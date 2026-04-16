"""
Search and fetch module.

Search strategy (in priority order):
  1. Reddit JSON API          — for all sites of type: reddit
  2. Google Custom Search API — when google_search.enabled=true in config.yaml
  3. Per-site search URL      — fallback for sites without Google Search

Fetch strategy:
  - type: reddit              → Reddit JSON API (structured, no JS needed)
  - fetch_method: playwright  → headless Chromium via Playwright (JS-rendered sites)
  - default                   → requests + BeautifulSoup (fast, for simple HTML)

Each fetched page is returned as:
  {url, title, html, site_name, query}
"""

import logging
import os
import time
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CollegeQABot/1.0; "
        "educational research; contact: research@example.com)"
    )
}


# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, limits: dict):
        self._limits = limits
        self._last: dict[str, float] = {}

    def wait(self, domain: str):
        delay = self._limits.get(domain, self._limits.get("default", 2.0))
        elapsed = time.time() - self._last.get(domain, 0.0)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last[domain] = time.time()


# ── Main agent ────────────────────────────────────────────────────────────────

class SearchAgent:
    def __init__(self, config: dict):
        self.config = config
        self.rate = RateLimiter(config.get("rate_limits", {"default": 2.0}))
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

        gs = config.get("google_search", {})
        self.google_enabled = gs.get("enabled", False)
        self.google_api_key = os.environ.get("GOOGLE_API_KEY", gs.get("api_key", ""))
        self.google_cx = os.environ.get("GOOGLE_CX", gs.get("cx", ""))

        self._playwright = None  # lazy-loaded

    def search_site(self, site: dict, query: str) -> list[dict]:
        """
        Find and fetch pages for a given site + query.
        Returns list of page dicts ready for qa_extractor.
        """
        if not site.get("enabled", True):
            return []

        site_type = site.get("type", "generic")
        logger.info(f"Searching [{site['name']}] for: {query[:60]}")

        if site_type == "reddit":
            return self._search_reddit(site, query)

        if self.google_enabled and self.google_api_key and self.google_cx:
            return self._search_google(site, query)

        return self._search_direct(site, query)

    # ── Reddit ────────────────────────────────────────────────────────────────

    def _search_reddit(self, site: dict, query: str) -> list[dict]:
        search_url = site["search_url"].replace("{query}", quote_plus(query))
        domain = urlparse(search_url).netloc
        self.rate.wait(domain)

        try:
            resp = self.session.get(search_url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Reddit search failed [{site['name']}]: {e}")
            return []

        pages = []
        posts = data.get("data", {}).get("children", [])
        for post in posts[:10]:
            pd = post.get("data", {})
            post_url = f"https://www.reddit.com{pd.get('permalink', '')}"
            self.rate.wait(domain)
            page = self._fetch_reddit_post(post_url + ".json", site["name"], query)
            if page:
                pages.append(page)
        return pages

    def _fetch_reddit_post(self, url: str, site_name: str, query: str) -> dict | None:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            post_data = data[0]["data"]["children"][0]["data"]
            comments = data[1]["data"]["children"]

            title = post_data.get("title", "")
            body = post_data.get("selftext", "")
            top_comments = []
            for c in comments[:8]:
                cd = c.get("data", {})
                if isinstance(cd, dict) and cd.get("body") and cd["body"] != "[deleted]":
                    top_comments.append(cd["body"])

            combined_html = f"<h1>{title}</h1><p>{body}</p>"
            for comment in top_comments:
                combined_html += f"<div class='comment'>{comment}</div>"

            permalink = post_data.get("permalink", "")
            return {
                "url": f"https://www.reddit.com{permalink}",
                "title": title,
                "html": combined_html,
                "site_name": site_name,
                "query": query,
            }
        except Exception as e:
            logger.warning(f"Failed to fetch Reddit post {url}: {e}")
            return None

    # ── Google Custom Search ──────────────────────────────────────────────────

    def _search_google(self, site: dict, query: str) -> list[dict]:
        """
        Use Google Custom Search API to find relevant URLs within the site,
        then fetch each with Playwright or requests.
        """
        site_query = f"site:{urlparse(site['base_url']).netloc} {query}"
        url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?key={self.google_api_key}&cx={self.google_cx}"
            f"&q={quote_plus(site_query)}&num=5"
        )
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            items = resp.json().get("items", [])
        except Exception as e:
            logger.warning(f"Google Search failed for [{site['name']}]: {e}")
            return self._search_direct(site, query)

        pages = []
        fetch_method = site.get("fetch_method", "requests")
        for item in items:
            page_url = item.get("link", "")
            if not page_url:
                continue
            domain = urlparse(page_url).netloc
            self.rate.wait(domain)
            page = self._fetch_page(page_url, site["name"], query, fetch_method)
            if page:
                pages.append(page)
        return pages

    # ── Direct per-site search (fallback) ─────────────────────────────────────

    def _search_direct(self, site: dict, query: str) -> list[dict]:
        search_url = site["search_url"].replace("{query}", quote_plus(query))
        fetch_method = site.get("fetch_method", "requests")
        domain = urlparse(search_url).netloc
        self.rate.wait(domain)

        html = self._fetch_html(search_url, fetch_method)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        links = self._extract_links(soup, site["base_url"])
        pages = []
        for link in links[:8]:
            self.rate.wait(urlparse(link).netloc)
            page = self._fetch_page(link, site["name"], query, fetch_method)
            if page:
                pages.append(page)
        return pages

    # ── Page fetching ─────────────────────────────────────────────────────────

    def _fetch_page(self, url: str, site_name: str, query: str, method: str = "requests") -> dict | None:
        html = self._fetch_html(url, method)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title else ""
        return {"url": url, "title": title, "html": html, "site_name": site_name, "query": query}

    def _fetch_html(self, url: str, method: str = "requests") -> str | None:
        if method == "playwright":
            return self._fetch_with_playwright(url)
        return self._fetch_with_requests(url)

    def _fetch_with_requests(self, url: str) -> str | None:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"requests fetch failed {url}: {e}")
            return None

    def _fetch_with_playwright(self, url: str) -> str | None:
        try:
            pw = self._get_playwright()
            page = pw["browser"].new_page()
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)  # allow JS to render
            html = page.content()
            page.close()
            return html
        except Exception as e:
            logger.warning(f"Playwright fetch failed {url}: {e}")
            return None

    def _get_playwright(self) -> dict:
        if self._playwright is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                raise ImportError("Run: pip install playwright && playwright install chromium")
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=True)
            self._playwright = {"pw": pw, "browser": browser}
        return self._playwright

    def close(self):
        if self._playwright:
            self._playwright["browser"].close()
            self._playwright["pw"].stop()
            self._playwright = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        base_domain = urlparse(base_url).netloc
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and base_domain in href:
                links.append(href)
            elif href.startswith("/"):
                links.append(base_url.rstrip("/") + href)
        return list(dict.fromkeys(links))
