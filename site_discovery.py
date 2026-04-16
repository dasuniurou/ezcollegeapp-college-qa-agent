"""
Site Discovery — searches for new college application Q&A websites,
evaluates their relevance with the LLM, and proposes additions to config.yaml.

Run independently: python site_discovery.py
Discovered sites are written back to config.yaml under the `sites` key.
"""

import logging
import time
from urllib.parse import quote_plus, urlparse

import requests
import yaml
from bs4 import BeautifulSoup

from llm_client import LLMClient, load_llm_from_config

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CollegeQASiteDiscovery/1.0; educational research)"
    )
}


class SiteDiscovery:
    def __init__(self, config: dict, config_path: str = "config.yaml"):
        self.config = config
        self.config_path = config_path
        self.llm: LLMClient = load_llm_from_config(config)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        disc_cfg = config.get("discovery", {})
        self.min_relevance = disc_cfg.get("min_relevance_score", 0.7)
        self.max_new = disc_cfg.get("max_new_sites_per_run", 5)
        self.search_queries = disc_cfg.get("search_queries", [])

    def run(self) -> list[dict]:
        """
        Search for new sites, evaluate them, and append qualifying ones to config.yaml.
        Returns the list of newly added site configs.
        """
        existing_urls = {s["base_url"] for s in self.config.get("sites", [])}
        candidates = self._find_candidates()
        added = []

        for candidate in candidates:
            if candidate["base_url"] in existing_urls:
                continue
            score = self._evaluate_site(candidate)
            logger.info(f"Site {candidate['base_url']} relevance={score:.2f}")
            if score >= self.min_relevance:
                site_entry = self._build_site_entry(candidate, score)
                self.config["sites"].append(site_entry)
                existing_urls.add(candidate["base_url"])
                added.append(site_entry)
                logger.info(f"Added new site: {site_entry['name']}")
                if len(added) >= self.max_new:
                    break

        if added:
            self._save_config()

        return added

    # ── Discovery ─────────────────────────────────────────────────────────────

    def _find_candidates(self) -> list[dict]:
        candidates = []
        for query in self.search_queries:
            results = self._search_web(query)
            candidates.extend(results)
            time.sleep(2)
        return candidates

    def _search_web(self, query: str) -> list[dict]:
        """
        Use DuckDuckGo HTML search (no API key needed) to find candidate URLs.
        Returns list of {base_url, title, snippet}.
        """
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result")[:10]:
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            link_el = result.select_one(".result__url")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            raw_url = link_el.get_text(strip=True) if link_el else ""
            if not raw_url:
                continue
            if not raw_url.startswith("http"):
                raw_url = "https://" + raw_url
            parsed = urlparse(raw_url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            results.append({"base_url": base_url, "title": title, "snippet": snippet})

        return results

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate_site(self, candidate: dict) -> float:
        prompt = f"""Evaluate whether this website is a good source for college application Q&A data.
Good sources include: student forums, help centers, admissions blogs, Q&A communities focused on undergraduate college applications in the USA.

Site URL: {candidate['base_url']}
Title: {candidate['title']}
Snippet: {candidate['snippet']}

Respond with JSON only:
{{"relevance_score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""

        try:
            import json
            raw = self.llm.generate(prompt)
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1])
            data = json.loads(text)
            return float(data.get("relevance_score", 0.0))
        except Exception as e:
            logger.warning(f"Site evaluation failed for {candidate['base_url']}: {e}")
            return 0.0

    def _build_site_entry(self, candidate: dict, score: float) -> dict:
        parsed = urlparse(candidate["base_url"])
        name = parsed.netloc.replace(".", "_").replace("-", "_")
        return {
            "name": f"discovered_{name}",
            "enabled": True,
            "base_url": candidate["base_url"],
            "search_url": candidate["base_url"] + "/search?q={query}",
            "type": "generic",
            "notes": f"Auto-discovered. relevance_score={score:.2f}. {candidate.get('snippet', '')[:100]}",
        }

    # ── Config persistence ────────────────────────────────────────────────────

    def _save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        logger.info(f"config.yaml updated with new sites.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if not config.get("discovery", {}).get("enabled", True):
        print("Site discovery is disabled in config.yaml.")
        sys.exit(0)
    discovery = SiteDiscovery(config, config_path)
    added = discovery.run()
    print(f"Discovery complete. {len(added)} new site(s) added to config.yaml.")
    for s in added:
        print(f"  + {s['name']} — {s['base_url']}")
