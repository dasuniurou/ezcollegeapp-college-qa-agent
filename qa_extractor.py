"""
QA Extractor — uses the LLM to extract question/answer pairs from full page content.

Strategy:
  - Feed the full page (title + body + comments) to the LLM
  - LLM returns a list of {question, answer} pairs found on the page
  - Multiple pairs per page are allowed
  - answer may be blank if no good answer exists on the page
  - Question is always the priority — never discard a valid question
"""

import json
import logging
import re

from llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert at extracting college application questions and answers "
    "from online forum posts, blog articles, and help center pages."
)

EXTRACTION_PROMPT = """Read the following web page content from a college application resource.

Extract ALL distinct questions about undergraduate college applications that are present in this content.
For each question:
- Write it as a clear, self-contained question (reword if needed to make it standalone)
- Find the best answer from the content if one exists (comments, replies, article body)
- If no good answer exists, leave the answer blank — still include the question
- A good answer can be short or long — judge by usefulness, not length
- You may extract multiple questions from one page

Return ONLY a JSON array. No explanation. Example format:
[
  {{"question": "How do I choose a Common App essay topic?", "answer": "Focus on a specific moment that reveals who you are..."}},
  {{"question": "Can I reuse my Common App essay for supplements?", "answer": ""}}
]

Page title: {title}

Page content:
{content}"""


class QAExtractor:
    def __init__(self, llm: LLMClient, max_content_chars: int = 8000):
        self.llm = llm
        self.max_content_chars = max_content_chars

    def extract(self, page: dict) -> list[dict]:
        """
        Extract Q&A pairs from a page dict.
        Returns list of {question, answer, source_url, source_site, query}.
        """
        content = self._prepare_content(page)
        if not content.strip():
            logger.debug(f"Empty content for {page.get('url', '')}")
            return []

        title = page.get("title", "")
        pairs = self._llm_extract(title, content)

        results = []
        for pair in pairs:
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            if not q:
                continue
            results.append({
                "question": q,
                "answer": a,
                "source_url": page.get("url", ""),
                "source_site": page.get("site_name", ""),
                "query": page.get("query", ""),
            })

        logger.info(f"Extracted {len(results)} Q&A pair(s) from {page.get('url', '')[:80]}")
        return results

    # ── LLM extraction ────────────────────────────────────────────────────────

    def _llm_extract(self, title: str, content: str) -> list[dict]:
        prompt = EXTRACTION_PROMPT.format(
            title=title,
            content=content[:self.max_content_chars],
        )
        try:
            raw = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
            return self._parse_json(raw)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return []

    def _parse_json(self, text: str) -> list[dict]:
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        # Find JSON array in the response
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return [p for p in data if isinstance(p, dict) and p.get("question")]

    # ── Content preparation ───────────────────────────────────────────────────

    def _prepare_content(self, page: dict) -> str:
        """
        Build a clean text representation of the page.
        For Reddit pages (pre-structured HTML), parse accordingly.
        For generic HTML, use the stored plain text or strip tags.
        """
        html = page.get("html", "")
        if not html:
            return ""

        site_name = page.get("site_name", "")
        if "reddit" in site_name:
            return self._prepare_reddit(html)
        else:
            return self._prepare_generic(html)

    def _prepare_reddit(self, html: str) -> str:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        parts = []
        h1 = soup.find("h1")
        if h1:
            parts.append(f"POST TITLE: {h1.get_text(strip=True)}")
        p = soup.find("p")
        if p:
            parts.append(f"POST BODY: {p.get_text(separator=' ', strip=True)}")
        for i, div in enumerate(soup.find_all("div", class_="comment"), 1):
            text = div.get_text(separator=" ", strip=True)
            if text:
                parts.append(f"COMMENT {i}: {text}")
        return "\n\n".join(parts)

    def _prepare_generic(self, html: str) -> str:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
