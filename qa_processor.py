"""
QA Processor — lightweight classification step during the crawl pipeline.

Responsibilities (crawl time only):
  1. Relevance check — is this question about college applications?
  2. Topic label — which topic does it belong to?

Answer generation is NOT done here. Use qa_answerer.py for that.
"""

import json
import logging
from typing import Optional

from llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert college admissions counselor. "
    "You classify questions about undergraduate college applications."
)


class QAProcessor:
    def __init__(self, llm: LLMClient, relevance_threshold: float = 0.6):
        self.llm = llm
        self.relevance_threshold = relevance_threshold

    def process(self, raw: dict) -> Optional[dict]:
        """
        Classify a raw Q&A pair.
        Returns enriched dict with topic and relevance fields, or None if not relevant.
        """
        question = raw["question"]
        relevance_score, topic = self._classify(question)

        if relevance_score < self.relevance_threshold:
            logger.debug(f"Discarded (relevance={relevance_score:.2f}): {question[:80]}")
            return None

        return {
            **raw,
            "topic": topic,
            "is_relevant": True,
            "relevance_score": relevance_score,
            "quality_score": None,       # scored later by qa_answerer.py
            "generated_answer": "",      # generated later by qa_answerer.py
            "manual_answer": "",
            "best_answer_source": "original" if raw.get("answer") else "",
        }

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _classify(self, question: str) -> tuple[float, str]:
        topics = [
            "common_app_essay", "supplemental_essays", "activities_section",
            "financial_aid", "scholarships", "early_decision_early_action",
            "waitlist_strategy", "letters_of_recommendation", "standardized_tests",
            "college_list_building", "interviews", "demonstrated_interest",
            "transfer_applications", "international_students",
            "first_generation_students", "general_admissions", "not_related",
        ]
        prompt = f"""Is the following question related to undergraduate college applications in the USA?

Question: {question}

Respond with JSON only:
{{"relevance_score": <float 0.0-1.0>, "topic": "<one of: {' | '.join(topics)}>"}}"""

        try:
            raw = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
            data = json.loads(self._strip_fences(raw))
            score = float(data.get("relevance_score", 0.0))
            topic = data.get("topic", "general_admissions")
            return score, topic
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return 0.0, "general_admissions"

    def _strip_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return text
