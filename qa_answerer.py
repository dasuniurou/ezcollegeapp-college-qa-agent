"""
QA Answerer — standalone command that batch-generates answers for stored records.

Run independently after crawling:
  python qa_answerer.py                   # process all records needing answers
  python qa_answerer.py --limit 20        # process at most 20 records
  python qa_answerer.py --score-only      # score existing answers without generating
  python qa_answerer.py --config my.yaml

A record needs processing if:
  - answer is blank AND generated_answer is blank, OR
  - quality_score is below answer_quality_threshold

After processing, each record is updated in-place with:
  - quality_score        scored 1-5 (if original answer exists)
  - generated_answer     LLM-written answer (if original was blank or low quality)
  - best_answer_source   "original" | "generated"
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from llm_client import load_llm_from_config
from storage.json_store import JsonStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert college admissions counselor with 15+ years of experience "
    "helping students and parents navigate the undergraduate college application process. "
    "You provide accurate, practical, and encouraging guidance."
)


class QAAnswerer:
    def __init__(self, config: dict):
        self.config = config
        self.llm = load_llm_from_config(config)
        stor_cfg = config.get("storage", {})
        self.store = JsonStore(
            data_dir=stor_cfg.get("data_dir", "data"),
            records_per_file=stor_cfg.get("records_per_file", 100),
            indent=stor_cfg.get("indent", 2),
        )
        self.threshold = config.get("processing", {}).get("answer_quality_threshold", 3)

    def run(self, limit: int = None, score_only: bool = False):
        records = self.store.load_all()
        to_process = [r for r in records if self._needs_processing(r)]

        if limit:
            to_process = to_process[:limit]

        logger.info(f"Found {len(to_process)} record(s) to process out of {len(records)} total.")
        stats = {"scored": 0, "generated": 0, "skipped": 0}

        for i, record in enumerate(to_process, 1):
            logger.info(f"[{i}/{len(to_process)}] {record['question'][:80]}")
            updates = {}

            answer = record.get("answer", "")

            # Step 1: score the original answer if it exists and not yet scored
            if answer and record.get("quality_score") is None:
                score = self._score_answer(record["question"], answer)
                updates["quality_score"] = score
                stats["scored"] += 1
                logger.info(f"  Scored original answer: {score}/5")
            else:
                score = record.get("quality_score")

            # Step 2: generate answer if blank or low quality (unless score_only)
            if not score_only:
                needs_gen = (not answer and not record.get("generated_answer")) or \
                            (score is not None and score < self.threshold)
                if needs_gen:
                    generated = self._generate_answer(record["question"])
                    updates["generated_answer"] = generated
                    updates["best_answer_source"] = "generated"
                    stats["generated"] += 1
                    logger.info(f"  Generated answer ({len(generated)} chars)")
                elif answer and not record.get("best_answer_source"):
                    updates["best_answer_source"] = "original"

            if updates:
                self.store.update(record["id"], updates)
            else:
                stats["skipped"] += 1

        self._print_summary(stats, len(to_process))

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _score_answer(self, question: str, answer: str) -> int:
        import json
        prompt = f"""Score the quality of this answer to a college application question on a 1-5 scale.

1 = Wrong, irrelevant, or bot/auto reply
2 = Vague, emotional support only, no real information
3 = Partially helpful but incomplete or generic
4 = Good — accurate, useful, addresses the question
5 = Excellent — thorough, specific, and actionable

Question: {question}

Answer: {answer}

Respond with JSON only:
{{"quality_score": <int 1-5>, "reason": "<one sentence>"}}"""

        try:
            raw = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1])
            data = json.loads(text)
            return max(1, min(5, int(data.get("quality_score", 3))))
        except Exception as e:
            logger.warning(f"Scoring failed: {e}")
            return 3

    def _generate_answer(self, question: str) -> str:
        prompt = f"""A student or parent is asking the following question about undergraduate college applications.

Write a thorough, accurate, and actionable answer. Be specific and practical.
Use plain paragraphs — no markdown headers or bullet points unless they genuinely help.

Question: {question}"""

        try:
            return self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _needs_processing(self, record: dict) -> bool:
        answer = record.get("answer", "")
        generated = record.get("generated_answer", "")
        score = record.get("quality_score")

        # Needs generation: no answer at all
        if not answer and not generated:
            return True
        # Needs scoring: has answer but no score yet
        if answer and score is None:
            return True
        # Needs generation: has score but it's below threshold
        if score is not None and score < self.threshold and not generated:
            return True
        return False

    def _print_summary(self, stats: dict, total: int):
        print("\n" + "=" * 50)
        print("  QA Answerer Complete")
        print("=" * 50)
        print(f"  Records processed:   {total}")
        print(f"  Answers scored:      {stats['scored']}")
        print(f"  Answers generated:   {stats['generated']}")
        print(f"  Skipped:             {stats['skipped']}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Batch generate/score answers for stored Q&A records")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Max records to process")
    parser.add_argument("--score-only", action="store_true", help="Score existing answers without generating new ones")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    answerer = QAAnswerer(config)
    answerer.run(limit=args.limit, score_only=args.score_only)


if __name__ == "__main__":
    main()
