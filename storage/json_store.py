"""
JSON storage layer. Records are written into batch files under data/.
Each file holds up to `records_per_file` records as a pretty-printed JSON array.
File naming: data/YYYY-MM-DD_batch_NNN.json

Deduplication tracking (all in data/):
  visited_urls.json      - set of all fetched page URLs
  question_hashes.json   - set of MD5 hashes of stored questions
  search_log.json        - {site::query -> ISO date} of last search run
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JsonStore:
    def __init__(self, data_dir: str = "data", records_per_file: int = 100, indent: int = 2):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.records_per_file = records_per_file
        self.indent = indent

        self._current_file: Optional[Path] = None
        self._current_records: list = []
        self._batch_index: int = 0

        # Dedup indexes — loaded once into memory, persisted on every write
        self._visited_urls: set = self._load_set("visited_urls.json")
        self._question_hashes: set = self._load_set("question_hashes.json")
        self._search_log: dict = self._load_dict("search_log.json")

        self._load_current_batch()

    # ── public API — records ──────────────────────────────────────────────────

    def save(self, record: dict) -> str:
        """Append a validated record. Returns its assigned id."""
        if not record.get("id"):
            record["id"] = str(uuid.uuid4())
        if not record.get("timestamp"):
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        h = _hash_question(record["question"])
        record["question_hash"] = h
        self._question_hashes.add(h)
        self._persist_set("question_hashes.json", self._question_hashes)

        self._current_records.append(record)
        self._flush()

        if len(self._current_records) >= self.records_per_file:
            self._rotate()

        return record["id"]

    def update(self, record_id: str, fields: dict) -> bool:
        """Update fields on an existing record by id. Returns True if found."""
        for path in sorted(self.data_dir.glob("*_batch_*.json")):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    records = json.load(f)
                except json.JSONDecodeError:
                    continue
            for r in records:
                if r.get("id") == record_id:
                    r.update(fields)
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(records, f, indent=self.indent, ensure_ascii=False)
                    return True
        return False

    def load_all(self) -> list[dict]:
        records = []
        for path in sorted(self.data_dir.glob("*_batch_*.json")):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    records.extend(json.load(f))
                except json.JSONDecodeError:
                    pass
        return sorted(records, key=lambda r: r.get("timestamp", ""))

    def count(self) -> int:
        return sum(1 for _ in self.load_all())

    # ── public API — deduplication ────────────────────────────────────────────

    def question_exists(self, question: str) -> bool:
        return _hash_question(question) in self._question_hashes

    def url_visited(self, url: str) -> bool:
        return url in self._visited_urls

    def mark_url_visited(self, url: str):
        self._visited_urls.add(url)
        self._persist_set("visited_urls.json", self._visited_urls)

    def search_was_run(self, site: str, query: str, refresh_days: int) -> bool:
        """Returns True if this (site, query) was run within refresh_days."""
        key = f"{site}::{query}"
        last_run = self._search_log.get(key)
        if not last_run:
            return False
        last_dt = datetime.fromisoformat(last_run)
        age_days = (datetime.now(timezone.utc) - last_dt).days
        return age_days < refresh_days

    def mark_search_run(self, site: str, query: str):
        key = f"{site}::{query}"
        self._search_log[key] = datetime.now(timezone.utc).isoformat()
        self._persist_dict("search_log.json", self._search_log)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_current_batch(self):
        today = datetime.now().strftime("%Y-%m-%d")
        existing = sorted(self.data_dir.glob(f"{today}_batch_*.json"))
        if existing:
            latest = existing[-1]
            with open(latest, "r", encoding="utf-8") as f:
                try:
                    self._current_records = json.load(f)
                except json.JSONDecodeError:
                    self._current_records = []
            self._batch_index = int(latest.stem.split("_batch_")[-1])
            self._current_file = latest
            if len(self._current_records) >= self.records_per_file:
                self._rotate()
        else:
            self._batch_index = 1
            self._current_file = self.data_dir / f"{today}_batch_{self._batch_index:03d}.json"
            self._current_records = []

    def _flush(self):
        with open(self._current_file, "w", encoding="utf-8") as f:
            json.dump(self._current_records, f, indent=self.indent, ensure_ascii=False)

    def _rotate(self):
        self._batch_index += 1
        today = datetime.now().strftime("%Y-%m-%d")
        self._current_file = self.data_dir / f"{today}_batch_{self._batch_index:03d}.json"
        self._current_records = []

    def _load_set(self, filename: str) -> set:
        path = self.data_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return set(json.load(f))
                except json.JSONDecodeError:
                    pass
        return set()

    def _load_dict(self, filename: str) -> dict:
        path = self.data_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    pass
        return {}

    def _persist_set(self, filename: str, data: set):
        path = self.data_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(data), f)

    def _persist_dict(self, filename: str, data: dict):
        path = self.data_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ── helpers ───────────────────────────────────────────────────────────────────

def _hash_question(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode()).hexdigest()


def make_record(
    question: str,
    answer: str = "",
    source_url: str = "",
    source_site: str = "",
    topic: str = "",
    is_relevant: Optional[bool] = None,
    quality_score: Optional[int] = None,
    generated_answer: str = "",
    manual_answer: str = "",
    best_answer_source: str = "",
) -> dict:
    """Canonical record structure used throughout the pipeline."""
    return {
        "id": "",
        "timestamp": "",
        "question_hash": "",               # filled by JsonStore.save()
        "question": question.strip(),
        "answer": answer.strip(),
        "source_url": source_url,
        "source_site": source_site,
        "topic": topic,
        "is_relevant": is_relevant,
        "quality_score": quality_score,
        "generated_answer": generated_answer.strip(),
        "manual_answer": manual_answer.strip(),
        "best_answer_source": best_answer_source,  # original | generated | manual
    }
