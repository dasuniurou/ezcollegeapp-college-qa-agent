"""
Orchestrator — coordinates the full crawl pipeline (Stages 1–4).

Pipeline per (site × query):
  1. Skip if (site, query) was run within refresh_days        [search_log.json]
  2. Search: get candidate page URLs
  3. For each URL:
     a. Skip if URL already visited                           [visited_urls.json]
     b. Fetch full page (Reddit API or Playwright or requests)
     c. Mark URL visited
  4. LLM extracts Q&A pairs from full page content
  5. For each pair:
     a. Skip if question hash already stored                  [question_hashes.json]
     b. LLM classifies relevance + topic
     c. Discard if relevance < threshold
     d. Store record (answer may be blank)
  6. Mark (site, query) as run

Answer generation is NOT done here. Run: python qa_answerer.py

Usage:
  python orchestrator.py                      # full run
  python orchestrator.py --queries 3          # limit queries (good for testing)
  python orchestrator.py --discover           # run site discovery first
  python orchestrator.py --discover-only      # only discover, no crawl
  python orchestrator.py --config my.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from llm_client import load_llm_from_config
from search_agent import SearchAgent
from qa_extractor import QAExtractor
from qa_processor import QAProcessor
from storage.json_store import JsonStore, make_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        proc_cfg = config.get("processing", {})
        stor_cfg = config.get("storage", {})
        self.refresh_days = stor_cfg.get("refresh_days", 7)

        self.llm = load_llm_from_config(config)
        self.searcher = SearchAgent(config)
        self.extractor = QAExtractor(llm=self.llm)
        self.processor = QAProcessor(
            llm=self.llm,
            relevance_threshold=proc_cfg.get("relevance_threshold", 0.6),
        )
        self.store = JsonStore(
            data_dir=stor_cfg.get("data_dir", "data"),
            records_per_file=stor_cfg.get("records_per_file", 100),
            indent=stor_cfg.get("indent", 2),
        )

    def run(self, max_queries: int = None):
        sites = [s for s in self.config.get("sites", []) if s.get("enabled", True)]
        queries = self.config.get("search", {}).get("queries", [])
        if max_queries:
            queries = queries[:max_queries]

        stats = {
            "search_skipped": 0,
            "pages_fetched": 0,
            "url_skipped": 0,
            "pairs_extracted": 0,
            "question_skipped": 0,
            "irrelevant_skipped": 0,
            "saved": 0,
        }

        logger.info(f"Starting pipeline: {len(sites)} sites × {len(queries)} queries")

        try:
            for query in queries:
                for site in sites:
                    self._process_site_query(site, query, stats)
        finally:
            self.searcher.close()

        self._print_summary(stats)
        return stats

    def _process_site_query(self, site: dict, query: str, stats: dict):
        # Dedup Level 3: skip (site, query) pairs run recently
        if self.store.search_was_run(site["name"], query, self.refresh_days):
            logger.info(f"Skipping [{site['name']}] '{query[:50]}' (run within {self.refresh_days}d)")
            stats["search_skipped"] += 1
            return

        pages = self.searcher.search_site(site, query)

        for page in pages:
            url = page.get("url", "")

            # Dedup Level 1: skip already-visited URLs
            if url and self.store.url_visited(url):
                logger.debug(f"URL already visited: {url}")
                stats["url_skipped"] += 1
                continue

            if url:
                self.store.mark_url_visited(url)
            stats["pages_fetched"] += 1

            # LLM extraction: full page → list of Q&A pairs
            pairs = self.extractor.extract(page)
            stats["pairs_extracted"] += len(pairs)

            for raw in pairs:
                question = raw["question"]

                # Dedup Level 2: skip duplicate questions by hash
                if self.store.question_exists(question):
                    logger.debug(f"Duplicate question: {question[:60]}")
                    stats["question_skipped"] += 1
                    continue

                # LLM classification: relevance + topic
                processed = self.processor.process(raw)
                if processed is None:
                    stats["irrelevant_skipped"] += 1
                    continue

                record = make_record(
                    question=processed["question"],
                    answer=processed.get("answer", ""),
                    source_url=processed.get("source_url", ""),
                    source_site=processed.get("source_site", ""),
                    topic=processed.get("topic", ""),
                    is_relevant=processed.get("is_relevant"),
                    best_answer_source=processed.get("best_answer_source", ""),
                )
                self.store.save(record)
                stats["saved"] += 1
                logger.info(f"Saved: {question[:80]}")

        # Mark (site, query) as done — even if 0 results, don't retry for refresh_days
        self.store.mark_search_run(site["name"], query)

    def _print_summary(self, stats: dict):
        print("\n" + "=" * 55)
        print("  Crawl Pipeline Complete")
        print("=" * 55)
        print(f"  (site,query) skipped:    {stats['search_skipped']}")
        print(f"  Pages fetched:           {stats['pages_fetched']}")
        print(f"  URLs skipped (visited):  {stats['url_skipped']}")
        print(f"  Q&A pairs extracted:     {stats['pairs_extracted']}")
        print(f"  Questions skipped (dup): {stats['question_skipped']}")
        print(f"  Skipped (irrelevant):    {stats['irrelevant_skipped']}")
        print(f"  Records saved:           {stats['saved']}")
        print(f"  Total in store:          {self.store.count()}")
        print("=" * 55)
        print("  Run 'python qa_answerer.py' to generate answers.")
        print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="College Application Q&A Crawl Agent")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--discover", action="store_true", help="Run site discovery before crawling")
    parser.add_argument("--discover-only", action="store_true", help="Only run site discovery")
    parser.add_argument("--queries", type=int, default=None, help="Limit number of queries")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    if args.discover or args.discover_only:
        from site_discovery import SiteDiscovery
        logger.info("Running site discovery...")
        discovery = SiteDiscovery(config, args.config)
        added = discovery.run()
        logger.info(f"Site discovery added {len(added)} new site(s).")
        if args.discover_only:
            return
        config = load_config(args.config)

    orchestrator = Orchestrator(config)
    orchestrator.run(max_queries=args.queries)


if __name__ == "__main__":
    main()
