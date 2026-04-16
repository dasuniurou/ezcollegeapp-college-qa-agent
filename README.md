# College Application Q&A Agent

An autonomous agent that crawls college application websites and forums, extracts question-answer pairs using LLMs, evaluates quality, and builds a high-quality dataset for RAG and educational use.

---

## Architecture

```
config.yaml                      ← all sites, queries, topics, LLM, and settings
     │
orchestrator.py                  ← Stage 1–4: crawl, extract, classify, store
     ├── search_agent.py          ← find URLs (Reddit API / Google Search / Playwright)
     ├── qa_extractor.py          ← LLM extracts Q&A pairs from full page content
     ├── qa_processor.py          ← LLM classifies relevance + topic (no generation)
     └── storage/
           ├── json_store.py      ← batch JSON files + 3-level dedup tracking
           └── (faiss_store.py)   ← planned vector index for RAG

qa_answerer.py                   ← Stage 5: batch answer generation (run separately)
site_discovery.py                ← discover new sites, update config.yaml
llm_client.py                    ← unified wrapper: Claude / OpenAI / Gemini
```

---

## Full Workflow

### Stage 1 — Search (per site × query)

```
orchestrator.py loops over every enabled site and every query in config.yaml

For each (site, query):
  ┌─ Check search_log.json
  │   Was this (site, query) run within refresh_days?
  │   YES → skip entirely (saves API calls and time)
  │   NO  → proceed
  │
  └─ search_agent.py finds candidate page URLs using:
       type: reddit      → Reddit JSON API search endpoint
       google enabled    → Google Custom Search API (site:domain query)
       fallback          → direct per-site search URL
```

### Stage 2 — Fetch (per URL)

```
For each candidate URL:
  ┌─ Check visited_urls.json
  │   Already fetched?
  │   YES → skip
  │   NO  → fetch and mark as visited
  │
  └─ search_agent.py fetches the full page:
       type: reddit         → Reddit JSON API (post + top 8 comments)
       fetch_method: playwright → headless Chromium (JS-rendered sites)
       default              → requests + BeautifulSoup (fast, static HTML)
```

### Stage 3 — Extract (per page)

```
qa_extractor.py feeds the full page to the LLM:

  Input:  page title + body + comments (up to 8000 chars)
  Prompt: "Extract all college application questions from this content.
           For each, find the best answer if one exists. Return JSON array."
  Output: [{question, answer}, {question, answer}, ...]

Key behaviors:
  - Multiple Q&A pairs can come from one page
  - answer may be blank — question is always kept
  - LLM rewrites questions to be clear and self-contained
  - Short answers are fine — quality not length is what matters
```

### Stage 4 — Classify + Store (per Q&A pair)

```
For each extracted pair:
  ┌─ Check question_hashes.json
  │   Is this question already stored? (MD5 exact match)
  │   YES → skip
  │   NO  → proceed
  │
  ├─ qa_processor.py (LLM call):
  │     Is this relevant to college applications? (score 0–1)
  │     If score < relevance_threshold → discard
  │     If score ≥ threshold → assign topic label
  │
  └─ json_store.py saves record to data/YYYY-MM-DD_batch_NNN.json
       Fields saved: question, answer (may be blank), source_url,
                     source_site, topic, is_relevant, quality_score=null,
                     generated_answer="", manual_answer=""
```

### Stage 5 — Answer Generation (separate command)

```
python qa_answerer.py

Scans all stored records and finds those that need processing:
  - answer is blank AND generated_answer is blank, OR
  - quality_score < answer_quality_threshold

For each:
  ├─ Score the original answer (1–5) if it exists
  └─ Generate a new answer if blank or score < threshold

Updates records in-place. Does NOT re-crawl.

Options:
  --limit N        process at most N records
  --score-only     score existing answers without generating new ones
```

### Future Stage 6 — Manual / RAG Answers

```
Option A: Edit JSON directly
  Set "manual_answer" field and "best_answer_source": "manual"

Option B: RAG pipeline (planned)
  python rag_answerer.py
  Uses FAISS vector index to find relevant context, then LLM generates answer
```

---

## Deduplication

Three JSON tracking files live in `data/` and are checked **before any LLM call**:

| File | What it prevents | Key |
|---|---|---|
| `search_log.json` | Re-searching same (site, query) too soon | `"site_name::query"` → ISO date |
| `visited_urls.json` | Re-fetching same pages | URL string |
| `question_hashes.json` | Storing duplicate questions | MD5 of lowercased question |

Re-run the same command freely — already-processed work is always skipped.

---

## Record Schema

```json
{
  "id": "uuid4",
  "timestamp": "2026-04-16T10:00:00+00:00",
  "question_hash": "md5hex",
  "question": "How do I write the Common App essay?",
  "answer": "Original scraped answer (may be blank)",
  "source_url": "https://www.reddit.com/r/ApplyingToCollege/...",
  "source_site": "reddit_applying_to_college",
  "topic": "common_app_essay",
  "is_relevant": true,
  "quality_score": null,
  "generated_answer": "",
  "manual_answer": "",
  "best_answer_source": "original | generated | manual | (blank)"
}
```

`quality_score` and `generated_answer` are filled by `qa_answerer.py`, not during crawling.

---

## Setup

### 1. Activate your conda environment

```bash
conda activate edai_agent   # or whichever env you use
```

> **Important:** Always activate your environment before running any command.
> All `pip install` and `python` commands must use the same environment.
> To verify: `which python3` should point inside your env's directory.

### 2. Install dependencies

```bash
cd college_qa_agent
pip install -r requirements.txt
playwright install chromium
```

### 3. Set API keys in `.env`

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...

# Optional: for Google Custom Search
GOOGLE_API_KEY=...
GOOGLE_CX=...
```

### 4. Choose your LLM in `config.yaml`

```yaml
llm:
  default_model: openai   # or: claude | gemini
```

### 5. (Optional) Enable Google Custom Search

```yaml
google_search:
  enabled: true
  api_key: ""   # or set GOOGLE_API_KEY in .env
  cx: ""        # or set GOOGLE_CX in .env
```

Get credentials at: https://developers.google.com/custom-search/v1/overview

---

## Usage

### Crawl (collect Q&A pairs)

```bash
# Full run
python orchestrator.py

# Test with 3 queries only
python orchestrator.py --queries 3

# Discover new sites first, then crawl
python orchestrator.py --discover

# Only discover new sites, skip crawl
python orchestrator.py --discover-only
```

### Generate answers (run after crawling)

```bash
# Generate answers for all blank/low-quality records
python qa_answerer.py

# Process at most 20 records
python qa_answerer.py --limit 20

# Score existing answers without generating new ones
python qa_answerer.py --score-only
```

### Discover new sites

```bash
python site_discovery.py
```

---

## Evolving the Dataset

| Task | How |
|---|---|
| Add search queries | Edit `search.queries` in `config.yaml` |
| Add new topics | Edit `topics` in `config.yaml` |
| Add a site manually | Add entry under `sites` in `config.yaml` |
| Disable a site | Set `enabled: false` on the site entry |
| Re-run stale queries | Reduce `storage.refresh_days` or delete `data/search_log.json` |
| Re-fetch visited URLs | Delete `data/visited_urls.json` |
| Full reset | Delete all files in `data/` |

---

## Data Files

```
data/
  2026-04-16_batch_001.json    ← Q&A records (up to 100 per file)
  2026-04-16_batch_002.json
  visited_urls.json            ← all fetched URLs
  question_hashes.json         ← MD5 hashes of all stored questions
  search_log.json              ← (site, query) run history with timestamps
```

Batch files are human-readable JSON arrays compatible with:
```python
import pandas as pd
df = pd.read_json("data/2026-04-16_batch_001.json")
```
```python
from datasets import load_dataset
ds = load_dataset("json", data_files="data/*.json")
```

---

## Target Sources

| Site | Type | Fetch method |
|---|---|---|
| r/ApplyingToCollege | Reddit | Reddit JSON API |
| r/chanceme | Reddit | Reddit JSON API |
| r/collegeadmissions | Reddit | Reddit JSON API |
| College Confidential | Forum | Playwright |
| CommonApp Help Center | Help center | Playwright |
| PrepScholar Blog | Blog | Playwright |
| CollegeVine | Blog + Q&A | Playwright |
| Niche | Blog + Reviews | Playwright |
