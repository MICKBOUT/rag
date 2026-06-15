*This project has been created as part of the 42 curriculum by mmatisse.*

# RAG Against The Machine

> **Retrieval-Augmented Generation over the vLLM repository.**
> BM25 retrieval · DSPy generation · Qwen3-0.6B · sliding-window chunking · local vLLM inference

---

## Description

**RAG Against The Machine** is a fully local Retrieval-Augmented Generation (RAG) system
built to answer technical questions about the [vLLM](https://github.com/vllm-project/vllm) project.

The pipeline works in four stages:

1. **Ingest** — chunk Python source files (AST-aware) and Markdown docs (sliding window) from the vLLM repository.
2. **Index** — build a BM25 index with `bm25s` for fast lexical retrieval.
3. **Retrieve** — given a question, return the top-*k* source spans (file path + character offsets).
4. **Generate** — pass the top chunks as context to a local Qwen3-0.6B server (vLLM, OpenAI-compatible) via DSPy `ChainOfThought`.

The system is exposed through a Python Fire CLI (`student`) and produces structured JSON output compatible with the moulinette grader.

---

## System Architecture  

```
data/raw/vllm-0.10.1/          ← vLLM source tree
        │
        ▼
 ast_cleaning.py               ← chunking layer
  ├── Python files  →  AST-aware callable/class extraction
  └── Markdown files → sliding-window (1 000 chars, stride 500)
        │
        ▼
 indexing.py                   ← BM25 index builder (bm25s)
  ├── data/processed/bm25_index/    ← serialised retriever + corpus
  └── data/processed/chunks/chunks.jsonl
        │
        ▼
 retrieval.py                  ← bm25s.BM25 query, returns SearchResult list
        │
        ▼
 generation.py                 ← DSPy ChainOfThought → vLLM server (OpenAI API)
        │
        ▼
 pipeline.py                   ← dataset search, IoU-based recall@k evaluation
        │
        ▼
 cli.py  +  student.py         ← Python Fire CLI entry point
```

**Validation** is enforced at every CLI boundary via Pydantic v2 models (`validation.py`),
and all errors surface as readable messages rather than stack traces.

---

## Chunking Strategy

Two strategies are applied depending on file type:

### Python files — AST-aware extraction

Each `.py` file is parsed with Python's `ast` module.
`ContextualParser` walks the AST and records every class, method, function, nested
function, and async function, together with:

- the full source lines it spans
- the list of callees (for functions)
- the nesting context (for symbol path reconstruction)

Each callable/class becomes one chunk whose `text` field is a structured header
(`FILE / TYPE / SYMBOL / CALLS / …`) followed by the raw source code.
Character offsets (`first_character_index`, `last_character_index`) are derived directly
from the AST line numbers, making them exact and evaluation-safe.

### Markdown files — sliding window

Markdown is **not** split at heading boundaries (that was found to leave gold spans
uncovered when a relevant paragraph sits under a deep heading).
Instead a sliding window of **1 000 characters** with a **500-character stride** is
applied to the raw text, producing overlapping chunks that cover every byte of the file.

Each chunk's `text` is prefixed with `FILE:` and `HEADING:` metadata (the heading
stack active at the chunk's start offset), so BM25 still has structural signal while the
span remains contiguous.

### Chunk size cap

All chunks are capped at `--max_chunk_size` characters (default **2 000**).
The cap is applied to the BM25 text and to the source span independently, so span
indices are never corrupted by prefix overhead.

---

## Retrieval Method

**BM25** via [`bm25s`](https://github.com/xhluca/bm25s):

- English stop-word filtering (`stopwords="en"`)
- The full corpus (text + metadata) is serialised alongside the index so loading is
  instant on subsequent runs
- Metadata versioning: `max_chunk_size` and `folder_path` are stored in
  `bm25_index/metadata.json`; a mismatch triggers automatic rebuilding

No vector embeddings are used in the mandatory part (pure lexical BM25).

---

## Performance Analysis

Results on `dataset_docs_public.json` (100 questions, IoU threshold 5 %):

| Metric | Score |
|--------|-------|
| Recall@1 | ~0.55 |
| Recall@3 | ~0.68 |
| **Recall@5** | **~0.81** |
| Recall@10 | ~0.84 |

Target: **≥ 0.80 Recall@5** on docs questions.
The sliding-window chunker (replacing the earlier heading-split approach) is the main
driver of the improvement; the span-index decoupling fix in `_limit_entry_text` also
eliminated a class of IoU undercount.

---

## Bonus Features

### Local LLM inference via vLLM

All generation goes through a locally-served `vllm serve` process on
`http://localhost:8000/v1`.  No external API is called.

`### `top_context_chunks` context pruning

Only the top-*n* retrieved chunks are passed as context to the LLM (default `n=3`),
avoiding context-window overload on small models.

### Checkpoint & resume for `answer_dataset`

Interrupted runs write partial results to the output file.  On restart, already-answered
questions are skipped automatically.

### Concurrent generation

`answer_dataset` accepts `--concurrency N` to process multiple questions in parallel
using `ThreadPoolExecutor`.  Each worker instantiates its own `dspy.LM` to avoid
shared-state issues.

### English stop-word filtering

BM25 tokenisation strips English stop words (`'a'`, `'the'`, `'is'`, …) before indexing
and at query time, improving term-frequency signal on technical vocabulary.

### Dspy Caching

dspy use cache that avoid regenerating the same question-context pair multiple times,
which is especially useful when tuning generation parameters.

---

## Requirements

- Python ≥ 3.10
- [`uv`](https://github.com/astral-sh/uv) (package manager)
- A running **vLLM** OpenAI-compatible server:

  ```bash
  vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000
  ```

  CPU-only build (no CUDA / ROCm required):

  ```bash
  VLLM_TARGET_DEVICE=cpu vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000
  ```

---

## Instructions

### Install

```bash
uv sync
```

If the disk / cache is restricted:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

### Index the repository (first run)

```bash
uv run student index --max_chunk_size 2000
```

The index is stored under `data/processed/bm25_index/` and reused on subsequent
commands.  Delete the directory to force a rebuild.

### Search a single query

```bash
uv run student search "What does the usage property return in vLLM's KV cache manager?" --k 10
```

### Search a full dataset

```bash
uv run student search_dataset \
  data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results
```

### Evaluate retrieval quality

```bash
uv run student evaluate \
  data/output/search_results/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json
```

### Answer a single question

```bash
uv run student answer \
  "What does the usage property return in vLLM's KV cache manager?" \
  --model Qwen/Qwen3-0.6B \
  --k 10
```

### Answer a full dataset

```bash
uv run student answer_dataset \
  data/output/search_results/dataset_docs_public.json \
  --model Qwen/Qwen3-0.6B \
  --concurrency 1 \
  --timeout_seconds 300 \
  --max_tokens 128
```

### List available datasets

```bash
uv run student datasets
```

### Show active configuration

```bash
uv run student show_config
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **BM25 over TF-IDF** | `bm25s` serialises the full corpus with the index, so load time is O(1) and no rebuild is needed between commands. |
| **AST chunking for Python** | Function/class boundaries are the natural unit of code search; heading splits would cut across multi-method classes. |
| **Sliding window for Markdown** | Heading-split chunking left gold spans uncovered when answers straddle section boundaries; overlapping windows guarantee coverage. |
| **Decoupled span vs. BM25 text length** | Markdown chunks prepend `FILE:/HEADING:` prefixes, inflating `len(text)`.  Using `len(text)` to clip `last_character_index` corrupted IoU scores; the span is now measured independently. |
| **DSPy `ChainOfThought`** | Structured prompting with a typed signature (`context`, `question` → `answer`) reduces hallucination and makes the LM call swappable. |
| **Pydantic v2 validation at CLI boundary** | Every command validates its arguments before touching the filesystem or network, giving actionable error messages instead of tracebacks. |
| **Checkpoint writes on `answer_dataset`** | CPU-only inference is slow; checkpointing every `--checkpoint_interval` questions means a crash doesn't lose all progress. |

---

## Challenges Faced

**Heading-split chunking leaving gold spans uncovered**
The first chunker split Markdown at `#` headings.  When a gold span started a few lines
before a heading boundary, the relevant text ended up in the wrong chunk.  Switching to
sliding-window chunking with 50 % overlap fixed this and pushed Recall@5 up ~6 points.

**`_limit_entry_text` corrupting span indices**
Markdown BM25 text includes a `FILE:/HEADING:` prefix not present in the source file.
`len(text) > max_chunk_size` was used as the condition to clip `last_character_index`,
which clipped spans that were actually within budget.  The fix computes span length
independently from BM25 text length.

**CPU-only vLLM throughput**
`concurrency=8` overwhelmed the CPU server, causing cascading timeouts that masked
as misleading "connection refused" errors.  The solution was `--concurrency 1
--timeout_seconds 300`.  The error-message handler was also patched to distinguish
timeout from refused-connection.

**`max_chunk_size` not propagated through the call chain**
The CLI correctly accepted `--max_chunk_size`, but `pipeline.py` was calling
`search_dataset_to_file` without forwarding the argument, so the index was always
rebuilt with the default value.  A systematic audit of all call sites fixed the propagation.

---

## Output Formats

**Search results** (`data/output/search_results/<dataset>.json`):

```json
{
  "search_results": [
    {
      "question_id": "uuid",
      "question_str": "Question text",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/...",
          "first_character_index": 0,
          "last_character_index": 123
        }
      ]
    }
  ],
  "k": 10
}
```

**Answered results** (`data/output/search_results_and_answer/<dataset>.json`):

```json
{
  "answers": [
    {
      "question_id": "uuid",
      "question_str": "Question text",
      "answer": "Generated answer",
      "retrieved_sources": [ ... ]
    }
  ]
}
```

---

## Suggested Run Order

```bash
uv run student index --max_chunk_size 2000
uv run student search_dataset data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10
uv run student evaluate \
  data/output/search_results/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run student answer_dataset \
  data/output/search_results/dataset_docs_public.json \
  --concurrency 1 --timeout_seconds 300 --max_tokens 128
```

---

## Resources

- [vLLM documentation](https://docs.vllm.ai)
- [bm25s — fast BM25 in Python](https://github.com/xhluca/bm25s)
- [DSPy — programming language models](https://github.com/stanfordnlp/dspy)
- [Qwen3 model family](https://huggingface.co/Qwen)
- [RAG survey (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Python `ast` module documentation](https://docs.python.org/3/library/ast.html)

**AI usage in this project:**
Claude (Anthropic) was used to assist with: debugging the span-index corruption bug,
designing the sliding-window chunker, writing Pydantic validation models, diagnosing
`ThreadPoolExecutor` / vLLM timeout interactions, and drafting this README.
All generated code was reviewed, tested, and understood before inclusion.