# RAG Against The Machine

Retrieval-Augmented Generation over the vLLM repository, using BM25 for retrieval and a local vLLM OpenAI-compatible server for generation.

## What This Project Does

- Indexes the vLLM source tree under `data/raw/vllm-0.10.1`
- Chunks Python and Markdown files with source offsets
- Builds a BM25 index in `data/processed/bm25_index`
- Saves structured chunks in `data/processed/chunks/chunks.jsonl`
- Searches unanswered questions and writes moulinette-compatible JSON
- Generates answers from retrieved sources using a local vLLM server
- Evaluates retrieval against the answered datasets with recall@k

## Requirements

- Python 3.10+
- `uv`
- A running vLLM OpenAI-compatible server on `http://localhost:8000/v1`

This project is configured to use:

- `Qwen/Qwen3-0.6B` for generation
- BM25 for retrieval

## Setup

```bash
uv sync
```

If you already have the environment and want to avoid a sync step:

```bash
uv run --no-sync student show_config
```

## Start vLLM

Run the model outside Python. The project only sends HTTP requests to the server.

```bash
vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000
```

Set the model name used by the CLI:

```bash
export VLLM_MODEL="Qwen/Qwen3-0.6B"
```

## CLI Commands

The CLI is exposed through Python Fire.

```bash
uv run student index --max_chunk_size 2000
uv run student search "What does the usage property return in vLLM's KV cache manager?" --k 10
uv run student search_dataset data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10
uv run student evaluate data/output/search_results/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run student answer "What does the usage property return in vLLM's KV cache manager?" --model Qwen/Qwen3-0.6B
uv run student answer_dataset data/output/search_results/dataset_docs_public.json --model Qwen/Qwen3-0.6B
uv run student datasets
```

If your environment has a restricted cache or `uv` tries to re-resolve packages, use:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync student <command>
```

## Data Flow

1. `student index`
   - Reads `data/raw/vllm-0.10.1`
   - Chunks code and Markdown
   - Writes `data/processed/bm25_index/`
   - Writes `data/processed/chunks/chunks.jsonl`

2. `student search_dataset`
   - Reads an unanswered dataset
   - Retrieves the top `k` sources per question
   - Writes `data/output/search_results/<dataset_name>.json`

3. `student evaluate`
   - Compares retrieved sources to the answered dataset
   - Reports recall@1, recall@3, recall@5, recall@10
   - Uses a minimum IoU overlap threshold of 5%

4. `student answer_dataset`
   - Reads the saved search results JSON
   - Fetches chunk text from the indexed corpus
   - Calls the local vLLM server
   - Writes `data/output/search_results_and_answer/<dataset_name>.json`

## Output Formats

Search results:

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

Answered results:

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
      ],
      "answer": "Generated answer"
    }
  ],
  "k": 10
}
```

## Implementation Notes

- BM25 is the retrieval method.
- Markdown is chunked by headings and include directives.
- Python is chunked with AST-aware callable/class extraction.
- Chunk size is capped at 2000 characters by default.
- `answer_dataset` supports checkpointing and resume, so interrupted runs can continue from the last saved JSON file.

## Suggested Run Order

1. `student index`
2. `student search_dataset ...`
3. `student evaluate ...`
4. `student answer_dataset ...`

That is the order that best matches the grading pipeline.


part on stop word ('a', 'the') stopwords="en"