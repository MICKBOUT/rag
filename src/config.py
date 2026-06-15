from pathlib import Path


class Config:
    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_OUTPUT_DIR = Path("data/output/search_results_and_answer")
    DEFAULT_OUTPUT_DIR_ANSWER = Path("data/output/search_results_and_answer")
    DEFAULT_MAX_TOKENS = 256
    DEFAULT_SEARCH_K = 10
    DEFAULT_TOP_CONTEXT_CHUNKS = 3
    DEFAULT_TIMEOUT_SECONDS = 60.0
    DEFAULT_CONCURRENCY = 8
    DEFAULT_CHECKPOINT_INTERVAL = 1
    DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
    MIN_IOU_THRESHOLD = 0.05
    INDEX_PATH = "data/processed/bm25_index"
    CHUNKS_PATH = "data/processed/chunks/chunks.jsonl"
    RAW_ROOT = "data/raw/vllm-0.10.1"
    DEFAULT_MAX_CHUNK_SIZE = 2000
    REPO_ROOT = Path(__file__).resolve().parents[1]
