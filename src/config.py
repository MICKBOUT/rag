from pathlib import Path


class Config:
    """
    Configuration constants used throughout the application.

    This class centralizes default values and file-system paths so that they
    can be easily referenced and changed in one place.

    Attributes:
        DEFAULT_BASE_URL (str): Default API base URL.
            Default: "http://localhost:8000/v1".
        DEFAULT_OUTPUT_DIR (Path): Directory for saving search-only results.
            Default: Path("data/output/search_results").
        DEFAULT_OUTPUT_DIR_ANSWER (Path): Directory for saving search results
            together with generated answers.
            Default: Path("data/output/search_results_and_answer").
        DEFAULT_MAX_TOKENS (int): Default maximum number of tokens for model
            responses. Default: 512.
        DEFAULT_SEARCH_K (int): Number of documents to retrieve per search.
            Default: 10.
        DEFAULT_TOP_CONTEXT_CHUNKS (Optional[int]): Number of top context
            chunks to include when building prompts. If None, a dynamic or
            external default is used.
        DEFAULT_TIMEOUT_SECONDS (float): General default timeout value in
            seconds. Default: 60.0.
        DEFAULT_CONCURRENCY (int): Default number of concurrent
            operations/threads.
            Default: 8.
        DEFAULT_CHECKPOINT_INTERVAL (int): Interval (in steps or iterations)
            at which checkpoints or progress are recorded. Default: 1.
        DEFAULT_MODEL (str): Default model identifier used for inference.
            Default: "Qwen/Qwen3-0.6B".
        MIN_IOU_THRESHOLD (float): Minimum Intersection-over-Union threshold
            used in
            overlap/duplication heuristics. Default: 0.05.
        INDEX_PATH (str): Filesystem path to the prepared BM25 index.
            Default: "data/processed/bm25_index".
        CHUNKS_PATH (str): Filesystem path to the JSONL file containing
            document chunks.
            Default: "data/processed/chunks/chunks.jsonl".
        RAW_ROOT (str): Root directory containing raw dataset files.
            Default: "data/raw/vllm-0.10.1".
        DEFAULT_MAX_CHUNK_SIZE (int): Maximum size (e.g., characters or tokens)
            for a document chunk when splitting. Default: 2000.
        REPO_ROOT (Path): Root directory of the repository
            (resolved from this file).
        TIMEOUT_SECONDS_SINGLE_QUESTION (float): Timeout used for
            single-question interactions. Default: 60.0.
        TIMEOUT_SECONDS_MULTIPLE_QUESTION (float): Timeout used for
            multi-question or batch interactions. Default: 600.0.
    """
    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_OUTPUT_DIR = "data/output/search_results"
    DEFAULT_OUTPUT_DIR_ANSWER = "data/output/search_results_and_answer"
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_SEARCH_K = 10
    DEFAULT_TOP_CONTEXT_CHUNKS = None
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
    TIMEOUT_SECONDS_SINGLE_QUESTION = 60.0
    TIMEOUT_SECONDS_MULTIPLE_QUESTION = 600.0
