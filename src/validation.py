from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ValidationError,
)


def _fmt_errors(exc: "ValidationError") -> str:
    """
    Return a compact, human-readable summary of Pydantic validation errors.
    """
    lines: list[str] = []
    for error in exc.errors():
        loc = " -> ".join(
            str(p) for p in error["loc"]) if error["loc"] else "input"
        lines.append(f"  • {loc}: {error['msg']}")
    return "\n".join(lines)


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise ValueError(f"file not found: '{value}'")
    if not path.is_file():
        raise ValueError(f"path is not a file: '{value}'")
    return path


def _json_file(value: str) -> Path:
    path = _existing_file(value)
    if path.suffix.lower() != ".json":
        raise ValueError(f"expected a .json file, got: '{value}'")
    return path


_K_FIELD = Field(
    default=10, ge=1, le=1000,
    description="Number of results to retrieve (1-1000)"
)
_MAX_CHUNK_SIZE_FIELD = Field(
    default=2000, ge=1, le=100_000,
    description="Maximum chunk size in characters"
)
_TEMPERATURE_FIELD = Field(
    default=0.0, ge=0.0, le=2.0,
    description="Sampling temperature (0.0-2.0)"
)
_MAX_TOKENS_FIELD = Field(
    default=256, ge=1, le=32_768,
    description="Maximum tokens to generate (1-32768)"
)
_TOP_CONTEXT_CHUNKS_FIELD = Field(
    default=3, ge=1, le=50,
    description="Number of context chunks (1-50)"
)
_MAX_CONTEXT_CHARS_FIELD = Field(
    default=12_000, ge=100, le=200_000,
    description="Max context characters (100-200000)"
)
_TIMEOUT_FIELD = Field(
    default=60.0, gt=0.0, le=3600.0,
    description="Timeout in seconds (>0, <=3600)"
)
_CONCURRENCY_FIELD = Field(
    default=1, ge=1, le=64,
    description="Concurrency level (1-64)"
)
_CHECKPOINT_INTERVAL_FIELD = Field(
    default=1, ge=1,
    description="Checkpoint every N answers (>=1)"
)


class IndexParams(BaseModel):
    folder_path: str = "data/raw/vllm-0.10.1"
    index_path: str = "data/processed/bm25_index"
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"folder not found: '{value}'")
        if not path.is_dir():
            raise ValueError(f"path is not a directory: '{value}'")
        return value


class SearchParams(BaseModel):
    query: str
    k: int = _K_FIELD
    folder_path: str = "data/raw/vllm-0.10.1"
    index_path: str = "data/processed/bm25_index"
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty or whitespace")
        return value

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"folder not found: '{value}'")
        if not path.is_dir():
            raise ValueError(f"path is not a directory: '{value}'")
        return value


class SearchDatasetParams(BaseModel):
    dataset_path: str
    k: int = _K_FIELD
    save_directory: str = "data/output/search_results"
    folder_path: str = "data/raw/vllm-0.10.1"
    index_path: str = "data/processed/bm25_index"
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("dataset_path")
    @classmethod
    def dataset_must_be_json(cls, value: str) -> str:
        _json_file(value)
        return value

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"folder not found: '{value}'")
        if not path.is_dir():
            raise ValueError(f"path is not a directory: '{value}'")
        return value


class AnswerParams(BaseModel):
    question: str
    k: int = _K_FIELD
    model: str = "Qwen/Qwen3-0.6B"
    base_url: str = "http://localhost:8000/v1"
    top_context_chunks: int = _TOP_CONTEXT_CHUNKS_FIELD
    max_context_chars: int = _MAX_CONTEXT_CHARS_FIELD
    temperature: float = _TEMPERATURE_FIELD
    max_tokens: int = _MAX_TOKENS_FIELD
    timeout_seconds: float = _TIMEOUT_FIELD
    folder_path: str = "data/raw/vllm-0.10.1"
    index_path: str = "data/processed/bm25_index"
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace")
        return value

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError(
                f"base_url must start with http:// or https://, got: '{value}'"
            )
        return value.rstrip("/")

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model name must not be empty")
        return value

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        path = Path(value)
        if not path.exists():
            raise ValueError(f"folder not found: '{value}'")
        if not path.is_dir():
            raise ValueError(f"path is not a directory: '{value}'")
        return value

    @model_validator(mode="after")
    def top_chunks_le_k(self) -> "AnswerParams":
        if self.top_context_chunks > self.k:
            raise ValueError(
                f"top_context_chunks ({self.top_context_chunks}) "
                f"must be <= k ({self.k})"
            )
        return self


class AnswerDatasetParams(BaseModel):
    student_search_results_path: str
    model: str = "Qwen/Qwen3-0.6B"
    base_url: str = "http://localhost:8000/v1"
    top_context_chunks: int = _TOP_CONTEXT_CHUNKS_FIELD
    max_context_chars: int = _MAX_CONTEXT_CHARS_FIELD
    temperature: float = _TEMPERATURE_FIELD
    max_tokens: int = _MAX_TOKENS_FIELD
    timeout_seconds: float = Field(default=600.0, gt=0.0, le=3600.0)
    concurrency: int = _CONCURRENCY_FIELD
    checkpoint_interval: int = _CHECKPOINT_INTERVAL_FIELD
    save_directory: str = "data/output/search_results_and_answer"

    @field_validator("student_search_results_path")
    @classmethod
    def results_must_be_json(cls, value: str) -> str:
        _json_file(value)
        return value

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError(
                f"base_url must start with http:// or https://, got: '{value}'"
            )
        return value.rstrip("/")

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model name must not be empty")
        return value


class EvaluateParams(BaseModel):
    student_results_path: str
    dataset_path: str
    minimal_iou_threshold: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="IoU threshold (0.0-1.0)"
    )
    threshold: float | None = Field(
        default=None,
        description="Pass/fail recall@5 threshold (0.0-1.0)"
    )

    @field_validator("student_results_path")
    @classmethod
    def results_must_be_json(cls, value: str) -> str:
        _json_file(value)
        return value

    @field_validator("dataset_path")
    @classmethod
    def dataset_must_be_json(cls, value: str) -> str:
        _json_file(value)
        return value

    @field_validator("threshold", mode="before")
    @classmethod
    def threshold_range(cls, value: Any) -> Any:
        if value is None:
            return value
        fval = float(value)
        if not (0.0 <= fval <= 1.0):
            raise ValueError(
                f"threshold must be between 0.0 and 1.0, got {fval}"
            )
        return fval
