from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Shared fields
_K_FIELD = Field(
    default=10,
    ge=1,
    le=1000,
    description="Number of results to retrieve (1-1000)",
)
_MAX_CHUNK_SIZE_FIELD = Field(
    default=2000,
    ge=1,
    le=100_000,
    description="Maximum chunk size in characters",
)
_MAX_TOKENS_FIELD = Field(
    default=256,
    ge=1,
    le=32_768,
    description="Maximum tokens to generate",
)
_TOP_CONTEXT_CHUNKS_FIELD = Field(
    default=3,
    ge=1,
    le=50,
    description="Number of context chunks",
)
_TIMEOUT_FIELD = Field(
    default=60.0,
    gt=0.0,
    le=3600.0,
    description="Timeout in seconds",
)


# Base model
class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )


# Reusable validators
def validate_existing_directory(path_str: str) -> str:
    path = Path(path_str)

    if not path.exists():
        raise ValueError(f"directory not found: '{path_str}'")

    if not path.is_dir():
        raise ValueError(f"path is not a directory: '{path_str}'")

    return path_str


def validate_existing_file(path_str: str) -> str:
    path = Path(path_str)

    if not path.exists():
        raise ValueError(f"file not found: '{path_str}'")

    if not path.is_file():
        raise ValueError(f"path is not a file: '{path_str}'")

    return path_str


# Index
class IndexParams(StrictBaseModel):
    folder_path: str = "data/raw/vllm-0.10.1"
    index_path: str = "data/processed/bm25_index"
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        return validate_existing_directory(value)


# Search
class SearchParams(StrictBaseModel):
    query: str
    k: int = _K_FIELD
    folder_path: str = "data/raw/vllm-0.10.1"
    index_path: str = "data/processed/bm25_index"
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        return validate_existing_directory(value)


# Search Dataset
class SearchDatasetParams(StrictBaseModel):
    dataset_path: str
    k: int = _K_FIELD
    save_directory: str = "data/output/search_results"
    folder_path: str = "data/raw/vllm-0.10.1"
    index_path: str = "data/processed/bm25_index"
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("dataset_path")
    @classmethod
    def dataset_must_exist(cls, value: str) -> str:
        return validate_existing_file(value)

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        return validate_existing_directory(value)


# Answer
class AnswerParams(StrictBaseModel):
    question: str
    k: int = _K_FIELD
    model: str = "Qwen/Qwen3-0.6B"
    base_url: str = "http://localhost:8000/v1"
    top_context_chunks: int = _TOP_CONTEXT_CHUNKS_FIELD
    max_tokens: int = _MAX_TOKENS_FIELD
    timeout_seconds: float = _TIMEOUT_FIELD
    folder_path: str = "data/raw/vllm-0.10.1"
    index_path: str = "data/processed/bm25_index"
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        if not (
            value.startswith("http://")
            or value.startswith("https://")
        ):
            raise ValueError(
                "base_url must start with "
                "'http://' or 'https://'"
            )

        return value.rstrip("/")

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        return validate_existing_directory(value)

    @model_validator(mode="after")
    def top_chunks_must_be_le_k(self):
        if self.top_context_chunks > self.k:
            raise ValueError(
                f"top_context_chunks ({self.top_context_chunks}) "
                f"must be <= k ({self.k})"
            )
        return self


# Answer Dataset
class AnswerDatasetParams(StrictBaseModel):
    student_search_results_path: str
    model: str = "Qwen/Qwen3-0.6B"
    base_url: str = "http://localhost:8000/v1"
    top_context_chunks: int = _TOP_CONTEXT_CHUNKS_FIELD
    max_tokens: int = _MAX_TOKENS_FIELD
    timeout_seconds: float = Field(
        default=600.0,
        gt=0.0,
        le=3600.0,
    )
    concurrency: int = Field(
        default=1,
        ge=1,
        le=128,
    )
    checkpoint_interval: int = Field(
        default=1,
        ge=1,
    )
    save_directory: str = (
        "data/output/search_results_and_answer"
    )

    @field_validator("student_search_results_path")
    @classmethod
    def file_must_exist(cls, value: str) -> str:
        return validate_existing_file(value)

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        if not (
            value.startswith("http://")
            or value.startswith("https://")
        ):
            raise ValueError(
                "base_url must start with "
                "'http://' or 'https://'"
            )

        return value.rstrip("/")


# Evaluate
class EvaluateParams(StrictBaseModel):
    student_results_path: str
    dataset_path: str
    minimal_iou_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @field_validator("student_results_path")
    @classmethod
    def student_results_must_exist(cls, value: str) -> str:
        return validate_existing_file(value)

    @field_validator("dataset_path")
    @classmethod
    def dataset_must_exist(cls, value: str) -> str:
        return validate_existing_file(value)


class DatasetsParams(StrictBaseModel):
    root: str = "data/datasets"

    @field_validator("root")
    @classmethod
    def root_must_exist(cls, value: str) -> str:
        return validate_existing_directory(value)
