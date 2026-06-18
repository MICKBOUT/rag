from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import uuid
import json

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .config import Config


_K_FIELD = Field(
    default=10,
    ge=1,
    le=1000,
    description="Number of results to retrieve (1-1000)",
)
_MAX_CHUNK_SIZE_FIELD = Field(
    default=Config.DEFAULT_MAX_CHUNK_SIZE,
    ge=1,
    le=100_000,
    description="Maximum chunk size in characters",
)
_MAX_TOKENS_FIELD = Field(
    default=Config.DEFAULT_MAX_TOKENS,
    ge=1,
    le=32_768,
    description="Maximum tokens to generate",
)
_TOP_CONTEXT_CHUNKS_FIELD = Field(
    default=Config.DEFAULT_TOP_CONTEXT_CHUNKS,
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


# fire data validation
class StrictBaseModel(BaseModel):
    """Base model with strict Pydantic configuration.

    Models inheriting from `StrictBaseModel` will forbid extra fields
    and enforce strict types, reducing surprises when parsing user
    input.
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )


def validate_existing_directory(path_str: str) -> str:
    """Validate that `path_str` points to an existing directory.

    Args:
        path_str: The path string to validate.

    Returns:
        The original `path_str` if validation succeeds.

    Raises:
        ValueError: If the path does not exist or is not a directory.
    """

    path = Path(path_str)

    if not path.exists():
        raise ValueError(f"directory not found: '{path_str}'")

    if not path.is_dir():
        raise ValueError(f"path is not a directory: '{path_str}'")

    return path_str


def validate_existing_file(path_str: str) -> str:
    """Validate that `path_str` points to an existing file.

    Args:
        path_str: The path string to validate.

    Returns:
        The original `path_str` if validation succeeds.

    Raises:
        ValueError: If the path does not exist or is not a file.
    """

    path = Path(path_str)

    if not path.exists():
        raise ValueError(f"file not found: '{path_str}'")

    if not path.is_file():
        raise ValueError(f"path is not a file: '{path_str}'")

    return path_str


class IndexParams(StrictBaseModel):
    folder_path: str = Config.RAW_ROOT
    index_path: str = Config.INDEX_PATH
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        """Pydantic field validator ensuring the index folder exists.

        Args:
            value: The folder path string to validate.

        Returns:
            The validated folder path string.
        """

        return validate_existing_directory(value)


class SearchParams(StrictBaseModel):
    query: str
    k: int = _K_FIELD
    folder_path: str = Config.RAW_ROOT
    index_path: str = Config.INDEX_PATH
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, value: str) -> str:
        """Ensure the `query` field is not empty or whitespace only.

        Args:
            value: The query string to validate.

        Returns:
            The validated query string.

        Raises:
            ValueError: If the query is empty after stripping.
        """

        if not value.strip():
            raise ValueError("query must not be empty")
        return value

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        """Ensure the folder path exists for search operations.

        Delegates to `validate_existing_directory`.
        """

        return validate_existing_directory(value)


class SearchDatasetParams(StrictBaseModel):
    dataset_path: str
    k: int = _K_FIELD
    save_directory: str | Path = Config.DEFAULT_OUTPUT_DIR
    folder_path: str = Config.RAW_ROOT
    index_path: str = Config.INDEX_PATH
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("dataset_path")
    @classmethod
    def dataset_must_exist(cls, value: str) -> str:
        """Validate that the provided dataset file exists.

        Args:
            value: Path to the dataset file.

        Returns:
            The validated file path string.
        """
        validate_existing_file(value)

        with open(value, "r", encoding="utf-8") as f:
            content = json.load(f)
        RagDatasetUnanswered.model_validate(content)
        return value

    @field_validator("folder_path")
    @classmethod
    def folder_must_exist(cls, value: str) -> str:
        """Validate that the source folder for the dataset exists.

        Delegates to `validate_existing_directory`.
        """
        return validate_existing_directory(value)


class AnswerParams(StrictBaseModel):
    question: str
    k: int = _K_FIELD
    model: str = Config.DEFAULT_MODEL
    base_url: str = Config.DEFAULT_BASE_URL
    top_context_chunks: int | None = _TOP_CONTEXT_CHUNKS_FIELD
    max_tokens: int = _MAX_TOKENS_FIELD
    timeout_seconds: float = _TIMEOUT_FIELD
    folder_path: str = Config.RAW_ROOT
    index_path: str = Config.INDEX_PATH
    max_chunk_size: int = _MAX_CHUNK_SIZE_FIELD

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, value: str) -> str:
        """Ensure the `question` field is not empty or whitespace only.

        Args:
            value: The question string to validate.

        Returns:
            The validated question string.

        Raises:
            ValueError: If the question is empty after stripping.
        """

        if not value.strip():
            raise ValueError("question must not be empty")
        return value

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, value: str) -> str:
        """Validate that a non-empty `model` name was provided.

        Args:
            value: The model name string.

        Returns:
            The validated model string.

        Raises:
            ValueError: If the model name is empty after stripping.
        """

        if not value.strip():
            raise ValueError("model must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        """Ensure `base_url` begins with http:// or https://.

        Trims any trailing slash from the value before returning it.

        Args:
            value: The base URL string to validate.

        Returns:
            The normalized base URL with no trailing slash.

        Raises:
            ValueError: If the URL does not start with an allowed scheme.
        """

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
        """Validate that the folder path exists for answer generation.

        Delegates to `validate_existing_directory`.
        """

        return validate_existing_directory(value)

    @model_validator(mode="after")
    def top_chunks_must_be_le_k(self) -> AnswerParams:
        """Model-level validator ensuring `top_context_chunks <= k`.

        This validator runs after model initialization and raises a
        `ValueError` if `top_context_chunks` is greater than `k`.

        Returns:
            The validated model instance (`self`).
        """

        if (
            self.top_context_chunks is not None and
            self.top_context_chunks > self.k
        ):
            raise ValueError(
                f"top_context_chunks ({self.top_context_chunks}) "
                f"must be <= k ({self.k})"
            )
        return self


class AnswerDatasetParams(StrictBaseModel):
    student_search_results_path: str
    model: str = Config.DEFAULT_MODEL
    base_url: str = Config.DEFAULT_BASE_URL
    top_context_chunks: int | None = _TOP_CONTEXT_CHUNKS_FIELD
    max_tokens: int = _MAX_TOKENS_FIELD
    timeout_seconds: float = Field(default=600.0, gt=0.0, le=3600.0)
    concurrency: int = Field(default=1, ge=1, le=128)
    checkpoint_interval: int = Field(default=1, ge=1)
    save_directory: str | Path = Config.DEFAULT_OUTPUT_DIR_ANSWER

    @field_validator("student_search_results_path")
    @classmethod
    def file_must_exist(cls, value: str) -> str:
        """Validate that the provided student search results file exists.

        Args:
            value: Path to the student search results file.

        Returns:
            The validated file path string.
        """

        return validate_existing_file(value)

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, value: str) -> str:
        """Ensure `model` is not empty for dataset answering.

        Args:
            value: The model string to validate.

        Returns:
            The validated model string.
        """

        if not value.strip():
            raise ValueError("model must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        """Validate `base_url` begins with an HTTP scheme and normalize it.

        Returns the value with any trailing slash removed.
        """

        if not (
            value.startswith("http://")
            or value.startswith("https://")
        ):
            raise ValueError(
                "base_url must start with "
                "'http://' or 'https://'"
            )

        return value.rstrip("/")


class EvaluateParams(StrictBaseModel):
    student_results_path: str
    dataset_path: str
    minimal_iou_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("student_results_path")
    @classmethod
    def student_results_must_exist(cls, value: str) -> str:
        """Validate that the student's results file exists.

        Args:
            value: Path to the student's results file.

        Returns:
            The validated file path string.
        """
        return validate_existing_file(value)

    @field_validator("dataset_path")
    @classmethod
    def dataset_must_exist(cls, value: str) -> str:
        """Validate that the dataset file exists for evaluation.

        Args:
            value: Path to the dataset file.

        Returns:
            The validated file path string.
        """
        validate_existing_file(value)
        with open(value, "r", encoding="utf-8") as f:
            content = json.load(f)
        RagDatasetAnswered.model_validate(content)
        return value


class SearchDatasetOutput(StrictBaseModel):
    file_path: str

    @field_validator("file_path")
    @classmethod
    def dataset_check(cls, value: str) -> str:
        validate_existing_file(value)
        with open(value, "r", encoding="utf-8") as f:
            content = json.load(f)
        StudentSearchResults.model_validate(content)
        return value


class AnswerDatasetOutput(StrictBaseModel):
    file_path: str

    @field_validator("file_path")
    @classmethod
    def dataset_check(cls, value: str) -> str:
        validate_existing_file(value)
        with open(value, "r", encoding="utf-8") as f:
            content = json.load(f)
        StudentSearchResultsAndAnswer.model_validate(content)
        return value


# datasets validataion
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class RagDatasetUnanswered(BaseModel):
    rag_questions: list[UnansweredQuestion]


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDatasetAnswered(BaseModel):
    rag_questions: list[AnsweredQuestion]


# not implemented yet
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: Sequence[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: Sequence[MinimalAnswer]
