import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..indexing import load_or_build_index
from ..models import SearchResult
from ..retrieval import search
from ..config import Config
from ..validation import IndexParams
from .pipeline_common import load_questions


def _question_text(question: dict[str, Any]) -> str:
    """Return the question text for a question dict.

    Args:
        question: A mapping representing a question; expected to have a
            `question` key.

    Returns:
        The question text coerced to `str` or an empty string if absent.
    """

    return str(question.get("question", ""))


def _retrieved_sources(
        results: list[SearchResult], k: int) -> list[dict[str, Any]]:
    """Return the minimal source descriptors for the top-k results.

    Args:
        results: A list of `SearchResult` objects.
        k: Number of top results to include.

    Returns:
        A list of dicts each containing `file_path`,
        `first_character_index`, and `last_character_index` for the top-k
        results.
    """

    return [result.to_source_dict() for result in results[:k]]


def _save_search_results(
        payload: dict[str, Any], path: Path) -> str:
    """Persist a search-results payload to `output_path` as JSON.

    Args:
        payload: The dictionary payload to write (will be JSON-serialized).
        output_path: Destination file path for the JSON output.

    Returns:
        The `Path` to the written file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(path)


def search_dataset(
        dataset_path: str | Path,
        *,
        k: int = 10,
        retriever: Any | None = None,
        corpus: list[dict[str, Any]] | None = None,
        max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE) -> dict[str, Any]:
    """Search each question in a dataset and return retrieval outputs.

    This function loads or builds an index if no `retriever` and `corpus`
    are provided, then runs the retriever for each question, collecting
    the top-k source descriptors.

    Args:
        dataset_path: Path to the dataset file containing questions.
        k: Number of retrieval results to return per question.
        retriever: Optional pre-built retriever instance.
        corpus: Optional corpus associated with `retriever`.
        max_chunk_size: Chunk size to use when building an index if one
            must be created.

    Returns:
        A dict with keys `search_results` (list of per-question dicts)
        and `k` (the retrieval `k`).
    """

    if retriever is None or corpus is None:
        retriever, corpus = load_or_build_index(index_params=IndexParams(
            max_chunk_size=max_chunk_size,
        ))

    questions = load_questions(dataset_path)
    search_results: list[dict[str, Any]] = []

    for question in tqdm(questions, desc="Searching questions"):
        results = search(_question_text(question), retriever, corpus, k=k)
        search_results.append({
            "question_id": question.get("question_id"),
            "question_str": _question_text(question),
            "retrieved_sources": _retrieved_sources(results, k),
        })

    return {
        "search_results": search_results,
        "k": k,
    }


def search_dataset_to_file(
        dataset_path: str | Path,
        *,
        k: int = 10,
        output_dir: str = Config.DEFAULT_OUTPUT_DIR_ANSWER,
        retriever: Any | None = None,
        corpus: list[dict[str, Any]] | None = None,
        max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE) -> str:
    """Run `search_dataset` and write the search results to a file.

    Args:
        dataset_path: Path to the dataset file.
        k: Number of retrieval results per question.
        output_dir: Directory where the output file will be written.
        retriever: Optional pre-built retriever instance.
        corpus: Optional corpus associated with `retriever`.
        max_chunk_size: Chunk size to use when building an index if one
            must be created.

    Returns:
        The `Path` to the saved search-results file.
    """

    payload = search_dataset(
        dataset_path=dataset_path,
        k=k,
        retriever=retriever,
        corpus=corpus,
        max_chunk_size=max_chunk_size,
    )
    output_path = Path(output_dir) / Path(dataset_path).name
    return _save_search_results(payload, output_path)
