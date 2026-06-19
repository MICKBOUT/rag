from typing import Any, cast
from pathlib import Path
import json

import bm25s

from . import build_and_save_index
from .index_common import resolve_repo_path, _metadata_path
from ..validation import IndexParams


def _corpus_has_absolute_paths(corpus: list[dict[str, Any]]) -> bool:
    """Return True if any corpus entry contains an absolute file path.

    This is used as a safety check to detect indexes created on another
    machine or with absolute paths that would not be portable.

    Args:
        corpus: List of corpus entries, each possibly containing a
            `file_path` key.

    Returns:
        `True` if any `file_path` is absolute, otherwise `False`.
    """

    for entry in corpus:
        file_path = str(entry.get("file_path", ""))
        if Path(file_path).is_absolute():
            return True
    return False


def _read_index_metadata(index_path: str) -> dict[str, Any] | None:
    """Read metadata.json from an index path if present.

    Args:
        index_path: Path to the index folder.

    Returns:
        The parsed metadata dict, or `None` if the metadata file does
        not exist.
    """

    path = _metadata_path(index_path)
    if not path.exists():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], metadata)


def _load_index(
        index_params: IndexParams
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
    """Load a saved BM25 index and validate compatibility.

    This attempts to load the index from `index_path` and performs a
    series of checks to ensure the corpus exists, uses relative paths,
    and that the stored `max_chunk_size` matches the requested value.

    Args:
        index_path: Path to the saved BM25 index.
        max_chunk_size: Expected max chunk size for compatibility.

    Returns:
        A tuple of the loaded `bm25s.BM25` retriever and the corpus list.

    Raises:
        ValueError: When the index is missing required data or appears
            incompatible and should be rebuilt.
    """

    resolved_index_path = resolve_repo_path(index_params.index_path)
    retriever = bm25s.BM25.load(str(resolved_index_path), load_corpus=True)
    corpus = retriever.corpus

    if corpus is None:
        raise ValueError("Corpus is None — rebuilding.")

    normalized_corpus = list(corpus)
    if _corpus_has_absolute_paths(normalized_corpus):
        raise ValueError("Corpus paths are absolute — rebuilding.")

    metadata = _read_index_metadata(str(resolved_index_path))
    if metadata is None:
        raise ValueError("Index metadata missing — rebuilding.")
    if int(metadata.get("max_chunk_size", -1)) != index_params.max_chunk_size:
        raise ValueError("Index chunk size changed — rebuilding.")

    return retriever, normalized_corpus


def load_or_build_index(
        index_params: IndexParams
) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
    """Load an existing index or build a new one if loading fails.

    This convenience wrapper first attempts to load the index using
    `_load_index`. If that fails for any reason it falls back to
    `build_and_save_index` to create a fresh index from `folder_path`.

    Args:
        folder_path: Source folder to index if building is required.
        index_path: Path to the index to load or create.
        max_chunk_size: Max chunk size parameter for load/build.

    Returns:
        A tuple with the retriever and the normalized corpus list.
    """

    try:
        return _load_index(index_params)
    except Exception:
        print("Index not loaded, building from scratch")
        return build_and_save_index(index_params)
