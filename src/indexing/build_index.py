from typing import Any
import json

import bm25s

from .index_common import resolve_repo_path, _metadata_path
from ..config import Config
from ..parsing import get_ready_to_index_data
from ..validation import IndexParams


def _entry_text(entry: dict[str, Any]) -> str:
    """Return the text content for a corpus entry.

    Args:
        entry: A corpus entry mapping expected to contain a "text"
            key.

    Returns:
        The `text` value coerced to `str`, or an empty string if
        missing.
    """
    return str(entry.get("text", ""))


def _write_chunks(corpus: list[dict[str, Any]]) -> None:
    """Write the serialized corpus to the configured chunks file.

    Each entry is written as a single JSON line to `Config.CHUNKS_PATH`.

    Args:
        corpus: A list of corpus entry dicts to persist.
    """
    path = resolve_repo_path(Config.CHUNKS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in corpus:
            handle.write(
                json.dumps(entry, ensure_ascii=False) + "\n"
            )


def _limit_entry_text(
        entry: dict[str, Any], max_chunk_size: int) -> dict[str, Any]:
    """Return a copy of `entry` whose text/span is clipped to `max_chunk_size`.

    This helper ensures that both the BM25 `text` and the source span
    (`last_character_index`) do not exceed `max_chunk_size`. If clipping
    is necessary a shallow copy of `entry` is returned with the
    adjusted fields; otherwise the original `entry` is returned.

    Args:
        entry: The corpus entry to potentially limit.
        max_chunk_size: Maximum allowed characters for text/span.

    Returns:
        A dict representing the possibly-limited entry.
    """
    text = entry.get("text", "")
    span_length = max(
        0, (
            int(entry["last_character_index"]) -
            int(entry["first_character_index"])
        )
    )

    limited_entry = dict(entry)
    if span_length <= max_chunk_size:
        return entry

    limited_entry["text"] = text[:max_chunk_size]
    limited_entry["last_character_index"] = (
        limited_entry["first_character_index"] + max_chunk_size
    )

    return limited_entry


def _write_index_metadata(
        index_path: str, *, max_chunk_size: int, folder_path: str) -> None:
    """Write index metadata to the index folder.

    Stores information about how the index was created so subsequent
    loads can validate compatibility (e.g. `max_chunk_size`).

    Args:
        index_path: Path to the index directory.
        max_chunk_size: Maximum chunk size used when building the
            index.
        folder_path: The source folder path that was indexed.
    """
    metadata = {
        "max_chunk_size": max_chunk_size,
        "folder_path": folder_path,
    }
    _metadata_path(index_path).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_and_save_index(
        index_params: IndexParams
        ) -> tuple[bm25s.BM25, list[dict[str, Any]]]:
    """Build a BM25 index from source files and persist it.

    This function collects ready-to-index data from `folder_path`,
    applies chunk size limiting, tokenizes the corpus for BM25,
    builds and saves the retriever and also persists the chunk file
    and index metadata.

    Args:
        folder_path: Source folder to index.
        index_path: Path where the BM25 index will be saved.
        max_chunk_size: Maximum chunk size used when generating
            corpus entries.

    Returns:
        A tuple containing the constructed `bm25s.BM25` retriever and
        the list of corpus entry dicts.

    Raises:
        ValueError: If no corpus entries were produced from the source
            folder.
    """
    resolved_index_path = resolve_repo_path(index_params.index_path)
    corpus = [
        _limit_entry_text(entry, index_params.max_chunk_size)
        for entry in get_ready_to_index_data(index_params.folder_path)
    ]
    if not corpus:
        raise ValueError(
            f"No source files were indexed from {index_params.folder_path}. "
            "Check that the corpus path is correct."
        )
    corpus_texts = [_entry_text(entry) for entry in corpus]

    corpus_tokens = bm25s.tokenize(corpus_texts)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(resolved_index_path), corpus=corpus)
    _write_chunks(corpus)
    _write_index_metadata(
        str(resolved_index_path),
        max_chunk_size=index_params.max_chunk_size,
        folder_path=index_params.folder_path,
    )

    print(f"Index built and saved ({len(corpus)} documents).")
    return retriever, corpus
