from collections.abc import Sequence
from typing import Any

import bm25s

from .models import SearchResult
from .config import Config

_STOPWORDS = "en"


def _resolve_entry(
        raw: Any, corpus: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Resolve a retrieval hit to a corpus entry.

    The `raw` value returned by the retriever may be either a mapping
    (already the corpus entry) or an index into the `corpus`. This
    helper normalizes both cases and returns the dictionary describing
    the corpus entry.

    Args:
        raw: Either a dict representing a corpus entry or an index
            (int or stringified int) into `corpus` identifying the
            entry.
        corpus: Sequence of corpus entry dicts that `raw` may index
            into if it is not already a dict.

    Returns:
        The resolved corpus entry as a dict.
    """

    if isinstance(raw, dict):
        return raw
    return corpus[int(raw)]


def search(
        question: str,
        retriever: bm25s.BM25,
        corpus: Sequence[dict[str, Any]],
        k: int = Config.DEFAULT_SEARCH_K) -> list[SearchResult]:
    """Run BM25 retrieval for a question and return typed search results.

    This function tokenizes the question, retrieves the top-k documents
    using the provided `retriever`, and converts each hit into a
    `SearchResult` instance with rank and score information.

    Args:
        question: question string to search for.
        retriever: A `bm25s.BM25` retriever instance already indexed.
        corpus: The corpus sequence used by the retriever; used to
            resolve doc indices to full entries if necessary.
        k: Number of top documents to retrieve (default: 5).

    Returns:
        A list of `SearchResult` objects in rank order (1..k).
    """

    question_tokens = bm25s.tokenize(question, stopwords=_STOPWORDS)
    docs, scores = retriever.retrieve(question_tokens, k=k)

    results: list[SearchResult] = []
    for rank, raw in enumerate(docs[0], start=1):
        entry = _resolve_entry(raw, corpus)
        results.append(
            SearchResult.from_entry(
                entry=entry,
                rank=rank,
                score=float(scores[0, rank - 1]),
                question=question
            )
        )

    return results
