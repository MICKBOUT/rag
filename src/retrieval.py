from collections.abc import Sequence
from typing import Any

import bm25s

from models import SearchResult

_STOPWORDS = "en"


def _resolve_entry(
        raw: Any, corpus: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return corpus[int(raw)]


def search(
        query: str,
        retriever: bm25s.BM25,
        corpus: Sequence[dict[str, Any]],
        k: int = 5) -> list[SearchResult]:
    query_tokens = bm25s.tokenize(query, stopwords=_STOPWORDS)
    docs, scores = retriever.retrieve(query_tokens, k=k)

    results: list[SearchResult] = []
    for rank, raw in enumerate(docs[0], start=1):
        entry = _resolve_entry(raw, corpus)
        results.append(
            SearchResult.from_entry(
                entry,
                rank=rank,
                score=float(scores[0, rank - 1]),
            )
        )

    return results
