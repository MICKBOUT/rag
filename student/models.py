from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SearchResult:
    """Represents a single search result returned by the retriever.

    Attributes:
        rank: Rank position of the result (0-based or 1-based depending on
            upstream code).
        score: Relevance score produced by the retriever.
        text: The BM25-processed text of the chunk.
        file_path: Path to the source file for the chunk.
        first_character_index: Start index of the chunk in the source file.
        last_character_index: End index of the chunk in the source file.
        kind: Kind or type of the source (e.g. "code", "markdown").
        heading_path: Hierarchical heading path for the chunk if any.
        symbol: Optional symbol name associated with the chunk.
        calls: List of call targets referenced in the chunk.
    """

    rank: int
    score: float
    text: str
    file_path: str
    first_character_index: int
    last_character_index: int
    kind: str
    question: str
    heading_path: list[str] = field(default_factory=list)
    symbol: str | None = None
    calls: list[str] = field(default_factory=list)

    @classmethod
    def from_entry(
            cls,
            entry: dict[str, Any],
            question: str,
            *,
            rank: int,
            score: float) -> SearchResult:
        """Create a `SearchResult` from a raw corpus entry.

        Args:
            entry: Raw dictionary representing a corpus entry.
            rank: Rank to assign to the resulting `SearchResult`.
            score: Retriever score for this entry.

        Returns:
            A populated `SearchResult` instance.
        """

        return cls(
            rank=rank,
            score=score,
            text=str(entry.get("text", "")),
            file_path=str(entry.get("file_path", "unknown")),
            first_character_index=int(entry.get("first_character_index", 0)),
            last_character_index=int(entry.get("last_character_index", 0)),
            kind=str(entry.get("kind", "unknown")),
            heading_path=list(entry.get("heading_path") or []),
            symbol=entry.get("symbol"),
            calls=list(entry.get("calls") or []),
            question=question,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the `SearchResult` to a JSON-serializable dict.

        Returns:
            A dict containing the serializable fields of the result.
        """

        return {
            "rank": self.rank,
            "score": self.score,
            "text": self.text,
            "file_path": self.file_path,
            "first_character_index": self.first_character_index,
            "last_character_index": self.last_character_index,
            "kind": self.kind,
            "heading_path": self.heading_path,
            "symbol": self.symbol,
            "calls": self.calls,
        }

    def to_source_dict(self) -> dict[str, Any]:
        """Return a minimal dict identifying the source span for this result.

        The returned mapping contains only the `file_path`,
        `first_character_index`, and `last_character_index` which are
        sufficient to locate the original source text.

        Returns:
            A dict with keys `file_path`, `first_character_index`, and
            `last_character_index`.
        """

        return {
            "file_path": self.file_path,
            "first_character_index": self.first_character_index,
            "last_character_index": self.last_character_index,
        }


@dataclass(slots=True)
class GeneratedAnswer:
    """Represents a generated answer for a question.

    Attributes:
        question_id: Identifier for the question this answer corresponds to.
        question: The original question text.
        answer: Generated answer text.
        retrieved_sources: List of source descriptors used to produce the
            answer (each is a dict with `file_path`, `first_character_index`,
            `last_character_index`, etc.).
        model: Name of the language model used.
        base_url: Base API URL used for the LM provider.
        max_tokens: Maximum tokens requested for generation.
        search_k: Retrieval `k` used when collecting context.
        top_context_chunks: Number of top chunks included in the context.
    """

    question_id: str
    question: str
    answer: str
    retrieved_sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the `GeneratedAnswer` to a JSON-serializable dict.

        Returns:
            A dict suitable for JSON serialization and writing to disk.
        """

        return {
            "question_id": self.question_id,
            "question": self.question,
            "retrieved_sources": self.retrieved_sources,
            "answer": self.answer,
        }
