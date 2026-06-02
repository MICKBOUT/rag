from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SearchResult:
    rank: int
    score: float
    text: str
    file_path: str
    first_character_index: int
    last_character_index: int
    kind: str
    heading_path: list[str] = field(default_factory=list)
    symbol: str | None = None
    calls: list[str] = field(default_factory=list)

    @classmethod
    def from_entry(
            cls,
            entry: dict[str, Any],
            *,
            rank: int,
            score: float) -> "SearchResult":
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
        )

    def to_dict(self) -> dict[str, Any]:
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
        return {
            "file_path": self.file_path,
            "first_character_index": self.first_character_index,
            "last_character_index": self.last_character_index,
        }
