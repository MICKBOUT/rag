from typing import Any
from dataclasses import asdict, dataclass, field

from pathlib import Path
from ..config import Config


@dataclass(slots=True)
class IndexChunk:
    """
    Representation of a contiguous piece of source or markdown suitable for
        indexing.

    Attributes:
        text (str): The text content of the chunk to be indexed.
        file_path (str): Displayable path to the source file containing the
            chunk.
        first_character_index (int): Absolute character index of the first
            character of the chunk within the file.
        last_character_index (int): Absolute character index of the last
            character of the chunk within the file.
        kind (str): A short identifier describing the chunk type
            (e.g. 'python_class', 'markdown').
        heading_path (list[str]): Hierarchical heading path for Markdown
            chunks (defaults to empty).
        symbol (str | None): Fully qualified symbol name for code chunks
            (e.g. 'Class.method'), if applicable.
        calls (list[str]): List of function/method call names extracted from
            code chunks.
    """
    text: str
    file_path: str
    first_character_index: int
    last_character_index: int
    kind: str
    heading_path: list[str] = field(default_factory=list)
    symbol: str | None = None
    calls: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def display_path(path: Path) -> str:
    """
    Return a compact display path for a file, preferring repository-relative,
    then CWD-relative, else absolute.

    Args:
        path (Path): Path to display.

    Returns:
        str: String representation of path relative to repository root if
            possible, otherwise relative to the current working directory,
            otherwise the absolute path.
    """
    if path.is_absolute():
        try:
            return str(path.relative_to(Config.REPO_ROOT))
        except ValueError:
            try:
                return str(path.relative_to(Path.cwd()))
            except ValueError:
                return str(path)
    return str(path)
