from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..config import Config
from .parse_py_file import get_ready_to_index_py_file
from .parse_md_file import get_ready_to_index_md_file
from .parser_common import IndexChunk


def get_ready_to_index_data(
        folder_path: str = Config.RAW_ROOT) -> list[dict[str, Any]]:
    """
    Traverse a folder tree and produce indexable chunks for Python and
    Markdown files.

    This is the top-level orchestration that:
    - Walks the given folder for .py files, parses them to extract
        class/function chunks.
    - Walks for .md files and produces sliding-window markdown chunks with
        heading context.
    - Returns a list of serialized IndexChunk records ready for
        indexing/storage.

    Args:
        folder_path (str): Root folder to scan (relative or absolute).
            Defaults to Config.RAW_ROOT

    Returns:
        list[dict[str, Any]]: A list of dictionaries produced by
            IndexChunk.to_record() for all discovered chunks.
    """
    candidate = Path(folder_path)
    if candidate.is_absolute():
        folder_root = candidate
    else:
        folder_root = (Config.REPO_ROOT / candidate).resolve()

    clean_data_lst: list[IndexChunk] = []
    py_files = list(folder_root.rglob("*.py"))
    for file in tqdm(py_files, "Parsing python files"):
        clean_data_lst.extend(
            get_ready_to_index_py_file(file))

    md_files = list(folder_root.rglob("*.md")) + list(
        folder_root.rglob("*.txt"))
    for file in tqdm(md_files, "Parsing markdown files"):
        clean_data_lst.extend(
            get_ready_to_index_md_file(file))

    return [chunk.to_record() for chunk in clean_data_lst]


if __name__ == "__main__":
    get_ready_to_index_data()
