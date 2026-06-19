from pathlib import Path
from typing import Any
import json


def load_questions(dataset_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate questions from a dataset JSON file.

    The dataset file is expected to contain a top-level `rag_questions`
    list. If the file cannot be read or contains invalid JSON, a
    `RuntimeError` is raised. If the `rag_questions` field is missing
    or not a list, a `ValueError` is raised.

    Args:
        dataset_path: Path to the dataset JSON file.

    Returns:
        A list of question dicts extracted from the dataset.
    """

    path = Path(dataset_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(
            f"Could not read dataset file '{path}': {e}"
        ) from e
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Dataset file '{path}' contains invalid JSON: {e}"
        ) from e
    questions = payload.get("rag_questions", [])
    if not isinstance(questions, list):
        raise ValueError(
            "Invalid dataset format: rag_questions must be a list")
    return list(questions)
