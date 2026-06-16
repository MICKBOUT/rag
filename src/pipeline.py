import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from indexing import load_or_build_index
from models import SearchResult
from retrieval import search
from config import Config


@dataclass(slots=True)
class EvaluationSummary:
    """Summary statistics for an evaluation run.

    Attributes:
        questions_evaluated: Total number of questions considered.
        questions_with_sources: Number of questions that had gold sources.
        recall_at_1: Average recall@1 across evaluated questions.
        recall_at_3: Average recall@3 across evaluated questions.
        recall_at_5: Average recall@5 across evaluated questions.
        recall_at_10: Average recall@10 across evaluated questions.
        passed: Optional boolean indicating whether the evaluation met
            a provided threshold for success; `None` if not applicable.
    """

    questions_evaluated: int
    questions_with_sources: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the `EvaluationSummary` to a JSON-serializable dict.

        Returns:
            A dict containing the summary statistics.
        """

        return {
            "questions_evaluated": self.questions_evaluated,
            "questions_with_sources": self.questions_with_sources,
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "passed": self.passed,
        }


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
        retriever, corpus = load_or_build_index(max_chunk_size=max_chunk_size)

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


def save_search_results(
        payload: dict[str, Any], output_path: str | Path) -> Path:
    """Persist a search-results payload to `output_path` as JSON.

    Args:
        payload: The dictionary payload to write (will be JSON-serialized).
        output_path: Destination file path for the JSON output.

    Returns:
        The `Path` to the written file.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def search_dataset_to_file(
        dataset_path: str | Path,
        *,
        k: int = 10,
        output_dir: str | Path = Config.DEFAULT_OUTPUT_DIR_ANSWER,
        retriever: Any | None = None,
        corpus: list[dict[str, Any]] | None = None,
        max_chunk_size: int = Config.DEFAULT_MAX_CHUNK_SIZE) -> Path:
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
        dataset_path,
        k=k,
        retriever=retriever,
        corpus=corpus,
        max_chunk_size=max_chunk_size,
    )
    output_path = Path(output_dir) / Path(dataset_path).name
    return save_search_results(payload, output_path)


def _interval_iou(
        left: dict[str, Any], right: dict[str, Any]) -> float:
    """Compute the intersection-over-union (IoU) of two source spans.

    The IoU is computed only when both spans reference the same file.
    Indices are treated as inclusive ranges. If files differ or the
    ranges do not overlap, the function returns 0.0.

    Args:
        left: A source descriptor with `file_path`,
            `first_character_index`, and `last_character_index`.
        right: A second source descriptor to compare against.

    Returns:
        IoU as a float in [0.0, 1.0].
    """

    if left.get("file_path") != right.get("file_path"):
        return 0.0

    left_start = int(left.get("first_character_index", 0))
    left_end = int(left.get("last_character_index", 0))
    right_start = int(right.get("first_character_index", 0))
    right_end = int(right.get("last_character_index", 0))

    intersection_start = max(left_start, right_start)
    intersection_end = min(left_end, right_end)
    if intersection_end < intersection_start:
        return 0.0

    intersection = (intersection_end - intersection_start) + 1
    left_size = (left_end - left_start) + 1
    right_size = (right_end - right_start) + 1
    union = left_size + right_size - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _recall_at_k(
        student_sources: list[dict[str, Any]],
        gold_sources: list[dict[str, Any]],
        threshold: float) -> float:
    """Compute recall@k comparing student and gold source lists.

    For each gold source this function checks whether any of the
    provided `student_sources` has IoU >= `threshold` with it. Recall
    is defined as the fraction of gold sources that were matched.

    Args:
        student_sources: List of source descriptors returned by the
            system (top-k for a question).
        gold_sources: List of gold source descriptors for the question.
        threshold: IoU threshold to consider a match successful.

    Returns:
        A float in [0.0, 1.0] representing recall for the provided
        gold source set.
    """

    if not gold_sources:
        return 0.0

    found = 0
    for gold_source in gold_sources:
        if any(
            _interval_iou(student_source, gold_source) >= threshold
            for student_source in student_sources
        ):
            found += 1
    return found / len(gold_sources)


def evaluate_search_results(
        student_results_path: str | Path,
        dataset_path: str | Path,
        *,
        minimal_iou_threshold: float = Config.MIN_IOU_THRESHOLD,
        threshold: float | None = None) -> EvaluationSummary:
    """Evaluate a student's retrieval outputs against a gold dataset.

    The function reads the student's `search_results` JSON file and the
    gold dataset, computes recall@1/3/5/10 using IoU-based matching, and
    aggregates results into an `EvaluationSummary`.

    Args:
        student_results_path: Path to the student's search-results JSON.
        dataset_path: Path to the gold dataset JSON containing questions
            and gold `sources` for each question.
        minimal_iou_threshold: IoU threshold used to count a retrieved
            source as matching a gold source.
        threshold: Optional pass/fail threshold applied to recall@5. If
            provided, `EvaluationSummary.passed` will be set accordingly.

    Returns:
        An `EvaluationSummary` containing aggregated recall metrics and
        pass/fail status (if `threshold` was provided).
    """

    student_payload = json.loads(
        Path(student_results_path).read_text(encoding="utf-8"))
    student_results = student_payload.get("search_results", [])
    if not isinstance(student_results, list):
        raise ValueError(
            "Invalid student results format: search_results must be a list")

    student_by_id = {
        str(item.get("question_id")): item
        for item in student_results
    }
    dataset_questions = load_questions(dataset_path)

    totals = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
    questions_with_sources = 0

    for question in dataset_questions:
        question_id = str(question.get("question_id"))
        gold_sources = list(question.get("sources") or [])
        if not gold_sources:
            continue

        questions_with_sources += 1
        student_item = student_by_id.get(question_id, {})
        retrieved_sources = list(student_item.get("retrieved_sources") or [])

        for k in totals:
            totals[k] += _recall_at_k(
                retrieved_sources[:k],
                gold_sources,
                minimal_iou_threshold,
            )

    questions_evaluated = len(dataset_questions)
    if questions_with_sources == 0:
        summary = EvaluationSummary(
            questions_evaluated=questions_evaluated,
            questions_with_sources=0,
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            passed=None,
        )
    else:
        recall_at_1 = totals[1] / questions_with_sources
        recall_at_3 = totals[3] / questions_with_sources
        recall_at_5 = totals[5] / questions_with_sources
        recall_at_10 = totals[10] / questions_with_sources
        passed = None
        if threshold is not None:
            passed = recall_at_5 >= threshold
        summary = EvaluationSummary(
            questions_evaluated=questions_evaluated,
            questions_with_sources=questions_with_sources,
            recall_at_1=recall_at_1,
            recall_at_3=recall_at_3,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            passed=passed,
        )

    print("Evaluation Results")
    print("========================================")
    if threshold is not None:
        status = "PASS" if summary.passed else "FAIL"
        print(f"{status} (threshold={threshold:.2f})")

    return summary
