import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from indexing import load_or_build_index
from models import SearchResult
from retrieval import search


DEFAULT_OUTPUT_DIR = Path("data/output/search_results")
MIN_IOU_THRESHOLD = 0.05


@dataclass(slots=True)
class EvaluationSummary:
    questions_evaluated: int
    questions_with_sources: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
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
    payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    questions = payload.get("rag_questions", [])
    if not isinstance(questions, list):
        raise ValueError(
            "Invalid dataset format: rag_questions must be a list")
    return list(questions)


def _question_text(question: dict[str, Any]) -> str:
    return str(question.get("question", ""))


def _retrieved_sources(
        results: list[SearchResult], k: int) -> list[dict[str, Any]]:
    return [result.to_source_dict() for result in results[:k]]


def search_dataset(
        dataset_path: str | Path,
        *,
        k: int = 10,
        retriever: Any | None = None,
        corpus: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if retriever is None or corpus is None:
        retriever, corpus = load_or_build_index()

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
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        retriever: Any | None = None,
        corpus: list[dict[str, Any]] | None = None) -> Path:
    payload = search_dataset(
        dataset_path,
        k=k,
        retriever=retriever,
        corpus=corpus,
    )
    output_path = Path(output_dir) / Path(dataset_path).name
    return save_search_results(payload, output_path)


def _interval_iou(
        left: dict[str, Any], right: dict[str, Any]) -> float:
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
        minimal_iou_threshold: float = MIN_IOU_THRESHOLD,
        threshold: float | None = None) -> EvaluationSummary:
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
    print(f"Questions evaluated: {summary.questions_evaluated}")
    print(f"Questions with sources: {summary.questions_with_sources}")
    print(f"Recall@1: {summary.recall_at_1:.3f}")
    print(f"Recall@3: {summary.recall_at_3:.3f}")
    print(f"Recall@5: {summary.recall_at_5:.3f}")
    print(f"Recall@10: {summary.recall_at_10:.3f}")
    if threshold is not None:
        status = "PASS" if summary.passed else "FAIL"
        print(f"{status} (threshold={threshold:.2f})")

    return summary
