import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm

from indexing import load_chunks, load_or_build_index
from models import GeneratedAnswer, SearchResult
from retrieval import search


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OUTPUT_DIR = Path("data/output/search_results_and_answer")
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_SEARCH_K = 10
DEFAULT_TOP_CONTEXT_CHUNKS = 3
DEFAULT_MAX_CONTEXT_CHARS = 12_000
DEFAULT_MAX_CHUNK_CHARS = 2_000
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_CONCURRENCY = 8
DEFAULT_CHECKPOINT_INTERVAL = 1
QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>"

SYSTEM_PROMPT = (
    "You answer questions about the vLLM repository using only the provided "
    "context. Do not explain your reasoning, do not show chain of thought, "
    "and do not invent details. Return only the final answer. If the "
    "context is insufficient, say that you cannot find a supported answer."
)


@dataclass(slots=True)
class GenerationConfig:
    model: str
    base_url: str = DEFAULT_BASE_URL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    search_k: int = DEFAULT_SEARCH_K
    top_context_chunks: int = DEFAULT_TOP_CONTEXT_CHUNKS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _select_context(
        results: list[SearchResult],
        config: GenerationConfig,
) -> list[SearchResult]:
    selected: list[SearchResult] = []
    total_chars = 0

    for result in results:
        if len(selected) >= config.top_context_chunks:
            break

        chunk_text = _truncate_text(result.text, DEFAULT_MAX_CHUNK_CHARS)
        projected = total_chars + len(chunk_text)
        if selected and projected > config.max_context_chars:
            break

        selected.append(result)
        total_chars = projected

    return selected


def _format_context_block(result: SearchResult, rank: int) -> str:
    heading = " > ".join(result.heading_path) if result.heading_path else ""
    calls = ", ".join(result.calls) if result.calls else ""
    lines = [
        f"[{rank}] FILE: {result.file_path}",
        (
            "SPAN: "
            f"{result.first_character_index}-{result.last_character_index}"
        ),
        f"KIND: {result.kind}",
        f"SCORE: {result.score:.4f}",
    ]
    if heading:
        lines.append(f"HEADING: {heading}")
    if result.symbol:
        lines.append(f"SYMBOL: {result.symbol}")
    if calls:
        lines.append(f"CALLS: {calls}")
    lines.append("TEXT:")
    lines.append(_truncate_text(result.text, DEFAULT_MAX_CHUNK_CHARS))
    return "\n".join(lines)


def _source_key(source: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(source.get("file_path", "")),
        int(source.get("first_character_index", 0)),
        int(source.get("last_character_index", 0)),
    )


def _build_corpus_lookup(
        corpus: list[dict[str, Any]]
) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (
            str(entry.get("file_path", "")),
            int(entry.get("first_character_index", 0)),
            int(entry.get("last_character_index", 0)),
        ): entry
        for entry in corpus
    }


def _format_source_block(
        source: dict[str, Any],
        entry: dict[str, Any],
        rank: int,
) -> str:
    heading_path = list(entry.get("heading_path") or [])
    heading = " > ".join(heading_path) if heading_path else ""
    kind = str(entry.get("kind", "unknown"))
    lines = [
        f"[{rank}] FILE: {source.get('file_path', 'unknown')}",
        (
            "SPAN: "
            f"{source.get('first_character_index', 0)}-"
            f"{source.get('last_character_index', 0)}"
        ),
        f"KIND: {kind}",
    ]
    if heading:
        lines.append(f"HEADING: {heading}")
    symbol = entry.get("symbol")
    if symbol:
        lines.append(f"SYMBOL: {symbol}")
    calls = entry.get("calls") or []
    if calls:
        lines.append(f"CALLS: {', '.join(str(call) for call in calls)}")
    lines.append("TEXT:")
    lines.append(
        _truncate_text(str(entry.get("text", "")), DEFAULT_MAX_CHUNK_CHARS)
    )
    return "\n".join(lines)


def _build_qwen_prompt(
        question: str,
        context: str,
) -> str:
    return (
        f"{QWEN_IM_START}system\n"
        f"{SYSTEM_PROMPT}{QWEN_IM_END}\n"
        f"{QWEN_IM_START}user\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer with only the final answer. /no_think\n"
        f"{QWEN_IM_END}\n"
        f"{QWEN_IM_START}assistant\n"
    )


def _call_openai_compatible_completion(
        *,
        model: str,
        prompt: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
) -> str:
    url = f"{base_url.rstrip('/')}/completions"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": [QWEN_IM_END],
        "top_p": 1.0,
        "stream": False,
    }).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"vLLM request failed with HTTP {error.code}: {body}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not reach vLLM server at {base_url}: {error.reason}"
        ) from error

    try:
        response_payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON returned by vLLM server: {raw[:500]}"
        ) from error

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"Missing choices in vLLM response: {response_payload}"
        )

    choice = choices[0]
    content = choice.get("text")
    if not isinstance(content, str):
        message = choice.get("message", {})
        content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(
            f"Missing completion content in vLLM response: {response_payload}"
        )

    index = content.find("</think>")
    if index >= 0:
        content = content[index + 8:]

    return content.strip()


def generate_answer(
        question: dict[str, Any],
        results: list[SearchResult],
        *,
        config: GenerationConfig,
) -> GeneratedAnswer:
    question_id = str(question.get("question_id", ""))
    question_str = str(question.get("question", ""))
    selected = _select_context(results, config)
    context = "\n\n".join(
        _format_context_block(result, rank)
        for rank, result in enumerate(selected, start=1)
    )
    prompt = _build_qwen_prompt(question_str, context)
    answer = _call_openai_compatible_completion(
        model=config.model,
        prompt=prompt,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout_seconds,
    )
    return GeneratedAnswer(
        question_id=question_id,
        question_str=question_str,
        answer=answer,
        retrieved_sources=[
            result.to_source_dict() for result in results[:config.search_k]
        ],
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        search_k=config.search_k,
        top_context_chunks=config.top_context_chunks,
    )


def generate_answer_from_sources(
        question_id: str,
        question_str: str,
        retrieved_sources: list[dict[str, Any]],
        *,
        corpus_lookup: dict[tuple[str, int, int], dict[str, Any]],
        config: GenerationConfig,
) -> GeneratedAnswer:
    selected_sources = retrieved_sources[:config.top_context_chunks]
    blocks: list[str] = []
    for rank, source in enumerate(selected_sources, start=1):
        entry = corpus_lookup.get(_source_key(source))
        if entry is None:
            continue
        blocks.append(_format_source_block(source, entry, rank))

    prompt = _build_qwen_prompt(question_str, "\n\n".join(blocks))
    answer = _call_openai_compatible_completion(
        model=config.model,
        prompt=prompt,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout_seconds,
    )
    return GeneratedAnswer(
        question_id=question_id,
        question_str=question_str,
        answer=answer,
        retrieved_sources=list(retrieved_sources[:config.search_k]),
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        search_k=config.search_k,
        top_context_chunks=config.top_context_chunks,
    )


def _resolve_model(model: str | None) -> str:
    if model:
        return model

    env_model = os.environ.get("VLLM_MODEL")
    if env_model:
        return env_model

    raise ValueError(
        "No model provided. Pass model=... or set the VLLM_MODEL "
        "environment variable."
    )


def answer_question(
        question: str,
        *,
        model: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        search_k: int = DEFAULT_SEARCH_K,
        top_context_chunks: int = DEFAULT_TOP_CONTEXT_CHUNKS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retriever: Any | None = None,
        corpus: list[dict[str, Any]] | None = None,
) -> GeneratedAnswer:
    if retriever is None or corpus is None:
        retriever, corpus = load_or_build_index()

    config = GenerationConfig(
        model=_resolve_model(model),
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        search_k=search_k,
        top_context_chunks=top_context_chunks,
        max_context_chars=max_context_chars,
        timeout_seconds=timeout_seconds,
    )
    results = search(question, retriever, corpus, k=config.search_k)
    generated = generate_answer(
        {"question_id": "", "question": question},
        results,
        config=config,
    )
    return generated


def _load_search_results(
        student_search_results_path: str | Path) -> dict[str, Any]:
    payload = json.loads(
        Path(student_search_results_path).read_text(encoding="utf-8")
    )
    search_results = payload.get("search_results", [])
    if not isinstance(search_results, list):
        raise ValueError(
            "Invalid student results format: search_results must be a list"
        )
    return cast(dict[str, Any], payload)


def _load_existing_answers(
        output_path: Path) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    answers = payload.get("search_results", [])
    if not isinstance(answers, list):
        return {}

    existing: dict[str, dict[str, Any]] = {}
    for item in answers:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id", ""))
        if question_id:
            existing[question_id] = item
    return existing


def answer_dataset(
        student_search_results_path: str | Path,
        *,
        model: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        top_context_chunks: int = DEFAULT_TOP_CONTEXT_CHUNKS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        concurrency: int = DEFAULT_CONCURRENCY,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        output_path: str | Path | None = None,
        retriever: Any | None = None,
        corpus: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = _load_search_results(student_search_results_path)
    search_results = list(payload.get("search_results", []))
    if corpus is None:
        try:
            corpus = load_chunks()
        except FileNotFoundError:
            _, corpus = load_or_build_index()

    corpus_lookup = _build_corpus_lookup(corpus)
    output_path_obj = Path(output_path) if output_path is not None else None
    existing_answers = (
        _load_existing_answers(output_path_obj)
        if output_path_obj is not None
        else {}
    )

    config = GenerationConfig(
        model=_resolve_model(model),
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_context_chunks=top_context_chunks,
        max_context_chars=max_context_chars,
        timeout_seconds=timeout_seconds,
    )

    answers = _answer_student_items_with_resume(
        search_results,
        corpus_lookup,
        config,
        existing_answers=existing_answers,
        concurrency=concurrency,
        checkpoint_interval=max(1, checkpoint_interval),
        output_path=output_path_obj,
        base_payload=payload,
    )

    return {
        "search_results": answers,
        "k": int(payload.get("k", config.search_k)),
    }


def _answer_student_item(
        item: dict[str, Any],
        corpus_lookup: dict[tuple[str, int, int], dict[str, Any]],
        config: GenerationConfig,
) -> dict[str, Any]:
    question_id = str(item.get("question_id", ""))
    question_str = str(item.get("question_str", item.get("question", "")))
    retrieved_sources = list(item.get("retrieved_sources") or [])
    generated = generate_answer_from_sources(
        question_id,
        question_str,
        retrieved_sources,
        corpus_lookup=corpus_lookup,
        config=config,
    )
    return {
        "question_id": generated.question_id,
        "question_str": generated.question_str,
        "retrieved_sources": generated.retrieved_sources,
        "answer": generated.answer,
    }


def _answer_student_items_concurrently(
        items: list[dict[str, Any]],
        corpus_lookup: dict[tuple[str, int, int], dict[str, Any]],
        config: GenerationConfig,
        *,
        concurrency: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(items)
    worker_count = max(1, concurrency)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                _answer_student_item,
                item,
                corpus_lookup,
                config,
            ): index
            for index, item in enumerate(items)
        }
        for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Generating answers"):
            index = future_to_index[future]
            results[index] = future.result()

    return [result for result in results if result is not None]


def _answer_student_items_with_resume(
        items: list[dict[str, Any]],
        corpus_lookup: dict[tuple[str, int, int], dict[str, Any]],
        config: GenerationConfig,
        *,
        existing_answers: dict[str, dict[str, Any]],
        concurrency: int,
        checkpoint_interval: int,
        output_path: Path | None,
        base_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    answers: list[dict[str, Any] | None] = [None] * len(items)
    completed = 0

    for index, item in enumerate(items):
        question_id = str(item.get("question_id", ""))
        if question_id and question_id in existing_answers:
            answers[index] = existing_answers[question_id]

    pending_indexes = [
        index for index, item in enumerate(items)
        if answers[index] is None
    ]

    if not pending_indexes:
        return [answer for answer in answers if answer is not None]

    if concurrency <= 1:
        for index in tqdm(pending_indexes, desc="Generating answers"):
            answers[index] = _answer_student_item(
                items[index],
                corpus_lookup,
                config,
            )
            completed += 1
            if (
                output_path is not None
                and completed % checkpoint_interval == 0
            ):
                save_answers(
                    {
                        **base_payload,
                        "search_results": [
                            answer for answer in answers if answer is not None
                        ],
                    },
                    output_path,
                )
    else:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            future_to_index = {
                executor.submit(
                    _answer_student_item,
                    items[index],
                    corpus_lookup,
                    config,
                ): index
                for index in pending_indexes
            }
            for future in tqdm(
                    as_completed(future_to_index),
                    total=len(future_to_index),
                    desc="Generating answers"):
                index = future_to_index[future]
                answers[index] = future.result()
                completed += 1
                if (
                    output_path is not None
                    and completed % checkpoint_interval == 0
                ):
                    save_answers(
                        {
                            **base_payload,
                            "search_results": [
                                answer
                                for answer in answers
                                if answer is not None
                            ],
                        },
                        output_path,
                    )

    if output_path is not None:
        save_answers(
            {
                **base_payload,
                "search_results": [
                    answer for answer in answers if answer is not None
                ],
            },
            output_path,
        )

    return [answer for answer in answers if answer is not None]


def save_answers(
        payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def answer_dataset_to_file(
        student_search_results_path: str | Path,
        *,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        model: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        top_context_chunks: int = DEFAULT_TOP_CONTEXT_CHUNKS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        concurrency: int = DEFAULT_CONCURRENCY,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        corpus: list[dict[str, Any]] | None = None,
) -> Path:
    output_path = Path(output_dir) / Path(student_search_results_path).name
    answer_dataset(
        student_search_results_path,
        model=model,
        base_url=base_url,
        top_context_chunks=top_context_chunks,
        max_context_chars=max_context_chars,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        concurrency=concurrency,
        checkpoint_interval=checkpoint_interval,
        output_path=output_path,
        corpus=corpus,
    )
    return output_path
