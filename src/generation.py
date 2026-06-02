import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm

from indexing import load_or_build_index
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

SYSTEM_PROMPT = (
    "You answer questions about the vLLM repository using only the provided "
    "context. If the context is insufficient, say that you cannot find a "
    "supported answer in the retrieved sources. Be concise and factual. "
    "Do not invent details."
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


def _build_messages_from_sources(
        question: str,
        retrieved_sources: list[dict[str, Any]],
        corpus_lookup: dict[tuple[str, int, int], dict[str, Any]],
        config: GenerationConfig,
) -> list[dict[str, str]]:
    selected_sources = retrieved_sources[:config.top_context_chunks]
    blocks: list[str] = []
    for rank, source in enumerate(selected_sources, start=1):
        entry = corpus_lookup.get(_source_key(source))
        if entry is None:
            continue
        blocks.append(_format_source_block(source, entry, rank))

    context = "\n\n".join(blocks)
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer the question using only the retrieved context."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_messages(
        question: str, results: list[SearchResult], config: GenerationConfig
) -> list[dict[str, str]]:
    selected = _select_context(results, config)
    context = "\n\n".join(
        _format_context_block(result, rank)
        for rank, result in enumerate(selected, start=1)
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer the question using only the retrieved context."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _call_openai_compatible_chat(
        *,
        model: str,
        messages: list[dict[str, str]],
        base_url: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
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
    message = choice.get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(
            f"Missing message content in vLLM response: {response_payload}"
        )

    return content.strip()


def generate_answer(
        question: dict[str, Any],
        results: list[SearchResult],
        *,
        config: GenerationConfig,
) -> GeneratedAnswer:
    question_id = str(question.get("question_id", ""))
    question_str = str(question.get("question", ""))
    messages = _build_messages(question_str, results, config)
    answer = _call_openai_compatible_chat(
        model=config.model,
        messages=messages,
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
    messages = _build_messages_from_sources(
        question_str,
        retrieved_sources,
        corpus_lookup,
        config,
    )
    answer = _call_openai_compatible_chat(
        model=config.model,
        messages=messages,
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
        concurrency: int = 4,
        retriever: Any | None = None,
        corpus: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if retriever is None or corpus is None:
        retriever, corpus = load_or_build_index()

    payload = _load_search_results(student_search_results_path)
    search_results = list(payload.get("search_results", []))
    corpus_lookup = _build_corpus_lookup(corpus)

    config = GenerationConfig(
        model=_resolve_model(model),
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_context_chunks=top_context_chunks,
        max_context_chars=max_context_chars,
        timeout_seconds=timeout_seconds,
    )

    if concurrency <= 1:
        answers = []
        for item in tqdm(search_results, desc="Generating answers"):
            answers.append(
                _answer_student_item(item, corpus_lookup, config)
            )
    else:
        answers = _answer_student_items_concurrently(
            search_results,
            corpus_lookup,
            config,
            concurrency=concurrency,
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
        concurrency: int = 4,
        retriever: Any | None = None,
        corpus: list[dict[str, Any]] | None = None,
) -> Path:
    payload = answer_dataset(
        student_search_results_path,
        model=model,
        base_url=base_url,
        top_context_chunks=top_context_chunks,
        max_context_chars=max_context_chars,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        concurrency=concurrency,
        retriever=retriever,
        corpus=corpus,
    )
    output_path = Path(output_dir) / Path(student_search_results_path).name
    return save_answers(payload, output_path)
