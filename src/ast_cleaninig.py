import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm


INCLUDE_RE = re.compile(r'^\s*--8<--\s+"(?P<target>[^"]+)"\s*$')
MARKER_RE = re.compile(
    r'^\s*(?:<!--\s*|#\s*)?--8<--\s*'
    r'\[(?P<kind>start|end):(?P<name>[^\]]+)\]\s*(?:-->)?\s*$'
)
HEADING_RE = re.compile(r'^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$')
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class IndexChunk:
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


class ContextualParser(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[tuple[str, str]] = []
        self.results: list[dict[str, Any]] = []

    def _snapshot(self, node: Any, calls: list[str] | None = None) -> None:
        self.results.append({
            'name': node.name,
            'line': node.lineno,
            'end_line': node.end_lineno or node.lineno,
            'context': list(self.stack),
            'calls': calls or [],
        })

    # fix 3 — collect every function/method name called inside a node's body
    def _extract_calls(self, node: Any) -> list[str]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(dict.fromkeys(calls))  # deduplicate, preserve order

    def visit_ClassDef(self, node: Any) -> None:
        self.stack.append(('class', node.name))
        self._snapshot(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: Any) -> None:
        self.stack.append(('function', node.name))
        calls = self._extract_calls(node)
        self._snapshot(node, calls=calls)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: Any) -> None:
        self.visit_FunctionDef(node)


def classify(context: list[tuple[str, str]]) -> str:
    kind = context[-1][0]
    if kind == 'class':
        return 'class'
    if len(context) >= 2 and context[-2][0] == 'class':
        return 'method'
    if len(context) >= 2 and context[-2][0] == 'function':
        return 'nested_function'
    return 'function'


def _build_heading_path(
        inherited: list[str], local_headings: list[str]) -> list[str]:
    return inherited + local_headings


def _display_path(path: Path) -> str:
    if path.is_absolute():
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            try:
                return str(path.relative_to(Path.cwd()))
            except ValueError:
                return str(path)
    return str(path)


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _char_span_from_lines(
        line_starts: list[int], lines: list[str],
        start_line: int, end_line: int) -> tuple[int, int]:
    start_offset = line_starts[start_line - 1]
    end_line_text = lines[end_line - 1].rstrip("\r\n")
    end_offset = line_starts[end_line - 1] + len(end_line_text) - 1
    return start_offset, end_offset


def format_class_chunk(
        r: dict[str, Any],
        results: list[dict[str, Any]],
        file_name: str,
        line_starts: list[int],
        lines: list[str]) -> IndexChunk:
    symbol = ".".join(name for _, name in r['context'])
    methods = [
        res['name'] for res in results
        if len(res['context']) >= 2
        and res['context'][-2][1] == r['name']
        and res['context'][-2][0] == 'class'
    ]
    summary = "\n".join(
        f"- {m}" for m in methods
    ) if methods else "  (no methods)"
    start_offset, end_offset = _char_span_from_lines(
        line_starts, lines, r['line'], r['end_line'])
    return IndexChunk(
        text=(
            f"FILE: {file_name}\n"
            f"TYPE: class\n"
            f"SYMBOL: {symbol}\n"
            f"\nCLASS SUMMARY:\n{summary}\n"
        ),
        file_path=file_name,
        first_character_index=start_offset,
        last_character_index=end_offset,
        kind='python_class',
        symbol=symbol,
    )


def format_nested_chunk(
        r: dict[str, Any], lines: list[str], file_name: str,
        line_starts: list[int]) -> IndexChunk:
    symbol = ".".join(name for _, name in r['context'])
    parent_name = r['context'][-2][1]
    code = "\n".join(lines[r['line'] - 1:r['end_line']])
    start_offset, end_offset = _char_span_from_lines(
        line_starts, lines, r['line'], r['end_line'])
    return IndexChunk(
        text=(
            f"FILE: {file_name}\n"
            f"TYPE: nested_function\n"
            f"SYMBOL: {symbol}\n"
            f"PARENT: {parent_name}\n"
            f"\n{code}\n"
        ),
        file_path=file_name,
        first_character_index=start_offset,
        last_character_index=end_offset,
        kind='python_nested_function',
        symbol=symbol,
        calls=list(r.get('calls', [])),
    )


def format_callable_chunk(
        r: dict[str, Any],
        lines: list[str],
        file_name: str,
        kind: str,
        line_starts: list[int]) -> IndexChunk:
    symbol = ".".join(name for _, name in r['context'])
    parent = r['context'][-2][1] if len(r['context']) > 1 else None
    calls = r.get('calls', [])
    code = "\n".join(lines[r['line'] - 1:r['end_line']])
    start_offset, end_offset = _char_span_from_lines(
        line_starts, lines, r['line'], r['end_line'])

    calls_section = ""
    if calls:
        calls_section = "CALLS:\n" + "\n".join(
            f"- {c}()" for c in calls) + "\n"

    return IndexChunk(
        text=(
            f"FILE: {file_name}\n"
            f"TYPE: {kind}\n"
            f"SYMBOL: {symbol}\n"
            + (f"PARENT: {parent}\n" if parent else "")
            + (f"\n{calls_section}" if calls_section else "")
            + f"\n{code}\n"
        ),
        file_path=file_name,
        first_character_index=start_offset,
        last_character_index=end_offset,
        kind=f'python_{kind}',
        symbol=symbol,
        calls=list(calls),
    )


def get_ready_to_index_data(
        folder_path: str = "data/raw/vllm-0.10.1") -> list[dict[str, Any]]:
    clean_data_lst: list[IndexChunk] = []
    folder_root = _resolve_repo_path(folder_path)

    def get_ready_to_index_py_file(
            file_name: str | Path = "src/test_file.py") -> None:

        file_name_str = _display_path(Path(file_name))

        with open(file=file_name, mode="r", encoding="utf-8") as f:
            source = f.read()
        lines = source.splitlines()
        line_starts = [0]
        for match in re.finditer(r"\n", source):
            line_starts.append(match.end())

        tree = ast.parse(source)
        parser = ContextualParser()
        parser.visit(tree)

        for r in parser.results:
            kind = classify(r['context'])

            if kind == 'class':
                clean_data_lst.append(
                    format_class_chunk(
                        r, parser.results, file_name_str, line_starts, lines))
            elif kind == 'nested_function':
                clean_data_lst.append(format_nested_chunk(
                    r, lines, file_name_str, line_starts))
            else:
                clean_data_lst.append(
                    format_callable_chunk(
                        r, lines, file_name_str, kind, line_starts))

    def _line_starts(source: str) -> list[int]:
        starts = [0]
        for match in re.finditer(r"\n", source):
            starts.append(match.end())
        return starts

    def _anchor_bounds(
            source: str, anchor: str | None) -> tuple[int, int]:
        if anchor is None:
            return 1, len(source.splitlines(keepends=True))

        lines = source.splitlines(keepends=True)
        start_line = 1
        end_line = len(lines)
        found_start = False
        for idx, line in enumerate(lines, start=1):
            match = MARKER_RE.match(line)
            if match is None:
                continue
            marker_kind = match.group("kind")
            marker_name = match.group("name")
            if marker_kind == "start" and marker_name == anchor:
                start_line = idx + 1
                found_start = True
            elif (
                marker_kind == "end"
                and marker_name == anchor
                and found_start
            ):
                end_line = idx - 1
                break
        return start_line, end_line

    def _resolve_markdown_target(
            current_file: Path,
            raw_target: str,
            repo_root: Path) -> tuple[Path, str | None]:
        target_path, _, anchor = raw_target.partition(":")
        resolved = Path(target_path)
        if not resolved.is_absolute():
            root_candidate = (repo_root / target_path).resolve()
            if root_candidate.exists():
                resolved = root_candidate
            else:
                resolved = (current_file.parent / target_path).resolve()
        return resolved, anchor or None

    _MD_WINDOW = 1000
    _MD_STRIDE = 500

    def _heading_at_offset(
            lines: list[str],
            line_starts: list[int],
            char_offset: int) -> list[str]:
        """Active heading breadcrumb (all levels) at a given char offset."""
        active: list[str] = []
        for line_no, line in enumerate(lines, start=1):
            if line_starts[line_no - 1] >= char_offset:
                break
            m = HEADING_RE.match(line)
            if m:
                level = len(m.group("hashes"))
                title = m.group("title").strip()
                active = active[:level - 1] + [title]
        return active

    def _sliding_window_md_chunks(
            file_path: Path,
            source: str,
            line_starts: list[int],
            lines: list[str],
            start_line: int,
            end_line: int,
            inherited_headings: list[str],
    ) -> list[IndexChunk]:
        """Overlapping fixed-size chunks over the source char range.

        Span indices point into the *source file* directly — the
        FILE:/HEADING: prefix added to the BM25 text is NOT counted in
        first/last_character_index, so IoU against gold spans is accurate.
        """
        if not lines:
            return []

        file_str = _display_path(file_path)
        range_start = line_starts[start_line - 1]
        last_line_text = lines[end_line - 1].rstrip("\r\n")
        range_end = line_starts[end_line - 1] + len(last_line_text)

        source_slice = source[range_start:range_end]
        if not source_slice.strip():
            return []

        chunks: list[IndexChunk] = []
        window = _MD_WINDOW
        stride = _MD_STRIDE
        pos = 0
        total = len(source_slice)

        while pos < total:
            win_end = min(pos + window, total)

            # Snap to end of current line to avoid cutting mid-sentence
            newline_pos = source_slice.find("\n", win_end)
            if newline_pos != -1 and newline_pos < win_end + 120:
                win_end = newline_pos + 1

            chunk_text = source_slice[pos:win_end]
            if not chunk_text.strip():
                pos += stride
                continue

            abs_start = range_start + pos
            abs_end = range_start + win_end - 1

            heading_path = _build_heading_path(
                inherited_headings,
                _heading_at_offset(lines, line_starts, abs_start),
            )

            _sep = " > "
            if heading_path:
                bm25_text = (
                    "FILE: " + file_str + "\n"
                    "HEADING: " + _sep.join(heading_path) + "\n\n"
                    + chunk_text.strip()
                )
            else:
                bm25_text = "FILE: " + file_str + "\n\n" + chunk_text.strip()

            chunks.append(IndexChunk(
                text=bm25_text,
                file_path=file_str,
                first_character_index=abs_start,
                last_character_index=abs_end,
                kind='markdown',
                heading_path=heading_path,
            ))

            if win_end >= total:
                break
            pos += stride

        return chunks

    def _get_ready_to_index_md_file(
            file_name: str | Path,
            repo_root: Path,
            inherited_headings: list[str] | None = None,
            anchor: str | None = None,
            active_stack: set[tuple[Path, str | None]] | None = None,
    ) -> list[IndexChunk]:
        file_path = Path(file_name).resolve()
        if inherited_headings is None:
            inherited_headings = []
        if active_stack is None:
            active_stack = set()

        visit_key = (file_path, anchor)
        if visit_key in active_stack:
            return []
        active_stack.add(visit_key)
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError:
            active_stack.remove(visit_key)
            return []

        try:
            if not source:
                return []

            line_starts = _line_starts(source)
            start_line, end_line = _anchor_bounds(source, anchor)
            lines = source.splitlines(keepends=True)
            if not lines:
                return []

            return _sliding_window_md_chunks(
                file_path,
                source,
                line_starts,
                lines,
                start_line,
                end_line,
                inherited_headings,
            )
        finally:
            active_stack.remove(visit_key)

    files = list(folder_root.rglob("*.py"))
    for file in tqdm(files, "Parsing files"):
        get_ready_to_index_py_file(file)

    docs_root = folder_root
    md_files = [
        file for file in docs_root.rglob("*.md")
        if not file.name.endswith(".inc.md")
    ]
    for file in tqdm(md_files, "Parsing markdown files"):
        clean_data_lst.extend(
            _get_ready_to_index_md_file(file, docs_root))

    return [chunk.to_record() for chunk in clean_data_lst]


if __name__ == "__main__":
    get_ready_to_index_data()
