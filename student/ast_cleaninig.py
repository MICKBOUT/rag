import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm


MARKER_RE = re.compile(
    r'^\s*(?:)?\s*$'
)
HEADING_RE = re.compile(r'^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$')
REPO_ROOT = Path(__file__).resolve().parents[1]


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


class ContextualParser(ast.NodeVisitor):
    """
    AST visitor that records a contextual index of class and function
    definitions.

    The parser walks a Python AST and builds a list of dictionaries describing
    each encountered class, function, async function and nested functions
        including:
    - the definition name
    - start and end lines
    - the lexical context stack (class/function nesting)
    - a list of discovered function/method calls within each callable

    Attributes:
        stack (list[tuple[str, str]]): Stack of (kind, name) tuples
            representing current nesting.
        results (list[dict[str, Any]]): Accumulated snapshot dictionaries for
            each visited node.
    """
    def __init__(self) -> None:
        """
        Initialize the parser.

        Initializes the context stack and the results container.
        """
        self.stack: list[tuple[str, str]] = []
        self.results: list[dict[str, Any]] = []

    def _snapshot(self, node: Any, calls: list[str] | None = None) -> None:
        """
        Record a snapshot describing the given AST node.

        Args:
            node (ast.AST): AST node representing a class or function
                definition.
            calls (list[str] | None): Optional list of call names discovered
                in this node.

        Returns:
            None: Appends a dictionary snapshot to self.results with keys
                'name', 'line',
            'end_line', 'context', and 'calls'.
        """
        self.results.append({
            'name': node.name,
            'line': node.lineno,
            'end_line': node.end_lineno or node.lineno,
            'context': list(self.stack),
            'calls': calls or [],
        })

    def _extract_calls(self, node: Any) -> list[str]:
        """
        Extract a deduplicated list of function/method call names from an AST
        node.

        Args:
            node (ast.AST): The AST subtree to inspect.

        Returns:
            list[str]: Ordered list of unique call names found (function names
            for Name nodes and attribute names for Attribute calls).
        """
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(dict.fromkeys(calls))

    def visit_ClassDef(self, node: Any) -> None:
        """
        Handle class definitions: push context, snapshot the class, visit
            children, and pop context.

        Args:
            node (ast.ClassDef): The class definition node.

        Returns:
            None
        """
        self.stack.append(('class', node.name))
        self._snapshot(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: Any) -> None:
        """
        Handle function definitions: push context, extract calls, snapshot,
        visit children, and pop context.

        Args:
            node (ast.FunctionDef): The function definition node.

        Returns:
            None
        """
        self.stack.append(('function', node.name))
        calls = self._extract_calls(node)
        self._snapshot(node, calls=calls)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: Any) -> None:
        """
        Handle async function definitions by delegating to the normal function
        visitor.

        Args:
            node (ast.AsyncFunctionDef): The async function definition node.

        Returns:
            None
        """
        self.visit_FunctionDef(node)


def classify(context: list[tuple[str, str]]) -> str:
    """
    Classify a contextual item as 'class', 'method', 'nested_function', or
    'function'.

    Args:
        context (list[tuple[str, str]]): Context stack produced by
            ContextualParser where each element is a (kind, name) tuple.

    Returns:
        str: One of 'class', 'method', 'nested_function', or 'function'
            describing the kind.
    """
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
    """
    Combine inherited headings with local headings to form a full heading path.

    Args:
        inherited (list[str]): Headings inherited from parent documents or
            scopes.
        local_headings (list[str]): Headings discovered locally in the
            current document slice.

    Returns:
        list[str]: Concatenation of inherited and local heading segments.
    """
    return inherited + local_headings


def _display_path(path: Path) -> str:
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
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            try:
                return str(path.relative_to(Path.cwd()))
            except ValueError:
                return str(path)
    return str(path)


def _resolve_repo_path(path: str | Path) -> Path:
    """
    Resolve a possibly relative path against the repository root.

    Args:
        path (str | Path): A file or folder path, absolute or relative.

    Returns:
        pathlib.Path: Absolute Path resolved relative to the repository root
            when a relative path is supplied.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _char_span_from_lines(
        line_starts: list[int], lines: list[str],
        start_line: int, end_line: int) -> tuple[int, int]:
    """
    Compute absolute character indices for a span given line start offsets.

    Args:
        line_starts (list[int]): List of absolute character indices where each
            line begins.
        lines (list[str]): List of source lines.
        start_line (int): 1-based starting line number of the span.
        end_line (int): 1-based ending line number of the span.

    Returns:
        tuple[int, int]: (first_character_index, last_character_index)
            inclusive character offsets.
    """
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
    """
    Format an IndexChunk summarizing a class definition.

    Args:
        r (dict[str, Any]): Snapshot dict for the class
            (from ContextualParser.results).
        results (list[dict[str, Any]]): All parser results for the file to
            infer methods.
        file_name (str): Displayable file name/path.
        line_starts (list[int]): Line start character offsets for the file.
        lines (list[str]): File lines as a list.

    Returns:
        IndexChunk: A chunk describing the class, containing a small summary
            and offsets.
    """
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
    """
    Format an IndexChunk representing a nested function (function defined
    within another function).

    Args:
        r (dict[str, Any]): Snapshot dict for the nested function.
        lines (list[str]): File lines as a list.
        file_name (str): Displayable file name/path.
        line_starts (list[int]): Line start character offsets for the file.

    Returns:
        IndexChunk: A chunk containing the nested function source, parent
            information, and offsets.
    """
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
    """
    Format an IndexChunk for a top-level or method/function callable.

    Args:
        r (dict[str, Any]): Snapshot dict for the callable.
        lines (list[str]): File lines as a list.
        file_name (str): Displayable file name/path.
        kind (str): Kind label as returned by classify (e.g., 'function',
            'method').
        line_starts (list[int]): Line start character offsets for the file.

    Returns:
        IndexChunk: A chunk containing callable metadata (symbol, parent when
            applicable), call list and source.
    """
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
            Defaults to "data/raw/vllm-0.10.1".

    Returns:
        list[dict[str, Any]]: A list of dictionaries produced by
            IndexChunk.to_record() for all discovered chunks.
    """
    clean_data_lst: list[IndexChunk] = []
    folder_root = _resolve_repo_path(folder_path)

    def get_ready_to_index_py_file(
            file_name: str | Path) -> None:
        """
        Read and parse a Python file, extract classes/functions, and append
        formatted chunks to the enclosing list.

        Args:
            file_name (str | Path): Path to the Python file to parse.

        Returns:
            None: Side-effect appends IndexChunk instances (via format_*
                functions) to the outer clean_data_lst.
        """
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
        """
        Compute a list of character indices where each line in source begins.

        Args:
            source (str): File source text.

        Returns:
            list[int]: 1-to-1 list of start offsets for each line; first entry
                is 0.
        """
        starts = [0]
        for match in re.finditer(r"\n", source):
            starts.append(match.end())
        return starts

    def _anchor_bounds(
            source: str, anchor: str | None) -> tuple[int, int]:
        """
        Determine start and end line bounds for a source slice optionally
        delimited by markers.
        If anchor is None, returns bounds that cover the entire document.
        If anchors are present
        in the document, this function scans marker lines to find matching
        start/end markers.

        Args:
            source (str): Full file source text.
            anchor (str | None): Optional anchor name to limit the slice.

        Returns:
            tuple[int, int]: (start_line, end_line) inclusive 1-based line
                numbers describing the slice.
        """
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

    _MD_WINDOW = 1000
    _MD_STRIDE = 500

    def _heading_at_offset(
            lines: list[str],
            line_starts: list[int],
            char_offset: int) -> list[str]:
        """
        Compute a list of Markdown headings active at a given character offset.
        Scans lines in order and returns the hierarchical heading titles that
        precede the offset.

        Args:
            lines (list[str]): File lines (with or without line endings).
            line_starts (list[int]): Start character offsets for each line.
            char_offset (int): Absolute character offset for which to compute
                the heading stack.

        Returns:
            list[str]: Ordered list of heading titles representing the active
                heading path at offset.
        """
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
        """
        Produce sliding-window IndexChunks for a range of Markdown source.
        Slices the given source range into overlapping windows suitable for
        retrieval/indexing, attaches heading context, and computes absolute
        character offsets for each chunk.

        Args:
            file_path (Path): Path to the markdown file.
            source (str): Full file source text.
            line_starts (list[int]): Start character offsets for each line.
            lines (list[str]): File lines (with line endings preserved).
            start_line (int): 1-based starting line for the range to window.
            end_line (int): 1-based ending line for the range to window.
            inherited_headings (list[str]): Headings inherited from parent
                context.

        Returns:
            list[IndexChunk]: List of markdown IndexChunk objects covering the
                requested range.
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
            inherited_headings: list[str] | None = None,
    ) -> list[IndexChunk]:
        """
        Read a Markdown file and produce ready-to-index chunks for its content.
        Handles simple recursion protections and returns sliding-window chunks
        for the file range.

        Args:
            file_name (str | Path): Path to the markdown file.
            inherited_headings (list[str] | None): Optional heading path
                inherited from callers.

        Returns:
            list[IndexChunk]: Chunks produced for the Markdown file, or an
                empty list on error or if the file is empty.
        """
        file_path = Path(file_name).resolve()
        if inherited_headings is None:
            inherited_headings = []
        active_stack = set()

        visit_key = (file_path, None)
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
            start_line, end_line = _anchor_bounds(source, None)
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
    ]
    for file in tqdm(md_files, "Parsing markdown files"):
        clean_data_lst.extend(
            _get_ready_to_index_md_file(file))

    return [chunk.to_record() for chunk in clean_data_lst]


if __name__ == "__main__":
    get_ready_to_index_data()
