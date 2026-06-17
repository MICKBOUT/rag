from pathlib import Path
from typing import Any
import re

import ast

from .parser_common import display_path, IndexChunk


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


def get_ready_to_index_py_file(
        file_name: str | Path) -> list[IndexChunk]:
    """
    Read and parse a Python file, extract classes/functions, and append
    formatted chunks to the enclosing list.

    Args:
        file_name (str | Path): Path to the Python file to parse.

    Returns:
        None: Side-effect appends IndexChunk instances (via format_*
            functions) to the outer clean_data_lst.
    """
    file_name_str = display_path(Path(file_name))

    with open(file=file_name, mode="r", encoding="utf-8") as f:
        source = f.read()
    lines = source.splitlines()
    line_starts = [0]
    for match in re.finditer(r"\n", source):
        line_starts.append(match.end())

    tree = ast.parse(source)
    parser = ContextualParser()
    parser.visit(tree)

    clean_data_lst = []
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

    return clean_data_lst
