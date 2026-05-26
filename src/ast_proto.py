import ast
from typing import Any


class ContextualParser(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack = []
        self.results = []

    def _snapshot(self, node, calls=None):
        self.results.append({
            'name': node.name,
            'line': node.lineno,
            'end_line': node.end_lineno,
            'context': list(self.stack),
            'calls': calls or [],
        })

    # fix 3 — collect every function/method name called inside a node's body
    def _extract_calls(self, node) -> list[dict[Any, Any | None]]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(dict.fromkeys(calls))  # deduplicate, preserve order

    def visit_ClassDef(self, node):
        self.stack.append(('class', node.name))
        self._snapshot(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self.stack.append(('function', node.name))
        calls = self._extract_calls(node)
        self._snapshot(node, calls=calls)
        self.generic_visit(node)
        self.stack.pop()


def classify(context):
    kind = context[-1][0]
    if kind == 'class':
        return 'class'
    if len(context) >= 2 and context[-2][0] == 'class':
        return 'method'
    if len(context) >= 2 and context[-2][0] == 'function':
        return 'nested_function'
    return 'function'


# fix 1 — class chunk lists method names only, no bodies
def format_class_chunk(r, results, file_name):
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
    return (
        f"FILE: {file_name}\n"
        f"TYPE: class\n"
        f"SYMBOL: {symbol}\n"
        f"\nCLASS SUMMARY:\n{summary}\n"
    )


# fix 2 — nested functions include parent context line
def format_nested_chunk(r, lines, file_name):
    symbol = ".".join(name for _, name in r['context'])
    parent_name = r['context'][-2][1]
    code = "\n".join(lines[r['line'] - 1:r['end_line']])
    return (
        f"FILE: {file_name}\n"
        f"TYPE: nested_function\n"
        f"SYMBOL: {symbol}\n"
        f"PARENT: {parent_name}\n"
        f"\n{code}\n"
    )


# fix 3 — functions/methods include CALLS section
def format_callable_chunk(r, lines, file_name, kind):
    symbol = ".".join(name for _, name in r['context'])
    parent = r['context'][-2][1] if len(r['context']) > 1 else None
    calls = r.get('calls', [])
    code = "\n".join(lines[r['line'] - 1:r['end_line']])

    calls_section = ""
    if calls:
        calls_section = "CALLS:\n" + "\n".join(
            f"- {c}()" for c in calls) + "\n"

    return (
        f"FILE: {file_name}\n"
        f"TYPE: {kind}\n"
        f"SYMBOL: {symbol}\n"
        + (f"PARENT: {parent}\n" if parent else "")
        + (f"\n{calls_section}" if calls_section else "")
        + f"\n{code}\n"
    )


file_name = "test_file.py"
with open(file=file_name, mode="r") as f:
    source = f.read()
lines = source.splitlines()

tree = ast.parse(source)
parser = ContextualParser()
parser.visit(tree)

lst = []
for r in parser.results:
    kind = classify(r['context'])

    if kind == 'class':
        lst.append(format_class_chunk(r, parser.results, file_name))
    elif kind == 'nested_function':
        lst.append(format_nested_chunk(r, lines, file_name))
    else:
        lst.append(format_callable_chunk(r, lines, file_name, kind))

for i in lst:
    print(i)
