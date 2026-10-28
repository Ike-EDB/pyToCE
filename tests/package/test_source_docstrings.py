"""Source docstring checks for installable package functions."""

import ast
from pathlib import Path


def test_every_source_function_has_a_nonempty_docstring() -> None:
    """Check docstring presence for functions, methods and nested definitions."""

    package = Path(__file__).resolve().parents[2] / "src" / "pytoce"
    findings: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not (ast.get_docstring(node, clean=True) or "").strip():
                findings.append(f"{path.relative_to(package)}:{node.lineno}:{node.name}")
    assert not findings, "Source docstring findings:\n" + "\n".join(findings)
