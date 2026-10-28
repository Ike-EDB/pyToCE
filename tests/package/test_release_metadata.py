"""Checks for current package metadata and serialized schema names."""

import re
from pathlib import Path


def test_distributed_pytoce_schema_names_are_v1() -> None:
    """Require every pyToCE schema identifier and schema filename to use v1."""

    root = Path(__file__).resolve().parents[2]
    package = root / "src/pytoce"
    pattern = re.compile(r"pytoce_[a-z0-9_]*_schema_v(\d+)")
    findings: list[str] = []
    for path in sorted(package.rglob("*")):
        if path.suffix.lower() not in {".json", ".py", ".yaml"}:
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            if match.group(1) != "1":
                findings.append(f"{path.relative_to(root)}: {match.group(0)}")
    schema_files = tuple(package.rglob("data/schemas/*.json"))
    assert schema_files
    assert all(path.name.endswith("_schema_v1.json") for path in schema_files)
    assert not findings, "Non-v1 pyToCE schema names remain:\n" + "\n".join(findings)
