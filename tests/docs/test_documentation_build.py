"""Public documentation release boundary."""

import ast
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

_OPTIONAL_TUTORIAL = "optional_workflows/pyaesa_allocated_carrying_capacity.ipynb"
_WORKFLOW_CALLS = (
    "set_workspace",
    "download_data",
    "prepare_backgrounds",
    "run_mfa",
    "run_lca",
    "run_asr",
    "run_optimization",
)


def _tutorials(root: Path) -> tuple[Path, ...]:
    """Discover the authoritative repository notebooks for validation."""

    return tuple(sorted((root / "tutorials").rglob("*.ipynb")))


def _code_tree(path: Path) -> ast.Module:
    """Parse all code cells in one notebook as a Python module."""

    notebook = nbformat.read(path, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    return ast.parse(code, filename=str(path))


def test_tutorial_structure_and_notebook_state() -> None:
    """Validate distributed notebooks and their cleared execution state."""

    root = Path(__file__).resolve().parents[2]
    tutorials = _tutorials(root)

    assert tutorials
    for path in tutorials:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        _code_tree(path)
        assert all(
            cell.execution_count is None and not cell.outputs
            for cell in notebook.cells
            if cell.cell_type == "code"
        ), path


def test_core_tutorials_call_the_public_workflows_directly() -> None:
    """Expose every public workflow directly after process workspace selection."""

    root = Path(__file__).resolve().parents[2]
    documented = set()
    for path in sorted((root / "tutorials/core_workflow").glob("*.ipynb")):
        calls = [
            statement.value.func.id
            for statement in _code_tree(path).body
            if isinstance(statement, (ast.Assign, ast.Expr))
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id in _WORKFLOW_CALLS
        ]
        assert calls and calls[0] == "set_workspace", path
        assert set(calls) - {"set_workspace"}, path
        documented.update(calls)
    assert documented == set(_WORKFLOW_CALLS)


def test_tutorials_contain_no_mojibake_or_control_characters() -> None:
    """Reject encoding damage that valid notebook JSON and Sphinx can miss."""

    root = Path(__file__).resolve().parents[2]
    for path in _tutorials(root):
        notebook = nbformat.read(path, as_version=4)
        text = "\n".join(cell.source for cell in notebook.cells)
        assert "\N{LATIN CAPITAL LETTER A WITH TILDE}" not in text, path
        assert "\N{REPLACEMENT CHARACTER}" not in text, path
        assert not any(0x80 <= ord(character) <= 0x9F for character in text), path


def test_pyaesa_h12_mapping_preserves_each_source_region() -> None:
    """Preserve regional totals when distributing annual values across the 12 world regions."""

    root = Path(__file__).resolve().parents[2]
    resource_directory = root / "tutorials" / "optional_workflows" / "data" / "pyaesa_plastics"
    mapping = pd.read_csv(resource_directory / "agg_reg_h12_plast.csv").set_index(
        ["original_classification", "aggregated_mrio"]
    )
    assert mapping.columns.tolist() == [f"weight::{year}" for year in range(1995, 2025)]
    assert np.isfinite(mapping.to_numpy()).all()
    assert (mapping.to_numpy() >= 0).all()
    np.testing.assert_allclose(
        mapping.groupby(level="original_classification").sum().to_numpy(),
        1.0,
        rtol=0,
        atol=1e-12,
    )


def test_pyaesa_tutorial_uses_public_calls_and_fixed_capacity_statistics() -> None:
    """Keep public pyaesa calls and unambiguous allocated carrying capacity statistic labels."""

    root = Path(__file__).resolve().parents[2]
    path = root / "tutorials" / _OPTIONAL_TUTORIAL
    tree = _code_tree(path)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {"set_pyaesa_workspace", "uncertainty_acc", "process_mrio"} <= calls
    assert not any(
        (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.partition(".")[0] == "pytoce"
        )
        or (
            isinstance(node, ast.Import)
            and any(alias.name.partition(".")[0] == "pytoce" for alias in node.names)
        )
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mkdir"
        and any(
            keyword.arg == "exist_ok"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
    )
    statistic_mapping = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "statistic_columns"
            for target in node.targets
        )
    )
    assert statistic_mapping == {
        "mean": "mean",
        "min": "min",
        "max": "max",
        "p5": "p5",
        "p25": "p25",
        "p50": "median",
        "p75": "p75",
        "p95": "p95",
    }
    default_statistics = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SELECTED_STATISTICS"
            for target in node.targets
        )
    )
    assert default_statistics == ("p5", "p50", "p95")


@pytest.mark.parametrize(
    "filename",
    ("pytoce_article.pdf", "pytoce_methodological_appendix.pdf"),
)
def test_article_and_methodological_appendix_pdfs_are_readable(
    tmp_path: Path,
    filename: str,
) -> None:
    """Check that the bundled article and Methodological appendix PDFs can be read and rendered."""

    root = Path(__file__).resolve().parents[2]
    source = root / "src/pytoce/set_workspace/data/methodological_notes" / filename
    information = subprocess.run(
        ["pdfinfo", str(source)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    information_fields = {
        key: value.strip()
        for line in information.splitlines()
        if ":" in line
        for key, value in (line.split(":", 1),)
    }
    assert int(information_fields["Pages"]) >= 1
    assert information_fields["Encrypted"].split()[0] == "no"
    assert information_fields["JavaScript"] == "no"

    page_text = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(source), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    assert page_text.strip()

    rendered = tmp_path / "page"
    subprocess.run(
        ["pdftoppm", "-f", "1", "-singlefile", "-r", "72", "-png", str(source), str(rendered)],
        check=True,
    )
    assert rendered.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_sphinx_documentation_builds_without_warnings(tmp_path: Path) -> None:
    """Build every public page and linked notebook with warnings treated as errors."""

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "html"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            str(root / "docs"),
            str(output),
        ],
        cwd=root,
        check=True,
    )
    assert (output / "index.html").is_file()
    assert (output / "tutorials" / "core_workflow" / "5m_run_asr.html").is_file()
    assert (output / "tutorials" / "core_workflow" / "5o_run_optimization.html").is_file()
    assert (
        output / "tutorials" / "optional_workflows" / "pyaesa_allocated_carrying_capacity.html"
    ).is_file()
