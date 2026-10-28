"""Monitoring ASRs use matched completed LCA results and shared assessment rules."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from asr_fixtures import BACKGROUND, FOREGROUND, asr_workspace, write_lca

from pytoce import run_asr
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_lca.calculation.result_inputs import select_lca_result
from pytoce.run_optimization.input.allocated_cc import select_allocated_cc_result
from pytoce.run_optimization.input.allocated_cc_product import compile_allocated_cc_result
from pytoce.shared.runtime.filesystem import canonical_json_bytes, file_sha256

__all__ = ("asr_workspace",)


def _run(**kwargs):
    """Select the fixture's exact completed LCA product through the public API."""
    return run_asr(
        allocated_cc_result="monitoring_cc",
        foreground_scenario=FOREGROUND,
        background_scenario=BACKGROUND,
        lcia_method="ef31",
        **kwargs,
    )


def test_monitoring_reports_annual_and_independent_cumulative_asr(asr_workspace: Path) -> None:
    """Check independently known ratios, years, global values, and arbitrary case labels."""
    report = _run()
    (item,) = report.results
    annual = pd.read_parquet(item.annual_asr)
    cumulative = pd.read_parquet(item.cumulative_gwp100_asr)
    assert report.status == item.status == "computed"
    assert report.workspace == asr_workspace and report.elapsed_seconds >= 0
    assert len(annual) == 2 * 13 * 82 * 16
    assert len(cumulative) == 2 * 13
    assert set(annual.allocated_cc_case) == {"p5", "p95"}
    p5 = annual.loc[annual.allocated_cc_case.eq("p5")]
    assert p5.loc[p5.year.eq(2019), "asr"].eq(2.0).all()
    assert not p5.loc[p5.year.eq(2019), "assessed_for_compliance"].any()
    assert p5.loc[p5.year.ge(2020), "asr"].eq(0.5).all()
    assert p5.loc[p5.year.ge(2020), "compliant"].all()
    assert cumulative.loc[cumulative.allocated_cc_case.eq("p5"), "asr"].eq(1.0).all()
    assert cumulative.loc[cumulative.allocated_cc_case.eq("p95"), "asr"].eq(0.5).all()
    assert cumulative.compliant.all()
    assert annual.loc[annual.region.eq("global") & annual.year.eq(2020), "score"].eq(12.0).all()
    serialized = report.as_dict()
    assert serialized["workspace"] == str(asr_workspace)
    assert serialized["results"][0]["annual_asr"] == str(item.annual_asr)
    manifest = json.loads(item.manifest_path.read_text())
    assert manifest["complete"] and manifest["allocated_cc_cases"] == ["p5", "p95"]
    assert "lca_manifest_sha256" in manifest["identity_payload"]
    assert "global" in (item.result_directory / "README.md").read_text()


def test_case_selection_reuse_refresh_and_input_preservation(asr_workspace: Path) -> None:
    """Keep completed inputs unchanged and avoid opening them when reusing ASR results."""
    first = _run(allocated_cc_case="p5")
    item = first.results[0]
    original = item.annual_asr.read_bytes()
    assert _run(allocated_cc_case=["p5", "p5"]).status == "reused"
    assert _run(allocated_cc_case="p5", refresh=True).status == "updated"
    assert item.annual_asr.read_bytes() == original
    source = next(
        (asr_workspace / "results/monitoring/lca_results").glob("*/annual_scores.parquet")
    )
    source.unlink()
    assert _run(allocated_cc_case="p5").status == "reused"
    with pytest.raises(FileNotFoundError):
        _run(allocated_cc_case="p5", refresh=True)
    assert item.annual_asr.read_bytes() == original


def test_mixed_item_status_is_updated(asr_workspace: Path) -> None:
    """Reuse one pair, compute another, and preserve both inputs."""
    _run()
    write_lca(asr_workspace, background="remind_ssp2_ndc")
    report = run_asr(
        allocated_cc_result="monitoring_cc",
        foreground_scenario=FOREGROUND,
        background_scenario=[BACKGROUND, "remind_ssp2_ndc"],
        lcia_method="ef31",
    )
    assert report.status == "updated"
    assert {item.status for item in report.results} == {"computed", "reused"}


def test_default_lca_selection_keeps_pb_results_out_of_the_asr(asr_workspace: Path) -> None:
    """The default EF/PB LCA product supplies the same matched EF assessment."""
    expected = pd.read_parquet(_run().results[0].annual_asr)
    write_lca(asr_workspace, groups=("ef31", "pb_lcia"))
    report = run_asr(
        allocated_cc_result="monitoring_cc",
        foreground_scenario=FOREGROUND,
        background_scenario=BACKGROUND,
    )
    pd.testing.assert_frame_equal(pd.read_parquet(report.results[0].annual_asr), expected)


@pytest.mark.parametrize("refresh", [1, "False", None])
def test_refresh_requires_boolean(refresh: object) -> None:
    """Reject invalid replacement controls before importing scientific inputs."""
    with pytest.raises(ValueError, match="refresh must"):
        _run(refresh=refresh)


@pytest.mark.parametrize("method", ["pb_lcia", ("one", "method")])
def test_annual_assessment_requires_all_ef_categories(method) -> None:
    """Reject partial or incompatible annual assessments."""
    with pytest.raises(ValueError, match="complete 'ef31'"):
        run_asr(allocated_cc_result="monitoring_cc", lcia_method=method)


@pytest.mark.parametrize("entry", ["extra.txt", "nested"])
def test_refresh_preserves_unrelated_files(asr_workspace: Path, entry: str) -> None:
    """Reject replacement of a completed scope containing additional user material."""
    item = _run().results[0]
    extra = item.result_directory / entry
    extra.write_text("preserve") if entry.endswith(".txt") else extra.mkdir()
    with pytest.raises(ValueError, match="Existing files have not been changed"):
        _run(refresh=True)
    assert extra.exists()


def test_incomplete_result_is_preserved(asr_workspace: Path) -> None:
    """Keep interrupted results intact and give an actionable replacement error."""
    item = _run().results[0]
    item.manifest_path.unlink()
    with pytest.raises(ValueError, match="completed ASR result"):
        _run()
    assert item.annual_asr.is_file()


def test_result_name_length_is_bounded(asr_workspace: Path) -> None:
    """Leave room for atomic replacement on supported filesystem paths."""
    with pytest.raises(ValueError, match="exceeds 212"):
        run_asr(allocated_cc_result="a" * 213, background_scenario=BACKGROUND)


def test_conflicting_lca_parents_require_an_explicit_choice(asr_workspace: Path) -> None:
    """Different scientific parents cannot be chosen merely by upstream reporting years."""
    other = write_lca(asr_workspace, years=(2050,))
    manifest_path = other / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity_payload"]["mfa_result_identity"] = "another-scientific-parent"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="different scientific parents"):
        _run()
    assert manifest_path.is_file()


def test_failed_publication_removes_staging_and_preserves_previous_result(
    asr_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the completed result after a write failure."""
    item = _run().results[0]
    original = item.manifest_path.read_bytes()
    module = importlib.import_module("pytoce.run_asr.results")

    def fail(*args, **kwargs):
        """Fail at the output publication boundary."""
        raise OSError("invented disk failure")

    monkeypatch.setattr(module, "replace_directory", fail)
    with pytest.raises(OSError, match="disk failure"):
        _run(refresh=True)
    assert item.manifest_path.read_bytes() == original
    assert not list(item.result_directory.parent.glob(".asr.*"))


@pytest.mark.parametrize(
    "mutation", ["hash", "factor", "missing_method", "duplicate", "missing", "unit", "nan"]
)
def test_monitoring_rejects_incompatible_or_damaged_lca_inputs(
    asr_workspace: Path,
    mutation: str,
) -> None:
    """Check the score-reading boundary rather than silently dropping or repairing input rows."""
    reference = select_lca_result(
        asr_workspace,
        foreground_scenario=FOREGROUND,
        background_scenario=BACKGROUND,
        selection=normalized_lcia_method("ef31"),
    )
    manifest = reference.manifest
    path = reference.directory / "annual_scores.parquet"
    frame = pd.read_parquet(path)
    if mutation == "hash":
        path.write_bytes(b"damaged")
    elif mutation == "factor":
        manifest["identity_payload"]["lcia_methods"][0]["factor_identity"] = "other"
    elif mutation == "missing_method":
        manifest["identity_payload"]["lcia_methods"].pop(0)
    else:
        if mutation == "duplicate":
            frame = pd.concat((frame, frame.iloc[:1]))
        elif mutation == "missing":
            frame = frame.iloc[1:]
        elif mutation == "unit":
            frame.loc[0, "unit"] = "incompatible"
        else:
            frame.loc[0, "score"] = np.nan
        frame.to_parquet(path, index=False)
        manifest["artifacts"][path.name]["artifact_sha256"] = file_sha256(path)
    (reference.directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError):
        _run()
    assert not list((asr_workspace / "results/monitoring/asr_results").glob("*/manifest.json"))


def test_output_tables_use_shared_math_for_signed_burdens(asr_workspace: Path) -> None:
    """Preserve the signs of negative environmental burdens and their ratios."""
    from pytoce.run_asr.results import _assessment_tables

    result = compile_allocated_cc_result(asr_workspace, "monitoring_cc")
    capacities = select_allocated_cc_result(result, ("p5",))
    methods = pd.read_parquet(result.result_directory / "lcia_method_registry.parquet")
    scores = -np.ones((12, 82, 16), dtype=np.float64)
    tables = _assessment_tables(scores, capacities, methods)
    assert tables["annual_asr.parquet"].asr.eq(-0.5).all()
    np.testing.assert_allclose(tables["cumulative_gwp100_asr.parquet"].asr, -82.0 / 85.0)
