"""Public optimization preflight and response-parent boundary coverage."""

from dataclasses import replace
from pathlib import Path

import pytest
from result_store_fixtures import _save_result
from selected_parent_fixtures import _BACKGROUND, _write_baseline_result

from pytoce.prepare_backgrounds.preparation.premise import ACTIVE_BACKGROUND_SCENARIOS
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    PreparedDataError,
)
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_lca.run_lca import lca_result_directory
from pytoce.run_optimization.calculation.response_builder import (
    ensure_optimization_response_bank,
    response_method_selection,
)
from pytoce.run_optimization.run_optimization import (
    RunOptimizationError,
    _run_foreground,
    normalize_run_optimization_request,
)
from pytoce.run_optimization.runtime.result_store import OptimizationResultTarget


def _request():
    """Return the canonical smallest public optimization request."""

    return normalize_run_optimization_request(
        foreground_scenario="study_parent",
        allocated_cc_result="study_result",
        optimization_lcia_method="ef31",
        lcia_method="ef31",
        optimization_block="demand_lifetime_frontier",
        allocated_cc_case="central",
    )


def test_foreground_preflight_preserves_visible_incomplete_result(tmp_path: Path) -> None:
    """Reject an incomplete result and preserve its existing files."""

    request = _request()
    target = OptimizationResultTarget(
        workspace=tmp_path,
        foreground_scenario="study_parent",
        allocated_cc_result=request.allocated_cc_result,
        optimization_lcia_selection="ef31",
        readable_request_selection=request.readable_request_selection,
        initial_grid_step_pct=20,
        endpoint_refinement_steps_pct=(10, 5),
    )
    selected = target.selected_results(request.result_lcia_selection)
    selected.mkdir(parents=True)
    marker = selected / "user-file.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(RunOptimizationError, match="exists without a completion manifest"):
        _run_foreground(tmp_path, request, "study_parent")

    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_foreground_preflight_reports_missing_baseline_lca_results(tmp_path: Path) -> None:
    """Stop before MFA, allocated carrying capacity loading or search if baseline LCA is missing."""

    request = _request()

    with pytest.raises(
        RunOptimizationError,
        match="Cannot use the required LCA results.*foreground_scenario='study_parent'",
    ):
        _run_foreground(tmp_path, request, "study_parent")


def test_old_search_method_product_cannot_satisfy_corrected_request(tmp_path: Path) -> None:
    """Keep old completed products while the new method admits its own required parents."""
    request = _request()
    target = OptimizationResultTarget(
        workspace=tmp_path,
        foreground_scenario="study_parent",
        allocated_cc_result=request.allocated_cc_result,
        optimization_lcia_selection="ef31",
        readable_request_selection=request.readable_request_selection,
        initial_grid_step_pct=request.initial_grid_step_pct,
        endpoint_refinement_steps_pct=request.endpoint_refinement_steps_pct,
    )
    corrected = target.selected_results(request.result_lcia_selection)
    old_root = target.root.with_name(request.readable_request_selection)
    old_product = old_root / corrected.relative_to(target.root)
    _save_result(old_product)
    before = {path.name: path.read_bytes() for path in old_product.iterdir()}

    with pytest.raises(RunOptimizationError, match="Cannot use the required LCA results"):
        _run_foreground(tmp_path, request, "study_parent")

    assert corrected != old_product
    assert not corrected.exists()
    assert {path.name: path.read_bytes() for path in old_product.iterdir()} == before


def test_response_bank_build_requires_prepared_shared_parent(tmp_path: Path) -> None:
    """Fail before any response solve when the shared prepared background is unavailable."""

    selection = response_method_selection(normalized_lcia_method(("ef31",)))
    observed_progress: list[str] = []

    with pytest.raises(PreparedDataError, match="Run prepare_backgrounds first"):
        ensure_optimization_response_bank(
            tmp_path,
            selection=selection,
            member_progress=observed_progress.append,
        )

    assert observed_progress == []


def test_foreground_preflight_rejects_conflicting_completed_lca_parents(tmp_path: Path) -> None:
    """Reject different MFA parents before loading MFA or launching candidate workers."""
    request = _request()
    manifests = {}
    for background in ACTIVE_BACKGROUND_SCENARIOS:
        directory = lca_result_directory(
            tmp_path,
            "study_parent",
            background.background_scenario,
            request.lcia_method,
            (2019, 2050, 2100),
        )
        _write_baseline_result(
            directory,
            foreground_scenario="study_parent",
            background_scenario=background.background_scenario,
        )
        manifests[directory] = (directory / "manifest.json").read_bytes()
    conflicting = lca_result_directory(
        tmp_path, "study_parent", _BACKGROUND, request.lcia_method, (2050, 2100)
    )
    _write_baseline_result(
        conflicting,
        foreground_scenario="study_parent",
        years=(2050, 2100),
        mfa_identity="mfa::refreshed_parent",
    )
    manifests[conflicting] = (conflicting / "manifest.json").read_bytes()

    with pytest.raises(RunOptimizationError, match="Ambiguous baseline LCA results") as caught:
        _run_foreground(tmp_path, request, "study_parent")

    default = lca_result_directory(
        tmp_path, "study_parent", _BACKGROUND, request.lcia_method, (2019, 2050, 2100)
    )
    assert str(default) in str(caught.value)
    assert str(conflicting) in str(caught.value)
    assert "run_lca(refresh=True)" in str(caught.value)
    assert "new workspace" in str(caught.value)
    assert all(
        (path / "manifest.json").read_bytes() == content for path, content in manifests.items()
    )


def test_refresh_preserves_completed_product_when_required_parents_are_missing(
    tmp_path: Path,
) -> None:
    """A refresh request admits its owned destination before missing parents stop execution."""
    request = replace(_request(), refresh=True)
    target = OptimizationResultTarget(
        workspace=tmp_path,
        foreground_scenario="study_parent",
        allocated_cc_result=request.allocated_cc_result,
        optimization_lcia_selection="ef31",
        readable_request_selection=request.readable_request_selection,
        initial_grid_step_pct=request.initial_grid_step_pct,
        endpoint_refinement_steps_pct=request.endpoint_refinement_steps_pct,
    )
    directory = target.selected_results(request.result_lcia_selection)
    _save_result(directory)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(RunOptimizationError, match="Cannot use the required LCA results"):
        _run_foreground(tmp_path, request, "study_parent")
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_completed_lca_parents_do_not_substitute_for_missing_mfa(tmp_path: Path) -> None:
    """Deleting a required MFA after LCA completion remains an actionable boundary."""
    request = _request()
    for background in ACTIVE_BACKGROUND_SCENARIOS:
        directory = lca_result_directory(
            tmp_path,
            "study_parent",
            background.background_scenario,
            request.lcia_method,
            (2019, 2050, 2100),
        )
        _write_baseline_result(
            directory,
            foreground_scenario="study_parent",
            background_scenario=background.background_scenario,
        )
    with pytest.raises(RunOptimizationError, match="No completed MFA result was found"):
        _run_foreground(tmp_path, request, "study_parent")
