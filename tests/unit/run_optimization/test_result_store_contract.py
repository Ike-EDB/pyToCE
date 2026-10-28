"""Persistence-contract tests for selected optimization results."""

from pathlib import Path

import pytest
from result_store_fixtures import _save_result

from pytoce.run_optimization.runtime import result_store


def test_result_owner_and_refresh_reject_nonregular_entries(tmp_path: Path) -> None:
    """Preserve paths outside the exact package-owned regular-file tree."""

    unavailable = tmp_path / "not-a-directory"
    unavailable.write_text("user-owned", encoding="utf-8")
    with pytest.raises(result_store.OptimizationResultError, match="non-linked directory"):
        result_store.validate_result_for_refresh(unavailable)

    directory = tmp_path / "refresh"
    _save_result(directory)
    artifact = directory / "selected_coordinates.parquet"
    artifact.unlink()
    artifact.mkdir()
    with pytest.raises(result_store.OptimizationResultError, match="not a regular file"):
        result_store.validate_result_for_refresh(directory)
    assert artifact.is_dir()


def test_write_preserves_an_incomplete_visible_directory(tmp_path: Path) -> None:
    """Preserve a visible result directory lacking its manifest."""

    directory = tmp_path / "incomplete"
    directory.mkdir()
    marker = directory / "user-note.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(result_store.OptimizationResultError, match="no completion manifest"):
        _save_result(directory)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_results_guide_rejects_a_directory_in_place_of_the_file(tmp_path: Path) -> None:
    """Preserve an existing non-file guide rather than replacing its contents."""

    target = result_store.OptimizationResultTarget(
        workspace=tmp_path,
        foreground_scenario="remind_mfa_default",
        allocated_cc_result="pyaesa_h12",
        optimization_lcia_selection="ef31",
        readable_request_selection="blocks_integrated",
        initial_grid_step_pct=20,
        endpoint_refinement_steps_pct=(10, 5),
    )
    guide = tmp_path / "results/remind_mfa_default/optimization_results" / "README.md"
    guide.mkdir(parents=True)

    with pytest.raises(result_store.OptimizationResultError, match="not a regular file"):
        result_store.ensure_optimization_results_guide(target)
    assert guide.is_dir()
