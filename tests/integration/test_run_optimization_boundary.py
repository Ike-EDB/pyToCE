from pathlib import Path

import pandas as pd
import pytest

from pytoce import run_optimization, set_workspace
from pytoce.run_optimization.run_optimization import (
    RunOptimizationError,
    normalize_run_optimization_request,
)
from pytoce.run_optimization.runtime.result_store import (
    OptimizationResultTarget,
    StagedTable,
    stage_selected_life_cycle_scores,
    write_optimization_result,
)

_TABLES = (
    "selected_coordinates.parquet",
    "selected_annual_interventions.parquet",
    "selection_roles.parquet",
    "selected_annual_acc_asr.parquet",
    "selected_environmental_scores.parquet",
    "selected_life_cycle_stage_scores.parquet",
    "selected_feedstock_requirements.parquet",
)


def _request_target(
    workspace: Path, foreground: str = "frozen_2019_default_baseline"
) -> tuple[OptimizationResultTarget, str]:
    """Return the selected-result location for one parent and the default request."""
    request = normalize_run_optimization_request(
        foreground_scenario=foreground,
        allocated_cc_result="example_allocated_cc",
    )
    target = OptimizationResultTarget(
        workspace=workspace,
        foreground_scenario=foreground,
        allocated_cc_result=request.allocated_cc_result,
        optimization_lcia_selection="ef31",
        readable_request_selection=request.readable_request_selection,
        initial_grid_step_pct=20,
        endpoint_refinement_steps_pct=(10, 5),
    )
    return target, request.result_lcia_selection


def _write_completed_result(
    workspace: Path, foreground: str = "frozen_2019_default_baseline"
) -> Path:
    """Write the smallest complete selected result accepted by the real store."""
    target, result_lcia_selection = _request_target(workspace, foreground)
    selected_directory = target.selected_results(result_lcia_selection)
    frame = pd.DataFrame({"coordinate": ["fixture"], "value": [-1.0]})
    with stage_selected_life_cycle_scores(selected_directory) as writer:
        writer.start(frame)
        writer.write(frame)
        tables: dict[str, pd.DataFrame | StagedTable] = {name: frame for name in _TABLES}
        tables["selected_life_cycle_stage_scores.parquet"] = writer.finish()
        write_optimization_result(
            selected_directory,
            product_identity=f"pytoce_optimization_selected_results_v1::{foreground}",
            parent_identities={"request": f"fixture-request::{foreground}"},
            method_identity="fixture-method",
            tables=tables,
            summary={"selection": "fixture"},
        )
    return selected_directory


def test_run_optimization_reports_missing_required_lca_results(tmp_path: Path) -> None:
    set_workspace(tmp_path)

    with pytest.raises(RunOptimizationError, match="Cannot use the required LCA results"):
        run_optimization(
            foreground_scenario="frozen_2019_default_baseline",
            allocated_cc_result="example_allocated_cc",
        )


def test_run_optimization_reuses_a_completed_result_without_opening_parents(
    tmp_path: Path,
) -> None:
    workspace = set_workspace(tmp_path).workspace
    selected_directory = _write_completed_result(workspace)

    report = run_optimization(
        foreground_scenario="frozen_2019_default_baseline",
        allocated_cc_result="example_allocated_cc",
    )

    assert report.status == "reused"
    assert report.as_dict() == {
        "status": "reused",
        "workspace": str(workspace),
        "elapsed_seconds": report.elapsed_seconds,
        "products": [
            {
                "status": "reused",
                "foreground_scenario": "frozen_2019_default_baseline",
                "product_identity": (
                    "pytoce_optimization_selected_results_v1::frozen_2019_default_baseline"
                ),
                "result_directory": str(selected_directory),
                "manifest_path": str(selected_directory / "manifest.json"),
            }
        ],
    }


def test_run_optimization_reuses_each_selected_parent_independently(tmp_path: Path) -> None:
    """Read each completed result once."""
    workspace = set_workspace(tmp_path).workspace
    foregrounds = ("frozen_2019_default_baseline", "study_parent")
    directories = tuple(_write_completed_result(workspace, name) for name in foregrounds)

    report = run_optimization(
        foreground_scenario=("study_parent", "frozen_2019_default_baseline", "study_parent"),
        allocated_cc_result="example_allocated_cc",
    )

    assert report.status == "reused"
    assert tuple(product.foreground_scenario for product in report.products) == foregrounds
    assert tuple(product.result_directory for product in report.products) == directories
    assert len({product.product_identity for product in report.products}) == 2
    assert all(product.status == "reused" for product in report.products)


def test_completed_other_parent_cannot_replace_missing_selected_parent(tmp_path: Path) -> None:
    """Require the chosen parent's LCA even when another requested result is reusable."""
    workspace = set_workspace(tmp_path).workspace
    completed = _write_completed_result(workspace)
    with pytest.raises(
        RunOptimizationError,
        match="Cannot use the required LCA results.*foreground_scenario='study_parent'",
    ):
        run_optimization(
            foreground_scenario=("frozen_2019_default_baseline", "study_parent"),
            allocated_cc_result="example_allocated_cc",
        )
    assert (completed / "manifest.json").is_file()


def test_run_optimization_refresh_checks_required_lca_results(tmp_path: Path) -> None:
    workspace = set_workspace(tmp_path).workspace
    selected_directory = _write_completed_result(workspace)

    with pytest.raises(RunOptimizationError, match="Cannot use the required LCA results"):
        run_optimization(
            foreground_scenario="frozen_2019_default_baseline",
            allocated_cc_result="example_allocated_cc",
            refresh=True,
        )

    assert (selected_directory / "manifest.json").is_file()


def test_run_optimization_preserves_an_incomplete_result_directory(tmp_path: Path) -> None:
    workspace = set_workspace(tmp_path).workspace
    target, result_lcia_selection = _request_target(workspace)
    selected_directory = target.selected_results(result_lcia_selection)
    selected_directory.mkdir(parents=True)
    note = selected_directory / "user_note.txt"
    note.write_text("keep\n", encoding="utf-8")

    with pytest.raises(RunOptimizationError, match="exists without a completion manifest"):
        run_optimization(
            foreground_scenario="frozen_2019_default_baseline",
            allocated_cc_result="example_allocated_cc",
        )

    assert note.read_text(encoding="utf-8") == "keep\n"


def test_run_optimization_translates_an_invalid_results_guide(tmp_path: Path) -> None:
    workspace = set_workspace(tmp_path).workspace
    guide = workspace / "results/frozen_2019_default_baseline/optimization_results/README.md"
    guide.mkdir(parents=True)

    with pytest.raises(RunOptimizationError, match="Cannot use.*README.md"):
        run_optimization(
            foreground_scenario="frozen_2019_default_baseline",
            allocated_cc_result="example_allocated_cc",
        )

    assert guide.is_dir()
