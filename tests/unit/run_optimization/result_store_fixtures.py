"""Complete selected-result products shared by persistence owner tests."""

from pathlib import Path

import pandas as pd

from pytoce.run_optimization.runtime import result_store

_SELECTED_TABLES = (
    "selected_coordinates.parquet",
    "selected_annual_interventions.parquet",
    "selection_roles.parquet",
    "selected_annual_acc_asr.parquet",
    "selected_environmental_scores.parquet",
    "selected_life_cycle_stage_scores.parquet",
    "selected_feedstock_requirements.parquet",
)
_PARENTS = {
    "allocated_cc": "pytoce_allocated_cc_v1::parent",
    "request": "pytoce_optimization_request_v1::parent",
}
_METHOD = "pytoce_method_selection_v1::ef31"
_PRODUCT = "pytoce_optimization_selected_results_v1::product"
_SUMMARY = {"lifetime_ceiling_years": 150, "sensitivity_performed": False}


def _tables(names: tuple[str, ...], *, value: float = -1.5) -> dict[str, pd.DataFrame]:
    """Return deterministic result tables for the requested filenames."""

    return {
        name: pd.DataFrame(
            {
                "coordinate": [name.removesuffix(".parquet")],
                "value": [value],
            }
        )
        for name in names
    }


def _save_result(
    directory: Path,
    *,
    tables: dict[str, pd.DataFrame] | None = None,
    summary: dict[str, object] | None = None,
    product_identity: str = _PRODUCT,
    refresh: bool = False,
) -> result_store.SavedOptimizationResult:
    """Write one complete selected optimization result."""

    frames = _tables(_SELECTED_TABLES) if tables is None else tables
    stage = frames["selected_life_cycle_stage_scores.parquet"]
    with result_store.stage_selected_life_cycle_scores(directory) as writer:
        writer.start(stage)
        writer.write(stage)
        completed: dict[str, pd.DataFrame | result_store.StagedTable] = dict(frames)
        completed["selected_life_cycle_stage_scores.parquet"] = writer.finish()
        return result_store.write_optimization_result(
            directory,
            product_identity=product_identity,
            parent_identities=_PARENTS,
            method_identity=_METHOD,
            tables=completed,
            summary=_SUMMARY if summary is None else summary,
            refresh=refresh,
        )
