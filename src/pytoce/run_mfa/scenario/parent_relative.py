"""Optimization-relative scenario assignments and compiled-table identity."""

from typing import Any, cast

import numpy as np
import pandas as pd

from pytoce.shared.runtime.filesystem import payload_sha256

from .rules import (
    _OPTIMIZATION_PARENT_OVERLAY_MODE,
    OPTIMIZATION_PARENT_OVERLAY_ORIGIN,
    _derive_lifetime_std,
)
from .scenario_contract import ScenarioEvaluation

OPTIMIZATION_PARENT_OVERLAY_METHOD_IDENTITY = "pytoce_optimization_parent_overlay_v3"
_OPTIMIZATION_PARENT_OVERLAY_SCHEMA = "pytoce_optimization_foreground_scenario_schema_v1"
_COORDINATE_COLUMNS = ("region", "material", "good", "process", "route")
_COMPILED_COLUMNS = (
    "year",
    *_COORDINATE_COLUMNS,
    "parameter_id",
    "unit",
    "parameter_role",
    "resolved_value",
    "value_mode",
    "value_origin",
    "controlling_change_point_year",
    "default_value",
    "reference_value",
    "deferred_propensity",
    "deferred_denominator_id",
    "direct_dependencies",
    "invalidated_descendants",
)


def compiled_scenario_table_identity(frame: pd.DataFrame) -> str:
    """Return the deterministic identity used by compiled scenario manifests."""

    canonical = frame.copy()
    for column in canonical.columns:
        if canonical[column].dtype == object:
            canonical[column] = canonical[column].where(canonical[column].notna(), "<null>")
    hashes = pd.util.hash_pandas_object(canonical, index=False, categorize=False).to_numpy()
    schema = [(column, str(frame[column].dtype)) for column in frame.columns]
    return payload_sha256(
        {
            "schema": schema,
            "rows": len(frame),
            "row_hashes": [int(value) for value in hashes],
        }
    )


def compile_parent_relative_scenario_assignments(
    parent: ScenarioEvaluation,
    foreground_scenario: str,
    assignments: pd.DataFrame,
) -> ScenarioEvaluation:
    """Overlay complete prospective optimization trajectories on one parent.

    Unselected rows retain the parent's exact values and execution metadata,
    except the spread derived from a changed lifetime mean.
    A trajectory which changes no parent value returns the parent object and
    identity unchanged. The admitted parent table identity is retained; only the
    newly derived table is hashed.
    """

    key_columns = ["parameter_id", "year", *_COORDINATE_COLUMNS]
    assignment_values = assignments.loc[:, [*key_columns, "resolved_value"]].rename(
        columns={"resolved_value": "_overlay_resolved_value"}
    )
    assignment_values["_overlay_assignment"] = True
    combined = parent.compiled_parameters.merge(
        assignment_values,
        on=key_columns,
        how="left",
        sort=False,
    )
    assigned = combined["_overlay_assignment"].notna()
    assigned_values = combined.loc[assigned, "_overlay_resolved_value"].to_numpy(dtype=np.float64)
    parent_values = combined.loc[assigned, "resolved_value"].to_numpy(dtype=np.float64)
    changed_assigned = assigned.copy()
    changed_assigned.loc[assigned] = assigned_values != parent_values
    if not changed_assigned.any():
        return parent

    combined.loc[assigned, "resolved_value"] = assigned_values
    combined.loc[changed_assigned, "value_mode"] = _OPTIMIZATION_PARENT_OVERLAY_MODE
    combined.loc[changed_assigned, "value_origin"] = OPTIMIZATION_PARENT_OVERLAY_ORIGIN
    combined.loc[changed_assigned, "controlling_change_point_year"] = combined.loc[
        changed_assigned, "year"
    ].to_numpy()
    combined.loc[changed_assigned, "reference_value"] = parent_values[
        assigned_values != parent_values
    ]
    compiled = cast(
        pd.DataFrame,
        combined.drop(columns=["_overlay_resolved_value", "_overlay_assignment"]).loc[
            :, list(_COMPILED_COLUMNS)
        ],
    )
    compiled = compiled.sort_values(
        ["parameter_id", "year", *_COORDINATE_COLUMNS], kind="stable"
    ).reset_index(drop=True)
    _derive_lifetime_std(compiled)
    overlay_points: list[dict[str, object]] = []
    overlay_frame = combined.loc[
        changed_assigned,
        [*key_columns, "resolved_value"],
    ].sort_values(key_columns, kind="stable")
    for row in overlay_frame.itertuples(index=False):
        point: dict[str, object] = {
            "parameter": str(row.parameter_id),
            "year": int(cast(Any, row.year)),
            "mode": _OPTIMIZATION_PARENT_OVERLAY_MODE,
            "interpolation": "annual_explicit",
            "value": float(cast(Any, row.resolved_value)),
        }
        for column in _COORDINATE_COLUMNS:
            value = getattr(row, column)
            if pd.notna(value):
                point[column] = str(value)
        overlay_points.append(point)
    normalized_change_points = (
        *parent.normalized_change_points,
        *overlay_points,
    )
    compiled_table_sha256 = compiled_scenario_table_identity(compiled)
    identity_payload = {
        "schema_version": _OPTIMIZATION_PARENT_OVERLAY_SCHEMA,
        "method_identity": OPTIMIZATION_PARENT_OVERLAY_METHOD_IDENTITY,
        "parent_foreground_scenario_identity": parent.foreground_scenario_identity,
        "parent_compiled_table_sha256": parent.compiled_table_sha256,
        "realized_compiled_table_sha256": compiled_table_sha256,
        "normalized_change_points_sha256": payload_sha256(normalized_change_points),
    }
    return ScenarioEvaluation(
        foreground_scenario=foreground_scenario,
        foreground_scenario_identity=(
            f"{_OPTIMIZATION_PARENT_OVERLAY_SCHEMA}::{payload_sha256(identity_payload)}"
        ),
        ssp_scenario="SSP2",
        compiled_parameters=compiled,
        normalized_change_points=normalized_change_points,
        compiled_table_sha256=compiled_table_sha256,
    )
