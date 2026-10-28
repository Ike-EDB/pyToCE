import json
from pathlib import Path

import numpy as np
import pandas as pd
from scenario_fixtures import compile_scenario_payload

from pytoce import set_workspace
from pytoce.run_mfa.model.result_tables import _resolved_input_view
from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.run_mfa.scenario.parent_relative import (
    compile_parent_relative_scenario_assignments,
)


def test_resolved_input_view_retains_the_requested_persistent_rule() -> None:
    compiled = pd.DataFrame(
        {
            "year": [2024, 2025, 2026],
            "region": ["EUR"] * 3,
            "material": [None] * 3,
            "good": [None] * 3,
            "process": [None] * 3,
            "route": [None] * 3,
            "parameter_id": ["collection_rate"] * 3,
            "unit": ["fraction"] * 3,
            "parameter_role": ["editable_input"] * 3,
            "resolved_value": [0.5, 0.48, 0.56],
            "value_mode": [
                "retain_default",
                "multiplier_from_default",
                "multiplier_from_default",
            ],
            "value_origin": [
                "remind_mfa_default",
                "user_foreground_scenario",
                "user_foreground_scenario",
            ],
            "controlling_change_point_year": pd.array(
                [pd.NA, 2025, 2025],
                dtype="Int64",
            ),
            "default_value": [0.5, 0.6, 0.7],
            "reference_value": [0.5, 0.6, 0.7],
            "direct_dependencies": ["discard"] * 3,
            "invalidated_descendants": ["collection|treatment"] * 3,
        }
    )
    change_points = [
        {
            "parameter": "collection_rate",
            "region": "EUR",
            "year": 2025,
            "mode": "multiplier_from_default",
            "interpolation": "step",
            "value": 0.8,
        }
    ]

    resolved = _resolved_input_view(compiled, change_points, pd.DataFrame())

    assert pd.isna(resolved.loc[0, "requested_value"])
    assert resolved.loc[1:, "requested_value"].tolist() == [0.8, 0.8]
    assert resolved.loc[1:, "resolved_value"].tolist() == [0.48, 0.56]
    np.testing.assert_allclose(
        resolved.loc[1:, "ratio_to_default"].to_numpy(dtype=float),
        [0.8, 0.8],
        rtol=1e-14,
        atol=0.0,
    )


def test_parent_overlay_retains_unmodified_parent_request_provenance() -> None:
    parent = compile_scenario_payload(
        {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": "custom_parent",
            "ssp_scenario": "SSP2",
            "changes": [
                {
                    "parameter": "collection_rate",
                    "region": "USA",
                    "changes": [
                        {
                            "year": 2030,
                            "mode": "multiplier_from_default",
                            "value": 0.9,
                        }
                    ],
                }
            ],
        }
    )
    columns = [
        "parameter_id",
        "year",
        "region",
        "material",
        "good",
        "process",
        "route",
        "resolved_value",
    ]
    assignments = parent.compiled_parameters.loc[
        parent.compiled_parameters["parameter_id"].eq("collection_rate")
        & parent.compiled_parameters["region"].eq("USA"),
        columns,
    ].reset_index(drop=True)
    assignments.loc[assignments["year"].ge(2050), "resolved_value"] = 0.8
    candidate = compile_parent_relative_scenario_assignments(
        parent,
        "optimization_candidate",
        assignments,
    )

    resolved = _resolved_input_view(
        candidate.compiled_parameters,
        candidate.normalized_change_points,
        pd.DataFrame(),
    )
    usa = resolved.loc[
        resolved["parameter_id"].eq("collection_rate") & resolved["region"].eq("USA")
    ].set_index("year")
    assert usa.loc[2030:2049, "requested_value"].eq(0.9).all()
    assert usa.loc[2030:2049, "value_origin"].eq("user_foreground_scenario").all()
    assert usa.loc[2050:2100, "requested_value"].eq(0.8).all()
    assert usa.loc[2050:2100, "value_origin"].eq("optimization_parent_overlay").all()


def test_resolved_input_view_uses_endogenous_waste_trade_mass() -> None:
    compiled = pd.DataFrame(
        {
            "year": [2020, 2021],
            "region": ["EUR", "EUR"],
            "material": [None, None],
            "good": [None, None],
            "process": [None, None],
            "route": [None, None],
            "parameter_id": ["waste_exports", "waste_exports"],
            "unit": ["Mt", "Mt"],
            "parameter_role": ["editable_input", "editable_input"],
            "resolved_value": [np.nan, np.nan],
            "deferred_propensity": [0.2, 0.2],
            "value_mode": ["retain_2019_propensity_pretrade_collected"] * 2,
            "value_origin": ["user_foreground_scenario"] * 2,
            "controlling_change_point_year": pd.array([2020, 2020], dtype="Int64"),
            "default_value": [0.0, 0.0],
            "reference_value": [0.2, 0.2],
            "direct_dependencies": ["collected_mass"] * 2,
            "invalidated_descendants": ["waste_trade|treatment"] * 2,
        }
    )
    change_points = [
        {
            "parameter": "waste_exports",
            "region": "EUR",
            "year": 2020,
            "mode": "retain_2019_propensity_pretrade_collected",
            "interpolation": "step",
        }
    ]
    ledger = pd.DataFrame(
        {
            "year": [2020, 2021],
            "region": ["EUR", "EUR"],
            "source_input_waste_imports_mt": [0.0, 0.0],
            "source_input_waste_exports_mt": [2.0, 3.0],
        }
    )

    resolved = _resolved_input_view(compiled, change_points, ledger)

    assert resolved["resolved_value"].tolist() == [2.0, 3.0]
    assert resolved["requested_value"].isna().all()
    assert resolved["input_status"].eq("scenario_value").all()


def test_resolved_input_view_preserves_scalar_and_grouped_trajectory_targets(
    tmp_path: Path,
) -> None:
    """Write authored targets and calculate group residuals separately."""

    workspace = set_workspace(tmp_path).workspace
    payload = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": "trajectory_targets",
        "ssp_scenario": "SSP2",
        "changes": [
            {
                "parameter": "collection_rate",
                "region": "EUR",
                "trajectory": {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_mode": "absolute",
                    "target_value": 0.8,
                },
            },
            {
                "parameter_group": "collected_waste_treatment_shares",
                "selector": "uniform_all_h12",
                "trajectory": {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_values": {
                        "mechanical_recycling_rate": 0.4,
                        "chemical_recycling_rate": 0.1,
                        "incineration_rate": 0.3,
                        "landfill_share": 0.2,
                    },
                },
            },
            {
                "parameter_group": "virgin_feedstock_shares",
                "selector": "uniform_all_h12",
                "trajectory": {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_values": {
                        "fossil_share": 0.7,
                        "bio_production_rate": 0.2,
                        "daccu_production_rate": 0.1,
                    },
                },
            },
        ],
    }
    source = workspace / "data_raw/foreground_scenarios/scenarios/trajectory_targets.yaml"
    source.write_text(json.dumps(payload), encoding="utf-8")
    compiled = compile_foreground_scenario(workspace, "trajectory_targets")
    table = pd.read_parquet(compiled.compiled_parameters_path)
    resolved = _resolved_input_view(
        table,
        compiled.normalized_change_points,
        pd.DataFrame(),
    )
    pd.testing.assert_frame_equal(resolved[table.columns], table, check_exact=True)
    for year in (2030, 2050, 2100):
        target = resolved.loc[
            resolved["year"].eq(year) & resolved["region"].eq("EUR"),
            ["parameter_id", "requested_value"],
        ].set_index("parameter_id")
        for parameter, value in (
            ("collection_rate", 0.8),
            ("mechanical_recycling_rate", 0.4),
            ("chemical_recycling_rate", 0.1),
            ("incineration_rate", 0.3),
            ("bio_production_rate", 0.2),
            ("daccu_production_rate", 0.1),
        ):
            assert target.loc[parameter, "requested_value"] == value
