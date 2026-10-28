"""Stored scenario and method selection for optimization workers."""

import json
from pathlib import Path

import pandas as pd

from pytoce import set_workspace
from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.run_mfa.scenario.parent_relative import compiled_scenario_table_identity
from pytoce.run_optimization.calculation.worker_inputs import (
    _load_parent_scenario,
)


def test_parent_scenario_loader_accepts_exact_completed_artifacts(tmp_path: Path) -> None:
    """Preserve the compiled table and both forms of normalized scenario controls."""

    workspace = set_workspace(tmp_path).workspace
    name = "worker_parent"
    scenario_path = workspace / "data_raw/foreground_scenarios/scenarios" / f"{name}.yaml"
    scenario = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": name,
        "ssp_scenario": "SSP2",
        "changes": [
            {
                "parameter": "collection_rate",
                "region": "EUR",
                "changes": [
                    {"year": 2050, "mode": "absolute", "value": 0.7, "interpolation": "step"}
                ],
            },
            {
                "parameter": "collection_rate",
                "region": "USA",
                "trajectory": {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_mode": "absolute",
                    "target_value": 0.8,
                },
            },
        ],
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    compiled = compile_foreground_scenario(workspace, name)
    table = pd.read_parquet(compiled.compiled_parameters_path)
    manifest = json.loads(compiled.manifest_path.read_text(encoding="utf-8"))
    payload = {
        "foreground_scenario": compiled.foreground_scenario,
        "compiled_scenario_identity": compiled.foreground_scenario_identity,
        "compiled_scenario_manifest_sha256": compiled.manifest_sha256,
    }

    observed = _load_parent_scenario(workspace, payload)

    assert observed.foreground_scenario == name
    assert observed.foreground_scenario_identity == compiled.foreground_scenario_identity
    assert observed.ssp_scenario == "SSP2"
    assert observed.compiled_table_sha256 == compiled_scenario_table_identity(table)
    assert observed.compiled_table_sha256 == manifest["compiled_table"]["canonical_sha256"]
    pd.testing.assert_frame_equal(observed.compiled_parameters, table, check_exact=True)
    assert observed.normalized_change_points == (
        {
            "parameter": "collection_rate",
            "year": 2050,
            "mode": "absolute",
            "interpolation": "step",
            "region": "EUR",
            "value": 0.7,
        },
        {
            "mode": "linear_from_2019",
            "target_year": 2050,
            "target_mode": "absolute",
            "parameter": "collection_rate",
            "region": "USA",
            "target_value": 0.8,
        },
    )
