"""Focused scenario compilation contracts."""

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from scenario_fixtures import _workspace, compile_scenario_payload

from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.run_mfa.scenario.scenario_contract import (
    ScenarioError,
)
from pytoce.set_workspace.set_workspace import set_workspace


def test_scenario_rejects_an_unknown_schema_with_v1_action() -> None:
    with pytest.raises(ScenarioError, match="expected.*schema_v1.*installed template"):
        compile_scenario_payload(
            {
                "schema_version": "invalid_schema",
                "foreground_scenario": "invalid_scenario",
                "ssp_scenario": "SSP2",
                "changes": [],
            }
        )


def test_reserved_default_rejects_user_changes(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    path = root / "data_raw" / "foreground_scenarios" / "scenarios" / "remind_mfa_default.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["description"] = "User changed this reserved scenario."
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ScenarioError, match="reserved"):
        compile_foreground_scenario(root, "remind_mfa_default")


def test_installed_scenario_schema_matches_the_compiler_grammar(tmp_path: Path) -> None:
    report = set_workspace(tmp_path)
    schema_path = (
        report.data_raw / "foreground_scenarios" / "schemas" / "foreground_scenario_schema_v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    def document(
        change_point: dict[str, object],
        *,
        parameter: str = "collection_rate",
    ) -> dict[str, object]:
        return {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": "schema_fixture",
            "ssp_scenario": "SSP2",
            "changes": [
                {
                    "parameter": parameter,
                    "region": "EUR",
                    "changes": [change_point],
                }
            ],
        }

    valid = (
        {"year": 2030, "mode": "absolute", "value": 0.7},
        {"year": 2030, "mode": "retain_default"},
        {"year": 2030, "mode": "absolute", "value": 0.7, "interpolation": "step"},
    )
    invalid = (
        {"year": 2030, "mode": "absolute"},
        {"year": 2030, "mode": "retain_default", "value": 0.7},
        {"year": 2030, "mode": "retain_2019", "interpolation": "linear"},
        {"year": 2030, "mode": "absolute", "value": 0.7, "unknown": True},
    )

    assert all(validator.is_valid(document(point)) for point in valid)
    assert all(not validator.is_valid(document(point)) for point in invalid)
    deferred = {"year": 2020, "mode": "retain_2019_propensity_pretrade_collected"}
    assert validator.is_valid(document(deferred, parameter="waste_imports"))
    assert validator.is_valid(document(deferred, parameter="waste_exports"))
    assert not validator.is_valid(document(deferred))
    assert not validator.is_valid(document({**deferred, "value": 1.0}, parameter="waste_imports"))
    assert not validator.is_valid(
        document({**deferred, "interpolation": "linear"}, parameter="waste_exports")
    )
    scalar_trajectory = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": "schema_fixture",
        "ssp_scenario": "SSP2",
        "changes": [
            {
                "parameter": "collection_rate",
                "selector": "uniform_all_h12",
                "trajectory": {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_mode": "absolute",
                    "target_value": 0.8,
                },
            }
        ],
    }
    treatment_trajectory = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": "schema_fixture",
        "ssp_scenario": "SSP2",
        "changes": [
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
            }
        ],
    }
    assert validator.is_valid(scalar_trajectory)
    assert validator.is_valid(treatment_trajectory)
    assert not validator.is_valid(
        {
            **treatment_trajectory,
            "changes": [
                {
                    **treatment_trajectory["changes"][0],
                    "trajectory": {
                        **treatment_trajectory["changes"][0]["trajectory"],
                        "target_values": {
                            "mechanical_recycling_rate": 0.4,
                            "chemical_recycling_rate": 0.1,
                            "incineration_rate": 0.5,
                        },
                    },
                }
            ],
        }
    )
