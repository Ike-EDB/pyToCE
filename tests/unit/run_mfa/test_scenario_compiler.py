import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from scenario_fixtures import _define, _workspace, compile_scenario_payload

from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.run_mfa.scenario.scenario_contract import (
    ScenarioError,
    ScenarioFeasibilityError,
)


@pytest.mark.parametrize(
    ("bio_rule", "dac_share", "invalid"),
    [
        ({"changes": [{"year": 2020, "mode": "absolute", "value": 0.8}]}, 0.3, True),
        (
            {
                "trajectory": {
                    "mode": "linear_from_2019",
                    "target_year": 2030,
                    "target_mode": "absolute",
                    "target_value": 0.8,
                }
            },
            0.200000000000001,
            True,
        ),
        ({"changes": [{"year": 2020, "mode": "absolute", "value": 0.8}]}, 0.2, False),
    ],
)
def test_feedstock_partition_is_admitted_before_compiled_publication(
    tmp_path: Path,
    bio_rule: dict[str, object],
    dac_share: float,
    invalid: bool,
) -> None:
    """Check resolved annual share overlap while preserving valid values and source bytes."""
    root = _workspace(tmp_path)
    name = "feedstock_partition"
    payload = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": name,
        "ssp_scenario": "SSP2",
        "changes": [
            {"parameter": "bio_production_rate", "region": "EUR", **bio_rule},
            {
                "parameter": "daccu_production_rate",
                "region": "EUR",
                "changes": [
                    {"year": 2020, "mode": "absolute", "value": 0.0},
                    {"year": 2050, "mode": "absolute", "value": dac_share},
                ],
            },
        ],
    }
    _define(root, name, json.dumps(payload))
    source = root / "data_raw/foreground_scenarios/scenarios" / f"{name}.yaml"
    source_bytes = source.read_bytes()
    if invalid:
        with pytest.raises(
            ScenarioFeasibilityError,
            match="Biogenic and DACCU shares for EUR in 2050 total .*must total at most 1",
        ):
            compile_foreground_scenario(root, name)
        assert source.read_bytes() == source_bytes
        assert not (root / "data_processed/foreground_scenarios" / name).exists()
        return

    compiled = compile_foreground_scenario(root, name)
    table = pd.read_parquet(compiled.compiled_parameters_path)
    shares = table.loc[
        table["region"].eq("EUR")
        & table["year"].eq(2050)
        & table["parameter_id"].isin(("bio_production_rate", "daccu_production_rate"))
    ].set_index("parameter_id")["resolved_value"]
    assert shares.to_dict() == {"bio_production_rate": 0.8, "daccu_production_rate": 0.2}
    assert shares.sum() == 1.0
    assert source.read_bytes() == source_bytes


def test_default_compiles_inputs_only_for_every_future_year(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    first = compile_foreground_scenario(root, "remind_mfa_default")
    second = compile_foreground_scenario(root, "remind_mfa_default")
    table = pd.read_parquet(first.compiled_parameters_path)

    assert first.status == "computed"
    assert second.status == "reused"
    assert first.foreground_scenario_identity == second.foreground_scenario_identity
    assert table["year"].min() == 2020
    assert table["year"].max() == 2100
    assert str(table["year"].dtype) == "int64"
    assert set(table["parameter_role"]) == {"editable_input", "derived_input"}
    assert set(table["value_mode"]) == {"retain_default"}
    assert {
        "virgin_production",
        "discard",
        "stock_inflow",
        "stock_outflow",
        "collected_mass",
    }.isdisjoint(table["parameter_id"])
    assert table.groupby("parameter_id")["year"].nunique().eq(81).all()


def test_stepwise_relative_lifetimes_preserve_reference_and_change_years(tmp_path: Path) -> None:
    """Persist successive default/2019-relative YAML rules for only the selected good."""
    root = _workspace(tmp_path)
    name = "stepwise_relative_lifetime"
    _define(
        root,
        name,
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: stepwise_relative_lifetime
ssp_scenario: SSP2
changes:
  - parameter: lifetime_mean
    good: Transportation
    changes:
      - {year: 2030, mode: delta_from_default, value: 2.0}
      - {year: 2040, mode: multiplier_from_2019, value: 1.25}
      - {year: 2050, mode: delta_from_2019, value: 3.0}
""".strip()
        + "\n",
    )
    compiled = compile_foreground_scenario(root, name)
    table = pd.read_parquet(compiled.compiled_parameters_path)
    means = table.loc[table["parameter_id"].eq("lifetime_mean")]
    transportation = means.loc[means["good"].eq("Transportation")].set_index("year")

    assert transportation.index.tolist() == list(range(2020, 2101))
    # The admitted source mean for Transportation is 13 years, including 2019.
    assert transportation["default_value"].eq(13.0).all()
    assert transportation["reference_value"].eq(13.0).all()
    for start, end, mode, expected in (
        (2020, 2029, "retain_default", 13.0),
        (2030, 2039, "delta_from_default", 13.0 + 2.0),
        (2040, 2049, "multiplier_from_2019", 13.0 * 1.25),
        (2050, 2100, "delta_from_2019", 13.0 + 3.0),
    ):
        interval = transportation.loc[start:end]
        assert interval["resolved_value"].eq(expected).all()
        assert interval["value_mode"].eq(mode).all()
        if start == 2020:
            assert interval["controlling_change_point_year"].isna().all()
        else:
            assert interval["controlling_change_point_year"].eq(start).all()

    unchanged = means.loc[means["good"].ne("Transportation")]
    assert unchanged["good"].nunique() == 6
    assert unchanged["resolved_value"].eq(unchanged["default_value"]).all()
    assert unchanged["value_mode"].eq("retain_default").all()
    assert unchanged["controlling_change_point_year"].isna().all()


def test_in_memory_payload_matches_file_compilation_without_writing_files(
    tmp_path: Path,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": "in_memory_collection_schedule",
        "ssp_scenario": "SSP2",
        "changes": [
            {
                "parameter": "collection_rate",
                "region": "EUR",
                "changes": [
                    {"year": 2030, "mode": "absolute", "value": 0.7},
                    {
                        "year": 2050,
                        "mode": "multiplier_from_default",
                        "value": 0.9,
                    },
                ],
            }
        ],
    }

    compiled_from_payload = compile_scenario_payload(payload)
    assert list(tmp_path.iterdir()) == []

    root = _workspace(tmp_path)
    _define(
        root,
        "in_memory_collection_schedule",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: in_memory_collection_schedule
ssp_scenario: SSP2
changes:
  - parameter: collection_rate
    region: EUR
    changes:
      - {year: 2030, mode: absolute, value: 0.7}
      - {year: 2050, mode: multiplier_from_default, value: 0.9}
""".strip()
        + "\n",
    )
    compiled = compile_foreground_scenario(root, "in_memory_collection_schedule")
    compiled_table = pd.read_parquet(compiled.compiled_parameters_path)

    assert (
        compiled_from_payload.foreground_scenario_identity == compiled.foreground_scenario_identity
    )
    pd.testing.assert_frame_equal(
        compiled_from_payload.compiled_parameters,
        compiled_table,
        check_exact=True,
    )
    manifest = json.loads(compiled.manifest_path.read_text(encoding="utf-8"))
    assert (
        list(compiled_from_payload.normalized_change_points)
        == manifest["normalized_scenario"]["change_points"]
    )


def test_description_and_independent_entry_order_do_not_change_identity(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    name = "equivalent_authoring"
    first_definition = """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: equivalent_authoring
ssp_scenario: SSP2
description: First authoring description.
changes:
  - parameter: collection_rate
    region: EUR
    changes:
      - {year: 2030, mode: absolute, value: 0.7}
  - parameter: collection_rate
    region: USA
    changes:
      - {year: 2040, mode: multiplier_from_default, value: 0.8}
""".strip()
    second_definition = """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: equivalent_authoring
ssp_scenario: SSP2
description: A different human-readable description.
changes:
  - parameter: collection_rate
    region: USA
    changes:
      - {year: 2040, mode: multiplier_from_default, value: 0.8}
  - parameter: collection_rate
    region: EUR
    changes:
      - {year: 2030, mode: absolute, value: 0.7}
""".strip()
    _define(root, name, first_definition + "\n")
    first = compile_foreground_scenario(root, name)
    first_manifest = first.manifest_path.read_bytes()
    _define(root, name, second_definition + "\n")

    second = compile_foreground_scenario(root, name, refresh=True)

    assert second.status == "computed"
    assert second.foreground_scenario_identity == first.foreground_scenario_identity
    assert second.manifest_path.read_bytes() == first_manifest


@pytest.mark.parametrize(
    ("schedule", "message"),
    (
        (
            {
                "trajectory": {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_mode": [],
                    "target_value": 0.8,
                }
            },
            "trajectory.target_mode .* is not valid",
        ),
        (
            {"changes": [{"year": 2030, "mode": "absolute", "value": 10**400}]},
            "Mode 'absolute' requires a finite numeric value",
        ),
        (
            {
                "trajectory": {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_mode": "absolute",
                    "target_value": 10**400,
                }
            },
            "trajectory.target_value must be a finite numeric value",
        ),
    ),
    ids=("nontext_target_mode", "overflow_change_point", "overflow_trajectory"),
)
def test_invalid_yaml_scalars_report_scenario_errors_before_publication(
    tmp_path: Path,
    schedule: dict[str, object],
    message: str,
) -> None:
    """Translate malformed YAML scalars at admission and preserve their source."""
    root = _workspace(tmp_path)
    name = "invalid_scalar"
    payload = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": name,
        "ssp_scenario": "SSP2",
        "changes": [{"parameter": "collection_rate", "region": "EUR", **schedule}],
    }
    _define(root, name, yaml.safe_dump(payload))
    source = root / "data_raw/foreground_scenarios/scenarios" / f"{name}.yaml"
    source_bytes = source.read_bytes()

    with pytest.raises(ScenarioError, match=message):
        compile_foreground_scenario(root, name)

    assert source.read_bytes() == source_bytes
    assert not (root / "data_processed/foreground_scenarios" / name).exists()


def test_deferred_waste_trade_modes_follow_compiled_null_contract(tmp_path: Path) -> None:
    """Write null masses for both trade modes whose quantities are calculated during MFA."""
    root = _workspace(tmp_path)
    name = "deferred_mode_contract"
    payload = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": name,
        "ssp_scenario": "SSP2",
        "changes": [
            {
                "parameter": parameter,
                "region": "EUR",
                "changes": [{"year": 2020, "mode": mode}],
            }
            for parameter, mode in (
                ("waste_imports", "retain_2019_propensity_pretrade_collected"),
                ("waste_exports", "retain_default_propensity_pretrade_collected"),
            )
        ],
    }
    _define(root, name, yaml.safe_dump(payload))
    compiled = compile_foreground_scenario(root, name)
    table = pd.read_parquet(compiled.compiled_parameters_path)
    schema_path = (
        root
        / "data_raw/foreground_scenarios/schemas/compiled_annual_input_parameters_schema_v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    unresolved = table.loc[table["resolved_value"].isna()]
    assert set(unresolved["value_mode"]) == set(schema["nullable_resolved_value_modes"])
    assert set(unresolved["parameter_id"]) == {"waste_imports", "waste_exports"}
    assert set(unresolved["deferred_denominator_id"]) == {"local_pretrade_collected_waste"}
