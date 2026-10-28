"""Focused scenario compilation contracts."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scenario_fixtures import _define, _workspace, compile_scenario_payload

from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.run_mfa.scenario.scenario_contract import (
    ScenarioConflictError,
    ScenarioError,
    ScenarioFeasibilityError,
)


def test_scalar_trajectory_uses_each_regional_2019_start_and_persists_target() -> None:
    compiled = compile_scenario_payload(
        {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": "collection_trajectory",
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
    )
    rows = compiled.compiled_parameters.loc[
        compiled.compiled_parameters["parameter_id"].eq("collection_rate")
    ]
    target = rows.loc[rows["year"].isin({2050, 2075, 2100})]
    np.testing.assert_allclose(target["resolved_value"], 0.8, rtol=0.0, atol=0.0)
    first = rows.loc[rows["year"].eq(2020)]
    expected = first["reference_value"] + (0.8 - first["reference_value"]) / 31.0
    np.testing.assert_allclose(first["resolved_value"], expected, rtol=0.0, atol=1e-15)
    assert set(rows["value_mode"]) == {"linear_from_2019_absolute"}
    assert set(rows["controlling_change_point_year"]) == {2050}
    assert len(compiled.normalized_change_points) == 12
    assert {point["region"] for point in compiled.normalized_change_points} == set(rows["region"])


@pytest.mark.parametrize(
    ("target_mode", "target_value", "expected"),
    (
        ("multiplier_from_2019", 1.25, lambda reference: reference * 1.25),
        ("delta_from_2019", 2.0, lambda reference: reference + 2.0),
    ),
)
def test_scalar_trajectory_relative_target_modes(
    target_mode: str,
    target_value: float,
    expected: Callable[[float], float],
) -> None:
    compiled = compile_scenario_payload(
        {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": f"lifetime_{target_mode}",
            "ssp_scenario": "SSP2",
            "changes": [
                {
                    "parameter": "lifetime_mean",
                    "good": "Transportation",
                    "trajectory": {
                        "mode": "linear_from_2019",
                        "target_year": 2075,
                        "target_mode": target_mode,
                        "target_value": target_value,
                    },
                }
            ],
        }
    )
    rows = compiled.compiled_parameters.loc[
        compiled.compiled_parameters["parameter_id"].eq("lifetime_mean")
        & compiled.compiled_parameters["good"].eq("Transportation")
        & compiled.compiled_parameters["year"].eq(2100)
    ]
    reference = float(rows["reference_value"].iloc[0])
    assert float(rows["resolved_value"].iloc[0]) == pytest.approx(expected(reference))


def test_scenario_rejects_nonfinite_values_created_by_relative_arithmetic() -> None:
    payload = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": "overflowing_stock",
        "ssp_scenario": "SSP2",
        "changes": [
            {
                "parameter": "stock_projection",
                "selector": "uniform_all_h12_goods",
                "changes": [
                    {
                        "year": 2020,
                        "mode": "multiplier_from_default",
                        "value": 1e308,
                    }
                ],
            }
        ],
    }

    with pytest.raises(ScenarioFeasibilityError, match="resolved values must all be finite"):
        compile_scenario_payload(payload)


def test_grouped_share_trajectories_are_complete_and_persist() -> None:
    treatment_targets = {
        "mechanical_recycling_rate": 0.4,
        "chemical_recycling_rate": 0.1,
        "incineration_rate": 0.3,
        "landfill_share": 0.2,
    }
    feedstock_targets = {
        "fossil_share": 0.7,
        "bio_production_rate": 0.2,
        "daccu_production_rate": 0.1,
    }
    compiled = compile_scenario_payload(
        {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": "grouped_trajectories",
            "ssp_scenario": "SSP2",
            "changes": [
                {
                    "parameter_group": "collected_waste_treatment_shares",
                    "selector": "uniform_all_h12",
                    "trajectory": {
                        "mode": "linear_from_2019",
                        "target_year": 2050,
                        "target_values": treatment_targets,
                    },
                },
                {
                    "parameter_group": "virgin_feedstock_shares",
                    "selector": "uniform_all_h12",
                    "trajectory": {
                        "mode": "linear_from_2019",
                        "target_year": 2050,
                        "target_values": feedstock_targets,
                    },
                },
            ],
        }
    )
    table = compiled.compiled_parameters
    for year in (2050, 2075, 2100):
        rows = table.loc[table["year"].eq(year)]
        treatment = rows.loc[
            rows["parameter_id"].isin(
                {
                    "mechanical_recycling_rate",
                    "chemical_recycling_rate",
                    "incineration_rate",
                }
            )
        ].pivot(index="region", columns="parameter_id", values="resolved_value")
        feedstock = rows.loc[
            rows["parameter_id"].isin({"bio_production_rate", "daccu_production_rate"})
        ].pivot(index="region", columns="parameter_id", values="resolved_value")
        np.testing.assert_allclose(treatment["mechanical_recycling_rate"], 0.4, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(treatment["chemical_recycling_rate"], 0.1, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(treatment["incineration_rate"], 0.3, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(1.0 - treatment.sum(axis=1), 0.2, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(feedstock["bio_production_rate"], 0.2, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(feedstock["daccu_production_rate"], 0.1, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(1.0 - feedstock.sum(axis=1), 0.7, rtol=0.0, atol=1e-15)
    normalized = list(compiled.normalized_change_points)
    assert len(normalized) == 24
    assert {point["parameter_group"] for point in normalized} == {
        "collected_waste_treatment_shares",
        "virgin_feedstock_shares",
    }
    assert all(point["target_year"] == 2050 for point in normalized)


def test_calculated_landfill_share_rejects_an_independent_trajectory() -> None:
    """Require treatment-share trajectories to use their complete parameter group."""

    with pytest.raises(
        ScenarioError,
        match="parameter_group='collected_waste_treatment_shares'.*all four",
    ):
        compile_scenario_payload(
            {
                "schema_version": "pytoce_foreground_scenario_schema_v1",
                "foreground_scenario": "independent_landfill_share",
                "ssp_scenario": "SSP2",
                "changes": [
                    {
                        "parameter": "landfill_share",
                        "region": "EUR",
                        "trajectory": {
                            "mode": "linear_from_2019",
                            "target_year": 2050,
                            "target_value": 0.2,
                        },
                    }
                ],
            }
        )


def test_landfill_share_error_names_the_first_mismatching_year() -> None:
    """Report the year where a fixed landfill share first stops matching the residual."""

    with pytest.raises(
        ScenarioFeasibilityError,
        match=(
            r"landfill_share is 0\.23723445897878392 for EUR in 2021, .*"
            r"leave 0\.23344327693192679"
        ),
    ):
        compile_scenario_payload(
            {
                "schema_version": "pytoce_foreground_scenario_schema_v1",
                "foreground_scenario": "fixed_landfill_share",
                "ssp_scenario": "SSP2",
                "changes": [
                    {
                        "parameter": "landfill_share",
                        "region": "EUR",
                        "changes": [
                            {
                                "year": 2020,
                                "mode": "absolute",
                                "value": 0.23723445897878392,
                            }
                        ],
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("trajectory", "message"),
    (
        (
            {
                "mode": "linear_from_2019",
                "target_year": 2019,
                "target_values": {
                    "mechanical_recycling_rate": 0.4,
                    "chemical_recycling_rate": 0.1,
                    "incineration_rate": 0.3,
                    "landfill_share": 0.2,
                },
            },
            "2020 through 2100",
        ),
        (
            {
                "mode": "linear_from_2019",
                "target_year": 2050,
                "target_values": {
                    "mechanical_recycling_rate": 0.4,
                    "chemical_recycling_rate": 0.1,
                    "incineration_rate": 0.3,
                },
            },
            "requires every exact member",
        ),
        (
            {
                "mode": "linear_from_2019",
                "target_year": 2050,
                "target_values": {
                    "mechanical_recycling_rate": -0.1,
                    "chemical_recycling_rate": 0.2,
                    "incineration_rate": 0.4,
                    "landfill_share": 0.5,
                },
            },
            "nonnegative and sum to one",
        ),
        (
            {
                "mode": "linear_from_2019",
                "target_year": 2050,
                "target_values": {
                    "mechanical_recycling_rate": 0.4,
                    "chemical_recycling_rate": 0.1,
                    "incineration_rate": 0.3,
                    "landfill_share": 0.3,
                },
            },
            "nonnegative and sum to one",
        ),
    ),
)
def test_grouped_share_trajectory_rejects_invalid_contract(
    trajectory: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ScenarioError, match=message):
        compile_scenario_payload(
            {
                "schema_version": "pytoce_foreground_scenario_schema_v1",
                "foreground_scenario": "invalid_group",
                "ssp_scenario": "SSP2",
                "changes": [
                    {
                        "parameter_group": "collected_waste_treatment_shares",
                        "selector": "uniform_all_h12",
                        "trajectory": trajectory,
                    }
                ],
            }
        )


def test_grouped_and_individual_trajectory_overlap_is_rejected() -> None:
    with pytest.raises(ScenarioConflictError, match="cannot be combined"):
        compile_scenario_payload(
            {
                "schema_version": "pytoce_foreground_scenario_schema_v1",
                "foreground_scenario": "overlapping_group",
                "ssp_scenario": "SSP2",
                "changes": [
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
                    {
                        "parameter": "bio_production_rate",
                        "region": "EUR",
                        "changes": [{"year": 2050, "mode": "absolute", "value": 0.2}],
                    },
                ],
            }
        )


def test_sparse_rules_persist_as_rules_against_annual_references(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        "collection_schedule",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: collection_schedule
ssp_scenario: SSP2
changes:
  - parameter: collection_rate
    region: EUR
    changes:
      - year: 2025
        mode: retain_2019
      - year: 2050
        mode: absolute
        value: 0.75
  - parameter: collection_rate
    region: USA
    changes:
      - year: 2030
        mode: multiplier_from_default
        value: 0.8
""".strip()
        + "\n",
    )
    compiled = compile_foreground_scenario(root, "collection_schedule")
    table = pd.read_parquet(compiled.compiled_parameters_path)
    collection = table.loc[table["parameter_id"].eq("collection_rate")]
    eur = collection.loc[collection["region"].eq("EUR")].set_index("year")
    usa = collection.loc[collection["region"].eq("USA")].set_index("year")

    assert eur.loc[2024, "resolved_value"] == eur.loc[2024, "default_value"]
    assert eur.loc[2025, "resolved_value"] == pytest.approx(eur.loc[2025, "reference_value"])
    assert eur.loc[2049, "resolved_value"] == pytest.approx(eur.loc[2025, "resolved_value"])
    assert eur.loc[2050, "resolved_value"] == pytest.approx(0.75)
    assert eur.loc[2100, "resolved_value"] == pytest.approx(0.75)
    assert usa.loc[2030, "resolved_value"] == pytest.approx(0.8 * usa.loc[2030, "default_value"])
    assert usa.loc[2100, "resolved_value"] == pytest.approx(0.8 * usa.loc[2100, "default_value"])
    assert usa.loc[2030, "resolved_value"] != pytest.approx(usa.loc[2100, "resolved_value"])


def test_conflicting_change_points_fail_independently_of_file_order(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        "conflict",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: conflict
ssp_scenario: SSP2
changes:
  - parameter: collection_rate
    region: EUR
    changes:
      - {year: 2030, mode: absolute, value: 0.5}
  - parameter: collection_rate
    region: EUR
    changes:
      - {year: 2030, mode: absolute, value: 0.6}
""".strip()
        + "\n",
    )
    with pytest.raises(ScenarioConflictError, match="Duplicate or overlapping"):
        compile_foreground_scenario(root, "conflict")


def test_no_implicit_scalar_broadcast_or_non_ssp2(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        "implicit",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: implicit
ssp_scenario: SSP2
changes:
  - parameter: collection_rate
    changes:
      - {year: 2030, mode: absolute, value: 0.5}
""".strip()
        + "\n",
    )
    with pytest.raises(ScenarioError, match="requires explicit region"):
        compile_foreground_scenario(root, "implicit")

    _define(
        root,
        "wrong_ssp",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: wrong_ssp
ssp_scenario: SSP1
changes: []
""".strip()
        + "\n",
    )
    with pytest.raises(ScenarioError, match="only accepted"):
        compile_foreground_scenario(root, "wrong_ssp")


def test_user_treatment_partition_is_not_silently_normalized(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        "invalid_partition",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: invalid_partition
ssp_scenario: SSP2
changes:
  - parameter: mechanical_recycling_rate
    region: EUR
    changes:
      - {year: 2030, mode: absolute, value: 0.8}
  - parameter: chemical_recycling_rate
    region: EUR
    changes:
      - {year: 2030, mode: absolute, value: 0.3}
  - parameter: incineration_rate
    region: EUR
    changes:
      - {year: 2030, mode: absolute, value: 0.1}
""".strip()
        + "\n",
    )
    with pytest.raises(ScenarioFeasibilityError, match="must total at most 1"):
        compile_foreground_scenario(root, "invalid_partition")


def test_material_change_requires_complete_new_inflow_partition(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        "incomplete_material",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: incomplete_material
ssp_scenario: SSP2
changes:
  - parameter: plastic_family_share
    region: EUR
    material: PET
    good: Packaging
    changes:
      - {year: 2030, mode: absolute, value: 0.5}
""".strip()
        + "\n",
    )
    with pytest.raises(ScenarioFeasibilityError, match="Provide all plastic families"):
        compile_foreground_scenario(root, "incomplete_material")


@pytest.mark.parametrize(
    ("name", "body", "message"),
    (
        (
            "unknown_top",
            "unexpected: true\nchanges: []",
            "Unsupported top-level",
        ),
        (
            "unknown_entry",
            "changes:\n"
            "  - parameter: collection_rate\n"
            "    region: EUR\n"
            "    unexpected: true\n"
            "    changes:\n"
            "      - {year: 2030, mode: retain_default}",
            "Unsupported parameter-entry",
        ),
        (
            "unknown_change",
            "changes:\n"
            "  - parameter: collection_rate\n"
            "    region: EUR\n"
            "    changes:\n"
            "      - {year: 2030, mode: retain_default, unexpected: true}",
            "Unsupported change-point",
        ),
        (
            "null_value",
            "changes:\n"
            "  - parameter: collection_rate\n"
            "    region: EUR\n"
            "    changes:\n"
            "      - {year: 2030, mode: retain_default, value: null}",
            "omit the field",
        ),
        (
            "fractional_year",
            "changes:\n"
            "  - parameter: collection_rate\n"
            "    region: EUR\n"
            "    changes:\n"
            "      - {year: 2020.5, mode: retain_default}",
            "integer year",
        ),
        (
            "non_text_key",
            "1: unsupported\nchanges: []",
            "cannot be read as UTF-8 YAML",
        ),
    ),
)
def test_yaml_grammar_rejects_unknown_fields_and_explicit_null(
    tmp_path: Path,
    name: str,
    body: str,
    message: str,
) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        name,
        (
            "schema_version: pytoce_foreground_scenario_schema_v1\n"
            f"foreground_scenario: {name}\n"
            "ssp_scenario: SSP2\n"
            f"{body}\n"
        ),
    )

    with pytest.raises(ScenarioError, match=message):
        compile_foreground_scenario(root, name)


def test_yaml_grammar_rejects_duplicate_mapping_keys(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        "duplicate_key",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: duplicate_key
ssp_scenario: SSP2
description: first
description: second
changes: []
""".strip()
        + "\n",
    )

    with pytest.raises(ScenarioError, match="cannot be read as UTF-8 YAML"):
        compile_foreground_scenario(root, "duplicate_key")


@pytest.mark.parametrize(
    ("parameter", "suffix", "message"),
    (
        (
            "collection_rate",
            "",
            "not valid",
        ),
        (
            "waste_imports",
            "\n        value: 1.0",
            "does not accept value",
        ),
        (
            "waste_exports",
            "\n        interpolation: linear",
            "Ordinary change points are stepwise",
        ),
    ),
)
def test_deferred_waste_trade_mode_rejects_illegal_authoring(
    tmp_path: Path,
    parameter: str,
    suffix: str,
    message: str,
) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        "illegal_deferred_trade",
        (
            "schema_version: pytoce_foreground_scenario_schema_v1\n"
            "foreground_scenario: illegal_deferred_trade\n"
            "ssp_scenario: SSP2\n"
            "changes:\n"
            f"  - parameter: {parameter}\n"
            "    region: EUR\n"
            "    changes:\n"
            "      - year: 2020\n"
            "        mode: retain_2019_propensity_pretrade_collected"
            f"{suffix}\n"
        ),
    )

    with pytest.raises(ScenarioError, match=message):
        compile_foreground_scenario(root, "illegal_deferred_trade")
