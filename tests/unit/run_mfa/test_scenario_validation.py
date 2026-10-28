"""User-reachable foreground-scenario validation contracts."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue
from pytoce.run_mfa.scenario.compiler import _compile_payload
from pytoce.run_mfa.scenario.rules import _validate_scenario_payload
from pytoce.run_mfa.scenario.scenario_contract import (
    ScenarioConflictError,
    ScenarioError,
    ScenarioFeasibilityError,
)

_SCHEMA = "pytoce_foreground_scenario_schema_v1"


def _payload(changes: object) -> dict[str, object]:
    """Return the smallest named SSP2 scenario around supplied authoring rules."""

    return {
        "schema_version": _SCHEMA,
        "foreground_scenario": "validation_case",
        "ssp_scenario": "SSP2",
        "changes": changes,
    }


def _compile(changes: object) -> None:
    """Compile one scenario through the canonical scenario owner."""

    _compile_payload(
        _validate_scenario_payload(_payload(changes), expected_name="validation_case"),
        source_path=Path("validation_case.yaml"),
    )


@pytest.mark.parametrize(
    ("value", "expected_name", "message"),
    (
        ([], "validation_case", "must be a YAML mapping"),
        (
            {
                "schema_version": _SCHEMA,
                "foreground_scenario": 7,
                "ssp_scenario": "SSP2",
                "changes": [],
            },
            "validation_case",
            "foreground_scenario must be text",
        ),
        (_payload([]), "different_name", "filename stem must equal"),
        ({**_payload([]), "description": 7}, "validation_case", "description must be text"),
        ({**_payload([]), "changes": {}}, "validation_case", "changes must be a list"),
    ),
)
def test_scenario_document_rejects_ambiguous_top_level_types_and_identity(
    value: object,
    expected_name: str,
    message: str,
) -> None:
    """Reject documents that cannot identify one readable SSP2 scenario."""

    with pytest.raises(ScenarioError, match=message):
        _validate_scenario_payload(value, expected_name=expected_name)


@pytest.mark.parametrize(
    ("entry", "error_type", "message"),
    (
        (
            {
                "parameter": "collection_rate",
                "region": "EUR",
                "material": "PE",
                "changes": [{"year": 2030, "mode": "absolute", "value": 0.5}],
            },
            ScenarioError,
            "does not accept coordinate fields",
        ),
        (
            {
                "parameter": "collection_rate",
                "selector": "uniform_all_goods",
                "changes": [{"year": 2030, "mode": "absolute", "value": 0.5}],
            },
            ScenarioError,
            "Selector .* is not valid",
        ),
        (
            {
                "parameter": "stock_projection",
                "selector": "uniform_all_h12_goods",
                "changes": [{"year": 2030, "mode": "absolute", "value": 2.0}],
            },
            ScenarioError,
            "relative-mode only",
        ),
        (
            {
                "parameter": "collection_rate",
                "selector": "uniform_all_h12",
                "region": "EUR",
                "changes": [{"year": 2030, "mode": "absolute", "value": 0.5}],
            },
            ScenarioError,
            "cannot be combined",
        ),
        (
            {
                "parameter": "collection_rate",
                "region": "EUR",
                "regions": ["USA"],
                "changes": [{"year": 2030, "mode": "absolute", "value": 0.5}],
            },
            ScenarioConflictError,
            "either region or regions",
        ),
        (
            {
                "parameter": "collection_rate",
                "regions": [],
                "changes": [{"year": 2030, "mode": "absolute", "value": 0.5}],
            },
            ScenarioError,
            "regions must be a non-empty list",
        ),
        (
            {
                "parameter": "collection_rate",
                "region": "Atlantis",
                "changes": [{"year": 2030, "mode": "absolute", "value": 0.5}],
            },
            ScenarioError,
            "unknown region",
        ),
        (
            {
                "parameter": "collection_rate",
                "regions": ["EUR", "EUR"],
                "changes": [{"year": 2030, "mode": "absolute", "value": 0.5}],
            },
            ScenarioConflictError,
            "selector contains duplicates",
        ),
    ),
)
def test_parameter_selectors_preserve_explicit_physical_coordinates(
    entry: Mapping[str, object],
    error_type: type[ScenarioError],
    message: str,
) -> None:
    """Reject implicit broadcasts, conflicting selectors, and unknown coordinates."""

    with pytest.raises(error_type, match=message):
        _compile([dict(entry)])


@pytest.mark.parametrize(
    ("entry", "message"),
    (
        (
            {
                "parameter": "collection_rate",
                "region": "EUR",
                "changes": ["2030: 0.5"],
            },
            "change point must be a mapping",
        ),
        (
            {
                "parameter": "collection_rate",
                "region": "EUR",
                "changes": [{"year": 2030, "mode": "absolute"}],
            },
            "requires a finite numeric value",
        ),
        (
            {
                "parameter": "collection_rate",
                "region": "EUR",
                "changes": [{"year": 2030, "mode": "absolute", "value": float("inf")}],
            },
            "requires a finite numeric value",
        ),
    ),
)
def test_change_points_require_finite_explicit_numeric_values(
    entry: Mapping[str, object],
    message: str,
) -> None:
    """Reject scientific schedules with unparseable or nonfinite annual values."""

    with pytest.raises(ScenarioError, match=message):
        _compile([dict(entry)])


@pytest.mark.parametrize(
    ("changes", "error_type", "message"),
    (
        (["not a mapping"], ScenarioError, "change entry must be a mapping"),
        (
            [
                {
                    "parameter_group": "unknown_share_group",
                    "selector": "uniform_all_h12",
                    "trajectory": {},
                }
            ],
            ScenarioError,
            "Unknown foreground parameter group",
        ),
        (
            [
                {
                    "parameter": "collection_rate",
                    "parameter_group": "virgin_feedstock_shares",
                    "trajectory": {},
                }
            ],
            ScenarioError,
            "exactly one of 'parameter' or 'parameter_group'",
        ),
        (
            [{"region": "EUR", "changes": []}],
            ScenarioError,
            "exactly one of 'parameter' or 'parameter_group'",
        ),
        (
            [{"parameter": "collection_rate", "region": "EUR"}],
            ScenarioError,
            "exactly one of 'changes' or 'trajectory'",
        ),
        (
            [
                {
                    "parameter": "collection_rate",
                    "region": "EUR",
                    "changes": [],
                    "trajectory": {},
                }
            ],
            ScenarioError,
            "exactly one of 'changes' or 'trajectory'",
        ),
    ),
)
def test_scenario_entries_select_one_parameter_and_one_schedule_form(
    changes: Sequence[object],
    error_type: type[ScenarioError],
    message: str,
) -> None:
    """Require each entry to have one unambiguous owner and schedule form."""

    with pytest.raises(error_type, match=message):
        _compile(list(changes))


def _group_entry(trajectory: object, **extra: object) -> dict[str, object]:
    """Return one treatment-share group trajectory authoring entry."""

    return {
        "parameter_group": "collected_waste_treatment_shares",
        "selector": "uniform_all_h12",
        "trajectory": trajectory,
        **extra,
    }


@pytest.mark.parametrize(
    ("entry", "message"),
    (
        (
            _group_entry(
                {"mode": "linear_from_2019", "target_year": 2050, "target_values": {}},
                region="EUR",
            ),
            "Grouped trajectories accept only",
        ),
        (
            {
                **_group_entry({"mode": "linear_from_2019"}),
                "selector": "uniform_all_goods",
            },
            "require selector 'uniform_all_h12'",
        ),
        (_group_entry([]), "trajectory must be a mapping"),
        (
            _group_entry(
                {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_values": {},
                    "unexpected": True,
                }
            ),
            "Unsupported trajectory fields",
        ),
        (
            _group_entry({"mode": "step", "target_year": 2050, "target_values": {}}),
            "trajectory.mode must be 'linear_from_2019'",
        ),
        (
            _group_entry(
                {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_mode": "absolute",
                    "target_values": {},
                }
            ),
            "require explicit target_values",
        ),
        (
            _group_entry(
                {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_values": [],
                }
            ),
            "require target_values",
        ),
    ),
)
def test_grouped_trajectories_reject_ambiguous_share_schedules(
    entry: Mapping[str, object],
    message: str,
) -> None:
    """Require a single all-region trajectory with explicit named shares."""

    with pytest.raises(ScenarioError, match=message):
        _compile([dict(entry)])


@pytest.mark.parametrize(
    ("entry", "message"),
    (
        (
            {"parameter": "missing_parameter", "region": "EUR", "changes": []},
            "Unknown foreground parameter",
        ),
        (
            {"parameter": "reuse_rate", "region": "EUR", "changes": []},
            "is not editable",
        ),
        (
            {"parameter": "collection_rate", "region": "EUR", "changes": []},
            "requires change points",
        ),
        (
            {
                "parameter": "waste_imports",
                "region": "EUR",
                "trajectory": {
                    "mode": "linear_from_2019",
                    "target_year": 2050,
                    "target_mode": "absolute",
                    "target_value": 1.0,
                },
            },
            "does not support trajectories",
        ),
    ),
)
def test_parameter_schedules_reject_unknown_unsupported_or_empty_controls(
    entry: Mapping[str, object],
    message: str,
) -> None:
    """Permit prospective schedules only for editable parameters and supported forms."""

    with pytest.raises(ScenarioError, match=message):
        _compile([dict(entry)])


@pytest.mark.parametrize(
    ("trajectory", "message"),
    (
        ([], "trajectory must be a mapping"),
        (
            {
                "mode": "linear_from_2019",
                "target_year": 2050,
                "target_mode": "absolute",
                "target_value": 0.8,
                "unexpected": True,
            },
            "Unsupported trajectory fields",
        ),
        (
            {
                "mode": "step",
                "target_year": 2050,
                "target_mode": "absolute",
                "target_value": 0.8,
            },
            "trajectory.mode must be 'linear_from_2019'",
        ),
        (
            {
                "mode": "linear_from_2019",
                "target_year": 2050,
                "target_values": {"collection_rate": 0.8},
            },
            "Scalar trajectories require target_value",
        ),
        (
            {
                "mode": "linear_from_2019",
                "target_year": 2050,
                "target_mode": "retain_default",
                "target_value": 0.8,
            },
            "trajectory.target_mode .* is not valid",
        ),
        (
            {
                "mode": "linear_from_2019",
                "target_year": 2050,
                "target_mode": "absolute",
                "target_value": "not numeric",
            },
            "trajectory.target_value must be a finite numeric value",
        ),
        (
            {
                "mode": "linear_from_2019",
                "target_year": 2050,
                "target_mode": "absolute",
                "target_value": float("nan"),
            },
            "trajectory.target_value must be a finite numeric value",
        ),
    ),
)
def test_scalar_trajectories_reject_ambiguous_or_nonfinite_targets(
    trajectory: object,
    message: str,
) -> None:
    """Require one finite scalar target with an accepted trajectory mode."""

    entry = {
        "parameter": "collection_rate",
        "region": "EUR",
        "trajectory": trajectory,
    }
    with pytest.raises(ScenarioError, match=message):
        _compile([entry])


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    (
        ("collection_rate", -0.01, "must be at least 0"),
        ("lifetime_mean", 0.0, "must be greater than 0"),
        ("collection_rate", 1.01, "must be at most 1"),
    ),
)
def test_compiled_scientific_values_respect_inclusive_and_exclusive_bounds(
    parameter: str,
    value: float,
    message: str,
) -> None:
    """Enforce the catalog bounds after annual scenario rules are resolved."""

    coordinate = {"good": "Packaging"} if parameter == "lifetime_mean" else {"region": "EUR"}
    entry = {
        "parameter": parameter,
        **coordinate,
        "changes": [{"year": 2030, "mode": "absolute", "value": value}],
    }
    with pytest.raises(ScenarioFeasibilityError, match=message):
        _compile([entry])


@pytest.mark.parametrize("rounded_partition", (False, True))
def test_material_share_partition_preserves_admitted_rounding(rounded_partition: bool) -> None:
    """Reject deficient partitions and preserve valid rounded shares."""

    definition = load_parameter_catalogue("plastic_family_share")
    assert not isinstance(definition, tuple)
    materials = definition.coordinates["material"]
    values = (
        {material: {"HDPE": 1.0, "PP": 5e-13}.get(material, 0.0) for material in materials}
        if rounded_partition
        else dict.fromkeys(materials, 0.9 / len(materials))
    )
    changes = [
        {
            "parameter": "plastic_family_share",
            "region": "EUR",
            "material": material,
            "good": "Packaging",
            "changes": [
                {
                    "year": 2030,
                    "mode": "absolute",
                    "value": values[material],
                }
            ],
        }
        for material in materials
    ]

    if rounded_partition:
        evaluation, _ = _compile_payload(
            _validate_scenario_payload(_payload(changes), expected_name="validation_case"),
            source_path=Path("validation_case.yaml"),
        )
        table = evaluation.compiled_parameters
        selected = table.loc[
            table["parameter_id"].eq("plastic_family_share")
            & table["region"].eq("EUR")
            & table["good"].eq("Packaging")
            & table["year"].eq(2030)
        ].set_index("material")["resolved_value"]
        assert selected.to_dict() == values
        assert selected.sum() > 1.0
    else:
        with pytest.raises(ScenarioFeasibilityError, match="shares must total 1"):
            _compile(changes)


def test_matching_landfill_assertion_accepts_the_exact_treatment_residual() -> None:
    """Accept a user assertion when landfill equals the active route-share residual."""

    changes = [
        {
            "parameter": parameter,
            "region": "EUR",
            "changes": [{"year": 2030, "mode": "absolute", "value": value}],
        }
        for parameter, value in (
            ("mechanical_recycling_rate", 0.4),
            ("chemical_recycling_rate", 0.1),
            ("incineration_rate", 0.3),
            ("landfill_share", 0.2),
        )
    ]

    evaluation, _manifest = _compile_payload(
        _validate_scenario_payload(_payload(changes), expected_name="validation_case"),
        source_path=Path("validation_case.yaml"),
    )
    active = evaluation.compiled_parameters.loc[
        evaluation.compiled_parameters["year"].ge(2030)
        & evaluation.compiled_parameters["region"].eq("EUR")
        & evaluation.compiled_parameters["parameter_id"].isin(
            {
                "mechanical_recycling_rate",
                "chemical_recycling_rate",
                "incineration_rate",
            }
        )
    ]
    totals = active.groupby("year")["resolved_value"].sum()
    np.testing.assert_allclose(1.0 - totals, 0.2, rtol=0.0, atol=1e-15)
