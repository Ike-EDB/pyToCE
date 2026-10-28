"""Checks that depend only on foreground scenario inputs."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from .catalogue import ParameterDefinition
from .rules import (
    _COORDINATE_COLUMNS,
    _DEFAULT_DEFERRED_TRADE_MODE,
    _DEFERRED_TRADE_MODE,
    _Rule,
)
from .scenario_contract import MATERIAL_SHARE_SUM_TOLERANCE, ScenarioFeasibilityError


def _validate_bounds(
    compiled: pd.DataFrame,
    definitions: Mapping[str, ParameterDefinition],
) -> None:
    """Check resolved parameter values against their catalog bounds."""
    for parameter_id, rows in compiled.groupby("parameter_id", sort=False):
        definition = definitions[str(parameter_id)]
        resolved_rows = rows.loc[
            ~rows["value_mode"].isin({_DEFERRED_TRADE_MODE, _DEFAULT_DEFERRED_TRADE_MODE})
        ]
        values = resolved_rows["resolved_value"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ScenarioFeasibilityError(f"{parameter_id}: resolved values must all be finite")
        below_minimum = (
            (
                values < definition.minimum
                if definition.minimum_inclusive
                else values <= definition.minimum
            )
            if definition.minimum is not None
            else np.zeros(len(values), dtype=bool)
        )
        if below_minimum.any():
            row = resolved_rows.iloc[int(np.flatnonzero(below_minimum)[0])]
            observed = float(row["resolved_value"])
            bound = cast(float, definition.minimum)
            comparison = "at least" if definition.minimum_inclusive else "greater than"
            raise ScenarioFeasibilityError(
                f"{parameter_id} is {observed:.17g} at {_row_coordinates(row)} but must be "
                f"{comparison} {bound:.17g}. Correct this value in the scenario YAML.",
            )
        above_maximum = (
            values > definition.maximum
            if definition.maximum is not None
            else np.zeros(len(values), dtype=bool)
        )
        if above_maximum.any():
            row = resolved_rows.iloc[int(np.flatnonzero(above_maximum)[0])]
            observed = float(row["resolved_value"])
            bound = cast(float, definition.maximum)
            raise ScenarioFeasibilityError(
                f"{parameter_id} is {observed:.17g} at {_row_coordinates(row)} but must be at "
                f"most {bound:.17g}. Correct this value in the scenario YAML.",
            )


def _row_coordinates(row: pd.Series) -> dict[str, object]:
    """Format one parameter row's coordinates for an error message."""
    return {
        "year": int(row["year"]),
        **{column: row[column] for column in _COORDINATE_COLUMNS if pd.notna(row[column])},
    }


def _validate_feedstock_partition(compiled: pd.DataFrame) -> None:
    """Require biogenic and DACCU fractions to leave a nonnegative fossil share."""
    members = ("bio_production_rate", "daccu_production_rate")
    shares = compiled.loc[compiled["parameter_id"].isin(members)].pivot(
        index=["year", "region"], columns="parameter_id", values="resolved_value"
    )
    total = shares["bio_production_rate"] + shares["daccu_production_rate"]
    invalid = total.gt(1.0)
    if invalid.any():
        year, region = invalid[invalid].index[0]
        observed = float(total.loc[(year, region)])
        raise ScenarioFeasibilityError(
            f"Biogenic and DACCU shares for {region} in {year} total {observed:.17g}; "
            "they must total at most 1. Reduce bio_production_rate or "
            "daccu_production_rate in the scenario YAML."
        )


def _validate_treatment_partition(
    compiled: pd.DataFrame,
    assertion_rules: Sequence[_Rule],
) -> None:
    """Require treatment shares to form the specified yearly partition."""
    members = (
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
    )
    table = (
        compiled.loc[compiled["parameter_id"].isin(members)]
        .pivot(index=["year", "region"], columns="parameter_id", values="resolved_value")
        .sort_index()
    )
    changed = (
        compiled.loc[compiled["parameter_id"].isin(members)]
        .assign(changed=lambda frame: frame["value_origin"].eq("user_foreground_scenario"))
        .groupby(["year", "region"], sort=False)["changed"]
        .any()
        .reindex(table.index, fill_value=False)
    )
    total = table.loc[:, list(members)].sum(axis=1)
    # Check partitions changed by the scenario. run_mfa applies the
    # least-change projection to calculated regional treatment flows.
    invalid = total.gt(1.0 + 1e-12) & changed
    if invalid.any():
        year, region = invalid[invalid].index[0]
        observed = float(total.loc[(year, region)])
        raise ScenarioFeasibilityError(
            f"Treatment shares for {region} in {year} total {observed:.17g}. Mechanical "
            "recycling, chemical recycling, and incineration must total at most 1; reduce "
            "those values in the scenario YAML.",
        )
    landfill = 1.0 - total
    for rule in assertion_rules:
        region = str(rule.coordinate[0])
        later = [
            item
            for item in assertion_rules
            if item.coordinate == rule.coordinate and item.year > rule.year
        ]
        end = min((item.year - 1 for item in later), default=2100)
        expected = landfill.loc[(slice(rule.year, end), region)]
        matches = np.isclose(
            expected.to_numpy(),
            cast(float, rule.value),
            rtol=0.0,
            atol=1e-12,
        )
        if not bool(matches.all()):
            first_mismatch = int(np.flatnonzero(~matches)[0])
            year = int(expected.index[first_mismatch][0])
            residual = float(expected.iloc[first_mismatch])
            raise ScenarioFeasibilityError(
                f"landfill_share is {rule.value} for {region} in {year}, but the other "
                f"treatment shares leave {residual:.17g}. Set landfill_share "
                "to that residual or revise the treatment shares in the scenario YAML."
            )


def _validate_material_partitions(
    compiled: pd.DataFrame,
    rules: Sequence[_Rule],
    definitions: Mapping[str, ParameterDefinition],
) -> None:
    """Require material-specific rule groups to form complete yearly partitions."""
    material_rules = [rule for rule in rules if rule.parameter_id == "plastic_family_share"]
    materials = definitions["plastic_family_share"].coordinates["material"]
    changed_groups: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for rule in material_rules:
        region, material, good = (str(value) for value in rule.coordinate)
        changed_groups[(rule.year, region, good)].add(material)
    for key, selected in changed_groups.items():
        if selected != set(materials):
            missing = sorted(set(materials).difference(selected))
            raise ScenarioFeasibilityError(
                f"The plastic_family_share change for {key} is missing {missing}. Provide all "
                "plastic families at the same change point."
            )
    shares = compiled.loc[compiled["parameter_id"].eq("plastic_family_share")]
    totals = shares.groupby(["year", "region", "good"], sort=False)["resolved_value"].sum()
    matches = np.isclose(totals, 1.0, rtol=0.0, atol=MATERIAL_SHARE_SUM_TOLERANCE)
    if not matches.all():
        key = totals.index[int(np.flatnonzero(~matches)[0])]
        raise ScenarioFeasibilityError(
            f"plastic_family_share totals {float(totals.loc[key]):.17g} for {key}; the shares "
            "must total 1. Correct the values in the scenario YAML."
        )
