"""Prepare optimization demand, lifetime, and circularity baseline values."""

from dataclasses import dataclass
from math import fsum
from typing import cast

import numpy as np
import pandas as pd

from pytoce.shared.runtime.filesystem import canonical_table_sha256, payload_sha256

from ..scenario.catalogue import load_parameter_catalogue
from ..scenario.scenario_contract import (
    ScenarioEvaluation,
    ScenarioFeasibilityError,
)

OPTIMIZATION_CIRCULARITY_PARENT_METHOD_IDENTITY = "pytoce_optimization_circularity_parent_schema_v1"
_OPTIMIZATION_CIRCULARITY_PARENT_IDENTITY_PREFIX = (
    "pytoce_optimization_circularity_parent_schema_v1::"
)
_OPTIMIZATION_CIRCULARITY_PARENT_COLUMNS = (
    "year",
    "region",
    "collection_rate",
    "mechanical_recycling_rate",
    "chemical_recycling_rate",
    "incineration_rate",
    "landfill_share",
    "fossil_share",
    "bio_production_rate",
    "daccu_production_rate",
)
OPTIMIZATION_DEMAND_LIFETIME_PARENT_METHOD_IDENTITY = (
    "pytoce_optimization_demand_lifetime_parent_v1"
)
_OPTIMIZATION_DEMAND_LIFETIME_PARENT_IDENTITY_PREFIX = (
    "pytoce_optimization_demand_lifetime_parent_v1::"
)
_OPTIMIZATION_STOCK_TARGET_IDENTITY_PREFIX = "pytoce_optimization_parent_stock_target_v1::"
_OPTIMIZATION_LIFETIME_IDENTITY_PREFIX = "pytoce_optimization_parent_lifetime_v1::"
_OPTIMIZATION_STOCK_TARGET_COLUMNS = ("year", "region", "good", "stock_projection")
_OPTIMIZATION_LIFETIME_COLUMNS = ("year", "good", "lifetime_mean", "lifetime_std")


@dataclass(frozen=True, slots=True)
class OptimizationDemandLifetimeParent:
    """Actual accepted parent stock targets and compiled lifetime trajectories."""

    identity: str
    method_identity: str
    parent_foreground_scenario_identity: str
    physical_state_identity: str
    stock_target_identity: str
    lifetime_identity: str
    stock_targets: pd.DataFrame
    lifetimes: pd.DataFrame


@dataclass(frozen=True, slots=True)
class OptimizationCircularityParent:
    """Realized regional treatment and the admitted collection and feedstock controls."""

    identity: str
    method_identity: str
    physical_state_identity: str
    foreground_scenario: str
    table: pd.DataFrame


def optimization_demand_lifetime_parent(
    derived_outputs: pd.DataFrame,
    *,
    physical_state_identity: str,
    parent: ScenarioEvaluation,
) -> OptimizationDemandLifetimeParent:
    """Extract demand and lifetime trajectories from admitted MFA tables.

    Stock targets come from the required baseline MFA result. This distinction is essential for
    socioeconomic parents, whose stock target is resolved from population and
    GDP per capita inside the REMIND-MFA adapter. Lifetimes remain direct
    compiled parent controls. The returned tables cover exactly 2020--2100,
    all 12 world regions in the catalog, and all seven end uses.
    """

    stock_rows = derived_outputs.loc[
        derived_outputs["parameter_id"].eq("plastics_in_use_stock")
        & derived_outputs["year"].between(2020, 2100),
        ["year", "region", "material", "good", "amount"],
    ]
    stock_rows = stock_rows.sort_values(
        ["year", "region", "good", "material"],
        kind="stable",
        ignore_index=True,
    )
    stock_targets = (
        stock_rows.groupby(
            ["year", "region", "good"],
            observed=True,
            sort=True,
        )["amount"]
        .agg(lambda values: fsum(float(value) for value in values))
        .rename("stock_projection")
        .reset_index()
        .loc[:, list(_OPTIMIZATION_STOCK_TARGET_COLUMNS)]
        .sort_values(["year", "region", "good"], kind="stable", ignore_index=True)
    )
    lifetime_rows = parent.compiled_parameters.loc[
        parent.compiled_parameters["parameter_id"].isin({"lifetime_mean", "lifetime_std"}),
        ["parameter_id", "year", "good", "resolved_value"],
    ].copy()
    # Use explicit int64 years for the parent table's platform-independent identity.
    lifetime_rows["year"] = lifetime_rows["year"].astype(np.int64)
    lifetimes = (
        lifetime_rows.pivot(
            index=["year", "good"],
            columns="parameter_id",
            values="resolved_value",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    lifetimes = cast(
        pd.DataFrame,
        lifetimes.loc[:, list(_OPTIMIZATION_LIFETIME_COLUMNS)],
    ).sort_values(["year", "good"], kind="stable", ignore_index=True)
    parent_compiled_identity = canonical_table_sha256(parent.compiled_parameters)
    stock_table_identity = canonical_table_sha256(stock_targets)
    lifetime_table_identity = canonical_table_sha256(lifetimes)
    stock_target_identity = _OPTIMIZATION_STOCK_TARGET_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": OPTIMIZATION_DEMAND_LIFETIME_PARENT_METHOD_IDENTITY,
            "parent_foreground_scenario_identity": parent.foreground_scenario_identity,
            "physical_state_identity": physical_state_identity,
            "stock_target_table_identity": stock_table_identity,
        }
    )
    lifetime_identity = _OPTIMIZATION_LIFETIME_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": OPTIMIZATION_DEMAND_LIFETIME_PARENT_METHOD_IDENTITY,
            "parent_foreground_scenario_identity": parent.foreground_scenario_identity,
            "parent_compiled_table_identity": parent_compiled_identity,
            "lifetime_table_identity": lifetime_table_identity,
        }
    )
    identity = _OPTIMIZATION_DEMAND_LIFETIME_PARENT_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": OPTIMIZATION_DEMAND_LIFETIME_PARENT_METHOD_IDENTITY,
            "parent_foreground_scenario_identity": parent.foreground_scenario_identity,
            "physical_state_identity": physical_state_identity,
            "stock_target_identity": stock_target_identity,
            "lifetime_identity": lifetime_identity,
        }
    )
    return OptimizationDemandLifetimeParent(
        identity=identity,
        method_identity=OPTIMIZATION_DEMAND_LIFETIME_PARENT_METHOD_IDENTITY,
        parent_foreground_scenario_identity=parent.foreground_scenario_identity,
        physical_state_identity=physical_state_identity,
        stock_target_identity=stock_target_identity,
        lifetime_identity=lifetime_identity,
        stock_targets=stock_targets,
        lifetimes=lifetimes,
    )


def optimization_circularity_parent(
    model_flows: pd.DataFrame,
    material_realization: pd.DataFrame,
    *,
    physical_state_identity: str,
    parent: ScenarioEvaluation,
) -> OptimizationCircularityParent:
    """Derive regional controls from admitted MFA results and scenario inputs.

    Collection divides collected polymer by discarded polymer. Every treatment
    share uses all collected waste after trade, including constituents that
    cannot enter chemical processing. Virgin-feed biogenic and DAC controls
    come directly from the compiled scenario. Amounts are Mt and returned controls are fractions.

    Exactly zero waste retains the compiled controls. Optimization initialization requires those
    controls to define a simplex.
    """
    columns = list(_OPTIMIZATION_CIRCULARITY_PARENT_COLUMNS[2:])
    controls = (
        "collection_rate",
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
        "bio_production_rate",
        "daccu_production_rate",
    )
    compiled = parent.compiled_parameters
    selected = compiled.loc[
        compiled["year"].between(2020, 2100) & compiled["parameter_id"].isin(controls)
    ]
    table = selected.pivot(
        index=["year", "region"], columns="parameter_id", values="resolved_value"
    )
    regions = load_parameter_catalogue("collection_rate").coordinates["region"]
    grid = pd.MultiIndex.from_product([range(2020, 2101), regions], names=["year", "region"])
    table = table.reindex(grid).loc[:, list(controls)].astype(float)
    flow_totals = (
        model_flows.loc[
            model_flows["year"].between(2020, 2100)
            & model_flows["source_flow_name"].isin(("use => eol", "eol => collected"))
        ]
        .groupby(["year", "region", "source_flow_name"], sort=True)["amount"]
        .agg(fsum)
        .unstack("source_flow_name")
        .reindex(grid)
    )
    discarded = flow_totals["use => eol"].to_numpy(dtype=float)
    table["collection_rate"] = np.divide(
        flow_totals["eol => collected"].to_numpy(dtype=float),
        discarded,
        out=table["collection_rate"].to_numpy(copy=True),
        where=discarded != 0.0,
    )
    treatment_names = (
        "mechanical_treatment",
        "chemical_treatment",
        "incineration_treatment",
        "landfill_treatment",
    )
    material_totals = (
        material_realization.loc[
            material_realization["year"].between(2020, 2100)
            & material_realization["flow_name"].isin(("collected_after_trade", *treatment_names))
        ]
        .groupby(["year", "region", "flow_name"], sort=True)["amount"]
        .agg(fsum)
        .unstack("flow_name")
        .reindex(grid)
    )
    waste = material_totals["collected_after_trade"].to_numpy(dtype=float)
    treatment_columns = columns[1:5]
    table["landfill_share"] = 1.0 - np.array(
        [
            fsum(row)
            for row in table.loc[:, treatment_columns[:3]].itertuples(index=False, name=None)
        ]
    )
    unrepresented = (waste == 0.0) & table["landfill_share"].lt(0.0).to_numpy()
    if np.any(unrepresented):
        first = np.flatnonzero(unrepresented)[0]
        year, region = grid[first]
        raise ScenarioFeasibilityError(
            f"The completed MFA treatment rates cannot be summarized at {year}:{region}: "
            "zero collected waste has no realized treatment partition, and the compiled "
            "mechanical, chemical and incineration controls sum above one. Use a foreground "
            "whose compiled controls and realized regional amounts define the required "
            "optimization inputs; preserve the completed MFA result.",
            code="optimization_parent_not_representable",
            context={"family": "treatment", "coordinate": f"{year}:{region}"},
        )
    table[treatment_columns] = np.divide(
        material_totals.loc[:, list(treatment_names)].to_numpy(dtype=float),
        waste[:, None],
        out=table.loc[:, treatment_columns].to_numpy(copy=True),
        where=waste[:, None] != 0.0,
    )
    table["fossil_share"] = 1.0 - (table["bio_production_rate"] + table["daccu_production_rate"])
    table = table.reset_index().loc[:, list(_OPTIMIZATION_CIRCULARITY_PARENT_COLUMNS)]
    table = table.sort_values(["year", "region"], kind="stable", ignore_index=True)
    table.columns.name = None
    identity = _OPTIMIZATION_CIRCULARITY_PARENT_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": OPTIMIZATION_CIRCULARITY_PARENT_METHOD_IDENTITY,
            "physical_state_identity": physical_state_identity,
            "foreground_scenario_identity": parent.foreground_scenario_identity,
            "table_identity": canonical_table_sha256(table),
        }
    )
    return OptimizationCircularityParent(
        identity=identity,
        method_identity=OPTIMIZATION_CIRCULARITY_PARENT_METHOD_IDENTITY,
        physical_state_identity=physical_state_identity,
        foreground_scenario=parent.foreground_scenario,
        table=table,
    )
