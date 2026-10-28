"""MFA result tables and compact candidate projections."""

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from .cohorts import ProspectiveCohortResult
from .flows import NOMINAL_ELEMENT_BASIS, NOMINAL_FEEDSTOCK_METHOD, MFAFlowState

_INPUT_COORDINATES = ("region", "material", "good", "process", "route")
_PROCESS_STAGE = {
    "sysenv": "system_environment",
    "new_polymer": "polymer_production",
    "chemical_feedstock": "chemical_process_boundary",
    "chemical_preparation_reject": "chemical_preparation_reject_boundary",
    "manufacturing_loss": "manufacturing_principal_loss_boundary",
    "waterborne_loss": "mechanical_recovery_wastewater_inlet",
    "airborne_loss": "mechanical_recovery_airborne_principal",
    "processing": "processing",
    "fabrication": "fabrication",
    "primary_market": "primary_market",
    "intermediate_market": "intermediate_market",
    "good_market": "entry_into_use",
    "use": "in_use",
    "eol": "end_of_life_generation",
    "collected": "collection_sorting",
    "reclmech": "mechanical_recycling",
    "reclchem": "chemical_recycling",
    "landfill": "landfill",
    "incineration": "incineration",
    "mismanaged": "uncontrolled_or_mismanaged",
    "waste_market": "waste_market",
}


def _cohort_state_table(
    cohort: ProspectiveCohortResult,
    *,
    foreground_scenario: str,
) -> pd.DataFrame:
    """Expand nonzero cohort stocks and outflows by event year, cohort year, region, and good."""
    frames: list[pd.DataFrame] = []
    years = np.asarray(cohort.years, dtype=int)
    regions = np.asarray(cohort.regions, dtype=object)
    goods = np.asarray(cohort.goods, dtype=object)
    first_year = int(years[0])
    for quantity, values in (
        ("stock", cohort.stock_by_cohort),
        ("outflow", cohort.outflow_by_cohort),
    ):
        event_i, cohort_i, region_i, good_i = np.nonzero(values != 0.0)
        cohort_years = years[cohort_i]
        frames.append(
            pd.DataFrame(
                {
                    "foreground_scenario": foreground_scenario,
                    "event_year": years[event_i],
                    "cohort_year": cohort_years,
                    "region": regions[region_i],
                    "good": goods[good_i],
                    "stock_quantity": quantity,
                    "amount": values[event_i, cohort_i, region_i, good_i],
                    "unit": "Mt",
                    "time_interval_years": 1.0,
                    "cohort_origin_status": np.where(
                        cohort_years == first_year,
                        (
                            f"model_boundary_initial_{first_year}_cohort_"
                            f"not_pre{first_year}_production_year"
                        ),
                        "model_inflow_cohort",
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _cohort_material_identity_table(
    cohort: ProspectiveCohortResult,
    *,
    foreground_scenario: str,
) -> pd.DataFrame:
    """Expand material shares for every cohort, region, material, and good."""
    cohort_i, region_i, material_i, good_i = np.indices(cohort.cohort_material_shares.shape)
    years = np.asarray(cohort.years, dtype=int)
    cohort_years = years[cohort_i.ravel()]
    return pd.DataFrame(
        {
            "foreground_scenario": foreground_scenario,
            "cohort_year": cohort_years,
            "region": np.asarray(cohort.regions, dtype=object)[region_i.ravel()],
            "material": np.asarray(cohort.materials, dtype=object)[material_i.ravel()],
            "good": np.asarray(cohort.goods, dtype=object)[good_i.ravel()],
            "material_share": cohort.cohort_material_shares.ravel(),
            "unit": "fraction",
            "material_identity_status": np.where(
                cohort_years <= 2019,
                "frozen_historical_material_identity",
                cohort.material_identity_method,
            ),
        }
    )


def _optimization_cohort_state_table(
    cohort: ProspectiveCohortResult,
) -> pd.DataFrame:
    """Return prospective discard cohorts used to assign carbon origin."""

    years = np.asarray(cohort.years, dtype=int)
    values = cohort.outflow_by_cohort
    event_i, cohort_i, region_i, good_i = np.nonzero(values > 0.0)
    keep = years[event_i] >= 2020
    event_i = event_i[keep]
    cohort_i = cohort_i[keep]
    region_i = region_i[keep]
    good_i = good_i[keep]
    cohort_years = years[cohort_i]
    return pd.DataFrame(
        {
            "event_year": years[event_i],
            "cohort_year": cohort_years,
            "region": np.asarray(cohort.regions, dtype=object)[region_i],
            "good": np.asarray(cohort.goods, dtype=object)[good_i],
            "stock_quantity": "outflow",
            "amount": values[event_i, cohort_i, region_i, good_i],
        }
    )


def _optimization_cohort_material_identity_table(
    cohort: ProspectiveCohortResult,
    cohort_state: pd.DataFrame,
) -> pd.DataFrame:
    """Project material shares only for cohorts that discard prospectively."""

    required = cohort_state.loc[:, ["cohort_year", "region", "good"]].drop_duplicates()
    years = {int(value): index for index, value in enumerate(cohort.years)}
    regions = {str(value): index for index, value in enumerate(cohort.regions)}
    goods = {str(value): index for index, value in enumerate(cohort.goods)}
    cohort_i = np.asarray([years[int(value)] for value in required["cohort_year"]], dtype=int)
    region_i = np.asarray([regions[str(value)] for value in required["region"]], dtype=int)
    good_i = np.asarray([goods[str(value)] for value in required["good"]], dtype=int)
    shares = cohort.cohort_material_shares[cohort_i, region_i, :, good_i]
    row_i, material_i = np.nonzero(shares != 0.0)
    cohort_years = required["cohort_year"].to_numpy(dtype=int)[row_i]
    return pd.DataFrame(
        {
            "cohort_year": cohort_years,
            "region": required["region"].to_numpy(dtype=object)[row_i],
            "material": np.asarray(cohort.materials, dtype=object)[material_i],
            "good": required["good"].to_numpy(dtype=object)[row_i],
            "material_share": shares[row_i, material_i],
        }
    )


def _cohort_composition_table(
    cohort: ProspectiveCohortResult,
) -> pd.DataFrame:
    """List nominal C and Other Elements mass fractions by plastic family."""
    element_i, material_i = np.indices(cohort.carbon_content.shape)
    return pd.DataFrame(
        {
            "element": np.asarray(cohort.elements, dtype=object)[element_i.ravel()],
            "material": np.asarray(cohort.materials, dtype=object)[material_i.ravel()],
            "content_fraction": cohort.carbon_content.ravel(),
            "unit": "fraction",
            "parameter_origin": "remind_mfa_input",
            "computation_origin": "nominal_remind_element_partition",
        }
    )


def _requested_value_lookup(
    change_points: Sequence[Mapping[str, object]],
) -> dict[tuple[object, ...], float | None]:
    """Index scenario change-point values by parameter, year, and coordinates."""
    lookup: dict[tuple[object, ...], float | None] = {}
    for point in change_points:
        if point.get("mode") == "linear_from_2019":
            target_year = point["target_year"]
            if isinstance(point.get("parameter"), str):
                key = (
                    point["parameter"],
                    target_year,
                    *(point.get(coordinate) for coordinate in _INPUT_COORDINATES),
                )
                lookup[key] = float(cast(float, point["target_value"]))
                continue
            raw_targets = point.get("target_values")
            raw_targets = cast(Mapping[str, object], raw_targets)
            for parameter_id, raw_value in raw_targets.items():
                key = (
                    str(parameter_id),
                    target_year,
                    point.get("region"),
                    None,
                    None,
                    None,
                    None,
                )
                lookup[key] = float(cast(float, raw_value))
            continue
        key = (
            point["parameter"],
            point["year"],
            *(point.get(coordinate) for coordinate in _INPUT_COORDINATES),
        )
        raw_value = point.get("value")
        lookup[key] = None if raw_value is None else float(cast(float, raw_value))
    return lookup


def _resolved_input_view(
    compiled: pd.DataFrame,
    change_points: Sequence[Mapping[str, object]],
    waste_trade_propensity_ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Describe resolved annual inputs and their changes from the defaults.

    Deferred waste imports and exports are filled with pre-balancing amounts
    in Mt from the propensity ledger. Explicit scenario values are joined to their
    controlling change points. The returned table adds the requested value,
    the signed difference ``resolved_value - default_value`` in
    ``absolute_difference``, the ratio to a nonzero default, and an exact-default
    or scenario-value status to every compiled parameter row. Derived lifetime
    spread has no independently requested value.
    """
    result = compiled.copy()
    if not waste_trade_propensity_ledger.empty:
        for parameter_id, amount_column in (
            ("waste_imports", "source_input_waste_imports_mt"),
            ("waste_exports", "source_input_waste_exports_mt"),
        ):
            mask = result["parameter_id"].eq(parameter_id) & result["value_mode"].isin(
                {
                    "retain_2019_propensity_pretrade_collected",
                    "retain_default_propensity_pretrade_collected",
                }
            )
            trade_lookup = waste_trade_propensity_ledger.set_index(["year", "region"])[
                amount_column
            ]
            coordinates = pd.MultiIndex.from_frame(result.loc[mask, ["year", "region"]])
            result.loc[mask, "resolved_value"] = trade_lookup.reindex(coordinates).to_numpy()
    lookup = _requested_value_lookup(change_points)
    requested: list[float] = []
    for row in result.itertuples(index=False):
        if row.value_origin == "remind_mfa_default" or row.parameter_role == "derived_input":
            requested.append(np.nan)
            continue
        key = cast(
            tuple[object, ...],
            (
                row.parameter_id,
                row.controlling_change_point_year,
                *(getattr(row, coordinate) for coordinate in _INPUT_COORDINATES),
            ),
        )
        value = lookup[key]
        requested.append(np.nan if value is None else value)
    result["requested_value"] = requested
    result["absolute_difference"] = result["resolved_value"] - result["default_value"]
    result["ratio_to_default"] = np.where(
        result["default_value"].ne(0.0),
        result["resolved_value"] / result["default_value"],
        np.nan,
    )
    result["input_status"] = np.where(
        result["absolute_difference"].eq(0.0),
        "exact_default",
        "scenario_value",
    )
    return result


def _group_flow(
    flows: pd.DataFrame,
    names: tuple[str, ...],
    dimensions: list[str],
) -> pd.DataFrame:
    """Sum selected flows within each combination of the requested dimensions."""
    selected = flows.loc[flows["source_flow_name"].isin(names)]
    return selected.groupby(dimensions, observed=True, sort=True)["amount"].sum().reset_index()


def _derived_outputs(
    cohort: ProspectiveCohortResult,
    accepted_flows: pd.DataFrame,
) -> pd.DataFrame:
    """Build stock, flow, production, collection, and landfill-share output rows."""
    years = np.asarray(cohort.years, dtype=int)
    regions = np.asarray(cohort.regions, dtype=object)
    materials = np.asarray(cohort.materials, dtype=object)
    goods = np.asarray(cohort.goods, dtype=object)
    frames: list[pd.DataFrame] = []
    for parameter_id, values in (
        ("plastics_in_use_stock", cohort.material_stock),
        ("stock_inflow", cohort.material_inflow),
        ("stock_outflow", cohort.material_outflow),
    ):
        year_i, region_i, material_i, good_i = np.indices(values.shape)
        frames.append(
            pd.DataFrame(
                {
                    "parameter_id": parameter_id,
                    "year": years[year_i.ravel()],
                    "region": regions[region_i.ravel()],
                    "material": materials[material_i.ravel()],
                    "good": goods[good_i.ravel()],
                    "process": None,
                    "route": None,
                    "amount": values.ravel(),
                    "unit": "Mt",
                    "parameter_role": "derived_output",
                    "derivation_status": "computed",
                }
            )
        )

    flow_specs = (
        ("discard", ("use => eol",), ["year", "region", "material", "good"]),
        (
            "collected_mass",
            ("eol => collected",),
            ["year", "region", "material", "good"],
        ),
        (
            "mechanically_recycled_resin",
            ("reclmech => processing",),
            ["year", "region", "material"],
        ),
        (
            "new_polymer_production",
            ("new_polymer => processing", "new_polymer => primary_market"),
            ["year", "region", "material"],
        ),
    )
    for parameter_id, names, dimensions in flow_specs:
        table = _group_flow(accepted_flows, names, dimensions)
        table["parameter_id"] = parameter_id
        for coordinate in ("region", "material", "good", "process", "route"):
            if coordinate not in table:
                table[coordinate] = None
        table["unit"] = "Mt"
        table["parameter_role"] = "derived_output"
        table["derivation_status"] = "computed"
        frames.append(
            table[
                [
                    "parameter_id",
                    "year",
                    "region",
                    "material",
                    "good",
                    "process",
                    "route",
                    "amount",
                    "unit",
                    "parameter_role",
                    "derivation_status",
                ]
            ]
        )

    treatment = _group_flow(
        accepted_flows,
        (
            "collected => reclmech",
            "collected => reclchem",
            "collected => incineration",
            "collected => landfill",
        ),
        ["year", "region", "source_flow_name"],
    )
    pivot = treatment.pivot(
        index=["year", "region"],
        columns="source_flow_name",
        values="amount",
    )
    total = pivot.sum(axis=1)
    landfill = pivot["collected => landfill"]
    share = landfill / total.where(total.ne(0.0))
    frame = share.rename("amount").reset_index()
    frame["parameter_id"] = "landfill_share"
    frame["material"] = None
    frame["good"] = None
    frame["process"] = None
    frame["route"] = "landfill"
    frame["unit"] = "fraction"
    frame["parameter_role"] = "derived_output"
    frame["derivation_status"] = np.where(
        total.to_numpy() == 0.0,
        "undefined_zero_total_treatment",
        "computed_residual_share",
    )
    frames.append(
        frame[
            [
                "parameter_id",
                "year",
                "region",
                "material",
                "good",
                "process",
                "route",
                "amount",
                "unit",
                "parameter_role",
                "derivation_status",
            ]
        ]
    )
    output = pd.concat(frames, ignore_index=True)
    return output.loc[output["year"].between(2019, 2100)].reset_index(drop=True)


def mfa_flow_table(
    state: MFAFlowState,
    cohort: ProspectiveCohortResult,
    *,
    foreground_scenario: str,
    describe_flows: bool = True,
) -> pd.DataFrame:
    """Project polymer mass to nominal elements, with optional ordinary report labels.

    Candidate foreground inputs retain coordinates, quantities and units;
    ordinary output additionally explains processes, stage and element basis.
    """
    frames: list[pd.DataFrame] = []
    for name, values in state.flows.items():
        source, target = name.split(" => ")
        axes = np.indices(values.shape)
        base = {
            "foreground_scenario": foreground_scenario,
            "year": np.asarray(state.years)[axes[0].ravel()],
            "region": np.asarray(state.regions, dtype=object)[axes[1].ravel()],
            "material": np.asarray(state.materials, dtype=object)[axes[2].ravel()],
            "good": np.asarray(state.goods, dtype=object)[axes[3].ravel()]
            if values.ndim == 4
            else "all",
            "source_flow_name": name,
            "unit": "Mt",
            "custody": "h12_physical_custody",
        }
        if describe_flows:
            base.update(
                source_process=source,
                target_process=target,
                life_cycle_stage=_PROCESS_STAGE[target],
                basis=NOMINAL_ELEMENT_BASIS,
            )
        for element_index, element in enumerate(cohort.elements):
            fractions = cohort.carbon_content[element_index, axes[2].ravel()]
            frames.append(
                pd.DataFrame(
                    {
                        **base,
                        "element": element,
                        "amount": values.ravel() * fractions,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def nominal_feedstock_table(state: MFAFlowState) -> pd.DataFrame:
    """Report nominal virgin-feed polymer equivalents separately from flows."""
    frames: list[pd.DataFrame] = []
    for origin, values in state.nominal_feedstock_equivalents.items():
        time, region, material = np.indices(values.shape)
        frames.append(
            pd.DataFrame(
                {
                    "year": np.asarray(state.years)[time.ravel()],
                    "region": np.asarray(state.regions, dtype=object)[region.ravel()],
                    "material": np.asarray(state.materials, dtype=object)[material.ravel()],
                    "feedstock": origin,
                    "amount": values.ravel(),
                    "unit": "Mt polymer equivalent",
                    "basis": NOMINAL_FEEDSTOCK_METHOD,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
