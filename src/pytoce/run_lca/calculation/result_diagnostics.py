"""Scientific alignment and source diagnostics for completed LCA results."""

from math import fsum
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .foreground import H12_REGIONS
from .timex import PrincipalTimeline

MISMANAGED_STAGES = (
    "mismanaged_open_burning_treatment",
    "mismanaged_open_dump_treatment",
)
MISMANAGED_ACTIVATION_COLUMNS = (
    "year",
    "region",
    "life_cycle_stage",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "good",
    "ultimate_carbon_origin",
    "source_flow_names",
    "source_parent_amount_kg",
    "source_allocation_share",
    "origin_fraction",
    "lca_fate",
    "fate_origin",
    "provider_material",
    "material_component_share",
    "infiltration_class",
    "infiltration_share",
    "route_id",
    "route_component_id",
    "provider_service_id",
    "provider_anchor_year",
    "provider_database",
    "provider_code",
    "provider_class",
    "analogue_status",
    "material_functional_unit",
    "styrene_component_share",
    "butadiene_analogue_component_share",
    "sbr_share_in_material",
    "sbr_proxy_component_role",
    "whole_tyre_constituents_excluded",
    "no_treatment_sensitivity_executed",
    "declared_provider_source_role",
    "provider_source_database",
    "provider_source_role",
    "general_background_database",
    "general_background_anchor_year",
    "general_background_role",
    "selected_background_member",
    "temporal_provider_share",
    "amount",
    "mfa_carbon_amount_kg",
)
_FATE_OUTPUT_COLUMNS = (
    "foreground_scenario",
    "background_scenario",
    "year",
    "h12_region",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "good",
    "ultimate_carbon_origin",
    "source_flow_names",
    "parent_mass_kg",
    "parent_mass_unit",
    "lca_fate",
    "fate_share",
    "fate_origin",
    "fate_share_source_id",
    "fate_share_source_doi",
    "fate_share_dataset_doi",
    "provider_material",
    "material_component_share",
    "infiltration_class",
    "infiltration_share",
    "route_id",
    "route_component_id",
    "provider_service_id",
    "provider_name",
    "provider_reference_product",
    "provider_unit",
    "provider_location",
    "provider_member",
    "provider_anchor_year",
    "provider_database",
    "provider_code",
    "provider_class",
    "analogue_status",
    "material_functional_unit",
    "styrene_component_share",
    "butadiene_analogue_component_share",
    "sbr_share_in_material",
    "sbr_proxy_component_role",
    "whole_tyre_constituents_excluded",
    "no_treatment_sensitivity_executed",
    "declared_provider_source_role",
    "provider_source_database",
    "provider_source_role",
    "general_background_database",
    "general_background_anchor_year",
    "general_background_role",
    "selected_background_member",
    "temporal_provider_share",
    "final_provider_demand_amount_kg",
    "provider_demand_unit",
    "mfa_carbon_amount_kg",
    "closure_residual_kg",
    "provider_resolution_status",
)
_MISMANAGED_CLOSURE_COLUMNS = (
    "foreground_scenario",
    "background_scenario",
    "year",
    "h12_region",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "good",
    "ultimate_carbon_origin",
    "parent_mismanaged_mass_kg",
    "resolved_route_component_count",
    "resolved_provider_count",
    "open_burning_demand_kg",
    "open_dump_demand_kg",
    "allocated_mismanaged_mass_kg",
    "allocation_residual_kg",
    "source_flow_name",
    "unit",
)

_ACTIVATION_LINEAGE_COLUMNS = (
    "foreground_scenario",
    "background_scenario",
    "ssp_scenario",
    "year",
    "h12_region",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "good",
    "custody_status",
    "source_flow_names",
    "material_cycle_source",
    "ultimate_carbon_origin",
    "source_parent_amount_kg",
    "source_allocation_share",
    "origin_fraction",
    "provider_demand_type",
    "life_cycle_stage",
    "stage_group",
    "mapping_basis",
    "lca_boundary",
    "burden_attribution_basis",
    "cutoff_cycle_role",
    "physical_event_basis",
    "physical_event_year",
    "lca_fate",
    "fate_origin",
    "route_id",
    "route_component_id",
    "provider_service_id",
    "provider_role",
    "provider_material",
    "material_component_share",
    "infiltration_class",
    "infiltration_share",
    "provider_class",
    "analogue_status",
    "throughput_multiplier",
    "resolution_status",
    "material_functional_unit",
    "styrene_component_share",
    "butadiene_analogue_component_share",
    "sbr_share_in_material",
    "sbr_proxy_component_role",
    "whole_tyre_constituents_excluded",
    "no_treatment_sensitivity_executed",
    "declared_provider_source_role",
    "provider_member",
    "provider_anchor_year",
    "provider_database",
    "provider_code",
    "provider_source_database",
    "provider_source_role",
    "general_background_database",
    "general_background_anchor_year",
    "general_background_role",
    "selected_background_member",
    "temporal_provider_share",
    "final_provider_demand_amount_kg",
    "provider_demand_unit",
    "mfa_carbon_amount_kg",
    "branch_lineage_id",
)


def spool_activation_lineage(
    timelines: dict[str, PrincipalTimeline],
    target: Any,
    *,
    foreground_scenario: str,
    background_scenario: str,
) -> None:
    """Write provider activation lineage one activity year at a time.

    The 12 regional timelines are combined for each observed year, converted to
    the documented lineage columns, and appended to one Parquet stream in year
    order. The first year fixes the Arrow schema used by every later batch.
    ``target`` remains open and is flushed after the Parquet writer closes.
    """

    first_year, *remaining_years = sorted(
        {
            int(year)
            for region in H12_REGIONS
            for year in timelines[region].activations["year"].unique()
        }
    )
    first_activations = pd.concat(
        [
            timelines[region].activations.loc[timelines[region].activations["year"].eq(first_year)]
            for region in H12_REGIONS
            if timelines[region].activations["year"].eq(first_year).any()
        ],
        ignore_index=True,
    )
    first_lineage = _activation_lineage(
        first_activations,
        foreground_scenario=foreground_scenario,
        background_scenario=background_scenario,
    )
    first_table = pa.Table.from_pandas(first_lineage, preserve_index=False)
    lineage_schema = first_table.schema
    writer = pq.ParquetWriter(target, lineage_schema, compression="zstd")
    try:
        writer.write_table(first_table)
        del first_activations
        del first_lineage
        del first_table
        for year in remaining_years:
            activations = pd.concat(
                [
                    timelines[region].activations.loc[
                        timelines[region].activations["year"].eq(year)
                    ]
                    for region in H12_REGIONS
                    if timelines[region].activations["year"].eq(year).any()
                ],
                ignore_index=True,
            )
            lineage = _activation_lineage(
                activations,
                foreground_scenario=foreground_scenario,
                background_scenario=background_scenario,
            )
            table = pa.Table.from_pandas(
                lineage,
                schema=lineage_schema,
                preserve_index=False,
            )
            writer.write_table(table)
            del activations
            del lineage
            del table
    finally:
        writer.close()
    target.flush()


def restore_activation_lineage(target: Any) -> pd.DataFrame:
    """Read the sorted activation rows saved during the calculation."""

    target.seek(0)
    return pd.read_parquet(target)


def _stable_sum(values: pd.Series) -> float:
    """Return a reproducible floating-point sum."""
    return fsum(float(value) for value in values)


def _provider_metadata(bd: Any, activations: pd.DataFrame) -> pd.DataFrame:
    """Index provider metadata by Brightway database and activity code."""
    rows = []
    for database, code in (
        activations[["provider_database", "provider_code"]]
        .drop_duplicates(ignore_index=True)
        .itertuples(index=False, name=None)
    ):
        activity = bd.get_node(database=str(database), code=str(code))
        rows.append(
            {
                "provider_database": str(database),
                "provider_code": str(code),
                "provider_name": str(activity.get("name", "")),
                "provider_reference_product": str(activity.get("reference product", "")),
                "provider_unit": str(activity.get("unit", "")),
                "provider_location": str(activity.get("location", "")),
            }
        )
    return pd.DataFrame(rows)


def mismanaged_result_tables(
    bd: Any,
    activations: pd.DataFrame,
    *,
    foreground_scenario: str,
    background_scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build open-burning, open-dump, and mass-balance tables.

    Args:
        bd: Active Brightway data interface. Its ``get_node`` method supplies
            the name, reference product, unit, and location of each resolved
            provider activity.
        activations: Annual foreground provider activations. ``amount`` and
            ``source_parent_amount_kg`` are in kilograms.
        foreground_scenario: Readable plastics intervention scenario copied to
            every output row.
        background_scenario: REMIND background member copied to every output row
            and recorded as the provider member.

    Returns:
        A pair of tables. The first retains one resolved open-burning or open-dump
        provider component and anchor per activity year, world region, material,
        constituent, product, virgin feedstock route, good, and ultimate carbon
        origin. The second contains one mass-balance row at those coordinates
        before fate, treatment route, and anchor are split. Both
        tables use kilograms for mass and provider demand. If no mismanaged
        activations exist, both tables are empty but retain their exact schemas.

    Raises:
        KeyError: If ``activations`` omits a column required by the mismanaged
            result schemas.

    Notes:
        These tables cover fixed kilogram-service routes. Actual PA fiber mixtures
        are retained separately in the PA receiving inventory and event tables.
        Parent mass is ``source_parent_amount_kg * origin_fraction``. The fate
        residual is the stable sum of final provider demands minus that parent
        mass. The mass-balance residual is open-burning plus open-dump demand
        minus parent mismanaged mass. This function adds scenario lineage.
    """
    selected = activations.loc[activations["life_cycle_stage"].isin(MISMANAGED_STAGES)].copy()
    if selected.empty:
        return (
            pd.DataFrame(columns=list(_FATE_OUTPUT_COLUMNS)),
            pd.DataFrame(columns=list(_MISMANAGED_CLOSURE_COLUMNS)),
        )
    selected["foreground_scenario"] = foreground_scenario
    selected["background_scenario"] = background_scenario
    selected["h12_region"] = selected.pop("region")
    selected["parent_mass_kg"] = selected["source_parent_amount_kg"] * selected["origin_fraction"]
    selected["fate_share"] = selected["source_allocation_share"]
    selected["final_provider_demand_amount_kg"] = selected.pop("amount")
    selected["parent_mass_unit"] = "kilogram"
    selected["provider_demand_unit"] = "kilogram"
    selected["fate_share_source_id"] = "cottom_2024_nature_633_101_108"
    selected["fate_share_source_doi"] = "10.1038/s41586-024-07758-6"
    selected["fate_share_dataset_doi"] = "10.5061/dryad.8cz8w9gxb"
    selected["provider_member"] = background_scenario
    selected["provider_resolution_status"] = "resolved_member_local"
    selected = selected.merge(
        _provider_metadata(bd, selected),
        on=["provider_database", "provider_code"],
        how="left",
    )
    coordinates = [
        "foreground_scenario",
        "background_scenario",
        "year",
        "h12_region",
        "material",
        "constituent",
        "product",
        "virgin_feedstock_route",
        "good",
        "ultimate_carbon_origin",
    ]
    allocated = selected.groupby(coordinates, observed=True, sort=False)[
        "final_provider_demand_amount_kg"
    ].transform(_stable_sum)
    selected["closure_residual_kg"] = allocated - selected["parent_mass_kg"]
    fate = selected[list(_FATE_OUTPUT_COLUMNS)].sort_values(
        [
            "year",
            "h12_region",
            "material",
            "constituent",
            "product",
            "virgin_feedstock_route",
            "good",
            "ultimate_carbon_origin",
            "lca_fate",
            "route_component_id",
            "provider_anchor_year",
        ],
        kind="stable",
        ignore_index=True,
    )

    fate_amounts = (
        fate.groupby([*coordinates, "lca_fate"], observed=True, sort=False)[
            "final_provider_demand_amount_kg"
        ]
        .agg(_stable_sum)
        .unstack("lca_fate", fill_value=0)
        .rename(
            columns={
                "open_burning": "open_burning_demand_kg",
                "open_dump": "open_dump_demand_kg",
            }
        )
        .reset_index()
    )
    parent = (
        fate.groupby(coordinates, observed=True, sort=False)
        .agg(
            parent_mismanaged_mass_kg=("parent_mass_kg", "first"),
            resolved_route_component_count=("route_component_id", "nunique"),
        )
        .reset_index()
    )
    provider_counts = (
        fate[[*coordinates, "provider_database", "provider_code"]]
        .drop_duplicates(ignore_index=True)
        .groupby(coordinates, observed=True, sort=False)
        .size()
        .rename("resolved_provider_count")
        .reset_index()
    )
    parent = parent.merge(
        provider_counts,
        on=coordinates,
        how="left",
    )
    coverage = parent.merge(
        fate_amounts,
        on=coordinates,
        how="left",
    )
    coverage["allocated_mismanaged_mass_kg"] = (
        coverage["open_burning_demand_kg"] + coverage["open_dump_demand_kg"]
    )
    coverage["allocation_residual_kg"] = (
        coverage["allocated_mismanaged_mass_kg"] - coverage["parent_mismanaged_mass_kg"]
    )
    coverage["source_flow_name"] = "eol => mismanaged"
    coverage["unit"] = "kilogram"
    coverage = coverage.sort_values(
        [
            "year",
            "h12_region",
            "material",
            "constituent",
            "product",
            "virgin_feedstock_route",
            "good",
            "ultimate_carbon_origin",
        ],
        kind="stable",
        ignore_index=True,
    )
    return fate, coverage[list(_MISMANAGED_CLOSURE_COLUMNS)]


def _activation_lineage(
    activations: pd.DataFrame,
    *,
    foreground_scenario: str,
    background_scenario: str,
) -> pd.DataFrame:
    """Return the resolved annual provider demands as a sorted lineage table."""

    result = activations.copy()
    result.insert(0, "foreground_scenario", foreground_scenario)
    result.insert(1, "background_scenario", background_scenario)
    result.insert(2, "ssp_scenario", "SSP2")
    result["h12_region"] = result.pop("region")
    result["provider_member"] = background_scenario
    result["final_provider_demand_amount_kg"] = result.pop("amount")
    result["provider_demand_unit"] = "kilogram"
    return result[list(_ACTIVATION_LINEAGE_COLUMNS)].sort_values(
        [
            "year",
            "h12_region",
            "material",
            "constituent",
            "product",
            "virgin_feedstock_route",
            "good",
            "life_cycle_stage",
            "ultimate_carbon_origin",
            "lca_fate",
            "route_component_id",
            "provider_anchor_year",
            "provider_database",
            "provider_code",
            "branch_lineage_id",
        ],
        kind="stable",
        ignore_index=True,
    )
