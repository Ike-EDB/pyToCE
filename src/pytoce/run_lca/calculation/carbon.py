"""MFA, native treatment and actual PA direct principal-carbon diagnostics."""

import json
from collections.abc import Iterable, Iterator
from math import fsum
from typing import Any, cast

import pandas as pd

from pytoce.prepare_backgrounds.preparation.providers.material_carbon import CARBON_ORIGINS
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingFlow,
)

from .burdens import pa_event_burden
from .pa_projection import PaProjectedEvent, group_pa_receiving_amounts

CARBON_CONSISTENCY_METHOD_IDENTITY = "mfa_native_direct_treatment_carbon_v11"
DIRECT_TREATMENT_CARBON_FLOWS_IDENTITY = "pytoce_direct_treatment_carbon_flows_v5"

TREATMENT_STAGES = {
    "chemical_recycling_preparation",
    "chemical_recycling_conversion",
    "incineration_treatment",
    "mechanical_recycling_reject_incineration_treatment",
    "landfill_treatment",
    "mechanical_recycling_reject_landfill_treatment",
    "mismanaged_open_burning_treatment",
    "mismanaged_open_dump_treatment",
}
_TREATMENT_CARBON_BASIS = {
    stage: (
        "processed_constituent_carbon_after_preparation"
        if stage == "chemical_recycling_conversion"
        else "treatment_gate_constituent_carbon"
    )
    for stage in TREATMENT_STAGES
}
_GROUP_COLUMNS = (
    "year",
    "region",
    "life_cycle_stage",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "good",
    "material_cycle_source",
    "ultimate_carbon_origin",
    "lca_fate",
    "fate_origin",
    "provider_database",
    "provider_code",
)
_CARBON_FRACTIONS = {
    # Carbon atomic mass divided by the molecular mass converts each native
    # gas quantity from kg species to kg carbon, once after origin attribution.
    "carbon dioxide": 12.0 / 44.0,
    "methane": 12.0 / 16.0,
    "carbon monoxide": 12.0 / 28.0,
}
_DIRECT_CARBON_FLOW_COLUMNS = (
    "year",
    "h12_region",
    "life_cycle_stage",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "good",
    "material_cycle_source",
    "mfa_carbon_origin",
    "mfa_carbon_boundary",
    "lca_fate",
    "fate_origin",
    "provider_database",
    "provider_anchor_year",
    "provider_code",
    "owner_index",
    "owner_database",
    "owner_code",
    "owner_name",
    "owner_scale_per_provider_unit",
    "owner_scope",
    "source_role",
    "origin_mapping_status",
    "flow_database",
    "flow_code",
    "flow_name",
    "flow_unit",
    "flow_compartment_json",
    "carbon_species",
    "flow_carbon_origin",
    "principal_carbon_fraction",
    "flow_amount_per_provider_unit",
    "provider_demand_amount",
    "flow_amount",
    "carbon_amount",
    "carbon_unit",
)
_CONSISTENCY_COLUMNS = (
    "year",
    "h12_region",
    "life_cycle_stage",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "good",
    "material_cycle_source",
    "mfa_carbon_origin",
    "mfa_carbon_boundary",
    "lca_fate",
    "fate_origin",
    "provider_database",
    "provider_code",
    "provider_name",
    "provider_reference_product",
    "provider_unit",
    "provider_location",
    "provider_demand_amount",
    "mfa_treatment_input_carbon_kg",
    "native_direct_co2_carbon_kg",
    "native_direct_ch4_carbon_kg",
    "native_direct_co_carbon_kg",
    "native_direct_selected_carbon_kg",
    "selected_native_minus_mfa_carbon_kg",
    "native_direct_carbon_flow_keys_json",
    "comparison_scope",
    "comparison_additivity",
    "carbon_unit",
)


def pa_direct_carbon_flows(projected: Iterable[PaProjectedEvent]) -> pd.DataFrame:
    """Report actual PA principal CO2, CH4 and CO at their activity year.

    Each receiving function has already calculated its complete mixed load,
    including maintenance and water treatment during spinning or recycling.
    Its output-specific principal fractions exclude retained finish and fresh
    treatment reagents from this diagnostic; those auxiliaries remain in the
    full LCA inventory. Biological and atmospheric-DAC principal carbon stay
    separate even when emitted through the same native non-fossil flow.

    ``flow_amount`` is kg principal-attributable gas and ``carbon_amount`` is
    kg C, after one temporal share. ``receiving_flow_amount`` is the complete
    receiving function's kg gas, including auxiliaries, repeated across its
    origin rows. Source records identify the coefficient authorities. Provider
    demand and native-producer columns remain unset for these modeled flows.
    """
    rows: list[dict[str, object]] = []
    for projected_event in projected:
        event = projected_event.event
        burden = pa_event_burden(event)
        for group in group_pa_receiving_amounts(projected_event.receiving.amounts):
            flow = group.endpoint
            if not isinstance(flow, PaReceivingFlow) or flow.unit != "kilogram":
                continue
            species = _carbon_species(flow.name)
            if species is None:
                continue
            for origin, principal in zip(CARBON_ORIGINS, group.principal_amounts, strict=True):
                if principal == 0.0:
                    continue
                # Grouping has already applied each native output's principal
                # attribution. Apply the source-anchor share only here; dump
                # class allocation is already in the activity and load.
                amount = principal * projected_event.temporal_share
                rows.append(
                    {
                        "year": event.year,
                        "h12_region": event.region,
                        "life_cycle_stage": burden.demand_type,
                        "material": "Fibres",
                        "constituent": event.constituent,
                        "product": "polymer",
                        "virgin_feedstock_route": "",
                        "good": "all",
                        "material_cycle_source": burden.material_cycle_source,
                        "mfa_carbon_origin": origin,
                        "mfa_carbon_boundary": "actual_pa_principal_receiving_outputs",
                        "lca_fate": burden.lca_fate,
                        "fate_origin": burden.fate_origin,
                        "owner_scope": "actual_receiving_function",
                        "source_role": "modelled_direct_pa_receiving",
                        "origin_mapping_status": "actual_pa_principal_origin",
                        "flow_database": flow.key[0],
                        "flow_code": flow.key[1],
                        "flow_name": flow.name,
                        "flow_unit": flow.unit,
                        "flow_compartment_json": json.dumps(
                            flow.categories, ensure_ascii=True, separators=(",", ":")
                        ),
                        "carbon_species": species,
                        "flow_carbon_origin": _flow_origin(flow.name),
                        "principal_carbon_fraction": principal / group.amount,
                        "flow_amount": amount,
                        "carbon_amount": amount * _CARBON_FRACTIONS[species],
                        "carbon_unit": "kilogram carbon",
                        "stage_group": burden.stage_group,
                        "physical_event_basis": burden.physical_event_basis,
                        "receiving_function": group.receiving_function,
                        "infiltration_class": projected_event.infiltration_class,
                        "source_anchor_year": projected_event.source_anchor_year,
                        "source_database": projected_event.source_database,
                        "matrix_database": projected_event.matrix_database,
                        "temporal_share": projected_event.temporal_share,
                        "source_records_json": group.source_records_json,
                        "receiving_flow_amount": group.amount * projected_event.temporal_share,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            *_DIRECT_CARBON_FLOW_COLUMNS,
            "stage_group",
            "physical_event_basis",
            "receiving_function",
            "infiltration_class",
            "source_anchor_year",
            "source_database",
            "matrix_database",
            "temporal_share",
            "source_records_json",
            "receiving_flow_amount",
        ],
    )


def _stable_sum(values: pd.Series) -> float:
    """Return a reproducible floating-point sum."""
    return fsum(float(value) for value in values)


def _stable_group_sum(
    frame: pd.DataFrame,
    group_columns: tuple[str, ...],
    value_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate only repeated coordinates while preserving exact ``fsum``.

    Resolved Timex treatment rows are normally already unique at the saved
    carbon grain. Calling a Python aggregation callback for every singleton is
    both unnecessary and disproportionately expensive. Repeated coordinates
    still use the established stable summation rule.
    """

    columns = [*group_columns, *value_columns]
    repeated = frame.duplicated(list(group_columns), keep=False)
    parts = [frame.loc[~repeated, columns]]
    if repeated.any():
        parts.append(
            frame.loc[repeated, columns]
            .groupby(
                list(group_columns),
                observed=True,
                sort=False,
                dropna=False,
            )[list(value_columns)]
            .agg(_stable_sum)
            .reset_index()
        )
    return pd.concat(parts, ignore_index=True).sort_values(
        list(group_columns),
        kind="stable",
        ignore_index=True,
    )


def _provider_metadata(provider: Any) -> dict[str, str]:
    """Return descriptive fields for one provider activity."""
    return {
        "provider_name": str(provider.get("name", "")),
        "provider_reference_product": str(provider.get("reference product", "")),
        "provider_unit": str(provider.get("unit", "")),
        "provider_location": str(provider.get("location", "")),
    }


def _nodes_for_keys(
    bd: Any,
    keys: set[tuple[str, str]],
) -> dict[tuple[str, str], Any]:
    """Read requested activity codes by iterating each owning database once."""

    codes_by_database: dict[str, set[str]] = {}
    for database, code in keys:
        codes_by_database.setdefault(database, set()).add(code)
    resolved: dict[tuple[str, str], Any] = {}
    for database, codes in sorted(codes_by_database.items()):
        for node in bd.Database(database):
            key = cast(tuple[str, str], node.key)
            if key[1] in codes:
                resolved[key] = node
    return resolved


def _carbon_species(name: str) -> str | None:
    """Classify one biosphere flow as carbon dioxide, monoxide, or methane."""
    lowered = name.casefold()
    for species in _CARBON_FRACTIONS:
        if lowered == species or lowered.startswith(species + ","):
            return species
    return None


def _principal_carbon_exchanges(
    owner: Any,
    spec: dict[str, object],
    flow_nodes: dict[tuple[str, str], Any],
    *,
    origin: str | None = None,
) -> Iterator[tuple[Any, str, float, float]]:
    """Yield declared waste-derived gases per provider unit, excluding auxiliaries.

    Fractional native owners use the source-derived share. For an already split
    non-fossil owner, its principal row uses one and its auxiliary row zero.
    Unmixed treatment owners retain their existing all-principal gas boundary.
    TPS rows retain actual origin fractions per exchange because principal
    ingredients and auxiliaries can emit the same native flow at different
    rates. Their repeated rows are combined only after applying those fractions.
    """
    actual_origin = spec["origin_mapping_status"] == "actual_tps_principal_origin"
    origin_index = CARBON_ORIGINS.index(cast(str, origin)) if actual_origin else None
    fractions = (
        {
            (str(row["flow_database"]), str(row["flow_code"])): float(cast(float, row["fraction"]))
            for row in cast(list[dict[str, object]], spec["principal_carbon_fractions"])
        }
        if "principal_carbon_fractions" in spec
        else None
    )
    scale = _owner_scale(spec) / float(owner.rp_exchange()["amount"])
    total_amounts: dict[tuple[str, str], list[float]] = {}
    principal_amounts: dict[tuple[str, str], list[float]] = {}
    species_by_flow: dict[tuple[str, str], str] = {}
    for exchange in owner.exchanges():
        if str(exchange["type"]) != "biosphere":
            continue
        key = cast(tuple[str, str], exchange["input"])
        flow = flow_nodes[key]
        if str(flow.get("unit", "")) != "kilogram":
            continue
        species = _carbon_species(str(flow.get("name", "")))
        if species is None:
            continue
        if origin_index is not None:
            fraction = float(exchange["principal_carbon_origin_fractions"][origin_index])
            amount = scale * float(exchange["amount"])
            total_amounts.setdefault(key, []).append(amount)
            principal_amounts.setdefault(key, []).append(amount * fraction)
            species_by_flow[key] = species
        else:
            fraction = fractions[key] if fractions is not None else 1.0
        if origin_index is None and fraction != 0.0:
            yield flow, species, scale * float(exchange["amount"]) * fraction, fraction
    for key, values in principal_amounts.items():
        amount = fsum(values)
        if amount != 0.0:
            yield flow_nodes[key], species_by_flow[key], amount, amount / fsum(total_amounts[key])


def _direct_carbon_per_provider_unit(
    owner: Any,
    *,
    spec: dict[str, object],
    flow_nodes: dict[tuple[str, str], Any],
    origin: str | None = None,
) -> dict[str, object]:
    """Return principal waste-derived CO2, CH4, and CO carbon per provider unit."""
    amounts: dict[str, list[float]] = {species: [] for species in _CARBON_FRACTIONS}
    flow_keys: set[tuple[str, str]] = set()
    for flow, species, amount, _fraction in _principal_carbon_exchanges(
        owner, spec, flow_nodes, origin=origin
    ):
        amounts[species].append(amount * _CARBON_FRACTIONS[species])
        flow_keys.add((str(flow.key[0]), str(flow.key[1])))
    species_totals = {species: fsum(values) for species, values in amounts.items()}
    return {
        "co2_carbon_kg_per_provider_unit": species_totals["carbon dioxide"],
        "ch4_carbon_kg_per_provider_unit": species_totals["methane"],
        "co_carbon_kg_per_provider_unit": species_totals["carbon monoxide"],
        "native_direct_carbon_flow_keys_json": json.dumps(
            sorted([list(key) for key in flow_keys]),
            ensure_ascii=True,
            separators=(",", ":"),
        ),
    }


def _direct_owner_specs(provider: Any) -> tuple[dict[str, object], ...]:
    """Return the declared direct-carbon owners per provider service unit."""
    return tuple(
        cast(
            list[dict[str, object]],
            json.loads(str(provider["direct_treatment_carbon_owners_json"])),
        )
    )


def _owner_key(spec: dict[str, object]) -> tuple[str, str]:
    """Build the direct-carbon ownership key for one provider exchange."""
    return str(spec["owner_database"]), str(spec["owner_code"])


def _owner_scale(spec: dict[str, object]) -> float:
    """Read the direct treatment activity amount per provider service unit."""
    return float(cast(float | int | str, spec["owner_scale_per_provider_unit"]))


def _carbon_context(
    bd: Any,
    provider_keys: set[tuple[str, str]],
) -> tuple[
    dict[tuple[str, str], Any],
    dict[tuple[str, str], tuple[dict[str, object], ...]],
    dict[tuple[str, str], Any],
    dict[tuple[str, str], Any],
]:
    """Resolve provider, declared direct-carbon owner, and biosphere-flow records."""
    provider_cache = _nodes_for_keys(bd, provider_keys)
    specs_by_provider = {
        key: _direct_owner_specs(provider) for key, provider in provider_cache.items()
    }
    owner_keys = {_owner_key(spec) for specs in specs_by_provider.values() for spec in specs}
    owner_nodes = {
        **provider_cache,
        **_nodes_for_keys(bd, owner_keys - provider_keys),
    }
    flow_keys = {
        cast(tuple[str, str], exchange["input"])
        for owner_key in owner_keys
        for exchange in owner_nodes[owner_key].exchanges()
        if str(exchange["type"]) == "biosphere"
    }
    return (
        provider_cache,
        specs_by_provider,
        owner_nodes,
        _nodes_for_keys(bd, flow_keys),
    )


def _flow_origin(name: str) -> str:
    """Read the native fossil, non_fossil or unspecified label from a flow name."""
    lowered = name.casefold()
    if "non-fossil" in lowered:
        return "non_fossil"
    if "fossil" in lowered:
        return "fossil"
    return "unspecified"


def _diagnostic_demands(
    grouped: pd.DataFrame,
    group_columns: tuple[str, ...],
    specs_by_provider: dict[tuple[str, str], tuple[dict[str, object], ...]],
) -> pd.DataFrame:
    """Restore TPS demand while retaining each input-carbon origin.

    Foreground attribution splits one TPS provider demand by its inlet carbon
    shares. Those shares are unsuitable for species-specific output: glucan
    and glycerol form different methane fractions, and auxiliaries remain
    outside principal carbon. Sum only those input-origin demand rows, keeping
    the historical formulation, world region, event, fate and dated provider.
    Each input-carbon total then compares with its actual output-origin gases.
    Native treatment services retain their existing attributed demand.
    """
    actual_keys = {
        key
        for key, specs in specs_by_provider.items()
        if any(spec["origin_mapping_status"] == "actual_tps_principal_origin" for spec in specs)
    }
    actual = pd.Series(
        [(row.provider_database, row.provider_code) in actual_keys for row in grouped.itertuples()],
        index=grouped.index,
    )
    native = grouped.loc[~actual].assign(diagnostic_origin="")
    if not actual.any():
        return native
    coordinates = tuple(name for name in group_columns if name != "ultimate_carbon_origin")
    tps = grouped.loc[actual]
    physical = _stable_group_sum(tps, coordinates, ("amount",))
    tps = tps.drop(columns="amount").merge(physical, on=list(coordinates), how="left", sort=False)
    tps["diagnostic_origin"] = tps["ultimate_carbon_origin"]
    return pd.concat([native, tps], ignore_index=True).sort_values(
        list(group_columns), kind="stable", ignore_index=True
    )


def direct_treatment_carbon_flows(
    bd: Any,
    provider_activations: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate principal waste-derived treatment gases by MFA event.

    Args:
        bd: Open Brightway project containing the prepared provider and biosphere data.
        provider_activations: Annual provider demands with world region, life cycle stage,
            material carbon origin, provider key, amount, and MFA carbon mass.

    Returns:
        One row per treatment activation, direct carbon activity, and carbon flow. Flow
        amounts use the native biosphere unit after the declared principal-carbon
        fraction; ``carbon_amount`` is expressed in
        kilograms of carbon. An empty frame is returned when no selected treatment
        activation has a declared direct carbon activity.

    Notes:
        Only direct CO2, CH4, and CO exchanges from the activities declared by each
        provider are included. Wastewater/sludge rows exclude fossil flocculant
        carbon, which remains in the LCA alongside upstream supply-chain carbon.
    """

    selected = provider_activations.loc[
        provider_activations["life_cycle_stage"].isin(TREATMENT_STAGES)
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    grouped = _stable_group_sum(
        selected,
        (*_GROUP_COLUMNS, "provider_anchor_year"),
        ("amount", "mfa_carbon_amount_kg"),
    )
    provider_keys = {
        (str(item.provider_database), str(item.provider_code))
        for item in grouped.itertuples(index=False)
    }
    provider_cache, specs_by_provider, owner_nodes, flow_nodes = _carbon_context(
        bd,
        provider_keys,
    )
    grouped = _diagnostic_demands(
        grouped, (*_GROUP_COLUMNS, "provider_anchor_year"), specs_by_provider
    )
    unit_rows: list[dict[str, object]] = []
    join_columns = ["provider_database", "provider_code", "diagnostic_origin"]
    for database, code, origin in (
        grouped[join_columns].drop_duplicates().itertuples(index=False, name=None)
    ):
        provider_key = (str(database), str(code))
        for owner_index, spec in enumerate(specs_by_provider[provider_key]):
            owner_key = _owner_key(spec)
            owner = owner_nodes[owner_key]
            owner_scale = _owner_scale(spec)
            for flow, species, native_amount, fraction in _principal_carbon_exchanges(
                owner, spec, flow_nodes, origin=str(origin) if origin else None
            ):
                flow_key = cast(tuple[str, str], flow.key)
                unit_rows.append(
                    {
                        "provider_database": provider_key[0],
                        "provider_code": provider_key[1],
                        "diagnostic_origin": origin,
                        "owner_index": owner_index,
                        "owner_database": owner_key[0],
                        "owner_code": owner_key[1],
                        "owner_name": str(owner.get("name", "")),
                        "owner_scale_per_provider_unit": owner_scale,
                        "owner_scope": str(spec["owner_scope"]),
                        "source_role": str(spec["source_role"]),
                        "origin_mapping_status": str(spec["origin_mapping_status"]),
                        "flow_database": flow_key[0],
                        "flow_code": flow_key[1],
                        "flow_name": str(flow.get("name", "")),
                        "flow_unit": str(flow.get("unit", "")),
                        "flow_compartment_json": json.dumps(
                            list(flow.get("categories", ())),
                            ensure_ascii=True,
                            separators=(",", ":"),
                        ),
                        "carbon_species": species,
                        "flow_carbon_origin": _flow_origin(str(flow.get("name", ""))),
                        "principal_carbon_fraction": fraction,
                        "flow_amount_per_provider_unit": native_amount,
                        "carbon_fraction": _CARBON_FRACTIONS[species],
                    }
                )
    if not unit_rows:
        return pd.DataFrame(columns=list(_DIRECT_CARBON_FLOW_COLUMNS))
    result = grouped.merge(
        pd.DataFrame(unit_rows),
        on=join_columns,
        how="inner",
        sort=False,
    )
    result["flow_amount"] = result["amount"] * result["flow_amount_per_provider_unit"]
    result["carbon_amount"] = result["flow_amount"] * result.pop("carbon_fraction")
    result["mfa_carbon_boundary"] = result["life_cycle_stage"].map(_TREATMENT_CARBON_BASIS)
    result["carbon_unit"] = "kilogram carbon"
    result = result.rename(
        columns={
            "region": "h12_region",
            "ultimate_carbon_origin": "mfa_carbon_origin",
            "amount": "provider_demand_amount",
        }
    )
    return result[list(_DIRECT_CARBON_FLOW_COLUMNS)]


def carbon_consistency_table(
    bd: Any,
    provider_activations: pd.DataFrame,
) -> pd.DataFrame:
    """Compare MFA treatment-input carbon with principal waste-derived gases.

    Constituent coordinates remain separate even when they share a provider.
    The carbon boundary distinguishes preparation-gate from processed-feed
    carbon. These comparisons track principal material carbon at the stated process steps.
    """

    selected = provider_activations.loc[
        provider_activations["life_cycle_stage"].isin(TREATMENT_STAGES)
    ]
    if selected.empty:
        return pd.DataFrame()
    grouped = _stable_group_sum(
        selected,
        _GROUP_COLUMNS,
        ("amount", "mfa_carbon_amount_kg"),
    )
    provider_keys = {
        (str(item.provider_database), str(item.provider_code))
        for item in grouped.itertuples(index=False)
    }
    provider_cache, specs_by_provider, owner_nodes, flow_nodes = _carbon_context(
        bd,
        provider_keys,
    )
    grouped = _diagnostic_demands(grouped, _GROUP_COLUMNS, specs_by_provider)
    join_columns = ["provider_database", "provider_code", "diagnostic_origin"]
    unit_rows = []
    for database, code, origin in (
        grouped[join_columns].drop_duplicates().itertuples(index=False, name=None)
    ):
        key = (str(database), str(code))
        specs = specs_by_provider[key]
        units = tuple(
            _direct_carbon_per_provider_unit(
                owner_nodes[_owner_key(spec)],
                spec=spec,
                flow_nodes=flow_nodes,
                origin=str(origin) if origin else None,
            )
            for spec in specs
        )
        flow_keys = {
            tuple(value)
            for unit in units
            for value in cast(
                list[list[str]],
                json.loads(str(unit["native_direct_carbon_flow_keys_json"])),
            )
        }
        unit = {
            "co2_carbon_kg_per_provider_unit": fsum(
                cast(float, unit["co2_carbon_kg_per_provider_unit"]) for unit in units
            ),
            "ch4_carbon_kg_per_provider_unit": fsum(
                cast(float, unit["ch4_carbon_kg_per_provider_unit"]) for unit in units
            ),
            "co_carbon_kg_per_provider_unit": fsum(
                cast(float, unit["co_carbon_kg_per_provider_unit"]) for unit in units
            ),
            "native_direct_carbon_flow_keys_json": json.dumps(
                sorted([list(value) for value in flow_keys]),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        }
        unit_rows.append(
            {
                "provider_database": key[0],
                "provider_code": key[1],
                "diagnostic_origin": origin,
                **_provider_metadata(provider_cache[key]),
                **unit,
            }
        )
    result = grouped.merge(
        pd.DataFrame(unit_rows),
        on=join_columns,
        how="inner",
        sort=False,
    )
    result["native_direct_co2_carbon_kg"] = result["amount"] * result.pop(
        "co2_carbon_kg_per_provider_unit"
    )
    result["native_direct_ch4_carbon_kg"] = result["amount"] * result.pop(
        "ch4_carbon_kg_per_provider_unit"
    )
    result["native_direct_co_carbon_kg"] = result["amount"] * result.pop(
        "co_carbon_kg_per_provider_unit"
    )
    result["native_direct_selected_carbon_kg"] = [
        fsum(values)
        for values in zip(
            result["native_direct_co2_carbon_kg"],
            result["native_direct_ch4_carbon_kg"],
            result["native_direct_co_carbon_kg"],
            strict=True,
        )
    ]
    result["selected_native_minus_mfa_carbon_kg"] = (
        result["native_direct_selected_carbon_kg"] - result["mfa_carbon_amount_kg"]
    )
    result["comparison_scope"] = "mfa_treatment_input_vs_native_direct_co2_ch4_co_emissions"
    result["comparison_additivity"] = "non_additive_custody_diagnostic"
    result["mfa_carbon_boundary"] = result["life_cycle_stage"].map(_TREATMENT_CARBON_BASIS)
    result["carbon_unit"] = "kilogram carbon"
    result = result.rename(
        columns={
            "region": "h12_region",
            "ultimate_carbon_origin": "mfa_carbon_origin",
            "amount": "provider_demand_amount",
            "mfa_carbon_amount_kg": "mfa_treatment_input_carbon_kg",
        }
    )
    return result[list(_CONSISTENCY_COLUMNS)]
