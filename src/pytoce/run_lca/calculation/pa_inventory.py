"""Bind actual PA receiving quantities to native products and elementary flows."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import pandas as pd

from ...prepare_backgrounds.preparation.providers.pa_receiving_records import PaReceivingActivity
from .burdens import pa_event_burden
from .contribution_accumulation import _StageCoordinate
from .direct_contributions import DirectContribution
from .pa_projection import PaProjectedEvent, group_pa_receiving_amounts

_NATIVE_ACTIVATION_COLUMNS = (
    "region",
    "year",
    "matrix_database",
    "provider_database",
    "provider_code",
    "provider_activity_id",
    "provider_product_id",
    "provider_anchor_year",
    "amount",
    "stage_database",
    "stage_code",
    "life_cycle_stage",
    "stage_group",
    "cutoff_cycle_role",
    "material_cycle_source",
    "physical_event_basis",
    "physical_event_year",
    "source_flow_names",
    "lca_fate",
    "fate_origin",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "ultimate_carbon_origin",
    "good",
    "receiving_function",
    "source_records_json",
    "endpoint_unit",
)
_RECEIVING_INVENTORY_COLUMNS = (
    "year",
    "h12_region",
    "material",
    "constituent",
    "product",
    "good",
    "life_cycle_stage",
    "stage_group",
    "cutoff_cycle_role",
    "material_cycle_source",
    "physical_event_basis",
    "physical_event_year",
    "source_flow_names",
    "lca_fate",
    "fate_origin",
    "infiltration_class",
    "source_anchor_year",
    "source_database",
    "matrix_database",
    "temporal_share",
    "receiving_function",
    "endpoint_kind",
    "endpoint_database",
    "endpoint_code",
    "endpoint_name",
    "endpoint_unit",
    "endpoint_location",
    "endpoint_categories_json",
    "product_database",
    "product_code",
    "source_records_json",
    "unweighted_amount",
    "amount",
    "principal_fossil_amount",
    "principal_biogenic_amount",
    "principal_atmospheric_dac_amount",
)


@dataclass(frozen=True, slots=True)
class PaReceivingInventory:
    """Native-unit purchases, direct inventory and portable receiving provenance.

    ``activations`` enters the existing native matrix and contribution owners.
    ``direct_inventory`` has region/year/flow_id/amount; ``direct_contributions``
    retains its process for attribution. Both represent the same
    direct quantities, which enter the annual inventory only once. ``records``
    preserves portable endpoint identities and quantities in their native units.
    """

    activations: pd.DataFrame
    direct_inventory: pd.DataFrame
    direct_contributions: tuple[DirectContribution, ...]
    records: pd.DataFrame


def _physical_receiving_fields(value: Any) -> dict[str, Any]:
    """Retain fields and structured emission coordinates in JSON."""
    result = {
        field.name: getattr(value, field.name) for field in fields(value) if field.name != "amounts"
    }
    # These calculated dictionaries use native emission or compartment tuples, which cannot be JSON
    # object keys. Preserve the entire ordered coordinate as an array, with its value unchanged.
    for name in result.keys() & {
        "emissions",
        "native_emission_factors",
        "phosphate_water_kg",
        "biosphere_amounts",
        "principal_origin_fractions",
    }:
        result[name] = [
            {"coordinate": coordinate, "value": amount}
            for coordinate, amount in sorted(result[name].items())
        ]
    return result


def pa_receiving_physical_events(projected: Iterable[PaProjectedEvent]) -> pd.DataFrame:
    """Retain receiving quantities before temporal weighting.

    Event JSON contains the typed input fields: principal kg, finish dose and
    water, with carbon-origin axes. Receiving JSON retains every calculated
    field except ``amounts``; the native inventory table owns supplier and
    elementary quantities. Nested names and units follow the dataclasses.

    Each row is one event, dump class and source anchor. Its payload
    is unweighted, on the class-allocated event basis. Apply ``temporal_share``
    once across source anchors. Parent and child indicators describe the same connected process
    quantities.
    """
    columns = (
        "schema",
        "year",
        "h12_region",
        "constituent",
        "life_cycle_stage",
        "infiltration_class",
        "source_anchor_year",
        "source_database",
        "matrix_database",
        "temporal_share",
        "physical_event_json",
        "physical_receiving_json",
    )
    records = []
    for row in projected:
        records.append(
            {
                "schema": "pytoce_pa_receiving_physical_event_schema_v1",
                "year": row.event.year,
                "h12_region": row.event.region,
                "constituent": row.event.constituent,
                "life_cycle_stage": pa_event_burden(row.event).demand_type,
                "infiltration_class": row.infiltration_class,
                "source_anchor_year": row.source_anchor_year,
                "source_database": row.source_database,
                "matrix_database": row.matrix_database,
                "temporal_share": row.temporal_share,
                "physical_event_json": json.dumps(
                    asdict(row.event), sort_keys=True, separators=(",", ":")
                ),
                "physical_receiving_json": json.dumps(
                    row.receiving,
                    default=_physical_receiving_fields,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def build_pa_receiving_inventory(
    bd: Any,
    projected: Iterable[PaProjectedEvent],
    regional_roots: Mapping[str, tuple[str, str]],
) -> PaReceivingInventory:
    """Bind grouped actual receiving to the existing native inventory calculation.

    Each activity's public reference-product exchange supplies the identifier
    used by the inventory solver. Receiving coefficients retain their native signs
    and endpoint units: this boundary applies only the temporal share. Elementary quantities bypass
    the solver
    and join the same annual inventory before method switching.

    Complete package source records are trusted. Each used native endpoint is
    resolved once; the function assembles the inventory from prepared references. The response bank
    separately enumerates native suppliers and flows needed
    by conditional operations; unused endpoints are not calculated here.
    """
    targets: dict[tuple[str, str], tuple[int, int, str, str]] = {}
    flows: dict[tuple[str, str], int] = {}
    activations: list[dict[str, object]] = []
    direct_inventory: list[tuple[str, int, int, float]] = []
    direct: list[DirectContribution] = []
    records: list[dict[str, object]] = []
    for item in projected:
        event = item.event
        spec = pa_event_burden(event)
        source_flows = " + ".join(spec.source_flow_names)
        stage: _StageCoordinate = (
            event.region,
            event.year,
            spec.demand_type,
            spec.stage_group,
            spec.cutoff_cycle_role,
            spec.material_cycle_source,
            spec.physical_event_basis,
            event.year,
            source_flows,
            spec.lca_fate,
            spec.fate_origin,
        )
        # The burden owner already aggregates goods for PA fiber.
        # These unsplit mixtures have no single virgin route or carbon origin.
        common = {
            "year": event.year,
            "material": "Fibres",
            "constituent": event.constituent,
            "product": "polymer",
            "good": "all",
            "life_cycle_stage": spec.demand_type,
            "stage_group": spec.stage_group,
            "cutoff_cycle_role": spec.cutoff_cycle_role,
            "material_cycle_source": spec.material_cycle_source,
            "physical_event_basis": spec.physical_event_basis,
            "physical_event_year": event.year,
            "source_flow_names": source_flows,
            "lca_fate": spec.lca_fate,
            "fate_origin": spec.fate_origin,
        }
        for row in group_pa_receiving_amounts(item.receiving.amounts):
            endpoint = row.endpoint
            amount = row.amount * item.temporal_share
            product_database = product_code = ""
            if isinstance(endpoint, PaReceivingActivity):
                target = targets.get(endpoint.key)
                if target is None:
                    activity = bd.get_node(database=endpoint.key[0], code=endpoint.key[1])
                    product = activity.rp_exchange().input
                    target = (
                        int(activity.id),
                        int(product.id),
                        str(product.key[0]),
                        str(product.key[1]),
                    )
                    targets[endpoint.key] = target
                activity_id, product_id, product_database, product_code = target
                root = regional_roots[event.region]
                activations.append(
                    {
                        **common,
                        "region": event.region,
                        "matrix_database": item.matrix_database,
                        "provider_database": endpoint.key[0],
                        "provider_code": endpoint.key[1],
                        "provider_activity_id": activity_id,
                        "provider_product_id": product_id,
                        "provider_anchor_year": item.source_anchor_year,
                        "amount": amount,
                        "stage_database": root[0],
                        "stage_code": root[1],
                        "virgin_feedstock_route": "",
                        "ultimate_carbon_origin": "",
                        "receiving_function": row.receiving_function,
                        "source_records_json": row.source_records_json,
                        "endpoint_unit": endpoint.unit,
                    }
                )
                kind, location, categories = "native_supplier", endpoint.location, ()
            else:
                flow_id = flows.get(endpoint.key)
                if flow_id is None:
                    flow_id = int(bd.get_node(database=endpoint.key[0], code=endpoint.key[1]).id)
                    flows[endpoint.key] = flow_id
                direct_inventory.append((event.region, event.year, flow_id, amount))
                direct.append(
                    DirectContribution(
                        stage,
                        row.receiving_function,
                        row.source_records_json,
                        endpoint.unit,
                        flow_id,
                        amount,
                    )
                )
                kind, location, categories = "elementary_flow", "", endpoint.categories
            records.append(
                {
                    **common,
                    "h12_region": event.region,
                    "infiltration_class": item.infiltration_class,
                    "source_anchor_year": item.source_anchor_year,
                    "source_database": item.source_database,
                    "matrix_database": item.matrix_database,
                    "temporal_share": item.temporal_share,
                    "receiving_function": row.receiving_function,
                    "endpoint_kind": kind,
                    "endpoint_database": endpoint.key[0],
                    "endpoint_code": endpoint.key[1],
                    "endpoint_name": endpoint.name,
                    "endpoint_unit": endpoint.unit,
                    "endpoint_location": location,
                    "endpoint_categories_json": json.dumps(categories),
                    "product_database": product_database,
                    "product_code": product_code,
                    "source_records_json": row.source_records_json,
                    "unweighted_amount": row.amount,
                    "amount": amount,
                    "principal_fossil_amount": row.principal_amounts[0] * item.temporal_share,
                    "principal_biogenic_amount": row.principal_amounts[1] * item.temporal_share,
                    "principal_atmospheric_dac_amount": row.principal_amounts[2]
                    * item.temporal_share,
                }
            )
    return PaReceivingInventory(
        pd.DataFrame.from_records(activations, columns=_NATIVE_ACTIVATION_COLUMNS),
        pd.DataFrame.from_records(
            direct_inventory, columns=("region", "year", "flow_id", "amount")
        ),
        tuple(direct),
        pd.DataFrame.from_records(records, columns=_RECEIVING_INVENTORY_COLUMNS),
    )
