"""Admit the selected PA washing, pelletizing and complete delivery sources.

Original WEEE water and dry-input calibration determine washing water.
Completed dated operations supply utilities, cooling, equipment and the native
particulate records separately. The operating-function compiler defines the feed omissions and the
selected complete operating function.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum
from typing import cast

from .catalogue import provider_service_code
from .native_recipe import _self_production_amount
from .pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
    PaReceivingOperation,
    PaReceivingRow,
    _activity,
    _activity_from_payload,
    _flow,
    _flow_from_payload,
    _operation_from_payload,
    _receiving_row,
)
from .recipe_context import _RecipeContext

_PREPARATION = "recycling_weee_preparation"
_FINISHING = "recycling_pet_finishing"
_DELIVERY = "recycling_pet_pellet_delivery"
_PM = "Particulate Matter, > 10 um"
_PREPARATION_WATER = "wastewater from recycling of waste PP"


@dataclass(frozen=True, slots=True)
class PaRecyclingReceivingReference:
    """Original WEEE washing water and dry-feed basis per output kg.

    ``dry_feed_kg_per_reference_kg`` is the allocated original dry-feed basis.
    ``wash_water_kg_per_reference_kg`` is the original tap-water purchase
    divided by production. Their ratio gives kg fresh wash water/kg incoming
    dry principal-plus-finish mixture, separately from recovery yield.
    """

    preparation: PaReceivingActivity
    dry_feed_kg_per_reference_kg: float
    wash_water_kg_per_reference_kg: float


@dataclass(frozen=True, slots=True)
class PaRecyclingReceivingSources:
    """Complete dated operations with separately indexed water and dust.

    Operation rows are per positive reference-product kg, with signed native
    production retained as provenance. Preparation utility throughput is the
    incoming dry mixture divided by the original dry-feed basis. Dust rows receive the actual
    postwash airborne principal and finish quantities from the material calculation.
    Finishing dust is likewise separate from recovered-pellet operating duties.

    The preparation wastewater indices identify only the overlapping donor
    PP-water function to replace with actual finish receiving. Finishing keeps
    its independent cooling water and complete wastewater function. Pellet
    delivery retains one prepared delivery endpoint, including native
    freight and support. Its rows and production basis remain inside that graph.
    """

    preparation: PaReceivingOperation
    finishing: PaReceivingOperation
    preparation_pm_rows: tuple[int, ...]
    finishing_pm_rows: tuple[int, ...]
    preparation_water_rows: tuple[int, ...]
    preparation_wastewater_rows: tuple[int, ...]
    pellet_delivery: PaReceivingActivity
    evaporated_water: PaReceivingFlow


def admit_pa_recycling_receiving_reference(
    context: _RecipeContext,
) -> PaRecyclingReceivingReference:
    """Read original wash water using the admitted dry-feed/production convention."""
    services = cast(list[dict[str, object]], context.catalogue["services"])
    service = next(item for item in services if item["provider_service_id"] == _PREPARATION)
    construction = cast(dict[str, object], service["construction"])
    base = cast(dict[str, object], construction["base"])
    original = context.source.activity(cast(dict[str, object], base["selector"]), year=2025)
    exchanges = context.exchanges(original)
    production = _self_production_amount(original, exchanges, method="PA washing water")
    wash_water = fsum(
        float(row["amount"])
        for row in exchanges
        if row["type"] == "technosphere" and context.input(row)["reference product"] == "tap water"
    )
    basis = next(
        item for item in context.material_recipe.recycling_operations if item.operation == "weee"
    )
    return PaRecyclingReceivingReference(
        _activity(original), basis.dry_feed_kg_per_reference_kg, wash_water / production
    )


def _prepared_operation(context: _RecipeContext, service_id: str) -> PaReceivingOperation:
    """Detach a completed operation using its admitted rows."""
    from bw2data import get_node

    node = get_node(database=context.provider_database, code=provider_service_code(service_id))
    exchanges = tuple(node.exchanges())
    # The completed operation compiler already owns its singular positive
    # production and valid rows. Normalize its output basis once for detached
    # use, preserving duplicate rows and their independent uncertainties.
    production = float(next(row["amount"] for row in exchanges if row["type"] == "production"))
    rows: list[PaReceivingRow] = []
    for native in exchanges:
        if native["type"] == "production":
            continue
        endpoint = context.input(native)
        rows.append(
            _receiving_row(
                _activity(endpoint) if native["type"] == "technosphere" else _flow(endpoint),
                native,
                1 / production,
            )
        )
    return PaReceivingOperation(_activity(node), production, tuple(rows))


def admit_pa_recycling_receiving_sources(context: _RecipeContext) -> PaRecyclingReceivingSources:
    """Admit completed dated operations and indices for actual PA wash replacement."""
    from bw2data import get_node

    preparation = _prepared_operation(context, _PREPARATION)
    finishing = _prepared_operation(context, _FINISHING)
    # Both selected operations emit the same existing coarse-particle identity.
    # Index exact endpoints, leaving any other direct flow with its operation.
    pm = context.biosphere.flow({"name": _PM, "unit": "kilogram", "categories": ["air"]})
    preparation_pm = tuple(
        index for index, row in enumerate(preparation.rows) if row.endpoint.key == pm.key
    )
    finishing_pm = tuple(
        index for index, row in enumerate(finishing.rows) if row.endpoint.key == pm.key
    )
    water = tuple(
        index
        for index, row in enumerate(preparation.rows)
        if isinstance(row.endpoint, PaReceivingActivity)
        and row.endpoint.reference_product == "tap water"
    )
    wastewater = tuple(
        index
        for index, row in enumerate(preparation.rows)
        if isinstance(row.endpoint, PaReceivingActivity)
        and row.endpoint.reference_product == _PREPARATION_WATER
    )
    delivery = get_node(database=context.provider_database, code=provider_service_code(_DELIVERY))
    return PaRecyclingReceivingSources(
        preparation,
        finishing,
        preparation_pm,
        finishing_pm,
        water,
        wastewater,
        _activity(delivery),
        _flow(
            context.biosphere.flow({"name": "Water", "unit": "cubic meter", "categories": ["air"]})
        ),
    )


def recycling_reference_from_payload(
    payload: Mapping[str, object],
) -> PaRecyclingReceivingReference:
    """Restore the trusted original dry-input and wash-water reference."""
    return PaRecyclingReceivingReference(
        _activity_from_payload(cast(Mapping[str, object], payload["preparation"])),
        float(cast(float, payload["dry_feed_kg_per_reference_kg"])),
        float(cast(float, payload["wash_water_kg_per_reference_kg"])),
    )


def recycling_sources_from_payload(payload: Mapping[str, object]) -> PaRecyclingReceivingSources:
    """Restore dated row order, indices, and endpoints."""
    return PaRecyclingReceivingSources(
        _operation_from_payload(cast(Mapping[str, object], payload["preparation"])),
        _operation_from_payload(cast(Mapping[str, object], payload["finishing"])),
        tuple(cast(list[int], payload["preparation_pm_rows"])),
        tuple(cast(list[int], payload["finishing_pm_rows"])),
        tuple(cast(list[int], payload["preparation_water_rows"])),
        tuple(cast(list[int], payload["preparation_wastewater_rows"])),
        _activity_from_payload(cast(Mapping[str, object], payload["pellet_delivery"])),
        _flow_from_payload(cast(Mapping[str, object], payload["evaporated_water"])),
    )
