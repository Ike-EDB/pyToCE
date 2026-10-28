"""Blend and deliver the actual PA fiber finish, with complete residue receiving.

The seven ingredients include the FAS supplier's aqueous carrier.
The selected whole mixing function and fitted drum are supplied once. Water
for dilution and drum recovery is added later at the fiber application site.
"""

import json
from dataclasses import asdict
from math import fsum
from typing import cast

from ..chemical_residue_load import ChemicalResidueLoad
from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .pa_finish_material import (
    PA_FINISH_CONTAINER_PARTS_KG,
    PA_FINISH_CONTAINER_TARE_KG,
    _pa_finish_reference,
    _PAFinishReference,
)
from .recipe_context import _RecipeContext, _scaled_native_exchange


def _pa_blending_residue_load(reference: _PAFinishReference) -> ChemicalResidueLoad:
    """Receive actual lost concentrate with its water and separate carbon origins.

    Michel (via Doka) correlation estimates nominal organic CHONS heat. Sodium/phosphorus
    have no added fuel coefficient. Shared receiving owns their actual capture,
    reagents and ash. The donor paint chemistry and waste mixture are absent.
    """
    residue = reference.blending_residue
    elements = residue.elements_kg
    water = reference.blending_residue_water_kg
    # Michel (via Doka WWcalc DT18) gives HHV in MJ/kg element. The study's NCV
    # deducts 2.4417 MJ/kg incoming and formed water (Harvey 1998, NISTIR
    # 5078 Table 1, 2441.7 kJ/kg at 25 C), using nominal 9 kg water/kg H.
    energy = fsum(
        elements[element] * factor
        for element, factor in {
            "C": 34.016,
            "H": 124.265,
            "O": -9.8324,
            "N": 6.276,
            "S": 19.079,
        }.items()
    ) - 2.4417 * (9 * elements["H"] + water)
    return ChemicalResidueLoad(
        wet_mass_kg=fsum(residue.compounds_kg.values()) + water,
        carrier_water_kg=water,
        active_elements_kg=elements,
        net_energy_mj=energy,
        nonfuel_salt_elements_kg={},
        biogenic_carbon_kg=residue.biogenic_carbon_kg,
        # Newly formed water follows the same CIAAW H/O masses as the formulation. Use these
        # formulation masses for the water balance.
        waste_formed_water_kg=elements["H"] * 18.015 / (2 * 1.008),
    )


def _pa_finish_concentrate_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Supply delivered technical concentrate per kg finished principal PA.

    Complete native mixing supports are normalized by dry-output production,
    then multiplied by actual nonaqueous finish. The filled delivery uses
    concentrate and fitted tare; empty delivery is already in
    the complete HDPE container supplier. The emptied whole drum goes to its
    next reconditioning cycle.
    """
    _service_trace(context, service)
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    reference = _pa_finish_reference(str(construction["pa_grade"]))
    key = database_name, provider_service_code(service_id)
    rows: list[dict[str, object]] = [{"input": key, "type": "production", "amount": 1.0}]
    ingredients = cast(dict[str, dict[str, object]], construction["ingredients"])
    for name, amount in reference.ingredient_demands_kg.items():
        rows.append(
            {
                "input": (
                    database_name,
                    provider_service_code(str(ingredients[name]["service_id"])),
                ),
                "type": "technosphere",
                "amount": amount,
            }
        )
    mixing, trace = context.activity(
        cast(dict[str, object], construction["mixing_source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="complete_concentrate_mixing_operation",
        semantic_class="operation_only_extraction",
    )
    native = context.exchanges(mixing)
    production = _self_production_amount(mixing, native, method="PA concentrate mixing")
    retained = []
    for exchange in native:
        if exchange["type"] == "production":
            continue
        node = context.input(exchange)
        keep = (
            str(node["reference product"])
            in {"chemical factory, organics", "electricity, medium voltage", "nitrogen, liquid"}
            if exchange["type"] == "technosphere"
            else str(node["name"]) == "Nitrogen"
        )
        if keep:
            retained.append(exchange)
            # Full alkyd mixing originally supplies 0.07133 kWh and 0.019 kg N2/kg dry output, plus
            # its factory and N2 vent. Preserve dated amounts/uncertainty.
            rows.append(
                _scaled_native_exchange(exchange, reference.dry_concentrate_kg / production)
            )
    trace["retained_technosphere_json"] = context.exchange_json(
        row for row in retained if row["type"] == "technosphere"
    )
    trace["retained_biosphere_json"] = context.exchange_json(
        row for row in retained if row["type"] == "biosphere"
    )
    trace["removed_technosphere_json"] = context.exchange_json(
        row for row in native if row["type"] == "technosphere" and row not in retained
    )
    trace["removed_biosphere_json"] = context.exchange_json(
        row for row in native if row["type"] == "biosphere" and row not in retained
    )
    # The container owner supplies these same stationary installed masses.
    # Its native dated manufacture is independent of this filled-cargo basis.
    tare = PA_FINISH_CONTAINER_TARE_KG * reference.drum_count
    rows.append(
        {
            "input": (
                database_name,
                provider_service_code(
                    str(cast(dict[str, object], construction["container"])["service_id"])
                ),
            ),
            "type": "technosphere",
            "amount": reference.drum_count,
        }
    )
    freight, freight_trace = context.activity(
        cast(dict[str, object], construction["delivery_source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="filled_concentrate_delivery",
        semantic_class="source_backed_portfolio",
    )
    freight_rows = context.exchanges(freight)
    freight_production = _self_production_amount(freight, freight_rows, method="PA finish delivery")
    # Retain the selected lubricant market's three transport duties, excluding
    # all lubricant purchases. Actual filled cargo is concentrate plus tare;
    # dilution water and already supplied empty-package delivery are absent.
    transport = [
        exchange
        for exchange in freight_rows
        if exchange["type"] == "technosphere" and context.input(exchange)["unit"] == "ton kilometer"
    ]
    rows.extend(
        _scaled_native_exchange(exchange, (reference.concentrate_kg + tare) / freight_production)
        for exchange in transport
    )
    freight_trace["retained_technosphere_json"] = context.exchange_json(transport)
    freight_trace["removed_technosphere_json"] = context.exchange_json(
        row for row in freight_rows if row["type"] == "technosphere" and row not in transport
    )
    residue = _pa_blending_residue_load(reference)
    rows.append(
        {
            "input": (
                database_name,
                provider_service_code(
                    str(cast(dict[str, object], construction["residue"])["service_id"])
                ),
            ),
            "type": "technosphere",
            "amount": -residue.wet_mass_kg,
        }
    )
    trace["project_terms_json"] = json.dumps(
        {
            "pa_finish_reference": asdict(reference),
            "delivered_fitted_tare_kg": tare,
            "whole_drum_receiving_reuse_cycle_kg": tare,
            "whole_drum_parts_kg": {
                name: float(amount) * reference.drum_count
                for name, amount in PA_FINISH_CONTAINER_PARTS_KG.items()
            },
        },
        sort_keys=True,
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": rows,
    }
