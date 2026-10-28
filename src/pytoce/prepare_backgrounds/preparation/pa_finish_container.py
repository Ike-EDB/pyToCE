"""Supply the complete HDPE container selected for delivered PA finish."""

from typing import cast

from .providers.catalogue import provider_service_code
from .providers.pa_finish_material import (
    PA_FINISH_CONTAINER_CAPACITY_L,
    PA_FINISH_CONTAINER_PARTS_KG,
    PA_FINISH_CONTAINER_TARE_KG,
)
from .providers.recipe_context import _RecipeContext


def pa_finish_container(context: _RecipeContext) -> dict[str, object]:
    """Manufacture one fitted HDPE drum and supply its initial empty delivery.

    RIPA/EY 2015 supplies the nominal 55-US-gallon body and closure masses.
    Its explicit revised recipe uses 53% virgin and 47% recycled body resin.
    Complete native resin, forming and finished-seal functions supply this
    packaging proxy; the report's plant electricity and fuel are not added.

    Reported component masses are treated as installed masses, with native
    forming losses supplied once in this nominal whole-container proxy.
    The caller owns the actual 100 kg contents, filling, filled delivery,
    manual product recovery and intact handoff to the next reconditioning cycle.
    """
    service = "pa_finish_container"
    key = (context.provider_database, provider_service_code(service))
    body = PA_FINISH_CONTAINER_PARTS_KG["hdpe_body"]
    closures = PA_FINISH_CONTAINER_PARTS_KG["hdpe_closures"]
    seals = PA_FINISH_CONTAINER_PARTS_KG["technical_seals"]
    conversion = {item.operation: item for item in context.material_recipe.conversion}
    # Main converted polymer is excluded from these native forming services.
    # Their admitted finished/service factors retain native processing losses;
    # the complete recycled-pellet market supplies the body PCR fraction.
    body_feed = body / conversion["conversion_blow"].finished_kg_per_native_service_kg
    closure_feed = closures / conversion["conversion_injection"].finished_kg_per_native_service_kg
    tare = PA_FINISH_CONTAINER_TARE_KG
    # RIPA's 200 miles and 250 empty drums/truck represent this 55-gallon class. The native
    # vehicle's 15.96 t average payload converts the operation to the source's volume allocation.
    distance_km = 200 * 1.609344
    empty_haul = 15.96 * distance_km / 250
    supplies = (
        (
            "virgin_hdpe",
            "market for polyethylene, high density, granulate",
            "polyethylene, high density, granulate",
            "kilogram",
            0.53 * body_feed + closure_feed,
        ),
        (
            "recycled_hdpe",
            "market for polyethylene, high density, pellets, recycled",
            "polyethylene, high density, pellets, recycled",
            "kilogram",
            0.47 * body_feed,
        ),
        ("body_forming", "blow moulding", "blow moulding", "kilogram", body_feed),
        (
            "closure_forming",
            "injection moulding",
            "injection moulding",
            "kilogram",
            closure_feed,
        ),
        (
            "technical_seals",
            "synthetic rubber production",
            "synthetic rubber",
            "kilogram",
            seals,
        ),
        (
            "initial_delivery",
            "transport, freight, lorry, >32 metric ton, diesel, EURO 6",
            "transport, freight, lorry, >32 metric ton, diesel, EURO 6",
            "ton kilometer",
            empty_haul,
        ),
    )
    exchanges: list[dict[str, object]] = [{"input": key, "type": "production", "amount": 1.0}]
    for component, name, product, unit, quantity in supplies:
        activity, _ = context.activity(
            {"name": name, "reference_product": product, "unit": unit, "exact_location": "RER"},
            source_role="native_ecoinvent_provider",
            service_id=service,
            component_id=component,
            semantic_class="complete_container_manufacture",
        )
        exchanges.append(
            {
                "input": cast(tuple[str, str], activity.key),
                "type": "technosphere",
                "amount": quantity,
            }
        )
    return {
        "database": key[0],
        "code": key[1],
        "name": "manufacture and empty delivery of PA finish HDPE container",
        "reference product": "PA finish HDPE container, delivered empty",
        "unit": "unit",
        "location": "RER",
        "type": "process",
        "provider_service_id": service,
        "fitted_tare_kg": tare,
        "installed_parts_kg": dict(PA_FINISH_CONTAINER_PARTS_KG),
        "source_balance_outputs": {
            "container_capacity_l": PA_FINISH_CONTAINER_CAPACITY_L,
            "empty_drum_physical_transport_tkm": tare / 1000 * distance_km,
            "empty_drum_native_transport_tkm": empty_haul,
        },
        "comment": (
            "PA finish packaging uses the RIPA/EY (2015) 55-US-gallon HDPE drum: "
            "21 lb body, 6 oz PE closures and 1 oz gasket, carrying 100 kg of finish. "
            "The body contains 53% virgin and 47% recycled HDPE. Native European "
            "forming inventories include forming losses; recycled-HDPE pellets "
            "and a technical-seal inventory supply the recycled resin and gasket. "
            "Empty delivery uses 200 miles and 250 drums per truck. Initial "
            "manufacture is charged once, followed by intact container reuse "
            "without credit or amortization."
        ),
        "exchanges": exchanges,
    }
