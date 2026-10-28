"""Scale complete recovery operations to actual dry input and pellet output."""

from math import fsum
from typing import cast

from pytoce.download_data.sources.ecoinvent_materials import (
    POLYOLEFIN_PELLET_RETENTION,
)

from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .native_resolution import _matches
from .recipe_context import _copied_native_exchange, _RecipeContext
from .tps_residue_load import _tps_material_load


def _polyethylene_recovery_recipe(
    context: _RecipeContext, construction: dict[str, object]
) -> dict[str, object]:
    """Supply preparation, finishing and freight per kg incoming dry polyethylene.

    The admitted material recipe determines finished pellets ``P``. Flakes sent
    to extrusion are ``F = P / 0.97``; the native allocated flake purchase is
    not another yield. Preparation demand is the reciprocal of native
    dry feed per reference output. Linked native operations retain production
    ``p`` and all operating exchanges, so Brightway applies their ``1/p``
    normalization once. Freight uses actual ``F`` and ``P`` separately.

    Principal feed, product and rejects are owned by the MFA model. They are omitted from the
    operation extracts; utility emissions,
    capital and their native exchange metadata remain with those operations.
    """
    operation = str(construction["operation"])
    basis = next(
        item for item in context.material_recipe.recycling_operations if item.operation == operation
    )
    recovery = next(
        item
        for item in context.material_recipe.mechanical_recovery
        if item.name == f"{operation}_granulate"
    )
    pellets = recovery.recovered_kg_per_input_kg
    quantities = (
        ("preparation", 1.0 / basis.dry_feed_kg_per_reference_kg),
        ("finishing", pellets),
        ("flake_delivery", pellets / POLYOLEFIN_PELLET_RETENTION),
        ("pellet_delivery", pellets),
    )
    return {
        "components": [
            {**cast(dict[str, object], construction[role]), "amount": amount}
            for role, amount in quantities
        ]
    }


def _pellet_recovery_recipe(
    context: _RecipeContext, construction: dict[str, object]
) -> dict[str, object]:
    """Supply complete PP, PS or PVC recovery and delivery per kg dry input.

    These native operations already include extrusion to pellets. Their
    admitted dry feed per reference output determines the operating demand;
    the recovery recipe independently determines delivered pellets.
    Native production normalizes each operating exchange once in Brightway.
    Principal feed and rejects belong to the material system, while retained
    process inputs, emissions, equipment and their metadata remain native.
    """
    operation = str(construction["operation"])
    basis = next(
        item for item in context.material_recipe.recycling_operations if item.operation == operation
    )
    recovery = next(
        item
        for item in context.material_recipe.mechanical_recovery
        if item.name == f"{operation}_granulate"
    )
    return {
        "components": [
            {
                **cast(dict[str, object], construction["preparation"]),
                "amount": 1.0 / basis.dry_feed_kg_per_reference_kg,
            },
            {
                **cast(dict[str, object], construction["pellet_delivery"]),
                "amount": recovery.recovered_kg_per_input_kg,
            },
        ]
    }


def _pet_recovery_recipe(
    context: _RecipeContext, construction: dict[str, object]
) -> dict[str, object]:
    """Supply PET preparation, finishing and both deliveries per dry input kg.

    The material recipe supplies finished pellets ``P``. Its admitted finishing
    ratio ``g`` gives flakes ``F = g*P``. Preparation uses the native allocated
    dry feed per reference output ``b``; its demand is ``1/b``. Native operations
    retain their production, so Brightway performs each reference normalization
    once. Bottle PET and bio-PET use the declared food-grade finisher; fiber PET
    uses generic pelletizing. Their sorting and recovery stay shared.
    """
    bases = {item.operation: item for item in context.material_recipe.recycling_operations}
    recovery = next(
        item for item in context.material_recipe.mechanical_recovery if item.name == "pet_granulate"
    )
    pellets = recovery.recovered_kg_per_input_kg
    quantities = (
        ("preparation", 1.0 / bases["pet"].dry_feed_kg_per_reference_kg),
        ("finishing", pellets),
        ("flake_delivery", pellets * bases["pet_finishing"].dry_feed_kg_per_reference_kg),
        ("pellet_delivery", pellets),
    )
    return {
        "components": [
            {**cast(dict[str, object], construction[role]), "amount": amount}
            for role, amount in quantities
        ]
    }


def _weee_recovery_recipe(
    context: _RecipeContext, construction: dict[str, object]
) -> dict[str, object]:
    """Supply integrated preparation and pelletizing per kg dry target polymer.

    The complete WEEE operating function treats all incoming dry material;
    its allocated dry feed/reference ratio ``b`` gives demand ``1/b``. The
    separately admitted recovery supplies pellet mass ``P`` for
    generic PET finishing and final bulk delivery. The integrated site has
    no intermediate flake consignment. The material calculation supplies the principal feed and
    actual residues.
    """
    basis = next(
        item for item in context.material_recipe.recycling_operations if item.operation == "weee"
    )
    recovery = next(
        item
        for item in context.material_recipe.mechanical_recovery
        if item.name == str(construction["recovery"])
    )
    pellets = recovery.recovered_kg_per_input_kg
    return {
        "components": [
            {**cast(dict[str, object], construction[role]), "amount": amount}
            for role, amount in (
                ("preparation", 1.0 / basis.dry_feed_kg_per_reference_kg),
                ("finishing", pellets),
                ("pellet_delivery", pellets),
            )
        ]
    }


def _phb_recovery_recipe(
    context: _RecipeContext, construction: dict[str, object]
) -> dict[str, object]:
    """Supply complete PHB conditioning, pellet delivery and actual residue haul.

    The material recipe gives recovered pellets ``P`` and the water/air losses
    per kg incoming dry principal. Their remainder ``B`` is hauled once to
    controlled treatment, whose separate demand belongs to material custody.
    The inclusive industrial energy transfer replaces PET electricity; water,
    equipment and dust retain their native operation per recovered kg.
    """
    recovery = next(
        item for item in context.material_recipe.mechanical_recovery if item.name == "phb_granulate"
    )
    pellets = recovery.recovered_kg_per_input_kg
    residue = 1.0 - fsum(
        (pellets, recovery.waterborne_kg_per_input_kg, recovery.airborne_kg_per_input_kg)
    )
    electricity = cast(dict[str, object], construction["electricity"])
    return {
        "components": [
            {**cast(dict[str, object], construction["support"]), "amount": pellets},
            {**electricity, "amount": pellets * float(cast(float, electricity["amount"]))},
            {**cast(dict[str, object], construction["pellet_delivery"]), "amount": pellets},
            {**cast(dict[str, object], construction["residue_haul"]), "amount": residue},
        ]
    }


def _tps_recovery_recipe(
    context: _RecipeContext,
    construction: dict[str, object],
    services: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Condition TPS per kg incoming principal of its historical formulation.

    Principal is glucan plus glycerol; retained moisture and impurities travel
    with it. The virgin formulation supplies water ``w`` and technical mass
    ``k`` per principal kg. With recovered
    fraction ``a``, extrusion electricity and delivered moist pellets use
    ``a*k``; facility throughput uses ``k``. Fresh water ``w`` replaces the
    inferred evaporation ``w``. The admitted managed principal remainder ``b``
    buys ``b*k`` wet residue haulage. Material custody activates its terminal
    treatment separately.
    """
    formulation = services[str(construction["formulation_service_id"])]
    composition = cast(dict[str, object], formulation["representative_method"])
    technical_mass = float(cast(float, composition["gross_technical_output_kg"]))
    water = float(cast(float, composition["co_delivered_water_kg"]))
    recovery = next(
        item for item in context.material_recipe.mechanical_recovery if item.name == "tps_compound"
    )
    retained = recovery.recovered_kg_per_input_kg
    managed = 1.0 - fsum(
        (retained, recovery.waterborne_kg_per_input_kg, recovery.airborne_kg_per_input_kg)
    )
    source = cast(dict[str, object], formulation["construction"])
    electricity = next(
        item
        for item in cast(list[dict[str, object]], source["components"])
        if item.get("component_id") == "extrusion_electricity"
    )
    facility = cast(dict[str, object], construction["facility"])
    return {
        "components": [
            {**electricity, "amount": float(cast(float, electricity["amount"])) * retained},
            {**facility, "amount": float(cast(float, facility["amount"])) * technical_mass},
            {**cast(dict[str, object], construction["fresh_water"]), "amount": water},
            {
                **cast(dict[str, object], construction["pellet_delivery"]),
                "amount": retained * technical_mass,
            },
            {
                **cast(dict[str, object], construction["residue_haul"]),
                "amount": managed * technical_mass,
            },
        ],
        # TPS conditioning replaces the carried water that evaporates. Its
        # biosphere flow is in m3; the adopted water density is 1000 kg/m3.
        "biosphere_terms": [{"selector": construction["evaporated_water"], "amount": water / 1000}],
    }


def _tps_residue_treatment_recipe(
    construction: dict[str, object], services: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Supply landfill treatment per kg actual principal TPS waste.

    The historical formulation supplies wet mass ``Q`` from glucan, glycerol,
    retained reagents, protein and moisture. Its landfill accepts a negative
    waste reference, so one principal kg buys ``-Q`` kg landfill. The MFA
    supplies the actual waste amount separately. Manufacturing and recycling operations own their
    industrial-residue haulage. External collection and inbound transport of
    discarded products remain outside the declared end-of-life boundary.
    """
    material = _tps_material_load(services, str(construction["formulation_service_id"]))
    return {
        "components": [
            {
                **cast(dict[str, object], construction["treatment"]),
                "amount": -material.wet_mass_kg,
            },
        ]
    }


def _residue_haul_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Retain dated waste haulage while replacing the donor disposal mixture.

    The selected GB market has negative reference production ``p``. The haul
    service produces ``-p`` kg with every retained exchange unchanged; Brightway
    supplies ``B*q/(-p)`` transport for actual PHB or wet TPS residue ``B``. Excluding the
    three declared disposal links leaves the dated freight supplier selected
    by premise, including its original uncertainty.

    Raises:
        ValueError: If the external market has no finite negative reference.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    _service_trace(context, service)
    base = cast(dict[str, object], construction["base"])
    activity, row = context.activity(
        cast(dict[str, object], base["selector"]),
        source_role=str(base["source_role"]),
        service_id=service_id,
        component_id="plastic_residue_haul",
        semantic_class="service_normalization",
    )
    exchanges = context.exchanges(activity)
    production = _self_production_amount(activity, exchanges, method="Plastic residue haul")
    if production >= 0:
        raise ValueError(
            f"Plastic residue-haul source {activity.key!r} has reference production "
            f"{production!r}. "
            "A finite negative waste-input reference is required; use the supported "
            "waste-market source in a new workspace."
        )
    omissions = cast(list[dict[str, object]], construction["omit_disposal"])
    retained = []
    removed = []
    for exchange in exchanges:
        if exchange["type"] == "production":
            continue
        if exchange["type"] == "technosphere" and any(
            _matches(context.input(exchange), selector) for selector in omissions
        ):
            removed.append(exchange)
        else:
            retained.append(exchange)
    row["native_production_amount"] = production
    row["retained_technosphere_json"] = context.exchange_json(
        exchange for exchange in retained if exchange["type"] == "technosphere"
    )
    row["removed_technosphere_json"] = context.exchange_json(removed)
    row["scaling_algebra"] = "project haul production=-p; native transport q unchanged; demand=B"
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": [
            {
                "input": (database_name, provider_service_code(service_id)),
                "amount": -production,
                "type": "production",
            },
            *(_copied_native_exchange(exchange) for exchange in retained),
        ],
    }


def _pet_food_grade_finish_recipe(
    context: _RecipeContext,
    construction: dict[str, object],
    services: dict[str, dict[str, object]],
    *,
    service_id: str,
) -> dict[str, object]:
    """Combine APR finishing meters with complete native gas supply and firing.

    The whole-process electricity meter replaces direct electricity in the
    generic finisher and gas furnaces. Fuel-normalized NG and propane operations
    retain their fuel supply, equipment and emissions. LPG uses its actual
    supply and fuel carbon; the propane furnace transfers equipment and other
    combustion emissions on equal fuel energy. Read fuel properties from the
    original source, while dated native fuel demand determines the operation
    scale. Allocated LPG production properties are not fuel composition.
    """
    propane = cast(dict[str, object], construction["propane"])
    definition = cast(dict[str, object], services[str(propane["service_id"])]["construction"])
    base = cast(dict[str, object], definition["base"])
    selector = cast(dict[str, object], base["selector"])
    denominator = cast(dict[str, object], definition["denominator_selector"])
    original = context.source.activity(selector, year=2025)
    source_propane = next(
        exchange
        for exchange in context.exchanges(original)
        if exchange["type"] == "technosphere" and _matches(context.input(exchange), denominator)
    )
    propane_heat = float(source_propane["properties"]["heating value, net"]["amount"])
    activity, _row = context.activity(
        selector,
        source_role=str(base["source_role"]),
        service_id=service_id,
        component_id="lpg_combustion_transfer",
        semantic_class="service_normalization",
    )
    exchanges = context.exchanges(activity)
    production = next(
        float(exchange["amount"]) for exchange in exchanges if exchange["type"] == "production"
    )
    native_propane = fsum(
        float(exchange["amount"])
        for exchange in exchanges
        if exchange["type"] == "technosphere" and _matches(context.input(exchange), denominator)
    )
    lpg = cast(dict[str, object], construction["lpg"])
    lpg_selector = cast(dict[str, object], lpg["selector"])
    original_lpg = context.source.activity(lpg_selector, year=2025)
    source_lpg = next(
        exchange
        for exchange in context.exchanges(original_lpg)
        if exchange["type"] == "technosphere" and exchange["input"] == original_lpg.key
    )
    properties = source_lpg["properties"]
    lpg_heat = float(properties["heating value, net"]["amount"])
    lpg_carbon = float(properties["carbon content, fossil"]["amount"])
    lpg_mass = float(cast(float, lpg["amount"]))
    support = lpg_mass * lpg_heat * production / (native_propane * propane_heat)
    # Bottle-PET finishing burns the APR LPG quantity: 44/12 converts its
    # oxidized fossil carbon from kg C to kg CO2 (nominal molar masses).
    carbon_dioxide = (
        lpg_mass * lpg_carbon * (44.0 / 12.0) * float(cast(float, construction["carbon_oxidation"]))
    )
    # The adopted complete sulfur oxidation uses M(SO2)/M(S) = 64/32 = 2.
    sulfur_dioxide = lpg_mass * float(cast(float, construction["sulfur_mass_fraction"])) * 2.0
    context.semantic_rows.append(
        {
            "background_scenario": context.background_scenario,
            "anchor_year": context.year,
            "provider_service_id": service_id,
            "semantic_class": "service_normalization",
            "ledger_entry_type": "pet_lpg_combustion_transfer",
            "original_propane_database": original.key[0],
            "original_propane_code": original.key[1],
            "propane_net_heat_mj_per_kg": propane_heat,
            "native_propane_kg": native_propane,
            "native_production_mj": production,
            "original_lpg_database": original_lpg.key[0],
            "original_lpg_code": original_lpg.key[1],
            "lpg_net_heat_mj_per_kg": lpg_heat,
            "lpg_fossil_carbon_kg_per_kg": lpg_carbon,
            "lpg_kg": lpg_mass,
            "transferred_operation_mj": support,
            "scaling_algebra": "LPG support=m_LPG*h_LPG*p/(q_propane*h_propane)",
        }
    )
    return {
        "components": [
            *cast(list[dict[str, object]], construction["components"]),
            propane,
            lpg,
            {**cast(dict[str, object], construction["lpg_support"]), "amount": support},
        ],
        "biosphere_terms": [
            {"selector": construction["carbon_dioxide"], "amount": carbon_dioxide},
            {"selector": construction["sulfur_dioxide"], "amount": sulfur_dioxide},
        ],
    }
