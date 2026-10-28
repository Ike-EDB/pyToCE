"""Compile complete Swiss sludge co-incineration with FLUWA and actual ash exits."""

import json
from dataclasses import asdict
from math import fsum
from typing import Any, cast

from .catalogue import provider_service_code
from .leachate_load import _WastewaterLoad
from .material_carbon import CARBON_ORIGINS, PrincipalCarbonOriginFractions
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _copied_native_exchange, _RecipeContext, _scaled_native_exchange
from .swiss_sludge_load import (
    AIR_OXYGEN,
    ELEMENT_NAMES,
    SUPPLIES,
    SURFACE,
    Emission,
    SwissSludgeLoad,
    SwissSludgeSource,
    calculate_swiss_sludge,
)


def swiss_sludge_load(
    context: _RecipeContext,
    construction: dict[str, object],
    wastewater: _WastewaterLoad,
) -> SwissSludgeLoad:
    """Apply original Swiss composition and operating quantities to actual sludge.

    The initial ecoinvent source determines operating quantities. Later premise
    suppliers retain their dated technology while these source equations remain
    fixed. Original dry mass is per wet reference kg; element contents are
    dry-mass fractions. Only exchanges are divided by the native negative
    production magnitude.
    """
    original = context.source.activity(cast(dict[str, object], construction["source"]), year=2025)
    rows = context.exchanges(original)
    magnitude = -_self_production_amount(original, rows, method="Swiss sludge source")
    production = next(ex for ex in rows if ex["type"] == "production")
    props = cast(dict[str, dict[str, object]], production["properties"])
    dry = float(cast(float, props["dry mass"]["amount"]))
    elements = {
        e: dry * float(cast(float, props[f"{name} content"]["amount"]))
        for e, name in ELEMENT_NAMES.items()
        if f"{name} content" in props
    }
    inputs: dict[str, float] = {}
    emissions: dict[Emission, float] = {}
    for ex in rows:
        if ex["type"] == "production":
            continue
        node = context.input(ex)
        amount = float(ex["amount"]) / magnitude
        if ex["type"] == "technosphere":
            product = str(node["reference product"])
            inputs[product] = inputs.get(product, 0.0) + amount
        else:
            key = (str(node["name"]), str(node["unit"]), tuple(node["categories"]))
            emissions[key] = emissions.get(key, 0.0) + amount
    return calculate_swiss_sludge(SwissSludgeSource(elements, inputs, emissions), wastewater)


def _sludge_reference(
    exchange: Any, key: tuple[str, str], product: str, load: SwissSludgeLoad
) -> dict[str, object]:
    """Retain signed native production and disclose actual dry/nominal composition."""
    properties: dict[str, object] = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": load.dry_mass_kg / load.wet_mass_kg, "unit": "kg"},
        "water in wet mass": {"amount": load.water_kg / load.wet_mass_kg, "unit": "kg"},
        "carbon allocation": {"amount": load.elements_kg["C"] / load.wet_mass_kg, "unit": "kg"},
    }
    properties.update(
        {
            f"{ELEMENT_NAMES[e]} content": {"amount": q / load.dry_mass_kg, "unit": "dimensionless"}
            for e, q in load.elements_kg.items()
        }
    )
    for origin in ("fossil", "non-fossil"):
        carbon = fsum(
            x.carbon_kg for x in load.carbon if (x.origin == "fossil") == (origin == "fossil")
        )
        properties[f"carbon content, {origin}"] = {
            "amount": carbon / load.dry_mass_kg,
            "unit": "dimensionless",
        }
    result = _copied_native_exchange(exchange)
    for field in ("flow", "activity", "production volume", "classifications"):
        result.pop(field, None)
    result.update(
        input=key,
        name=product,
        properties=properties,
        comment=(
            "Swiss co-treatment: actual sludge elements; "
            "separately qualified nominal H/O and dry mass."
        ),
    )
    return result


def _principal_gas_shares(load: SwissSludgeLoad, fossil: bool) -> PrincipalCarbonOriginFractions:
    """Partition principal diagnostic carbon within the actual emitted origin group."""
    group = fsum(x.carbon_kg for x in load.carbon if (x.origin == "fossil") == fossil)
    return cast(
        PrincipalCarbonOriginFractions,
        tuple(
            fsum(
                x.carbon_kg
                for x in load.carbon
                if x.principal and x.origin == origin and (x.origin == "fossil") == fossil
            )
            / group
            for origin in CARBON_ORIGINS
        ),
    )


def swiss_sludge_dataset(
    context: _RecipeContext, service: dict[str, object], load: SwissSludgeLoad, database_name: str
) -> dict[str, object]:
    """Bind complete dated operations to one negative wet-kg Swiss treatment service.

    Collection is calculated separately. This furnace contains original operating
    quantities, process water, polyacrylamide and native catalyst supplies, spent
    catalyst in actual ash, both ash markets and the original hydroxide receiver
    under cut-off allocation. No recovered-energy credits
    or second solidification-water purchase are generated.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    selector = cast(dict[str, object], construction["source"])
    _service_trace(context, service)
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="Swiss_sludge_incineration",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    p = _self_production_amount(activity, native, method="Swiss sludge incineration")
    scale = -p / load.wet_mass_kg
    production = next(ex for ex in native if ex["type"] == "production")
    generated = [
        _sludge_reference(
            production,
            (database_name, provider_service_code(service_id)),
            str(service["reference_product"]),
            load,
        )
    ]
    changes: list[dict[str, object]] = []
    pam, _ = context.activity(
        cast(dict[str, object], construction["flocculant"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="Swiss_APC_PAM",
        semantic_class="service_normalization",
    )
    for duty, (product, _) in SUPPLIES.items():
        quantity = load.duties[duty]
        # PA finish/phosphate sludge contains no source toxic-metal allocand;
        # its real zero FeCl3 duty has no exchange or lognormal rescaling.
        if quantity == 0:
            continue
        # premise 2.4.6 can rename the gas-heat product while preserving its
        # declared reheat duty. Both names denote this one selected source.
        products = {product, "heat, district or industrial"} if duty == "reheat" else {product}
        rows = [
            ex
            for ex in native
            if ex["type"] == "technosphere" and context.input(ex)["reference product"] in products
        ]
        factor = quantity * scale / fsum(float(ex["amount"]) for ex in rows)
        final = []
        for ex in rows:
            row = _scaled_native_exchange(ex, factor)
            if duty == "PAM":
                row.update(input=cast(tuple[str, str], pam.key), name=str(pam["reference product"]))
                for field in ("flow", "activity", "properties"):
                    row.pop(field, None)
            generated.append(row)
            final.append(row)
        changes.append(
            {
                "duty": duty,
                "source_rows_json": context.exchange_json(rows),
                "generated_rows_json": context.exchange_json(final),
            }
        )
    if load.duties["CaO"] > 0:
        # Phosphate-ester sludge has positive source-equation neutralization
        # after FLUWA. The native donor has no lime row. G73 = B73/F73 supplies
        # packed lime at the source's 95% active grade through one complete RER
        # market; its delivery and manufacturing inventory remain unchanged.
        lime, _ = context.activity(
            cast(dict[str, object], construction["quicklime"]),
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id="Swiss_APC_quicklime",
            semantic_class="service_normalization",
        )
        quantity = load.duties["CaO"] / 0.95 * scale
        row = {
            "input": cast(tuple[str, str], lime.key),
            "type": "technosphere",
            "amount": quantity,
            "loc": quantity,
            "uncertainty type": 0,
        }
        generated.append(row)
        changes.append(
            {
                "duty": "CaO",
                "source_rows_json": "[]",
                "generated_rows_json": context.exchange_json([row]),
            }
        )
    for role, amount in (
        ("bottom_ash", -load.bottom_ash.treatment_mass_kg),
        ("residual_ash", -load.residual_ash.treatment_mass_kg),
    ):
        generated.append(
            {
                "input": (
                    database_name,
                    provider_service_code(
                        str(cast(dict[str, object], construction[role])["service_id"])
                    ),
                ),
                "type": "technosphere",
                "amount": amount * scale,
                "loc": amount * scale,
                "uncertainty type": 0,
            }
        )
    hydroxide_rows = [
        ex
        for ex in native
        if ex["type"] == "technosphere"
        and context.input(ex)["reference product"] == "metalliferous hydroxide sludge"
    ]
    hfactor = -load.hydroxide_kg * scale / fsum(float(ex["amount"]) for ex in hydroxide_rows)
    final = []
    for ex in hydroxide_rows:
        row = _scaled_native_exchange(ex, hfactor)
        # Positive recycled-content cut-off receiver; negative outgoing input.
        # No Waelz purchase or avoided-zinc credit is included.
        row["swiss_hydroxide_cutoff"] = {
            "mass_kg": load.hydroxide_kg,
            "water_kg": load.hydroxide_water_kg,
            "elements_kg": load.phases["hydroxide"],
            "carbon": [asdict(x) for x in load.hydroxide_carbon],
        }
        row.pop("properties", None)
        generated.append(row)
        final.append(row)
    changes.append(
        {
            "duty": "hydroxide_cutoff",
            "source_rows_json": context.exchange_json(hydroxide_rows),
            "generated_rows_json": context.exchange_json(final),
        }
    )
    original = context.source.activity(selector, year=2025)
    biosphere: dict[Emission, list[Any]] = {}
    for ex in context.exchanges(original):
        if ex["type"] == "biosphere":
            node = context.input(ex)
            key = (str(node["name"]), str(node["unit"]), tuple(node["categories"]))
            biosphere.setdefault(key, []).append(ex)
    for key, amount in load.emissions.items():
        if amount == 0:
            continue
        name, unit, categories = key
        final = []
        if key in {("Chloride", "kilogram", SURFACE), ("Oxygen", "kilogram", AIR_OXYGEN)}:
            # Known chloride uses its specified identity. Net oxygen demand replaces the uncertain
            # native gross-air amount.
            flow = context.biosphere.flow(
                {"name": name, "unit": unit, "categories": list(categories)}
            )
            row = {
                "input": cast(tuple[str, str], flow.key),
                "type": "biosphere",
                "name": name,
                "unit": unit,
                "amount": amount * scale,
                "loc": amount * scale,
                "uncertainty type": 0,
            }
            generated.append(row)
            final.append(row)
        else:
            rows = biosphere[key]
            factor = amount * scale / fsum(float(ex["amount"]) for ex in rows)
            for ex in rows:
                row = _scaled_native_exchange(ex, factor)
                if name.startswith(("Carbon dioxide,", "Carbon monoxide,", "Methane,")):
                    row["principal_carbon_origin_fractions"] = _principal_gas_shares(
                        load, name.endswith(", fossil")
                    )
                generated.append(row)
                final.append(row)
        changes.append(
            {
                "duty": name + "/" + "/".join(categories),
                "source_rows_json": context.exchange_json(biosphere.get(key, [])),
                "generated_rows_json": context.exchange_json(final),
            }
        )
    # Every altered native row is recorded as replaced.
    for kind in ("technosphere", "biosphere"):
        trace[f"retained_{kind}_json"] = "[]"
        trace[f"removed_{kind}_json"] = context.exchange_json(
            ex for ex in native if ex["type"] == kind
        )
    terms = asdict(load)
    terms["emissions"] = [
        {"name": n, "unit": u, "categories": c, "amount": q}
        for (n, u, c), q in load.emissions.items()
    ]
    terms["scaled_or_replaced_native_rows"] = changes
    terms["boundary"] = (
        "CH municipal co-treatment; empirical H/O and oxide service masses; "
        "actual C/N/P/S/Na/Fe/Co/Ca/K/Mg and auxiliaries; native process-water boundary; "
        "internal energy sharing without export credits; native supply chains as supplied."
    )
    trace["project_terms_json"] = json.dumps(terms, sort_keys=True, separators=(",", ":"))
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }
