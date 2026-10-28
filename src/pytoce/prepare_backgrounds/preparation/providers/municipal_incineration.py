"""Compile municipal incineration from actual material and carbon pools."""

import json
from dataclasses import asdict
from math import fsum
from typing import cast

from .catalogue import provider_service_code
from .material_carbon import CARBON_ORIGINS
from .municipal_incineration_emissions import _Emission
from .municipal_incineration_load import (
    _MUNICIPAL_ELEMENTS,
    _MUNICIPAL_SUPPLIES,
    MunicipalIncinerationLoad,
    MunicipalIncinerationMaterial,
    MunicipalIncinerationSource,
)
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _RecipeContext, _scaled_native_exchange


def _municipal_incineration_source(
    context: _RecipeContext, construction: dict[str, object]
) -> MunicipalIncinerationSource:
    """Admit original intensive composition and negative-reference operating rows.

    Source properties are per kg, independent of production amount.
    Material quantities are supplied separately from dated supplier shares.
    """
    original = context.source.activity(cast(dict[str, object], construction["source"]), year=2025)
    rows = context.exchanges(original)
    basis = -_self_production_amount(original, rows, method="municipal incineration source")
    reference = next(row for row in rows if row["type"] == "production")
    properties = cast(dict[str, dict[str, float]], reference["properties"])
    names = {f"{item.property_name} content": el for el, item in _MUNICIPAL_ELEMENTS.items()}
    names.update({"hydrogen content": "H", "oxygen content": "O"})
    elements = {
        names[name]: properties["dry mass"]["amount"] * prop["amount"]
        for name, prop in properties.items()
        if name in names
    }
    inputs: dict[str, float] = {}
    gases: dict[_Emission, float] = {}
    for row in rows:
        if row["type"] == "production":
            continue
        node = context.input(row)
        amount = float(row["amount"]) / basis
        if row["type"] == "technosphere":
            product = str(node["reference product"])
            inputs[product] = inputs.get(product, 0.0) + amount
        else:
            key = (str(node["name"]), str(node["unit"]), tuple(node["categories"]))
            gases[key] = gases.get(key, 0.0) + amount
    return MunicipalIncinerationSource(
        elements, properties["water in wet mass"]["amount"], inputs, gases
    )


def _municipal_carbon_groups(
    material: MunicipalIncinerationMaterial,
) -> tuple[tuple[str, float, tuple[float, float, float]], ...]:
    """Partition only furnace waste-carbon gases and their principal subsets.

    The common municipal C transfers apply to every supplied material pool.
    Fresh polyacrylamide for air pollution control joins ash after the furnace;
    it and embedded fuels retain their own carbon. DAC uses native non-fossil flows.
    """
    groups = []
    for category in ("fossil", "non-fossil"):
        pools = tuple(
            pool for pool in material.carbon if (pool.origin == "fossil") == (category == "fossil")
        )
        carbon = fsum(pool.carbon_kg for pool in pools)
        if carbon == 0:
            continue
        fractions = cast(
            tuple[float, float, float],
            tuple(
                fsum(pool.carbon_kg for pool in pools if pool.principal and pool.origin == origin)
                / carbon
                for origin in CARBON_ORIGINS
            ),
        )
        groups.append((category, carbon / material.elements_kg["C"], fractions))
    return tuple(groups)


def _municipal_bottom_ash_water_fraction(
    context: _RecipeContext, selector: dict[str, object]
) -> float:
    """Read the selected original bottom-ash water fraction per wet reference kg.

    Road's paint source reports 0.173671 kg water and 0.826329 kg dry material.
    Native properties are intensive and independent of signed production;
    future inventories can omit them, so original admission owns this ratio.
    """
    original = context.source.activity(selector, year=2025)
    production = next(row for row in context.exchanges(original) if row["type"] == "production")
    properties = production["properties"]
    water = float(properties["water in wet mass"]["amount"])
    return water / (float(properties["dry mass"]["amount"]) + water)


def _municipal_incineration_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    load: MunicipalIncinerationLoad,
    database_name: str,
    *,
    material_name: str,
    principal_origin: str,
    principal_elements_kg: dict[str, float],
    production_comment: str,
) -> dict[str, object]:
    """Compile one principal-kg municipal function and its signed actual ash demands.

    The native furnace's negative, nonunit production is normalized here.
    Calculated material-specific operating quantities stay fixed while supplier
    shares follow the selected year and background. External collection and
    inbound transport are outside this furnace function.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    key = (database_name, provider_service_code(service_id))
    _service_trace(context, service)
    activity, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id=f"actual_{material_name.lower()}_municipal_incineration",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    production = _self_production_amount(activity, native, method="municipal incineration")
    material = load.material
    dry_mass = material.wet_mass_kg - material.water_kg
    total_carbon = material.elements_kg["C"]
    properties: dict[str, object] = {
        "wet mass": {"amount": material.wet_mass_kg, "unit": "kg"},
        "dry mass": {"amount": dry_mass, "unit": "kg"},
        "water in wet mass": {"amount": material.water_kg, "unit": "kg"},
        "carbon allocation": {"amount": total_carbon, "unit": "kg"},
    }
    physical = dict(material.elements_kg)
    for element, amount in material.bulk_inert_elements_kg.items():
        physical[element] = physical.get(element, 0.0) + amount
    for element, amount in physical.items():
        name = {"H": "hydrogen", "O": "oxygen", "K": "potassium", "Mg": "magnesium"}.get(element)
        if name is None:
            name = _MUNICIPAL_ELEMENTS[element].property_name
        properties[f"{name} content"] = {"amount": amount / dry_mass, "unit": "dimensionless"}
    for category in ("fossil", "non-fossil"):
        properties[f"carbon content, {category}"] = {
            "amount": fsum(
                pool.carbon_kg
                for pool in material.carbon
                if (pool.origin == "fossil") == (category == "fossil")
            )
            / dry_mass,
            "unit": "dimensionless",
        }
    generated: list[dict[str, object]] = [
        {
            "input": key,
            "type": "production",
            "amount": 1.0,
            "name": str(service["reference_product"]),
            "unit": "kilogram",
            "properties": properties,
            "comment": production_comment,
        }
    ]
    pam, _ = context.activity(
        cast(dict[str, object], construction["flocculant"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id=f"{material_name}_APC_PAM",
        semantic_class="service_normalization",
    )
    retained = []
    for duty, (product, _) in _MUNICIPAL_SUPPLIES.items():
        products = {product}
        if duty == "reheat":
            # Premise 2.4.6 relinks the selected gas-heat function to its dated general
            # district/industrial heat product.
            products.add("heat, district or industrial")
        rows = [
            ex
            for ex in native
            if ex["type"] == "technosphere" and context.input(ex)["reference product"] in products
        ]
        amount = load.duties[duty]
        if amount == 0:
            continue  # These absent material drivers create no reagent purchase.
        multiplier = amount / fsum(float(ex["amount"]) for ex in rows)
        for ex in rows:
            row = _scaled_native_exchange(ex, multiplier)
            if duty == "PAM":
                # Keep the source dry flocculant dose; replace its generic
                # inorganic manufacture proxy by the complete actual PAM supply.
                row.update(input=cast(tuple[str, str], pam.key), name=str(pam["reference product"]))
                for field in ("flow", "activity", "properties"):
                    row.pop(field, None)
            generated.append(row)
        retained.extend(rows)
    for role, amount in (
        ("bottom_ash", -load.bottom_ash.treatment_mass_kg),
        ("residual_ash", -load.residual_ash.treatment_mass_kg),
    ):
        target = str(cast(dict[str, object], construction[role])["service_id"])
        generated.append(
            {
                "input": (database_name, provider_service_code(target)),
                "type": "technosphere",
                "amount": amount,
                "loc": amount,
                "uncertainty type": 0,
            }
        )
    original = context.source.activity(cast(dict[str, object], construction["source"]), year=2025)
    original_biosphere = {}
    for ex in context.exchanges(original):
        if ex["type"] == "biosphere":
            flow = context.input(ex)
            template = (str(flow["name"]), str(flow["unit"]), tuple(flow["categories"]))
            original_biosphere.setdefault(template, []).append(ex)
    carbon_groups = _municipal_carbon_groups(material)
    principal_fractions: dict[tuple[str, str], float] = {}
    for (name, unit, categories), amount in load.emissions.items():
        template = (name, unit, categories)
        carbon_gas = name.startswith(("Carbon dioxide,", "Carbon monoxide,", "Methane,"))
        partitions = carbon_groups if carbon_gas else (("", 1.0, (0.0, 0.0, 0.0)),)
        for category, share, fractions in partitions:
            target_name = name.rsplit(", ", 1)[0] + ", " + category if carbon_gas else name
            target = context.biosphere.flow(
                {"name": target_name, "unit": unit, "categories": list(categories)}
            )
            if template in load.native_emission_factors:
                rows = original_biosphere[template]
                multiplier = amount * share / fsum(float(ex["amount"]) for ex in rows)
                emissions = [_scaled_native_exchange(ex, multiplier) for ex in rows]
            else:
                # Reconstructed carbon, water, heat and elemental coefficients
                # have no donor-composition uncertainty distribution.
                emissions = [
                    {
                        "type": "biosphere",
                        "amount": amount * share,
                        "loc": amount * share,
                        "uncertainty type": 0,
                    }
                ]
            for row in emissions:
                row.update(input=cast(tuple[str, str], target.key), name=target_name, unit=unit)
                if carbon_gas:
                    for field in ("flow", "activity", "properties"):
                        row.pop(field, None)
                    row["principal_carbon_origin_fractions"] = fractions
                    principal_fractions[cast(tuple[str, str], target.key)] = fsum(fractions)
                generated.append(row)
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = context.exchange_json(
        ex for ex in native if ex["type"] == "technosphere" and ex not in retained
    )
    trace["retained_biosphere_json"] = "[]"
    trace["removed_biosphere_json"] = context.exchange_json(
        ex for ex in native if ex["type"] == "biosphere"
    )
    terms = asdict(load)
    # JSON stores ordered records with explicit name, unit, and compartment fields.
    terms["emissions"] = [
        {"name": n, "unit": u, "categories": list(c), "amount": q}
        for (n, u, c), q in sorted(load.emissions.items())
    ]
    terms["native_emission_factors"] = [
        {"name": n, "unit": u, "categories": list(c), "factor": q}
        for (n, u, c), q in sorted(load.native_emission_factors.items())
    ]
    terms.update(
        principal_origin=principal_origin,
        principal_elements_kg=principal_elements_kg,
        native_production=production,
        public_treatment_kg=1.0,
        initial_haul_owner="outside municipal treatment",
        terminal_supply_boundary="unchanged native inventories",
        # Native carbon properties describe the modeled organic reference. Complete incoming C
        # includes the separately excluded limestone pool.
        physical_elements_kg={
            **physical,
            "C": physical["C"] + material.geological_carbon_kg,
        },
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
        "direct_treatment_carbon_owners_json": json.dumps(
            [
                {
                    "owner_database": database_name,
                    "owner_code": key[1],
                    "owner_scale_per_provider_unit": 1.0,
                    "owner_scope": "direct_biosphere_only",
                    "source_role": "member_local_frontier_service",
                    "origin_mapping_status": f"actual_{material_name.lower()}_principal_origin",
                    "principal_carbon_fractions": [
                        {"flow_database": flow[0], "flow_code": flow[1], "fraction": fraction}
                        for flow, fraction in sorted(principal_fractions.items())
                    ],
                }
            ],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "exchanges": generated,
    }
