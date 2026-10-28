"""Compile sludge treatment from the selected RoW source inventories.

Source composition and normalized original quantities determine sludge
treatment. Compilation preserves dated supplier shares, negative production,
uncertainty and provenance for the admitted material inputs.
Swiss FLUWA treatment is calculated separately in swiss_sludge_incineration.
"""

import json
from dataclasses import asdict
from math import fsum
from typing import Any, cast

from .catalogue import provider_service_code
from .material_carbon import CARBON_ORIGINS, PrincipalCarbonOriginFractions
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _copied_native_exchange, _RecipeContext, _scaled_native_exchange
from .sludge_incineration_load import (
    _SUPPLY,
    _SURFACE,
    _URBAN,
    SludgeIncinerationSource,
    _Emission,
    _IncinerationLoad,
)

# Native production properties give element mass fractions in dry sludge. Specified source elements
# define each denominator.
_ELEMENT_NAMES = {
    "O": "oxygen",
    "H": "hydrogen",
    "C": "carbon",
    "S": "sulfur",
    "N": "nitrogen",
    "P": "phosphorus",
    "B": "boron",
    "Cl": "chlorine",
    "Br": "bromine",
    "F": "fluorine",
    "I": "iodine",
    "Ag": "silver",
    "As": "arsenic",
    "Ba": "barium",
    "Cd": "cadmium",
    "Co": "cobalt",
    "Cr": "chromium",
    "Cu": "copper",
    "Hg": "mercury",
    "Mn": "manganese",
    "Mo": "molybdenum",
    "Ni": "nickel",
    "Pb": "lead",
    "Sb": "antimony",
    "Se": "selenium",
    "Sn": "tin",
    "V": "vanadium",
    "Zn": "zinc",
    "Be": "beryllium",
    "Sc": "scandium",
    "Sr": "strontium",
    "Ti": "titanium",
    "Tl": "thallium",
    "W": "tungsten",
    "Si": "silicon",
    "Fe": "iron",
    "Ca": "calcium",
    "Al": "aluminium",
    "K": "potassium",
    "Mg": "magnesium",
    "Na": "sodium",
}


def _sludge_source(
    context: _RecipeContext, construction: dict[str, object]
) -> SludgeIncinerationSource:
    """Read the original negative-reference source quantities once.

    The selected rest-of-world or global plant treats landfill wastewater sludge
    with 70% water, in kg. Dry-mass properties are per reference product kg;
    the signed production scales exchanges, not those product properties.
    Element masses follow the properties reported in the native inventory.
    """
    original = context.source.activity(cast(dict[str, object], construction["source"]), year=2025)
    rows = context.exchanges(original)
    reference = -_self_production_amount(original, rows, method="leachate sludge incineration")
    production = next(ex for ex in rows if ex["type"] == "production")
    properties = cast(dict[str, dict[str, object]], production["properties"])
    dry = float(cast(float, properties["dry mass"]["amount"]))
    elements = {
        el: dry * float(cast(float, properties[f"{name} content"]["amount"]))
        for el, name in _ELEMENT_NAMES.items()
        if f"{name} content" in properties
    }
    inputs: dict[str, float] = {}
    emissions: dict[_Emission, float] = {}
    for ex in rows:
        if ex["type"] == "production":
            continue
        node = context.input(ex)
        amount = float(ex["amount"]) / reference
        if ex["type"] == "technosphere":
            product = str(node["reference product"])
            inputs[product] = inputs.get(product, 0.0) + amount
        else:
            key = (str(node["name"]), str(node["unit"]), tuple(node["categories"]))
            emissions[key] = emissions.get(key, 0.0) + amount
    return SludgeIncinerationSource(elements, inputs, emissions)


def _sludge_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    load: _IncinerationLoad,
    database_name: str,
) -> dict[str, object]:
    """Compile one negative wet-kg furnace from a completed leachate sludge load.

    Wastewater and thermal quantities are already calculated. The sludge market
    supplies collection/haulage; this operation supplies the furnace, air pollution
    control and support, and links both ash markets. Native catalyst suppliers
    retain the TiO2/V2O5 demand; their spent mass is already in the ash quantities.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    _service_trace(context, service)
    activity, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="actual_sludge_incineration",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    p = _self_production_amount(activity, native, method="leachate sludge incineration")
    scale = -p / load.wet_mass_kg
    production = next(ex for ex in native if ex["type"] == "production")
    reference = _copied_native_exchange(production)
    for field in ("flow", "activity", "production volume", "classifications"):
        reference.pop(field, None)
    properties: dict[str, object] = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": load.dry_mass_kg / load.wet_mass_kg, "unit": "kg"},
        "water in wet mass": {"amount": load.water_kg / load.wet_mass_kg, "unit": "kg"},
        "carbon allocation": {"amount": load.elements_kg["C"] / load.wet_mass_kg, "unit": "kg"},
    }
    properties.update(
        {
            f"{_ELEMENT_NAMES[el]} content": {
                "amount": mass / load.dry_mass_kg,
                "unit": "dimensionless",
            }
            for el, mass in load.elements_kg.items()
        }
    )
    for origin in ("fossil", "non-fossil"):
        mass = fsum(
            x.carbon_kg for x in load.carbon if (x.origin == "fossil") == (origin == "fossil")
        )
        properties[f"carbon content, {origin}"] = {
            "amount": mass / load.dry_mass_kg,
            "unit": "dimensionless",
        }
    reference.update(
        input=(database_name, provider_service_code(service_id)),
        name=str(service["reference_product"]),
        properties=properties,
        comment=(
            "Actual leachate sludge C/N/S/metals; dry mass uses separately traced nominal H/O. "
            "Native donor properties remain in provenance."
        ),
    )
    generated = [reference]
    retained: list[Any] = []
    # Required flocculant is the complete GLO polyacrylamide market. The source
    # generic-inorganic dose remains causal; only its manufacture identity changes.
    pam, _ = context.activity(
        cast(dict[str, object], construction["flocculant"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="APC_PAM",
        semantic_class="service_normalization",
    )
    for duty, (product, _) in _SUPPLY.items():
        products = {product}
        if duty == "reheat":
            # Premise 2.4.6 renames the original gas-heat market product.
            # Preserve its dated suppliers for the same waste-N reheat duty;
            # the dose above still comes from the original source.
            products.add("heat, district or industrial")
        rows = [
            ex
            for ex in native
            if ex["type"] == "technosphere" and context.input(ex)["reference product"] in products
        ]
        amount = load.duties[duty]
        amount *= scale
        if amount == 0:
            continue  # Actual metals can give zero FeCl3 for biogenic TPS.
        multiplier = amount / fsum(float(ex["amount"]) for ex in rows)
        for ex in rows:
            row = _scaled_native_exchange(ex, multiplier)
            if duty == "PAM":
                row.update(input=cast(tuple[str, str], pam.key), name=str(pam["reference product"]))
                for field in ("flow", "activity", "properties"):
                    row.pop(field, None)
            generated.append(row)
        retained.extend(rows)
    for role, amount in (
        ("bottom_ash", -load.bottom_ash.treatment_mass_kg),
        ("residual_ash", -load.residual_ash.treatment_mass_kg),
    ):
        # Each ash market owns its native denominator; pass its signed demand.
        amount *= scale
        child_id = str(cast(dict[str, object], construction[role])["service_id"])
        generated.append(
            {
                "input": (database_name, provider_service_code(child_id)),
                "type": "technosphere",
                "amount": amount,
                "loc": amount,
                "uncertainty type": 0,
            }
        )
    original = context.source.activity(cast(dict[str, object], construction["source"]), year=2025)
    biosphere: dict[_Emission, list[Any]] = {}
    for ex in context.exchanges(original):
        if ex["type"] == "biosphere":
            flow = context.input(ex)
            key = (str(flow["name"]), str(flow["unit"]), tuple(flow["categories"]))
            biosphere.setdefault(key, []).append(ex)
    new_apc = {"Chloride", *cast(list[str], construction["additional_apc_water_flows"])}
    for key, amount in load.emissions.items():
        if amount == 0:
            continue  # Absence is explicit in the typed target load.
        # Actual UPR glass and road mineral thermal phases can be absent from
        # the selected donor. Their admitted Doka coefficients bind existing
        # flow identities; every positive native template keeps uncertainty.
        new_mineral = (
            key[0]
            in {
                "Silicon",
                "Calcium II",
                "Aluminium III",
                "Titanium ion",
                "Potassium I",
                "Magnesium",
                "Sodium I",
                "Sulfate",
                "Sulfur dioxide",
            }
            and key[2] in {_URBAN, _SURFACE}
            and key not in biosphere
        )
        if not (new_mineral or (key[0] in new_apc and key[2] == _SURFACE)):
            template_key = key
            if key[0].startswith(("Carbon dioxide,", "Carbon monoxide,", "Methane,")):
                species, origin = key[0].rsplit(", ", 1)
                templates = cast(dict[str, str], construction["carbon_templates"])
                template_key = (f"{species}, {templates[origin]}", key[1], key[2])
            rows = biosphere[template_key]
            multiplier = amount * scale / fsum(float(ex["amount"]) for ex in rows)
            for ex in rows:
                row = _scaled_native_exchange(ex, multiplier)
                if template_key != key:
                    # The plastic-leachate source has only fossil templates.
                    # Keep its distribution but use the declared existing native
                    # non-fossil counterpart for actual biogenic/DAC sludge C.
                    node = context.biosphere.flow(
                        {"name": key[0], "unit": key[1], "categories": list(key[2])}
                    )
                    row.update(input=cast(tuple[str, str], node.key), name=key[0])
                    for field in ("flow", "activity", "properties"):
                        row.pop(field, None)
                if key[0].startswith(("Carbon dioxide,", "Carbon monoxide,", "Methane,")):
                    # The existing thermal model distributes every gas species
                    # homogeneously over actual sludge C. Split only diagnostic
                    # shares inside this native origin group; retain PAM/protein
                    # emissions in the unchanged inventory, outside principal C.
                    fossil = key[0].endswith(", fossil")
                    group_carbon = fsum(
                        pool.carbon_kg
                        for pool in load.carbon
                        if (pool.origin == "fossil") == fossil
                    )
                    principal_origins = cast(
                        PrincipalCarbonOriginFractions,
                        tuple(
                            fsum(
                                pool.carbon_kg
                                for pool in load.carbon
                                if pool.principal
                                and pool.origin == actual_origin
                                and (pool.origin == "fossil") == fossil
                            )
                            / group_carbon
                            for actual_origin in CARBON_ORIGINS
                        ),
                    )
                    row["principal_carbon_origin_fractions"] = principal_origins
                generated.append(row)
        else:
            # Actual APC and admitted mineral phases use their existing native
            # elementary identities, preserving the source records.
            name, unit, categories = key
            node = context.biosphere.flow(
                {"name": name, "unit": unit, "categories": list(categories)}
            )
            generated.append(
                {
                    "input": cast(tuple[str, str], node.key),
                    "name": name,
                    "unit": unit,
                    "type": "biosphere",
                    "amount": amount * scale,
                    "loc": amount * scale,
                    "uncertainty type": 0,
                }
            )
    for kind in ("technosphere", "biosphere"):
        trace[f"retained_{kind}_json"] = context.exchange_json(
            ex for ex in retained if ex["type"] == kind
        )
        trace[f"removed_{kind}_json"] = context.exchange_json(
            ex for ex in native if ex["type"] == kind and ex not in retained
        )
    terms = asdict(load)
    terms["process_emission_boundary"] = (
        "Doka mixed-waste process allocation: chlorinated VOC Cl belongs to co-treated "
        "reference waste, not target sludge/APC; C/H debits are nominal attribution."
    )
    terms["emissions"] = [
        {"name": name, "unit": unit, "categories": categories, "amount": amount}
        for (name, unit, categories), amount in load.emissions.items()
    ]
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
