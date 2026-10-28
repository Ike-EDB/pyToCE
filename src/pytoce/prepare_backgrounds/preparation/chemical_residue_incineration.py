"""Treat declared chemical-manufacturing residues through complete incineration.

The CH hazardous-waste function supplies operating coefficients; its linked
residual-material landfill supplies terminal functions. Actual wet throughput,
known elements, fresh reagents and separate nonfuel salts replace the donor
composition. For PA6/PA66 finish manufacture, unknown filter-aid mineral retains
the complete noncarbon environmental function and its full bulk disposal mass.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum
from typing import Any, cast

from .chemical_residue_load import (
    _AMMONIA,
    _FERRIC,
    _GAS_HEAT,
    _HCL,
    _NAOH,
    _OIL_HEAT,
    _PEROXIDE,
    _TMT,
    _URBAN,
    ChemicalResidueAshSource,
    ChemicalResidueLoad,
    ChemicalResidueSource,
    _ChemicalResidueReference,
    _residue_added_releases,
    _residue_ash_additions,
    _residue_ash_release,
    _residue_reference,
    _residue_waste_release,
)
from .providers.native_recipe import _self_production_amount
from .providers.recipe_context import (
    _copied_native_exchange,
    _RecipeContext,
    _scaled_native_exchange,
)

# PA finish-manufacturing residues and the actual PA receiving model share these complete native
# functions. CH supplies the hazardous-waste operating reference; RoW supplies its linked
# residual-material landfill. Their donor compositions are adapted to the declared waste, with
# native reagent and catalyst manufacture retained. PAM replaces the donor's inorganic proxy because
# the flocculant itself is known to be polyacrylamide.
CHEMICAL_RESIDUE_SOURCE: dict[str, object] = {
    "name": "treatment of hazardous waste, hazardous waste incineration",
    "reference_product": "hazardous waste, for incineration",
    "unit": "kilogram",
    "exact_location": "CH",
}
CHEMICAL_RESIDUE_FIXED_SOURCE: dict[str, object] = {
    "name": "process-specific burdens, hazardous waste incineration plant",
    "reference_product": "process-specific burdens, hazardous waste incineration plant",
    "unit": "kilogram",
    "exact_location": "CH",
}
CHEMICAL_RESIDUE_ASH_SOURCE: dict[str, object] = {
    "name": (
        "treatment of residues, MSWI-WWT-SLF, municipal solid waste, residual material landfill"
    ),
    "reference_product": "residues, MSWI-WWT-SLF, municipal solid waste",
    "unit": "kilogram",
    "exact_location": "RoW",
}
CHEMICAL_RESIDUE_PAM_SOURCE: dict[str, object] = {
    "name": "market for polyacrylamide",
    "reference_product": "polyacrylamide",
    "unit": "kilogram",
    "exact_location": "GLO",
}


@dataclass(slots=True)
class _NativeTreatment:
    """One traced native treatment with positive normalization."""

    activity: Any
    rows: tuple[Any, ...]
    reference: float
    production: Any
    supplies: dict[str, float]
    releases: dict[tuple[str, tuple[str, ...]], float]
    trace: dict[str, object]


def _native_treatment(
    context: _RecipeContext, selector: dict[str, object], owner: str
) -> _NativeTreatment:
    """Read a selected native operation and its allocated supplies once."""
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=owner,
        component_id=str(selector["name"]),
        semantic_class="service_normalization",
    )
    rows = context.exchanges(activity)
    reference = abs(_self_production_amount(activity, rows, method=owner))
    supplies: dict[str, float] = {}
    releases: dict[tuple[str, tuple[str, ...]], float] = {}
    for row in rows:
        if row["type"] == "production":
            continue
        node = context.input(row)
        amount = float(row["amount"]) / reference
        if row["type"] == "technosphere":
            product = str(node["reference product"])
            supplies[product] = supplies.get(product, 0.0) + amount
        else:
            key = str(node["name"]), tuple(node["categories"])
            releases[key] = releases.get(key, 0.0) + amount
    return _NativeTreatment(
        activity,
        rows,
        reference,
        next(r for r in rows if r["type"] == "production"),
        supplies,
        releases,
        trace,
    )


def _chemical_residue_source(
    context: _RecipeContext,
    hwi: _NativeTreatment,
    fixed: _NativeTreatment,
    source_selector: dict[str, object],
    gas_heat_product: str,
    *,
    integrated_pa_site: bool,
) -> ChemicalResidueSource:
    """Admit the exact native chemical-residue operation and supporting boiler.

    The PA spinning site already owns heat, electricity and ordinary water.
    Its integrated cleaning therefore requires no stand-alone boiler lookup.
    Off-site treatment retains the selected boiler's allocated kg/MJ and kWh/MJ.
    """
    # Premise 2.4.6 extraction drops exchange properties. Use the original
    # intensive composition with the dated operating rows below.
    original = context.source.activity(source_selector, year=2025)
    production = next(r for r in context.exchanges(original) if r["type"] == "production")
    # Native waste composition is reported in mg element/kg wet waste in the
    # source comment. Dividing by 10^6 yields kg/kg; carbon's allocated product
    # property already uses kg carbon/kg waste and keeps its native basis.
    donor = {
        el: float(value) / 1e6
        for el, value in re.findall(r"; ([A-Z][a-z]?) ([0-9.]+)", str(original["comment"]))
    }
    donor["C"] = float(production["properties"]["carbon allocation"]["amount"])
    oil = electricity = 0.0
    if not integrated_pa_site:
        oil_row = next(
            r
            for r in hwi.rows
            if r["type"] == "technosphere" and context.input(r)["reference product"] == _OIL_HEAT
        )
        boiler = context.input(oil_row)
        boiler_rows = context.exchanges(boiler)
        boiler_p = _self_production_amount(
            boiler, boiler_rows, method="chemical-residue supporting boiler"
        )
        oil = fsum(
            float(r["amount"]) / boiler_p
            for r in boiler_rows
            if r["type"] == "technosphere"
            and context.input(r)["reference product"] == "light fuel oil"
        )
        electricity = fsum(
            float(r["amount"]) / boiler_p
            for r in boiler_rows
            if r["type"] == "technosphere"
            and str(context.input(r)["reference product"]).startswith("electricity,")
        )
    return ChemicalResidueSource(
        donor_elements_kg=donor,
        # The heat equation names the original reheat duty. Premise supplies dated technology-mixed
        # industrial heat. Preserve its input key.
        waste_supplies={
            _GAS_HEAT if product == gas_heat_product else product: amount
            for product, amount in hwi.supplies.items()
        },
        waste_releases=hwi.releases,
        fixed_supplies={
            _GAS_HEAT if product == gas_heat_product else product: amount
            for product, amount in fixed.supplies.items()
        },
        fixed_releases=fixed.releases,
        oil_kg_per_mj=oil,
        boiler_electricity_kwh_per_mj=electricity,
    )


def _chemical_residue_ash_source(
    context: _RecipeContext,
    ash: _NativeTreatment,
    source_selector: dict[str, object],
) -> ChemicalResidueAshSource:
    """Pair original ash composition with the selected dated release coefficients.

    Original source properties are intensive kg/kg values, unaffected by native
    production magnitude. Dated releases already use positive throughput
    normalization. Premise omits those exchange properties from future sources.
    """
    original = context.source.activity(source_selector, year=2025)
    production = next(r for r in context.exchanges(original) if r["type"] == "production")
    properties = production["properties"]
    return ChemicalResidueAshSource(
        dry_fraction=float(properties["dry mass"]["amount"]),
        element_contents={
            element: float(properties[f"{element} content"]["amount"])
            for element in (
                "carbon",
                "nitrogen",
                "phosphorus",
                "sulfur",
                "sodium",
                "calcium",
                "vanadium",
                "iron",
                "nickel",
                "silicon",
                "aluminium",
                "potassium",
                "magnesium",
            )
        },
        releases=ash.releases,
    )


def _residue_emission(
    context: _RecipeContext, name: str, unit: str, categories: tuple[str, ...], amount: float
) -> dict[str, object]:
    """Bind a new receiving-function quantity to an existing native flow."""
    flow = context.biosphere.flow({"name": name, "unit": unit, "categories": list(categories)})
    return {
        "input": cast(tuple[str, str], flow.key),
        "name": name,
        "unit": unit,
        "type": "biosphere",
        "amount": amount,
        "loc": amount,
        "uncertainty type": 0,
    }


def _residue_link(key: tuple[str, str], amount: float) -> dict[str, object]:
    """Record a signed demand."""
    return {
        "input": key,
        "type": "technosphere",
        "amount": amount,
        "loc": amount,
        "uncertainty type": 0,
    }


def _residue_dataset(
    source: _NativeTreatment,
    key: tuple[str, str],
    product: str,
    rows: list[dict[str, object]],
    properties: dict[str, object],
    reference: _ChemicalResidueReference,
) -> dict[str, object]:
    """Normalize complete treatment to negative one kg with source provenance."""
    production = _scaled_native_exchange(source.production, 1 / source.reference)
    for field in ("flow", "activity", "production volume", "classifications"):
        production.pop(field, None)
    production.update(input=key, name=product, properties=properties)
    return {
        "name": product + ", complete treatment",
        "reference product": product,
        "unit": "kilogram",
        "location": str(source.activity["location"]),
        "type": "process",
        "comment": "Declared chemical-manufacturing residue treatment; original source: "
        + str(source.activity["name"])
        + ". Intact potassium sulfate remains separate from organic carbon and reactive "
        "elements. Heat and phase transfers follow the source assumptions. Linked fuel and "
        "terminal services receive no export credit.",
        "chemical_residue_reference": reference,
        "exchanges": [production, *rows],
    }


def _residue_trace(
    context: _RecipeContext,
    source: _NativeTreatment,
    native_pairs: list[tuple[Any, dict[str, object]]],
    reference: Mapping[str, object],
) -> None:
    """Separate unchanged native rows from actual scaled/replaced exchange pairs."""
    retained: list[Any] = []
    changed: list[dict[str, object]] = []
    for original, generated in native_pairs:
        if _copied_native_exchange(original) == generated:
            retained.append(original)
        else:
            changed.append(
                {
                    "source_rows_json": context.exchange_json([original]),
                    "generated_rows_json": context.exchange_json([generated]),
                }
            )
    for kind in ("technosphere", "biosphere"):
        source.trace[f"retained_{kind}_json"] = context.exchange_json(
            r for r in retained if r["type"] == kind
        )
        source.trace[f"removed_{kind}_json"] = context.exchange_json(
            r for r in source.rows if r["type"] == kind and r not in retained
        )
    source.trace["project_terms_json"] = json.dumps(
        {**reference, "scaled_or_replaced_native_rows": changed},
        sort_keys=True,
        separators=(",", ":"),
    )


def _residue_ash_dataset(
    context: _RecipeContext,
    source: _NativeTreatment,
    ash_source: ChemicalResidueAshSource,
    key: tuple[str, str],
    product: str,
    reference: _ChemicalResidueReference,
) -> dict[str, object]:
    """Retain full terminal support for actual residual elements and fresh reagents.

    Original waste N/S and unknown-mineral trace releases already occur in
    the furnace's combined native functions. This terminal receives the
    separately derived residual organic carbon, known mineral atoms and fresh
    reagent atoms. Native terminal supplies remain unchanged. Intact potassium
    sulfate uses Table A8 co-release, independently of organic-carbon/DOC and
    oxide-equivalent accounting.
    """
    b = reference["ash_mass_kg"]
    ash = reference["ash"]
    rows: list[dict[str, object]] = []
    native_pairs: list[tuple[Any, dict[str, object]]] = []
    for ex in source.rows:
        if ex["type"] == "production":
            continue
        node = context.input(ex)
        if ex["type"] == "technosphere":
            amount = float(ex["amount"]) / source.reference
        else:
            name, categories = str(node["name"]), tuple(node["categories"])
            amount = _residue_ash_release(
                name, categories, float(ex["amount"]) / source.reference, ash_source, reference
            )
            if amount is None:
                continue
        if amount != 0:
            row = _scaled_native_exchange(ex, amount / float(ex["amount"]))
            rows.append(row)
            native_pairs.append((ex, row))
    for name, categories, amount in _residue_ash_additions(reference):
        rows.append(_residue_emission(context, name, "kilogram", categories, amount))
    properties: dict[str, object] = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": 1.0, "unit": "kg"},
        "water in wet mass": {"amount": 0.0, "unit": "kg"},
        "carbon allocation": {"amount": ash["C"] / b, "unit": "kg"},
    }
    _residue_trace(context, source, native_pairs, reference)
    return _residue_dataset(source, key, product, rows, properties, reference)


def chemical_residue_datasets(
    context: _RecipeContext,
    load: ChemicalResidueLoad,
    *,
    incineration_key: tuple[str, str],
    ash_key: tuple[str, str],
    residue_name: str,
    include_collection: bool = True,
) -> dict[tuple[str, str], dict[str, object]]:
    """Construct complete HWI and ash records for one declared chemical residue.

    Both records retain negative waste production normalized to -1 kg. The
    caller demands ``-load.wet_mass_kg`` from ``incineration_key`` per its
    functional unit. Collection is included here once when ``include_collection``
    is true. Road-container residue selects direct HWI with no incoming haul;
    it passes false while retaining the stand-alone plant and energy functions.
    The fixed and waste-N catalyst shares retain the original complete native
    suppliers. Their oxide equivalents remain part of outgoing ash.
    The caller's distinct scientific ``residue_name`` also distinguishes
    semantic provider descriptors after premise relinking/regionalization.

    Integrated PA maintenance binds the shared equations through its portable
    receiving owner, inside the spinning site's water, energy and capital.
    """
    hwi = _native_treatment(context, CHEMICAL_RESIDUE_SOURCE, incineration_key[1])
    fixed = _native_treatment(context, CHEMICAL_RESIDUE_FIXED_SOURCE, incineration_key[1])
    ash_selector = CHEMICAL_RESIDUE_ASH_SOURCE
    ash = _native_treatment(context, ash_selector, ash_key[1])
    # Premise 2.4.6 relinks industrial natural-gas heat to the matching IAM heat mix. These are the
    # two declared source roles. Early technology frontiers still use original ordinary heat.
    gas_heat_product = (
        _GAS_HEAT
        if context.ordinary.database_name == context.source.database_name
        else "heat, district or industrial"
    )
    source = _chemical_residue_source(
        context,
        hwi,
        fixed,
        CHEMICAL_RESIDUE_SOURCE,
        gas_heat_product,
        integrated_pa_site=False,
    )
    ash_source = _chemical_residue_ash_source(context, ash, ash_selector)
    ref = _residue_reference(load, source, integrated_pa_site=False)
    m, q, nr = load.wet_mass_kg, ref["supplies"], ref["nitrogen_ratio"]
    supply = {
        _AMMONIA: q["NH3_waste"],
        gas_heat_product: q["NG_waste"],
        _OIL_HEAT: ref["oil_heat_mj"],
        _HCL: q["HCl"],
        _NAOH: q["NaOH"],
        _PEROXIDE: q["H2O2"],
        _FERRIC: q["FeCl3"],
        _TMT: q["TMT"],
        "calcium chloride": q["CaCl2"],
        "quicklime, milled, packed": q["CaO"],
        "chemical, inorganic": q["PAM"],
        # The donor apportions these catalyst supplies to waste nitrogen.
        # nr scales the native per-kg demand to the actual residue's nitrogen;
        # the fixed plant shares below remain per kg wet throughput. Chromium
        # is the native environmental proxy for the vanadium oxide.
        "titanium dioxide": hwi.supplies["titanium dioxide"] * nr,
        "chromium": hwi.supplies["chromium"] * nr,
        "hazardous waste incineration facility": hwi.supplies[
            "hazardous waste incineration facility"
        ]
        * m,
    }
    pam, _ = context.activity(
        CHEMICAL_RESIDUE_PAM_SOURCE,
        source_role="native_ecoinvent_provider",
        service_id=incineration_key[1],
        component_id="complete_PAM_supply",
        semantic_class="service_normalization",
    )
    generated: list[dict[str, object]] = []
    native_pairs: list[tuple[Any, dict[str, object]]] = []
    for ex in hwi.rows:
        if ex["type"] == "production":
            continue
        node = context.input(ex)
        if ex["type"] == "technosphere":
            product = str(node["reference product"])
            if product not in supply or supply[product] == 0:
                continue
            row = _scaled_native_exchange(
                ex, supply[product] / (m * hwi.supplies[product] * hwi.reference)
            )
            if product == "chemical, inorganic":
                row.update(input=cast(tuple[str, str], pam.key), name="polyacrylamide")
                for field in ("flow", "activity", "properties"):
                    row.pop(field, None)
        else:
            name, categories = str(node["name"]), tuple(node["categories"])
            amount = _residue_waste_release(
                name, categories, float(ex["amount"]) / hwi.reference, load, ash_source, ref
            )
            if amount is None:
                continue
            if name in {"Carbon dioxide, fossil", "Carbon monoxide, fossil"}:
                counterpart = name.replace("fossil", "non-fossil")
                nonfossil = ref["gases"][counterpart]
                if nonfossil != 0:
                    # Split the native source row and retain fuel/reagent carbon at its original
                    # origin. Preserve uncertainty/allocation and use its exact existing ecoinvent
                    # flow counterpart in the same compartment.
                    organic = _scaled_native_exchange(ex, nonfossil / (m * float(ex["amount"])))
                    flow = context.biosphere.flow(
                        {
                            "name": counterpart,
                            "unit": str(node["unit"]),
                            "categories": list(categories),
                        }
                    )
                    organic.update(input=cast(tuple[str, str], flow.key), name=counterpart)
                    for field in ("flow", "activity", "properties"):
                        organic.pop(field, None)
                    if counterpart in ref["principal_gas_origin_fractions"]:
                        organic["principal_carbon_origin_fractions"] = ref[
                            "principal_gas_origin_fractions"
                        ][counterpart]
                    generated.append(organic)
                    native_pairs.append((ex, organic))
            if amount == 0:
                continue
            row = _scaled_native_exchange(ex, amount / (m * float(ex["amount"])))
            if name in ref["principal_gas_origin_fractions"] and categories == _URBAN:
                row["principal_carbon_origin_fractions"] = ref["principal_gas_origin_fractions"][
                    name
                ]
        generated.append(row)
        native_pairs.append((ex, row))
    fixed_pairs: list[tuple[Any, dict[str, object]]] = []
    for ex in fixed.rows:
        if ex["type"] == "production":
            continue
        node = context.input(ex)
        if ex["type"] == "biosphere" and node["name"] == "Water":
            # Actual water outputs are bound below; HWI retains its native well.
            continue
        row = _scaled_native_exchange(ex, 1 / fixed.reference)
        generated.append(row)
        fixed_pairs.append((ex, row))
    generated.append(_residue_link(ash_key, -ref["ash_mass_kg"] / m))
    if include_collection:
        # Road containers select direct HWI. Other manufacturing residues
        # retain incoming collection.
        collection = _native_treatment(
            context,
            {
                "name": "market for hazardous waste, for incineration",
                "reference_product": "hazardous waste, for incineration",
                "unit": "kilogram",
                "exact_location": "CH",
            },
            incineration_key[1],
        )
        hauled = [
            r
            for r in collection.rows
            if r["type"] == "technosphere"
            and str(context.input(r)["reference product"]).startswith("transport, freight,")
        ]
        haul_pairs = [(r, _scaled_native_exchange(r, 1 / collection.reference)) for r in hauled]
        generated.extend(row for _, row in haul_pairs)
        _residue_trace(
            context,
            collection,
            haul_pairs,
            {
                "wet_handoff_kg": m,
                "scope": "CH industrial collection transfer, excluding original treatment mix",
            },
        )
    for name, unit, categories, amount in _residue_added_releases(load, ref):
        generated.append(_residue_emission(context, name, unit, categories, amount / m))
    _residue_trace(context, hwi, native_pairs, ref)
    _residue_trace(context, fixed, fixed_pairs, ref)
    properties: dict[str, object] = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": 1 - load.carrier_water_kg / m, "unit": "kg"},
        "water in wet mass": {"amount": load.carrier_water_kg / m, "unit": "kg"},
        "carbon allocation": {"amount": load.active_elements_kg["C"] / m, "unit": "kg"},
    }
    return {
        incineration_key: _residue_dataset(
            hwi, incineration_key, residue_name, generated, properties, ref
        ),
        ash_key: _residue_ash_dataset(
            context, ash, ash_source, ash_key, residue_name + ", incineration ash", ref
        ),
    }
