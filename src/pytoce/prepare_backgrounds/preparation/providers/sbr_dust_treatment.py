"""Incinerate captured SBR mixer dust and treat its complete cemented residue."""

import json
import re
from dataclasses import asdict, dataclass
from math import fsum
from typing import Any, cast

from ..hazardous_incineration import (
    ACID_CAPTURE_FRACTIONS,
    LIME_ACID_WEIGHTS,
    NAOH_ACID_WEIGHTS,
    acid_neutralization_mass,
    reagent_element_masses,
    scrubber_capture_mass,
)
from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _copied_native_exchange, _RecipeContext, _scaled_native_exchange
from .treatment_compilation import _replace_carbon_identity

_URBAN = ("air", "urban air close to ground")
_SURFACE = ("water", "surface water")
_LONG = ("water", "ground-, long-term")
_DOC = "DOC, Dissolved Organic Carbon"
_CARBON_INDICATORS = {
    _DOC,
    "TOC, Total Organic Carbon",
    "COD, Chemical Oxygen Demand",
    "BOD5, Biological Oxygen Demand",
}
_HCL = "hydrochloric acid, without water, in 30% solution state"
_NAOH = "sodium hydroxide, without water, in 50% solution state"
_PEROXIDE = "hydrogen peroxide, without water, in 50% solution state"
_FERRIC = "iron(III) chloride, without water, in 40% solution state"
_AMMONIA = "ammonia, anhydrous, liquid"
_NG = "heat, district or industrial, natural gas"
_FACILITY = "hazardous waste incineration facility"
# The admitted compound-22 nominal atom convention is C12/H1/O16/N14/S32;
# Zn65.38 and these fresh-reagent metal masses retain their stated references.
_ATOMS = {
    "C": 12.0,
    "H": 1.0,
    "O": 16.0,
    "N": 14.0,
    "S": 32.0,
    "Zn": 65.38,
    "Na": 22.99,
    "Cl": 35.45,
    "Ca": 40.078,
    "Fe": 55.845,
    "Ti": 47.867,
    "V": 50.942,
}
# Doka II Table 4.1 non-air fractions size scrubber chemicals. C/N
# partitioning is assumed from the Doka burnable-waste reference.
_NONAIR = {
    "O": 0.083,
    "H": 0.0,
    "C": 1 - 0.9889898,
    "S": 0.9994,
    "N": 0.011,
    "P": 0.9993,
    "B": 0.88,
    "Cl": 0.9997,
    "F": 0.9995,
    "Cd": 1 - 0.00899,
    "Co": 0.9993,
    "Cr": 1 - 0.0000000739,
    "Cu": 0.9993,
    "Hg": 1 - 0.0402,
    "Ni": 0.9993,
    "Pb": 1 - 0.00331,
    "Zn": 0.9993,
    "Si": 1 - 0.00233,
}
_METALS = ("Cd", "Co", "Cr", "Cu", "Hg", "Ni", "Pb", "Zn")
_ELEMENT_NAMES = {
    "C": "carbon",
    "H": "hydrogen",
    "O": "oxygen",
    "N": "nitrogen",
    "S": "sulfur",
    "Zn": "zinc",
    "Ca": "calcium",
    "Fe": "iron",
    "Ti": "titanium",
    "V": "vanadium",
}


@dataclass(frozen=True, slots=True)
class _SbrDustLoad:
    """Loads per kg dry dust; heat is MJ, water[air] and water[water] are m3.

    Water formed_kg/ammonia_dilution_kg are kg. Exact nominal inlet elements
    differ from empirical C/N phases, fresh reagents and oxide-equivalent disposal mass.
    Auxiliary fossil and tallow carbon retain their own feedstock origins.
    """

    elements: dict[str, float]
    carbon: dict[str, float]
    reagent: dict[str, float]
    fresh: dict[str, dict[str, float]]
    ash: dict[str, float]
    ash_mass: float
    gas: dict[str, float]
    co: dict[str, float]
    co2: dict[str, float]
    ash_carbon: dict[tuple[str, ...], float]
    ash_nitrate: dict[tuple[str, ...], float]
    lhv: float
    heat: float
    water: dict[str, float]


def _dust_molecule(formula: dict[str, int], mass: float) -> dict[str, float]:
    """Convert a stated nominal molecule or repeat unit into kg elements."""
    molar_mass = fsum(_ATOMS[el] * n for el, n in formula.items())
    return {el: mass * _ATOMS[el] * n / molar_mass for el, n in formula.items()}


def _dust_inlet(formulation: dict[str, object]) -> tuple[dict[str, float], dict[str, float], float]:
    """Entrain the eight dry compound-22 additives proportionally.

    Liquid processing oil and SBR bales are excluded from mixer powder.
    Tallow stearic acid supplies biogenic carbon. Composition uses the TMQ
    repeat unit, 6PPD/TBBS formulas and the C12H26 wax reference.
    """
    formulas = {
        "carbon_black_n330": {"C": 1},
        "zinc_oxide": {"Zn": 1, "O": 1},
        "stearic_acid": {"C": 18, "H": 36, "O": 2},
        "tmq": {"C": 12, "H": 15, "N": 1},
        "six_ppd": {"C": 18, "H": 24, "N": 2},
        "protective_wax": {"C": 12, "H": 26},
        "rubber_makers_sulfur": {"S": 1},
        "tbbs": {"C": 11, "H": 14, "N": 2, "S": 2},
    }
    charged = {
        str(row["component_id"]): float(cast(float, row["amount_kg_per_gross_principal_kg"]))
        for row in cast(list[dict[str, object]], formulation["ingredients"])
    }
    dry = fsum(charged[name] for name in formulas)
    pools = {
        name: _dust_molecule(formula, charged[name] / dry) for name, formula in formulas.items()
    }
    elements = {
        el: fsum(pool.get(el, 0.0) for pool in pools.values())
        for el in ("C", "H", "O", "N", "S", "Zn")
    }
    carbon = {"non-fossil": pools["stearic_acid"]["C"]}
    carbon["fossil"] = elements["C"] - carbon["non-fossil"]
    # NIST combustion enthalpies (kJ/mol numerically equal MJ/kmol) apply to elemental carbon black
    # and sulfur. Michel's (via Muñoz et al. (2007), Eq. 9) organic HHV correlation applies only to
    # the five organic additives.
    organic_coefficients = {"C": 34.016, "H": 124.265, "O": -9.8324, "N": 6.276, "S": 19.079}
    hhv = (
        pools["carbon_black_n330"]["C"] * 393.52 / 12
        + pools["rubber_makers_sulfur"]["S"] * 296.84 / 32
        + fsum(
            pools[name].get(el, 0.0) * coefficient
            for name in ("stearic_acid", "tmq", "six_ppd", "protective_wax", "tbbs")
            for el, coefficient in organic_coefficients.items()
        )
    )
    # Nominal 9 kg combustion water/kg H converts this dry dust's HHV to LHV using 2.4417 MJ/kg
    # water: Harvey (1998), NISTIR 5078 Table 1 gives 2441.7 kJ/kg at 25 C, divided by 1000. This is
    # a calorific reference. This dust exceeds the HWI 23 MJ/kg no-oil bound.
    return elements, carbon, hhv - 9 * elements["H"] * 2.4417


def _dust_inputs(context: _RecipeContext, activity: Any, reference: float) -> dict[str, float]:
    """Read original product doses per positive source reference."""
    result: dict[str, float] = {}
    for ex in context.exchanges(activity):
        if ex["type"] == "technosphere":
            product = str(context.input(ex)["reference product"])
            result[product] = result.get(product, 0.0) + float(ex["amount"]) / reference
    return result


def _dust_emissions(
    context: _RecipeContext, activity: Any, reference: float
) -> dict[tuple[str, tuple[str, ...]], float]:
    """Read original emissions with their exact native compartment axes."""
    result: dict[tuple[str, tuple[str, ...]], float] = {}
    for ex in context.exchanges(activity):
        if ex["type"] == "biosphere":
            node = context.input(ex)
            key = str(node["name"]), tuple(node["categories"])
            result[key] = result.get(key, 0.0) + float(ex["amount"]) / reference
    return result


def _sbr_dust_load(
    context: _RecipeContext, construction: dict[str, object], services: dict[str, dict[str, object]]
) -> _SbrDustLoad:
    """Apply original HWI equations to the actual dust load.

    Doka II section 4.1.3 adopts reagent functions, not municipal captures.
    Native doses remain the source anchors, including their rounding. New
    burnable-waste C/N phases partition current aggregate DOC/nitrate once
    between this furnace and its single combined residue terminal.
    """
    formulation = services[str(construction["formulation_service_id"])]
    elements, carbon, lhv = _dust_inlet(cast(dict[str, object], formulation["construction"]))
    original = context.source.activity(cast(dict[str, object], construction["source"]), year=2025)
    rows = context.exchanges(original)
    source_p = -_self_production_amount(original, rows, method="SBR dust HWI source")
    fixed = context.source.activity(
        cast(dict[str, object], construction["fixed_source"]), year=2025
    )
    fixed_p = _self_production_amount(fixed, context.exchanges(fixed), method="fixed HWI source")
    source = _dust_inputs(context, original, source_p)
    fixed_inputs = _dust_inputs(context, fixed, fixed_p)
    emissions = _dust_emissions(context, original, source_p)
    fixed_emissions = _dust_emissions(context, fixed, fixed_p)
    # The original documentation gives wet elemental ppm. Its unreported
    # elements remain unknown, while the actual declared dust has known pools.
    donor = {
        el: float(value) / 1e6
        for el, value in re.findall(r"; ([A-Z][a-z]?) ([0-9.]+)", str(original["comment"]))
    }
    production = next(ex for ex in rows if ex["type"] == "production")
    props = cast(dict[str, dict[str, object]], production["properties"])
    donor["C"] = float(cast(float, props["carbon allocation"]["amount"]))
    capture_ratio = scrubber_capture_mass(elements, _NONAIR) / scrubber_capture_mass(
        {el: donor[el] for el in _NONAIR}, _NONAIR
    )
    metals_ratio = elements["Zn"] * _NONAIR["Zn"] / fsum(donor[el] * _NONAIR[el] for el in _METALS)
    n_ratio = elements["N"] / donor["N"]
    reagent = {
        "HCl": source[_HCL] * capture_ratio,
        "PAM": source["chemical, inorganic"] * capture_ratio,
        "FeCl3": source[_FERRIC] * metals_ratio,
        "NaOH": source[_NAOH]
        * acid_neutralization_mass(elements, {el: NAOH_ACID_WEIGHTS[el] for el in ("N", "S")})
        / acid_neutralization_mass(donor, NAOH_ACID_WEIGHTS),
        "CaO": source["quicklime, milled, packed"]
        * elements["N"]
        * ACID_CAPTURE_FRACTIONS["N"]
        * LIME_ACID_WEIGHTS["N"]
        / acid_neutralization_mass(donor, LIME_ACID_WEIGHTS),
        "H2O2": source[_PEROXIDE] * elements["S"] / donor["S"],
        "NH3_fixed": fixed_inputs[_AMMONIA],
        "NH3_waste_N": source[_AMMONIA] * n_ratio,
        "NG_fixed": fixed_inputs[_NG],
        "NG_waste_N": source[_NG] * n_ratio,
    }
    # Retain the two original environmental suppliers. Only the waste-N share scales with actual
    # dust nitrogen; fixed catalyst is per wet kg. Native chromium supplies the V2O5 manufacturing
    # proxy.
    reagent["TiO2_waste_N"] = source["titanium dioxide"] * n_ratio
    reagent["V2O5_waste_N"] = source["chromium"] * n_ratio
    reagent["TiO2"] = fixed_inputs["titanium dioxide"] + reagent["TiO2_waste_N"]
    reagent["V2O5"] = fixed_inputs["chromium"] + reagent["V2O5_waste_N"]
    formulas = {
        "HCl": {"H": 1, "Cl": 1},
        "NaOH": {"Na": 1, "O": 1, "H": 1},
        "FeCl3": {"Fe": 1, "Cl": 3},
        "CaO": {"Ca": 1, "O": 1},
        "H2O2": {"H": 2, "O": 2},
        "PAM": {"C": 3, "H": 5, "N": 1, "O": 1},
        "TiO2": {"Ti": 1, "O": 2},
        "V2O5": {"V": 2, "O": 5},
    }
    fresh = reagent_element_masses(reagent, formulas, _ATOMS)
    gas = {
        name: emissions[name, _URBAN] * n_ratio
        for name in ("Ammonia", "Nitrogen oxides", "Dinitrogen monoxide", "Cyanide")
    }
    co_total = emissions["Carbon monoxide, fossil", _URBAN]
    co = {origin: co_total * c / elements["C"] for origin, c in carbon.items()}
    # C/N phases follow the burnable reference. CN consumes fossil N-additive carbon; thermal NOx
    # uses atmospheric nitrogen.
    co2 = {
        origin: (
            c * (1 - 0.0110001 - 0.0000101)
            - co[origin] * 12 / 28
            - (gas["Cyanide"] * 12 / 26 if origin == "fossil" else 0.0)
        )
        * 44
        / 12
        for origin, c in carbon.items()
    }
    ash_carbon = {
        _SURFACE: emissions[_DOC, _SURFACE] * elements["C"] / donor["C"]
        - 0.0000101 * elements["C"],
        _LONG: emissions[_DOC, _LONG] * elements["C"] / donor["C"],
    }
    ash_nitrate = {
        _SURFACE: emissions["Nitrate", _SURFACE] * n_ratio - 0.001 * elements["N"] * 62 / 14,
        _LONG: emissions["Nitrate", _LONG] * n_ratio,
    }
    # Doka II Table 4.1 reports each phase independently: retain .0428 S solid and .986 Zn solid at
    # their reported source precision.
    ash = {
        "C": 0.0110001 * elements["C"],
        "N": 0.01 * elements["N"],
        "S": 0.0428 * elements["S"],
        "Zn": 0.986 * elements["Zn"],
        "O": 0.083 * elements["O"],
        "H": 0.0,
    }
    # The fresh-APC endpoint retains full PAM, lime, precipitant Fe, and installed Ti/V catalyst in
    # the combined ash. Na/Cl remain aqueous; reagent H/O retain their assigned species.
    waste_oxide_mass = (
        ash["C"] + ash["N"] * 62 / 14 + ash["S"] * 80 / 32 + ash["Zn"] * (1 + 16 / 65.38)
    )
    for name in ("PAM", "CaO", "TiO2", "V2O5"):
        for el, amount in fresh[name].items():
            ash[el] = ash.get(el, 0.0) + amount
    ash["Fe"] = fresh["FeCl3"]["Fe"]
    # Doka calculation AR64 starts at C row 23: raw O/H are separate element
    # bookkeeping, excluded here because full oxide factors include oxygen.
    ash_mass = (
        waste_oxide_mass
        + reagent["PAM"]
        + reagent["CaO"]
        + ash["Fe"] * (1 + 24 / 55.845)
        + reagent["TiO2"]
        + reagent["V2O5"]
    )
    # HWI net electricity is 10% of LHV less .287 kWh/kg own consumption.
    # This is a heat descriptor only: no recovered-energy credit is purchased.
    net_electricity = 0.1 * lhv - 0.287 * 3.6
    heat = lhv - net_electricity + reagent["NG_fixed"] + reagent["NG_waste_N"]
    ammonia = reagent["NH3_fixed"] + reagent["NH3_waste_N"]
    ammonia_air = fixed_emissions["Ammonia", _URBAN] + gas["Ammonia"]
    formed_water = 9 * (elements["H"] + (ammonia - ammonia_air) * 3 / 17)
    water = {
        "air": fixed_emissions["Water", ("air",)] + formed_water / 1000,
        "water": fixed_emissions["Water", ("water",)],
        "formed_kg": formed_water,
        "ammonia_dilution_kg": 3 * ammonia,
    }
    return _SbrDustLoad(
        elements,
        carbon,
        reagent,
        fresh,
        ash,
        ash_mass,
        gas,
        co,
        co2,
        ash_carbon,
        ash_nitrate,
        lhv,
        heat,
        water,
    )


def _dust_supply(
    context: _RecipeContext, activity: Any, product: str, amount: float
) -> tuple[list[dict[str, object]], list[Any]]:
    """Give an actual fixed dose to the entire dated native supplier group.

    Premise 2.4.6 heat.py relinks the original natural-gas market to its
    industrial-heat mix in all four REMIND members. These two product names
    identify the same DeNOx reheating role, not alternative supplier choices.
    """
    products = {product, "heat, district or industrial"} if product == _NG else {product}
    rows = [
        ex
        for ex in context.exchanges(activity)
        if ex["type"] == "technosphere" and context.input(ex)["reference product"] in products
    ]
    multiplier = amount / fsum(float(ex["amount"]) for ex in rows)
    return [_scaled_native_exchange(ex, multiplier) for ex in rows], rows


def _dust_emission(
    context: _RecipeContext, name: str, unit: str, categories: tuple[str, ...], amount: float
) -> dict[str, object]:
    """Use an existing ecoinvent elementary identity for a new dust equation."""
    flow = context.biosphere.flow({"name": name, "unit": unit, "categories": list(categories)})
    return {
        "input": cast(tuple[str, str], flow.key),
        "type": "biosphere",
        "amount": amount,
        "name": name,
        "unit": unit,
        "loc": amount,
        "uncertainty type": 0,
    }


def _dust_reference(
    exchange: Any, key: tuple[str, str], product: str, properties: dict[str, object]
) -> dict[str, object]:
    """Preserve signed production while replacing donor composition and identity."""
    result = _copied_native_exchange(exchange)
    for field in ("flow", "activity", "production volume", "classifications"):
        result.pop(field, None)
    result.update(
        input=key,
        name=product,
        properties=properties,
        comment=(
            f"Actual SBR receiving reference: {product}. Element "
            "properties are distinct from oxide-equivalent ash disposal mass."
        ),
    )
    return result


def _dust_dataset(
    service: dict[str, object], exchanges: list[dict[str, object]]
) -> dict[str, object]:
    """Describe the captured-dust or its single combined residue treatment."""
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": str(service["provider_service_id"]),
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": exchanges,
    }


def _dust_trace(
    context: _RecipeContext,
    trace: dict[str, object],
    native: tuple[Any, ...],
    retained: list[Any],
    terms: dict[str, object],
) -> None:
    """Record all retained and replaced native functions with their actual loads."""
    for kind in ("technosphere", "biosphere"):
        trace[f"retained_{kind}_json"] = context.exchange_json(
            ex for ex in retained if ex["type"] == kind
        )
        trace[f"removed_{kind}_json"] = context.exchange_json(
            ex for ex in native if ex["type"] == kind and ex not in retained
        )
    trace["project_terms_json"] = json.dumps(terms, sort_keys=True, separators=(",", ":"))


def _sbr_dust_incineration_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Replace HWI donor chemistry with dry compound-22 captured powder.

    Full fixed support remains once, with original catalyst suppliers and
    separate fixed and waste-N quantities. Original flattened ash burdens and releases are
    replaced by one complete ash child; only direct water releases stay here.
    The fixed recipe is self-supporting, so the old auxiliary oil is removed.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    load = _sbr_dust_load(context, construction, services)
    _service_trace(context, service)
    activity, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="captured_sbr_dust_hwi",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    scale = -_self_production_amount(activity, native, method="SBR dust HWI")
    production = next(ex for ex in native if ex["type"] == "production")
    properties: dict[str, object] = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": 1.0, "unit": "kg"},
        "water in wet mass": {"amount": 0.0, "unit": "kg"},
        "carbon allocation": {"amount": load.elements["C"], "unit": "kg"},
    }
    properties.update(
        {
            f"{_ELEMENT_NAMES[el]} content": {"amount": amount, "unit": "dimensionless"}
            for el, amount in load.elements.items()
        }
    )
    properties.update(
        {
            f"carbon content, {origin}": {"amount": amount, "unit": "dimensionless"}
            for origin, amount in load.carbon.items()
        }
    )
    generated = [
        _dust_reference(
            production,
            (database_name, provider_service_code(service_id)),
            str(service["reference_product"]),
            properties,
        )
    ]
    original = context.source.activity(cast(dict[str, object], construction["source"]), year=2025)
    source_p = -_self_production_amount(
        original, context.exchanges(original), method="SBR dust HWI source"
    )
    source_inputs = _dust_inputs(context, original, source_p)
    retained: list[Any] = []
    for product, amount in (
        (_FACILITY, source_inputs[_FACILITY]),
        ("titanium dioxide", load.reagent["TiO2_waste_N"]),
        ("chromium", load.reagent["V2O5_waste_N"]),
        (_AMMONIA, load.reagent["NH3_waste_N"]),
        (_NG, load.reagent["NG_waste_N"]),
        (_HCL, load.reagent["HCl"]),
        (_NAOH, load.reagent["NaOH"]),
        (_PEROXIDE, load.reagent["H2O2"]),
        (_FERRIC, load.reagent["FeCl3"]),
        ("quicklime, milled, packed", load.reagent["CaO"]),
    ):
        copied, rows = _dust_supply(context, activity, product, amount * scale)
        generated.extend(copied)
        retained.extend(rows)
    pam, _ = context.activity(
        cast(dict[str, object], construction["flocculant"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="complete_pam_flocculant",
        semantic_class="service_normalization",
    )
    copied, rows = _dust_supply(
        context, activity, "chemical, inorganic", load.reagent["PAM"] * scale
    )
    for row in copied:
        row.update(input=cast(tuple[str, str], pam.key), name=str(pam["reference product"]))
        for field in ("flow", "activity", "properties"):
            row.pop(field, None)
    generated.extend(copied)
    retained.extend(rows)
    fixed, fixed_trace = context.activity(
        cast(dict[str, object], construction["fixed_source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="complete_fixed_hwi",
        semantic_class="service_normalization",
    )
    fixed_rows = context.exchanges(fixed)
    fixed_p = _self_production_amount(fixed, fixed_rows, method="fixed HWI")
    # Catalyst quantities and the spent oxide mass use the original reference. A later source may
    # change supplier fractions; preserve those fractions and retain the fixed amount charged to the
    # dust.
    fixed_retained = []
    for product, oxide in (("titanium dioxide", "TiO2"), ("chromium", "V2O5")):
        amount = load.reagent[oxide] - load.reagent[oxide + "_waste_N"]
        copied, rows = _dust_supply(context, fixed, product, amount * scale)
        generated.extend(copied)
        fixed_retained.extend(rows)
    for ex in fixed_rows:
        if ex["type"] == "production":
            continue
        node = context.input(ex)
        if ex["type"] == "technosphere" and node["reference product"] in {
            "titanium dioxide",
            "chromium",
        }:
            continue
        if ex["type"] == "biosphere" and node["name"] == "Water":
            amount = load.water[str(tuple(node["categories"])[0])] * scale
            generated.append(_scaled_native_exchange(ex, amount / float(ex["amount"])))
        else:
            generated.append(_scaled_native_exchange(ex, scale / fixed_p))
        fixed_retained.append(ex)
    generated.append(
        {
            "input": (
                database_name,
                provider_service_code(
                    str(cast(dict[str, object], construction["ash"])["service_id"])
                ),
            ),
            "type": "technosphere",
            "amount": -load.ash_mass * scale,
            "loc": -load.ash_mass * scale,
            "uncertainty type": 0,
        }
    )
    for ex in native:
        if ex["type"] != "biosphere":
            continue
        node = context.input(ex)
        name, categories = str(node["name"]), tuple(node["categories"])
        if name in {"Carbon dioxide, fossil", "Carbon monoxide, fossil"}:
            for origin, amount in (
                load.co2 if name.startswith("Carbon dioxide") else load.co
            ).items():
                row = _scaled_native_exchange(ex, amount * scale / float(ex["amount"]))
                flow = context.biosphere.flow(
                    {
                        "name": name.replace("fossil", origin),
                        "unit": "kilogram",
                        "categories": list(categories),
                    }
                )
                _replace_carbon_identity(row, flow)
                generated.append(row)
            retained.append(ex)
            continue
        if name in load.gas and categories == _URBAN:
            amount = load.gas[name]
        elif name == "Sulfur dioxide":
            amount = 0.0006 * load.elements["S"] * 64 / 32
        elif name == "Zinc II" and categories == _URBAN:
            amount = 0.0007 * load.elements["Zn"]
        elif name in _CARBON_INDICATORS and categories == _SURFACE:
            amount = 0.0000101 * load.elements["C"]
            if name.startswith(("BOD5", "COD,")):
                amount *= 32 / 12
        elif name == "Nitrate" and categories == _SURFACE:
            amount = 0.001 * load.elements["N"] * 62 / 14
        elif name == "Sulfate" and categories == _SURFACE:
            # Table4.1 gives .957 of incoming S to direct water; nominal
            # SO4/S=96/32 converts that elemental amount to kg sulfate.
            amount = 0.957 * load.elements["S"] * 96 / 32
        elif name == "Zinc II" and categories == _SURFACE:
            amount = 0.0134 * load.elements["Zn"]
        elif name == "Hazardous waste disposed":
            amount = float(ex["amount"]) / scale
        else:
            continue
        generated.append(_scaled_native_exchange(ex, amount * scale / float(ex["amount"])))
        retained.append(ex)
    generated.extend(
        _dust_emission(context, name, unit, categories, amount * scale)
        for name, unit, categories, amount in (
            ("Sodium I", "kilogram", _SURFACE, load.fresh["NaOH"]["Na"]),
            ("Chloride", "kilogram", _SURFACE, load.fresh["HCl"]["Cl"] + load.fresh["FeCl3"]["Cl"]),
            ("Heat, waste", "megajoule", _URBAN, 0.83 * load.heat),
            ("Heat, waste", "megajoule", _SURFACE, 0.17 * load.heat),
        )
    )
    terms = asdict(load)
    # JSON needs named phases; the in-memory dictionaries keep actual axes.
    terms["ash_carbon"] = {"surface": load.ash_carbon[_SURFACE], "long": load.ash_carbon[_LONG]}
    terms["ash_nitrate"] = {"surface": load.ash_nitrate[_SURFACE], "long": load.ash_nitrate[_LONG]}
    _dust_trace(context, trace, native, retained, terms)
    _dust_trace(
        context,
        fixed_trace,
        fixed_rows,
        fixed_retained,
        {
            "reference_multiplier": scale / fixed_p,
        },
    )
    return _dust_dataset(service, generated)


def _sbr_dust_ash_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Treat one combined HWI ash with original dust and fresh-reagent custody.

    Native cement portfolios, two landfill support shares, facility, signed
    hydrated cement and other native terminal supplies remain unchanged.
    Original dust DOC/nitrate equals the HWI aggregate less direct releases;
    fresh PAM has separate source carbon and native nitrate coefficients.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    hwi = services[str(construction["incineration_service_id"])]
    load = _sbr_dust_load(context, cast(dict[str, object], hwi["construction"]), services)
    _service_trace(context, service)
    selector = cast(dict[str, object], construction["source"])
    original = context.source.activity(selector, year=2025)
    source_rows = context.exchanges(original)
    source_p = -_self_production_amount(original, source_rows, method="SBR dust ash source")
    source_production = next(ex for ex in source_rows if ex["type"] == "production")
    properties = cast(dict[str, dict[str, object]], source_production["properties"])
    dry = float(cast(float, properties["dry mass"]["amount"]))
    coefficients = _dust_emissions(context, original, source_p)
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="actual_sbr_dust_ash_solidification_landfill",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    p = _self_production_amount(activity, native, method="SBR dust ash treatment")
    scale = -p / load.ash_mass
    actual = {_ELEMENT_NAMES[el]: value for el, value in load.ash.items()}
    actual_mass = fsum(actual.values())
    target_properties: dict[str, object] = {
        "wet mass": {"amount": actual_mass / load.ash_mass, "unit": "kg"},
        "dry mass": {"amount": actual_mass / load.ash_mass, "unit": "kg"},
        "water in wet mass": {"amount": 0.0, "unit": "kg"},
        "carbon allocation": {"amount": actual["carbon"] / load.ash_mass, "unit": "kg"},
    }
    target_properties.update(
        {
            f"{el} content": {"amount": amount / actual_mass, "unit": "dimensionless"}
            for el, amount in actual.items()
        }
    )
    for origin, c in load.carbon.items():
        carbon = 0.0110001 * c + (load.fresh["PAM"]["C"] if origin == "fossil" else 0.0)
        target_properties[f"carbon content, {origin}"] = {
            "amount": carbon / actual_mass,
            "unit": "dimensionless",
        }
    production = next(ex for ex in native if ex["type"] == "production")
    generated = [
        _dust_reference(
            production,
            (database_name, provider_service_code(service_id)),
            str(service["reference_product"]),
            target_properties,
        )
    ]
    retained: list[Any] = []
    for ex in native:
        if ex["type"] == "production":
            continue
        if ex["type"] == "technosphere":
            generated.append(_copied_native_exchange(ex))
            retained.append(ex)
            continue
        node = context.input(ex)
        name, categories = str(node["name"]), tuple(node["categories"])
        if name == "Organic carbon, placed in landfill":
            amount = actual["carbon"] * scale
        elif categories[0] == "inventory indicator":
            amount = coefficients[name, categories] * -p
        elif name in _CARBON_INDICATORS:
            donor_c = dry * float(cast(float, properties["carbon content"]["amount"]))
            amount = coefficients[name, categories] * load.fresh["PAM"]["C"] / donor_c
            # Retain native PAM oxygen-demand ratios; dust organic carbon
            # uses 32 kg O2 per 12 kg C.
            amount += load.ash_carbon[categories] * (
                32 / 12 if name.startswith(("BOD5", "COD,")) else 1
            )
            amount *= scale
        elif name == "Nitrate":
            # Original dust nitrate already has its HWI/ash partition. Fresh PAM N adds only its
            # native ash release: kg nitrate/kg source ash divided by kg N/kg source ash.
            donor_n = dry * float(cast(float, properties["nitrogen content"]["amount"]))
            amount = (
                load.ash_nitrate[categories]
                + coefficients[name, categories] * load.fresh["PAM"]["N"] / donor_n
            ) * scale
        else:
            element = "sulfur" if name == "Sulfate" else name.split()[0].lower()
            if element not in actual:
                continue
            donor = dry * float(cast(float, properties[f"{element} content"]["amount"]))
            amount = coefficients[name, categories] * actual[element] / donor * scale
        generated.append(_scaled_native_exchange(ex, amount / float(ex["amount"])))
        retained.append(ex)
    # The original ash lacks Ti rows. Table A8's cumulative .2973 fraction
    # includes its .0004956 short part; subtract it once for additional long.
    generated.extend(
        _dust_emission(
            context, "Titanium ion", "kilogram", phase, load.ash["Ti"] * fraction * scale
        )
        for phase, fraction in ((_SURFACE, 0.0004956), (_LONG, 0.2973 - 0.0004956))
    )
    _dust_trace(
        context,
        trace,
        native,
        retained,
        {
            "oxide_equivalent_disposal_kg_per_captured_dust_kg": load.ash_mass,
            "actual_ash_elements_kg_per_captured_dust_kg": load.ash,
            "actual_atom_mass_kg_per_captured_dust_kg": actual_mass,
            "original_terminal_c_kg": 0.0110001 * load.elements["C"]
            - fsum(load.ash_carbon.values()),
            "original_terminal_n_kg": 0.01 * load.elements["N"]
            - fsum(load.ash_nitrate.values()) * 14 / 62,
            "terminal_supply_boundary": "unchanged native inventories",
        },
    )
    return _dust_dataset(service, generated)
