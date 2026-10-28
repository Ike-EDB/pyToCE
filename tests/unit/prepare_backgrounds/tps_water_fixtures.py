"""Invented external inventories for actual TPS collection and wastewater tests.

All native quantities in this fixture are deliberately invented controls, per
m3 wastewater or kg sludge reference as stated. They are not licensed vectors.
Different original/future production magnitudes, treatment fractions and supplier
mixes expose double normalization and accidental retention of municipal loads.
The three packaged TPS formulations themselves remain unchanged.
"""

from copy import deepcopy
from math import log

from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.waste_gas import _GasTreatment

ORIGINS = ("biogenic", "fossil", "atmospheric_dac")
RAW_WATER = ("water",)
SURFACE = ("water", "surface water")
URBAN = ("air", "urban air close to ground")
RURAL = ("air", "non-urban air or from high stacks")
AIR_RESOURCE = ("natural resource", "in air")
LAND = ("natural resource", "land")
WASTE = ("inventory indicator", "waste")
LEACHATE = "leachate, SLF, municipal solid waste"
INCINERATION = "sewage sludge, 70% water, WWT-SLF, municipal solid waste"
LANDFARMING = "sewage sludge, 97% water, WWT-SLF, municipal solid waste"
CARBON_INDICATORS = (
    "TOC, Total Organic Carbon",
    "DOC, Dissolved Organic Carbon",
    "COD, Chemical Oxygen Demand",
    "BOD5, Biological Oxygen Demand",
)
RAW_ELEMENTS = (
    "Hydrogen",
    "Oxygen",
    "Nitrogen",
    "Phosphorus",
    "Sulfur",
    "Chlorine",
    "Sodium I",
    "Magnesium",
    "Zinc II",
    "Cobalt II",
    "Calcium II",
    "Potassium I",
)
EFFLUENT_IONS = (
    "Chloride",
    "Sodium I",
    "Magnesium",
    "Zinc II",
    "Iron ion",
    "Cobalt II",
    "Calcium II",
    "Potassium I",
    "Sulfate",
    "Nitrate",
    "Phosphate",
)

# Invented portfolio quantities per m3 native treatment. Their totals must be
# replaced by actual loads; only the dated supplier ratios remain. Negative
# waste demands pair with negative-production treatment leaves.
PORTFOLIOS = {
    "electricity": ("electricity, medium voltage", "kilowatt hour", 9.0),
    "heat": ("heat, district or industrial, natural gas", "megajoule", 11.0),
    "iron": ("iron sulfate", "kilogram", 0.3),
    "pam": ("polyacrylamide", "kilogram", 0.2),
    "sand": ("inert waste, for final disposal", "kilogram", -0.03),
    "paper": ("waste graphical paper", "kilogram", -0.04),
    "plastic": ("waste plastic, mixture", "kilogram", -0.05),
}


def flow_key(name: str, categories: tuple[str, ...]) -> tuple[str, str]:
    """Give each external flow a stable, invented name/compartment identifier."""
    return "biosphere", "water_test|" + "|".join((name, *categories))


def biosphere_inventory() -> dict:
    """List required native-shaped flows independently of the production mappings."""
    descriptors = {(name, "kilogram", RAW_WATER) for name in (*RAW_ELEMENTS, *CARBON_INDICATORS)}
    descriptors.update((name, "kilogram", SURFACE) for name in (*EFFLUENT_IONS, *CARBON_INDICATORS))
    descriptors.update(("Water", "cubic meter", c) for c in (RAW_WATER, SURFACE, URBAN, RURAL))
    descriptors.update(("Heat, waste", "megajoule", c) for c in (SURFACE, URBAN, RURAL))
    descriptors.update(
        (name, "kilogram", c)
        for name in ("Carbon dioxide, fossil", "Carbon dioxide, non-fossil", "Dinitrogen monoxide")
        for c in (URBAN, RURAL)
    )
    descriptors.update(
        {
            ("Oxygen", "kilogram", AIR_RESOURCE),
            ("Methane, non-fossil", "kilogram", URBAN),
            ("Occupation, industrial area", "square meter-year", LAND),
            ("Non-hazardous waste disposed", "kilogram", WASTE),
        }
    )
    return {
        flow_key(name, categories): {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "emission",
        }
        for name, unit, categories in sorted(descriptors)
    }


def exchange(key: tuple[str, str], amount: float, kind: str = "technosphere") -> dict:
    """Attach valid signed lognormal uncertainty to invented nonzero exchanges."""
    return {
        "input": key,
        "amount": amount,
        "type": kind,
        "uncertainty type": 2,
        "loc": log(abs(amount)) + 0.03,
        "scale": 0.21,
        "negative": amount < 0,
        "minimum": min(amount / 2, amount * 2),
        "maximum": max(amount / 2, amount * 2),
        "comment": "Invented external source control",
        "allocation": "cut-off control",
    }


def activity(
    database: str,
    code: str,
    name: str,
    product: str,
    unit: str,
    production: float,
    rows: list[dict],
    *,
    location: str = "RoW",
) -> dict:
    """Scale extensive per-reference rows, preserving intensive source metadata."""
    key = database, code
    reference = {
        "input": key,
        "amount": production,
        "type": "production",
        "uncertainty type": 0,
        "loc": production,
        "name": product,
        "unit": unit,
        "flow": "invented source flow",
        "activity": "invented source activity",
        "production volume": 123.0,
        "classifications": [("fixture", "source")],
        "properties": {"carbon content": {"amount": 0.25, "unit": "dimensionless"}},
    }
    scaled = []
    for row in rows:
        copy = deepcopy(row)
        factor = abs(production)
        copy["amount"] *= factor
        copy["loc"] += log(factor)
        copy["minimum"] *= factor
        copy["maximum"] *= factor
        scaled.append(copy)
    return {
        "name": name,
        "reference product": product,
        "unit": unit,
        "location": location,
        "type": "process",
        "exchanges": [reference, *scaled],
    }


def native_inventory(database: str, *, future: bool) -> dict:
    """Build full tiny collection, plant, two sludge markets and their suppliers."""
    result = {}
    shares = (0.7, 0.3) if future else (0.25, 0.75)
    # Distinct screenings supplier locations check that replacement preserves both geographies.
    screening_shares = (0.6, 0.4) if future else (0.2, 0.8)
    for prefix, (product, unit, quantity) in PORTFOLIOS.items():
        # Use a different industrial-heat product label in the future fixture.
        # Supplier identities and MJ quantities remain those of this fixture member.
        if future and prefix == "heat":
            product = "heat, district or industrial"
        for suffix, location in (("a", "RoW"), ("b", "RER")):
            code = f"{prefix}_{suffix}"
            result[database, code] = activity(
                database,
                code,
                f"invented supply of {product}",
                product,
                unit,
                -1 if quantity < 0 else 1,
                [exchange(flow_key("Carbon dioxide, fossil", RURAL), 0.005, "biosphere")],
                location=location,
            )
    for code, product, unit, p in (
        ("limestone", "limestone, crushed, washed", "kilogram", 1),
        ("retirement", "waste polyethylene", "kilogram", -1),
        ("sewer_a", "sewer grid", "kilometer", 1),
        ("sewer_b", "sewer grid", "kilometer", 1),
        ("terminal_haul", "transport, freight, lorry, diesel, unspecified", "ton kilometer", 1),
    ):
        result[database, code] = activity(
            database,
            code,
            f"invented {code}",
            product,
            unit,
            p,
            [exchange(flow_key("Carbon dioxide, fossil", RURAL), 0.005, "biosphere")],
            location="GLO" if code == "terminal_haul" else "RoW",
        )
    for role, product in (("incineration", INCINERATION), ("landfarming", LANDFARMING)):
        result[database, role] = activity(
            database,
            role,
            f"treatment of {product}",
            product,
            "kilogram",
            -1,
            [exchange(flow_key("Carbon dioxide, fossil", RURAL), 0.02, "biosphere")],
        )
        # Invented 50 km terminal wet haul, distinct from the source-model
        # internal 10 km dry haul. Future transportation intensity doubles.
        result[database, f"{role}_market"] = activity(
            database,
            f"{role}_market",
            f"market for {product}",
            product,
            "kilogram",
            -(8 if role == "incineration" else 10)
            if future
            else -(4 if role == "incineration" else 6),
            [
                exchange((database, role), -1),
                exchange((database, "terminal_haul"), 0.1 if future else 0.05),
            ],
        )
    plant_rows = []
    for prefix, (_, _, quantity) in PORTFOLIOS.items():
        mix = screening_shares if prefix in {"paper", "plastic"} else shares
        for suffix, share in zip(("a", "b"), mix, strict=True):
            plant_rows.append(exchange((database, f"{prefix}_{suffix}"), quantity * share))
    # Limestone kg/m3 encodes embodied construction electricity. Actual operating
    # electricity is separate; the real Infra coefficient is tested at its use.
    plant_rows.extend(
        [
            exchange((database, "limestone"), 0.006 if future else 0.004),
            exchange((database, "retirement"), -0.003),
            exchange((database, "incineration_market"), -0.8),
            exchange((database, "landfarming_market"), -0.6),
            exchange(flow_key("Occupation, industrial area", LAND), 0.015, "biosphere"),
            exchange(flow_key("Carbon dioxide, fossil", URBAN), 77.0, "biosphere"),
            exchange(flow_key("Water", SURFACE), 0.9, "biosphere"),
        ]
    )
    result[database, "plant"] = activity(
        database,
        "plant",
        f"treatment of {LEACHATE}, wastewater treatment",
        LEACHATE,
        "cubic meter",
        -7 if future else -3,
        plant_rows,
    )
    # Native-shape raw emissions supply templates.
    market_rows = [
        exchange((database, "plant"), -0.6 if future else -0.8),
        exchange((database, "sewer_a"), 0.0002),
        exchange((database, "sewer_b"), 0.0003),
        *[
            exchange(flow_key(name, RAW_WATER), 0.1, "biosphere")
            for name in (*RAW_ELEMENTS, *CARBON_INDICATORS)
        ],
        exchange(flow_key("Water", RAW_WATER), 0.2, "biosphere"),
    ]
    result[database, "collection"] = activity(
        database,
        "collection",
        f"market for {LEACHATE}",
        LEACHATE,
        "cubic meter",
        -5 if future else -2,
        market_rows,
    )
    return result


def water_catalogue() -> dict:
    """Add direct-owner declarations while retaining all three real TPS formulations."""
    catalogue = load_provider_catalogue()
    method = "invented_tps_water_boundary"
    registry = catalogue["method_authority_registry"]
    registry["exact_source_bindings"][method] = ["ecoinvent_3_12_cutoff"]
    registry["exact_method_authorities"][method] = {
        **registry["project_recipe_defaults"],
        "limitation": "Synthetic inventories for collection and wastewater tests.",
    }
    for origin in ORIGINS:
        for role in (
            "collection",
            "plant",
            "incineration_market",
            "landfarming_market",
            "incineration_terminal",
            "landfarming_terminal",
        ):
            market = role.endswith("market")
            sludge = "incineration" if role.startswith("incineration") else "landfarming"
            product = (
                LEACHATE
                if role in {"collection", "plant"}
                else (INCINERATION if sludge == "incineration" else LANDFARMING)
            )
            name = (
                f"market for {product}"
                if market or role == "collection"
                else f"treatment of {product}, wastewater treatment"
                if role == "plant"
                else f"treatment of {product}"
            )
            construction = {
                "kind": "tps_sludge_market" if market else "project_recipe",
                "source_role": "native_ecoinvent_provider",
                "formulation_service_id": f"tps_{origin}",
                "source": {
                    "name": name,
                    "reference_product": product,
                    "unit": "cubic meter" if product == LEACHATE else "kilogram",
                    "exact_location": "RoW",
                },
            }
            if role == "collection":
                construction.update(
                    kind="tps_leachate_market", plant={"service_id": f"test_tps_plant_{origin}"}
                )
            elif role == "plant":
                construction.update(
                    kind="tps_wastewater_treatment",
                    incineration_reference_product=INCINERATION,
                    landfarming_reference_product=LANDFARMING,
                    incineration={"service_id": f"test_tps_incineration_market_{origin}"},
                    landfarming={"service_id": f"test_tps_landfarming_market_{origin}"},
                )
            elif market:
                construction.update(
                    sludge_fraction=sludge,
                    treatment={"service_id": f"test_tps_{sludge}_terminal_{origin}"},
                )
            else:
                construction["components"] = []
            service_id = f"test_tps_{role}_{origin}"
            catalogue["services"].append(
                {
                    "provider_service_id": service_id,
                    "name": service_id,
                    "reference_product": product,
                    "unit": construction["source"]["unit"],
                    "location": "RoW",
                    "stage_owner": "manufacturing_auxiliary_treatment",
                    "foreground_service": False,
                    "frontier_service": True,
                    "citation": method,
                    "construction": construction,
                }
            )
    return catalogue


def gas_boundary() -> _GasTreatment:
    """Supply a typed gas-owner output with independent deterministic test controls.

    Per principal kg, the test input contains 0.0004 kg methane, 0.01 kg O2,
    0.00002 m3 reaction water, and 0.8 MJ unbound heat.
    These synthetic controls check that inventory transfer applies once.
    """

    def row(key, amount, kind):
        """Represent an already-calculated deterministic boundary exchange."""
        return {"input": key, "type": kind, "amount": amount, "uncertainty type": 0, "loc": amount}

    return _GasTreatment(
        biosphere_exchanges=(
            row(flow_key("Methane, non-fossil", URBAN), 0.0004, "biosphere"),
            row(flow_key("Oxygen", AIR_RESOURCE), 0.01, "biosphere"),
            row(flow_key("Water", URBAN), 0.00002, "biosphere"),
        ),
        heat_air_mj=0.8,
        heat_soil_mj=0.0,
        internal_electricity_kwh=0.0,
        internal_heat_mj=0.0,
        trace={"boundary": "invented conditional gas output; no collection fraction"},
    )


def terminal_inventory(database: str, origin: str, *, future: bool) -> dict:
    """Use complete tiny external sludge terminals; their actual chemistry is tested elsewhere."""
    result = {}
    for role, product in (("incineration", INCINERATION), ("landfarming", LANDFARMING)):
        code = provider_service_code(f"test_tps_{role}_terminal_{origin}")
        result[database, code] = activity(
            database,
            code,
            f"invented complete {role}",
            product,
            "kilogram",
            -(17 if role == "incineration" else 19)
            if future
            else -(11 if role == "incineration" else 13),
            [exchange(flow_key("Carbon dioxide, fossil", RURAL), 0.02, "biosphere")],
        )
    return result
