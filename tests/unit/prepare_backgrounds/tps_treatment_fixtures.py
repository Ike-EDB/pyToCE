"""Invented native inventories for the TPS sludge and ash compiler boundaries.

Descriptors reproduce the supported external semantics; amounts are deliberately
small invented controls, using synthetic source quantities. All quantities below
are per kg native reference unless their unit says otherwise. Original and
future inventories use different production magnitudes, utility mixes and
terminal support. The three packaged TPS formulations themselves are unchanged.
"""

from copy import deepcopy
from math import log

from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    load_provider_catalogue,
)

URBAN = ("air", "urban air close to ground")
SURFACE = ("water", "surface water")
LONG_WATER = ("water", "ground-, long-term")
AIR_RESOURCE = ("natural resource", "in air")
RURAL = ("air", "non-urban air or from high stacks")
GROUND = ("water", "ground-")
SOIL = ("soil", "agricultural")
ORIGINS = ("biogenic", "fossil", "atmospheric_dac")
SLUDGE = "sewage sludge, 70% water, WWT-SLF, municipal solid waste"
BOTTOM = "bottom ash, MSWI-WWT-SLF, municipal solid waste"
RESIDUAL = "residues, MSWI-WWT-SLF, municipal solid waste"
LANDFARM = "sewage sludge, 97% water, WWT-SLF, municipal solid waste"

# The synthetic dry assay assigns .78 to six major elements and .22/25 to each remaining element.
# Hg/Cd describe the municipal donor. Dry fractions are .3 for sludge and .8 for ash; these
# intensive properties retain their values across production references.
ELEMENT_NAMES = {
    "O": "oxygen",
    "H": "hydrogen",
    "C": "carbon",
    "S": "sulfur",
    "N": "nitrogen",
    "P": "phosphorus",
    "B": "boron",
    "Cl": "chlorine",
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
    "Si": "silicon",
    "Fe": "iron",
    "Ca": "calcium",
    "Al": "aluminium",
    "K": "potassium",
    "Mg": "magnesium",
    "Na": "sodium",
}
DRY_FRACTIONS = dict.fromkeys(ELEMENT_NAMES, 0.22 / 25)
DRY_FRACTIONS.update(C=0.4, H=0.05, O=0.25, N=0.04, S=0.02, P=0.02)

# Invented active-chemical kg/kg wet sludge; reheat is MJ/kg and facility is
# units/kg. Large water relative to NH3 makes the whole-meter dilution feasible.
# The prospective source triples doses to detect accidental dated-dose adoption.
SUPPLIES = {
    "NH3": ("ammonia, anhydrous, liquid", 0.0002, "kilogram"),
    "reheat": ("heat, district or industrial, natural gas", 0.06, "megajoule"),
    "TiO2": ("titanium dioxide", 0.00001, "kilogram"),
    "V2O5": ("chromium trioxide, flakes", 0.000002, "kilogram"),
    "PAM": ("chemical, inorganic", 0.000003, "kilogram"),
    "HCl": ("hydrochloric acid, without water, in 30% solution state", 0.0004, "kilogram"),
    "FeCl3": ("iron(III) chloride, without water, in 40% solution state", 0.0001, "kilogram"),
    "CaO": ("quicklime, milled, packed", 0.0003, "kilogram"),
    "NaOH": ("sodium hydroxide, without water, in 50% solution state", 0.0005, "kilogram"),
    "water": ("water, decarbonised", 0.8, "kilogram"),
    "facility": ("municipal waste incineration facility", 0.000002, "unit"),
}

# Independently listed identities check production bindings. Ash release coefficients include source
# stoichiometry.
IONS = {
    "Aluminium III": "Al",
    "Antimony ion": "Sb",
    "Arsenic ion": "As",
    "Barium II": "Ba",
    "Cadmium II": "Cd",
    "Calcium II": "Ca",
    "Chromium VI": "Cr",
    "Cobalt II": "Co",
    "Copper ion": "Cu",
    "Iron ion": "Fe",
    "Lead II": "Pb",
    "Magnesium": "Mg",
    "Manganese II": "Mn",
    "Mercury II": "Hg",
    "Molybdenum VI": "Mo",
    "Nickel II": "Ni",
    "Nitrate": "N",
    "Phosphate": "P",
    "Potassium I": "K",
    "Selenium IV": "Se",
    "Silicon": "Si",
    "Sodium I": "Na",
    "Sulfate": "S",
    "Tin ion": "Sn",
    "Vanadium V": "V",
    "Zinc II": "Zn",
}
INDICATORS = (
    "DOC, Dissolved Organic Carbon",
    "TOC, Total Organic Carbon",
    "COD, Chemical Oxygen Demand",
    "BOD5, Biological Oxygen Demand",
)
TRACE_ORGANICS = (
    "Benzene",
    "Benzo(a)pyrene",
    "Hexachlorobenzene",
    "Pentachlorobenzene",
    "Pentachlorophenol",
    "Toluene",
)
PROCESS = (
    *TRACE_ORGANICS,
    "NMVOC, non-methane volatile organic compounds",
    "Particulate Matter, < 2.5 um",
    "Particulate Matter, > 2.5 um and < 10um",
    "Dioxins, measured as 2,3,7,8-tetrachlorodibenzo-p-dioxin",
)


def flow_key(name: str, categories: tuple[str, ...]) -> tuple[str, str]:
    """Return an invented external flow code, separate from licensed UUIDs."""
    return "biosphere", name + "::" + "/".join(categories)


def biosphere_inventory() -> dict:
    """Supply native-named elementary flows with invented identifiers only."""
    descriptors = {(name, "kilogram", URBAN) for name in PROCESS}
    descriptors.update(
        (name, "kilogram", URBAN)
        for name in ("Ammonia", "Nitrogen oxides", "Dinitrogen monoxide", "Cyanide")
    )
    descriptors.update(
        (f"{gas}, {origin}", "kilogram", URBAN)
        for gas in ("Carbon dioxide", "Carbon monoxide", "Methane")
        for origin in ("fossil", "non-fossil")
    )
    descriptors.update(
        (name, "kilogram", compartment)
        for name in (
            *IONS,
            *INDICATORS,
            "Boron",
            "Titanium ion",
            "Chloride",
            "Organic carbon, placed in landfill",
        )
        for compartment in (SURFACE, LONG_WATER)
    )
    descriptors.update(
        (name, unit, compartment)
        for name, unit in (("Heat, waste", "megajoule"), ("Water", "cubic meter"))
        for compartment in (SURFACE, URBAN)
    )
    descriptors.add(("Oxygen", "kilogram", AIR_RESOURCE))
    descriptors.add(("Chromium III", "kilogram", SURFACE))
    descriptors.update(
        (name, "kilogram", SOIL) for name in (*IONS, "Boron", "Chromium III", "Sulfur")
    )
    descriptors.update(
        (name, "kilogram", RURAL)
        for name in (
            "Ammonia",
            "Nitrogen oxides",
            "Dinitrogen monoxide",
            "Carbon dioxide, fossil",
            "Carbon dioxide, non-fossil",
        )
    )
    descriptors.update(("Water", "cubic meter", compartment) for compartment in (RURAL, GROUND))
    descriptors.add(("Phosphate", "kilogram", GROUND))
    descriptors.add(("Heat, waste", "megajoule", SOIL))
    descriptors.update(
        (name, "kilogram", SURFACE)
        for name in (
            "Non-hazardous waste disposed",
            "Hazardous waste disposed",
            "Waste mass, total, placed in landfill",
        )
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
    """Give positive supply/releases a real lognormal distribution and metadata."""
    row = {
        "input": key,
        "amount": amount,
        "type": kind,
        "uncertainty type": 0,
        "loc": amount,
        "comment": "Invented external inventory control",
        "allocation": "cut-off control",
    }
    if amount > 0:
        row.update(
            {
                "uncertainty type": 2,
                "loc": log(amount) + 0.03,
                "scale": 0.21,
                "minimum": amount / 2,
                "maximum": amount * 2,
            }
        )
    return row


def _properties(dry: float) -> dict:
    """Return intensive invented source properties, with dry fractions summing to one."""
    return {
        "dry mass": {"amount": dry, "unit": "kg"},
        "water in wet mass": {"amount": 1 - dry, "unit": "kg"},
        "carbon allocation": {"amount": dry * 0.4, "unit": "kg"},
        **{
            f"{ELEMENT_NAMES[el]} content": {"amount": q, "unit": "dimensionless"}
            for el, q in DRY_FRACTIONS.items()
        },
    }


def _activity(
    database: str,
    code: str,
    name: str,
    product: str,
    unit: str,
    production: float,
    rows: list[dict],
    *,
    dry: float | None = None,
) -> dict:
    """Scale extensive exchanges to p while keeping product properties intensive."""
    key = database, code
    p = exchange(key, production, "production")
    p.update(
        name=product,
        unit=unit,
        flow="invented reference identity",
        activity="invented source activity",
        classifications=[("test", "source")],
    )
    if dry is not None:
        p["properties"] = _properties(dry)
    scaled = []
    for row in rows:
        copy = deepcopy(row)
        copy["amount"] *= abs(production)
        if copy["uncertainty type"] == 2:
            copy["loc"] += log(abs(production))
            copy["minimum"] *= abs(production)
            copy["maximum"] *= abs(production)
        else:
            copy["loc"] = copy["amount"]
        scaled.append(copy)
    return {
        "name": name,
        "reference product": product,
        "unit": unit,
        "location": "GLO" if code == "actual_pam" else "RoW",
        "type": "process",
        "exchanges": [p, *scaled],
    }


def native_inventory(database: str, *, future: bool) -> dict:
    """Build complete tiny native plants, suppliers, ash markets and terminals.

    Sludge p=-2/-5, ash terminals p=-3/-7 and markets p=-4/-6 exercise
    independent signed normalization. Future support intensities double and
    chemical group totals triple, while supplier shares change .25/.75 to .7/.3.
    Original trace gas releases are 1e-10 kg/kg; ash releases are 1e-7/2e-7 kg/kg
    for short/additional-long terms. Future releases are twice these values.
    None is a copied native coefficient.
    """
    result = {}
    carbon = flow_key("Carbon dioxide, fossil", URBAN)
    for duty, (product, _, unit) in SUPPLIES.items():
        # The future fixture changes the industrial-heat supplier product label.
        # The reheat duty retains its original nitrogen-based calculation.
        if future and duty == "reheat":
            product = "heat, district or industrial"
        for suffix in ("a", "b"):
            code = f"{duty}_{suffix}"
            result[database, code] = _activity(
                database,
                code,
                f"Invented {duty} {suffix}",
                product,
                unit,
                1,
                [exchange(carbon, 0.02, "biosphere")],
            )
    leaves = {
        "actual_pam": ("market for polyacrylamide", "polyacrylamide", "kilogram", 1),
        "tmt": ("Invented obsolete TMT", "triazine-compound", "kilogram", 1),
        "freight": ("Invented lorry", "transport, freight, lorry", "ton kilometer", 1),
        "support": ("Invented disposal support", "landfill support", "unit", 1),
        "cement": ("Invented cement", "cement", "kilogram", 1),
        "hydration": ("Invented hydration water", "tap water", "kilogram", 1),
        "hydrated": ("Invented hydrated cement fate", "hydrated cement", "kilogram", -1),
        "spreading": ("Invented vacuum tanker spreading", "slurry spreading", "cubic meter", 1),
    }
    for code, (name, product, unit, p) in leaves.items():
        result[database, code] = _activity(
            database, code, name, product, unit, p, [exchange(carbon, 0.02, "biosphere")]
        )
    factor = 2 if future else 1
    for code, product, terminal in (
        ("bottom", BOTTOM, "slag compartment"),
        ("residual", RESIDUAL, "residual material landfill"),
    ):
        rows = [exchange((database, "support"), 0.02 * factor)]
        if code == "residual":
            # Invented native cement, water and negative-production disposal
            # rows exercise unchanged source retention at the ash owner.
            rows += [
                exchange((database, "cement"), 0.4),
                exchange((database, "hydration"), 0.6),
                exchange((database, "hydrated"), -0.8),
            ]
        for name in (*IONS, "Boron", *INDICATORS, "Organic carbon, placed in landfill"):
            for compartment, q in ((SURFACE, 1e-7), (LONG_WATER, 2e-7)):
                rows.append(exchange(flow_key(name, compartment), q * factor, "biosphere"))
        for name in (
            "Non-hazardous waste disposed",
            "Hazardous waste disposed",
            "Waste mass, total, placed in landfill",
        ):
            rows.append(exchange(flow_key(name, SURFACE), 0.3 * factor, "biosphere"))
        result[database, code] = _activity(
            database,
            code,
            f"treatment of {product}, {terminal}",
            product,
            "kilogram",
            -7 if future else -3,
            rows,
            dry=0.8,
        )
        result[database, f"{code}_market"] = _activity(
            database,
            f"{code}_market",
            f"market for {product}",
            product,
            "kilogram",
            -6 if future else -4,
            [exchange((database, code), -1), exchange((database, "freight"), 0.05 * factor)],
            dry=0.8,
        )
    rows = []
    shares = (0.7, 0.3) if future else (0.25, 0.75)
    for duty, (_, dose, _) in SUPPLIES.items():
        for suffix, share in zip(("a", "b"), shares, strict=True):
            rows.append(
                exchange((database, f"{duty}_{suffix}"), dose * share * (3 if future else 1))
            )
    rows += [
        exchange((database, "tmt"), 0.00001),
        exchange((database, "bottom_market"), -0.02),
        exchange((database, "residual_market"), -0.01),
    ]
    for name in (*PROCESS, "Ammonia", "Nitrogen oxides", "Dinitrogen monoxide", "Cyanide"):
        rows.append(exchange(flow_key(name, URBAN), 1e-10 * factor, "biosphere"))
    for gas in ("Carbon dioxide", "Carbon monoxide", "Methane"):
        for origin in ("fossil", "non-fossil"):
            rows.append(exchange(flow_key(f"{gas}, {origin}", URBAN), 1e-10 * factor, "biosphere"))
    for original_name in (*IONS, *INDICATORS):
        name = "Chromium III" if original_name == "Chromium VI" else original_name
        rows.append(exchange(flow_key(name, SURFACE), 1e-7 * factor, "biosphere"))
    rows.append(exchange(flow_key("Non-hazardous waste disposed", SURFACE), 0.3, "biosphere"))
    for compartment, heat in ((URBAN, 3), (SURFACE, 1)):
        rows += [
            exchange(flow_key("Heat, waste", compartment), heat, "biosphere"),
            exchange(flow_key("Water", compartment), 0.0001, "biosphere"),
        ]
    rows.append(exchange(flow_key("Oxygen", AIR_RESOURCE), 0.04, "biosphere"))
    result[database, "sludge"] = _activity(
        database,
        "sludge",
        f"treatment of {SLUDGE}, municipal incineration",
        SLUDGE,
        "kilogram",
        -5 if future else -2,
        rows,
        dry=0.3,
    )
    # Agricultural donor: .03 kg dry matter/kg wet sludge. Invented local N
    # fates are 10% each to four gases/ions, P 10% each to surface/ground water,
    # and 80/20 moisture evaporation/drainage. Other soil rows are gross
    # deposition. The 99 MJ donor heat intentionally conflicts with bulk HHV.
    rows = [exchange((database, "spreading"), factor / 1030)]
    for name, molecular, nitrogen in (
        ("Ammonia", 17, 14),
        ("Dinitrogen monoxide", 44, 28),
        ("Nitrogen oxides", 46, 14),
    ):
        rows.append(
            exchange(flow_key(name, RURAL), 0.1 * 0.03 * 0.04 * molecular / nitrogen, "biosphere")
        )
    rows.append(exchange(flow_key("Nitrate", SURFACE), 0.1 * 0.03 * 0.04 * 62 / 14, "biosphere"))
    for compartment in (SURFACE, GROUND):
        rows.append(
            exchange(flow_key("Phosphate", compartment), 0.1 * 0.03 * 0.02 * 95 / 31, "biosphere")
        )
    for name, element in IONS.items():
        if element in {"N", "P"}:
            continue
        name = {"Chromium VI": "Chromium III", "Sulfate": "Sulfur"}.get(name, name)
        rows.append(exchange(flow_key(name, SOIL), 0.03 * DRY_FRACTIONS[element], "biosphere"))
    for origin, share in (("fossil", 0.3), ("non-fossil", 0.7)):
        rows.append(
            exchange(
                flow_key(f"Carbon dioxide, {origin}", RURAL),
                0.03 * 0.4 * 44 / 12 * share,
                "biosphere",
            )
        )
    rows += [
        exchange(flow_key("Water", RURAL), 0.97 * 0.8 / 1000, "biosphere"),
        exchange(flow_key("Water", GROUND), 0.97 * 0.2 / 1000, "biosphere"),
        exchange(flow_key("Heat, waste", SOIL), 99, "biosphere"),
        exchange(flow_key("Non-hazardous waste disposed", SURFACE), 1, "biosphere"),
    ]
    result[database, "landfarming"] = _activity(
        database,
        "landfarming",
        f"treatment of {LANDFARM}, landfarming",
        LANDFARM,
        "kilogram",
        -11 if future else -8,
        rows,
        dry=0.03,
    )
    return result


def treatment_catalogue() -> dict:
    """Declare direct treatment owners using the invented native source graph."""
    catalogue = load_provider_catalogue()
    registry = catalogue["method_authority_registry"]
    method = "invented_tps_treatment_boundary"
    registry["exact_source_bindings"][method] = ["ecoinvent_3_12_cutoff"]
    registry["exact_method_authorities"][method] = {
        **registry["project_recipe_defaults"],
        "limitation": "Synthetic inventories for direct treatment tests.",
    }
    declarations = []
    for origin in ORIGINS:
        declarations.append(
            {
                "provider_service_id": f"tps_sludge_landfarming_{origin}",
                "name": "TPS landfarming",
                "reference_product": LANDFARM,
                "unit": "kilogram",
                "location": "RoW",
                "stage_owner": "manufacturing_auxiliary_treatment",
                "citation": method,
                "foreground_service": False,
                "frontier_service": True,
                "construction": {
                    "kind": "tps_sludge_landfarming",
                    "carbon_templates": {"fossil": "fossil", "non-fossil": "non-fossil"},
                    "source_role": "native_ecoinvent_provider",
                    "formulation_service_id": f"tps_{origin}",
                    "source": {
                        "name": f"treatment of {LANDFARM}, landfarming",
                        "reference_product": LANDFARM,
                        "unit": "kilogram",
                        "exact_location": "RoW",
                    },
                },
            }
        )
        for role, product in (
            ("sludge_incineration", SLUDGE),
            ("bottom_ash_market", BOTTOM),
            ("bottom_ash_treatment", BOTTOM),
            ("residual_ash_market", RESIDUAL),
            ("residual_ash_treatment", RESIDUAL),
        ):
            service_id = f"tps_{role}_{origin}"
            market = role.endswith("market")
            fraction = "bottom" if role.startswith("bottom") else "residual"
            terminal = "slag compartment" if fraction == "bottom" else "residual material landfill"
            name = f"market for {product}" if market else f"treatment of {product}, {terminal}"
            construction = {
                "kind": "tps_ash_market" if market else "tps_ash_treatment",
                "source_role": "native_ecoinvent_provider",
                "formulation_service_id": f"tps_{origin}",
                "source": {
                    "name": name,
                    "reference_product": product,
                    "unit": "kilogram",
                    "exact_location": "RoW",
                },
                "ash_fraction": fraction,
            }
            if role == "sludge_incineration":
                construction.update(
                    kind="tps_sludge_incineration",
                    additional_apc_water_flows=[],
                    carbon_templates={"fossil": "fossil", "non-fossil": "non-fossil"},
                    source={
                        "name": f"treatment of {SLUDGE}, municipal incineration",
                        "reference_product": SLUDGE,
                        "unit": "kilogram",
                        "exact_location": "RoW",
                    },
                    flocculant={
                        "name": "market for polyacrylamide",
                        "reference_product": "polyacrylamide",
                        "unit": "kilogram",
                        "exact_location": "GLO",
                    },
                    bottom_ash={"service_id": f"tps_bottom_ash_market_{origin}"},
                    residual_ash={"service_id": f"tps_residual_ash_market_{origin}"},
                )
            elif market:
                construction["treatment"] = {"service_id": f"tps_{fraction}_ash_treatment_{origin}"}
            declarations.append(
                {
                    "provider_service_id": service_id,
                    "name": service_id,
                    "reference_product": product,
                    "unit": "kilogram",
                    "location": "RoW",
                    "stage_owner": "manufacturing_auxiliary_treatment",
                    "citation": method,
                    "foreground_service": False,
                    "frontier_service": True,
                    "construction": construction,
                }
            )
    catalogue["services"].extend(declarations)
    return catalogue
