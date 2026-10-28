"""Synthetic Swiss treatment inventories for package and Brightway tests.

The reusable fixture supplies source metadata, uncertainty, nonunit production,
support and ash disposal. This module adds Swiss product descriptors,
peroxide/hydroxide quantities and Ti flows; no licensed inventories are used.
"""

from copy import deepcopy

from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _CH_WASTEWATER_PROFILE,
    _LeachateCarbon,
    _LeachateLoad,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pdo_manufacture import _pdo_reference
from tests.unit.prepare_backgrounds.tps_treatment_fixtures import (
    BOTTOM as OLD_BOTTOM,
)
from tests.unit.prepare_backgrounds.tps_treatment_fixtures import (
    LONG_WATER,
    SURFACE,
    URBAN,
    _activity,
    exchange,
    flow_key,
    treatment_catalogue,
)
from tests.unit.prepare_backgrounds.tps_treatment_fixtures import (
    RESIDUAL as OLD_RESIDUAL,
)
from tests.unit.prepare_backgrounds.tps_treatment_fixtures import (
    SLUDGE as OLD_SLUDGE,
)
from tests.unit.prepare_backgrounds.tps_treatment_fixtures import (
    biosphere_inventory as base_biosphere,
)
from tests.unit.prepare_backgrounds.tps_treatment_fixtures import (
    native_inventory as base_inventory,
)

SLUDGE = "sewage sludge, 70% water, WWT, WW, average"
BOTTOM = "bottom ash, MSWI[F]-WWT, WW, average"
RESIDUAL = "residues, MSWI[F]-WWT, WW, average"


def biosphere_inventory():
    """Add Swiss phosphorus/sulfur and Ti/Cl flow descriptors to synthetic records."""
    records = base_biosphere()
    for name, categories in (
        ("Phosphorus", URBAN),
        ("Sulfur dioxide", URBAN),
        ("Iron ion", URBAN),
        ("Sodium I", URBAN),
        ("Cobalt II", URBAN),
        ("Calcium II", URBAN),
        ("Potassium I", URBAN),
        ("Magnesium", URBAN),
        ("Chloride", SURFACE),
        ("Titanium ion", SURFACE),
        ("Titanium ion", LONG_WATER),
    ):
        records[flow_key(name, categories)] = {
            "name": name,
            "unit": "kilogram",
            "categories": categories,
            "type": "emission",
        }
    return records


def native_inventory(database, *, future):
    """Create synthetic Swiss plants with different original/future production amounts."""
    records = base_inventory(database, future=future)
    renames = {OLD_SLUDGE: SLUDGE, OLD_BOTTOM: BOTTOM, OLD_RESIDUAL: RESIDUAL}
    for record in records.values():
        for field in ("name", "reference product"):
            for old, new in renames.items():
                record[field] = record[field].replace(old, new)
        if record["location"] == "RoW":
            record["location"] = "CH"
    plant = records[database, "sludge"]
    plant["name"] += " FAE"
    # The synthetic original Swiss plant buys no lime. A changed sludge input
    # cannot scale a nonexistent source row; package compilation must select
    # the separate complete RER quicklime market.
    plant["exchanges"] = [
        row for row in plant["exchanges"] if row["input"][1] not in {"CaO_a", "CaO_b"}
    ]
    lime = deepcopy(records[database, "CaO_a"])
    lime.update(name="market for quicklime, milled, packed", location="RER")
    lime["exchanges"][0]["input"] = database, "packed_lime"
    records[database, "packed_lime"] = lime
    magnitude = 5 if future else 2
    carbon = flow_key("Carbon dioxide, fossil", URBAN)
    for suffix, share in (("a", 0.7 if future else 0.25), ("b", 0.3 if future else 0.75)):
        records[database, f"peroxide_{suffix}"] = _activity(
            database,
            f"peroxide_{suffix}",
            f"Invented peroxide {suffix}",
            "hydrogen peroxide, without water, in 50% solution state",
            "kilogram",
            1,
            [exchange(carbon, 0.02, "biosphere")],
        )
        plant["exchanges"].append(
            exchange(
                (database, f"peroxide_{suffix}"), magnitude * 0.0006 * share * (3 if future else 1)
            )
        )
    records[database, "hydroxide"] = _activity(
        database,
        "hydroxide",
        "Invented mixed-metal cut-off",
        "metalliferous hydroxide sludge",
        "kilogram",
        1,
        [],
    )
    plant["exchanges"].append(exchange((database, "hydroxide"), -magnitude * 0.003))
    for name in (
        "Phosphorus",
        "Sulfur dioxide",
        "Iron ion",
        "Sodium I",
        "Cobalt II",
        "Calcium II",
        "Potassium I",
        "Magnesium",
    ):
        plant["exchanges"].append(exchange(flow_key(name, URBAN), magnitude * 1e-8, "biosphere"))
    records[database, "freight"]["location"] = "RER"
    # The synthetic Swiss source contains Ti flows. Subtract the added Ti dry
    # fraction from carbon so the synthetic ash composition still totals one.
    for fraction in ("bottom", "residual"):
        node = records[database, fraction]
        production = node["exchanges"][0]
        production["properties"]["titanium content"] = {"amount": 0.001, "unit": "dimensionless"}
        production["properties"]["carbon content"]["amount"] -= 0.001
        for categories, amount in ((SURFACE, 1e-8), (LONG_WATER, 2e-8)):
            node["exchanges"].append(
                exchange(
                    flow_key("Titanium ion", categories),
                    abs(production["amount"]) * amount,
                    "biosphere",
                )
            )
    return records


def catalogue():
    """Declare treatment services for package compilation and source admission."""
    result = treatment_catalogue()
    old = {x["provider_service_id"]: x for x in result["services"]}
    names = {
        "tps_sludge_incineration_fossil": "swiss_sludge",
        "tps_bottom_ash_market_fossil": "swiss_bottom_market",
        "tps_bottom_ash_treatment_fossil": "swiss_bottom_terminal",
        "tps_residual_ash_market_fossil": "swiss_residual_market",
        "tps_residual_ash_treatment_fossil": "swiss_residual_terminal",
    }
    for previous, new in names.items():
        service = deepcopy(old[previous])
        service["provider_service_id"] = new
        service["location"] = "CH"
        con = service["construction"]
        con.pop("formulation_service_id", None)
        for field in ("reference_product", "name"):
            for donor, target in (
                (OLD_SLUDGE, SLUDGE),
                (OLD_BOTTOM, BOTTOM),
                (OLD_RESIDUAL, RESIDUAL),
            ):
                service[field] = service[field].replace(donor, target)
        for field in ("name", "reference_product"):
            for donor, target in (
                (OLD_SLUDGE, SLUDGE),
                (OLD_BOTTOM, BOTTOM),
                (OLD_RESIDUAL, RESIDUAL),
            ):
                con["source"][field] = con["source"][field].replace(donor, target)
        con["source"]["exact_location"] = "CH"
        if new == "swiss_sludge":
            con["kind"] = "swiss_sludge_incineration"
            con["source"]["name"] += " FAE"
            con["bottom_ash"] = {"service_id": "swiss_bottom_market"}
            con["residual_ash"] = {"service_id": "swiss_residual_market"}
        elif new.endswith("market"):
            con["treatment"] = {"service_id": names[con["treatment"]["service_id"]]}
        result["services"].append(service)
    return result


def wastewater(case):
    """Calculate Swiss wastewater treatment from PA6/PDO or synthetic acetone inputs.

    These are package-constructed inlet quantities, not native exchange vectors.
    The PA6 carbon labels are supplied controls for testing origin bookkeeping.
    """
    if case == "pdo":
        reference = _pdo_reference("fossil")
        return _wastewater_load(
            reference.load,
            volume_m3=reference.solution_volume_m3,
            incoming_water_kg=reference.water_kg,
            evaporation_fraction=0.0,
            profile=_CH_WASTEWATER_PROFILE,
        )
    if case == "finish":
        # Retained seven-ingredient PA6 reference per kg dry principal:
        # collected fraction .0571223984097466; add .000071 kg caprolactam
        # at nominal CHNO/113.
        elements = {
            "C": 0.000608081881600144,
            "H": 0.00010074945554805862,
            "O": 0.00011017587242293554,
            "N": 1.002191024548823e-5,
            "P": 1.9909535435986513e-6,
            "S": 2.8238783848686604e-6,
            "Na": 2.0249629614132736e-6,
        }
        carbon = (
            _LeachateCarbon("principal_biogenic", True, "biogenic", 0.5 * elements["C"]),
            _LeachateCarbon("principal_DAC", True, "atmospheric_dac", 0.3 * elements["C"]),
            _LeachateCarbon("fixed_fossil_finish", False, "fossil", 0.2 * elements["C"]),
        )
        # Use the fixture's collected solution, water and COD inputs; the water
        # value excludes the dry collected material from a 0.9 kg solution.
        volume, water, cod = 0.0009, 0.8991641310852935, 0.0023
        bod = 0.91 * elements["C"]
    else:
        # Synthetic inlet: 0.01319616 kg acetone and 0.181021790436203 kg water, with a separate
        # hydraulic volume. Source molecular weights determine acetone C/H/O quantities.
        elements = {
            "C": 8.186935834710743 / 1000,
            "H": 1.3741455867768595 / 1000,
            "O": 3.6350785785123962 / 1000,
        }
        carbon = (_LeachateCarbon("acetone", False, "fossil", elements["C"]),)
        volume = 0.197725790436203 / 1000
        water = 181.02179043620298 / 1000
        cod = 32 / 12 * elements["C"] + 8 * elements["H"] - elements["O"]
        # Original raw municipal BOD/TOC descriptor.
        bod = 1.7586552173908316 * elements["C"]
    load = _LeachateLoad(
        elements,
        carbon,
        0.1225 * elements["C"],
        0.64 * elements["C"],
        elements["C"],
        0.6114130434782609 * elements["C"],
        cod,
        bod,
    )
    return _wastewater_load(
        load,
        volume_m3=volume,
        incoming_water_kg=water,
        evaporation_fraction=0.0,
        profile=_CH_WASTEWATER_PROFILE,
    )
