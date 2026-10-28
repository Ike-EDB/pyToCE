"""Invented complete chemical-residue sources extending the native HWI fixture."""

import math
from collections.abc import Generator
from contextlib import contextmanager
from math import log
from pathlib import Path
from typing import Any

from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from tests.unit.prepare_backgrounds.native_hwi_fixtures import (
    native_hwi_biosphere,
    native_hwi_inventory,
)


def apply_premise_chemical_receiving_metadata(bd: Any, database: str) -> None:
    """Give a disposable future receiver this fixture's missing-property and NEU heat metadata.

    Call after preparing the fixture's chemical support and before receiving.
    Only future native metadata changes; original calibration and all dated
    input quantities remain intact. These are real Brightway exchange writes.
    """
    # Remove fixture exchange properties to test receiving with original HWI
    # carbon and ash calibration.
    for code in ("mc_hwi", "mc_fixed", "mc_ash"):
        activity = bd.get_node(database=database, code=code)
        for exchange in activity.exchanges():
            if "properties" in exchange:
                del exchange["properties"]
                exchange.save()
    # Use an NEU industrial heat descriptor in this future fixture;
    # retain its linked heat input quantities.
    heat = bd.get_node(database=database, code="mc_ng")
    heat["name"] = "market for heat, district or industrial"
    heat["reference product"] = "heat, district or industrial"
    heat["location"] = "NEU"
    heat.save()


def chemical_residue_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Supply exact elementary semantics with invented IDs and no CFs."""
    data = native_hwi_biosphere()
    extra = [
        ("cr_nitrogen_air", "Nitrogen", ("air",)),
        ("cr_cyanide", "Cyanide", ("air", "urban air close to ground")),
        ("cr_so2", "Sulfur dioxide", ("air", "urban air close to ground")),
        ("cr_nickel_air", "Nickel II", ("air", "urban air close to ground")),
        ("cr_silicon_air", "Silicon", ("air", "urban air close to ground")),
        ("cr_phosphorus_air", "Phosphorus", ("air", "urban air close to ground")),
        ("cr_calcium_air", "Calcium II", ("air", "urban air close to ground")),
        ("cr_potassium_air", "Potassium I", ("air", "urban air close to ground")),
        ("cr_titanium_air", "Titanium ion", ("air", "urban air close to ground")),
        ("cr_aluminium_air", "Aluminium III", ("air", "urban air close to ground")),
        ("cr_magnesium_air", "Magnesium", ("air", "urban air close to ground")),
        ("cr_sulfate_air", "Sulfate", ("air", "urban air close to ground")),
        ("cr_lithium_surface", "Lithium I", ("water", "surface water")),
    ]
    for suffix, cats in (
        ("surface", ("water", "surface water")),
        ("long", ("water", "ground-, long-term")),
    ):
        extra += [
            (f"cr_{name}_{suffix}", name, cats)
            for name in (
                "Sulfate",
                "Iron ion",
                "Nickel II",
                "Silicon",
                "Phosphate",
                "Potassium I",
                "Aluminium III",
                "Magnesium",
            )
        ]
    for code, name, cats in extra:
        data[("biosphere", code)] = {
            "name": name,
            "unit": "kilogram",
            "categories": cats,
            "type": "emission",
        }
    return data


def chemical_residue_native_inventory(
    catalogue: dict[str, Any], database: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """Keep nonunit native reference amounts and distinguish collection from ash haul.

    Coefficients deliberately differ from the retained industrial capture.
    Actual HWI and water services are real source-backed constructions,
    while native suppliers below are small invented datasets.
    """
    data = native_hwi_inventory(catalogue, database, original=True)

    def row(code: str, quantity: float, basis: float, *, bio: bool = False) -> dict[str, Any]:
        """Retain a visible lognormal dose and an independent source basis."""
        amount = quantity * basis
        return {
            "input": ("biosphere" if bio else database, code),
            "type": "biosphere" if bio else "technosphere",
            "amount": amount,
            "uncertainty type": 2,
            "loc": math.log(amount) + 0.02,
            "scale": 0.17,
            "minimum": amount / 3,
            "maximum": amount * 3,
            "allocation": "invented chemical source",
        }

    for code, product in (
        ("cr_peroxide", "hydrogen peroxide, without water, in 50% solution state"),
        ("cr_ferric", "iron(III) chloride, without water, in 40% solution state"),
        ("cr_calcium_chloride", "calcium chloride"),
    ):
        data[(database, code)] = {
            "name": f"invented {code} supply",
            "reference product": product,
            "unit": "kilogram",
            "location": "GLO",
            "exchanges": [{"input": (database, code), "type": "production", "amount": 1.0}],
        }
    hwi = data[(database, "mc_hwi")]
    hwi["exchanges"] += [
        row("cr_peroxide", 0.004, 3),
        row("cr_ferric", 0.006, 3),
        row("cr_calcium_chloride", 0.008, 3),
        row("cr_cyanide", 0.000004, 3, bio=True),
        row("mc_nh3_air", 0.000007, 3, bio=True),
        row("mc_nox_air", 0.0001, 3, bio=True),
        row("cr_so2", 0.00002, 3, bio=True),
        row("cr_nickel_air", 0.000001, 3, bio=True),
        row("cr_silicon_air", 0.000002, 3, bio=True),
        row("cr_phosphorus_air", 0.000003, 3, bio=True),
        row("cr_Phosphate_surface", 0.0002, 3, bio=True),
        row("cr_Phosphate_long", 0.0003, 3, bio=True),
        row("cr_Nickel II_surface", 0.000004, 3, bio=True),
        row("cr_Nickel II_long", 0.000009, 3, bio=True),
        row("cr_Silicon_surface", 0.00003, 3, bio=True),
        row("cr_Silicon_long", 0.00008, 3, bio=True),
        row("mc_nitrate_short", 0.0001, 3, bio=True),
        row("mc_nitrate_long", 0.0003, 3, bio=True),
        row("cr_Sulfate_surface", 0.05, 3, bio=True),
        row("cr_Sulfate_long", 0.03, 3, bio=True),
    ]
    ash = data[(database, "mc_ash")]
    ash["exchanges"] += [
        row("cr_Sulfate_surface", 0.003, 5, bio=True),
        row("cr_Sulfate_long", 0.02, 5, bio=True),
        row("cr_Iron ion_surface", 0.001, 5, bio=True),
        row("cr_Iron ion_long", 0.003, 5, bio=True),
        row("cr_Nickel II_surface", 0.002, 5, bio=True),
        row("cr_Nickel II_long", 0.005, 5, bio=True),
        row("cr_Silicon_surface", 0.006, 5, bio=True),
        row("cr_Silicon_long", 0.03, 5, bio=True),
        # Positive native element releases deliberately differ from rounded
        # source-table fractions; actual ash must retain each row coefficient.
        row("cr_Aluminium III_surface", 0.000031, 5, bio=True),
        row("cr_Aluminium III_long", 0.018, 5, bio=True),
        row("cr_Potassium I_surface", 0.023, 5, bio=True),
        row("cr_Potassium I_long", 0.071, 5, bio=True),
        row("cr_Magnesium_surface", 0.000004, 5, bio=True),
        row("cr_Magnesium_long", 0.0027, 5, bio=True),
        # Invented kg PO4 per kg native dry ash. Its P content is separately
        # 0.025 kg/kg; the target must use that basis exactly once.
        row("cr_Phosphate_surface", 0.004, 5, bio=True),
        row("cr_Phosphate_long", 0.006, 5, bio=True),
    ]
    ash["exchanges"][0]["properties"]["sulfur content"] = {"amount": 0.012}
    ash["exchanges"][0]["properties"].update(
        {
            "iron content": {"amount": 0.04},
            "nickel content": {"amount": 0.02},
            "silicon content": {"amount": 0.03},
            "phosphorus content": {"amount": 0.025},
        }
    )
    for code, product in (
        ("cr_road", "transport, freight, lorry, unspecified"),
        ("cr_rail", "transport, freight, train, fleet average"),
    ):
        data[(database, code)] = {
            "name": "invented " + code,
            "reference product": product,
            "unit": "ton kilometer",
            "location": "CH",
            "exchanges": [{"input": (database, code), "type": "production", "amount": 1.0}],
        }
    data[(database, "cr_collection")] = {
        "name": "market for hazardous waste, for incineration",
        "reference product": "hazardous waste, for incineration",
        "unit": "kilogram",
        "location": "CH",
        "exchanges": [
            {"input": (database, "cr_collection"), "type": "production", "amount": -1.0},
            {"input": (database, "mc_hwi"), "type": "technosphere", "amount": -1.0},
            row("cr_road", 0.04, 1),
            row("cr_rail", 0.007, 1),
        ],
    }
    return data


def triethanolamine_native_inventory(
    catalogue: dict[str, Any], database: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """Add two invented nonunit TEA producers and their distinct native supply mixes."""
    data = chemical_residue_native_inventory(catalogue, database)

    def leaf(code: str, name: str, product: str, unit: str, location: str) -> None:
        """Add a complete small utility/feed supplier with an allocated production row."""
        data[(database, code)] = {
            "name": name,
            "reference product": product,
            "unit": unit,
            "location": location,
            "exchanges": [{"input": (database, code), "type": "production", "amount": 1.0}],
        }

    leaf(
        "tea_steam",
        "market for heat, from steam, in chemical industry",
        "heat, from steam, in chemical industry",
        "megajoule",
        "RER",
    )
    leaf("tea_water", "market group for tap water", "tap water", "kilogram", "RER")
    cooling = next(
        s for s in catalogue["services"] if s["provider_service_id"] == "ambient_cooling_operation"
    )["construction"]
    for component in cooling["components"]:
        selector = component["selector"]
        if component["component_id"] != "tap_water":
            leaf(
                "tea_cooling_power",
                selector["name"],
                selector["reference_product"],
                selector["unit"],
                "RER",
            )
    for region, basis, share in (("RER", 2.0, 0.5), ("RoW", 3.0, 0.2)):
        code = "tea_" + region
        leaf(code, "triethanolamine production", "triethanolamine", "kilogram", region)
        rows: list[dict[str, Any]] = []

        def dose(target: str, amount: float, *, bio: bool = False) -> dict[str, Any]:
            """Give each source exchange its own preserved uncertainty and formula."""
            value = amount * basis
            return {
                "input": ("biosphere" if bio else database, target),
                "type": "biosphere" if bio else "technosphere",
                "amount": value,
                "uncertainty type": 2 if value > 0 else 0,
                "loc": math.log(value) + 0.01 if value > 0 else value,
                "scale": 0.2,
                "minimum": value / 2,
                "maximum": value * 2,
                "formula": "3*44.053/(146.19*0.95)",
                "allocation": "invented TEA native allocation",
            }

        eo = 3 * 44.053 / (146.19 * 0.95)
        for suffix, fraction in (("a", share), ("b", 1 - share)):
            for product, unit in (
                ("ethylene oxide", "kilogram"),
                ("electricity, medium voltage", "kilowatt hour"),
            ):
                input_code = code + "_" + suffix + ("_eo" if unit == "kilogram" else "_power")
                leaf(input_code, "invented " + input_code, product, unit, region)
                rows.append(dose(input_code, fraction * (eo if unit == "kilogram" else 0.4)))
        leaf(code + "_n2", "invented nitrogen", "nitrogen, liquid", "kilogram", region)
        rows += [
            dose("mc_ammonia", 17.03 / (149.19 * 0.95)),
            dose("mc_factory", 4e-10),
            dose(code + "_n2", 0.019),
            dose("mc_ng", 2.0),
            dose("tea_steam", 0.2),
            dose("market_row", -0.00002),
            dose("tea_eo_air", 0.002 * eo, bio=True),
            dose("tea_eo_water", 0.004 * eo, bio=True),
            dose("tea_n2_air", 0.019, bio=True),
            dose("mc_nh3_air", 0.002 * 17.03 / (149.19 * 0.95), bio=True),
            dose("mc_co2_fossil", 0.08, bio=True),
            dose("mc_nitrate_short", 0.02, bio=True),
            dose("mc_water_air", 0.001, bio=True),
        ]
        data[(database, code)]["exchanges"] = [
            {"input": (database, code), "type": "production", "amount": basis},
            *rows,
        ]
        data[(database, code)]["comment"] = (
            "Invented allocated TEA production, original 146.19 EO denominator."
        )
        # Evaluated EcoSpold doses can omit an optional formula. Their native
        # amount/uncertainty remain sufficient for this source construction.
        next(r for r in rows if r["input"][1] == code + "_n2").pop("formula")
    return data


def triethanolamine_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Add native TEA air and retired water semantics to the invented flow set."""
    data = chemical_residue_biosphere()
    # Reuse the fixture's unspecified-air nitrogen flow under the TEA identifier.
    data[("biosphere", "tea_n2_air")] = data.pop(("biosphere", "cr_nitrogen_air"))
    for code, name, categories in (
        ("tea_eo_air", "Ethylene oxide", ("air", "urban air close to ground")),
        ("tea_eo_water", "Ethylene oxide", ("water", "surface water")),
    ):
        data[("biosphere", code)] = {
            "name": name,
            "unit": "kilogram",
            "categories": categories,
            "type": "emission",
        }
    return data


def _maintenance_inventory(
    catalogue: dict[str, Any], database: str, *, original: bool
) -> dict[tuple[str, str], dict[str, Any]]:
    """Add outgoing ash freight and distinguish the dated native reference units.

    All quantities are invented. The future terminal changes its two support
    suppliers while keeping the same chemistry and five-kg water duty.
    Its doubled source reference amounts must cancel during normalization.
    """
    data = chemical_residue_native_inventory(catalogue, database)
    ash = data[database, "mc_ash"]
    ash["exchanges"].append(
        {
            "input": (database, "cr_road"),
            "type": "technosphere",
            # 0.025 tonne-km per kg ash, on the fixture's five-kg production.
            "amount": 0.025 * 5,
            "loc": 0.025 * 5,
            "uncertainty type": 0,
        }
    )
    if not original:
        for code in ("mc_hwi", "mc_fixed", "mc_ash"):
            for row in data[database, code]["exchanges"]:
                factor = 2.0
                if code == "mc_ash" and row["input"][1] == "mc_ash_support_a":
                    factor *= 0.8 / 0.25
                elif code == "mc_ash" and row["input"][1] == "mc_ash_support_b":
                    factor *= 0.2 / 0.75
                row["amount"] *= factor
                if "loc" in row:
                    row["loc"] = (
                        row["loc"] + log(factor)
                        if row.get("uncertainty type") == 2
                        else row["loc"] * factor
                    )
                for bound in ("minimum", "maximum"):
                    if bound in row:
                        row[bound] *= factor
    return data


@contextmanager
def pa_maintenance_context(
    tmp_path: Path, *, original: bool
) -> Generator[tuple[Any, _RecipeContext, str]]:
    """Prepare the same small complete receiving graph for mixed and pure origins."""
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    names = BrightwayNames(tmp_path / "brightway", "pa_maintenance", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(chemical_residue_biosphere(), searchable=False)
        bd.Database("original").write(
            _maintenance_inventory(catalogue, "original", original=True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                _maintenance_inventory(catalogue, database, original=False), searchable=False
            )
        if not original:
            apply_premise_chemical_receiving_metadata(bd, database)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario=None if original else "remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database=database,
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        yield bd, context, database
