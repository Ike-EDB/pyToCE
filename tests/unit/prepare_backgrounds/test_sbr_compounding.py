"""Compound-22 air releases and complete supplied functions through the compiler."""

from csv import DictReader
from math import fsum, log
from pathlib import Path
from typing import Any, cast

import pytest
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import _process

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import catalogue as catalogue_owner
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


@pytest.mark.parametrize("year", (2025, 2050))
def test_sbr_inputs_and_air_use_their_actual_material_bases(tmp_path: Path, year: int) -> None:
    """Compile SBR air, pulse supply, bags and dated nonunit supplied functions.

    All supplier inventories are invented. Their production of two units and
    direct fossil CO2 of 0.4/0.6 kg make normalization and future selection
    observable. The waste treatments instead produce -4 kg, emitting 1.6/2.4 kg
    CO2 and 0.008/0.012 kg SO2. These are invented signed, nonunit controls,
    with synthetic emission factors. Real packaged recipe data and its parsed EcoSpold
    material-balance calculation determine the SBR quantities. The fixed CSV was independently
    derived from EPA 2008 compound-22 Mixing X/Extrude F/Hot Air E occurrences,
    the original TRC Volume II I.1-43 five-metal vector, and the admitted POM
    phase reference. The CSV supplies independent expected emissions and native flow names. All
    fixture flows are synthetic nodes.
    """
    fixture = Path(__file__).parents[2] / "fixtures" / "sbr_compounding" / "air_emissions.csv"
    with fixture.open(encoding="utf-8", newline="") as stream:
        air_per_gross_compound = {
            row["flow_name"]: fsum(
                float(row[f"{stage}_kg_per_gross_compound"])
                for stage in ("mixing", "extrusion", "hot_air")
            )
            for row in DictReader(stream)
        }
    catalogue = catalogue_owner.load_provider_catalogue()
    services = {
        str(s["provider_service_id"]): s for s in cast(list[dict[str, Any]], catalogue["services"])
    }
    service = services["others_esbr_compounding_conversion"]
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="sbr_productive",
        biosphere_database="biosphere",
        source_database="original",
    )
    # Independent native names protect exact selectors from catalog-derived
    # fixture masking. Ordinary sulfur/electricity use their admitted premise
    # geography; the complete delivered-air function remains RER at both dates.
    supplied = (
        (
            "scrap_lorry",
            "market for transport, freight, lorry, unspecified",
            "transport, freight, lorry, diesel, unspecified",
            "ton kilometer",
            "RoW",
        ),
        ("carbon_black", "market for carbon black", "carbon black", "kilogram", "GLO"),
        ("zinc", "market for zinc oxide", "zinc oxide", "kilogram", "GLO"),
        ("stearic", "market for stearic acid", "stearic acid", "kilogram", "GLO"),
        (
            "organic",
            "market for chemical, organic, basic precursors and materials",
            "chemical, organic, basic precursors and materials",
            "kilogram",
            "GLO",
        ),
        ("wax", "market for paraffin", "paraffin", "kilogram", "GLO"),
        ("oil", "market for base oil", "base oil", "kilogram", "GLO"),
        ("sulfur", "market for sulfur", "sulfur", "kilogram", "GLO"),
        (
            "electricity",
            "market group for electricity, medium voltage",
            "electricity, medium voltage",
            "kilowatt hour",
            "RER",
        ),
        (
            "compressed_air",
            "market for compressed air, 600 kPa gauge",
            "compressed air, 600 kPa gauge",
            "cubic meter",
            "RER",
        ),
        (
            "air_electricity",
            "market group for electricity, low voltage",
            "electricity, low voltage",
            "kilowatt hour",
            "RER",
        ),
        (
            "other_air_electricity",
            "market group for electricity, low voltage",
            "electricity, low voltage",
            "kilowatt hour",
            "CH",
        ),
        (
            "machinery_oil",
            "market for lubricating oil",
            "lubricating oil",
            "kilogram",
            "RER",
        ),
        (
            "spent_oil",
            "treatment of waste mineral oil, hazardous waste incineration",
            "waste mineral oil",
            "kilogram",
            "CH",
        ),
        (
            "oil_lorry",
            "market for transport, freight, lorry, unspecified",
            "transport, freight, lorry, diesel, unspecified",
            "ton kilometer",
            "RER",
        ),
        (
            "oil_rail",
            "market for transport, freight, train, fleet average",
            "transport, freight, train, fleet average",
            "ton kilometer",
            "CH",
        ),
        (
            "bag_felt",
            "market for textile, nonwoven polyester",
            "textile, nonwoven polyester",
            "kilogram",
            "GLO",
        ),
        (
            "spent_bag",
            "treatment of waste polyethylene terephthalate, municipal incineration",
            "waste polyethylene terephthalate",
            "kilogram",
            "GLO",
        ),
        (
            "bag_offcut",
            "market group for waste polyethylene terephthalate",
            "waste polyethylene terephthalate",
            "kilogram",
            "RER",
        ),
        (
            "bag_gas_heat",
            "market group for heat, district or industrial, natural gas",
            "heat, district or industrial, natural gas",
            "megajoule",
            "RER",
        ),
        (
            "bag_other_heat",
            "market group for heat, district or industrial, other than natural gas",
            "heat, district or industrial, other than natural gas",
            "megajoule",
            "RER",
        ),
        (
            "bag_factory",
            "plastic processing factory construction",
            "plastic processing factory",
            "unit",
            "RER",
        ),
        (
            "delivery_barge",
            "market group for transport, freight, inland waterways, barge, diesel",
            "transport, freight, inland waterways, barge, diesel",
            "ton kilometer",
            "GLO",
        ),
        (
            "delivery_van",
            "market group for transport, freight, light commercial vehicle, fleet average",
            "transport, freight, light commercial vehicle, fleet average",
            "ton kilometer",
            "GLO",
        ),
        (
            "delivery_lorry",
            "market group for transport, freight, lorry, diesel, unspecified",
            "transport, freight, lorry, diesel, unspecified",
            "ton kilometer",
            "GLO",
        ),
        (
            "delivery_ship",
            "market for transport, freight, sea, container ship, heavy fuel oil",
            "transport, freight, sea, container ship, heavy fuel oil",
            "ton kilometer",
            "GLO",
        ),
        (
            "delivery_rail",
            "market group for transport, freight, train, fleet average",
            "transport, freight, train, fleet average",
            "ton kilometer",
            "GLO",
        ),
    )
    # Synthetic donors each produce two filter units. Distinct amounts and distributions check
    # retained duties and supplier records.
    fabrication_inputs = {
        "electricity": 0.44,
        "bag_gas_heat": 1.2,
        "bag_other_heat": 1.6,
        "bag_factory": 4e-8,
        "bag_offcut": -0.01,
    }
    delivery_inputs = {
        "delivery_barge": 0.002,
        "delivery_van": 0.006,
        "delivery_lorry": 0.1,
        "delivery_ship": 0.2,
        "delivery_rail": 0.04,
    }
    # Invented unequal air-market purchases, nonunit compressor production,
    # distinct LV suppliers and varying electricity coefficients expose wrong
    # averaging, normalization, voltage, date, or repeated complete air supply.
    air_suppliers = (
        ("air_best", 0.2, 2, "air_electricity", 0.5, 0.03, 0.1),
        ("air_optimized", 0.6, 3, "other_air_electricity", 0.9, 0.06, 0.2),
        ("air_average", 1.2, 4, "air_electricity", 1.6, 0.09, 0.3),
    )
    removed_inputs = {
        "donor_fleece": (
            "polar fleece production, energy use only",
            "polar fleece, energy use only",
        ),
        "donor_pet": (
            "market for polyethylene terephthalate, granulate, amorphous",
            "polyethylene terephthalate, granulate, amorphous",
        ),
        "donor_rolling": ("sheet rolling, steel", "sheet rolling, steel"),
        "donor_low_alloy_steel": ("market for steel, low-alloyed", "steel, low-alloyed"),
        "donor_steel": ("market for steel, unalloyed", "steel, unalloyed"),
        "donor_zinc_coat": ("zinc coating, coils", "zinc coat, coils"),
    }
    source_operation_records = {}
    with selected_brightway_project(names) as bd:
        from bw2calc import LCA

        bd.Database("biosphere").write(
            {
                ("biosphere", "co2"): {
                    "name": "Carbon dioxide, fossil",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                },
                ("biosphere", "so2"): {
                    "name": "Sulfur dioxide",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                },
                **{
                    ("biosphere", name): {
                        "name": name,
                        "unit": "kilogram",
                        "categories": ("air",),
                        "type": "emission",
                    }
                    for name in air_per_gross_compound
                },
            },
            searchable=False,
        )
        for database, dated in (("original", False), ("future", True)):
            nodes = {}
            for code, name, product, unit, location in supplied:
                if dated and code == "sulfur":
                    location = "World"
                if dated and code in {"electricity", "air_electricity"}:
                    location = "EUR"
                is_treatment = code in {
                    "spent_oil",
                    "spent_bag",
                    "bag_offcut",
                }
                emissions: list[dict[str, object]] = [
                    {
                        "input": ("biosphere", "co2"),
                        "amount": (0.6 if dated else 0.4) * (4 if is_treatment else 1),
                        "type": "biosphere",
                        "uncertainty type": 0,
                    }
                ]
                if is_treatment:
                    emissions.append(
                        {
                            "input": ("biosphere", "so2"),
                            "amount": 0.012 if dated else 0.008,
                            "type": "biosphere",
                            "uncertainty type": 0,
                        }
                    )
                nodes[database, code] = _process(
                    database,
                    code,
                    name,
                    product,
                    location=location,
                    production_amount=-4 if is_treatment else 2,
                    exchanges=emissions,
                )
                nodes[database, code]["unit"] = unit
            for code, purchase, production, power_code, power, oil, direct_co2 in air_suppliers:
                dated_factor = 1.5 if dated else 1
                nodes[database, code] = _process(
                    database,
                    code,
                    code,
                    "compressed air, 600 kPa gauge",
                    location="RER",
                    production_amount=production,
                    exchanges=[
                        {
                            "input": (database, power_code),
                            "type": "technosphere",
                            "amount": power * dated_factor,
                            "uncertainty type": 2,
                            "loc": log(power * dated_factor),
                            "scale": 0.23,
                        },
                        {
                            "input": (database, "machinery_oil"),
                            "type": "technosphere",
                            "amount": oil,
                        },
                        {"input": (database, "spent_oil"), "type": "technosphere", "amount": -oil},
                        {"input": ("biosphere", "co2"), "type": "biosphere", "amount": direct_co2},
                    ],
                )
                nodes[database, code]["unit"] = "cubic meter"
                nodes[database, "compressed_air"]["exchanges"].append(
                    {"input": (database, code), "type": "technosphere", "amount": purchase}
                )
            for code, (name, product) in removed_inputs.items():
                nodes[database, code] = _process(database, code, name, product)
                if code == "donor_zinc_coat":
                    nodes[database, code]["unit"] = "square meter"
            for code, location in (("donor_filter", "RER"), ("donor_other_filter", "RoW")):
                nodes[database, code] = _process(
                    database,
                    code,
                    "air filter production, central unit, 600 m3/h",
                    "air filter, central unit, 600 m3/h",
                    location=location,
                    production_amount=2,
                )
                nodes[database, code]["unit"] = "unit"
            for donor_code, retained in (
                ("donor_filter", fabrication_inputs),
                ("donor_market", delivery_inputs),
            ):
                donor_records = [
                    {
                        "input": (database, code),
                        "type": "technosphere",
                        "amount": amount * (1.5 if dated else 1),
                        "uncertainty type": 2,
                        "loc": log(abs(amount * (1.5 if dated else 1))),
                        "scale": 0.17,
                        "negative": amount < 0,
                        "comment": "Invented retained donor exchange",
                    }
                    for code, amount in retained.items()
                ]
                source_operation_records[database, donor_code] = [
                    dict(row) for row in donor_records
                ]
                if donor_code == "donor_filter":
                    nodes[database, donor_code]["exchanges"].extend(donor_records)
                    nodes[database, donor_code]["exchanges"].extend(
                        {"input": (database, code), "type": "technosphere", "amount": 100}
                        for code in removed_inputs
                    )
                else:
                    nodes[database, donor_code] = _process(
                        database,
                        donor_code,
                        "market for air filter, central unit, 600 m3/h",
                        "air filter, central unit, 600 m3/h",
                        location="GLO",
                        production_amount=2,
                        exchanges=donor_records
                        + [
                            {"input": (database, code), "type": "technosphere", "amount": 1}
                            for code in ("donor_filter", "donor_other_filter")
                        ],
                    )
                    nodes[database, donor_code]["unit"] = "unit"
            bd.Database(database).write(nodes, searchable=False)
        original = _ActivityResolver(bd, "original")
        database = "original" if year == 2025 else "future"
        ordinary = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=year,
            provider_database="compiled",
            source=original,
            ordinary=ordinary,
            technology=ordinary,
            biosphere=_FlowResolver(bd, "biosphere"),
            material_recipe=physical_material_recipe(),
            catalogue=catalogue,
            method_authorities={
                str(row["method_authority_id"]): row
                for row in catalogue_owner._method_authority_rows(catalogue)
            },
            complete_source_ids=catalogue_owner._service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        compiled = cast(dict[str, Any], _service_dataset(context, service, services, "compiled"))
        bag_children: dict[str, dict[str, Any]] = {
            # The complete composed-landfill chain has its own real linked
            # test. This small function isolates parent mass and the 136 km haul.
            "scrap": _process(
                "compiled",
                catalogue_owner.provider_service_code("sbr_filled_fossil_landfill"),
                "Invented complete filled scrap landfill",
                "filled SBR scrap",
                production_amount=-4,
                exchanges=[
                    {
                        "input": ("biosphere", "co2"),
                        "type": "biosphere",
                        "amount": 1.6 if year == 2025 else 2.4,
                    },
                    {
                        "input": ("biosphere", "so2"),
                        "type": "biosphere",
                        "amount": 0.008 if year == 2025 else 0.012,
                    },
                ],
            ),
            # The direct dust owner is independently exercised with real
            # source chemistry in test_sbr_dust_treatment. This small signed
            # child isolates compounding's mass/haul attachment boundary.
            "dust": _process(
                "compiled",
                catalogue_owner.provider_service_code("sbr_captured_mixer_dust_incineration"),
                "Invented complete captured-dust treatment",
                "captured SBR mixer dust",
                production_amount=-4,
                exchanges=[
                    {
                        "input": ("biosphere", "co2"),
                        "type": "biosphere",
                        "amount": 1.6 if year == 2025 else 2.4,
                    },
                    {
                        "input": ("biosphere", "so2"),
                        "type": "biosphere",
                        "amount": 0.008 if year == 2025 else 0.012,
                    },
                ],
            ),
            "fabrication": _service_dataset(
                context, services["others_esbr_filter_bag_fabrication"], services, "compiled"
            ),
            "delivery": _service_dataset(
                context, services["others_esbr_filter_bag_delivery"], services, "compiled"
            ),
        }
        child_keys = {
            name: next(row["input"] for row in child["exchanges"] if row["type"] == "production")
            for name, child in bag_children.items()
        }
        for name, donor in (("fabrication", "donor_filter"), ("delivery", "donor_market")):
            records = [
                row for row in bag_children[name]["exchanges"] if row["type"] == "technosphere"
            ]
            assert records == source_operation_records[database, donor]
            (child_production,) = [
                row for row in bag_children[name]["exchanges"] if row["type"] == "production"
            ]
            assert child_production["amount"] == 2
        # The independent principal reference is from TRC/EPA monomer factors,
        # corrected 2.66e-5 kg/kg extrusion PM, and Carli's 5% post-release trim.
        styrene = 1.858 * fsum((1.4786094907418637e-6, 3.9285095052594246e-8, 4.505773049721235e-7))
        butadiene = 1.858 * fsum((1.3054683728346342e-7, 7.82694259043066e-8))
        feed = 1 / (0.95 * (1 - styrene - butadiene - 0.0000266))
        # Compound 22: nine auxiliary charges total .858 kg/kg gross SBR.
        # The three distinct specialty chemicals share one complete supplier.
        expected = {
            "carbon_black": 0.585,
            "zinc": 0.1,
            "stearic": 0.02,
            "organic": 0.02 + 0.01 + 0.018,
            "wax": 0.015,
            "oil": 0.07,
            "sulfur": 0.02,
        }
        expected = {code: amount * feed for code, amount in expected.items()}
        dust_kg = 0.00045 * 0.993 * 1.858 * feed
        expected[child_keys["dust"][1]] = -dust_kg
        # Carli Table 15 uses 7.1 MJ/kg gross compound; EPA's fan design uses
        # .0152278920236 kWh/kg finished filled compound. Apply the fan dose
        # to the retained compound mass after all declared releases.
        finished_compound = 1.8468991701714175
        scrap_kg = 0.09720521948270619
        expected[child_keys["scrap"][1]] = -scrap_kg
        expected["scrap_lorry"] = 0.136 * scrap_kg
        # Independently admitted EPA collector coefficient per kg finished
        # compound; productive Carli machinery remains on gross throughput.
        fan_per_finished_compound = 0.015227892023639535
        expected["electricity"] = (
            7.1 / 3.6 * 1.858 * feed + fan_per_finished_compound * finished_compound
        )
        # RER pipe-extrusion maintenance: kg lubricant/kg gross compound.
        # CH waste-oil market rows 834544-834545 supply only collection tkm/kg
        # oil; the full negative-production HWI replaces its disposal mix.
        lubricant = 0.000143 * 1.858 * feed
        expected["machinery_oil"] = lubricant
        expected["spent_oil"] = -lubricant
        expected["oil_lorry"] = 0.02577 * lubricant
        expected["oil_rail"] = 0.00356 * lubricant
        expected["oil_lorry"] += 0.0258 * dust_kg
        expected["oil_rail"] += 0.0036 * dust_kg
        # Independent retained-piece areas from the admitted Donaldson/INDRO
        # construction: body .7924507968 m2 + disc .01843107700906449 m2.
        # EPA collector area / Donaldson active area / 3 years / annual output
        # gives replacement frequency, then actual retained compound P applies.
        bag_mass = (0.7924507968 + 0.01843107700906449) * 0.5424919595981168
        replacement_rate = 295.2755905511811 / (0.706063104 * 3 * 1_700_000)
        bag_kg = bag_mass * replacement_rate * finished_compound
        expected[child_keys["fabrication"][1]] = bag_kg / 0.05
        expected[child_keys["delivery"][1]] = bag_kg / 0.4
        expected["bag_felt"] = 1.05 * bag_kg
        expected["spent_bag"] = -bag_kg
        expected["oil_lorry"] += 0.02577 * bag_kg
        expected["oil_rail"] += 0.00356 * bag_kg
        expected["bag_factory"] = finished_compound * 7.407407407407407e-10
        # The EPA exhaust/pulse transfer supplies this normal m3 quantity per
        # kg finished compound; the complete native operating chain stays below.
        delivered_air = 0.029485817473476825 * finished_compound
        expected["compressed_air"] = delivered_air
        actual = {
            row["input"][1]: row["amount"]
            for row in compiled["exchanges"]
            if row["type"] == "technosphere"
        }
        assert actual == pytest.approx(expected, rel=1e-12, abs=0)
        assert all(
            row["input"][0] == ("compiled" if row["input"] in child_keys.values() else database)
            for row in compiled["exchanges"]
            if row["type"] == "technosphere"
        )
        # The vector checks named releases, .007 mixer dust/metals, .00363 hot-air POM, and
        # phase-unlisted NMVOC. Named and particle descriptors describe overlapping releases. Eight
        # unmatched substances retain their source descriptions.
        expected_biosphere = {
            name: amount * 1.858 * feed for name, amount in air_per_gross_compound.items()
        }
        actual_biosphere = {
            row["input"][1]: row["amount"]
            for row in compiled["exchanges"]
            if row["type"] == "biosphere"
        }
        assert actual_biosphere == pytest.approx(expected_biosphere, rel=1e-12, abs=0)
        (production,) = [row for row in compiled["exchanges"] if row["type"] == "production"]
        assert production["amount"] == 1
        assert "sbr_compounding" == service["construction"]["kind"]
        assert context.semantic_rows[0]["semantic_class"] == "service_normalization"
        assert (
            "trc_1995_sbr_compound22"
            in context.complete_source_ids["others_esbr_compounding_conversion"]
        )
        assert (
            "ecoinvent_sbr_productive_lubrication"
            in context.complete_source_ids["others_esbr_compounding_conversion"]
        )
        # Solve the compiled inventory with nonunit production and complete retained supplier
        # emissions.
        key = production["input"]
        bd.Database("compiled").write(
            {key: compiled, **{child_keys[name]: child for name, child in bag_children.items()}},
            searchable=False,
        )
        lca = LCA({bd.get_node(database=key[0], code=key[1]): 1})
        lca.lci()
        # Positive suppliers emit 0.2/0.3 kg CO2 per unit. Negative-production
        # treatment adds 0.4/0.6 kg CO2 and 0.002/0.003 kg SO2 per kg waste.
        # These independent fixture ratios expose a wrong sign or a second
        # division by its -4 kg reference while retaining the whole treatment.
        child_codes = {child_key[1] for child_key in child_keys.values()}
        positive_supply = fsum(
            amount for code, amount in expected.items() if amount > 0 and code not in child_codes
        )
        # Synthetic child operations produce two units. Retain their specified inputs and negative
        # offcut; the adapted recipe omits frame/raw-fabric and finished-filter purchases.
        dated_factor = 1.5 if year == 2050 else 1
        positive_supply += (
            dated_factor
            / 2
            * (
                bag_kg / 0.05 * fsum(v for v in fabrication_inputs.values() if v > 0)
                + bag_kg / 0.4 * fsum(delivery_inputs.values())
            )
        )
        offcut_kg = bag_kg / 0.05 * 0.01 * dated_factor / 2
        # The complete native compressed-air service must still carry its
        # electricity, oil supply, oil fate and direct process emissions once.
        # Actual market purchases .2/.6/1.2 have production of 2 and the three
        # compressors have production of 2/3/4, preserving their unequal mix.
        positive_supply += (
            delivered_air
            / 2
            * fsum(
                purchase / production * (power * dated_factor + oil)
                for _, purchase, production, _, power, oil, _ in air_suppliers
            )
        )
        compressed_air_oil = (
            delivered_air
            / 2
            * fsum(
                purchase / production * oil
                for _, purchase, production, _, _, oil, _ in air_suppliers
            )
        )
        compressor_emissions = (
            delivered_air
            / 2
            * fsum(
                purchase / production * direct_co2
                for _, purchase, production, _, _, _, direct_co2 in air_suppliers
            )
        )
        # All synthetic waste treatments share each pollutant's factor per kg solids/oil.
        treated_units = lubricant + dust_kg + scrap_kg + bag_kg + offcut_kg + compressed_air_oil
        expected_biosphere["co2"] = (
            positive_supply * (0.2 if year == 2025 else 0.3)
            + treated_units * (0.4 if year == 2025 else 0.6)
            + compressor_emissions
        )
        expected_biosphere["so2"] = treated_units * (0.002 if year == 2025 else 0.003)
        inventory = {
            flow.key[1]: float(lca.inventory[lca.dicts.biosphere[flow.id], :].sum())
            for flow in bd.Database("biosphere")
        }
        # Brightway's dictionary processing stores exchange amounts as float32;
        # allow that precision in this solved LCI, including multiplied supplier
        # coefficients. The unprocessed compiled vectors above stay at 1e-12.
        assert inventory == pytest.approx(expected_biosphere, rel=2e-7, abs=0)
