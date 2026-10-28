"""Actual signed hydraulic owners retain source functions and remove sewage loads."""

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from water_treatment_fixtures import water_biosphere, water_native_inventory

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_source_closures,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext

# Doka GNIWWT source profiles with the fixture's fixed MAT10/MAP1000/EAT500.
# Independent equation evaluations: hydraulic kWh/m3, evaporated m3/m3 and
# urban emission share. Canada's rounded stage shares are normalized once.
_PROFILE_EXPECTED = {
    "row": (0.07529946518261003, 0.00023577882430111369, 0.7217062553373826),
    "ch": (0.07650225895009924, 0.0002361565703573985, 0.7518241277622788),
    "ca_qc": (0.07370938964385908, 0.00023127041206070294, 0.8674556028920686),
    "europe": (0.07622475593564804, 0.00023483667402689044, 0.8277305113123367),
}


@pytest.mark.parametrize("original", [True, False])
def test_signed_water_services_preserve_complete_hydraulics_and_dated_suppliers(
    tmp_path: Path, original: bool
) -> None:
    """Compile all four regional sources and solve each real collection/plant LCI.

    Plants produce -4 m3 and markets -2 m3, with a 25% treatment share.
    The solved plant demand must therefore be one sixteenth per m3 collected,
    while untouched sewers, screenings, sand and capital keep their full basis.
    The European fixture plant has no heat purchase, so the hydraulic service uses
    its declared complete external heater; the other three profiles retain their
    dated portfolios.
    """
    catalogue = load_provider_catalogue()
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], catalogue["services"])
        if str(row["provider_service_id"]).startswith(
            ("hydraulic_water_collection_", "hydraulic_water_treatment_")
        )
    }
    names = BrightwayNames(tmp_path / "brightway", "water", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(water_biosphere(), searchable=False)
        bd.Database("original").write(
            water_native_inventory(catalogue, "original", original=True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database("future").write(
                water_native_inventory(catalogue, "future", original=False), searchable=False
            )
        native_before = {
            node.key: [dict(ex) for ex in node.exchanges()]
            for name in {"original", database}
            for node in bd.Database(name)
        }
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        compiled = {
            ("providers", provider_service_code(sid)): _service_dataset(
                context, service, services, "providers"
            )
            for sid, service in services.items()
        }
        bd.Database("providers").write(compiled, searchable=False)
        for suffix in _PROFILE_EXPECTED:
            pid = "hydraulic_water_treatment_" + suffix
            mid = "hydraulic_water_collection_" + suffix
            plant = bd.get_node(database="providers", code=provider_service_code(pid))
            market = bd.get_node(database="providers", code=provider_service_code(mid))
            plant_production = next(iter(plant.production()))
            assert plant_production["amount"] == -4.0
            assert next(iter(market.production()))["amount"] == -2.0
            assert plant_production["properties"] == {
                "wet mass": {"amount": 1000.0, "unit": "kg"},
                "dry mass": {"amount": 0.0, "unit": "kg"},
            }
            amounts = {ex.input.key: ex["amount"] for ex in plant.technosphere()}
            assert {
                code: amounts[(database, code)]
                for code in ("limestone", "plant", "screenings", "sand")
            } == {"limestone": 0.008, "plant": 0.002, "screenings": -0.004, "sand": 0.006}
            assert not any((database, code) in amounts for code in ("iron", "pam", "sludge"))
            native_land = {
                ex["input"]: ex["amount"]
                for ex in native_before[(database, "plant_" + suffix)]
                if ex["input"][0] == "biosphere" and ex["input"][1].startswith("land_")
            }
            assert {
                ex.input.key: ex["amount"]
                for ex in plant.biosphere()
                if ex.input.key[1].startswith("land_")
            } == native_land
            assert {
                ex.input["name"]: ex.input["unit"]
                for ex in plant.biosphere()
                if ex.input.key[1].startswith("land_")
            } == {
                "Occupation, construction site": "square meter-year",
                "Occupation, industrial area": "square meter-year",
                "Occupation, urban/industrial fallow (non-use)": "square meter-year",
                "Transformation, from pasture, man made": "square meter",
                "Transformation, to industrial area": "square meter",
                "Transformation, to urban/industrial fallow (non-use)": "square meter",
            }
            assert not any(ex.input.key[1] in {"co2", "cod"} for ex in plant.biosphere())
            trace = next(
                row for row in context.technology_rows if row["provider_service_id"] == pid
            )
            terms = json.loads(cast(str, trace["project_terms_json"]))
            hydraulic, evaporation, urban = _PROFILE_EXPECTED[suffix]
            assert terms["electricity_kwh"] == pytest.approx(
                4 * hydraulic + 0.008 * 23375 / 13250, rel=2e-14
            )
            heat = 4 * 0.06893210265697261
            assert terms["delivered_heat_mj"] == pytest.approx(heat, rel=2e-14)
            share = 0.25 if original else 0.75
            assert amounts[(database, "electricity_a")] == pytest.approx(
                share * terms["electricity_kwh"], rel=2e-14
            )
            scaled = next(ex for ex in plant.technosphere() if ex.input.key[1] == "electricity_a")
            assert scaled["loc"] == pytest.approx(math.log(scaled["amount"]) + 0.03, rel=2e-14)
            assert scaled["scale"] == 0.2
            assert scaled["minimum"] == scaled["amount"] / 2
            assert scaled["maximum"] == scaled["amount"] * 2
            if suffix == "europe":
                assert amounts[(database, "heat_europe")] == pytest.approx(heat, rel=2e-14)
                assert (database, "heat_a") not in amounts
                assert (database, "heat_b") not in amounts
            else:
                assert amounts[(database, "heat_a")] == pytest.approx(heat / 4, rel=2e-14)
                assert amounts[(database, "heat_b")] == pytest.approx(heat * 3 / 4, rel=2e-14)
                assert (database, "heat_europe") not in amounts
            biosphere = {ex.input.key[1]: ex["amount"] for ex in plant.biosphere()}
            assert terms["evaporated_fraction"] == pytest.approx(evaporation, rel=2e-14)
            assert biosphere["water_urban"] == pytest.approx(4 * evaporation * urban, rel=2e-14)
            assert biosphere["water_rural"] == pytest.approx(
                4 * evaporation * (1 - urban), rel=2e-14
            )
            assert biosphere["heat_urban"] == pytest.approx(
                (3.6 * terms["electricity_kwh"] + heat) * urban, rel=2e-14
            )
            assert biosphere["surface"] + biosphere["water_urban"] + biosphere[
                "water_rural"
            ] == pytest.approx(4.0, rel=2e-14)
            assert biosphere["heat_urban"] + biosphere["heat_rural"] == pytest.approx(
                3.6 * terms["electricity_kwh"] + heat, rel=2e-14
            )
            assert biosphere["waste"] == 4000.0
            collection = {ex.input.key: ex["amount"] for ex in market.technosphere()}
            assert collection == {
                plant.key: -0.5,
                (database, "sewer_rural"): 0.02,
                (database, "sewer_urban"): 0.04,
            }
            assert [(ex.input.key[1], ex["amount"]) for ex in market.biosphere()] == [
                ("surface", 1.5)
            ]
            lca = LCA({market: -1.0})
            lca.lci()
            supply = cast(Any, lca.supply_array)
            assert supply[lca.dicts.activity[market.id]] == 0.5
            assert supply[lca.dicts.activity[plant.id]] == 0.0625
            co2 = bd.get_node(database="biosphere", code="co2")
            assert float(lca.inventory[lca.dicts.biosphere[co2.id], :].sum()) > 0.0
        for key, rows in native_before.items():
            assert [
                dict(ex) for ex in bd.get_node(database=key[0], code=key[1]).exchanges()
            ] == rows
