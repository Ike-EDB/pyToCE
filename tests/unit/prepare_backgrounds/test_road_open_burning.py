"""Actual road binder, minerals and wet bead coating through uncontrolled burning."""

import json
from math import fsum
from typing import Any, cast

import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from pa_open_burning_fixtures import (
    _AIR,
    _INDICATOR,
    _SOIL,
    open_burning_biosphere,
    open_burning_native_inventory,
)

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
from pytoce.run_lca.calculation.carbon import direct_treatment_carbon_flows

_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")


@pytest.mark.parametrize("future", (False, True), ids=("original", "future"))
def test_road_burning_keeps_actual_minerals_and_only_organic_carbon(tmp_path, future: bool) -> None:
    """Solve all origins with source factors once on the actual retained wet load."""
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    services = {
        row["provider_service_id"]: row
        for row in catalogue["services"]
        if row["construction"]["kind"] == "road_open_burning"
    }
    assert set(services) == {f"road_{origin}_open_burning" for origin in _ORIGINS}
    names = BrightwayNames(tmp_path / "brightway", "road_burning", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        biosphere = open_burning_biosphere("biosphere")
        # Invented keys, admitted native descriptors: existing PA flows already
        # contain soil S/Na. Bulk road minerals add these finite soil endpoints;
        # only the PDMS-bound Si has the source airborne partition.
        for index, (name, compartment) in enumerate(
            [
                (name, _SOIL)
                for name in (
                    "Titanium ion",
                    "Silicon",
                    "Calcium II",
                    "Aluminium III",
                    "Potassium I",
                    "Magnesium",
                )
            ]
            + [("Silicon", _AIR)]
        ):
            biosphere["biosphere", f"road_{index}"] = {
                "name": name,
                "unit": "kilogram",
                "categories": compartment,
                "type": "emission",
            }
        bd.Database("biosphere").write(biosphere, searchable=False)
        for database in ("original", "future"):
            bd.Database(database).write(
                open_burning_native_inventory(
                    database, original=database == "original", biosphere_database="biosphere"
                ),
                searchable=False,
            )
        before = {
            node.key: [dict(row) for row in node.exchanges()]
            for database in ("original", "future")
            for node in bd.Database(database)
        }
        dated = _ActivityResolver(bd, "future" if future else "original")
        year = 2050 if future else 2025
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=year,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                row["method_authority_id"]: row for row in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        bd.Database("providers").write(
            {
                ("providers", provider_service_code(sid)): _service_dataset(
                    context, service, services, "providers"
                )
                for sid, service in services.items()
            },
            searchable=False,
        )

        # Silikal/DBC retained premix after container loss and MMA release:
        # 0.21875 kg principal remains from one kg premix + 0.010 kg hardener.
        wet = 4.561737142857143
        bead = 0.2475 / 0.21875
        water = bead * 0.144 / 500.240
        pdms = bead * 0.060 / 500.240
        auxiliary_c = 0.00495 / 0.21875 * (168 / 242 + 240 / 330) + pdms * 24 / 74.085
        organic_c = 0.6 + auxiliary_c
        geological_c = 0.1485 / 0.21875 * 12 / 100.078
        titanium = 0.07425 / 0.21875 * 47.867 / 79.867
        glass = bead * 500 / 500.240
        potassium = glass * 0.3 / 100.07 * 78.1966 / 94.1966
        pdms_si = pdms * 28.085 / 74.085
        for origin in _ORIGINS:
            sid = f"road_{origin}_open_burning"
            node = bd.get_node(database="providers", code=provider_service_code(sid))
            assert node.rp_exchange()["amount"] == 1
            assert list(node.technosphere()) == []
            trace = next(
                row
                for row in context.semantic_rows
                if row["provider_service_id"] == sid
                and row["ledger_entry_type"] == "service_construction"
            )
            terms = json.loads(trace["project_terms_json"])
            assert terms["original_source_key"] == ["original", "burning"]
            assert terms["wet_mass_kg_per_principal_kg"] == pytest.approx(wet)
            assert terms["incoming_water_kg"] == pytest.approx(water)
            assert terms["geological_carbon_kg"] == pytest.approx(geological_c)
            assert terms["source_rounding_kg"] == pytest.approx(0, abs=2e-15)
            assert terms["input_elements_kg"]["C"] == pytest.approx(organic_c)
            assert terms["auxiliary_carbon_kg"] == pytest.approx({"fossil": auxiliary_c})
            rows = {
                (row.input["name"], tuple(row.input["categories"])): row for row in node.biosphere()
            }
            assert rows["Titanium ion", _SOIL]["amount"] == pytest.approx(titanium)
            assert rows["Potassium I", _SOIL]["amount"] == pytest.approx(potassium)
            assert ("Titanium ion", _AIR) not in rows
            assert ("Potassium I", _AIR) not in rows
            assert rows["Silicon", _AIR]["amount"] == pytest.approx(
                pdms_si * (1 - 999.99532744142095 / 1000)
            )
            assert rows["Carbon", _SOIL]["amount"] == pytest.approx(
                organic_c * 5.0150854170955386 / 1000
            )
            assert (
                rows["Organic carbon, placed in landfill", _INDICATOR]["amount"]
                == (rows["Carbon", _SOIL]["amount"])
            )
            assert rows["Non-hazardous waste disposed", _INDICATOR]["amount"] == pytest.approx(wet)
            for name, factor in (("Carbon monoxide", 0.0386), ("Methane", 0.0059)):
                assert fsum(
                    row["amount"] for key, row in rows.items() if key[0].startswith(name + ",")
                ) == pytest.approx(wet * factor)

            demand = 2.5
            lca = LCA({node: demand})
            lca.lci()
            activations = pd.DataFrame(
                [
                    {
                        "year": 2050,
                        "region": "EUR",
                        "life_cycle_stage": "mismanaged_open_burning_treatment",
                        "material": "Others",
                        "constituent": "road_marking_coatings",
                        "product": "polymer",
                        "virgin_feedstock_route": origin,
                        "good": "all",
                        "material_cycle_source": "virgin_feedstock",
                        "ultimate_carbon_origin": origin,
                        "lca_fate": "open_burning",
                        "fate_origin": "remind_mfa",
                        "provider_database": "providers",
                        "provider_code": node.key[1],
                        "provider_anchor_year": year,
                        "amount": demand,
                        "mfa_carbon_amount_kg": demand * 0.6,
                    }
                ]
            )
            diagnostic = direct_treatment_carbon_flows(bd, activations)
            assert set(diagnostic.mfa_carbon_origin) == {origin}
            assert set(diagnostic.origin_mapping_status) == {"actual_road_principal_origin"}
            for species, name, c in (
                ("carbon dioxide", "Carbon dioxide", 12 / 44),
                ("carbon monoxide", "Carbon monoxide", 12 / 28),
                ("methane", "Methane", 12 / 16),
            ):
                gases = [row for key, row in rows.items() if key[0].startswith(name + ",")]
                total = fsum(row["amount"] for row in gases)
                assert fsum(
                    diagnostic.loc[diagnostic.carbon_species.eq(species)].carbon_amount
                ) == (pytest.approx(demand * total * c * 0.6 / organic_c))
                for row in gases:
                    assert float(lca.inventory[lca.dicts.biosphere[row.input.id], :].sum()) == (
                        pytest.approx(demand * row["amount"], rel=2e-7)
                    )
        assert before == {
            node.key: [dict(row) for row in node.exchanges()]
            for database in ("original", "future")
            for node in bd.Database(database)
        }
