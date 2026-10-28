"""Protect actual PA concentrate supply, delivery and homogeneous-residue receiving."""

import json
from math import fsum, isfinite, log
from pathlib import Path
from typing import Any, cast

import pytest
from bw2calc import LCA
from chemical_residue_fixtures import apply_premise_chemical_receiving_metadata
from chemical_steam_fixtures import chemical_steam_biosphere, chemical_steam_native_inventory
from material_recipe_fixtures import physical_material_recipe
from pa_finish_concentrate_fixtures import concentrate_biosphere, concentrate_native_inventory

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
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


@pytest.mark.parametrize("original", [True, False])
def test_concentrate_keeps_dated_whole_operations_and_actual_fitted_cargo(
    tmp_path: Path, original: bool
) -> None:
    """Test PA concentrate mixing, packaging, delivery and residue treatment.

    Ingredient quantities and manufacturing inventories have separate tests.
    This test checks dry mixing, filled-container freight and signed demands
    to the residue and ash treatment inventories.
    """
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, Any]], catalogue["services"])
    }
    names = BrightwayNames(tmp_path / "brightway", "pa_concentrate", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {**chemical_steam_biosphere(), **concentrate_biosphere()}, searchable=False
        )
        bd.Database("original").write(
            {
                **chemical_steam_native_inventory(catalogue, "original"),
                **concentrate_native_inventory(catalogue, "original", original=True),
            },
            searchable=False,
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                {
                    **chemical_steam_native_inventory(catalogue, database),
                    **concentrate_native_inventory(catalogue, database, original=False),
                },
                searchable=False,
            )
        before = {
            node.key: [dict(row) for row in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
            if node.key[1].startswith(("pac_", "drum_"))
        }
        if not original:
            apply_premise_chemical_receiving_metadata(bd, database)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario=None if original else "remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="frontier",
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
        records = {
            ("frontier", provider_service_code(service_id)): _service_dataset(
                context, services[service_id], services, "frontier"
            )
            for service_id in (
                "hydraulic_water_collection_row",
                "hydraulic_water_treatment_row",
                "pa_finish_container",
            )
        }
        container_key = "frontier", provider_service_code("pa_finish_container")
        for grade, finish in (("pa6", 0.014), ("pa66", 0.0095)):
            sid = grade + "_finish_concentrate"
            key = "frontier", provider_service_code(sid)
            hkey = "frontier", provider_service_code(grade + "_finish_blending_treatment")
            akey = "frontier", provider_service_code(grade + "_finish_blending_treatment_ash")
            concentrate = _service_dataset(context, services[sid], services, "frontier")
            for suffix in ("finish_blending_treatment", "finish_blending_treatment_ash"):
                receiver_id = grade + "_" + suffix
                records["frontier", provider_service_code(receiver_id)] = _service_dataset(
                    context, services[receiver_id], services, "frontier"
                )
            rows = cast(list[dict[str, Any]], concentrate["exchanges"])
            trace = next(
                row
                for row in context.technology_rows
                if row["provider_service_id"] == sid
                and row["component_id"] == "complete_concentrate_mixing_operation"
            )
            terms = json.loads(str(trace["project_terms_json"]))
            reference = terms["pa_finish_reference"]

            # Seven dry fractions sum to one; FAS carrier water is added to the delivered payload.
            dry = finish
            water = finish * 0.03 * (1 / 0.28 - 1)
            payload = dry + water
            drum_count = payload / 100
            installed_parts = {
                "hdpe_body": 21 * 0.45359237,
                "hdpe_closures": 6 * (0.45359237 / 16),
                "technical_seals": 0.45359237 / 16,
            }
            tare = sum(installed_parts.values()) * drum_count
            assert records[container_key]["installed_parts_kg"] == installed_parts
            assert records[container_key]["fitted_tare_kg"] * drum_count == tare
            assert reference["dry_concentrate_kg"] == pytest.approx(dry)
            assert reference["supplier_water_kg"] == pytest.approx(water)
            assert rows[0] == {"input": key, "type": "production", "amount": 1.0}
            inputs = {row["input"]: row for row in rows if row["type"] == "technosphere"}
            expected = {
                ("frontier", provider_service_code(ingredient)): finish * share * 1.01683
                for ingredient, share in (
                    ("pa_tridecyl_stearate", 0.60),
                    ("pa_diglycerol_dioleate", 0.05),
                    ("pa_oleth10", 0.17),
                    ("pa_peg40_hco", 0.06),
                    ("pa_tea_monooleyl_phosphate", 0.04),
                    ("fatty_alcohol_sulfate_28_percent", 0.03),
                    ("pa_hmda_dimeramide", 0.05),
                )
            }
            expected.update(
                {
                    (database, "pac_power"): dry * (0.13 if original else 0.21),
                    (database, "pac_factory"): dry * (0.002 if original else 0.003),
                    (database, "pac_nitrogen"): dry * (0.03 if original else 0.04),
                    container_key: drum_count,
                    hkey: -0.01683 * payload,
                }
            )
            expected.update(
                {
                    (database, "pac_filled_" + mode): (payload + tare)
                    * dose
                    * (1 if original else 0.7)
                    for mode, dose in (("road", 0.083), ("rail", 0.012), ("ship", 0.031))
                }
            )
            # Exactly seven ingredient purchases; co-delivered water,
            # donor paint/lubricant feeds and empty delivery add no rows.
            assert {key: row["amount"] for key, row in inputs.items()} == pytest.approx(expected)
            assert len([row for row in rows if row["type"] == "technosphere"]) == 15
            emissions = [row for row in rows if row["type"] == "biosphere"]
            assert len(emissions) == 1
            assert emissions[0]["input"] == ("biosphere", "cr_nitrogen_air")
            assert emissions[0]["amount"] == pytest.approx(dry * (0.03 if original else 0.04))
            for row in [
                inputs[database, code]
                for code in (
                    "pac_power",
                    "pac_factory",
                    "pac_nitrogen",
                    "pac_filled_road",
                    "pac_filled_rail",
                    "pac_filled_ship",
                )
            ] + emissions:
                assert row["loc"] == pytest.approx(log(abs(row["amount"])) + 0.04)
                assert row["scale"] == 0.18
                assert row["minimum"] == pytest.approx(row["amount"] / 4)
                assert row["maximum"] == pytest.approx(row["amount"] * 4)
                assert row["allocation"] == "invented concentrate operation"
            assert terms["whole_drum_receiving_reuse_cycle_kg"] == pytest.approx(tare)
            assert terms["delivered_fitted_tare_kg"] == pytest.approx(tare)
            assert terms["whole_drum_parts_kg"] == pytest.approx(
                {name: amount * drum_count for name, amount in installed_parts.items()}
            )
            removed = json.loads(str(trace["removed_technosphere_json"]))
            assert {row["code"] for row in removed} == {"pac_paint_feed", "pac_paint_waste"}
            assert json.loads(str(trace["removed_biosphere_json"]))[0]["code"] == "mc_co2_fossil"

            receiving = cast(dict[str, Any], records[hkey]["chemical_residue_reference"])
            load = receiving["load"]
            material = reference["blending_residue"]
            elements = material["elements_kg"]
            assert load["active_elements_kg"] == elements
            assert all(elements[element] > 0 for element in ("C", "N", "P", "S", "Na"))
            assert 0 < load["biogenic_carbon_kg"] < elements["C"]
            assert load["biogenic_carbon_kg"] == material["biogenic_carbon_kg"]
            assert load["carrier_water_kg"] == pytest.approx(0.01683 * water)
            assert load["wet_mass_kg"] == pytest.approx(0.01683 * payload)
            assert fsum(elements.values()) + load["carrier_water_kg"] == pytest.approx(
                load["wet_mass_kg"]
            )
            assert load["net_energy_mj"] == pytest.approx(
                34.016 * elements["C"]
                + (124.265 - 9 * 2.4417) * elements["H"]
                - 9.8324 * elements["O"]
                + 6.276 * elements["N"]
                + 19.079 * elements["S"]
                - 2.4417 * 0.01683 * water
            )
            assert load["waste_formed_water_kg"] == pytest.approx(
                elements["H"] * 18.015 / (2 * 1.008)
            )
            assert load["nonfuel_salt_elements_kg"] == {}
            assert cast(list[dict[str, Any]], records[hkey]["exchanges"])[0]["amount"] == -1
            assert cast(list[dict[str, Any]], records[akey]["exchanges"])[0]["amount"] == -1

        # Solve the receiving graph. The seven ingredient references are
        # checked separately above.
        bd.Database("frontier").write(records, searchable=False)
        for grade in ("pa6", "pa66"):
            hkey = "frontier", provider_service_code(grade + "_finish_blending_treatment")
            akey = "frontier", provider_service_code(grade + "_finish_blending_treatment_ash")
            reference = cast(dict[str, Any], records[hkey]["chemical_residue_reference"])
            wet = reference["load"]["wet_mass_kg"]
            node = bd.get_node(database="frontier", code=hkey[1])
            lca = LCA({node: -wet})
            lca.lci()
            amounts = cast(Any, lca.supply_array)
            assert all(isfinite(value) for value in amounts)
            assert amounts[lca.dicts.activity[node.id]] == pytest.approx(wet)
            ash = bd.get_node(database="frontier", code=akey[1])
            assert amounts[lca.dicts.activity[ash.id]] == pytest.approx(reference["ash_mass_kg"])
        for key, previous in before.items():
            assert [
                dict(row) for row in bd.get_node(database=key[0], code=key[1]).exchanges()
            ] == previous
