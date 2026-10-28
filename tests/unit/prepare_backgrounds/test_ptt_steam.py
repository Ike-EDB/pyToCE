"""Retain complete chemical steam when PTT replaces only its delivered heating."""

import json
from math import log
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from bw2calc import LCA
from chemical_steam_fixtures import chemical_steam_biosphere, chemical_steam_native_inventory
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from ptt_steam_fixtures import ptt_steam_native_inventory

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
def test_ptt_changes_only_heat_and_preserves_complete_scaled_steam(
    tmp_path: Path, original: bool
) -> None:
    """Use the package compiler to detect missing steam inputs or double counting."""
    provider = prepared_provider_authority()
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, Any]], provider.catalogue["services"])
    }
    service = services["ptt_renewable_steam"]
    assert service["construction"]["kind"] == "ptt_renewable_steam"
    names = BrightwayNames(tmp_path / "brightway", "ptt", "biosphere", "source")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(chemical_steam_biosphere(), searchable=False)
        native = {
            **chemical_steam_native_inventory(provider.catalogue, "source"),
            **ptt_steam_native_inventory("source"),
        }
        bd.Database("source").write(native, searchable=False)
        database = "source" if original else "future"
        if not original:
            future = {
                **chemical_steam_native_inventory(provider.catalogue, database),
                **ptt_steam_native_inventory(database),
            }
            # Doubled dated nonproduction exchanges retain the original study reference.
            for code in ("ptt_kg_converter", "steam_row"):
                for row in future[database, code]["exchanges"]:
                    if row["type"] != "production":
                        row["amount"] *= 2
            bd.Database(database).write(future, searchable=False)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario=None if original else "remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="frontier",
            source=_ActivityResolver(bd, "source"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=provider.catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(provider.catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        ptt = cast(dict[str, Any], _service_dataset(context, service, services, "frontier"))
        key = ("frontier", provider_service_code("ptt_renewable_steam"))
        assert (ptt["database"], ptt["code"]) == key
        assert ptt["name"] == "PTT chemical steam production, biogas heat"
        assert ptt["native_note"] == "retain converter metadata"
        assert ptt["exchanges"][0] == {
            "input": key,
            "type": "production",
            "amount": 1.0,
            "comment": "Source reference amount differs from one kg.",
        }
        # The fixture supplies 8/2 = 4 MJ/kg and (2.4+1.5)/3 = 1.3 MJ heat/MJ.
        # All full non-heat rows scale by 4/3. No 74% or other efficiency enters.
        factor = 4 / 3
        reference = [
            row
            for row in native[("source", "steam_row")]["exchanges"]
            if row["type"] != "production"
            and row["input"][1] not in {"steam_heat_a", "steam_heat_b"}
        ]
        actual = cast(list[dict[str, Any]], ptt["exchanges"])[1:-1]
        assert len(actual) == len(reference)
        for source, target in zip(reference, actual, strict=True):
            assert target["amount"] == pytest.approx(source["amount"] * factor)
            assert target["input"] == (
                (database, source["input"][1])
                if source["type"] == "technosphere"
                else source["input"]
            )
            assert target["type"] == source["type"]
            for field in source.keys() - {"amount", "loc", "output", "input"}:
                assert target[field] == source[field]
            if "loc" in source:
                assert target["loc"] == pytest.approx(source["loc"] + log(factor))
        assert ptt["exchanges"][-1] == {
            "input": (database, "ptt_biogas"),
            "amount": pytest.approx(5.2),
            "type": "technosphere",
        }
        assert ptt["source_balance_outputs"]["native_conversion_mj_per_kg"] == 4
        assert ptt["source_balance_outputs"]["native_heating_mj_per_mj"] == pytest.approx(1.3)
        traces = {row["component_id"]: row for row in context.technology_rows}
        conversion = json.loads(
            cast(str, traces["complete_biogas_engine_heat"]["project_terms_json"])
        )
        assert conversion["native_reference_kg"] == 2
        assert conversion["native_conversion_exchange"]["scale"] == 0.15
        assert conversion["native_conversion_exchange"]["loc"] == log(8)
        assert conversion["conversion_source_key"] == ["source", "ptt_kg_converter"]
        assert conversion["complete_steam_source_key"] == ["source", "steam_row"]
        assert {row["input"][1] for row in conversion["original_heating_exchanges"]} == {
            "steam_heat_a",
            "steam_heat_b",
        }
        assert conversion["biogas_heating_mj_per_kg"] == pytest.approx(5.2)
        assert {row["native_database_name"] for row in traces.values()} == {database}
        bd.Database("frontier").write({key: ptt}, searchable=False)
        result = LCA({bd.get_node(database=key[0], code=key[1]): 1.0})
        result.lci()
        supply_array = cast(NDArray[np.float64] | None, result.supply_array)
        assert supply_array is not None
        for code in ("steam_power_a", "steam_power_b", "steam_water_a", "steam_water_b"):
            node = bd.get_node(database=database, code=code)
            assert supply_array[result.dicts.activity[node.id]] > 0
        for code in ("steam_heat_a", "steam_heat_b"):
            node = bd.get_node(database=database, code=code)
            assert supply_array[result.dicts.activity[node.id]] == 0
        biogas = bd.get_node(database=database, code="ptt_biogas")
        assert supply_array[result.dicts.activity[biogas.id]] == pytest.approx(5.2)
        # Retain original converter, heat sources, and regional steam vectors for their existing
        # consumers.
        for code in (
            "ptt_kg_converter",
            "steam_heat_a",
            "steam_heat_b",
            "steam_row",
            "steam_rer",
            "steam_row_market",
            "steam_rer_market",
        ):
            node = bd.get_node(database="source", code=code)
            assert [
                {k: v for k, v in dict(row).items() if k != "output"} for row in node.exchanges()
            ] == [
                {k: v for k, v in row.items() if k != "output"}
                for row in native[("source", code)]["exchanges"]
            ]
