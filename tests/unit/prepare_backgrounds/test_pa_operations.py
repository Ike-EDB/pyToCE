"""Check native PA operating wastewater and distinct actual-condensate treatment."""

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from pa_operations_fixtures import pa_operation_biosphere, pa_operation_native_inventory

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


@pytest.mark.parametrize("original", [True, False])
def test_real_operations_keep_native_wastewater_and_separate_condensate(
    tmp_path: Path, original: bool
) -> None:
    """Solve native site wastewater and actual condensate through their distinct plants."""
    catalogue = load_provider_catalogue()
    services = {s["provider_service_id"]: s for s in catalogue["services"]}
    suffixes = ("row", "ch", "ca_qc", "europe")
    owners = (
        "pa_ester_operation",
        "pa_amide_operation",
        "pa_condensate_collection",
        "pa_finish_nitrogen",
    )
    selected = (
        *owners,
        *(
            prefix + suffix
            for suffix in suffixes
            for prefix in ("hydraulic_water_collection_", "hydraulic_water_treatment_")
        ),
    )
    names = BrightwayNames(tmp_path / "brightway", "pa_operations", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(pa_operation_biosphere(), searchable=False)
        bd.Database("original").write(
            pa_operation_native_inventory(catalogue, "original", original=True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                pa_operation_native_inventory(catalogue, database, original=False), searchable=False
            )
        resolver = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=resolver,
            technology=resolver,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                r["method_authority_id"]: r for r in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        compiled = {
            ("providers", provider_service_code(sid)): _service_dataset(
                context, services[sid], services, "providers"
            )
            for sid in selected
        }
        bd.Database("providers").write(compiled, searchable=False)
        for sid, multiplier in ((owners[0], 1.0), (owners[1], 1.3)):
            node = bd.get_node(database="providers", code=provider_service_code(sid))
            source = bd.get_node(
                database=database, code="pa_ester" if sid == owners[0] else "pa_amide"
            )
            production = next(iter(source.production()))["amount"]
            native_wastewater = {
                ex.input.key: ex
                for ex in source.technosphere()
                if ex.input["reference product"] == "wastewater, average"
            }
            assert next(iter(node.production()))["amount"] == 1
            amounts = {ex.input.key: ex["amount"] for ex in node.technosphere()}
            assert amounts[database, "pa_factory"] == pytest.approx(0.003 * multiplier)
            assert amounts[database, "electricity_a"] == pytest.approx(0.3 * multiplier)
            assert amounts[database, "heat_a"] == pytest.approx(1.7 * multiplier)
            assert amounts[database, "pa_steam"] == pytest.approx(0.8 * multiplier)
            assert (database, "pa_reactant") not in amounts
            assert all(ex.input["name"] == "Water" for ex in node.biosphere())
            for i, suffix in enumerate(suffixes):
                key = database, "market_" + suffix
                assert amounts[key] == pytest.approx(-(i + 1) * 0.0001 * multiplier)
                native_row = native_wastewater[key]
                expected = {
                    **dict(native_row),
                    "output": node.key,
                    "amount": pytest.approx(native_row["amount"] / production),
                    "loc": pytest.approx(native_row["loc"] - math.log(production)),
                }
                generated = next(ex for ex in node.technosphere() if ex.input.key == key)
                assert dict(generated) == expected
            lca = LCA({node: 1.0})
            lca.lci()
            supply = cast(Any, lca.supply_array)
            assert all(math.isfinite(q) for q in supply)
            for i, suffix in enumerate(suffixes):
                plant = bd.get_node(database=database, code="plant_" + suffix)
                hydraulic = bd.get_node(
                    database="providers",
                    code=provider_service_code("hydraulic_water_treatment_" + suffix),
                )
                # Source market -2 m3 buys -0.5 m3 from plant -4 m3: demand / 16.
                assert supply[lca.dicts.activity[plant.id]] == pytest.approx(
                    (i + 1) * 0.0001 * multiplier / 16
                )
                assert supply[lca.dicts.activity[hydraulic.id]] == 0
            cod = bd.get_node(database="biosphere", code="cod")
            # Native market emits 12 per -2 m3; its plant emits 7 per -4 m3.
            assert lca.inventory[lca.dicts.biosphere[cod.id], :].sum() == pytest.approx(
                0.001 * multiplier * (12 / 2 + 7 / 16)
            )
        condensate = bd.get_node(database="providers", code=provider_service_code(owners[2]))
        assert next(iter(condensate.production()))["amount"] == -1
        assert sum(ex["amount"] for ex in condensate.technosphere()) == pytest.approx(-1)
        assert list(condensate.biosphere()) == []
        condensate_trace = next(
            row
            for row in context.technology_rows
            if row["component_id"] == "condensate_collection_portfolio"
        )
        # Only the donor's wastewater portfolio supplies this collection
        # function. Its CO2 and water flows must appear only as removed in the
        # published provider provenance.
        assert json.loads(str(condensate_trace["retained_biosphere_json"])) == []
        removed_biosphere = json.loads(str(condensate_trace["removed_biosphere_json"]))
        assert {row["code"] for row in removed_biosphere} == {"co2", "pa_water_air", "surface"}
        lca = LCA({condensate: -1.0})
        lca.lci()
        supply = cast(Any, lca.supply_array)
        for i, suffix in enumerate(suffixes):
            native_plant = bd.get_node(database=database, code="plant_" + suffix)
            hydraulic = bd.get_node(
                database="providers",
                code=provider_service_code("hydraulic_water_treatment_" + suffix),
            )
            assert supply[lca.dicts.activity[native_plant.id]] == 0
            assert supply[lca.dicts.activity[hydraulic.id]] == pytest.approx((i + 1) / 10 / 16)
        assert lca.inventory[lca.dicts.biosphere[cod.id], :].sum() == 0
        n2 = bd.get_node(database="providers", code=provider_service_code(owners[3]))
        purchases = {ex.input.key: ex for ex in n2.technosphere()}
        assert purchases[database, "pa_n2_a"]["amount"] == pytest.approx(0.3 if original else 0.8)
        assert purchases[database, "pa_n2_b"]["amount"] == pytest.approx(0.7 if original else 0.2)
        assert all(ex["scale"] == 0.17 for ex in purchases.values())
        assert len(list(n2.biosphere())) == 1
        assert next(iter(n2.biosphere()))["amount"] == pytest.approx(1)
        assert next(iter(n2.biosphere())).input["name"] == "Nitrogen"
