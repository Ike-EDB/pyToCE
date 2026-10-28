"""Keep known aqueous chloride distinct at actual mixed-water collection.

The existing external fixture supplies nonunit collection/plant references and
two sewer inputs. A chemically specified organic-P/NaCl load exercises the real
collection and wastewater owners; its quantities are independent test inputs.
"""

import json
from dataclasses import replace
from math import fsum

import pytest
from material_recipe_fixtures import physical_material_recipe
from tps_water_fixtures import (
    RAW_WATER,
    SURFACE,
    biosphere_inventory,
    flow_key,
    native_inventory,
    water_catalogue,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_source_closures,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _CH_WASTEWATER_PROFILE,
    _LeachateCarbon,
    _LeachateLoad,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_treatment import (
    _leachate_market_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


@pytest.fixture(scope="module")
def collection_boundary(tmp_path_factory):
    """Persist the existing source fixture in one isolated real Brightway project."""
    root = tmp_path_factory.mktemp("known_chloride")
    catalogue = water_catalogue()
    names = BrightwayNames(root / "brightway", "known_chloride", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere_inventory(), searchable=False)
        bd.Database("original").write(native_inventory("original", future=False), searchable=False)
        resolver = _ActivityResolver(bd, "original")
        context = _RecipeContext(
            background_scenario=None,
            year=2025,
            provider_database="providers",
            source=resolver,
            ordinary=resolver,
            technology=resolver,
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
        service = next(
            row
            for row in catalogue["services"]
            if row["provider_service_id"] == "test_tps_collection_biogenic"
        )
        yield context, service


@pytest.mark.parametrize("chloride_kg", (0.0, 0.01, 0.0355), ids=("unspecified", "partial", "full"))
def test_known_chloride_bypasses_once_without_changing_mixed_plant_load(
    collection_boundary, chloride_kg
):
    """Close the raw Cl split while preserving sodium, organic-P and full plant treatment."""
    context, service = collection_boundary
    # 0.1 mol C18H37O4P and 1 mol NaCl under explicit nominal atomic masses
    # C/H/O/P/Na/Cl = 12/1/16/31/23/35.5 g/mol, in 1 kg water.
    load = _LeachateLoad(
        elements_kg={"C": 0.0216, "H": 0.0037, "O": 0.0064, "P": 0.0031, "Na": 0.023, "Cl": 0.0355},
        carbon=(_LeachateCarbon("collected_monoester", False, "biogenic", 0.0216),),
        organic_hydrogen_kg=0.1225 * 0.0216,
        organic_oxygen_kg=0.64 * 0.0216,
        toc_kg=0.0216,
        doc_kg=0.68 * 0.0216,
        cod_kg_o2=0.0001 * 26.5 * 32,
        bod5_kg_o2=1.7586552173908316 * 0.0216,
    )
    load = replace(load, chloride_kg=chloride_kg)
    volume = (1 + fsum(load.elements_kg.values())) / 1000
    water = _wastewater_load(
        load,
        volume_m3=volume,
        incoming_water_kg=1.0,
        evaporation_fraction=0.01,
        profile=_CH_WASTEWATER_PROFILE,
    )
    assert water == _wastewater_load(
        replace(load, chloride_kg=0.0),
        volume_m3=volume,
        incoming_water_kg=1.0,
        evaporation_fraction=0.01,
        profile=_CH_WASTEWATER_PROFILE,
    )
    assert water.effluent.elements_kg["Cl"] == load.elements_kg["Cl"]
    assert water.final_sludge.elements_kg["Cl"] == 0
    assert 0 < water.effluent.elements_kg["Na"] < load.elements_kg["Na"]
    assert 0 < water.final_sludge.elements_kg["P"] < load.elements_kg["P"]
    assert water.duties.iron_sulfate_kg > 0
    assert water.duties.polyacrylamide_kg > 0

    dataset = _leachate_market_dataset(context, service, load, water, "providers")
    rows = dataset["exchanges"]
    production = next(row for row in rows if row["type"] == "production")
    plant = next(
        row
        for row in rows
        if row["input"] == ("providers", provider_service_code("test_tps_plant_biogenic"))
    )
    assert production["amount"] == -2
    assert plant["amount"] == -1.6
    native_sewers = []
    for exchange in context.source.node(("original", "collection")).exchanges():
        if exchange["type"] == "technosphere" and exchange["input"][1].startswith("sewer_"):
            source_row = dict(exchange)
            source_row.pop("output", None)
            native_sewers.append(source_row)
    assert [row for row in rows if row["input"][0] == "original"] == native_sewers
    bio = {row["input"]: row for row in rows if row["type"] == "biosphere"}
    assert len(bio) == sum(row["type"] == "biosphere" for row in rows)
    # Demand -volume against the source's -2 m3 production: 20% bypass once.
    collection_scale = volume / 2
    chlorine = bio.get(flow_key("Chlorine", RAW_WATER), {"amount": 0})
    chloride = bio.get(flow_key("Chloride", SURFACE), {"amount": 0})
    assert collection_scale * chlorine["amount"] == pytest.approx(0.2 * (0.0355 - chloride_kg))
    assert collection_scale * chloride["amount"] == pytest.approx(0.2 * chloride_kg)
    assert collection_scale * (chlorine["amount"] + chloride["amount"]) == pytest.approx(
        0.2 * 0.0355
    )
    for name, amount in (
        ("Sodium I", 0.023),
        ("Phosphorus", 0.0031),
        ("TOC, Total Organic Carbon", 0.0216),
    ):
        assert collection_scale * bio[flow_key(name, RAW_WATER)]["amount"] == pytest.approx(
            0.2 * amount
        )
    terms = json.loads(context.technology_rows[-1]["project_terms_json"])
    assert terms["raw_load_per_caller_reference"]["chloride_kg"] == chloride_kg
