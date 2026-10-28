"""Verify the explicit PA phosphorus reference through real landfarming records.

Invented original/future donors retain complete spreading and nonunit negative
production. These tests exercise the shared sludge owner and native stores;
PA receiving integration tests cover its use with complete treatment chains.
"""

import json
from math import fsum
from pathlib import Path
from typing import Any, cast

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from tps_treatment_fixtures import (
    GROUND,
    SOIL,
    SURFACE,
    URBAN,
    biosphere_inventory,
    flow_key,
    native_inventory,
    treatment_catalogue,
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
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.sludge_landfarming import (
    _landfarming_dataset,
)

_DONOR_ONLY_METALS = ("Scandium", "Silver I", "Tungsten")


@pytest.fixture(
    scope="module",
    params=((False, False), (False, True), (True, False), (True, True)),
    ids=("original_no_donor_p", "original_donor_p", "future_no_donor_p", "future_donor_p"),
)
def landfarming_boundary(request, tmp_path_factory):
    """Persist two small synthetic source nodes with optional phosphorus rows."""
    future, donor_phosphorus = request.param
    root: Path = tmp_path_factory.mktemp("pa_landfarming")
    names = BrightwayNames(root / "brightway", "pa_landfarming", "biosphere", "original")
    catalogue = treatment_catalogue()
    phosphate_keys = {flow_key("Phosphate", compartment) for compartment in (GROUND, SURFACE)}
    with selected_brightway_project(names) as bd:
        biosphere = biosphere_inventory()
        biosphere.update(
            {
                flow_key(name, SOIL): {
                    "name": name,
                    "categories": SOIL,
                    "unit": "kilogram",
                    "type": "emission",
                }
                for name in _DONOR_ONLY_METALS
            }
        )
        bd.Database("biosphere").write(biosphere, searchable=False)
        databases = ("original", "future") if future else ("original",)
        for database in databases:
            sources = {
                key: record
                for key, record in native_inventory(database, future=database == "future").items()
                if key[1] in {"landfarming", "spreading"}
            }
            # The synthetic municipal donor has trace-metal flows; actual PA quantities follow the
            # specified load.
            sources[database, "landfarming"]["exchanges"].extend(
                {
                    "input": flow_key(name, SOIL),
                    "amount": 0.001,
                    "unit": "kilogram",
                    "type": "biosphere",
                    "uncertainty type": 0,
                }
                for name in _DONOR_ONLY_METALS
            )
            if not donor_phosphorus:
                donor = sources[database, "landfarming"]
                donor["exchanges"] = [
                    row for row in donor["exchanges"] if row["input"] not in phosphate_keys
                ]
                production = next(row for row in donor["exchanges"] if row["type"] == "production")
                # This P-free donor has an unspecified inert remainder. The
                # selected actual-P calculation cannot rely on a donor P assay.
                production["properties"].pop("phosphorus content")
            bd.Database(database).write(sources, searchable=False)
        dated_database = "future" if future else "original"
        dated = _ActivityResolver(bd, dated_database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
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
        yield bd, context, dated_database, donor_phosphorus


@pytest.mark.parametrize("incoming_phosphorus", (0.0, 0.012))
def test_pa_equilibrium_replaces_only_phosphorus_and_keeps_complete_spreading(
    landfarming_boundary, incoming_phosphorus: float
) -> None:
    """Close actual P through two water outlets and crop uptake on the signed graph.

    Incoming quantities are synthetic boundary masses. Carbon and sodium remain in the same
    mixture, and the real WWT owner supplies sludge, PAM, moisture and shares.
    """
    bd, context, native_database, donor_phosphorus = landfarming_boundary
    wastewater = _wastewater_load(
        _LeachateLoad(
            elements_kg={"C": 0.05, "N": 0.002, "P": incoming_phosphorus, "Na": 0.02},
            carbon=(_LeachateCarbon("pa_finish_water", False, "biogenic", 0.05),),
            organic_hydrogen_kg=0.006125,
            organic_oxygen_kg=0.032,
            toc_kg=0.05,
            doc_kg=0.034,
            cod_kg_o2=0.14,
            bod5_kg_o2=0.09,
        ),
        volume_m3=0.003,
        incoming_water_kg=2.9,
        evaporation_fraction=0.01,
        profile=_CH_WASTEWATER_PROFILE,
    )
    service_id = "tps_sludge_landfarming_biogenic"
    service = next(
        row for row in context.catalogue["services"] if row["provider_service_id"] == service_id
    )
    provider = "pa_landfarming_with_p" if incoming_phosphorus else "pa_landfarming_without_p"
    native_record = _landfarming_dataset(
        context,
        service,
        wastewater,
        "pa_finish_terminal_water",
        provider,
        complete_carbon_oxidation=False,
    )
    native_terms = json.loads(context.technology_rows[-1]["project_terms_json"])
    record = _landfarming_dataset(
        context,
        service,
        wastewater,
        "pa_finish_terminal_water",
        provider,
        complete_carbon_oxidation=False,
        phosphorus_equilibrium=True,
    )
    terms = json.loads(context.technology_rows[-1]["project_terms_json"])
    rows = cast(list[dict[str, Any]], record["exchanges"])
    native_rows = cast(list[dict[str, Any]], native_record["exchanges"])
    donor_metals = {flow_key(name, SOIL) for name in _DONOR_ONLY_METALS}
    assert not any(row["input"] in donor_metals for row in rows + native_rows)
    donor = bd.get_node(database=native_database, code="landfarming")
    assert {
        row.input.key: row["amount"] for row in donor.biosphere() if row.input.key in donor_metals
    } == dict.fromkeys(donor_metals, 0.001)
    phosphate_keys = {flow_key("Phosphate", compartment) for compartment in (GROUND, SURFACE)}
    assert [row for row in rows if row["input"] not in phosphate_keys] == [
        row for row in native_rows if row["input"] not in phosphate_keys
    ]
    wet = wastewater.landfarmed_sludge_wet_kg
    production = next(row for row in rows if row["type"] == "production")
    reference = -production["amount"]
    assert reference == (11 if native_database == "future" else 8)
    phosphorus = wastewater.landfarming_dry_share * wastewater.final_sludge.elements_kg["P"]
    actual = {
        row["input"]: row["amount"] * wet / reference
        for row in rows
        if row["input"] in phosphate_keys
    }
    # Doka2021 report pp56-58: equilibrium rates in kg P/(ha year),
    # normalized by 12 kg P/(ha year). WWcalc C401 retains erosion1.0602.
    ground_p = phosphorus * (0.07 / 12)
    surface_p = phosphorus * ((0.175 + 1.0602) / 12)
    for compartment, expected_p in ((GROUND, ground_p), (SURFACE, surface_p)):
        key = flow_key("Phosphate", compartment)
        matching = [row for row in rows if row["input"] == key]
        assert len(matching) == (1 if incoming_phosphorus else 0)
        assert actual.get(key, 0.0) == pytest.approx(expected_p * 94.97 / 30.97, rel=2e-12, abs=0)
        for row in matching:
            assert row["unit"] == "kilogram"
            assert row["uncertainty type"] == 0
            assert "loc" not in row  # A donor distribution cannot quantify this reference.
    emitted_p = fsum(actual.values()) * 30.97 / 94.97
    assert emitted_p + terms["phosphorus_crop_uptake_kg"] == pytest.approx(
        phosphorus, rel=2e-12, abs=0
    )
    assert terms["phosphorus_crop_uptake_kg"] == pytest.approx(phosphorus - ground_p - surface_p)
    assert "report equilibrium" in terms["phosphorus_allocation"]
    assert "Established native" in native_terms["phosphorus_allocation"]
    native_phosphate = [row for row in native_rows if row["input"] in phosphate_keys]
    assert len(native_phosphate) == (2 if donor_phosphorus and incoming_phosphorus else 0)
    for row in native_phosphate:
        assert row["amount"] * wet / reference * 31 / 95 == pytest.approx(0.1 * phosphorus)
        assert row["uncertainty type"] == 2
    assert native_terms["phosphorus_crop_uptake_kg"] == pytest.approx(
        (0.8 if donor_phosphorus else 1.0) * phosphorus
    )

    bd.Database(provider).write(
        {(provider, provider_service_code(service_id)): record}, searchable=False
    )
    node = bd.get_node(database=provider, code=provider_service_code(service_id))
    lca = LCA({node: -wet})
    lca.lci()
    assert lca.supply_array is not None
    assert lca.supply_array[lca.dicts.activity[node.id]] == pytest.approx(wet / reference, rel=2e-7)
    spreading = bd.get_node(database=native_database, code="spreading")
    spreading_m3 = wet * (2 if native_database == "future" else 1) / 1030
    assert lca.supply_array[lca.dicts.activity[spreading.id]] == pytest.approx(
        spreading_m3, rel=2e-7
    )
    assert len(list(node.technosphere())) == 1
    upstream_carbon = bd.get_node(
        database="biosphere", code=flow_key("Carbon dioxide, fossil", URBAN)[1]
    )
    assert float(lca.inventory[lca.dicts.biosphere[upstream_carbon.id], :].sum()) == pytest.approx(
        0.02 * spreading_m3, rel=2e-7
    )
    for compartment, expected_p in ((GROUND, ground_p), (SURFACE, surface_p)):
        flow = bd.get_node(database="biosphere", code=flow_key("Phosphate", compartment)[1])
        if incoming_phosphorus:
            assert float(lca.inventory[lca.dicts.biosphere[flow.id], :].sum()) == pytest.approx(
                expected_p * 94.97 / 30.97, rel=2e-7
            )
