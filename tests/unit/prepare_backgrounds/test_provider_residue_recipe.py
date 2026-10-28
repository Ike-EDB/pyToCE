"""Compile a residue service and its separately accounted phosphorus load."""

from pathlib import Path

import pytest
from bw2calc import LCA
from provider_recipe_fixtures import _catalogue, _process, _write_bounded_frontier

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    provider_service_code,
)


def test_native_residue_treatment_and_excess_phosphorus_are_counted_once(tmp_path: Path) -> None:
    """Keep positive native treatment activity and add only the explicit excess."""

    volume = 2.0
    effluent_phosphorus = 1.0
    excess = effluent_phosphorus - 0.105 * volume
    parent_feed = 3.0
    catalogue = _catalogue()
    common = {
        "unit": "kilogram",
        "location": "RER",
        "stage_owner": "virgin_resin_production",
        "citation": "bounded_native_materials",
    }
    catalogue["services"] = [
        {
            **common,
            "provider_service_id": "residue_operation",
            "name": "residue operation",
            "reference_product": "recovered feed",
            "foreground_service": False,
            "construction": {
                "kind": "project_recipe",
                "components": [
                    {
                        "component_id": "aqueous_residue_treatment",
                        "source_role": "native_ecoinvent_provider",
                        "selector": {
                            "name": "wastewater treatment",
                            "reference_product": "wastewater",
                            "unit": "cubic meter",
                            "exact_location": "RER",
                        },
                        "amount": -volume,
                    }
                ],
                "biosphere_terms": [
                    {
                        "selector": {
                            "name": "Phosphorus",
                            "unit": "kilogram",
                            "categories": ["water", "surface water"],
                        },
                        "amount": excess,
                    }
                ],
            },
        },
        {
            **common,
            "provider_service_id": "finished_product",
            "name": "finished product",
            "reference_product": "finished product",
            "foreground_service": True,
            "construction": {
                "kind": "project_recipe",
                "components": [{"service_id": "residue_operation", "amount": parent_feed}],
            },
        },
    ]
    catalogue["routes"] = []
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="residue_compilation",
        biosphere_database="biosphere",
        source_database="ordinary",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "phosphorus"): {
                    "name": "Phosphorus",
                    "unit": "kilogram",
                    "categories": ("water", "surface water"),
                    "type": "emission",
                }
            },
            searchable=False,
        )
        treatment = _process(
            "ordinary",
            "wastewater",
            "wastewater treatment",
            "wastewater",
            production_amount=-1.0,
            exchanges=[
                {"input": ("biosphere", "phosphorus"), "amount": 0.02, "type": "biosphere"},
                {"input": ("ordinary", "utility"), "amount": 0.5, "type": "technosphere"},
            ],
        )
        treatment["unit"] = "cubic meter"
        bd.Database("ordinary").write(
            {
                ("ordinary", "wastewater"): treatment,
                ("ordinary", "utility"): _process("ordinary", "utility", "utility", "utility"),
            },
            searchable=False,
        )
        bd.Database("technology").write({}, searchable=False)
        native = bd.get_node(database="ordinary", code="wastewater")
        before = [exchange.as_dict() for exchange in native.exchanges()]
        frontier, _, _ = _write_bounded_frontier(bd, catalogue)
        product = bd.get_node(database=frontier, code=provider_service_code("finished_product"))
        assert {
            (exchange.input.key, exchange["type"]): exchange["amount"]
            for exchange in product.exchanges()
        } == {
            (product.key, "production"): 1.0,
            (("ordinary", "wastewater"), "technosphere"): -parent_feed * volume,
            (("biosphere", "phosphorus"), "biosphere"): parent_feed * excess,
        }
        lca = LCA({product: 1.0})
        lca.lci()
        utility = bd.get_node(database="ordinary", code="utility")
        assert lca.supply_array[lca.dicts.activity[native.id]] == pytest.approx(
            parent_feed * volume
        )
        assert lca.supply_array[lca.dicts.activity[utility.id]] == pytest.approx(
            parent_feed * volume * 0.5
        )
        assert float(lca.inventory.sum()) == pytest.approx(parent_feed * (volume * 0.02 + excess))
        assert [exchange.as_dict() for exchange in native.exchanges()] == before
