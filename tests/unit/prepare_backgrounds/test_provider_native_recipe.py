"""Focused provider recipe compilation contracts."""

from pathlib import Path

import pytest
from bw2calc import LCA
from provider_recipe_fixtures import (
    _catalogue,
    _process,
    _write_bounded_frontier,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)


@pytest.mark.parametrize(
    ("conversion_operation", "finished_fraction"),
    [("conversion_injection", 0.8), ("conversion_foaming", 1.0)],
)
def test_native_processing_preserves_waste_and_uses_its_declared_basis(
    tmp_path: Path,
    conversion_operation: str,
    finished_fraction: float,
) -> None:
    """Distinguish conversion yield from waste; preserve native treatment and metadata."""

    catalogue = _catalogue()
    conversion_service = next(
        row for row in catalogue["services"] if row["provider_service_id"] == "conversion_operation"
    )
    conversion_service["construction"]["operation"] = conversion_operation
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="input_normalized_operation",
        biosphere_database="biosphere",
        source_database="ordinary",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "emission"): {
                    "name": "Process emission",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                },
                ("biosphere", "resource"): {
                    "name": "Process resource",
                    "unit": "kilogram",
                    "categories": ("natural resource",),
                    "type": "natural resource",
                },
            },
            searchable=False,
        )
        bd.Database("ordinary").write(
            {
                ("ordinary", "native_mdi"): _process(
                    "ordinary",
                    "native_mdi",
                    "native MDI",
                    "native MDI",
                ),
                ("ordinary", "native_polyol"): _process(
                    "ordinary",
                    "native_polyol",
                    "native polyol",
                    "native polyol",
                ),
                ("ordinary", "native_starch"): _process(
                    "ordinary",
                    "native_starch",
                    "market for native starch",
                    "native starch",
                ),
                ("ordinary", "unspecified_formulation"): _process(
                    "ordinary",
                    "unspecified_formulation",
                    "unspecified formulation",
                    "unspecified formulation",
                ),
                ("ordinary", "alternative_mdi"): _process(
                    "ordinary",
                    "alternative_mdi",
                    "alternative MDI",
                    "alternative MDI",
                ),
                ("ordinary", "alternative_polyol"): _process(
                    "ordinary",
                    "alternative_polyol",
                    "alternative polyol",
                    "alternative polyol",
                ),
                ("ordinary", "alternative_epoxy"): _process(
                    "ordinary",
                    "alternative_epoxy",
                    "alternative epoxy",
                    "alternative epoxy",
                ),
                ("ordinary", "alternative_polyester"): _process(
                    "ordinary",
                    "alternative_polyester",
                    "alternative polyester",
                    "alternative polyester",
                ),
                ("ordinary", "alternative_amylose"): _process(
                    "ordinary",
                    "alternative_amylose",
                    "alternative amylose",
                    "alternative amylose",
                ),
                ("ordinary", "waste"): _process(
                    "ordinary",
                    "waste",
                    "waste polymer supply",
                    "waste polymer",
                ),
                ("ordinary", "energy"): _process(
                    "ordinary",
                    "energy",
                    "energy supply",
                    "energy",
                ),
                ("ordinary", "reject"): _process(
                    "ordinary",
                    "reject",
                    "plastic reject treatment",
                    "plastic reject",
                ),
                ("ordinary", "metal"): _process(
                    "ordinary",
                    "metal",
                    "metal recovery",
                    "recovered metal",
                ),
                ("ordinary", "polymer_auxiliary"): _process(
                    "ordinary",
                    "polymer_auxiliary",
                    "polymer formulation auxiliary",
                    "polymer auxiliary",
                ),
                ("ordinary", "conversion_waste"): _process(
                    "ordinary",
                    "conversion_waste",
                    "external conversion waste",
                    "waste plastic",
                    production_amount=-2.0,
                    exchanges=[
                        {"input": ("biosphere", "emission"), "amount": 0.6, "type": "biosphere"}
                    ],
                ),
                ("ordinary", "principal_polymer"): _process(
                    "ordinary",
                    "principal_polymer",
                    "principal polymer supply",
                    "principal polymer",
                ),
                ("ordinary", "conversion_scrap"): _process(
                    "ordinary",
                    "conversion_scrap",
                    "conversion polymer scrap",
                    "waste polymer",
                ),
                ("ordinary", "recycling"): _process(
                    "ordinary",
                    "recycling",
                    "native recycling",
                    "recycled polymer",
                    production_amount=1.5,
                    exchanges=[
                        {
                            "input": ("ordinary", "waste"),
                            "amount": 2.0,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ordinary", "energy"),
                            "amount": 4.0,
                            "type": "technosphere",
                            "uncertainty type": 2,
                            "loc": 1.3862943611198906,
                            "scale": 0.1,
                        },
                        {
                            "input": ("ordinary", "reject"),
                            "amount": -0.5,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ordinary", "metal"),
                            "amount": -0.1,
                            "type": "technosphere",
                            "uncertainty type": 2,
                            "loc": -2.3025850929940455,
                            "scale": 0.2,
                            "negative": True,
                        },
                        {
                            "input": ("biosphere", "emission"),
                            "amount": 0.2,
                            "type": "biosphere",
                            "uncertainty type": 2,
                            "loc": -1.6094379124341003,
                            "scale": 0.3,
                        },
                    ],
                ),
                ("ordinary", "conversion"): _process(
                    "ordinary",
                    "conversion",
                    "native conversion",
                    "conversion operation",
                    production_amount=2.0,
                    exchanges=[
                        {
                            "input": ("ordinary", "energy"),
                            "amount": 3.0,
                            "type": "technosphere",
                            "uncertainty type": 2,
                            "loc": 1.0986122886681098,
                            "scale": 0.15,
                        },
                        {
                            "input": ("ordinary", "polymer_auxiliary"),
                            "amount": 0.05,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ordinary", "conversion_waste"),
                            "amount": -0.2,
                            "type": "technosphere",
                            "uncertainty type": 2,
                            "loc": -1.6094379124341003,
                            "scale": 0.17,
                            "negative": True,
                        },
                        {
                            "input": ("biosphere", "emission"),
                            "amount": 0.4,
                            "type": "biosphere",
                            "uncertainty type": 2,
                            "loc": -0.916290731874155,
                            "scale": 0.2,
                        },
                    ],
                ),
                ("ordinary", "native_adhesive"): _process(
                    "ordinary",
                    "native_adhesive",
                    "native adhesive",
                    "native adhesive",
                    exchanges=[
                        {
                            "input": ("ordinary", "native_mdi"),
                            "amount": 0.2,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ordinary", "native_mdi"),
                            "amount": 0.3,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ordinary", "native_polyol"),
                            "amount": 0.4,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ordinary", "unspecified_formulation"),
                            "amount": 0.1,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ordinary", "energy"),
                            "amount": 2.0,
                            "type": "technosphere",
                        },
                        {
                            "input": ("biosphere", "emission"),
                            "amount": 0.25,
                            "type": "biosphere",
                        },
                        {
                            "input": ("biosphere", "resource"),
                            "amount": 0.75,
                            "type": "biosphere",
                        },
                    ],
                ),
                ("ordinary", "native_fibre"): _process(
                    "ordinary",
                    "native_fibre",
                    "native fibre operation",
                    "finished fibre",
                    exchanges=[
                        {
                            "input": ("ordinary", "principal_polymer"),
                            "amount": 1.02,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ordinary", "conversion_scrap"),
                            "amount": -0.02,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ordinary", "energy"),
                            "amount": 4.0,
                            "type": "technosphere",
                        },
                        {
                            "input": ("biosphere", "emission"),
                            "amount": 0.3,
                            "type": "biosphere",
                        },
                    ],
                ),
                ("ordinary", "native_glucose"): _process(
                    "ordinary",
                    "native_glucose",
                    "native glucose production",
                    "glucose",
                    exchanges=[
                        {
                            "input": ("ordinary", "native_starch"),
                            "amount": 0.9,
                            "type": "technosphere",
                        },
                        {
                            "input": ("ordinary", "energy"),
                            "amount": 5.0,
                            "type": "technosphere",
                        },
                        {
                            "input": ("biosphere", "emission"),
                            "amount": 0.4,
                            "type": "biosphere",
                        },
                    ],
                ),
            },
            searchable=False,
        )

        native_conversion = bd.get_node(database="ordinary", code="conversion")
        native_production = next(iter(native_conversion.production()))
        native_production["uncertainty type"] = 3
        native_production["loc"] = 2.0
        native_production["scale"] = 0.1
        native_production.save()
        native_before = {
            code: [
                dict(exchange)
                for exchange in bd.get_node(database="ordinary", code=code).exchanges()
            ]
            for code in ("recycling", "conversion")
        }
        frontier_database, context, semantic_rows = _write_bounded_frontier(
            bd,
            catalogue,
        )

        operation = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("mechanical_recycling_operation"),
        )
        exchanges = {
            (exchange["type"], exchange.input.key): float(exchange["amount"])
            for exchange in operation.exchanges()
            if exchange["type"] != "production"
        }
        assert exchanges == {
            ("biosphere", ("biosphere", "emission")): 0.2,
            ("technosphere", ("ordinary", "energy")): 4.0,
            ("technosphere", ("ordinary", "metal")): -0.1,
            ("technosphere", ("ordinary", "waste")): 2.0,
        }
        assert next(iter(operation.production()))["amount"] == 2.0
        recycling_lca = LCA({operation: 1.0})
        recycling_lca.lci()
        for code, demand in (("energy", 2.0), ("metal", -0.05), ("waste", 1.0)):
            supplier = bd.get_node(database="ordinary", code=code)
            assert recycling_lca.supply_array[
                recycling_lca.dicts.activity[supplier.id]
            ] == pytest.approx(demand)
        assert float(recycling_lca.inventory.sum()) == pytest.approx(0.1)
        route_row = next(
            row for row in semantic_rows if row["ledger_entry_type"] == "route_mapping"
        )
        assert route_row["semantic_class"] == "service_normalization"
        normalization = next(
            row for row in semantic_rows if row["ledger_entry_type"] == "service_normalization"
        )
        assert normalization["native_normalization_input_amount"] == 2.0
        assert normalization["project_functional_unit"] == ("one kilogram normalized input")
        omission = next(
            row
            for row in semantic_rows
            if row["provider_service_id"] == "mechanical_recycling_operation"
            and row["ledger_entry_type"] == "native_exchange"
        )
        assert omission["removed_native_amount"] == -0.5
        assert omission["removed_amount_per_normalization_input"] == -0.25
        native = context.technology_rows[0]
        assert native["semantic_class"] == "service_normalization"
        assert '"name":"energy supply"' in native["retained_technosphere_json"]
        assert '"name":"plastic reject treatment"' in native["removed_technosphere_json"]

        conversion = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("conversion_operation"),
        )
        conversion_exchanges = {
            (exchange["type"], exchange.input.key): float(exchange["amount"])
            for exchange in conversion.exchanges()
        }
        assert conversion_exchanges == {
            ("biosphere", ("biosphere", "emission")): pytest.approx(0.4),
            (
                "production",
                (frontier_database, conversion["code"]),
            ): 2.0 * finished_fraction,
            ("technosphere", ("ordinary", "energy")): pytest.approx(3.0),
            ("technosphere", ("ordinary", "conversion_waste")): -0.2,
            (
                "technosphere",
                ("ordinary", "polymer_auxiliary"),
            ): pytest.approx(0.05),
        }
        conversion_balance = next(
            row for row in semantic_rows if row["ledger_entry_type"] == "conversion_service_balance"
        )
        assert conversion_balance["native_gross_production"] == 2.0
        assert conversion_balance["finished_kg_per_native_service_kg"] == finished_fraction
        finished_production = 2.0 * finished_fraction
        assert conversion_balance["finished_production"] == finished_production
        expected_production = (
            {key: value for key, value in dict(native_production).items() if key != "output"}
            if finished_fraction == 1.0
            else {"amount": finished_production, "type": "production"}
        )
        expected_production["input"] = conversion.key
        assert {
            key: value
            for key, value in dict(next(iter(conversion.production()))).items()
            if key != "output"
        } == expected_production

        conversion_lca = LCA({conversion: 1.0})
        conversion_lca.lci()
        energy = bd.get_node(database="ordinary", code="energy")
        polymer_auxiliary = bd.get_node(
            database="ordinary",
            code="polymer_auxiliary",
        )
        assert conversion_lca.supply_array[
            conversion_lca.dicts.activity[energy.id]
        ] == pytest.approx(3.0 / finished_production)
        assert conversion_lca.supply_array[
            conversion_lca.dicts.activity[polymer_auxiliary.id]
        ] == pytest.approx(0.05 / finished_production)
        waste = bd.get_node(database="ordinary", code="conversion_waste")
        assert conversion_lca.supply_array[
            conversion_lca.dicts.activity[waste.id]
        ] == pytest.approx((-0.2 / -2.0) / finished_production)
        assert float(conversion_lca.inventory.sum()) == pytest.approx(
            (0.4 + (-0.2 / -2.0) * 0.6) / finished_production
        )
        for code, generated, omitted in (
            ("recycling", operation, ("ordinary", "reject")),
            ("conversion", conversion, None),
        ):
            expected_records = [
                {key: value for key, value in exchange.items() if key != "output"}
                for exchange in native_before[code]
                if exchange["type"] != "production" and exchange["input"] != omitted
            ]
            actual_records = [
                {key: value for key, value in dict(exchange).items() if key != "output"}
                for exchange in generated.exchanges()
                if exchange["type"] != "production"
            ]
            assert sorted(actual_records, key=repr) == sorted(expected_records, key=repr)
            assert [
                dict(exchange)
                for exchange in bd.get_node(database="ordinary", code=code).exchanges()
            ] == native_before[code]

        complete_adhesive = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("complete_adhesive"),
        )
        adhesive_exchanges = [
            (exchange["type"], exchange.input.key, float(exchange["amount"]))
            for exchange in complete_adhesive.exchanges()
        ]
        assert sorted(adhesive_exchanges) == sorted(
            [
                ("biosphere", ("biosphere", "resource"), 0.75),
                ("production", complete_adhesive.key, 1.0),
                (
                    "technosphere",
                    (
                        frontier_database,
                        provider_catalogue.provider_service_code("alternative_mdi"),
                    ),
                    0.2,
                ),
                (
                    "technosphere",
                    (
                        frontier_database,
                        provider_catalogue.provider_service_code("alternative_mdi"),
                    ),
                    0.3,
                ),
                (
                    "technosphere",
                    (
                        frontier_database,
                        provider_catalogue.provider_service_code("alternative_polyol"),
                    ),
                    0.4,
                ),
                ("technosphere", ("ordinary", "energy"), 2.0),
                ("technosphere", ("ordinary", "unspecified_formulation"), 0.1),
            ]
        )
        adhesive_lca = LCA({complete_adhesive: 1.0})
        adhesive_lca.lci()
        alternative_mdi = bd.get_node(database="ordinary", code="alternative_mdi")
        alternative_polyol = bd.get_node(
            database="ordinary",
            code="alternative_polyol",
        )
        assert adhesive_lca.supply_array[
            adhesive_lca.dicts.activity[alternative_mdi.id]
        ] == pytest.approx(0.5)
        assert adhesive_lca.supply_array[
            adhesive_lca.dicts.activity[alternative_polyol.id]
        ] == pytest.approx(0.4)

        native_adhesive = bd.get_node(
            database="ordinary",
            code="native_adhesive",
        )
        assert sorted(
            float(exchange["amount"])
            for exchange in native_adhesive.technosphere()
            if exchange.input.key == ("ordinary", "native_mdi")
        ) == [0.2, 0.3]
        adhesive_source = next(
            row
            for row in context.technology_rows
            if row["provider_service_id"] == "complete_adhesive"
            and row["component_id"] == "native_operation"
        )
        assert '"name":"unspecified formulation"' in adhesive_source["retained_technosphere_json"]
        assert '"name":"energy supply"' in adhesive_source["retained_technosphere_json"]
        assert '"name":"native MDI"' in adhesive_source["removed_technosphere_json"]
        assert '"name":"native polyol"' in adhesive_source["removed_technosphere_json"]
        assert '"name":"Process resource"' in adhesive_source["retained_biosphere_json"]
        assert '"name":"Process emission"' in adhesive_source["removed_biosphere_json"]
        biosphere_omission = next(
            row
            for row in context.semantic_rows
            if row["provider_service_id"] == "complete_adhesive"
            and row["ledger_entry_type"] == "native_biosphere_omission"
        )
        assert biosphere_omission["removed_flow_name"] == "Process emission"
        assert biosphere_omission["source_unit_amount"] == pytest.approx(0.25)

        trimmed = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("scaled_operation_portfolio"),
        )
        trimmed_exchanges = {
            (exchange["type"], exchange.input.key): float(exchange["amount"])
            for exchange in trimmed.exchanges()
        }
        operation = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("operation_with_declared_omissions"),
        )
        assert trimmed_exchanges == {
            ("production", trimmed.key): 1.0,
            ("technosphere", operation.key): 0.25,
        }
        trimmed_lca = LCA({trimmed: 1.0})
        trimmed_lca.lci()
        energy = bd.get_node(database="ordinary", code="energy")
        emission = bd.get_node(database="biosphere", code="emission")
        assert trimmed_lca.supply_array[trimmed_lca.dicts.activity[energy.id]] == pytest.approx(1.0)
        assert float(
            trimmed_lca.inventory[trimmed_lca.dicts.biosphere[emission.id], :].sum()
        ) == pytest.approx(0.075)
        omission_rows = [
            row
            for row in context.semantic_rows
            if row["provider_service_id"] == "operation_with_declared_omissions"
            and row["ledger_entry_type"] == "native_exchange_omission"
        ]
        assert {
            (
                row["removed_input_name"],
                row["source_unit_amount"],
                row["reason"],
            )
            for row in omission_rows
        } == {
            (
                "principal polymer supply",
                1.02,
                "principal polymer is owned by the resin stage",
            ),
            (
                "conversion polymer scrap",
                -0.02,
                "conversion loss is owned by the process model",
            ),
        }
        balance_row = next(
            row
            for row in context.semantic_rows
            if row["provider_service_id"] == "operation_with_declared_omissions"
            and row["ledger_entry_type"] == "native_omission_balance"
        )
        assert balance_row["source_net_omitted_amount"] == pytest.approx(1.0)
        assert balance_row["native_production_amount"] == pytest.approx(1.0)
        assert balance_row["source_balance_difference"] == pytest.approx(0.0)
        assert balance_row["absolute_tolerance"] == pytest.approx(1e-12)

        coating = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("coating_blend"),
        )
        coating_exchanges = {
            (exchange["type"], exchange.input.key): float(exchange["amount"])
            for exchange in coating.exchanges()
        }
        assert coating_exchanges == {
            (
                "production",
                (frontier_database, coating["code"]),
            ): pytest.approx(1.0),
            ("technosphere", ("ordinary", "alternative_epoxy")): pytest.approx(0.5),
            (
                "technosphere",
                ("ordinary", "alternative_polyester"),
            ): pytest.approx(0.5),
        }
        coating_lca = LCA({coating: 1.0})
        coating_lca.lci()
        alternative_epoxy = bd.get_node(database="ordinary", code="alternative_epoxy")
        alternative_polyester = bd.get_node(
            database="ordinary",
            code="alternative_polyester",
        )
        assert coating_lca.supply_array[
            coating_lca.dicts.activity[alternative_epoxy.id]
        ] == pytest.approx(0.5)
        assert coating_lca.supply_array[
            coating_lca.dicts.activity[alternative_polyester.id]
        ] == pytest.approx(0.5)

        glucose = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("selected_glucose"),
        )
        glucose_exchanges = {
            (exchange["type"], exchange.input.key): float(exchange["amount"])
            for exchange in glucose.exchanges()
        }
        assert glucose_exchanges == {
            ("biosphere", ("biosphere", "emission")): pytest.approx(0.4),
            (
                "production",
                (frontier_database, glucose["code"]),
            ): pytest.approx(1.0),
            (
                "technosphere",
                (
                    frontier_database,
                    provider_catalogue.provider_service_code("alternative_amylose"),
                ),
            ): pytest.approx(0.9045226130653267),
            ("technosphere", ("ordinary", "energy")): pytest.approx(5.0),
        }
        glucose_lca = LCA({glucose: 1.0})
        glucose_lca.lci()
        alternative_amylose = bd.get_node(
            database="ordinary",
            code="alternative_amylose",
        )
        assert glucose_lca.supply_array[
            glucose_lca.dicts.activity[alternative_amylose.id]
        ] == pytest.approx(0.9045226130653267)
        glucose_source = next(
            row
            for row in context.technology_rows
            if row["provider_service_id"] == "selected_glucose"
            and row["component_id"] == "native_operation"
        )
        assert '"name":"energy supply"' in glucose_source["retained_technosphere_json"]
        assert '"name":"market for native starch"' in (glucose_source["removed_technosphere_json"])

        portfolio = bd.get_node(
            database=frontier_database,
            code=provider_catalogue.provider_service_code("source_dependent_portfolio"),
        )
        portfolio_lca = LCA({portfolio: 1.0})
        portfolio_lca.lci()
        expected_supplies = {
            "alternative_amylose": 0.5 * 0.9045226130653267,
            "alternative_epoxy": 0.3 * 0.5,
            "alternative_mdi": 0.2 * 0.5,
            "alternative_polyester": 0.3 * 0.5,
            "alternative_polyol": 0.2 * 0.4,
            "unspecified_formulation": 0.2 * 0.1,
            "energy": 0.2 * 2.0 + 0.5 * 5.0,
        }
        for code, expected in expected_supplies.items():
            supplier = bd.get_node(database="ordinary", code=code)
            assert portfolio_lca.supply_array[
                portfolio_lca.dicts.activity[supplier.id]
            ] == pytest.approx(expected)
        for code, expected in {"emission": 0.5 * 0.4, "resource": 0.2 * 0.75}.items():
            flow = bd.get_node(database="biosphere", code=code)
            assert float(
                portfolio_lca.inventory[portfolio_lca.dicts.biosphere[flow.id], :].sum()
            ) == pytest.approx(expected)
