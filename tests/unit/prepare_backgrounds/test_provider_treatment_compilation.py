"""Focused provider recipe compilation contracts."""

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import _catalogue, _process, _selector

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    catalogue as provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    native_resolution,
    recipe_compilation,
)
from pytoce.run_lca.calculation.carbon import (
    carbon_consistency_table,
    direct_treatment_carbon_flows,
)
from pytoce.run_lca.calculation.lcia import (
    ImpactCategory,
    characterize_principal_inventory,
)
from pytoce.run_lca.calculation.timex import (
    CharacterizationFactorBank,
    PrincipalInventory,
)


def test_treatment_origin_operation_changes_only_direct_carbon_identity(
    tmp_path: Path,
) -> None:
    """Retain native treatment exchanges and their quantitative metadata."""

    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="treatment_origin_operation",
        biosphere_database="biosphere",
        source_database="ordinary",
    )
    with selected_brightway_project(names) as bd:
        flow_rows = {
            "co2_fossil": (
                "Carbon dioxide, fossil",
                ("air", "urban air close to ground"),
            ),
            "co2_non_fossil": (
                "Carbon dioxide, non-fossil",
                ("air", "urban air close to ground"),
            ),
            "ch4_fossil": (
                "Methane, fossil",
                ("air", "urban air close to ground"),
            ),
            "ch4_non_fossil": (
                "Methane, non-fossil",
                ("air", "urban air close to ground"),
            ),
            "co_fossil": (
                "Carbon monoxide, fossil",
                ("air", "urban air close to ground"),
            ),
            "co_non_fossil": (
                "Carbon monoxide, non-fossil",
                ("air", "urban air close to ground"),
            ),
            "nitrogen_oxides": (
                "Nitrogen oxides",
                ("air", "urban air close to ground"),
            ),
        }
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": name,
                    "unit": "kilogram",
                    "categories": categories,
                    "type": "emission",
                }
                for code, (name, categories) in flow_rows.items()
            },
            searchable=False,
        )
        bd.Database("ordinary").write(
            {
                ("ordinary", "energy"): _process(
                    "ordinary",
                    "energy",
                    "electricity supply",
                    "electricity",
                ),
                ("ordinary", "treatment"): _process(
                    "ordinary",
                    "treatment",
                    "native plastic incineration",
                    "waste plastic treatment",
                    production_amount=-1.0,
                    exchanges=[
                        {
                            "input": ("ordinary", "energy"),
                            "amount": 0.25,
                            "type": "technosphere",
                            "uncertainty type": 2,
                            "loc": -1.4,
                            "scale": 0.2,
                            "allocation": "cut-off",
                        },
                        *[
                            {
                                "input": ("biosphere", code),
                                "amount": amount,
                                "type": "biosphere",
                                "name": flow_rows[code][0],
                                "unit": "kilogram",
                                "uncertainty type": 2,
                                "loc": loc,
                                "scale": 0.1,
                                "allocation": "native treatment allocation",
                            }
                            for code, amount, loc in (
                                ("co2_fossil", 2.2, 0.7),
                                ("ch4_fossil", 0.08, -2.5),
                                ("co_fossil", 0.028, -3.0),
                                ("nitrogen_oxides", 0.01, -4.0),
                            )
                        ],
                    ],
                ),
            },
            searchable=False,
        )

        catalogue = _catalogue()
        selector = _selector(
            "native plastic incineration",
            "waste plastic treatment",
            "kilogram",
        )
        fossil_service: dict[str, object] = {
            "provider_service_id": "incineration_fossil",
            "name": "native incineration service",
            "reference_product": "plastic incineration service",
            "unit": "kilogram",
            "location": "RER",
            "stage_owner": "incineration_treatment",
            "foreground_service": True,
            "citation": "bounded_native_materials",
            "provider_demand_type": "incineration_treatment",
            "ultimate_carbon_origins": ["fossil"],
            "construction": {
                "kind": "native_activity",
                "source_role": "native_ecoinvent_provider",
                "selector": selector,
                "amount": -1.0,
                "direct_treatment_carbon_owner": "native_activity",
                "carbon_attribution_status": "separable_treated_plastic_carbon",
            },
        }
        replacement_rules: list[dict[str, object]] = [
            {
                "selector": {
                    "name": source_name,
                    "unit": "kilogram",
                    "categories": ["air", "urban air close to ground"],
                },
                "replacement_selector": {
                    "name": replacement_name,
                    "unit": "kilogram",
                    "categories": ["air", "urban air close to ground"],
                },
                "attribution_basis": (
                    "The native exchange is direct carbon from the treated plastic."
                ),
                "carbon_attribution_status": "separable_treated_plastic_carbon",
                "mapping_status": "exact_non_fossil_counterpart",
                "limitation": "Only the declared direct waste-carbon flow changes.",
                "reason": "Preserve the selected plastic carbon origin at treatment.",
            }
            for source_name, replacement_name in (
                ("Carbon dioxide, fossil", "Carbon dioxide, non-fossil"),
                ("Methane, fossil", "Methane, non-fossil"),
                ("Carbon monoxide, fossil", "Carbon monoxide, non-fossil"),
            )
        ]
        cast(dict[str, object], fossil_service["construction"])[
            "direct_treatment_carbon_mappings"
        ] = replacement_rules
        non_fossil_service: dict[str, object] = {
            **fossil_service,
            "provider_service_id": "incineration_non_fossil",
            "name": "non-fossil incineration service",
            "ultimate_carbon_origins": ["biogenic", "atmospheric_dac"],
            "construction": {
                "kind": "treatment_origin_operation",
                "carbon_attribution_status": "separable_treated_plastic_carbon",
                "base": {
                    "source_role": "native_ecoinvent_provider",
                    "selector": selector,
                },
                "replace_biosphere": replacement_rules,
            },
        }
        catalogue["services"] = [
            *cast(list[dict[str, object]], catalogue["services"]),
            fossil_service,
            non_fossil_service,
        ]
        services = {
            str(service["provider_service_id"]): service
            for service in cast(list[dict[str, object]], catalogue["services"])
        }
        ordinary = native_resolution._ActivityResolver(bd, "ordinary")
        context = recipe_compilation._RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2025,
            provider_database="treatment_frontier",
            source=ordinary,
            ordinary=ordinary,
            technology=ordinary,
            biosphere=native_resolution._FlowResolver(bd, "biosphere"),
            material_recipe=physical_material_recipe(),
            catalogue=catalogue,
            method_authorities={
                str(row["method_authority_id"]): row
                for row in provider_catalogue._method_authority_rows(catalogue)
            },
            complete_source_ids=provider_catalogue._service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        database_name = "treatment_frontier"
        fossil = recipe_compilation._service_dataset(
            context,
            services["incineration_fossil"],
            services,
            database_name,
        )
        generated = recipe_compilation._service_dataset(
            context,
            services["incineration_non_fossil"],
            services,
            database_name,
        )
        fossil_owners = json.loads(str(fossil["direct_treatment_carbon_owners_json"]))
        generated_owners = json.loads(str(generated["direct_treatment_carbon_owners_json"]))
        assert [(row["owner_database"], row["owner_code"]) for row in fossil_owners] == [
            ("ordinary", "treatment")
        ]
        assert [(row["owner_database"], row["owner_code"]) for row in generated_owners] == [
            (
                database_name,
                provider_catalogue.provider_service_code("incineration_non_fossil"),
            )
        ]
        generated_exchanges = cast(list[dict[str, object]], generated["exchanges"])
        assert not any(
            exchange["input"] == ("ordinary", "treatment") for exchange in generated_exchanges
        )
        by_input = {
            cast(tuple[str, str], exchange["input"]): exchange
            for exchange in generated_exchanges
            if exchange["type"] != "production"
        }
        assert set(by_input) == {
            ("ordinary", "energy"),
            ("biosphere", "co2_non_fossil"),
            ("biosphere", "ch4_non_fossil"),
            ("biosphere", "co_non_fossil"),
            ("biosphere", "nitrogen_oxides"),
        }
        assert by_input[("ordinary", "energy")]["amount"] == pytest.approx(0.25)
        assert by_input[("ordinary", "energy")]["allocation"] == "cut-off"
        assert by_input[("biosphere", "nitrogen_oxides")]["amount"] == pytest.approx(0.01)
        for code, amount, loc in (
            ("co2_non_fossil", 2.2, 0.7),
            ("ch4_non_fossil", 0.08, -2.5),
            ("co_non_fossil", 0.028, -3.0),
        ):
            exchange = by_input[("biosphere", code)]
            assert exchange["amount"] == pytest.approx(amount)
            assert exchange["uncertainty type"] == 2
            assert exchange["loc"] == pytest.approx(loc)
            assert exchange["scale"] == pytest.approx(0.1)
            assert exchange["allocation"] == "native treatment allocation"
            assert exchange["name"] == flow_rows[code][0]

        treatment_rows = [
            row
            for row in context.semantic_rows
            if row.get("ledger_entry_type") == "treatment_carbon_mapping"
        ]
        assert len(treatment_rows) == 9
        assert {
            (row["provider_service_id"], row["ultimate_carbon_origin"]) for row in treatment_rows
        } == {
            ("incineration_fossil", "fossil"),
            ("incineration_non_fossil", "biogenic"),
            ("incineration_non_fossil", "atmospheric_dac"),
        }
        fossil_rows = [row for row in treatment_rows if row["ultimate_carbon_origin"] == "fossil"]
        assert all(
            row["source_flow_code"] == row["replacement_flow_code"]
            and row["mapping_status"] == "native_fossil_identity"
            for row in fossil_rows
        )
        assert all(row["source_amount"] == row["generated_amount"] for row in treatment_rows)
        assert all(
            row["source_carbon_kg_per_provider_unit"]
            == row["generated_carbon_kg_per_provider_unit"]
            for row in treatment_rows
        )
        assert all(row["uncertainty_metadata_preserved"] for row in treatment_rows)
        assert all(row["allocation_metadata_preserved"] for row in treatment_rows)

        bd.Database(database_name).write(
            {
                (database_name, provider_catalogue.provider_service_code(service_id)): dataset
                for service_id, dataset in (
                    ("incineration_fossil", fossil),
                    ("incineration_non_fossil", generated),
                )
            },
            searchable=False,
        )
        activations = pd.DataFrame.from_records(
            [
                {
                    "year": 2025,
                    "region": "EUR",
                    "life_cycle_stage": "incineration_treatment",
                    "material": "HDPE",
                    "constituent": "hdpe",
                    "product": "polymer",
                    "virgin_feedstock_route": "all",
                    "good": "all",
                    "material_cycle_source": "not_applicable",
                    "ultimate_carbon_origin": origin,
                    "lca_fate": "incineration",
                    "fate_origin": "remind_mfa",
                    "provider_database": database_name,
                    "provider_code": provider_catalogue.provider_service_code(service_id),
                    "provider_anchor_year": 2025,
                    "amount": 1.0,
                    "mfa_carbon_amount_kg": 6.0 / 7.0,
                }
                for origin, service_id in (
                    ("fossil", "incineration_fossil"),
                    ("biogenic", "incineration_non_fossil"),
                    ("atmospheric_dac", "incineration_non_fossil"),
                )
            ]
        )
        direct = direct_treatment_carbon_flows(bd, activations)
        consistency = carbon_consistency_table(bd, activations)
        assert len(direct) == 9
        for origin in ("fossil", "biogenic", "atmospheric_dac"):
            rows = direct.loc[direct["mfa_carbon_origin"] == origin].set_index("carbon_species")
            assert rows["flow_amount"].to_dict() == pytest.approx(
                {"carbon dioxide": 2.2, "methane": 0.08, "carbon monoxide": 0.028}
            )
            assert rows["carbon_amount"].to_dict() == pytest.approx(
                {"carbon dioxide": 0.6, "methane": 0.06, "carbon monoxide": 0.012}
            )
            assert set(rows["flow_carbon_origin"]) == {
                "fossil" if origin == "fossil" else "non_fossil"
            }
            assert set(rows["flow_compartment_json"]) == {'["air","urban air close to ground"]'}
        assert consistency["native_direct_selected_carbon_kg"].tolist() == pytest.approx(
            [0.672] * 3
        )

        biosphere_inventory = pd.DataFrame(
            [
                {
                    "year": 2025,
                    "flow_id": int(bd.get_node(database=input_key[0], code=input_key[1]).id),
                    "amount": float(exchange["amount"]),
                }
                for input_key, exchange in by_input.items()
                if exchange["type"] == "biosphere"
            ]
        )
        principal = PrincipalInventory(
            timeline=pd.DataFrame(),
            inventory=biosphere_inventory,
            base_static_score=0.0,
            provider_matrix_count=1,
            provider_count=1,
            provider_solve_batch_count=1,
        )
        ordinary_method = ImpactCategory(
            selection_order=1,
            method_id="bounded_climate",
            method_version="ordinary",
            impact_category_id="climate_ordinary",
            method_tuple=("bounded", "climate ordinary"),
            unit="kilogram carbon dioxide equivalent",
            factor_identity="bounded_ordinary",
        )
        non_fossil_co2_id = int(bd.get_node(database="biosphere", code="co2_non_fossil").id)
        inventory_before = principal.inventory.copy(deep=True)
        characterized = characterize_principal_inventory(
            principal,
            (ordinary_method,),
            CharacterizationFactorBank(
                factors={
                    ordinary_method.method_tuple: {non_fossil_co2_id: 0.0},
                }
            ),
            foreground_scenario="bounded_treatment",
            background_scenario="remind_ssp2_npi",
            background_dimensions={},
            region="EUR",
        )
        pd.testing.assert_frame_equal(principal.inventory, inventory_before)
        assert characterized.scores["score"].tolist() == pytest.approx([0.0])
