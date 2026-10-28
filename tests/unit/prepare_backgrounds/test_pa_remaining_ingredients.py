"""Check complete FAS delivery and actual PA phosphate workup connections."""

import math
from pathlib import Path
from typing import cast

import pytest
from material_recipe_fixtures import physical_material_recipe
from pa_remaining_ingredients_fixtures import (
    FAS_SUPPLIERS,
    remaining_ingredients_biosphere,
    remaining_ingredients_native_inventory,
    remaining_phosphate_boundary,
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

# Expected POCl3-workup charges are per kg nominal C24H52NO7P salt. Source recovery is 0.90. NaOH
# supply uses active mass; hydrolysis, neutralizer, and wash water are supplied separately.
_PHOSPHATE_INPUTS_KG = {
    "oleyl": 0.5994462203379577,
    "pocl3": 0.3603406254058194,
    "tea": 0.2997867420935915,
    "hexane": 0.3525314382973219,
    "mibk": 0.3525314382973219,
    "naoh": 0.2820023720304908,
    "di_water": 2.543094710639376,
}


@pytest.mark.parametrize("future", [False, True])
def test_fas_keeps_full_native_manufacture_and_actual_solution_distribution(
    tmp_path: Path, future: bool
) -> None:
    """Solve actual FAS resources with nonunit original/future native sources.

    The complete small native LCI includes utility, signed waste, infrastructure
    and freight. None of the eight material-supply rows belongs to distribution.
    """
    catalogue = load_provider_catalogue()
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], catalogue["services"])
    }
    names = BrightwayNames(
        tmp_path / "brightway", "pa_remaining_ingredients", "biosphere", "original"
    )
    with selected_brightway_project(names) as bd:
        import bw2calc as bc

        bd.Database("biosphere").write(remaining_ingredients_biosphere(), searchable=False)
        bd.Database("original").write(
            remaining_ingredients_native_inventory("original", future=False), searchable=False
        )
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(
                remaining_ingredients_native_inventory(database, future=True), searchable=False
            )
        before = {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        }
        dated = _ActivityResolver(bd, database)
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
        distribution_id = "fatty_alcohol_sulfate_distribution"
        fas_id = "fatty_alcohol_sulfate_28_percent"
        distribution, fas = (
            _service_dataset(context, services[sid], services, "providers")
            for sid in (distribution_id, fas_id)
        )
        distribution_rows = cast(list[dict[str, object]], distribution["exchanges"])
        output = 5.0 if future else 2.0
        production = next(ex for ex in distribution_rows if ex["type"] == "production")
        native_production = next(
            ex for ex in before[database, "fas_market"] if ex["type"] == "production"
        )
        expected_production = {
            field: value
            for field, value in native_production.items()
            if field
            not in {"output", "activity", "flow", "CAS number", "chemical formula", "properties"}
        }
        expected_production.update(
            input=("providers", provider_service_code(distribution_id)),
            name=services[distribution_id]["reference_product"],
            **{"reference product": services[distribution_id]["reference_product"]},
        )
        assert production == expected_production
        freight = {
            cast(tuple[str, str], ex["input"]): ex
            for ex in distribution_rows
            if ex["type"] == "technosphere"
        }
        assert set(freight) == {(database, f"freight_{index}") for index in range(4)}
        assert len(distribution_rows) == 5
        market_rows = before[database, "fas_market"]
        # All eight suppliers exist in the native market;
        # removing only the selected petrochemical supplier would double supply.
        assert {row["input"] for row in market_rows if row["input"][1].startswith("fas_")} == {
            (database, code) for code, _name, _location in FAS_SUPPLIERS
        } | {(database, "fas_market")}
        fields = (
            "amount",
            "uncertainty type",
            "loc",
            "scale",
            "minimum",
            "maximum",
            "negative",
            "allocation",
        )
        for key, row in freight.items():
            original_row = next(ex for ex in market_rows if ex["input"] == key)
            assert {field: row[field] for field in fields} == {
                field: original_row[field] for field in fields
            }
        omissions = [
            row
            for row in context.semantic_rows
            if row.get("ledger_entry_type") == "native_exchange_omission"
            and row["provider_service_id"] == distribution_id
        ]
        assert len(omissions) == 8
        assert [row["source_unit_amount"] for row in omissions] == [output / 8] * 8

        fas_rows = cast(list[dict[str, object]], fas["exchanges"])
        fas_technosphere = {
            cast(tuple[str, str], ex["input"]): float(cast(float, ex["amount"]))
            for ex in fas_rows
            if ex["type"] == "technosphere"
        }
        # The active reference is one kg, but distribution transports the
        # entire 28 percent solution. The remaining 72 percent is carrier water;
        # native manufacture retains its own utilities and environmental releases.
        assert fas_technosphere == pytest.approx(
            {
                (database, "fas_petrochemical_rer"): 1.0,
                (database, "tap_water"): (1 / 0.28 - 1),
                ("providers", provider_service_code(distribution_id)): 1 / 0.28,
            }
        )
        assert (database, "fas_market") not in fas_technosphere
        fas_reference = cast(dict[str, object], services[fas_id]["representative_method"])
        assert fas_reference["active_mass_fraction"] == 0.28
        assert fas_reference["carried_water_kg_per_active_kg"] == pytest.approx(0.72 / 0.28)
        fas_direct = {
            cast(tuple[str, str], ex["input"]): float(cast(float, ex["amount"]))
            for ex in fas_rows
            if ex["type"] == "biosphere"
        }
        assert fas_direct == {}
        bd.Database("providers").write(
            {
                ("providers", provider_service_code(distribution_id)): distribution,
                ("providers", provider_service_code(fas_id)): fas,
            },
            searchable=False,
        )
        persisted_distribution = bd.get_node(
            database="providers", code=provider_service_code(distribution_id)
        )
        assert [dict(ex) for ex in persisted_distribution.production()] == [
            {**expected_production, "output": persisted_distribution.key}
        ]
        lca = bc.LCA({bd.get_node(database="providers", code=provider_service_code(fas_id)): 1.0})
        lca.lci()
        inventory = {
            str(flow["code"]): float(lca.inventory[lca.dicts.biosphere[flow.id], :].sum())
            for flow in bd.Database("biosphere")
        }
        expected_fas = (0.4 if future else 0.2) + 0.3 * 0.1 + 0.1 * 0.3 + 0.02 * 0.5
        expected_freight = (
            math.fsum((index + 1) ** 2 * (0.04 if future else 0.03) * 0.1 for index in range(4))
            / 0.28
        )
        assert inventory == pytest.approx(
            {
                "fas_marker": expected_fas,
                "freight_marker": expected_freight,
                "water_marker": (1 / 0.28 - 1) * (0.02 if future else 0.01),
                "chemical_marker": 0.0,
            }
        )

        assert {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        } == before


@pytest.mark.parametrize("future", [False, True])
def test_phosphate_keeps_actual_frontier_water_and_dated_native_precursors(
    tmp_path: Path, future: bool
) -> None:
    """Keep phosphate treatment linked when synthetic source activity codes change.

    The compiled inventory links phosphate production to its operation and to
    wastewater, sludge, ash and solvent treatment. The future fixture renames
    all source activity codes and references; descriptors and quantities stay fixed.
    The parent keeps positive precursor/operation purchases and negative
    composition-specific liquid and solvent demands, with no substitute node.
    """
    with remaining_phosphate_boundary(tmp_path, future=future) as (bd, provider, selected, before):
        catalogue = provider.catalogue
        services = {
            str(row["provider_service_id"]): row
            for row in cast(list[dict[str, object]], catalogue["services"])
        }
        database = "future" if future else "source"
        frontier_database = selected["phosphate"][0]
        phosphate_id = "pa_tea_monooleyl_phosphate"
        phosphate = bd.get_node(
            database=frontier_database, code=provider_service_code(phosphate_id)
        )
        assert {node.key for node in bd.Database(frontier_database)} == {
            (frontier_database, provider_service_code(service_id)) for service_id in services
        }
        if future:
            assert selected["oleyl"][1].startswith("future_")
        phosphate_rows = [dict(ex) for ex in phosphate.exchanges()]
        phosphate_inputs = {
            cast(tuple[str, str], ex["input"]): float(cast(float, ex["amount"]))
            for ex in phosphate_rows
            if ex["type"] == "technosphere"
        }
        expected = {selected[code]: amount for code, amount in _PHOSPHATE_INPUTS_KG.items()}
        expected.update(
            {
                selected["operation"]: 0.7002132579064085,
                selected["collection"]: -3.084670670507235 / 1000,
                selected["solvents"]: -0.705062876594644,
            }
        )
        assert phosphate_inputs == pytest.approx(expected)
        assert not any(ex["type"] == "biosphere" for ex in phosphate_rows)
        assert math.fsum(_PHOSPHATE_INPUTS_KG.values()) == pytest.approx(
            1 + 3.084670670507235 + 0.705062876594644
        )
        assert _PHOSPHATE_INPUTS_KG["di_water"] == pytest.approx(
            0.7050628765946439 + 0.2820023720304908 + 1.5560294620142411
        )
        # CIAAW nominal salt contains six fossil TEA carbons and eighteen biogenic oleyl carbons.
        # Native supply retains its allocation.
        molar_mass = 24 * 12.011 + 52 * 1.008 + 14.007 + 7 * 15.999 + 30.973761998
        phosphate_reference = cast(
            dict[str, object], services[phosphate_id]["representative_method"]
        )
        assert phosphate_reference["product_fossil_kg_c"] == pytest.approx(6 * 12.011 / molar_mass)
        assert phosphate_reference["product_biogenic_kg_c"] == pytest.approx(
            18 * 12.011 / molar_mass
        )
        assert services["pa_phosphate_water_collection_ch"]["unit"] == "cubic meter"
        assert services["pa_phosphate_solvent_residue"]["unit"] == "kilogram"
        construction = cast(dict[str, object], services[phosphate_id]["construction"])
        receiving = cast(dict[str, dict[str, float]], construction["receiving_loads"])
        assert receiving["water"] == pytest.approx(
            {
                "water_kg": 2.5833167850919088,
                "sodium_chloride_kg": 0.412037118360841,
                "monooleylphosphate_kg": 0.0778014731007121,
                "phosphoric_acid_kg": 0.0115152939537735,
            }
        )
        assert receiving["solvents"] == pytest.approx(
            {"C6H14": 0.3525314382973219, "C6H12O": 0.3525314382973219}
        )
        # The collector's 1000 kg/m3 reference includes water, organics, and salts.
        assert math.fsum(receiving["water"].values()) / 1000 == pytest.approx(
            -phosphate_inputs[selected["collection"]]
        )
        collection = bd.get_node(database=frontier_database, code=selected["collection"][1])
        # The prepared receiver must retain the actual phosphate wash, rather
        # than the average municipal composition used as its technology source.
        # Its production properties express kg water per m3 of mixed wash.
        assert collection.rp_exchange()["properties"]["water in wet mass"] == {
            "amount": pytest.approx(
                receiving["water"]["water_kg"] / -phosphate_inputs[selected["collection"]]
            ),
            "unit": "kg",
        }
        assert math.fsum(receiving["solvents"].values()) == pytest.approx(
            -phosphate_inputs[selected["solvents"]]
        )
        assert {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"source", database}
            for node in bd.Database(db)
        } == before
