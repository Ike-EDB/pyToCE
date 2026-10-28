"""Protect PA ingredient charges, dated native supplies and known receiving loads."""

import json
import math
from pathlib import Path
from typing import cast

import pytest
from material_recipe_fixtures import physical_material_recipe
from pa_ester_ingredients_fixtures import (
    PA_ESTER_SERVICES,
    ester_biosphere,
    ester_native_inventory,
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

# Retained expected ingredient charges and service demands, per kg delivered ingredient.
# These use the admitted batch and nominal-product reference bases.
_INPUTS_KG = (
    {
        "glycerol": 0.778794992175274,
        "fatty_acid": 0.812222222222222,
        "potassium_carbonate": 0.00947183098591549,
        "sulfuric_acid": 0.00672134942288972,
    },
    {
        "fatty_acid": 1.45130807761861,
        "prepared_clay": 0.106235751281682,
        "lithium_carbonate": 0.00420879342509397,
        "phosphoric_acid": 0.0130617726985675,
        "water": 0.0468772509070811,
        "hmda": 0.0337264721810586,
    },
)
_SERVICE_DEMANDS = (
    {
        "pa_ester_operation": 1.29888888888889,
        "pa_finish_nitrogen": 0.019,
        "pa_condensate_collection": -0.000148806597756422,
        "pa_diglycerol_bottoms": -0.396077742878849,
        "pa_dioleate_settling": -0.0593099041307334,
    },
    {
        "pa_ester_operation": 0.976730336237325,
        "pa_amide_operation": 1.0,
        "pa_finish_nitrogen": 0.0185578763885092,
        "pa_condensate_collection": -0.0000583602261394681,
        "pa_dimer_monomer": -0.474577741381286,
        "pa_dimer_mineral_residue": -0.119973318766891,
    },
)


@pytest.mark.parametrize("future", [False, True])
def test_ester_ingredients_connect_actual_supplies_operations_and_receiving(
    tmp_path: Path, future: bool
) -> None:
    """Compile ingredient recipes through the dispatcher using synthetic source inventories.

    The component test checks complete supplier purchases and signed demands to
    the treatment inventories declared in the catalog. The connected PA test
    compiles those inventories and solves their LCI; no substitute treatment is used.
    Nonunit original/future inventories expose duplicate normalization and
    wrong-date freight while protecting native uncertainty and unchanged stores.
    """
    catalogue = load_provider_catalogue()
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], catalogue["services"])
    }
    complete_sources = _service_source_closures(catalogue)
    names = BrightwayNames(tmp_path / "brightway", "pa_ester_ingredients", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(ester_biosphere(), searchable=False)
        bd.Database("original").write(
            ester_native_inventory(catalogue, "original", future=False), searchable=False
        )
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(
                ester_native_inventory(catalogue, database, future=True), searchable=False
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
            complete_source_ids=complete_sources,
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        for index, sid in enumerate(PA_ESTER_SERVICES[1:]):
            dataset = _service_dataset(context, services[sid], services, "providers")
            exchanges = cast(list[dict[str, object]], dataset["exchanges"])
            physical = json.loads(str(dataset["pa_ester_reference_json"]))
            inputs = {
                cast(tuple[str, str], ex["input"]): ex
                for ex in exchanges
                if ex["type"] == "technosphere"
            }
            direct = {
                cast(tuple[str, str], ex["input"])[1]: float(cast(float, ex["amount"]))
                for ex in exchanges
                if ex["type"] == "biosphere"
            }
            expected = {(database, field): amount for field, amount in _INPUTS_KG[index].items()}
            expected.update(
                {
                    ("providers", provider_service_code(child)): amount
                    for child, amount in _SERVICE_DEMANDS[index].items()
                }
            )
            if index == 1:
                chemical = _INPUTS_KG[index]["lithium_carbonate"]
                for mode in range(4):
                    amount = chemical * (mode + 1) * (0.04 if future else 0.03)
                    expected[(database, f"freight_{mode}")] = amount
                    row = inputs[(database, f"freight_{mode}")]
                    assert row["loc"] == pytest.approx(math.log(amount) + 0.02)
                    assert row["scale"] == 0.17
                    assert row["minimum"] == pytest.approx(amount / 2)
                    assert row["maximum"] == pytest.approx(amount * 2)
                    assert row["allocation"] == "invented native allocation"
            assert {
                key: float(cast(float, ex["amount"])) for key, ex in inputs.items()
            } == pytest.approx(expected, rel=2e-13)
            assert direct["co2"] == pytest.approx(
                (0.00301615004029645, 0.00250683182444999)[index],
                rel=2e-13,
            )
            assert set(direct) == {"co2"}
            production = next(ex for ex in exchanges if ex["type"] == "production")
            assert production["input"] == ("providers", provider_service_code(sid))
            assert production["amount"] == 1.0
            properties = cast(dict[str, dict[str, object]], production["properties"])
            assert properties["wet mass"]["amount"] == 1.0
            assert properties["dry mass"]["amount"] == 1.0
            assert properties["water in wet mass"]["amount"] == 0.0
            fossil_atoms, bio_atoms, mass = (
                (0, 42, 695.079),
                (6, 216, 3445.602),
            )[index]
            assert properties["carbon content, fossil"]["amount"] == pytest.approx(
                fossil_atoms * 12.011 / mass
            )
            assert properties["carbon content, non-fossil"]["amount"] == pytest.approx(
                bio_atoms * 12.011 / mass
            )
            assert physical["technical_manufacturing_function_is_not_nominal_molecular_assay"]
            # Check gross material mass on the technical DGD/dimer reference.
            assert sum(_INPUTS_KG[index].values()) == pytest.approx(
                1
                + physical["condensate_kg"]
                + sum(physical["residue_wet_kg"].values())
                + sum(direct.values()),
                rel=2e-13,
            )
            residues = physical["residues"]
            if index == 0:
                assert residues["diglycerol_residue"]["known_compounds_kg"] == pytest.approx(
                    {"K2SO4": 0.0119423751167207}
                )
                assert residues["diglycerol_residue"][
                    "aggregate_organic_elements_kg"
                ] == pytest.approx(
                    {"C": 0.175091577531131, "H": 0.0320933034460065, "O": 0.176950486784991}
                )
                assert residues["settling_residue"]["known_compounds_kg"] == pytest.approx(
                    {
                        "C3H8O3": 0.00177929712392200,
                        "C6H14O5": 0.0545651118002747,
                        "C9H20O7": 0.00296549520653667,
                    }
                )
            else:
                assert residues["mineral_residue"]["known_compounds_kg"] == pytest.approx(
                    {"LiH2PO4": 0.0118395967896022, "H3PO4": 0.00189797069560610}
                )
                assert residues["mineral_residue"]["prepared_mineral_kg"] == pytest.approx(
                    0.106235751281682
                )
                assert residues["monomer_residue"]["biogenic_kg_c"] == pytest.approx(
                    0.363236750821867
                )
        assert {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        } == before


@pytest.mark.parametrize("future", [False, True])
def test_tridecyl_keeps_complete_dated_doa_and_original_acid_utilization(
    tmp_path: Path, future: bool
) -> None:
    """Compile actual target feeds, full native operation and alcohol freight.

    The synthetic original 0.4 kg acid/kg dioctyl adipate (DOA) gives g=1.0142847
    and the two expected target demands. The future donor deliberately has a
    different acid input and nonunit production, so using its feed quantities
    or dividing target demand by supplier production fails this witness.
    """
    catalogue = load_provider_catalogue()
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], catalogue["services"])
    }
    names = BrightwayNames(tmp_path / "brightway", "pa_tridecyl", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(ester_biosphere(), searchable=False)
        bd.Database("original").write(
            ester_native_inventory(catalogue, "original", future=False), searchable=False
        )
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(
                ester_native_inventory(catalogue, database, future=True), searchable=False
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
        sid = "pa_tridecyl_stearate"
        dataset = _service_dataset(context, services[sid], services, "providers")
        exchanges = cast(list[dict[str, object]], dataset["exchanges"])
        physical = json.loads(str(dataset["pa_tridecyl_stearate_reference_json"]))
        assert physical["original_adipic_acid_kg_per_kg_doa"] == 0.4
        assert physical["original_main_molar_utilization"] == pytest.approx(
            1.01428473676287446456186448796, rel=2e-14
        )
        # Expected feeds use original acid molar utilization for C18H36O2 and C13H28O per C31H62O2.
        expected_feeds = {
            "stearic_acid": 0.618093714167210211694533308327,
            "fatty_alcohol": 0.435331917200360095036581533079,
        }
        assert physical["native_inputs_kg"] == pytest.approx(expected_feeds, rel=2e-14)
        assert physical["native_environmental_function_is_target_proxy"]
        assert physical["main_feed_uncertainty_is_original_acid_proxy"]
        assert physical["technical_manufacturing_function_is_not_nominal_molecular_assay"]
        production = exchanges[0]
        assert production["input"] == ("providers", provider_service_code(sid))
        assert production["amount"] == 1.0
        properties = cast(dict[str, dict[str, object]], production["properties"])
        assert properties["wet mass"]["amount"] == 1.0
        assert properties["dry mass"]["amount"] == 1.0
        assert properties["water in wet mass"]["amount"] == 0.0
        assert properties["carbon content, fossil"]["amount"] == pytest.approx(
            13 * 12.011 / 466.835
        )
        assert properties["carbon content, non-fossil"]["amount"] == pytest.approx(
            18 * 12.011 / 466.835
        )
        assert len(exchanges) == 32
        assert sum(ex["type"] == "technosphere" for ex in exchanges) == 18
        assert sum(ex["type"] == "biosphere" for ex in exchanges) == 13
        native = before[(database, "doa")]
        assert sum(ex["input"] == (database, "ethylene") for ex in native) == 22
        assert not any(
            ex["input"] in {(database, "ethylene"), (database, "adipic")} for ex in exchanges
        )
        # Compare every dated support/emission record, with duplicate steam,
        # all four signed wastewater rows and all three deionized-water rows.
        # Only fossil process CO2 gains its matched non-fossil counterpart.
        expected_operation = []
        output = 5.0 if future else 2.0
        for ex in native:
            if ex["type"] == "production" or ex["input"] in {
                (database, "adipic"),
                (database, "ethylene"),
            }:
                continue
            parts = (
                (("co2", 13 / 31), ("co2_bio", 18 / 31))
                if ex["input"] == ("biosphere", "co2")
                else ((None, 1.0),)
            )
            for co2_code, fraction in parts:
                expected = {key: value for key, value in ex.items() if key != "output"}
                factor = fraction / output
                expected["amount"] = float(ex["amount"]) * factor
                expected["loc"] = float(ex["loc"]) + math.log(factor)
                expected["minimum"] = float(ex["minimum"]) * factor
                expected["maximum"] = float(ex["maximum"]) * factor
                if co2_code == "co2_bio":
                    expected.update(
                        input=("biosphere", "co2_bio"),
                        name="Carbon dioxide, non-fossil",
                        unit="kilogram",
                    )
                    if not future:
                        expected["properties"] = {
                            "carbon content, fossil": {"amount": 0.0, "unit": "dimensionless"},
                            "carbon content, non-fossil": {"amount": 0.27, "unit": "dimensionless"},
                            "dry mass": {"amount": 1.0, "unit": "kg"},
                        }
                expected_operation.append(expected)
        assert exchanges[1:26] == expected_operation
        main_rows = exchanges[26:28]
        for row, (field, amount) in zip(main_rows, expected_feeds.items(), strict=True):
            assert row["input"] == (database, field)
            assert row["name"] == bd.get_node(database=database, code=field)["name"]
            assert row["unit"] == "kilogram"
            assert row["amount"] == pytest.approx(amount, rel=2e-14)
            assert row["loc"] == pytest.approx(math.log(amount) + 0.02)
            assert row["scale"] == 0.17
            assert row["scale without pedigree"] == 0.03
            assert row["pedigree"] == {"reliability": 2, "completeness": 5}
            assert row["minimum"] == pytest.approx(amount / 2)
            assert row["maximum"] == pytest.approx(amount * 2)
            assert not set(row).intersection(
                {
                    "properties",
                    "flow",
                    "activity",
                    "CAS number",
                    "classifications",
                    "comment",
                    "allocation",
                }
            )
        for index, row in enumerate(exchanges[28:]):
            amount = expected_feeds["fatty_alcohol"] * (index + 1) * (0.04 if future else 0.03)
            assert row["input"] == (database, f"freight_{index}")
            assert row["amount"] == pytest.approx(amount)
            assert row["loc"] == pytest.approx(math.log(amount) + 0.02)
            assert row["scale"] == 0.17
            assert row["allocation"] == "invented native allocation"
        operation_trace = next(
            row
            for row in context.technology_rows
            if row["component_id"] == "integrated_tridecyl_stearate_environmental_function"
        )
        assert len(json.loads(str(operation_trace["removed_technosphere_json"]))) == 23
        assert len(json.loads(str(operation_trace["retained_technosphere_json"]))) == 12
        assert len(json.loads(str(operation_trace["retained_biosphere_json"]))) == 12
        assert {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        } == before
