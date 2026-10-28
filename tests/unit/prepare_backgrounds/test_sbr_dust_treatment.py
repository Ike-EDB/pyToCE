"""Actual captured-powder chemistry and complete linked HWI/ash boundaries."""

import json
from math import fsum, log
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from sbr_dust_fixtures import sbr_dust_biosphere, sbr_dust_native_inventory

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
    _route_contexts,
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext

_DUST = "sbr_captured_mixer_dust_incineration"
_ASH = "sbr_captured_mixer_dust_ash_treatment"


@pytest.mark.parametrize("original", (True, False))
def test_captured_dust_has_one_complete_treatment_with_fixed_actual_composition(
    tmp_path: Path, original: bool
) -> None:
    """Compile and solve synthetic original/future inventories through real Brightway.

    Invented source productions and supplier groups differ by date. Independent
    values below come from the published nominal compound-22 powder
    recipe; source doses and native emissions are deliberately artificial.
    """
    catalogue = load_provider_catalogue()
    services = {
        cast(str, row["provider_service_id"]): row
        for row in cast(list[dict[str, Any]], catalogue["services"])
    }
    # Three conversion routes share the fixed auxiliary-dust service across principal origins.
    routes = _route_contexts(cast(list[dict[str, object]], catalogue["routes"]), services)[_DUST]
    assert routes
    assert {row["ultimate_carbon_origin"] for row in routes} == {
        "fossil",
        "biogenic",
        "atmospheric_dac",
    }
    assert {row["virgin_feedstock_route"] for row in routes} == {"all"}
    hwi_recipe = services[_DUST]["construction"]
    assert hwi_recipe["source"] == {
        "name": "treatment of hazardous waste, hazardous waste incineration",
        "reference_product": "hazardous waste, for incineration",
        "unit": "kilogram",
        "exact_location": "CH",
    }
    assert services[_ASH]["construction"]["source"]["exact_location"] == "RoW"
    names = BrightwayNames(tmp_path / "brightway", "sbr_dust", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        from bw2calc import LCA

        bd.Database("biosphere").write(sbr_dust_biosphere(), searchable=False)
        bd.Database("original").write(
            sbr_dust_native_inventory(catalogue, "original", original=True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                sbr_dust_native_inventory(catalogue, database, original=False), searchable=False
            )
        before = {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        }
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                cast(str, row["method_authority_id"]): row
                for row in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        ids = (_DUST, _ASH)
        compiled = {
            ("providers", provider_service_code(sid)): _service_dataset(
                context, services[sid], services, "providers"
            )
            for sid in ids
        }
        bd.Database("providers").write(compiled, searchable=False)
        # These direct auxiliary owners are absent from the principal-polymer
        # treatment diagnostic. Their two original carbon pools stay in LCI.
        for sid in (_DUST, _ASH):
            record = compiled["providers", provider_service_code(sid)]
            assert record["provider_stage_owner"] == "manufacturing_auxiliary_treatment"
            assert "direct_treatment_carbon_owners_json" not in record
        hwi, ash = [
            bd.get_node(database="providers", code=provider_service_code(sid))
            for sid in (_DUST, _ASH)
        ]
        hwi_p, ash_p = (3, 5) if original else (6, 10)
        assert next(iter(hwi.production()))["amount"] == -hwi_p
        assert next(iter(ash.production()))["amount"] == -ash_p
        trace = next(
            row
            for row in context.technology_rows
            if row.get("component_id") == "captured_sbr_dust_hwi"
        )
        load = json.loads(cast(str, trace["project_terms_json"]))
        # Original eight dry charges total .788 kg/kg gross principal. These
        # independent nominal fractions include ZnO oxygen and exclude oil.
        expected_elements = {
            "C": 0.8218368573253222,
            "H": 0.010809364058856217,
            "O": 0.027810116064133774,
            "N": 0.006067157725955717,
            "S": 0.031523269206159624,
            "Zn": 0.10195323561957254,
        }
        assert load["elements"] == pytest.approx(expected_elements, rel=2e-13)
        assert fsum(load["elements"].values()) == pytest.approx(1, abs=3e-15)
        carbon, nitrogen, sulfur, zinc = (load["elements"][el] for el in ("C", "N", "S", "Zn"))
        biogenic = 0.02 / 0.788 * 216 / 284
        assert load["carbon"] == pytest.approx(
            {"non-fossil": biogenic, "fossil": carbon - biogenic}
        )
        assert load["lhv"] == pytest.approx(28.516189282237544, rel=2e-14)
        # Use retained receiving coefficients for actual and donor capture denominators.
        captured = (
            0.083 * load["elements"]["O"]
            + 0.0110102 * carbon
            + 0.011 * nitrogen
            + 0.9994 * sulfur
            + 0.9993 * zinc
        )
        donor_capture = fsum(
            (
                0.083 * 0.04,
                0.0110102 * 0.3,
                0.9994 * 0.02,
                0.011 * 0.01,
                0.9993 * 0.005,
                0.88 * 0.001,
                0.9997 * 0.05,
                0.9995 * 0.003,
                (1 - 0.00899) * 0.00001,
                0.9993 * 0.00002,
                (1 - 0.0000000739) * 0.00003,
                0.9993 * 0.00004,
                (1 - 0.0402) * 0.000005,
                0.9993 * 0.00002,
                (1 - 0.00331) * 0.00005,
                0.9993 * 0.0001,
                (1 - 0.00233) * 0.001,
            )
        )
        donor_metals = donor_capture - fsum(
            (
                0.083 * 0.04,
                0.0110102 * 0.3,
                0.9994 * 0.02,
                0.011 * 0.01,
                0.9993 * 0.005,
                0.88 * 0.001,
                0.9997 * 0.05,
                0.9995 * 0.003,
                (1 - 0.00233) * 0.001,
            )
        )
        assert load["reagent"]["HCl"] == pytest.approx(0.02 * captured / donor_capture)
        assert load["reagent"]["PAM"] == pytest.approx(0.01 * captured / donor_capture)
        assert load["reagent"]["FeCl3"] == pytest.approx(0.004 * 0.9993 * zinc / donor_metals)
        assert load["reagent"]["H2O2"] == pytest.approx(0.0002 * sulfur / 0.02)
        hwi_inputs = {}
        for ex in hwi.technosphere():
            hwi_inputs[ex.input.key] = hwi_inputs.get(ex.input.key, 0) + ex["amount"] / hwi_p
        assert hwi_inputs[ash.key] == pytest.approx(-load["ash_mass"])
        titanium = fsum(
            ex["amount"] / hwi_p
            for ex in hwi.technosphere()
            if ex.input["reference product"] == "titanium dioxide"
        )
        assert titanium == pytest.approx(0.0002 + 0.0001 * nitrogen / 0.01)
        assert hwi_inputs[database, "mc_v_proxy"] == pytest.approx(
            0.00005 + 0.00002 * nitrogen / 0.01
        )
        assert hwi_inputs[database, "mc_ng"] == pytest.approx(0.3 + 0.02 * nitrogen / 0.01)
        assert {key[1] for key in hwi_inputs}.isdisjoint(
            {
                "mc_oil",
                "mc_cement_a",
                "mc_ash_support_a",
                "mc_ash_facility",
                "mc_hydrated_cement",
                "mc_organic",
                "dust_calcium_chloride",
                "mc_fixed",
            }
        )
        # Source normalization preserves every dated HCl supplier fraction,
        # original uncertainty location offset, spread and scaled bounds.
        hcl_rows = [
            ex
            for ex in hwi.technosphere()
            if ex.input["reference product"].startswith("hydrochloric acid,")
        ]
        assert len(hcl_rows) == (1 if original else 2)
        assert fsum(ex["amount"] for ex in hcl_rows) / hwi_p == pytest.approx(
            load["reagent"]["HCl"]
        )
        if not original:
            assert hwi_inputs[database, "dust_second_hcl"] / hwi_inputs[
                database, "mc_hcl"
            ] == pytest.approx(3)
        for ex in hcl_rows:
            offset = 0.02 if ex.input.key[1] == "dust_second_hcl" else 0.03
            assert ex["loc"] == pytest.approx(log(ex["amount"]) + offset)
            assert ex["scale"] == (0.19 if ex.input.key[1] == "dust_second_hcl" else 0.21)
        # Dated waste-N supplier shares change; fixed catalyst
        # quantities continue to use the original ash-composition reference.
        if not original:
            assert hwi_inputs[database, "dust_second_ti"] == pytest.approx(
                0.75 * 0.0001 * nitrogen / 0.01
            )
        # Disposal mass includes full oxides, PAM, and installed catalyst once.
        expected_b = (
            0.0110001 * carbon
            + 0.01 * nitrogen * 62 / 14
            + 0.0428 * sulfur * 80 / 32
            + 0.986 * zinc * (1 + 16 / 65.38)
            + load["reagent"]["PAM"]
            + load["reagent"]["CaO"]
            + load["fresh"]["FeCl3"]["Fe"] * (1 + 24 / 55.845)
            + load["reagent"]["TiO2"]
            + load["reagent"]["V2O5"]
        )
        assert load["ash_mass"] == pytest.approx(expected_b, rel=2e-14)
        ash_inputs = {ex.input.key[1]: ex["amount"] / ash_p for ex in ash.technosphere()}
        assert ash_inputs == pytest.approx(
            {
                "mc_cement_a": 0.25,
                "mc_cement_b": 0.15,
                "mc_ash_support_a": 0.25 if original else 0.8,
                "mc_ash_support_b": 0.75 if original else 0.2,
                "mc_ash_facility": 0.0003,
                "mc_hydrated_cement": -1,
                "mc_tap_a": 0.6,
            }
        )
        emissions = {}
        for owner, p, mass in ((hwi, hwi_p, 1), (ash, ash_p, load["ash_mass"])):
            for ex in owner.biosphere():
                key = ex.input["name"], tuple(ex.input["categories"])
                emissions[key] = emissions.get(key, 0) + ex["amount"] / p * mass
        surface, long = ("water", "surface water"), ("water", "ground-, long-term")
        pam_c, pam_n = load["fresh"]["PAM"]["C"], load["fresh"]["PAM"]["N"]
        for phase, original_doc, pam_fraction, nitrate, ash_nitrate in (
            (surface, 0.00003, 0.1, 0.0001, 0.006),
            (long, 0.002, 0.4, 0.0002, 0.012),
        ):
            assert emissions["DOC, Dissolved Organic Carbon", phase] == pytest.approx(
                original_doc * carbon / 0.3 + pam_fraction * pam_c
            )
            assert emissions["COD, Chemical Oxygen Demand", phase] == pytest.approx(
                original_doc * carbon / 0.3 * 32 / 12 + 3 * pam_fraction * pam_c
            )
            assert emissions["Nitrate", phase] == pytest.approx(
                nitrate * nitrogen / 0.01 + pam_n * ash_nitrate / 0.03
            )
        assert emissions["Sulfate", surface] == pytest.approx(
            0.957 * sulfur * 3 + 0.0428 * sulfur * 0.012 / 0.04
        )
        assert emissions["Sulfate", long] == pytest.approx(0.0428 * sulfur * 0.024 / 0.04)
        assert emissions["Zinc II", surface] == pytest.approx(
            0.0134 * zinc + 0.986 * zinc * 0.004 / 0.01
        )
        assert emissions["Iron II", surface] == pytest.approx(load["ash"]["Fe"] * 0.0005 / 0.025)
        assert emissions["Calcium II", surface] == pytest.approx(
            load["fresh"]["CaO"]["Ca"] * 0.08 / 0.2
        )
        assert emissions["Titanium ion", long] == pytest.approx(
            load["ash"]["Ti"] * (0.2973 - 0.0004956)
        )
        assert emissions[
            "Organic carbon, placed in landfill", ("inventory indicator", "waste")
        ] == pytest.approx(0.0110001 * carbon + pam_c)
        assert emissions["Water", ("air",)] == pytest.approx(load["water"]["air"])
        assert emissions["Water", ("water",)] == pytest.approx(load["water"]["water"])
        # Preserve the fixed HWI source's 5 L invented well-water meter.
        abstracted = emissions["Water, well, in ground", ("natural resource", "in water")]
        assert abstracted == pytest.approx(0.005)
        assert (
            emissions["Water", ("air",)] + emissions["Water", ("water",)] - abstracted
        ) * 1000 == pytest.approx(load["water"]["formed_kg"], rel=2e-13)
        assert emissions["Heat, waste", ("air", "urban air close to ground")] == pytest.approx(
            0.83 * load["heat"]
        )
        assert emissions["Heat, waste", surface] == pytest.approx(0.17 * load["heat"])
        # LCA solves linked negative-production treatments and positive catalyst supplies.
        lca = LCA({hwi: -1})
        lca.lci()
        supply_array = cast(NDArray[np.float64] | None, lca.supply_array)
        assert supply_array is not None
        assert supply_array[lca.dicts.activity[hwi.id]] == pytest.approx(1 / hwi_p, rel=2e-7)
        assert supply_array[lca.dicts.activity[ash.id]] == pytest.approx(
            load["ash_mass"] / ash_p, rel=2e-7
        )
        for code in ("mc_tio2", "mc_v_proxy", "mc_pam"):
            node = bd.get_node(database=database, code=code)
            assert supply_array[lca.dicts.activity[node.id]] == pytest.approx(
                hwi_inputs[node.key], rel=2e-7
            )
        for row in context.semantic_rows:
            if row["provider_service_id"] in (_DUST, _ASH):
                assert row["semantic_class"] == "service_normalization"
        assert before == {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        }
