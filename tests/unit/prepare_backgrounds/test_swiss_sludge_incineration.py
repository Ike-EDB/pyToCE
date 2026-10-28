"""Verify Swiss FLUWA, supplies and signed synthetic inventories through Brightway."""

import json
from math import fsum, log
from typing import cast

import numpy as np
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from swiss_sludge_fixtures import biosphere_inventory, catalogue, native_inventory, wastewater

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.ash_treatment import (
    _ash_dataset,
    _ash_market_dataset,
    _AshReleaseProfile,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_source_closures,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.swiss_sludge_incineration import (
    swiss_sludge_dataset,
    swiss_sludge_load,
)


@pytest.mark.parametrize("future", [False, True], ids=["native", "dated"])
@pytest.mark.parametrize("case", ["finish", "acetone", "pdo"])
def test_swiss_treatment_preserves_actual_phases_complete_support_and_signed_lci(
    tmp_path, future, case
):
    """Solve synthetic linked inventories and reconcile calculated treatment outputs."""
    authority = catalogue()
    services = {s["provider_service_id"]: s for s in authority["services"]}
    names = BrightwayNames(tmp_path / "brightway", "swiss_sludge", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere_inventory(), searchable=False)
        bd.Database("original").write(native_inventory("original", future=False), searchable=False)
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(native_inventory(database, future=True), searchable=False)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=authority,
            material_recipe=physical_material_recipe(),
            method_authorities={
                cast(str, r["method_authority_id"]): r for r in _method_authority_rows(authority)
            },
            complete_source_ids=_service_source_closures(authority),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        ww = wastewater(case)
        load = swiss_sludge_load(context, services["swiss_sludge"]["construction"], ww)
        # Report-organic O/C is a calorific reference only. Source dry mass,
        # FLUWA allocands and ash continue to use their total oxide O ledger.
        raw_organic_o = (
            0.628571428571428
            * (ww.raw_sludge.elements_kg["C"] - ww.duties.polyacrylamide_elements_kg["C"])
            + ww.duties.polyacrylamide_elements_kg["O"]
        )
        organic_o = ww.incineration_dry_share * (
            (1 - 0.89) * raw_organic_o + 0.628571428571428 * ww.digested_sludge.elements_kg["C"]
        )
        hhv = (
            34.016 * load.elements_kg["C"]
            + 124.265 * load.allocated_hydrogen_kg
            - 9.8324 * organic_o
            + 6.276 * load.elements_kg["N"]
            + 19.079 * load.elements_kg["S"]
        )
        assert load.thermal["calorific_organic_oxygen_kg"] == pytest.approx(organic_o)
        assert load.thermal["HHV_MJ"] == pytest.approx(hhv, rel=2e-12, abs=0)
        assert load.thermal["environmental_heat_MJ"] > 0
        record = swiss_sludge_dataset(context, services["swiss_sludge"], load, "providers")

        def key(service_id):
            """Name one generated provider in the disposable database."""
            return "providers", provider_service_code(service_id)

        generated = {key("swiss_sludge"): record}
        for role, payload in (("bottom", load.bottom_ash), ("residual", load.residual_ash)):
            generated[key(f"swiss_{role}_market")] = _ash_market_dataset(
                context, services[f"swiss_{role}_market"], payload, "providers"
            )
            generated[key(f"swiss_{role}_terminal")] = _ash_dataset(
                context,
                services[f"swiss_{role}_terminal"],
                payload,
                "providers",
                release_profile=_AshReleaseProfile(()),
            )
        bd.Database("providers").write(generated, searchable=False)
        plant = bd.get_node(database="providers", code=provider_service_code("swiss_sludge"))
        production = next(iter(plant.production()))["amount"]
        assert production == (-5 if future else -2)
        factor = -production / load.wet_mass_kg
        amounts = {}
        for ex in plant.technosphere():
            amounts[ex.input.key] = amounts.get(ex.input.key, 0.0) + ex["amount"] / factor
        # Preserve the full source process-water duty and dated supplier shares.
        assert amounts[database, "water_a"] + amounts[database, "water_b"] == pytest.approx(
            load.duties["water"]
        )
        share = 0.7 if future else 0.25
        assert amounts[database, "water_a"] == pytest.approx((load.duties["water"]) * share)
        assert (database, "PAM_a") not in amounts and (database, "PAM_b") not in amounts
        assert (database, "tmt") not in amounts
        assert (database, "CaO_a") not in amounts and (database, "packed_lime") not in amounts
        assert load.duties["CaO"] == 0
        iron: float | None = None
        if case == "pdo":
            # Actual PDO cobalt gives 5.4849446127598155e-8 kg conventional
            # toxic-metal allocand. The independent invented donor assay gives
            # .0011628793719057524 kg, with .0001 kg active FeCl3/native kg.
            ferric = 0.0001 * 5.4849446127598155e-8 / 0.0011628793719057524
            assert load.duties["FeCl3"] == pytest.approx(ferric, rel=2e-12)
            for suffix, fraction in (("a", share), ("b", 1 - share)):
                assert amounts[database, f"FeCl3_{suffix}"] == pytest.approx(ferric * fraction)
                row = next(
                    ex
                    for ex in plant.technosphere()
                    if ex.input.key == (database, f"FeCl3_{suffix}")
                )
                assert row["loc"] == pytest.approx(log(ferric * fraction * factor) + 0.03)
                assert row["scale"] == 0.21 and row["allocation"] == "cut-off control"
            # Fresh precipitation follows furnace/FLUWA. FeCl3 uses its active-mass reference.
            iron = ferric * 55.85 / (55.85 + 3 * 35.45)
            assert load.residual_ash.auxiliary_elements_kg["Fe"] == pytest.approx(iron * 100 / 101)
            assert load.aqueous_elements_kg["Fe"] == pytest.approx(iron / 101)
            assert load.aqueous_elements_kg["Cl"] == pytest.approx(
                load.duties["HCl"] * 35.45 / 36.45 + ferric - iron
            )
            # The fixture's .3*.22/25 kg source Fe has independently evaluated
            # 2.836014219007564e-8 kg in treated water; its native flow is 1e-7.
            expected_iron = 1e-7 * load.phases["water"]["Fe"] / 2.836014219007564e-8 + iron / 101
            emitted_iron = next(
                ex
                for ex in plant.biosphere()
                if ex.input["name"] == "Iron ion"
                and tuple(ex.input["categories"]) == ("water", "surface water")
            )
            assert emitted_iron["amount"] / factor == pytest.approx(expected_iron)
            emitted_chloride = next(
                ex for ex in plant.biosphere() if ex.input["name"] == "Chloride"
            )
            assert emitted_chloride["amount"] / factor == pytest.approx(
                load.duties["HCl"] * 35.45 / 36.45 + ferric - iron
            )
        else:
            assert (database, "FeCl3_a") not in amounts and (database, "FeCl3_b") not in amounts
            assert load.duties["FeCl3"] == 0
        assert amounts[database, "actual_pam"] == pytest.approx(load.duties["PAM"])
        for oxide in ("TiO2", "V2O5"):
            assert load.duties[oxide] > 0
            for suffix, fraction in (("a", share), ("b", 1 - share)):
                expected = load.duties[oxide] * fraction
                assert amounts[database, f"{oxide}_{suffix}"] == pytest.approx(expected)
                row = next(
                    ex
                    for ex in plant.technosphere()
                    if ex.input.key == (database, f"{oxide}_{suffix}")
                )
                assert row["loc"] == pytest.approx(log(expected * factor) + 0.03)
                assert row["uncertainty type"] == 2 and row["scale"] == 0.21
                assert row["allocation"] == "cut-off control"
                assert row["minimum"] == pytest.approx(expected * factor / 2)
                assert row["maximum"] == pytest.approx(expected * factor * 2)
        assert amounts[key("swiss_bottom_market")] == pytest.approx(
            -load.bottom_ash.treatment_mass_kg
        )
        assert amounts[key("swiss_residual_market")] == pytest.approx(
            -load.residual_ash.treatment_mass_kg
        )
        assert amounts[database, "hydroxide"] == pytest.approx(-load.hydroxide_kg)
        assert load.allocands["lime_before_purchase"] <= 0
        assert load.thermal["process_water_remaining_kg"] > 0
        assert load.thermal["NH3_dilution_within_meter_kg"] < load.duties["water"]
        # Every actual C/N/P/S/Na/Fe input reaches exactly one of five phases.
        for element, amount in load.elements_kg.items():
            assert fsum(p[element] for p in load.phases.values()) == pytest.approx(
                amount, rel=2e-12, abs=1e-20
            )
        # Product-property dry mass equals the oxide-sized treatment basis,
        # including fresh PAM and catalyst oxides once, with residue water separate.
        for payload in (load.bottom_ash, load.residual_ash):
            atoms = fsum(payload.waste_elements_kg.values()) + fsum(
                payload.auxiliary_elements_kg.values()
            )
            atoms += payload.allocated_hydrogen_kg + payload.allocated_oxygen_kg
            assert atoms + payload.water_kg == pytest.approx(payload.treatment_mass_kg)
        water_out = fsum(
            ex["amount"] / factor * 1000 for ex in plant.biosphere() if ex.input["name"] == "Water"
        )
        residue_water = load.bottom_ash.water_kg + load.hydroxide_water_kg
        assert water_out + residue_water == pytest.approx(
            load.water_kg + load.duties["water"] + load.thermal["nominal_reaction_water_kg"]
        )
        oxygen = next(ex for ex in plant.biosphere() if ex.input["name"] == "Oxygen")
        assert oxygen["amount"] / factor == pytest.approx(load.allocands["oxygen"])
        assert oxygen["uncertainty type"] == 0
        nox = next(ex for ex in plant.biosphere() if ex.input["name"] == "Nitrogen oxides")
        assert nox["loc"] == pytest.approx(log(nox["amount"]) + 0.03)
        assert nox["scale"] == 0.21 and nox["allocation"] == "cut-off control"
        gases = [ex for ex in plant.biosphere() if ex.input["name"].startswith("Carbon dioxide,")]
        assert len(gases) == (1 if case == "acetone" else 2)
        for ex in gases:
            ratios = ex["principal_carbon_origin_fractions"]
            if case != "finish" or ex.input["name"].endswith(", fossil"):
                assert tuple(ratios) == (0.0, 0.0, 0.0)
            else:
                assert sum(ratios) == pytest.approx(1)
        # Normalize Swiss source Ti once and retain synthetic hydration once.
        terminal = bd.get_node(
            database="providers", code=provider_service_code("swiss_residual_terminal")
        )
        tp = next(iter(terminal.production()))["amount"]
        ti = fsum(
            ex["amount"] / -tp for ex in terminal.biosphere() if ex.input["name"] == "Titanium ion"
        )
        expected = (
            (1e-8 + 2e-8)
            / (0.8 * 0.001)
            * load.residual_ash.auxiliary_elements_kg["Ti"]
            / load.residual_ash.treatment_mass_kg
        )
        assert ti == pytest.approx(expected)
        if case == "pdo":
            assert iron is not None
            # The terminal receives waste Fe and fresh precipitant Fe. Original .8 dry fraction and
            # .22/25 Fe define source release factors; dated supplies provide the operating
            # inventories.
            total_iron = load.residual_ash.waste_elements_kg["Fe"] + iron * 100 / 101
            released_iron = fsum(
                ex["amount"] / -tp for ex in terminal.biosphere() if ex.input["name"] == "Iron ion"
            )
            expected = (1e-7 + 2e-7) / (0.8 * 0.22 / 25)
            assert released_iron == pytest.approx(
                expected * total_iron / load.residual_ash.treatment_mass_kg
            )
        hydration = next(
            ex for ex in terminal.technosphere() if ex.input.key == (database, "hydration")
        )
        assert hydration["amount"] / -tp == pytest.approx(0.6)
        trace = next(
            r
            for r in context.technology_rows
            if r["provider_service_id"] == "swiss_sludge"
            and r["component_id"] == "Swiss_sludge_incineration"
        )
        assert json.loads(cast(str, trace["retained_technosphere_json"])) == []
        assert json.loads(cast(str, trace["retained_biosphere_json"])) == []
        terms = json.loads(cast(str, trace["project_terms_json"]))
        assert any(x["duty"] == "hydroxide_cutoff" for x in terms["scaled_or_replaced_native_rows"])
        lca = LCA({plant: -load.wet_mass_kg})
        lca.lci()
        supply_array = cast(NDArray[np.float64] | None, lca.supply_array)
        assert supply_array is not None
        # Synthetic nonunit production must solve with one input,
        # full support and positive catalyst demand despite negative waste signs.
        assert supply_array[lca.dicts.activity[plant.id]] == pytest.approx(1 / factor, rel=2e-7)
        for oxide in ("TiO2", "V2O5"):
            for suffix, fraction in (("a", share), ("b", 1 - share)):
                supplier = bd.get_node(database=database, code=f"{oxide}_{suffix}")
                expected = load.duties[oxide] * fraction
                assert supply_array[lca.dicts.activity[supplier.id]] == pytest.approx(
                    expected, rel=2e-7, abs=0
                )
                fossil = next(iter(supplier.biosphere())).input
                assert lca.inventory[
                    lca.dicts.biosphere[fossil.id], lca.dicts.activity[supplier.id]
                ] == pytest.approx(0.02 * expected, rel=2e-7, abs=0)
