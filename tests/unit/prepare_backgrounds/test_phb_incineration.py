"""Verify PHB municipal treatment with small synthetic Brightway inventories.

The source quantities are invented. Expected elemental/energy balances use the
published equations independently; linked LCI checks protect signs, units and
each purchase occurring once. Selected linked LCI comparisons use a 2e-7
relative tolerance for float32-processed amounts.
"""

import json
import math
from typing import Any, cast

import numpy as np
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from pa_municipal_fixtures import ORIGINS, SURFACE, URBAN, _biosphere, _native

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
from pytoce.prepare_backgrounds.preparation.providers.phb_incineration import (
    _phb_incineration_load,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _route_contexts,
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


@pytest.fixture(scope="module", params=(True, False), ids=("original", "future"))
def boundary(request, tmp_path_factory):
    """Compile the actual service graph at two dates in disposable Brightway stores."""
    original = request.param
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    selected = {f"phb_{origin}_incineration" for origin in ORIGINS}
    selected |= {
        f"phb_{origin}_{fraction}_ash_{role}"
        for origin in ORIGINS
        for fraction in ("bottom", "residual")
        for role in ("market", "treatment")
    }
    services = {
        row["provider_service_id"]: row
        for row in catalogue["services"]
        if row["provider_service_id"] in selected
    }
    names = BrightwayNames(
        tmp_path_factory.mktemp("phb_thermal") / "brightway", "phb_thermal", "biosphere", "original"
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(_biosphere(), searchable=False)
        bd.Database("original").write(
            _native(catalogue, "original", original=True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                _native(catalogue, database, original=False), searchable=False
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
        compiled = {
            ("providers", provider_service_code(sid)): _service_dataset(
                context, service, services, "providers"
            )
            for sid, service in services.items()
        }
        bd.Database("providers").write(compiled, searchable=False)
        yield bd, context, services, database, original
        assert before == {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        }


def test_original_source_duties_and_nominal_phb_balance(boundary):
    """Close principal carbon, fresh auxiliaries, water, and energy for PHB."""
    bd, context, services, database, original = boundary
    load = _phb_incineration_load(context, services["phb_fossil_incineration"]["construction"])
    c, h, o = 0.558067, 6 * 1.008 / 86.09, 2 * 15.999 / 86.09
    phase = (
        0.004011064312571921,
        0,
        0.001004021104523617,
        0.0005020105522618085,
        0.0005020105522618085,
        0.9939808934783808,
    )
    expected_oxygen = 8 * h + c * (2 * sum(phase[:4]) + 32 / 12 * phase[5]) - o
    assert load.drivers["oxygen"] == pytest.approx(expected_oxygen, rel=2e-14)
    assert load.drivers["air"] == pytest.approx(2 * expected_oxygen / 32 / 0.21 * 22.413837)
    assert load.thermal["hhv_MJ"] == pytest.approx(34.016 * c + 124.265 * h - 9.8324 * o)
    assert load.thermal["net_electricity_MJ"] > 0
    assert load.thermal["environmental_heat_MJ"] > 0
    assert load.thermal["process_water_kg"] > 0
    emissions = load.emissions
    principal_air_c = (
        emissions["Carbon dioxide, fossil", "kilogram", URBAN] * 12 / 44
        + emissions["Carbon monoxide, fossil", "kilogram", URBAN] * 12 / 28
        + emissions["Methane, fossil", "kilogram", URBAN] * 12 / 16
        + emissions["NMVOC, non-methane volatile organic compounds", "kilogram", URBAN] * 12 / 13.85
    )
    formulas = {
        "Benzene": (6, 6, 0, 0),
        "Benzo(a)pyrene": (20, 12, 0, 0),
        "Hexachlorobenzene": (6, 0, 0, 6),
        "Pentachlorobenzene": (6, 1, 0, 5),
        "Pentachlorophenol": (6, 1, 1, 5),
        "Toluene": (7, 8, 0, 0),
    }
    principal_air_c += sum(
        emissions[name, "kilogram", URBAN]
        * 12
        * atoms[0]
        / (12 * atoms[0] + atoms[1] + 16 * atoms[2] + 35.45 * atoms[3])
        for name, atoms in formulas.items()
    )
    water_c = emissions["DOC, Dissolved Organic Carbon", "kilogram", SURFACE]
    assert principal_air_c + water_c + sum(
        x.carbon_kg
        for ash in (load.bottom_ash, load.residual_ash)
        for x in ash.carbon
        if x.principal
    ) == pytest.approx(c, rel=2e-14)
    assert emissions["COD, Chemical Oxygen Demand", "kilogram", SURFACE] == pytest.approx(
        water_c * 32 / 12
    )
    assert load.residual_ash.auxiliary_elements_kg["N"] == pytest.approx(
        load.duties["PAM"] * 14 / 71
    )
    assert load.residual_ash.auxiliary_elements_kg["Ti"] == pytest.approx(
        load.duties["TiO2"] * 47.867 / 79.865
    )
    assert "Cr" not in load.residual_ash.auxiliary_elements_kg
    outgoing_water = 1000 * sum(
        emissions["Water", "cubic meter", compartment] for compartment in (URBAN, SURFACE)
    )
    assert outgoing_water + load.bottom_ash.water_kg == pytest.approx(
        load.thermal["reaction_water_kg"] + load.duties["water"],
        rel=2e-14,
    )
    assert sum(
        emissions["Heat, waste", "megajoule", compartment] for compartment in (URBAN, SURFACE)
    ) == pytest.approx(load.thermal["environmental_heat_MJ"])
    node = bd.get_node(database="providers", code=provider_service_code("phb_fossil_incineration"))
    purchases = {ex.input.key: ex for ex in node.technosphere()}
    assert next(iter(node.production()))["amount"] == 1
    # Retain the whole flue-gas-cleaning water duty; reaction water is formed.
    purchased_water = load.duties["water"]
    assert purchases[database, "phb_water_a"]["amount"] == pytest.approx(
        purchased_water * (0.4 if original else 0.8)
    )
    assert purchases[database, "phb_water_b"]["amount"] == pytest.approx(
        purchased_water * (0.6 if original else 0.2)
    )
    assert purchased_water + load.thermal["reaction_water_kg"] == pytest.approx(
        outgoing_water + load.bottom_ash.water_kg, rel=2e-14
    )
    water_row = purchases[database, "phb_water_a"]
    assert water_row["loc"] == pytest.approx(math.log(water_row["amount"]) + 0.03)
    assert water_row["scale"] == 0.21
    assert (database, "mc_inorganic") not in purchases
    assert (database, "mc_pam") in purchases
    assert purchases[database, "mc_tio2"]["amount"] == pytest.approx(load.duties["TiO2"])
    assert purchases[database, "phb_v_proxy"]["amount"] == pytest.approx(load.duties["V2O5"])
    assert (database, "mc_cao") not in purchases and (database, "mc_naoh") not in purchases
    assert (database, "phb_haul") not in purchases
    assert not any(ex.input["name"] == "Hydrochloric acid" for ex in node.biosphere())
    owner = json.loads(node["direct_treatment_carbon_owners_json"])[0]
    assert owner["owner_code"] == node.key[1] and owner["owner_scale_per_provider_unit"] == 1


def test_origins_ash_and_complete_linked_lci(boundary):
    """Solve all actual origins and retain every independent ash, hydration and SCR duty."""
    bd, context, services, database, original = boundary
    load = _phb_incineration_load(context, services["phb_fossil_incineration"]["construction"])
    nodes = {
        origin: bd.get_node(
            database="providers", code=provider_service_code(f"phb_{origin}_incineration")
        )
        for origin in ORIGINS
    }
    for index, (origin, node) in enumerate(nodes.items()):
        for ex in node.biosphere():
            if ex.input["name"].startswith(("Carbon dioxide,", "Carbon monoxide,", "Methane,")):
                assert ("non-fossil" in ex.input["name"]) == (origin != "fossil")
                assert tuple(ex["principal_carbon_origin_fractions"]) == tuple(
                    float(index == i) for i in range(3)
                )
        lca = LCA({node: 1})
        lca.lci()
        supply_array = cast(NDArray[np.float64] | None, lca.supply_array)
        assert supply_array is not None

        def supply(database_name, code):
            """Read one solved activity multiplier, including independent reference basis."""
            activity = bd.get_node(database=database_name, code=code)
            return float(cast(NDArray[np.float64], supply_array)[lca.dicts.activity[activity.id]])

        for fraction, ash in (("bottom", load.bottom_ash), ("residual", load.residual_ash)):
            treatment = bd.get_node(
                database="providers",
                code=provider_service_code(f"phb_{origin}_{fraction}_ash_treatment"),
            )
            terminal_basis = 3 if original else 7
            assert supply("providers", treatment.key[1]) * terminal_basis == pytest.approx(
                ash.treatment_mass_kg, rel=2e-7
            )
            if fraction == "residual":
                nitrogen = sum(
                    ex["amount"] * 14 / 62
                    for ex in treatment.biosphere()
                    if ex.input["name"] == "Nitrate"
                )
                assert nitrogen / terminal_basis * ash.treatment_mass_kg == pytest.approx(
                    # Invented native short/later fractions are .1 and .902;
                    # their precision is retained rather than forced to one.
                    load.duties["PAM"] * 14 / 71 * 1.002,
                    rel=2e-14,
                )
                inputs = {
                    ex.input.key: ex["amount"] / terminal_basis for ex in treatment.technosphere()
                }
                assert inputs[database, "mc_tap_a"] == pytest.approx(0.6)
                assert inputs[database, "mc_cement_a"] == pytest.approx(0.4)
                assert inputs[database, "phb_hydrated"] == -1
            assert not any(ex.input["name"] == "Zinc II" for ex in treatment.biosphere())
        assert supply(database, "phb_haul") == pytest.approx(
            0.077 * (load.bottom_ash.treatment_mass_kg + load.residual_ash.treatment_mass_kg),
            rel=2e-7,
        )
        assert supply(database, "phb_hydrated") * (3 if original else 7) == pytest.approx(
            load.residual_ash.treatment_mass_kg, rel=2e-7
        )
        assert supply(database, "mc_tio2") == pytest.approx(load.duties["TiO2"], rel=2e-7)
        assert supply(database, "phb_v_proxy") == pytest.approx(load.duties["V2O5"], rel=2e-7)
        assert supply(database, "mc_pam") == pytest.approx(load.duties["PAM"], rel=2e-7)
    # Partition the PHB mixture before treatment and retain the fossil primer's origin.
    mixed = {nodes["fossil"]: 0.07, nodes["biogenic"]: 0.81, nodes["atmospheric_dac"]: 0.12}
    co2 = {
        origin: sum(
            ex["amount"]
            for ex in node.biosphere()
            if ex.input["name"].startswith("Carbon dioxide,")
        )
        for origin, node in nodes.items()
    }
    assert co2["fossil"] == co2["biogenic"] == co2["atmospheric_dac"]
    lca = LCA(mixed)
    lca.lci()
    mixed_supply = cast(NDArray[np.float64] | None, lca.supply_array)
    assert mixed_supply is not None
    for node, amount in mixed.items():
        assert mixed_supply[lca.dicts.activity[node.id]] == pytest.approx(amount, rel=2e-7)


def test_public_routes_reach_shared_phb_owner_without_pet_proxy():
    """Keep ordinary treatment and mechanical residue on each actual-origin PHB service."""
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    routes = [
        row
        for row in catalogue["routes"]
        if row["route_id"].startswith("bioplastics_incineration__pha__")
    ]
    assert len(routes) == 3
    assert {row["ultimate_carbon_origin"] for row in routes} == set(ORIGINS)
    contexts = _route_contexts(routes, services)
    assert "pet_incineration" not in contexts
    for origin in ORIGINS:
        assert f"phb_{origin}_incineration" in contexts
        assert f"phb_{origin}_residual_ash_treatment" in contexts
