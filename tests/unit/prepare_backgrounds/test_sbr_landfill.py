"""Verify actual SBR scrap and all linked landfill descendants with small sources."""

import json
from math import fsum

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from sbr_landfill_fixtures import (
    AIR,
    LAND,
    LONG,
    OWNERS,
    SURFACE,
    biosphere_inventory,
    native_inventory,
)
from tps_gas_fixtures import flow_key

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
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _ROW_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.sbr_landfill_load import (
    _sbr_landfill_load,
)

ORIGINS = ("fossil", "biogenic", "atmospheric_dac")


@pytest.mark.parametrize("origin", ORIGINS)
def test_filled_scrap_phases_and_conditional_water_match_source(origin):
    """Close every actual element and retain fixed auxiliaries independently of principal."""
    load = _sbr_landfill_load(origin)
    assert load.mass_kg == pytest.approx(0.09720521948270619, rel=2e-14)
    assert load.carbon[0].origin == origin
    assert load.carbon[0].carbon_kg == pytest.approx(0.047206472862196314)
    assert [(x.origin, x.carbon_kg) for x in load.carbon[1:]] == [
        ("fossil", 0.03633106199969588),
        ("biogenic", 0.0007595306139770401),
    ]
    for phase in load.phases.values():
        assert fsum(
            (phase.gas_kg, phase.short_leachate_kg, phase.later_leachate_kg, phase.stored_kg)
        ) == pytest.approx(phase.input_kg, rel=2e-14)
    assert load.phases["C"].gas_kg == pytest.approx(0.0008183476200799715)
    assert load.phases["C"].short_leachate_kg == pytest.approx(0.000024550428602399134)
    assert load.phases["C"].later_leachate_kg == pytest.approx(0.08345416742718688)
    assert load.phases["N"].stored_kg > 0
    # The independent source worksheet's original hydraulic reference is held
    # here, outside the recipe catalog. Loads remain conditional on treatment.
    water = _wastewater_load(
        load.leachate,
        volume_m3=0.0003843230869661498,
        incoming_water_kg=0.3842844862027998,
        evaporation_fraction=0.0,
        profile=_ROW_WASTEWATER_PROFILE,
    )
    assert water.final_sludge.dry_mass_kg == pytest.approx(0.0000334519259085493)
    assert water.final_sludge.elements_kg["C"] == pytest.approx(0.00001436851014090371)
    assert water.duties.iron_sulfate_kg == 0
    assert water.duties.polyacrylamide_kg == pytest.approx(4.3942955885436777e-7)
    assert water.duties.electricity_kwh == pytest.approx(0.00007145403468493781)
    assert water.duties.heat_mj == pytest.approx(0.00004761939558553325)
    assert water.water.effluent_kg == pytest.approx(0.383511138026858)
    assert load.gas.decomposition_heat_mj == pytest.approx(0.0009390478749130848)


@pytest.fixture(scope="module", params=(False, True), ids=("original", "future"))
def boundary(request, tmp_path_factory):
    """Write invented native nodes in a disposable project before real compilation."""
    future = request.param
    root = tmp_path_factory.mktemp("sbr_landfill")
    names = BrightwayNames(root / "brightway", "sbr_landfill", "biosphere", "original")
    cat = load_provider_catalogue()
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
            catalogue=cat,
            material_recipe=physical_material_recipe(),
            method_authorities={r["method_authority_id"]: r for r in _method_authority_rows(cat)},
            complete_source_ids=_service_source_closures(cat),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        yield bd, context, cat, database, future


@pytest.mark.parametrize("origin", ORIGINS)
def test_complete_chain_retains_sources_signs_and_actual_chemistry(boundary, origin):
    """Reach all 11 source owners through the package dispatcher and solve their LCI."""
    bd, context, cat, database, future = boundary
    services = {s["provider_service_id"]: s for s in cat["services"]}
    prefix = f"sbr_filled_{origin}_"
    provider = f"providers_{origin}"
    compiled = {}
    for role in OWNERS:
        sid = prefix + role
        compiled[provider, provider_service_code(sid)] = _service_dataset(
            context, services[sid], services, provider
        )
    bd.Database(provider).write(compiled, searchable=False)
    site = bd.get_node(database=provider, code=provider_service_code(prefix + "landfill"))
    load = _sbr_landfill_load(origin)
    lca = LCA({site: -load.mass_kg})
    lca.lci()
    assert lca.inventory.data.size > 0

    def emitted(role, name, categories):
        """Read a generated source amount per positive unit treated at its own gate."""
        node = bd.get_node(database=provider, code=provider_service_code(prefix + role))
        return (
            fsum(
                ex["amount"] for ex in node.biosphere() if ex["input"] == flow_key(name, categories)
            )
            / -node.rp_exchange()["amount"]
        )

    for role, (source_code, _, _, location) in OWNERS.items():
        node = bd.get_node(database=provider, code=provider_service_code(prefix + role))
        source = bd.get_node(database=database, code=source_code)
        assert node.rp_exchange()["amount"] == source.rp_exchange()["amount"] < 0
        assert node["location"] == location
    assert emitted("landfill", "TOC, Total Organic Carbon", LONG) * load.mass_kg == pytest.approx(
        0.08345416742718688
    )
    assert emitted("landfill", "Occupation, construction site", LAND) == pytest.approx(0.00025)
    assert emitted("landfill", "Transformation, from pasture, man made", LAND) == pytest.approx(
        0.00006
    )
    assert emitted("landfill", "Zinc II", AIR) * load.mass_kg == pytest.approx(
        4.4462551572808274e-10
    )
    # Independent filled-scrap source worksheet: MJ per kg retained principal.
    # The native signed production and actual scrap demand normalize each date.
    assert emitted("landfill", "Heat, waste", LONG) * load.mass_kg == pytest.approx(
        3.569394893742988, rel=2e-14
    )
    assert emitted("sludge_incineration", "Calcium II", SURFACE) > 0
    assert emitted("residual_ash_treatment", "Calcium II", SURFACE) > 0
    # The plastic-ash source uses Ti scaled by its original donor property.
    ash = bd.get_node(
        database=provider, code=provider_service_code(prefix + "residual_ash_treatment")
    )
    assert (
        len([ex for ex in ash.biosphere() if ex["input"] == flow_key("Titanium ion", SURFACE)]) == 1
    )
    hydration = [ex for ex in ash.technosphere() if ex.input["code"] == "thermal_hydration"]
    assert len(hydration) == 1
    assert hydration[0]["amount"] / -ash.rp_exchange()["amount"] == pytest.approx(0.6)
    # Original waste-N coefficients determine catalyst demand; dated
    # native suppliers retain their mix and full upstream emission inventories.
    thermal = bd.get_node(
        database=provider, code=provider_service_code(prefix + "sludge_incineration")
    )
    thermal_trace = next(
        row
        for row in context.technology_rows
        if row.get("provider_service_id") == prefix + "sludge_incineration"
        and row.get("component_id") == "actual_sludge_incineration"
    )
    thermal_load = json.loads(thermal_trace["project_terms_json"])
    wet = thermal_load["wet_mass_kg"]
    n_ratio = (wet + 100 * thermal_load["elements_kg"]["N"]) / (1 + 100 * 0.3 * 0.04)
    normalization = -thermal.rp_exchange()["amount"] / wet
    thermal_level = lca.supply_array[lca.dicts.activity[thermal.id]]
    shares = (0.7, 0.3) if future else (0.25, 0.75)
    for oxide, original_dose in (("TiO2", 0.00001), ("V2O5", 0.000002)):
        duty = original_dose * n_ratio
        assert duty > 0
        assert thermal_load["duties"][oxide] == pytest.approx(duty, rel=2e-12, abs=0)
        for suffix, share in zip(("a", "b"), shares, strict=True):
            supplier = bd.get_node(database=database, code=f"thermal_{oxide}_{suffix}")
            row = next(ex for ex in thermal.technosphere() if ex.input.key == supplier.key)
            expected = duty * share * normalization
            assert row["amount"] == pytest.approx(expected, rel=2e-12, abs=0)
            supplied = expected * thermal_level
            assert lca.supply_array[lca.dicts.activity[supplier.id]] == pytest.approx(
                supplied, rel=2e-7, abs=0
            )
            fossil = next(iter(supplier.biosphere())).input
            assert lca.inventory[
                lca.dicts.biosphere[fossil.id], lca.dicts.activity[supplier.id]
            ] == pytest.approx(0.02 * supplied, rel=2e-7, abs=0)
    # Retain complete native support, uncertainty, and dated nonunit references.
    original = bd.get_node(database=database, code="site")
    source_equipment = next(
        ex for ex in original.technosphere() if ex.input["code"] == "site_equipment"
    )
    equipment = next(ex for ex in site.technosphere() if ex.input["code"] == "site_equipment")
    for field in ("amount", "loc", "uncertainty type", "scale", "minimum", "maximum"):
        assert equipment[field] == source_equipment[field]
    terms = [
        json.loads(row["project_terms_json"])
        for row in context.technology_rows
        if row.get("provider_service_id") == prefix + "sludge_landfarming"
        and row.get("project_terms_json")
    ]
    field = next(item for item in terms if isinstance(item, dict) and "wet_kg" in item)
    wet = field["wet_kg"]
    co2 = fsum(
        emitted("sludge_landfarming", f"Carbon dioxide, {label}", AIR)
        for label in ("fossil", "non-fossil")
    )
    assert co2 * wet == pytest.approx(field["actual_elements_kg"]["C"] * 44 / 12)
    for role, compartment in (
        ("landfill", AIR),
        ("sludge_incineration", ("air", "urban air close to ground")),
        ("sludge_landfarming", AIR),
    ):
        assert emitted(role, "Carbon dioxide, fossil", compartment) > 0
        assert emitted(role, "Carbon dioxide, non-fossil", compartment) > 0
    routes = [
        r
        for r in cat["routes"]
        if r["provider_demand_type"] == "processing_conversion"
        and r["provider_service_id"].startswith("others_esbr_compounding_conversion")
    ]
    route = next(r for r in routes if r["ultimate_carbon_origin"] == origin)
    assert (
        services[route["provider_service_id"]]["construction"]["principal_carbon_origin"] == origin
    )
