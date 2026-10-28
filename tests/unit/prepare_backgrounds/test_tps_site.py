"""Test TPS landfill sites with linked synthetic gas and wastewater inventories.

The three catalog formulations pass through site, gas, collection, wastewater
and sludge market calculations. Sludge treatment uses small synthetic inventories;
its chemistry and complete provider preparation are tested separately.
No licensed sources or internal monkeypatches are used. Direct mappings
use relative 2e-12; solved LCIs use 3e-7 for bw_processing's float32 exchanges.
"""

import json
from copy import deepcopy
from math import fsum, log
from typing import cast

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from tps_gas_fixtures import GAS_SOURCES, RURAL, URBAN, flow_key
from tps_site_fixtures import (
    GROUND,
    ORIGINAL_LEACHATE,
    SITE_PRODUCTION,
    SOIL,
    biosphere_inventory,
    native_inventory,
    remap_terminal_biosphere,
    site_catalogue,
)
from tps_water_fixtures import LAND, ORIGINS, WASTE, terminal_inventory

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
    _ROW_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_treatment import (
    _leachate_market_dataset,
    _sludge_market_dataset,
    _wastewater_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.tps_residue_load import (
    _tps_landfill_gas_load,
    _tps_residue_load,
)
from pytoce.prepare_backgrounds.preparation.providers.tps_site import _tps_landfill_dataset
from pytoce.prepare_backgrounds.preparation.providers.waste_gas import (
    _landfill_gas,
    _wastewater_gas,
)


@pytest.fixture(scope="module")
def catalogue():
    """Retain actual TPS formulations and declare the direct site test boundary."""
    return site_catalogue()


@pytest.fixture(scope="module", params=(False, True), ids=("original", "future"))
def site_boundary(request, tmp_path_factory, catalogue):
    """Persist distinct original and dated inventories in a disposable project."""
    future = request.param
    root = tmp_path_factory.mktemp("tps_site")
    names = BrightwayNames(root / "brightway", "tps_site", "biosphere", "original")
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
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                cast(str, r["method_authority_id"]): r for r in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        yield bd, context, database, future


@pytest.fixture(scope="module", params=ORIGINS)
def compiled_site(request, site_boundary, catalogue):
    """Compile one principal kg's consistent wet site and all water children."""
    bd, context, database, future = site_boundary
    origin = request.param
    services = {s["provider_service_id"]: s for s in catalogue["services"]}
    residue = _tps_residue_load(services, f"tps_{origin}")
    # Original source m3/kg wet is independent of the future donor coefficient.
    # Liquid density is the explicit 1000 kg/m3 reference: actual dry solutes
    # displace water, rather than being added on top of the same liquid mass.
    source = bd.get_node(database="original", code="site")
    p = next(iter(source.production()))["amount"]
    leachate = next(ex for ex in source.technosphere() if ex.input["code"] == "collection")
    volume = leachate["amount"] / p * residue.wet_mass_kg
    dry_solutes = fsum(residue.leachate.elements_kg.values())
    water = _wastewater_load(
        residue.leachate,
        volume_m3=volume,
        incoming_water_kg=1000 * volume - dry_solutes,
        evaporation_fraction=0.01,  # Invented 1% climatic evaporation fixture.
        profile=_ROW_WASTEWATER_PROFILE,
    )
    construction: dict[str, object] = {"gas_sources": deepcopy(GAS_SOURCES)}
    gas = _landfill_gas(
        context, construction, _tps_landfill_gas_load(residue), f"test_tps_site_{origin}"
    )
    water_gas = _wastewater_gas(
        context, construction, water, f"test_tps_plant_{origin}", profile=_ROW_WASTEWATER_PROFILE
    )
    provider = f"site_{origin}"
    datasets = remap_terminal_biosphere(terminal_inventory(provider, origin, future=future))
    for role in ("incineration", "landfarming"):
        sid = f"test_tps_{role}_market_{origin}"
        datasets[provider, provider_service_code(sid)] = _sludge_market_dataset(
            context, services[sid], water, provider
        )
    sid = f"test_tps_plant_{origin}"
    datasets[provider, provider_service_code(sid)] = _wastewater_dataset(
        context,
        services[sid],
        residue.leachate,
        water,
        water_gas,
        provider,
        profile=_ROW_WASTEWATER_PROFILE,
    )
    sid = f"test_tps_collection_{origin}"
    datasets[provider, provider_service_code(sid)] = _leachate_market_dataset(
        context, services[sid], residue.leachate, water, provider
    )
    sid = f"test_tps_site_{origin}"
    datasets[provider, provider_service_code(sid)] = _tps_landfill_dataset(
        context, services[sid], residue, water, gas, provider
    )
    bd.Database(provider).write(datasets, searchable=False)
    site = bd.get_node(database=provider, code=provider_service_code(sid))
    lca = LCA({site: -residue.wet_mass_kg})
    lca.lci()
    return bd, context, database, future, origin, provider, residue, water, gas, water_gas, lca


def _amounts(node, kind):
    """Sum persisted exchanges by exact native input and retain repeated origins."""
    grouped = {}
    for ex in node.exchanges():
        if ex["type"] == kind:
            grouped.setdefault(ex.input.key, []).append(float(ex["amount"]))
    return {key: fsum(values) for key, values in grouped.items()}


def test_actual_site_mass_fates_and_native_metadata(compiled_site):
    """Replace donor chemistry while preserving dated full site support and distributions."""
    bd, context, native, future, origin, provider, residue, water, gas, _, _ = compiled_site
    sid = f"test_tps_site_{origin}"
    site = bd.get_node(database=provider, code=provider_service_code(sid))
    source = bd.get_node(database=native, code="site")
    p = SITE_PRODUCTION[int(future)]
    q = residue.wet_mass_kg
    scale = -p / q
    production = next(iter(site.production()))
    assert production["amount"] == p
    props = production["properties"]
    dry = fsum(x.mass_kg for x in residue.components)
    assert props["dry mass"]["amount"] == pytest.approx(dry / q, rel=2e-12, abs=0)
    assert props["water in wet mass"]["amount"] == pytest.approx(residue.water_kg / q)
    assert props["carbon allocation"]["amount"] == pytest.approx(
        residue.elements_kg["C"] / q, rel=2e-12, abs=0
    )
    assert dry + residue.water_kg == pytest.approx(q, rel=2e-12, abs=0)
    assert props["wet mass"]["amount"] == 1
    for name in ("fossil", "non-fossil"):
        carbon = fsum(
            x.input_carbon_kg
            for x in residue.carbon
            if (x.origin == "fossil") == (name == "fossil")
        )
        assert props[f"carbon content, {name}"]["amount"] == pytest.approx(
            carbon / dry, rel=2e-12, abs=0
        )
    assert not {"flow", "activity", "production volume", "classifications"} & production.keys()
    for field in ("uncertainty type", "loc"):
        assert production[field] == next(iter(source.production()))[field]

    tech = _amounts(site, "technosphere")
    collection = (provider, provider_service_code(f"test_tps_collection_{origin}"))
    assert water.volume_m3 == pytest.approx(ORIGINAL_LEACHATE * q, rel=2e-12, abs=0)
    assert water.water.incoming_kg + fsum(residue.leachate.elements_kg.values()) == pytest.approx(
        1000 * water.volume_m3, rel=2e-12, abs=0
    )
    assert tech[collection] == pytest.approx(p * ORIGINAL_LEACHATE, rel=2e-12, abs=0)
    assert (native, "collection") not in tech
    site_rows = {ex.input.key: ex for ex in site.technosphere()}
    for old in source.technosphere():
        if old.input["code"] == "collection":
            new = site_rows[collection]
            factor = new["amount"] / old["amount"]
            assert new["loc"] == pytest.approx(old["loc"] + log(factor), rel=2e-12, abs=0)
            assert "properties" not in new
        else:
            new = site_rows[old.input.key]
            for field in ("amount", "loc", "scale", "minimum", "maximum", "allocation"):
                assert new[field] == old[field]

    bio = _amounts(site, "biosphere")
    expected = {}
    for row in gas.biosphere_exchanges:
        expected.setdefault(row["input"], []).append(scale * row["amount"])
    for key, values in expected.items():
        assert bio[key] == pytest.approx(fsum(values), rel=2e-12, abs=0)
    # Fixed source sulfur species fractions, independent of the donor's .02/.01
    # invented emissions and of the dated provider. Protein alone has later S.
    # Landfill2020 MSWLF C90/C91 give mg S/L; D90/D91 normalize those amounts.
    # G90/G91 convert elemental S to species with 96/32 (SO4) and 34/32 (H2S).
    later_s = residue.protein_sulfur.later_leachate_kg
    sulfate_share = 40.00326505167857 / (40.00326505167857 + 2.9443208588856526)
    sulfate = 3 * sulfate_share * later_s
    sulfide = 34 / 32 * (1 - sulfate_share) * later_s
    assert bio.get(flow_key("Sulfate", GROUND), 0) == pytest.approx(
        scale * sulfate, rel=2e-12, abs=0
    )
    assert bio.get(flow_key("Hydrogen sulfide", GROUND), 0) == pytest.approx(
        scale * sulfide, rel=2e-12, abs=0
    )
    assert sulfate / 3 + sulfide * 32 / 34 == pytest.approx(later_s, rel=2e-12, abs=0)
    assert (later_s == 0) == (origin == "biogenic")
    for name in ("Chloride", "TOC, Total Organic Carbon", "Heat, waste"):
        assert flow_key(name, GROUND) not in bio
    assert bio[flow_key("Organic carbon, placed in landfill", WASTE)] == pytest.approx(
        scale * residue.elements_kg["C"], rel=2e-12, abs=0
    )
    for name in ("Waste mass, total, placed in landfill", "Non-hazardous waste disposed"):
        assert bio[flow_key(name, WASTE)] == -p
    admin = -p * (0.06 if future else 0.05)
    assert bio[flow_key("Heat, waste", RURAL)] == pytest.approx(
        scale * gas.heat_air_mj + 3.6 * admin,
        rel=2e-12,
        abs=0,
    )
    assert bio[flow_key("Heat, waste", SOIL)] == pytest.approx(
        scale * gas.heat_soil_mj, rel=2e-12, abs=0
    )
    for ex in source.biosphere():
        if ex.input["name"].startswith(("Occupation,", "Transformation,")):
            new = next(row for row in site.biosphere() if row.input.key == ex.input.key)
            for field in ("amount", "loc", "scale", "minimum", "maximum", "allocation"):
                assert new[field] == ex[field]
    trace = next(
        x
        for x in context.technology_rows
        if x.get("provider_service_id") == sid
        and x.get("component_id") == "TPS_actual_residue_landfill"
    )
    terms = json.loads(trace["project_terms_json"])
    assert terms["administration_kwh_per_native_reference"] == pytest.approx(admin)
    assert terms["later_sulfur_kg_per_principal_kg"] == later_s


def test_linked_site_water_gas_and_electricity_heat_once(compiled_site):
    """Propagate actual wet demand and one gas block through signed nonunit children."""
    bd, _, native, future, origin, provider, residue, water, gas, water_gas, lca = compiled_site
    q = residue.wet_mass_kg
    treated = 0.6 if future else 0.8

    def level(database, code):
        """Read a solved activity level through the supported Brightway dictionary."""
        node = bd.get_node(database=database, code=code)
        return lca.supply_array[lca.dicts.activity[node.id]]

    def total(name, categories):
        """Read the complete linked elementary inventory for one native flow."""
        node = bd.get_node(database="biosphere", code=flow_key(name, categories)[1])
        return float(lca.inventory[lca.dicts.biosphere[node.id], :].sum())

    for role, demand, p in (
        ("site", q, SITE_PRODUCTION[int(future)]),
        ("collection", water.volume_m3, -5 if future else -2),
        ("plant", treated * water.volume_m3, -7 if future else -3),
    ):
        assert level(provider, provider_service_code(f"test_tps_{role}_{origin}")) == pytest.approx(
            demand / -p, rel=3e-7, abs=0
        )
        assert level(native, role) == 0
    for code, amount in (
        ("site_equipment", 0.003 if future else 0.002),
        ("site_admin_a", 0.04 if future else 0.02),
        ("site_admin_b", 0.02 if future else 0.03),
        ("site_heat", 0.07 if future else 0.04),
        ("site_retirement", -0.001),
    ):
        node = bd.get_node(database=native, code=code)
        p = next(iter(node.production()))["amount"]
        assert level(native, code) == pytest.approx(q * amount / p, rel=3e-7, abs=0)
    # All three actual TPS gas mixtures meet their pumping load internally.
    # The independent site/plant supplies above and below remain with parents.
    # Native plant construction/retirement remain linked through its actual
    # treated volume even though the gas adaptation adds no operating oil.
    for code, coefficient in (("limestone", 0.006 if future else 0.004), ("retirement", -0.003)):
        node = bd.get_node(database=native, code=code)
        assert level(native, code) == pytest.approx(
            treated * water.volume_m3 * coefficient / next(iter(node.production()))["amount"],
            rel=3e-7,
            abs=0,
        )
    for code in ("chp", "furnace", "biogas", "natural_gas", "electricity"):
        assert level(native, code) == 0
    assert total("Occupation, construction site", LAND) == pytest.approx(
        q * (0.03 if future else 0.02), rel=3e-7, abs=0
    )
    assert total("Organic carbon, placed in landfill", WASTE) == pytest.approx(
        residue.elements_kg["C"], rel=3e-7, abs=0
    )
    assert total("Waste mass, total, placed in landfill", WASTE) == pytest.approx(
        q, rel=3e-7, abs=0
    )
    # Wastewater2021 Infra G28/G33 assigns 23375 kWh per 13250 kg limestone. Synthetic .004/.006 kg
    # limestone/m3 give original/future demands separately from the actual-load meter. Consumed
    # electricity contributes 3.6 MJ/kWh to allocated heat. Purchased heat uses its complete
    # producer; the WWT allocation includes delivered heat.
    water_e = (
        water.duties.electricity_kwh
        + water.volume_m3 * (0.006 if future else 0.004) * 23375 / 13250
    )
    air_heat = (
        gas.heat_air_mj
        + 3.6 * q * (0.06 if future else 0.05)
        + treated * (3.6 * water_e + water_gas.heat_air_mj + water.duties.heat_mj)
    )
    assert total("Heat, waste", RURAL) + total("Heat, waste", URBAN) == pytest.approx(
        air_heat, rel=3e-7, abs=0
    )
    assert total("Heat, waste", SOIL) == pytest.approx(gas.heat_soil_mj, rel=3e-7, abs=0)
    # Both real gas owners already bind reaction water and atmospheric oxygen.
    # The water parent adds only its separate biological and hydraulic duties.
    resource_air = ("natural resource", "in air")
    gas_oxygen = fsum(
        share * row["amount"]
        for result, share in ((gas, 1), (water_gas, treated))
        for row in result.biosphere_exchanges
        if row["input"] == flow_key("Oxygen", resource_air)
    )
    assert total("Oxygen", resource_air) == pytest.approx(
        gas_oxygen + treated * water.duties.biological_oxygen_kg, rel=3e-7, abs=0
    )
    gas_water = fsum(
        share * row["amount"]
        for result, share in ((gas, 1), (water_gas, treated))
        for row in result.biosphere_exchanges
        if row["input"] in (flow_key("Water", RURAL), flow_key("Water", URBAN))
    )
    assert total("Water", RURAL) + total("Water", URBAN) == pytest.approx(
        gas_water + treated * water.water.hydraulic_evaporation_kg / 1000,
        rel=3e-7,
        abs=0,
    )
