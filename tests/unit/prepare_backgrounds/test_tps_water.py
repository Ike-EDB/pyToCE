"""Test TPS collection/WWT mapping with three formulations and real tiny LCIs.

The gas result and sludge terminals are complete small external boundaries;
their chemistry is tested by their own owners. These tests exercise the real
WWT/market compilers with the linked sludge markets and terminal haulage.
Scalar mappings use relative 2e-12, zero absolute tolerance. Solved LCI amounts
use 2e-7 because bw_processing 1.6 stores exchange amounts in float32.
"""

import json
from math import fsum, log
from typing import cast

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from tps_water_fixtures import (
    AIR_RESOURCE,
    CARBON_INDICATORS,
    LAND,
    ORIGINS,
    RAW_WATER,
    RURAL,
    SURFACE,
    URBAN,
    WASTE,
    biosphere_inventory,
    flow_key,
    gas_boundary,
    native_inventory,
    terminal_inventory,
    water_catalogue,
)

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
    _tps_residue_load,
)


@pytest.fixture(scope="module")
def catalogue():
    """Keep real packaged formulations and add explicit test-only owner bindings."""
    return water_catalogue()


@pytest.fixture(scope="module", params=(False, True), ids=("original", "future"))
def water_boundary(request, tmp_path_factory, catalogue):
    """Persist supported original/future inventories in a disposable Brightway project."""
    future = request.param
    root = tmp_path_factory.mktemp("tps_water")
    names = BrightwayNames(root / "brightway", "tps_water", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere_inventory(), searchable=False)
        bd.Database("original").write(native_inventory("original", future=False), searchable=False)
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(native_inventory(database, future=True), searchable=False)
        resolver = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=resolver,
            technology=resolver,
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
def compiled_water(request, water_boundary, catalogue):
    """Compile one linked collection/plant/sludge graph per real formulation and date."""
    bd, context, database, future = water_boundary
    origin = request.param
    services = {s["provider_service_id"]: s for s in catalogue["services"]}
    residue = _tps_residue_load(services, f"tps_{origin}")
    # Independent site inputs: 2 L hydraulic service, 1.8 kg incoming
    # water and 1% hydraulic evaporation. No collection share enters this owner.
    wastewater = _wastewater_load(
        residue.leachate,
        volume_m3=0.002,
        incoming_water_kg=1.8,
        evaporation_fraction=0.01,
        profile=_ROW_WASTEWATER_PROFILE,
    )
    gas = gas_boundary()
    provider = f"water_{origin}"
    datasets = terminal_inventory(provider, origin, future=future)
    for role in ("incineration", "landfarming"):
        sid = f"test_tps_{role}_market_{origin}"
        datasets[provider, provider_service_code(sid)] = _sludge_market_dataset(
            context, services[sid], wastewater, provider
        )
    sid = f"test_tps_plant_{origin}"
    datasets[provider, provider_service_code(sid)] = _wastewater_dataset(
        context,
        services[sid],
        residue.leachate,
        wastewater,
        gas,
        provider,
        profile=_ROW_WASTEWATER_PROFILE,
    )
    sid = f"test_tps_collection_{origin}"
    datasets[provider, provider_service_code(sid)] = _leachate_market_dataset(
        context, services[sid], residue.leachate, wastewater, provider
    )
    bd.Database(provider).write(datasets, searchable=False)
    collection = bd.get_node(database=provider, code=provider_service_code(sid))
    # One principal kg's negative m3 demand, with independent negative
    # production at every market, plant and terminal boundary.
    lca = LCA({collection: -wastewater.volume_m3})
    lca.lci()
    return bd, context, database, future, origin, provider, residue, wastewater, gas, lca


def _amounts(node, kind):
    """Sum real persisted rows by exact input, retaining repeated gas/plant flow contributions."""
    grouped = {}
    for ex in node.exchanges():
        if ex["type"] == kind:
            grouped.setdefault(ex.input.key, []).append(float(ex["amount"]))
    return {key: fsum(values) for key, values in grouped.items()}


def test_collection_and_actual_wastewater(compiled_water):
    """Apply bypass once; map complete plant loads, utilities, fates and source uncertainty."""
    bd, context, native, future, origin, provider, residue, water, gas, lca = compiled_water
    collection = bd.get_node(
        database=provider, code=provider_service_code(f"test_tps_collection_{origin}")
    )
    plant = bd.get_node(database=provider, code=provider_service_code(f"test_tps_plant_{origin}"))
    source_plant = bd.get_node(database=native, code="plant")
    p_market, p_plant = (-5, -7) if future else (-2, -3)
    treated = 0.6 if future else 0.8  # Synthetic dated collection shares.
    scale = -p_plant / water.volume_m3
    graph_plant_level = -treated * water.volume_m3 / p_plant

    def level(database, code):
        """Read solved activity supply through the supported Brightway dictionary."""
        node = bd.get_node(database=database, code=code)
        return lca.supply_array[lca.dicts.activity[node.id]]

    assert level(provider, collection["code"]) == pytest.approx(
        -water.volume_m3 / p_market, rel=2e-7, abs=0
    )
    assert level(provider, plant["code"]) == pytest.approx(graph_plant_level, rel=2e-7, abs=0)
    assert level(native, "plant") == level(native, "collection") == 0
    ctech = _amounts(collection, "technosphere")
    assert ctech[plant.key] == pytest.approx(treated * p_market, rel=2e-12, abs=0)
    # Both full sewer inventories stay hydraulic, before any treated fraction.
    for code, amount in (("sewer_a", 0.0002), ("sewer_b", 0.0003)):
        assert ctech[native, code] == pytest.approx(-p_market * amount, rel=2e-12, abs=0)
        assert level(native, code) == pytest.approx(water.volume_m3 * amount, rel=2e-7, abs=0)

    cbio = _amounts(collection, "biosphere")
    bypass = -p_market * (1 - treated) / water.volume_m3
    raw_elements = {
        "Hydrogen": "H",
        "Oxygen": "O",
        "Nitrogen": "N",
        "Phosphorus": "P",
        "Sulfur": "S",
        "Sodium I": "Na",
        "Magnesium": "Mg",
        "Zinc II": "Zn",
    }
    raw_amounts = {
        name: residue.leachate.elements_kg[element] for name, element in raw_elements.items()
    }
    raw_amounts.update(
        zip(
            CARBON_INDICATORS,
            (
                residue.leachate.toc_kg,
                residue.leachate.doc_kg,
                residue.leachate.cod_kg_o2,
                residue.leachate.bod5_kg_o2,
            ),
            strict=True,
        )
    )
    raw_amounts["Water"] = water.water.incoming_kg / 1000  # kg water / (1000 kg/m3).
    expected_keys = {flow_key(name, RAW_WATER) for name, mass in raw_amounts.items() if mass != 0}
    # The two synthetic formulations retain NaCl/MgCl2 as dissolved chloride; the biogenic
    # formulation has zero salt.
    chloride = fsum(
        component.elements_kg["Cl"]
        for component in residue.components
        if component.component in {"sodium_chloride", "magnesium_chloride"}
    )
    assert residue.leachate.chloride_kg == residue.leachate.elements_kg["Cl"] == chloride
    assert chloride == 0 if origin == "biogenic" else chloride > 0
    chloride_key = flow_key("Chloride", SURFACE)
    if chloride:
        expected_keys.add(chloride_key)
    assert cbio.keys() == expected_keys
    assert cbio.get(chloride_key, 0) == pytest.approx(bypass * chloride, rel=2e-12, abs=0)
    assert flow_key("Chlorine", RAW_WATER) not in cbio
    for name, mass in raw_amounts.items():
        assert cbio.get(flow_key(name, RAW_WATER), 0) == pytest.approx(
            bypass * mass, rel=2e-12, abs=0
        )
    # The source indicator templates preserve distributions, even though
    # target raw loads replace their invented municipal concentrations.
    old_collection = bd.get_node(database=native, code="collection")
    for ex in collection.biosphere():
        if ex.input.key == chloride_key:
            # Known dissolved chloride uses its specified biosphere identity and deterministic
            # amount.
            assert ex["uncertainty type"] == 0
            assert ex["loc"] == ex["amount"]
            continue
        old = next(r for r in old_collection.biosphere() if r.input.key == ex.input.key)
        multiplier = ex["amount"] / old["amount"]
        assert ex["loc"] == pytest.approx(old["loc"] + log(multiplier), rel=2e-12, abs=0)
        assert ex["scale"] == old["scale"] == 0.21
        assert ex["comment"] == old["comment"]

    # Production magnitudes stay native; intensive composition properties
    # describe actual leachate per m3, independently of negative p.
    dry = fsum(residue.leachate.elements_kg.values())
    for node, p in ((collection, p_market), (plant, p_plant)):
        ref = next(iter(node.production()))
        assert ref["amount"] == p
        assert not {"flow", "activity", "production volume", "classifications"} & set(ref)
        props = ref["properties"]
        # The input contains 1.8 kg water in 0.002 m3 (900 kg/m3), with elements recorded
        # separately.
        assert props["wet mass"]["amount"] == pytest.approx(
            (water.water.incoming_kg + dry) / water.volume_m3
        )
        assert props["dry mass"]["amount"] == pytest.approx(dry / water.volume_m3, rel=2e-12, abs=0)
        assert props["water in wet mass"]["amount"] == pytest.approx(
            water.water.incoming_kg / water.volume_m3, rel=2e-12, abs=0
        )
        assert props["carbon allocation"]["amount"] == pytest.approx(
            residue.leachate.toc_kg / water.volume_m3, rel=2e-12, abs=0
        )
        assert props["carbon content"]["amount"] == pytest.approx(
            residue.leachate.toc_kg / dry, rel=2e-12, abs=0
        )
        assert fsum(
            props[f"carbon content, {o}"]["amount"] for o in ("fossil", "non-fossil")
        ) == pytest.approx(residue.leachate.toc_kg / dry, rel=2e-12, abs=0)

    tech = _amounts(plant, "technosphere")
    # Infra G28/G33 allocates 23375 kWh construction electricity per 13250 kg
    # limestone. Invented dated limestone is .004/.006 kg per m3 treatment.
    limestone_per_m3 = 0.006 if future else 0.004
    construction_kwh = -p_plant * limestone_per_m3 * 23375 / 13250
    electricity = scale * water.duties.electricity_kwh + construction_kwh
    targets = {
        "electricity": electricity,
        "heat": scale * water.duties.heat_mj,
        "iron": scale * water.duties.iron_sulfate_kg,
        "pam": scale * water.duties.polyacrylamide_kg,
        "sand": -scale * water.duties.sand_kg,
        "paper": -scale * water.duties.grit_biomass_kg,
        "plastic": -scale * water.duties.grit_plastic_kg,
    }
    ordinary_shares = (0.7, 0.3) if future else (0.25, 0.75)
    screening_shares = (0.6, 0.4) if future else (0.2, 0.8)
    for prefix, total in targets.items():
        shares = screening_shares if prefix in {"paper", "plastic"} else ordinary_shares
        for suffix, share in zip(("a", "b"), shares, strict=True):
            key = native, f"{prefix}_{suffix}"
            amount = total * share
            assert tech.get(key, 0) == pytest.approx(amount, rel=2e-12, abs=0)
            # Waste leaves produce -1; positive supplies produce +1.
            sign = -1 if prefix in {"sand", "paper", "plastic"} else 1
            assert level(*key) == pytest.approx(graph_plant_level * amount / sign, rel=2e-7, abs=0)
            if amount == 0:
                assert key not in tech
                continue
            new = next(r for r in plant.technosphere() if r.input.key == key)
            old = next(r for r in source_plant.technosphere() if r.input.key == key)
            multiplier = amount / old["amount"]
            assert new["loc"] == pytest.approx(old["loc"] + log(multiplier), rel=2e-12, abs=0)
            assert new["scale"] == old["scale"] == 0.21
            assert new["negative"] == old["negative"]
            assert new["minimum"] == pytest.approx(old["minimum"] * multiplier, rel=2e-12, abs=0)
            assert new["maximum"] == pytest.approx(old["maximum"] * multiplier, rel=2e-12, abs=0)
            assert new["comment"] == old["comment"]
    assert (
        water.duties.iron_sulfate_kg == 0
        if origin == "biogenic"
        else water.duties.iron_sulfate_kg > 0
    )
    # Waste-PE retirement belongs to the standing-plant function.
    assert tech[native, "retirement"] == pytest.approx(p_plant * 0.003, rel=2e-12, abs=0)
    assert level(native, "retirement") == pytest.approx(
        treated * water.volume_m3 * 0.003, rel=2e-7, abs=0
    )
    assert level(native, "limestone") == pytest.approx(
        treated * water.volume_m3 * limestone_per_m3, rel=2e-7, abs=0
    )
    bio = _amounts(plant, "biosphere")
    ions = {
        "Chloride": ("Cl", 1),
        "Sodium I": ("Na", 1),
        "Magnesium": ("Mg", 1),
        "Zinc II": ("Zn", 1),
        "Iron ion": ("Fe", 1),
        # WWcalc FA/FB24:26: kg sulfate/nitrate/phosphate per kg S/N/P.
        "Sulfate": ("S", 96.07 / 32.07),
        "Nitrate": ("N", 62.01 / 14.01),
        "Phosphate": ("P", 94.97 / 30.97),
    }
    for name, (element, factor) in ions.items():
        assert bio.get(flow_key(name, SURFACE), 0) == pytest.approx(
            scale * water.effluent.elements_kg[element] * factor, rel=2e-12, abs=0
        )
    for name, quantity in zip(
        CARBON_INDICATORS,
        (
            water.effluent.toc_kg,
            water.effluent.doc_kg,
            water.effluent.cod_kg_o2,
            water.effluent.bod5_kg_o2,
        ),
        strict=True,
    ):
        assert bio[flow_key(name, SURFACE)] == pytest.approx(scale * quantity, rel=2e-12, abs=0)
    assert bio[flow_key("Water", SURFACE)] == pytest.approx(
        scale * water.water.effluent_kg / 1000, rel=2e-12, abs=0
    )
    assert bio[flow_key("Non-hazardous waste disposed", WASTE)] == -p_plant * 1000
    assert bio[flow_key("Occupation, industrial area", LAND)] == pytest.approx(
        -p_plant * 0.015, rel=2e-12, abs=0
    )
    assert bio[flow_key("Oxygen", AIR_RESOURCE)] == pytest.approx(
        scale * (water.duties.biological_oxygen_kg + 0.01), rel=2e-12, abs=0
    )
    assert bio[flow_key("Methane, non-fossil", URBAN)] == pytest.approx(
        scale * 0.0004, rel=2e-12, abs=0
    )
    # GNIWWT AY337 partitions biological gases; Infra AT16 independently
    # partitions hydraulic evaporation and combined B133:B138 energy.
    urban_bio, urban_hydraulic = 0.6808092816046385, 0.7217062553373826
    heat = 3.6 * electricity + scale * (gas.heat_air_mj + water.duties.heat_mj)
    # 3.6 MJ/kWh and the admitted delivered-external-heat convention
    # include construction allocation once; no upstream CHP loss is inferred.
    for categories, b_share, h_share in (
        (URBAN, urban_bio, urban_hydraulic),
        (RURAL, 1 - urban_bio, 1 - urban_hydraulic),
    ):
        assert bio[flow_key("Heat, waste", categories)] == pytest.approx(
            h_share * heat, rel=2e-12, abs=0
        )
        reaction_water = 0.00002 if categories == URBAN else 0
        assert bio[flow_key("Water", categories)] == pytest.approx(
            scale * (h_share * water.water.hydraulic_evaporation_kg / 1000 + reaction_water),
            rel=2e-12,
            abs=0,
        )
        for carbon_origin in ("fossil", "non-fossil"):
            carbon = fsum(
                pool.biological_air_carbon_kg
                for pool in water.carbon
                if (pool.origin == "fossil") == (carbon_origin == "fossil")
            )
            # Doka nominal 44 kg CO2/12 kg C; source carbon origins are retained.
            assert bio.get(
                flow_key("Carbon dioxide, " + carbon_origin, categories), 0
            ) == pytest.approx(scale * b_share * carbon * 44 / 12, rel=2e-12, abs=0)
        # Two N atoms in N2O: 44 kg N2O/28 kg N on the same source convention.
        assert bio.get(flow_key("Dinitrogen monoxide", categories), 0) == pytest.approx(
            scale * b_share * water.biological_n2o_nitrogen_kg * 44 / 28, rel=2e-12, abs=0
        )
    # WWcalc B140 is MJ/kg biological element.
    biological = water.biological_air_elements_kg
    assert bio[flow_key("Heat, waste", SURFACE)] == pytest.approx(
        scale * (34.016 * biological["C"] + 19.079 * biological["S"] + 6.276 * biological["N"]),
        rel=2e-12,
        abs=0,
    )
    trace = next(
        r
        for r in context.technology_rows
        if r.get("provider_service_id") == f"test_tps_plant_{origin}"
        and r.get("component_id") == "actual_wastewater_treatment"
    )
    terms = json.loads(trace["project_terms_json"])
    assert terms["gas"] == gas.trace
    assert terms["hydraulic_urban_fraction"] == urban_hydraulic
    assert terms["biological_gas_urban_fraction"] == urban_bio
    assert terms["electricity_kwh_per_native_reference"] == pytest.approx(
        electricity, rel=2e-12, abs=0
    )
    # The solved elementary inventory independently crosses the collection
    # boundary: raw bypass is uncollected; plant and gas are treated once.
    for name, compartment, quantity in (
        ("Chlorine", RAW_WATER, 0),
        (
            "Chloride",
            SURFACE,
            (1 - treated) * chloride + treated * water.effluent.elements_kg["Cl"],
        ),
        ("DOC, Dissolved Organic Carbon", RAW_WATER, (1 - treated) * residue.leachate.doc_kg),
        ("DOC, Dissolved Organic Carbon", SURFACE, treated * water.effluent.doc_kg),
        ("Methane, non-fossil", URBAN, treated * 0.0004),
        ("Oxygen", AIR_RESOURCE, treated * (water.duties.biological_oxygen_kg + 0.01)),
    ):
        flow = bd.get_node(database="biosphere", code=flow_key(name, compartment)[1])
        # Brightway omits an ion with no exchange anywhere in the loaded
        # graph, as for chloride in this biogenic TPS fixture.
        row = lca.dicts.biosphere.get(flow.id)
        total = 0.0 if row is None else float(lca.inventory[row, :].sum())
        assert total == pytest.approx(quantity, rel=2e-7, abs=0)


def test_signed_sludge_markets_and_terminal_haul(compiled_water):
    """Normalize both signed markets and terminal outputs exactly once, retaining wet haul."""
    bd, _, native, future, origin, provider, _, water, _, lca = compiled_water
    treated = 0.6 if future else 0.8  # Same invented source collection boundary.
    haul = 0.1 if future else 0.05  # Invented native-market tkm/kg wet sludge.
    expected_haul = 0
    for role, wet, fraction, dry_fraction, water_mass in (
        (
            "incineration",
            water.incinerated_sludge_wet_kg,
            water.incineration_dry_share,
            0.3,
            water.water.incinerated_sludge_water_kg,
        ),
        (
            "landfarming",
            water.landfarmed_sludge_wet_kg,
            water.landfarming_dry_share,
            0.03,
            water.water.landfarmed_sludge_water_kg,
        ),
    ):
        # WWcalc C238/C236 gives sludge at 70%/97% water. The plant applies the disposal share.
        market = bd.get_node(
            database=provider, code=provider_service_code(f"test_tps_{role}_market_{origin}")
        )
        child = bd.get_node(
            database=provider, code=provider_service_code(f"test_tps_{role}_terminal_{origin}")
        )
        native_market = bd.get_node(database=native, code=f"{role}_market")
        p = next(iter(native_market.production()))["amount"]
        child_p = next(iter(child.production()))["amount"]
        expected_market_level = -treated * wet / p
        assert lca.supply_array[lca.dicts.activity[market.id]] == pytest.approx(
            expected_market_level, rel=2e-7, abs=0
        )
        assert lca.supply_array[lca.dicts.activity[child.id]] == pytest.approx(
            -treated * wet / child_p, rel=2e-7, abs=0
        )
        assert lca.supply_array[lca.dicts.activity[native_market.id]] == 0
        assert lca.supply_array[lca.dicts.activity[bd.get_node(database=native, code=role).id]] == 0
        rows = _amounts(market, "technosphere")
        assert rows[child.key] == p
        assert rows[native, "terminal_haul"] == pytest.approx(-p * haul, rel=2e-12, abs=0)
        expected_haul += treated * wet * haul
        dry = fraction * water.final_sludge.dry_mass_kg
        carbon = fraction * water.final_sludge.elements_kg["C"]
        reference = next(iter(market.production()))
        props = reference["properties"]
        assert reference["amount"] == p
        assert props["dry mass"]["amount"] == pytest.approx(dry_fraction, rel=2e-12, abs=0)
        assert props["water in wet mass"]["amount"] == pytest.approx(
            water_mass / wet, rel=2e-12, abs=0
        )
        assert props["carbon allocation"]["amount"] == pytest.approx(carbon / wet, rel=2e-12, abs=0)
        assert props["carbon content"]["amount"] == pytest.approx(carbon / dry, rel=2e-12, abs=0)
        link = next(r for r in market.technosphere() if r.input.key == child.key)
        assert link["properties"] == props
        assert not {"flow", "activity", "production volume", "classifications"} & set(link)
        old_link = next(
            r
            for r in native_market.technosphere()
            if r.input["reference product"] == market["reference product"]
        )
        for field in ("amount", "loc", "scale", "negative", "minimum", "maximum", "allocation"):
            assert link[field] == old_link[field]
        freight = next(r for r in market.technosphere() if r.input.key == (native, "terminal_haul"))
        old_freight = next(
            r for r in native_market.technosphere() if r.input.key == freight.input.key
        )
        for field in ("amount", "loc", "scale", "minimum", "maximum", "comment"):
            assert freight[field] == old_freight[field]
    freight_id = bd.get_node(database=native, code="terminal_haul").id
    assert lca.supply_array[lca.dicts.activity[freight_id]] == pytest.approx(
        expected_haul, rel=2e-7, abs=0
    )
