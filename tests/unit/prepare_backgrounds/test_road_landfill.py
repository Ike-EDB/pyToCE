"""Actual road mineral/carbon loads through complete sanitary and dump receivers."""

import json
from contextlib import contextmanager
from copy import deepcopy
from math import exp, fsum, log
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from pa_dump_receiving_fixtures import DUMP_CLASSES, LAND, LAND_ROWS
from road_landfill_fixtures import (
    LONG,
    MINERALS,
    ORIGINS,
    OWNERS,
    RURAL,
    SOIL,
    SURFACE,
    URBAN,
    dump_inventory,
    receiving_biosphere,
    road_conversion,
    sanitary_inventory,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _ROW_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.road_landfill_load import (
    _road_landfill_load,
)
from pytoce.prepare_backgrounds.preparation.providers.road_material import _road_material
from pytoce.run_lca.calculation.carbon import direct_treatment_carbon_flows


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize(
    "temperature,rain,end",
    (
        (16.743579766537, 1250.84108527132, 2863),
        (8, 350, 3171),
        (8, 550, 7789),
        (8, 900, 2658),
        (8, 1100, 1632),
        (8, 1900, 862),
    ),
)
def test_road_disposal_combines_three_accessibilities_without_geological_carbon_fate(
    origin,
    temperature,
    rain,
    end,
):
    """Independent source equations retain mass, actual minerals and organic pools."""
    conversion = road_conversion()
    origins = cast(
        tuple[float, float, float], tuple(float(origin == candidate) for candidate in ORIGINS)
    )
    material = _road_material(conversion, origins)
    site = _LandfillSite(temperature, rain, 1 if temperature != 8 else 1 - 2.0604 / 7)
    actual = _road_landfill_load(conversion, origin, site, carbonate_end_year=end)
    assert actual.wet_mass_kg == pytest.approx(material.mass_kg)
    assert actual.water_kg == material.water_kg > 0
    assert actual.geological_carbon_kg == material.limestone_elements_kg["C"] > 0
    assert fsum(
        actual.elements_kg.values()
    ) + actual.water_kg + actual.geological_carbon_kg == pytest.approx(actual.wet_mass_kg)
    assert actual.elements_kg["C"] == pytest.approx(0.6 + material.fossil_auxiliary_carbon_kg)
    assert [pool.origin for pool in actual.carbon] == [origin, "fossil"]
    assert actual.carbon[0].principal and not actual.carbon[1].principal
    assert actual.carbon[0].phases.input_kg == pytest.approx(0.6)
    # Original climate formula, independent of production helpers: rainfall m,
    # 60/220 source mean-MSW methane calibration, temperature reference +15 C.
    alpha = log(1 - 60 * (1 - exp(-29.372386439548432 * rain / 1000)) / 220)
    alpha *= log(1 - 60 * (1 - exp(-0.3 * (temperature + 15))) / 220) / log(1 - 60 / 220) ** 2
    releases = {
        "Ti": 0.05,
        "Si": 0.05,
        "Ca": 0.13024173278843815,
        "Al": 0.05,
        "K": 0.73130714062628965,
        "Mg": 0.61723443585572391,
        "Na": 4.13932455312919,
    }
    for element, release in releases.items():
        mass = (
            material.burnable_elements_kg.get(element, 0)
            + material.mineral_elements_kg.get(element, 0)
            + material.limestone_elements_kg.get(element, 0)
        )
        accessible = 0.01 * (
            material.burnable_elements_kg.get(element, 0)
            + material.mineral_elements_kg.get(element, 0)
        ) + 0.05 * material.limestone_elements_kg.get(element, 0)
        phase = actual.phases[element]
        assert phase.input_kg == pytest.approx(mass)
        assert phase.gas_kg == pytest.approx(accessible * alpha * release / 4001)
        assert phase.short_leachate_kg == pytest.approx(accessible * alpha * release * 4000 / 4001)
        assert phase.short_leachate_kg > 0
        assert fsum(
            (phase.gas_kg, phase.short_leachate_kg, phase.later_leachate_kg, phase.stored_kg)
        ) == pytest.approx(mass)
    energy = actual.gas.decomposition_trace
    assert fsum(
        energy[name]
        for name in (
            "raw_gas_hhv_mj",
            "heat_soil_mj",
            "short_leachate_hhv_mj",
            "later_leachate_hhv_mj",
            "stored_hhv_mj",
        )
    ) == pytest.approx(material.heat_potential_mj)
    assert actual.leachate.toc_kg == actual.phases["C"].short_leachate_kg


def test_road_water_receives_actual_titanium_potassium_and_magnesium():
    """Admitted D/N removal divides received Ti, K and Mg between sludge and effluent."""
    actual = _road_landfill_load(
        road_conversion(), "fossil", _LandfillSite(16, 1500, 1), carbonate_end_year=2863
    )
    # Use the declared hydraulic input to calculate removal.
    water = _wastewater_load(
        actual.leachate,
        volume_m3=0.02,
        incoming_water_kg=20 - fsum(actual.leachate.elements_kg.values()),
        evaporation_fraction=0,
        profile=_ROW_WASTEWATER_PROFILE,
    )
    for element, primary, total in (
        ("Ti", 0.27891639648419198, 0.92972132161397325),
        ("K", 0.015895638998058619, 0.052985463326862062),
        ("Mg", 0.02401950262181472, 0.0800650087393824),
    ):
        incoming = actual.leachate.elements_kg[element]
        f1, f2, f3 = _ROW_WASTEWATER_PROFILE.hydraulic.stages
        final = incoming * (primary * f1 + total * (f2 + f3))
        assert water.primary_sludge.elements_kg[element] == pytest.approx(incoming * primary)
        assert water.final_sludge.elements_kg[element] == pytest.approx(final)
        assert water.effluent.elements_kg[element] == pytest.approx(incoming - final)
        assert (
            water.biological_air_elements_kg[element]
            == water.digester_gas_elements_kg[element]
            == 0
        )


@contextmanager
def receiving_boundary(tmp_path, family, original):
    """Compile catalog declarations with synthetic original and dated inventories."""
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    kinds = (
        {"road_open_dump"}
        if family == "dump"
        else {
            "road_landfill",
            "road_leachate_collection",
            "road_wastewater_treatment",
            "road_sludge_collection",
            "road_sludge_incineration",
            "road_sludge_landfarming",
            "road_sludge_ash_market",
            "road_sludge_ash_treatment",
        }
    )
    services = {
        row["provider_service_id"]: row
        for row in catalogue["services"]
        if row["construction"]["kind"] in kinds
    }
    provider = PreparedProviderAuthority(
        identity="road_receiving_boundary",
        interface_database="road_interface",
        catalogue={**catalogue, "services": list(services.values()), "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(tmp_path / "brightway", "road_receiving", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(receiving_biosphere(), searchable=False)
        for database in ("original", "future"):
            data = (
                sanitary_inventory(database, future=database == "future")
                if family == "sanitary"
                else dump_inventory(catalogue, database, original=database == "original")
            )
            bd.Database(database).write(data, searchable=False)
        before = {
            node.key: (deepcopy(dict(node)), [dict(row) for row in node.exchanges()])
            for database in ("original", "future")
            for node in bd.Database(database)
        }
        database = "original" if original else "future"
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2025 if original else 2050,
            ordinary_database=database,
            technology_database=database,
            provider=provider,
            material_recipe=physical_material_recipe(),
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )
        yield bd, frontier, services, database
        assert before == {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("original", "future")
            for node in bd.Database(database)
        }


@pytest.fixture(scope="module", params=(True, False), ids=("original", "future"))
def sanitary_boundary(request, tmp_path_factory):
    """Reuse the prepared sanitary treatment chain across the three carbon origins."""
    with receiving_boundary(
        tmp_path_factory.mktemp("road_sanitary"), "sanitary", request.param
    ) as boundary:
        yield boundary


@pytest.fixture(scope="module", params=(True, False), ids=("original", "future"))
def dump_boundary(request, tmp_path_factory):
    """Reuse the five prepared dump sites across the three carbon origins."""
    with receiving_boundary(
        tmp_path_factory.mktemp("road_dump"), "dump", request.param
    ) as boundary:
        yield boundary


def emitted(node, name, categories):
    """Sum one bound elementary quantity on its native signed operation reference."""
    return fsum(
        row["amount"]
        for row in node.biosphere()
        if row.input["name"] == name and tuple(row.input["categories"]) == categories
    )


@pytest.mark.parametrize("origin", ORIGINS)
def test_actual_road_sanitary_chain_keeps_operations_and_all_mineral_receivers(
    sanitary_boundary, origin
):
    """Original material/climate survive dated supplier relinking and nonunit waste signs."""
    bd, frontier, services, database = sanitary_boundary
    nodes = {
        role: bd.get_node(
            database=frontier.database_name, code=provider_service_code(f"road_{origin}_{role}")
        )
        for role in OWNERS
    }
    site = nodes["landfill"]
    actual = _road_landfill_load(
        road_conversion(), origin, _LandfillSite(16, 1500, 1), carbonate_end_year=2863
    )
    properties = site.rp_exchange()["properties"]
    assert properties["wet mass"]["amount"] == pytest.approx(actual.wet_mass_kg)
    assert properties["dry mass"]["amount"] + properties["water in wet mass"][
        "amount"
    ] == pytest.approx(actual.wet_mass_kg)
    assert (
        properties["geological carbon outside organic fate"]["amount"]
        == actual.geological_carbon_kg
    )
    assert properties["carbon allocation"]["amount"] == actual.elements_kg["C"]
    # EcoSpold content properties are kg element/kg dry matter, including for wet coated beads.
    dry = properties["dry mass"]["amount"]
    for name, element in {
        "carbon": "C",
        "hydrogen": "H",
        "oxygen": "O",
        "nitrogen": "N",
        "sulfur": "S",
        "phosphorus": "P",
        "sodium": "Na",
        "silicon": "Si",
        "calcium": "Ca",
        "aluminium": "Al",
        "titanium": "Ti",
        "potassium": "K",
        "magnesium": "Mg",
    }.items():
        assert dry * properties[f"{name} content"]["amount"] == pytest.approx(
            actual.elements_kg[element], rel=1e-12
        )
    non_fossil = 0.0 if origin == "fossil" else 0.6
    assert dry * properties["carbon content, non-fossil"]["amount"] == pytest.approx(non_fossil)
    assert dry * properties["carbon content, fossil"]["amount"] == pytest.approx(
        actual.elements_kg["C"] - non_fossil
    )
    native = bd.get_node(database=database, code="site")
    multiplier = actual.wet_mass_kg / -native.rp_exchange()["amount"]
    source = next(row for row in native.technosphere() if row.input["code"] == "site_equipment")
    target = next(row for row in site.technosphere() if row.input["code"] == "site_equipment")
    assert target["amount"] == pytest.approx(source["amount"] * multiplier)
    assert target["loc"] == pytest.approx(source["loc"] + log(multiplier))
    assert target["scale"] == source["scale"]
    lca = LCA({site: 1.0})
    lca.lci()
    assert cast(NDArray[np.float64], lca.supply_array)[
        lca.dicts.activity[site.id]
    ] == pytest.approx(1)
    for role, (code, _, _) in OWNERS.items():
        if role != "landfill":
            assert (
                nodes[role].rp_exchange()["amount"]
                == bd.get_node(database=database, code=code).rp_exchange()["amount"]
                < 0
            )
    for name, (element, _) in MINERALS.items():
        assert emitted(site, name, LONG) == pytest.approx(actual.phases[element].later_leachate_kg)
        assert emitted(site, name, RURAL) == pytest.approx(actual.phases[element].gas_kg)
        assert emitted(nodes["leachate_collection"], name, SURFACE) > 0
        assert emitted(nodes["wastewater_treatment"], name, SURFACE) > 0
        assert emitted(nodes["sludge_landfarming"], name, SOIL) > 0
        assert emitted(nodes["sludge_incineration"], name, URBAN) > 0
        assert emitted(nodes["sludge_incineration"], name, SURFACE) > 0
        for role in ("bottom_ash_treatment", "residual_ash_treatment"):
            assert emitted(nodes[role], name, SURFACE) > 0
            assert emitted(nodes[role], name, LONG) > 0
    assert emitted(nodes["sludge_incineration"], "Sulfur dioxide", URBAN) > 0
    assert emitted(nodes["sludge_incineration"], "Sulfate", SURFACE) > 0
    assert emitted(nodes["sludge_landfarming"], "Sulfur", SOIL) > 0
    thermal = json.loads(
        next(
            row["project_terms_json"]
            for row in frontier.technology_rows
            if row.get("provider_service_id") == f"road_{origin}_sludge_incineration"
            and row.get("component_id") == "actual_sludge_incineration"
        )
    )
    scale = -nodes["sludge_incineration"].rp_exchange()["amount"] / thermal["wet_mass_kg"]
    # Independent original sludgeIncin52/59/60 K/L values are
    # fractions of incoming element mass released to treated water and air.
    for name, element, water_fraction, air_fraction in (
        ("Titanium ion", "Ti", 0.000263059743864643, 0.000829649961419258),
        ("Potassium I", "K", 0.00494050030057791, 0.00494050030057791),
        ("Magnesium", "Mg", 0.00131847077363618, 0.00105477661890894),
    ):
        incoming = thermal["elements_kg"][element]
        assert emitted(nodes["sludge_incineration"], name, URBAN) / scale == pytest.approx(
            incoming * air_fraction
        )
        assert emitted(nodes["sludge_incineration"], name, SURFACE) / scale == pytest.approx(
            incoming * water_fraction
        )
    # Fresh APC aqueous lime/alkali must remain in addition to the actual
    # road calcium/sodium. Their original paint donor has no such rows.
    for name, element, fraction in (
        ("Calcium II", "Ca", 0.00786987251060033),
        ("Sodium I", "Na", 0.00821018062397373),
    ):
        expected = (
            thermal["elements_kg"][element] * fraction + thermal["aqueous_apc_elements_kg"][element]
        )
        assert emitted(nodes["sludge_incineration"], name, SURFACE) / scale == pytest.approx(
            expected
        )
    sulfur = thermal["elements_kg"]["S"]
    assert emitted(nodes["sludge_incineration"], "Sulfate", SURFACE) / scale == pytest.approx(
        sulfur * 0.079758676892409 * 96 / 32
    )
    assert emitted(nodes["sludge_incineration"], "Sulfur dioxide", URBAN) / scale == pytest.approx(
        sulfur * 0.0053172451261606 * 64 / 32
    )
    bottom = thermal["bottom_ash"]
    assert bottom["water_kg"] == pytest.approx(
        (bottom["treatment_mass_kg"] - bottom["water_kg"]) * 0.186047 / 0.813953
    )
    # A7/A8 distinct receiving factors, tested on actual sludge ash with
    # its own moisture and full fresh-APC mineral contribution.
    for fraction, name, element, short, cumulative in (
        ("bottom", "Titanium ion", "Ti", 0.000004824, 0.2999),
        ("bottom", "Silicon", "Si", 0.0000369, 0.008593),
        ("residual", "Potassium I", "K", 0.2819, 1),
        ("residual", "Vanadium V", "V", 0.002454, 0.7711),
    ):
        ash = thermal[f"{fraction}_ash"]
        incoming = ash["waste_elements_kg"].get(element, 0) + ash["auxiliary_elements_kg"].get(
            element, 0
        )
        node = nodes[f"{fraction}_ash_treatment"]
        basis = ash["treatment_mass_kg"] / -node.rp_exchange()["amount"]
        assert emitted(node, name, SURFACE) * basis == pytest.approx(incoming * short)
        assert emitted(node, name, LONG) * basis == pytest.approx(incoming * (cumulative - short))
    field = json.loads(
        next(
            row["project_terms_json"]
            for row in frontier.technology_rows
            if row.get("provider_service_id") == f"road_{origin}_sludge_landfarming"
            and row.get("component_id") == "actual_sludge_landfarming"
        )
    )
    for name, (element, _) in MINERALS.items():
        amount = emitted(nodes["sludge_landfarming"], name, SOIL)
        assert amount * field["wet_kg"] / -nodes["sludge_landfarming"].rp_exchange()[
            "amount"
        ] == pytest.approx(field["actual_elements_kg"][element])
    # Negative waste demands divided by negative treatment production give
    # positive operations; no avoided-treatment credit occurs.
    assert all(
        cast(NDArray[np.float64], lca.supply_array)[lca.dicts.activity[node.id]] > 0
        for role, node in nodes.items()
        if role != "landfill"
    )
    trace = json.loads(site["direct_treatment_carbon_owners_json"])
    expected_owners = {
        nodes[role].key[1]
        for role in (
            "landfill",
            "wastewater_treatment",
            "sludge_incineration",
            "sludge_landfarming",
        )
    }
    assert {row["owner_code"] for row in trace} == expected_owners
    assert len(trace) == len(expected_owners)
    year = 2025 if database == "original" else 2050
    diagnostic = direct_treatment_carbon_flows(
        bd,
        pd.DataFrame(
            [
                {
                    "year": year,
                    "region": "EUR",
                    "life_cycle_stage": "landfill_treatment",
                    "material": "Others",
                    "constituent": "road",
                    "product": "polymer",
                    "virgin_feedstock_route": origin,
                    "good": "all",
                    "material_cycle_source": "virgin_feedstock",
                    "ultimate_carbon_origin": origin,
                    "lca_fate": "landfill",
                    "fate_origin": "remind_mfa",
                    "provider_database": frontier.database_name,
                    "provider_code": site.key[1],
                    "provider_anchor_year": year,
                    "amount": 1.0,
                    # The road functional unit contains one kg PMMA at 60% C;
                    # fossil auxiliaries and geological carbonate are separate.
                    "mfa_carbon_amount_kg": 0.6,
                }
            ]
        ),
    )
    assert set(diagnostic.owner_code) == expected_owners
    assert set(diagnostic.mfa_carbon_origin) == {origin}
    assert set(diagnostic.origin_mapping_status) == {"actual_road_principal_origin"}
    # The solved inventory independently supplies each receiving operation.
    # Compare only its principal-attributable gases; fossil auxiliaries stay
    # in LCI but outside this diagnostic, including for non-fossil principal.
    for spec in trace:
        owner = bd.get_node(database=spec["owner_database"], code=spec["owner_code"])
        supply = cast(NDArray[np.float64], lca.supply_array)[lca.dicts.activity[owner.id]]
        assert spec["owner_scale_per_provider_unit"] / owner.rp_exchange()[
            "amount"
        ] == pytest.approx(supply, rel=2e-7)
        expected = {}
        for row in owner.biosphere():
            if "principal_carbon_origin_fractions" not in row:
                continue
            amount = row["amount"] * row["principal_carbon_origin_fractions"][ORIGINS.index(origin)]
            expected[row.input.key[1]] = expected.get(row.input.key[1], 0.0) + supply * amount
        actual_flows = diagnostic.loc[diagnostic.owner_code.eq(owner.key[1])]
        for flow_code, amount in expected.items():
            assert actual_flows.loc[actual_flows.flow_code.eq(flow_code), "flow_amount"].sum() == (
                pytest.approx(amount, rel=2e-7)
            )


@pytest.mark.parametrize("origin", ORIGINS)
def test_road_dump_uses_five_original_climates_and_actual_minerals(dump_boundary, origin):
    """Hyperarid air/soil and positive-infiltration water retain the actual road burden."""
    bd, frontier, services, database = dump_boundary
    original = database == "original"
    for index, (_, suffix, rain) in enumerate(DUMP_CLASSES):
        sid = f"road_{origin}_open_dump_{suffix}"
        node = bd.get_node(database=frontier.database_name, code=provider_service_code(sid))
        actual = _road_landfill_load(
            road_conversion(),
            origin,
            _LandfillSite(8, rain, 1 - 2.0604 / 7),
            carbonate_end_year=services[sid]["construction"]["carbonate_end_year"],
        )
        assert (
            node.rp_exchange()["properties"]["geological carbon outside organic fate"]["amount"]
            == actual.geological_carbon_kg
        )
        assert not list(node.technosphere())
        for name, _, coefficient in LAND_ROWS:
            assert emitted(node, name, LAND) == pytest.approx(
                coefficient * actual.wet_mass_kg * (1 if original else 1.7)
            )
        for name, (element, _) in MINERALS.items():
            if index == 0:
                assert emitted(node, name, ("soil", "industrial")) == pytest.approx(
                    0.9
                    * (
                        actual.phases[element].short_leachate_kg
                        + actual.phases[element].later_leachate_kg
                    )
                )
            else:
                assert emitted(node, name, ("water", "ground-")) == pytest.approx(
                    actual.phases[element].short_leachate_kg
                )
                assert emitted(node, name, LONG) == pytest.approx(
                    actual.phases[element].later_leachate_kg
                )
