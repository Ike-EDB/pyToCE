"""Actual road coating and bulk fillers through complete municipal and ash LCI."""

import json
from copy import deepcopy
from math import fsum, log
from typing import Any

import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from road_municipal_fixtures import (
    LONG,
    ORIGINS,
    SURFACE,
    URBAN,
    road_municipal_biosphere,
    road_municipal_inventory,
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
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)
from pytoce.run_lca.calculation.carbon import direct_treatment_carbon_flows


def _terms(frontier, service_id, component):
    """Read the actual record retained by the shared preparation owner."""
    return json.loads(
        next(
            row["project_terms_json"]
            for row in frontier.technology_rows
            if row["provider_service_id"] == service_id and row["component_id"] == component
        )
    )


def _emitted(node, name, categories):
    """Sum one bound elementary species and compartment on its signed reference."""
    return fsum(
        row["amount"]
        for row in node.biosphere()
        if row.input["name"] == name and tuple(row.input["categories"]) == categories
    )


@pytest.mark.parametrize("original", (True, False), ids=("original", "future"))
def test_road_municipal_binds_actual_bulk_organic_origin_and_complete_ash(tmp_path, original):
    """All principal origins preserve actual wet mass and both complete paint ash graphs."""
    catalogue = load_provider_catalogue()
    services = {
        row["provider_service_id"]: row
        for row in catalogue["services"]
        if row["construction"]["kind"]
        in {"road_incineration", "road_ash_market", "road_ash_treatment"}
    }
    assert len(services) == 15
    provider = PreparedProviderAuthority(
        identity="road_municipal_boundary",
        interface_database="road_interface",
        catalogue={**catalogue, "services": list(services.values()), "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(tmp_path / "brightway", "road_municipal", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(road_municipal_biosphere(), searchable=False)
        for database in ("original", "future"):
            bd.Database(database).write(
                road_municipal_inventory(catalogue, database, original=database == "original"),
                searchable=False,
            )
        before = {
            node.key: (deepcopy(dict(node)), [dict(row) for row in node.exchanges()])
            for database in ("original", "future")
            for node in bd.Database(database)
        }
        database = "original" if original else "future"
        year = 2025 if original else 2050
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=database,
            technology_database=database,
            provider=provider,
            material_recipe=physical_material_recipe(),
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )
        for origin in ORIGINS:
            sid = f"road_{origin}_municipal_incineration"
            node = bd.get_node(database=frontier.database_name, code=provider_service_code(sid))
            load = _terms(frontier, sid, "actual_road_municipal_incineration")
            material = load["material"]
            assert node.rp_exchange()["amount"] == 1
            assert material["wet_mass_kg"] == pytest.approx(4.561737142857143)
            assert material["water_kg"] == pytest.approx(0.00032569509492586413)
            assert fsum(material["elements_kg"].values()) == pytest.approx(1.0453928491466953)
            bulk = material["bulk_inert_elements_kg"]
            geo_c = material["geological_carbon_kg"]
            assert fsum(bulk.values()) + geo_c == pytest.approx(3.516018598615522)
            assert geo_c == pytest.approx((0.1485 / 0.21875) * 12 / 100.078)
            assert load["physical_elements_kg"]["C"] == pytest.approx(
                material["elements_kg"]["C"] + geo_c
            )
            assert material["elements_kg"].keys() == {"C", "H", "O", "Si"}
            assert load["duties"]["NaOH"] == load["duties"]["CaO"] == 0
            # This fixture's original .7 kg aqueous-water meter excludes the
            # new bulk filler duty. Add only bulk quench at selected moisture;
            # no clipping or repair of the native aqueous coefficient is used.
            bulk_quench = (fsum(bulk.values()) + geo_c) * 0.173671 / 0.826329
            assert load["thermal"]["bulk_quench_water_kg"] == pytest.approx(bulk_quench)
            assert load["duties"]["water"] == pytest.approx(
                0.7 * load["drivers"]["aqueous"] / load["source_drivers"]["aqueous"] + bulk_quench
            )
            assert load["thermal"]["process_water_kg"] == pytest.approx(
                load["duties"]["water"] - load["bottom_ash"]["water_kg"]
            )
            assert load["thermal"]["process_water_kg"] > 0
            for code, share in (
                ("phb_water_a", 0.4 if original else 0.8),
                ("phb_water_b", 0.6 if original else 0.2),
            ):
                water = next(
                    row for row in node.technosphere() if row.input.key == (database, code)
                )
                assert water["amount"] == pytest.approx(load["duties"]["water"] * share)
                assert water["scale"] == 0.21
                assert water["loc"] == pytest.approx(log(water["amount"]) + 0.03)
            assert _emitted(node, "Sulfur dioxide", URBAN) == 0
            assert _emitted(node, "Sulfate", SURFACE) == 0
            # Bulk oxides bypass combustion. Trace PDMS silicon follows Municipal2020 phase
            # fractions.
            assert _emitted(node, "Silicon", URBAN) == pytest.approx(
                material["elements_kg"]["Si"] * 0.0000015575195263856684
            )
            assert _emitted(
                node, "Non-hazardous waste disposed", ("inventory indicator", "waste")
            ) == pytest.approx(material["wet_mass_kg"])
            # Original facility allocation applies to whole wet material; dated
            # water shares and complete utility chains remain independently linked.
            facility = next(
                row for row in node.technosphere() if row.input.key == (database, "phb_facility")
            )
            assert facility["amount"] == pytest.approx(0.00001 * material["wet_mass_kg"])
            assert facility["scale"] == 0.21
            assert facility["loc"] == pytest.approx(log(facility["amount"]) + 0.03)
            lca: Any = LCA({node: 1})
            lca.lci()
            for fraction in ("bottom", "residual"):
                ash = load[fraction + "_ash"]
                terminal = bd.get_node(
                    database=frontier.database_name,
                    code=provider_service_code(f"road_{origin}_municipal_{fraction}_ash_treatment"),
                )
                market = bd.get_node(
                    database=frontier.database_name,
                    code=provider_service_code(f"road_{origin}_municipal_{fraction}_ash_market"),
                )
                basis = -terminal.rp_exchange()["amount"]
                assert basis == (3 if original else 7)
                assert lca.supply_array[lca.dicts.activity[terminal.id]] * basis == pytest.approx(
                    ash["treatment_mass_kg"], rel=2e-7
                )
                native = bd.get_node(database=database, code=f"phb_{fraction}_terminal")
                assert [
                    {key: value for key, value in dict(row).items() if key != "output"}
                    for row in terminal.technosphere()
                ] == [
                    {key: value for key, value in dict(row).items() if key != "output"}
                    for row in native.technosphere()
                ]
                freight = [
                    row for row in market.technosphere() if row.input["unit"] == "ton kilometer"
                ]
                assert len(freight) == 3
                assert fsum(row["amount"] for row in freight) / -market.rp_exchange()[
                    "amount"
                ] == pytest.approx(0.077)
                assert all(row["scale"] == 0.21 for row in freight)
                source = bd.get_node(database="original", code=f"phb_{fraction}_terminal")
                native_doc = _emitted(source, "DOC, Dissolved Organic Carbon", SURFACE)
                props = source.rp_exchange()["properties"]
                actual_c = ash["waste_elements_kg"]["C"] + ash["auxiliary_elements_kg"].get("C", 0)
                expected_doc = (
                    native_doc
                    / -source.rp_exchange()["amount"]
                    * actual_c
                    / (props["dry mass"]["amount"] * props["carbon content"]["amount"])
                    * basis
                    / ash["treatment_mass_kg"]
                )
                assert _emitted(
                    terminal, "DOC, Dissolved Organic Carbon", SURFACE
                ) == pytest.approx(expected_doc)
                # Doka III A7/A8 source fractions for actual paint-missing atoms.
                profiles = (
                    ("Titanium ion", "Ti", 1, 0.000004824, 0.2999, 0.0004956, 0.2973),
                    ("Silicon", "Si", 1, 0.0000369, 0.008593, 0.002249, 1.0),
                    ("Calcium II", "Ca", 1, 0.004363, 1.0, 0.0001508, 0.09051),
                    ("Aluminium III", "Al", 1, 0.00001314, 0.8169, 0.0004956, 0.2973),
                    ("Potassium I", "K", 1, 0.1208, 1.0, 0.2819, 1.0),
                    ("Magnesium", "Mg", 1, 0.003799, 1.0, 0.0001897, 0.1138),
                    ("Sodium I", "Na", 1, 0.122, 1.0, 0.3753, 1.0),
                    ("Sulfate", "S", 32 / 96, 0.09119, 1.0, 0.1073, 1.0),
                )
                for name, element, elemental, bs, bc, rs, rc in profiles:
                    mass = ash["waste_elements_kg"].get(element, 0) + ash[
                        "auxiliary_elements_kg"
                    ].get(element, 0)
                    short, cumulative = (bs, bc) if fraction == "bottom" else (rs, rc)
                    normalization = basis / ash["treatment_mass_kg"] / elemental
                    assert _emitted(terminal, name, SURFACE) == pytest.approx(
                        mass * short * normalization
                    )
                    assert _emitted(terminal, name, LONG) == pytest.approx(
                        mass * (cumulative - short) * normalization
                    )
                if fraction == "bottom":
                    assert ash["geological_carbon_kg"] == geo_c
                    for element in ("Ti", "Ca", "Al", "K", "Mg", "Na", "S"):
                        assert ash["waste_elements_kg"][element] == bulk[element]
                    # Quench uses the selected paint .173671 wet fraction on
                    # whole inert mass plus the source raw organic slag pool.
                    raw_slag = (
                        fsum(
                            mass - bulk.get(element, 0)
                            for element, mass in ash["waste_elements_kg"].items()
                        )
                        + material["elements_kg"]["O"] * 24.5454545454545 / 1000
                    )
                    assert ash["water_kg"] == pytest.approx(
                        (raw_slag + fsum(bulk.values()) + geo_c) * 0.173671 / 0.826329
                    )
                else:
                    assert ash["geological_carbon_kg"] == 0
                    normalization = basis / ash["treatment_mass_kg"]
                    assert _emitted(terminal, "Vanadium V", SURFACE) == pytest.approx(
                        ash["auxiliary_elements_kg"]["V"] * 0.002454 * normalization
                    )
                    assert _emitted(terminal, "Vanadium V", LONG) == pytest.approx(
                        ash["auxiliary_elements_kg"]["V"] * (0.7711 - 0.002454) * normalization
                    )
                    native_n = _emitted(source, "Nitrate", SURFACE)
                    expected_n = (
                        native_n
                        / -source.rp_exchange()["amount"]
                        * ash["auxiliary_elements_kg"]["N"]
                        / (props["dry mass"]["amount"] * props["nitrogen content"]["amount"])
                        * normalization
                    )
                    assert _emitted(terminal, "Nitrate", SURFACE) == pytest.approx(expected_n)
                    nitrate = next(
                        row
                        for row in terminal.biosphere()
                        if row.input["name"] == "Nitrate"
                        and tuple(row.input["categories"]) == SURFACE
                    )
                    assert nitrate["scale"] == 0.21
                    assert nitrate["loc"] == pytest.approx(log(nitrate["amount"]) + 0.03)
            activation = pd.DataFrame(
                [
                    {
                        "year": year,
                        "region": "EUR",
                        "life_cycle_stage": "incineration_treatment",
                        "material": "Others",
                        "constituent": "road_marking",
                        "product": "polymer",
                        "virgin_feedstock_route": origin,
                        "good": "all",
                        "material_cycle_source": "virgin_feedstock",
                        "ultimate_carbon_origin": origin,
                        "lca_fate": "incineration",
                        "fate_origin": "remind_mfa",
                        "provider_database": frontier.database_name,
                        "provider_code": node.key[1],
                        "provider_anchor_year": year,
                        "amount": 1.0,
                        "mfa_carbon_amount_kg": 0.6,
                    }
                ]
            )
            diagnostic = direct_treatment_carbon_flows(bd, activation)
            assert set(diagnostic.origin_mapping_status) == {"actual_road_principal_origin"}
            assert set(diagnostic.mfa_carbon_origin) == {origin}
            co2 = _emitted(node, "Carbon dioxide, fossil", URBAN) + _emitted(
                node, "Carbon dioxide, non-fossil", URBAN
            )
            assert diagnostic.loc[
                diagnostic.carbon_species.eq("carbon dioxide"), "carbon_amount"
            ].sum() == pytest.approx(co2 * 12 / 44 * 0.6 / material["elements_kg"]["C"])
        assert before == {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("original", "future")
            for node in bd.Database(database)
        }
