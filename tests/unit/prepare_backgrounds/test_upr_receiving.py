"""Actual UPR minerals and carbon through complete dated receiving functions."""

import json
from contextlib import contextmanager
from copy import deepcopy
from math import fsum, log
from typing import Any, cast

import pandas as pd
import pytest
from bw2calc import LCA
from pa_dump_receiving_fixtures import DUMP_CLASSES, LAND_ROWS
from sbr_landfill_fixtures import OWNERS
from upr_receiving_fixtures import (
    AGRICULTURAL,
    LONG,
    MINERALS,
    ORIGINS,
    RURAL,
    SURFACE,
    URBAN,
    municipal_inventory,
    receiving_biosphere,
    sanitary_inventory,
    unmanaged_inventory,
    upr_recipe,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
    load_provider_catalogue,
    prepared_provider_authority,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.upr_landfill_load import (
    _upr_landfill_load,
)
from pytoce.prepare_backgrounds.preparation.providers.upr_material import _upr_material
from pytoce.run_lca.calculation.carbon import direct_treatment_carbon_flows
from pytoce.run_lca.calculation.providers import MaterialCarbon, ProviderRoutes


@contextmanager
def receiving_boundary(tmp_path, family, original):
    """Compile only the actual complete receiving family through the public preparation owner."""
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    kinds = {
        "municipal": {"upr_incineration", "upr_ash_market", "upr_ash_treatment"},
        "sanitary": {
            "upr_landfill",
            "upr_leachate_collection",
            "upr_wastewater_treatment",
            "upr_sludge_collection",
            "upr_sludge_incineration",
            "upr_sludge_landfarming",
            "upr_sludge_ash_market",
            "upr_sludge_ash_treatment",
        },
        "unmanaged": {"upr_open_dump", "upr_open_burning"},
    }[family]
    services = {
        row["provider_service_id"]: row
        for row in catalogue["services"]
        if row["construction"]["kind"] in kinds
    }
    provider = PreparedProviderAuthority(
        identity="upr_receiving_boundary",
        interface_database="upr_interface",
        catalogue={**catalogue, "services": list(services.values()), "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(tmp_path / "brightway", "upr_receiving", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(receiving_biosphere(family), searchable=False)
        for database in ("original", "future"):
            if family == "municipal":
                data = municipal_inventory(catalogue, database, original=database == "original")
            elif family == "sanitary":
                data = sanitary_inventory(database, future=database == "future")
            else:
                data = unmanaged_inventory(catalogue, database, original=database == "original")
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
            material_recipe=upr_recipe(),
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )
        yield bd, frontier, services, database
        assert before == {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("original", "future")
            for node in bd.Database(database)
        }


def emitted(node, name, categories):
    """Sum bound elementary exchanges in their flow unit on the activity's reference basis."""
    return fsum(
        row["amount"]
        for row in node.biosphere()
        if row.input["name"] == name and tuple(row.input["categories"]) == categories
    )


def technology_terms(frontier, service_id, component):
    """Read the one source-derived record retained by the real compiler."""
    return json.loads(
        next(
            row["project_terms_json"]
            for row in frontier.technology_rows
            if row["provider_service_id"] == service_id and row["component_id"] == component
        )
    )


@pytest.mark.parametrize("original", (True, False), ids=("original", "future"))
def test_upr_municipal_binds_actual_minerals_and_complete_signed_ash(tmp_path, original):
    """Route bulk minerals to bottom ash, preserving reacting atoms and dated supplies."""
    with receiving_boundary(tmp_path, "municipal", original) as (bd, frontier, services, database):
        assert len(services) == 15
        for origin in ORIGINS:
            sid = f"upr_{origin}_municipal_incineration"
            node = bd.get_node(database=frontier.database_name, code=provider_service_code(sid))
            load = technology_terms(frontier, sid, "actual_upr_municipal_incineration")
            assert node.rp_exchange()["amount"] == 1
            assert load["material"]["wet_mass_kg"] == pytest.approx(2.659625150379939)
            assert load["material"]["heat_potential_mj"] == pytest.approx(30.164081970657673)
            assert load["material"]["water_kg"] == 0
            reacting = load["material"]["elements_kg"]
            bulk = load["material"]["bulk_inert_elements_kg"]
            assert reacting.keys() == {"C", "H", "O", "Si", "Co"}
            assert bulk.keys() == {"Si", "Al", "Ca", "B", "O"}
            assert fsum((*reacting.values(), *bulk.values())) == pytest.approx(
                load["material"]["wet_mass_kg"]
            )
            # Retained reinforcement is 99% E-glass with 22% CaO. Bulk calcium retains the
            # inert-phase account; fresh APC calcium reaches water/residual ash.
            assert bulk["Ca"] == pytest.approx(0.99 * 1.6329013548044928 * 0.22 * 40.078 / 56.078)
            for element in ("B", "Al", "Ca"):
                assert load["bottom_ash"]["waste_elements_kg"][element] == bulk[element]
                assert load["residual_ash"]["waste_elements_kg"].get(element, 0) == 0
                assert load["air_elements_kg"].get(element, 0) == 0
                assert load["water_elements_kg"].get(element, 0) == 0
            assert emitted(node, "Aluminium III", URBAN) == 0
            assert emitted(node, "Calcium II", URBAN) == 0
            assert emitted(node, "Calcium II", SURFACE) == pytest.approx(
                load["aqueous_auxiliary_elements_kg"]["Ca"]
            )
            # Sizing organosiloxane Si remains reactive; the much larger bulk
            # glass/silica Si is preserved whole in bottom ash.
            assert reacting["Si"] > 0
            assert load["bottom_ash"]["waste_elements_kg"]["Si"] == pytest.approx(
                bulk["Si"] + reacting["Si"] * 0.90654415585828087
            )
            for compartment in (URBAN, SURFACE):
                assert emitted(node, "Silicon", compartment) == pytest.approx(
                    reacting["Si"] * 0.0000015575195263856684, rel=1e-12, abs=0
                )
            # Dispersed cobalt retains the original intensive source factors
            # and uncertainty, including on future native supplies.
            native = bd.get_node(database="original", code="phb_furnace")
            props = native.rp_exchange()["properties"]
            for compartment, coefficient in ((URBAN, 1e-7), (SURFACE, 3e-7)):
                expected = (
                    coefficient
                    * reacting["Co"]
                    / (props["dry mass"]["amount"] * props["cobalt content"]["amount"])
                )
                assert emitted(node, "Cobalt II", compartment) == pytest.approx(expected)
                row = next(
                    row
                    for row in node.biosphere()
                    if row.input["name"] == "Cobalt II"
                    and tuple(row.input["categories"]) == compartment
                )
                assert row["loc"] == pytest.approx(log(row["amount"]) + 0.03)
                assert row["scale"] == 0.21
            # Original bottom-ash water fraction applies to the new bulk mass
            # as well as reacting slag. Preserve the complete aqueous meter
            # and its two dated suppliers rather than subtracting this duty.
            water_fraction = 168427.10315919179 / 1e6
            bulk_quench = fsum(bulk.values()) * water_fraction / (1 - water_fraction)
            assert load["thermal"]["bulk_quench_water_kg"] == pytest.approx(bulk_quench)
            assert load["duties"]["water"] == pytest.approx(
                0.7 * load["drivers"]["aqueous"] / load["source_drivers"]["aqueous"] + bulk_quench
            )
            assert load["thermal"]["process_water_kg"] == pytest.approx(
                load["duties"]["water"] - load["bottom_ash"]["water_kg"]
            )
            assert load["thermal"]["process_water_kg"] > 0
            outgoing_water = 1000 * fsum(
                emitted(node, "Water", compartment) for compartment in (URBAN, SURFACE)
            )
            assert outgoing_water + load["bottom_ash"]["water_kg"] == pytest.approx(
                load["duties"]["water"] + load["thermal"]["reaction_water_kg"], rel=2e-14
            )
            for code, amount in (
                ("phb_water_a", load["duties"]["water"] * (0.4 if original else 0.8)),
                ("phb_water_b", load["duties"]["water"] * (0.6 if original else 0.2)),
                ("phb_facility", 0.00001 * load["material"]["wet_mass_kg"]),
            ):
                row = next(row for row in node.technosphere() if row.input.key == (database, code))
                assert row["amount"] == pytest.approx(amount)
                assert row["loc"] == pytest.approx(log(row["amount"]) + 0.03)
                assert row["scale"] == 0.21
            lca: Any = LCA({node: 1})
            lca.lci()
            for fraction in ("bottom", "residual"):
                ash = load[fraction + "_ash"]
                terminal = bd.get_node(
                    database=frontier.database_name,
                    code=provider_service_code(f"upr_{origin}_municipal_{fraction}_ash_treatment"),
                )
                basis = -terminal.rp_exchange()["amount"]
                assert basis == (3 if original else 7)
                assert lca.supply_array[lca.dicts.activity[terminal.id]] * basis == pytest.approx(
                    ash["treatment_mass_kg"], rel=2e-7
                )
                # Source A7/A8 fractions multiply actual element mass, once;
                # no donor B/Si/Ca row is available to silently substitute.
                profiles = (
                    ("Boron", "B", 0.007835, 0.8399 if fraction == "bottom" else 0.9911),
                    (
                        "Silicon",
                        "Si",
                        0.0000369 if fraction == "bottom" else 0.002249,
                        0.008593 if fraction == "bottom" else 1.0,
                    ),
                    (
                        "Calcium II",
                        "Ca",
                        0.004363 if fraction == "bottom" else 0.0001508,
                        1.0 if fraction == "bottom" else 0.09051,
                    ),
                )
                for name, element, short, cumulative in profiles:
                    mass = ash["waste_elements_kg"].get(element, 0) + ash[
                        "auxiliary_elements_kg"
                    ].get(element, 0)
                    normalization = basis / ash["treatment_mass_kg"]
                    assert emitted(terminal, name, SURFACE) == pytest.approx(
                        mass * short * normalization
                    )
                    assert emitted(terminal, name, LONG) == pytest.approx(
                        mass * (cumulative - short) * normalization
                    )
                for name in ("Aluminium III", "Cobalt II"):
                    for compartment in (SURFACE, LONG):
                        if name == "Aluminium III" and fraction == "residual":
                            assert emitted(terminal, name, compartment) == 0
                        else:
                            assert emitted(terminal, name, compartment) > 0
                native_ash = bd.get_node(database=database, code=f"phb_{fraction}_terminal")
                assert [
                    {key: value for key, value in dict(row).items() if key != "output"}
                    for row in terminal.technosphere()
                ] == [
                    {key: value for key, value in dict(row).items() if key != "output"}
                    for row in native_ash.technosphere()
                ]
            # Native municipal support remains supplied once, with distinct
            # market freight and cement/hydration descendants rather than a
            # second incoming UPR or auxiliary-material input.
            assert all(
                "upr_" not in row.input["name"]
                for row in node.technosphere()
                if row.input.key[0] != frontier.database_name
            )
            owner = json.loads(node["direct_treatment_carbon_owners_json"])[0]
            fractions = {
                (r["flow_database"], r["flow_code"]): r["fraction"]
                for r in owner["principal_carbon_fractions"]
            }
            co2 = [
                row for row in node.biosphere() if row.input["name"].startswith("Carbon dioxide,")
            ]
            total = fsum(row["amount"] for row in co2)
            tracked = fsum(row["amount"] * fractions[row.input.key] for row in co2)
            assert tracked / total == pytest.approx(0.70405325131712 / 0.731932863627673)


@pytest.mark.parametrize("original", (True, False), ids=("original", "future"))
def test_upr_sanitary_chain_preserves_minerals_through_water_soil_and_sludge_ash(
    tmp_path, original
):
    """Solve all eleven receiving nodes with complete native supplies and the UPR elemental load."""
    with receiving_boundary(tmp_path, "sanitary", original) as (bd, frontier, services, database):
        assert len(services) == 33
        conversion = next(
            row
            for row in upr_recipe().conversion
            if row.operation == "others_upr_hand_lay_up_conversion"
        )
        for origin in ORIGINS:
            nodes = {
                role: bd.get_node(
                    database=frontier.database_name,
                    code=provider_service_code(f"upr_{origin}_{role}"),
                )
                for role in OWNERS
            }
            site = nodes["landfill"]
            load = _upr_landfill_load(
                conversion,
                origin,
                _LandfillSite(16.743579766537, 1250.84108527132, 1),
                carbonate_end_year=2863,
            )
            assert site.rp_exchange()["amount"] == 1
            assert site.rp_exchange()["properties"]["wet mass"]["amount"] == pytest.approx(
                load.wet_mass_kg
            )
            lca: Any = LCA({site: 1})
            lca.lci()
            for role, node in nodes.items():
                if role != "landfill":
                    native = bd.get_node(database=database, code=OWNERS[role][0])
                    assert node.rp_exchange()["amount"] == native.rp_exchange()["amount"] < 0
                assert lca.supply_array[lca.dicts.activity[node.id]] > 0
            for name, element in MINERALS.items():
                assert emitted(site, name, LONG) == pytest.approx(
                    load.phases[element].later_leachate_kg
                )
                for role, categories in (
                    ("leachate_collection", SURFACE),
                    ("wastewater_treatment", SURFACE),
                    ("sludge_landfarming", AGRICULTURAL),
                ):
                    assert emitted(nodes[role], name, categories) > 0
            for name in ("Silicon", "Calcium II"):
                assert emitted(nodes["sludge_incineration"], name, URBAN) > 0
                assert emitted(nodes["sludge_incineration"], name, SURFACE) > 0
            thermal = technology_terms(
                frontier, f"upr_{origin}_sludge_incineration", "actual_sludge_incineration"
            )
            bottom = thermal["bottom_ash"]
            assert bottom["water_kg"] / (
                bottom["treatment_mass_kg"] - bottom["water_kg"]
            ) == pytest.approx(0.197802 / 0.802198)
            for fraction in ("bottom", "residual"):
                terminal = nodes[fraction + "_ash_treatment"]
                for name in MINERALS:
                    assert emitted(terminal, name, SURFACE) > 0
                    assert emitted(terminal, name, LONG) > 0
            # The site keeps native placement/support uncertainty while actual
            # leachate and its organic/mineral quantities enter one collected
            # hydraulic volume. All four carbon-owner scales match solved LCI.
            native = bd.get_node(database=database, code="site")
            source_row = next(
                row for row in native.technosphere() if row.input["code"] == "site_equipment"
            )
            target = next(
                row for row in site.technosphere() if row.input["code"] == "site_equipment"
            )
            scale = load.wet_mass_kg / -native.rp_exchange()["amount"]
            assert target["amount"] == pytest.approx(source_row["amount"] * scale)
            assert target["loc"] == pytest.approx(source_row["loc"] + log(scale))
            assert target["scale"] == source_row["scale"]
            for owner in json.loads(site["direct_treatment_carbon_owners_json"]):
                node = bd.get_node(database=owner["owner_database"], code=owner["owner_code"])
                assert owner["owner_scale_per_provider_unit"] / node.rp_exchange()[
                    "amount"
                ] == pytest.approx(lca.supply_array[lca.dicts.activity[node.id]], rel=2e-7)
            # Non-fossil carbon includes the declared reference bridge;
            # only tracked principal belongs to the principal diagnostic.
            co2 = [
                row
                for row in site.biosphere()
                if row.input["name"].startswith("Carbon dioxide,")
                and "principal_carbon_origin_fractions" in row
            ]
            assert fsum(
                row["amount"] * fsum(row["principal_carbon_origin_fractions"]) for row in co2
            ) / fsum(row["amount"] for row in co2) == pytest.approx(
                0.70405325131712 / 0.731932863627673
            )


@pytest.mark.parametrize("original", (True, False), ids=("original", "future"))
def test_upr_uncontrolled_functions_and_shared_selection_keep_actual_origin(tmp_path, original):
    """Bind six uncontrolled functions through the shared route/carbon boundary."""
    with receiving_boundary(tmp_path, "unmanaged", original) as (bd, frontier, services, database):
        assert len(services) == 18
        conversion = next(
            row
            for row in upr_recipe().conversion
            if row.operation == "others_upr_hand_lay_up_conversion"
        )
        material = _upr_material(conversion, (1, 0, 0))
        authority = prepared_provider_authority()
        routes = ProviderRoutes(
            authority.routes,
            identity=authority.identity,
            interface_database=frontier.database_name,
            material_carbon=MaterialCarbon(
                "actual_upr_carbon",
                (("Others", "upr"),),
                (((0.70405325131712, 0, 0), (0, 0.70405325131712, 0), (0, 0, 0.70405325131712)),),
            ),
        )
        year = 2025 if original else 2050
        for origin in ORIGINS:
            for _, suffix, precipitation in DUMP_CLASSES:
                sid = f"upr_{origin}_open_dump_{suffix}"
                node = bd.get_node(database=frontier.database_name, code=provider_service_code(sid))
                assert node.rp_exchange()["amount"] == 1
                assert list(node.technosphere()) == []
                for name, _, per_kg in LAND_ROWS:
                    rows = [row for row in node.biosphere() if row.input["name"] == name]
                    assert len(rows) == 2
                    assert fsum(row["amount"] for row in rows) == pytest.approx(
                        per_kg * material.mass_kg * (1 if original else 1.7)
                    )
                    assert all(row["scale"] == 0.12 for row in rows)
                load = _upr_landfill_load(
                    conversion,
                    origin,
                    _LandfillSite(8, precipitation, 1 - 2.0604 / 7),
                    carbonate_end_year=services[sid]["construction"]["carbonate_end_year"],
                )
                if precipitation == 350:
                    for name, element in MINERALS.items():
                        assert emitted(node, name, ("soil", "industrial")) == pytest.approx(
                            0.9
                            * (
                                load.phases[element].short_leachate_kg
                                + load.phases[element].later_leachate_kg
                            )
                        )
                else:
                    for name, element in MINERALS.items():
                        assert emitted(node, name, ("water", "ground-")) == pytest.approx(
                            load.phases[element].short_leachate_kg
                        )
                        assert emitted(node, name, LONG) == pytest.approx(
                            load.phases[element].later_leachate_kg
                        )
                lca: Any = LCA({node: 1})
                lca.lci()
                assert lca.supply_array[lca.dicts.activity[node.id]] == pytest.approx(1)
            sid = f"upr_{origin}_open_burning"
            node = bd.get_node(database=frontier.database_name, code=provider_service_code(sid))
            assert list(node.technosphere()) == []
            assert emitted(node, "Heat, waste", RURAL) == pytest.approx(material.heat_potential_mj)
            assert emitted(node, "Cobalt II", RURAL) > 0
            assert emitted(node, "Silicon", RURAL) > 0
            # Only dispersed organic sizing/catalyst residues burn; bulk glass
            # and gelcoat silica remain in the source's inert-residue pool.
            assert emitted(node, "Calcium II", RURAL) == 0
            rows = authority.routes.loc[authority.routes.provider_service_id.eq(sid)]
            assert not rows.empty
            route = rows.iloc[0]
            selected = routes.resolve(
                provider_demand_type=route.provider_demand_type,
                material=route.material,
                constituent=route.constituent,
                product=route["product"],
                year=year,
                mapping_basis=route.mapping_basis,
                virgin_feedstock_route=route.virgin_feedstock_route,
                ultimate_carbon_origin=origin,
            )
            assert {row.provider_service_id for row in selected} == {sid}
            assert all(row.throughput_multiplier == 1 for row in selected)
            activation = pd.DataFrame(
                [
                    {
                        "year": year,
                        "region": "EUR",
                        "life_cycle_stage": "mismanaged_open_burning_treatment",
                        "material": "Others",
                        "constituent": "upr",
                        "product": "polymer",
                        "virgin_feedstock_route": origin,
                        "good": "all",
                        "material_cycle_source": "virgin_feedstock",
                        "ultimate_carbon_origin": origin,
                        "lca_fate": "open_burning",
                        "fate_origin": "remind_mfa",
                        "provider_database": frontier.database_name,
                        "provider_code": node.key[1],
                        "provider_anchor_year": year,
                        "amount": 1.0,
                        "mfa_carbon_amount_kg": 0.70405325131712,
                    }
                ]
            )
            diagnostic = direct_treatment_carbon_flows(bd, activation)
            assert set(diagnostic.mfa_carbon_origin) == {origin}
            assert set(diagnostic.origin_mapping_status) == {"actual_upr_principal_origin"}
            co2 = fsum(
                row["amount"]
                for row in node.biosphere()
                if row.input["name"].startswith("Carbon dioxide,")
            )
            actual = diagnostic.loc[
                diagnostic.carbon_species.eq("carbon dioxide"), "carbon_amount"
            ].sum()
            assert actual == pytest.approx(co2 * 12 / 44 * 0.70405325131712 / 0.731932863627673)
