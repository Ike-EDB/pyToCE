"""Test municipal treatment and unmanaged disposal of retired compound 22.

Synthetic Brightway inventories reproduce the selected source descriptors.
The added zinc flows distinguish retained filler from the original source
composition; no licensed inventories are copied.
"""

import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from math import exp, fsum, log
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from infiltration_fixtures import infiltration_source_data
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from pa_dump_receiving_fixtures import (
    DUMP_CLASSES,
    GROUND,
    LAND,
    LAND_ROWS,
    LATER_AIR,
    LATER_GROUND,
    RURAL,
    SOIL,
    dump_biosphere,
    dump_native_inventory,
)
from pa_municipal_fixtures import (
    FIXTURE,
    LONG,
    ORIGINS,
    SURFACE,
    URBAN,
    pa_municipal_biosphere,
    pa_municipal_native_inventory,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    prepared_provider_authority,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.route_authorities import (
    resolve_open_dump_routes,
)
from pytoce.run_lca.calculation.burdens import BURDEN_SPECS, MT_TO_KG, _origin_demands
from pytoce.run_lca.calculation.carbon import direct_treatment_carbon_flows
from pytoce.run_lca.calculation.foreground import (
    _activation_ledger,
    _optimization_activation_ledger,
)
from pytoce.run_lca.calculation.providers import MaterialCarbon, ProviderRoutes

# Independent admitted material handoff, per retained principal kg. The material
# owner's formula/atom tests remain separate from these connected receiving tests.
MASS = 1.8468991701714175
PRINCIPAL_C = 0.8969229843817300
FOSSIL_AUX_C = 19 * 0.03633106199969588
BIOGENIC_AUX_C = 19 * 0.0007595306139770401
TOTAL_C = PRINCIPAL_C + FOSSIL_AUX_C + BIOGENIC_AUX_C
ZINC = 19 * 0.004223906787873709
GASES = {"Carbon dioxide": 12 / 44, "Methane": 12 / 16, "Carbon monoxide": 12 / 28}


def _biosphere(family):
    """Reuse the complete receiving semantics and add only the actual Zn endpoints."""
    data = pa_municipal_biosphere() if family == "municipal" else dump_biosphere("biosphere")
    existing = {(row["name"], tuple(row["categories"])) for row in data.values()}
    compartments = (
        (URBAN, SURFACE, LONG)
        if family == "municipal"
        else (RURAL, LATER_AIR, GROUND, LATER_GROUND, SOIL)
    )
    for categories in compartments:
        if ("Zinc II", categories) not in existing:
            data["biosphere", f"sbr_zinc_{len(data)}"] = {
                "name": "Zinc II",
                "unit": "kilogram",
                "categories": categories,
                "type": "emission",
            }
    return data


def _municipal_inventory(catalogue, database, original):
    """Build five synthetic treatment inventories with original zinc coefficients.

    Future source chemistry is absent and its synthetic zinc emissions differ;
    the calculation must still derive coefficients from the original composition.
    The ash fixture already contains positive zinc properties and releases.
    """
    data = pa_municipal_native_inventory(catalogue, database, original=original)
    keys = {
        (row["name"], tuple(row["categories"])): key for key, row in _biosphere("municipal").items()
    }
    furnace = data[database, "phb_furnace"]
    production = furnace["exchanges"][0]
    if original:
        production["properties"]["zinc content"] = {"amount": 0.002}
        production["properties"]["oxygen content"]["amount"] -= 0.002
    for categories, coefficient in ((URBAN, 1e-7), (SURFACE, 3e-7)):
        for share in (0.3, 0.7):
            amount = -production["amount"] * coefficient * share * (1 if original else 1.9)
            furnace["exchanges"].append(
                {
                    "input": keys["Zinc II", categories],
                    "type": "biosphere",
                    "amount": amount,
                    "uncertainty type": 2,
                    "loc": log(amount) + 0.03,
                    "scale": 0.21,
                    "minimum": amount / 2,
                    "maximum": amount * 2,
                    "allocation": "invented zinc allocation",
                }
            )
    return data


def _dump_inventory(catalogue, database, original):
    """Use five complete invented land functions with independently dated bases."""
    data = dump_native_inventory(catalogue, database, biosphere_database="biosphere")
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    biosphere = _biosphere("dump")
    for i, (_, suffix, precipitation) in enumerate(DUMP_CLASSES):
        node = data[database, f"site_{i}"]
        source = services[f"sbr_compound_22_fossil_open_dump_{suffix}"]["construction"]["source"]
        node.update(name=source["name"], location=source["exact_location"])
        node["reference product"] = source["reference_product"]
        old_basis = -node["exchanges"][0]["amount"]
        basis = i + (2 if original else 7)
        node["exchanges"][0]["amount"] = -basis
        rows = []
        for row in node["exchanges"]:
            if row["type"] != "biosphere" or biosphere[row["input"]]["categories"] != LAND:
                rows.append(row)
                continue
            for share in (0.2, 0.8):
                factor = basis / old_basis * share * (1 if original else 1.7)
                rows.append(
                    {**row, "amount": row["amount"] * factor, "loc": row["loc"] + log(factor)}
                )
        node["exchanges"] = rows
        # The source climate is fixed; future comments are deliberately different.
        node["comment"] = (
            f"(MAT) of {8 if original else 31} C; (MAP) of {precipitation if original else 1} mm."
        )
    return data


@contextmanager
def _receiving_boundary(tmp_path, family, year):
    """Compile catalog declarations and preserve synthetic original and dated inventories."""
    authority = prepared_provider_authority()
    kinds = (
        {"sbr_incineration", "sbr_ash_market", "sbr_ash_treatment"}
        if family == "municipal"
        else {"sbr_dump"}
    )
    services = {
        row["provider_service_id"]: row
        for row in cast(list[dict[str, Any]], authority.catalogue["services"])
        if row["construction"]["kind"] in kinds
    }
    assert len(services) == 15
    provider = PreparedProviderAuthority(
        identity=f"sbr_{family}_fixture",
        interface_database="sbr_receiving_interface",
        catalogue={**authority.catalogue, "services": list(services.values()), "routes": []},
        routes=pd.DataFrame(),
        method_authorities=authority.method_authorities,
    )
    names = BrightwayNames(tmp_path / "brightway", "sbr_receiving", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(_biosphere(family), searchable=False)
        for database in ("original", "future"):
            source = _municipal_inventory if family == "municipal" else _dump_inventory
            data = source(authority.catalogue, database, database == "original")
            if family == "dump" and database == "original":
                data.update(infiltration_source_data(database))
            bd.Database(database).write(data, searchable=False)
        before = {
            node.key: (deepcopy(dict(node)), [dict(row) for row in node.exchanges()])
            for database in ("original", "future")
            for node in bd.Database(database)
        }
        database = "original" if year == 2025 else "future"
        if family == "dump":
            # Shared preparation admits the five conditional weights from the
            # complete PE waste market before public route activation.
            resolved, _ = resolve_open_dump_routes("original", authority.routes)
            authority = replace(authority, routes=resolved)
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
        yield bd, authority, frontier, services, database
        assert before == {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("original", "future")
            for node in bd.Database(database)
        }


def _activation(authority, frontier, year, demand_type):
    """Compare ordinary/candidate demand and exercise every actual direct carbon owner."""
    routes = ProviderRoutes(
        authority.routes,
        identity=authority.identity,
        interface_database=frontier.database_name,
        material_carbon=MaterialCarbon(
            "sbr_receiving_carbon",
            (("Others", "non_tyre_elastomers"),),
            (
                tuple(
                    cast(
                        tuple[float, float, float],
                        tuple(PRINCIPAL_C if i == j else 0 for j in range(3)),
                    )
                    for i in range(3)
                ),
            ),
        ),
    )
    spec = next(row for row in BURDEN_SPECS if row.demand_type == demand_type)
    demands = pd.DataFrame(
        _origin_demands(
            spec,
            scenario="sbr_receiving_fixture",
            year=year,
            region="EUR",
            material="Others",
            constituent="non_tyre_elastomers",
            product="polymer",
            parent_amount_mt=2.5 / MT_TO_KG,
            allocation=1,
            carbon_mt=2.5 * PRINCIPAL_C * np.array((0.2, 0.3, 0.5)) / MT_TO_KG,
        )
    )
    ordinary = _activation_ledger(demands, routes, provider_service_database=frontier.database_name)
    candidate = _optimization_activation_ledger(demands, routes)
    pd.testing.assert_series_equal(
        ordinary.groupby("provider_service_id").amount.sum(),
        candidate.groupby("provider_service_id").amount.sum(),
    )
    assert ordinary.provider_service_id.str.startswith("sbr_compound_22_").all()
    assert fsum(ordinary.amount) == pytest.approx(2.5)
    assert (ordinary.amount > 0).all()
    # Tires retain their independent native-proxy route family.
    tyre = authority.routes.loc[
        authority.routes.material.eq("Elastomers (tyres)")
        & authority.routes.provider_demand_type.eq(demand_type)
    ]
    assert not tyre.empty
    assert not tyre.provider_service_id.str.startswith("sbr_compound_22_").any()
    return ordinary.rename(
        columns={
            "demand_type": "life_cycle_stage",
            "provider_service_database": "provider_database",
            "provider_service_code": "provider_code",
        }
    ).assign(provider_anchor_year=year)


def _emitted(node, name, categories):
    """Read bound elementary quantities on their actual compiled reference."""
    return fsum(
        row["amount"]
        for row in node.biosphere()
        if row.input["name"] == name and tuple(row.input["categories"]) == categories
    )


def _principal_diagnostic(bd, activation, node, origin, carbon_fraction):
    """Check full gas, fixed auxiliary origins, and the direct principal reporting subset."""
    # Retain the real class/route identity while separately probing each
    # endpoint on one principal kg with the corresponding principal C demand.
    template = activation.loc[activation.provider_code.eq(node.key[1])].copy()
    assert len(template) == 1
    template["amount"] = 1.0
    template["mfa_carbon_amount_kg"] = PRINCIPAL_C
    reported = direct_treatment_carbon_flows(bd, template)
    assert set(reported.origin_mapping_status) == {"actual_sbr_principal_origin"}
    assert set(reported.mfa_carbon_origin) == {origin}
    for name, kg_c in GASES.items():
        rows = [r for r in node.biosphere() if r.input["name"].startswith(name + ",")]
        if not rows:
            continue  # Unmanaged decomposition emits CH4 and CO2.
        total = fsum(r["amount"] for r in rows)
        fossil = fsum(r["amount"] for r in rows if r.input["name"] == name + ", fossil")
        assert fossil == pytest.approx(
            total * (FOSSIL_AUX_C + (PRINCIPAL_C if origin == "fossil" else 0)) / TOTAL_C
        )
        principal = fsum(r["amount"] * fsum(r["principal_carbon_origin_fractions"]) for r in rows)
        assert principal == pytest.approx(total * carbon_fraction)
        assert reported.loc[
            reported.carbon_species.eq(name.lower()), "carbon_amount"
        ].sum() == pytest.approx(principal * kg_c)
        assert 0 < principal < total


@pytest.mark.parametrize("year", (2025, 2050), ids=("original", "future"))
def test_compound22_municipal_keeps_zinc_auxiliaries_and_complete_signed_ash(tmp_path, year):
    """Solve all origins with original Zn calibration and dated complete native supplies."""
    with _receiving_boundary(tmp_path, "municipal", year) as (
        bd,
        authority,
        frontier,
        services,
        database,
    ):
        activation = _activation(authority, frontier, year, "incineration_treatment")
        for origin in ORIGINS:
            sid = f"sbr_compound_22_{origin}_incineration"
            node = bd.get_node(database=frontier.database_name, code=provider_service_code(sid))
            load = json.loads(
                cast(
                    str,
                    next(
                        row["project_terms_json"]
                        for row in frontier.technology_rows
                        if row["provider_service_id"] == sid
                        and row["component_id"] == "actual_sbr_municipal_incineration"
                    ),
                )
            )
            assert node.rp_exchange()["amount"] == 1
            assert load["material"]["wet_mass_kg"] == pytest.approx(MASS)
            assert load["material"]["water_kg"] == 0
            assert load["material"]["elements_kg"]["C"] == pytest.approx(TOTAL_C)
            assert load["material"]["elements_kg"]["Zn"] == pytest.approx(ZINC)
            assert len(load["material"]["carbon"]) == 3
            _principal_diagnostic(bd, activation, node, origin, PRINCIPAL_C / TOTAL_C)
            for categories, coefficient in ((URBAN, 1e-7), (SURFACE, 3e-7)):
                expected = coefficient * ZINC / (FIXTURE["dry_mass"] * 0.002)
                assert _emitted(node, "Zinc II", categories) == pytest.approx(expected)
                rows = [
                    r
                    for r in node.biosphere()
                    if r.input["name"] == "Zinc II" and tuple(r.input["categories"]) == categories
                ]
                assert len(rows) == 2
                assert sorted(r["amount"] / expected for r in rows) == pytest.approx([0.3, 0.7])
                for row in rows:
                    assert row["loc"] == pytest.approx(log(row["amount"]) + 0.03)
                    assert row["scale"] == 0.21
                    assert row["allocation"] == "invented zinc allocation"
            purchases = {row.input.key: row for row in node.technosphere()}
            assert (database, "phb_haul") not in purchases
            assert (database, "mc_inorganic") not in purchases
            for code, share in (
                ("phb_water_a", 0.4 if year == 2025 else 0.8),
                ("phb_water_b", 0.6 if year == 2025 else 0.2),
            ):
                row = purchases[database, code]
                assert row["amount"] == pytest.approx(load["duties"]["water"] * share)
                assert row["loc"] == pytest.approx(log(row["amount"]) + 0.03)
                assert row["allocation"] == "invented source allocation"
            assert purchases[database, "mc_pam"]["amount"] == pytest.approx(load["duties"]["PAM"])
            lca = LCA({node: 1})
            lca.lci()
            for fraction in ("bottom", "residual"):
                ash = load[fraction + "_ash"]
                market = bd.get_node(
                    database=frontier.database_name,
                    code=provider_service_code(f"sbr_compound_22_{origin}_{fraction}_ash_market"),
                )
                terminal = bd.get_node(
                    database=frontier.database_name,
                    code=provider_service_code(
                        f"sbr_compound_22_{origin}_{fraction}_ash_treatment"
                    ),
                )
                basis = 3 if year == 2025 else 7
                assert terminal.rp_exchange()["amount"] == -basis
                level = cast(NDArray[np.float64], lca.supply_array)[lca.dicts.activity[terminal.id]]
                assert level * basis == pytest.approx(ash["treatment_mass_kg"], rel=2e-7)
                assert purchases[market.key]["amount"] == pytest.approx(-ash["treatment_mass_kg"])
                actual_zn = ash["waste_elements_kg"]["Zn"] + ash["auxiliary_elements_kg"].get(
                    "Zn", 0
                )
                for categories in (SURFACE, LONG):
                    expected = (
                        1e-6
                        / (FIXTURE["ash_dry_mass"] * FIXTURE["ash_dry_elements"]["zinc"])
                        * actual_zn
                    )
                    assert _emitted(terminal, "Zinc II", categories) * level == pytest.approx(
                        expected, rel=2e-7
                    )
                assert actual_zn > 0
                assert ash["waste_elements_kg"]["C"] > 0
                if fraction == "residual":
                    assert ash["auxiliary_elements_kg"]["C"] > 0
                native = bd.get_node(database=database, code=f"phb_{fraction}_market")
                for row in native.technosphere():
                    if row.input["code"].endswith("terminal"):
                        continue
                    bound = next(r for r in market.technosphere() if r.input.key == row.input.key)
                    for field in ("amount", "loc", "scale", "minimum", "maximum", "allocation"):
                        assert bound[field] == row[field]
            ash_mass = (
                load["bottom_ash"]["treatment_mass_kg"] + load["residual_ash"]["treatment_mass_kg"]
            )
            for code, coefficient in (("phb_haul", 0.02), ("pa_rail", 0.03), ("pa_barge", 0.027)):
                supplier = bd.get_node(database=database, code=code)
                assert cast(NDArray[np.float64], lca.supply_array)[
                    lca.dicts.activity[supplier.id]
                ] == pytest.approx(coefficient * ash_mass, rel=2e-7)
            for name, _ in GASES.items():
                for row in node.biosphere():
                    if row.input["name"].startswith(name + ","):
                        expected = fsum(
                            r["amount"] for r in node.biosphere() if r.input.key == row.input.key
                        )
                        assert lca.inventory[
                            lca.dicts.biosphere[row.input.id], lca.dicts.activity[node.id]
                        ] == pytest.approx(expected, rel=2e-7)


@pytest.mark.parametrize("year", (2025, 2050), ids=("original", "future"))
def test_compound22_five_dumps_keep_original_climate_and_actual_zinc_destinations(tmp_path, year):
    """Bind dry compound to dated land functions and source-qualified Zn air/soil/water."""
    with _receiving_boundary(tmp_path, "dump", year) as (
        bd,
        authority,
        frontier,
        services,
        database,
    ):
        activation = _activation(authority, frontier, year, "mismanaged_open_dump_treatment")
        for origin in ORIGINS:
            for _, suffix, precipitation in DUMP_CLASSES:
                sid = f"sbr_compound_22_{origin}_open_dump_{suffix}"
                node = bd.get_node(database=frontier.database_name, code=provider_service_code(sid))
                terms = json.loads(
                    cast(
                        str,
                        next(
                            row["project_terms_json"]
                            for row in frontier.technology_rows
                            if row["provider_service_id"] == sid
                            and row["component_id"] == "actual_cured_sbr_open_dump"
                        ),
                    )
                )
                assert node.rp_exchange()["amount"] == 1
                assert node.rp_exchange()["properties"]["wet mass"]["amount"] == pytest.approx(MASS)
                assert list(node.technosphere()) == []
                assert not any(row.input["name"] == "Water" for row in node.biosphere())
                for name, _, coefficient in LAND_ROWS:
                    rows = [r for r in node.biosphere() if r.input["name"] == name]
                    expected = MASS * coefficient * (1 if year == 2025 else 1.7)
                    assert len(rows) == 2
                    assert fsum(r["amount"] for r in rows) == pytest.approx(expected)
                    for r in rows:
                        assert r["loc"] == pytest.approx(log(r["amount"]) + 0.03)
                        assert r["scale"] == 0.12
                # Source Clim response and row 48 fractions independently map
                # the admitted Zn input; the accepted later-release gate is q=1.
                rain = 60 * (1 - exp(-29.372386439548432 * precipitation / 1000))
                temperature = 60 * (1 - exp(-0.3 * (8 + 15)))
                alpha = log(1 - rain / 220) * log(1 - temperature / 220) / log(1 - 60 / 220) ** 2
                mobilized = ZINC * 0.01 * alpha * 0.04738342800955586
                gas_zn = mobilized * 0.00022217285047767166
                short = mobilized - gas_zn
                later = ZINC - gas_zn - short
                if precipitation == 350:
                    assert _emitted(node, "Zinc II", RURAL) == pytest.approx(gas_zn + 0.1 * short)
                    assert _emitted(node, "Zinc II", LATER_AIR) == pytest.approx(0.1 * later)
                    assert _emitted(node, "Zinc II", SOIL) == pytest.approx(0.9 * (short + later))
                    assert (
                        _emitted(node, "Zinc II", GROUND)
                        == _emitted(node, "Zinc II", LATER_GROUND)
                        == 0
                    )
                else:
                    assert _emitted(node, "Zinc II", RURAL) == pytest.approx(gas_zn)
                    assert _emitted(node, "Zinc II", GROUND) == pytest.approx(short)
                    assert _emitted(node, "Zinc II", LATER_GROUND) == pytest.approx(later)
                    assert (
                        _emitted(node, "Zinc II", SOIL) == _emitted(node, "Zinc II", LATER_AIR) == 0
                    )
                assert fsum(
                    _emitted(node, "Zinc II", c)
                    for c in (RURAL, LATER_AIR, SOIL, GROUND, LATER_GROUND)
                ) == pytest.approx(ZINC)
                _principal_diagnostic(bd, activation, node, origin, PRINCIPAL_C / TOTAL_C)
                carbon = fsum(
                    _emitted(node, species + ", " + label, RURAL) * kg_c
                    for species, kg_c in GASES.items()
                    for label in ("fossil", "non-fossil")
                )
                assert carbon == pytest.approx(TOTAL_C * 0.01 * alpha * 0.970873786407767)
                assert terms["actual_matrix_per_principal_kg"]["elements_kg"][
                    "Zn"
                ] == pytest.approx(ZINC)
                lca = LCA({node: 1})
                lca.lci()
                assert cast(NDArray[np.float64], lca.supply_array)[
                    lca.dicts.activity[node.id]
                ] == pytest.approx(1)
                assert np.count_nonzero(cast(NDArray[np.float64], lca.supply_array)) == 1
