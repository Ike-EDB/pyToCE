"""Compare SBR compound 22 burning quantities in ordinary and candidate calculations.

Synthetic original and future inventories are preserved while the compiled
burning service for each carbon origin is solved with Brightway LCI.
The checks retain direct air/soil flows and apply each treatment route once.
"""

import json
from copy import deepcopy
from math import fsum, log
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray

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
from pytoce.prepare_backgrounds.preparation.providers.sbr_material import (
    _sbr_compound_material,
)
from pytoce.run_lca.calculation.burdens import BURDEN_SPECS, MT_TO_KG, _origin_demands
from pytoce.run_lca.calculation.carbon import direct_treatment_carbon_flows
from pytoce.run_lca.calculation.foreground import (
    _activation_ledger,
    _optimization_activation_ledger,
)
from pytoce.run_lca.calculation.providers import MaterialCarbon, ProviderRoutes
from tests.unit.prepare_backgrounds.pa_open_burning_fixtures import (
    _AIR,
    _INDICATOR,
    _SOIL,
    open_burning_biosphere,
    open_burning_native_inventory,
)

_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
_GASES = (("Carbon dioxide", 12 / 44), ("Carbon monoxide", 12 / 28), ("Methane", 12 / 16))


def _assert_compiled_atom_balance(terms):
    """Reconcile SBR output element masses with their overlapping indicators."""
    # Nominal element masses in each molecule. SBR has no selected halogen
    # pool; the accepted zero-Cl domain consequently has no PCB/dioxin output.
    molecules: dict[str, dict[str, float]] = {
        "Benzene": {"C": 72, "H": 6},
        "Phenol": {"C": 72, "H": 6, "O": 16},
        "Styrene": {"C": 96, "H": 8},
        "Toluene": {"C": 84, "H": 8},
        "Formaldehyde": {"C": 12, "H": 2, "O": 16},
        "Acetaldehyde": {"C": 24, "H": 4, "O": 16},
        "Ethyl benzene": {"C": 96, "H": 10},
        "Carbon dioxide": {"C": 12, "O": 32},
        "Carbon monoxide": {"C": 12, "O": 16},
        "Methane": {"C": 12, "H": 4},
        "Nitrogen oxides": {"N": 14, "O": 32},
        "Dinitrogen monoxide": {"N": 28, "O": 16},
        "Sulfur dioxide": {"S": 32, "O": 32},
        "Water": {"H": 2, "O": 16},
        "Nitrogen": {"N": 28},
    }
    # Check EPA1997 Table3-7/SIM paired-run masses against nominal PAH carbon fractions.
    pah = (
        (0.9578 + 0.5780, 144 / 154),
        (13.6424 + 8.9577, 144 / 152),
        (2.3724 + 1.2910, 168 / 178),
        (3.1364 + 1.3425, 216 / 228),
        (3.1275 + 1.1167, 240 / 252),
        (3.7585 + 1.5016, 240 / 252),
        (2.8148 + 1.1340, 264 / 276),
        (1.6424 + 0.4085, 240 / 252),
        (3.5588 + 1.5136, 216 / 228),
        (0.4861 + 0.1595, 264 / 278),
        (5.1917 + 2.9436, 192 / 202),
        (4.7756 + 2.5360, 156 / 166),
        (2.7997 + 0.9547, 264 / 276),
        (18.9598 + 16.1032, 120 / 128),
        (8.9946 + 5.6546, 168 / 178),
        (6.1419 + 3.5157, 192 / 202),
    )
    pah_c = fsum(amount * carbon for amount, carbon in pah) / fsum(amount for amount, _ in pah)
    molecules["PAH, polycyclic aromatic hydrocarbons"] = {"C": pah_c, "H": 1 - pah_c}
    incoming = dict(terms["input_elements_kg"])
    incoming["O"] = incoming.get("O", 0) + terms["atmospheric_oxygen_kg"]
    incoming["N"] = incoming.get("N", 0) + terms["atmospheric_nitrogen_kg"]
    air, oxides, soil = (
        terms[field]
        for field in ("air_compounds_kg", "air_oxide_elements_kg", "residue_elements_kg")
    )
    assert terms["soil_compounds_kg"] == {}
    assert all(amount >= 0 for pool in (incoming, air, oxides, soil) for amount in pool.values())
    for element, amount in incoming.items():
        emitted = (
            fsum(
                mass * molecules[name].get(element, 0) / fsum(molecules[name].values())
                for name, mass in air.items()
            )
            + oxides.get(element, 0)
            + soil.get(element, 0)
        )
        assert emitted == pytest.approx(amount, rel=4e-15, abs=8e-16), element
    incoming_mass = (
        terms["wet_mass_kg_per_principal_kg"]
        + terms["atmospheric_oxygen_kg"]
        + terms["atmospheric_nitrogen_kg"]
    )
    assert incoming_mass - fsum((*air.values(), *oxides.values(), *soil.values())) == pytest.approx(
        terms["source_rounding_kg"], abs=5e-15
    )


@pytest.mark.parametrize("year", (2025, 2050), ids=("original", "future"))
def test_sbr_actual_burning_frontier_and_shared_principal_routes(tmp_path, year):
    """Run LCI for each carbon origin and compare ordinary and candidate burning quantities."""
    authority = prepared_provider_authority()
    services = {
        row["provider_service_id"]: row
        for row in cast(list[dict[str, Any]], authority.catalogue["services"])
        if row["construction"]["kind"] == "sbr_open_burning"
    }
    expected_ids = {f"sbr_compound_22_{origin}_open_burning" for origin in _ORIGINS}
    assert set(services) == expected_ids
    provider = PreparedProviderAuthority(
        identity="actual_sbr_burning_fixture",
        interface_database="sbr_interface",
        catalogue={**authority.catalogue, "services": list(services.values()), "routes": []},
        routes=pd.DataFrame(),
        method_authorities=authority.method_authorities,
    )
    biosphere = open_burning_biosphere("biosphere")
    for compartment in (_AIR, _SOIL):
        biosphere["biosphere", f"zinc_{compartment[0]}"] = {
            "name": "Zinc II",
            "unit": "kilogram",
            "categories": compartment,
            "type": "emission",
        }
    names = BrightwayNames(tmp_path / "brightway", "sbr_burning", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere, searchable=False)
        for database in ("original", "future"):
            native = open_burning_native_inventory(
                database, original=database == "original", biosphere_database="biosphere"
            )
            rows = native[database, "burning"]["exchanges"]
            rows[0]["amount"] = -3 if database == "original" else -7
            for row in rows[1:]:
                row.update(
                    {"uncertainty type": 2, "loc": log(row["amount"]), "scale": 0.12}
                    if database == "original"
                    else {"uncertainty type": 0, "loc": row["amount"]}
                )
            bd.Database(database).write(native, searchable=False)
        before = {
            node.key: (deepcopy(dict(node)), [dict(row) for row in node.exchanges()])
            for database in ("original", "future")
            for node in bd.Database(database)
        }
        dated = "original" if year == 2025 else "future"
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=dated,
            technology_database=dated,
            provider=provider,
            material_recipe=physical_material_recipe(),
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )
        traces = {}
        for origin in _ORIGINS:
            sid = f"sbr_compound_22_{origin}_open_burning"
            material = _sbr_compound_material(origin, principal_kg=1)
            node = bd.get_node(database=frontier.database_name, code=provider_service_code(sid))
            assert node.rp_exchange()["amount"] == 1
            assert list(node.technosphere()) == []
            trace = next(
                row
                for row in frontier.semantic_rows
                if row["provider_service_id"] == sid
                and row["ledger_entry_type"] == "service_construction"
            )
            terms = json.loads(cast(str, trace["project_terms_json"]))
            traces[sid] = terms
            assert terms["original_source_key"] == ["original", "burning"]
            assert terms["principal_carbon_origin"] == origin
            assert terms["input_elements_kg"] == material.elements_kg
            assert terms["principal_carbon_kg"] == material.principal_carbon_kg
            assert terms["auxiliary_carbon_kg"] == material.auxiliary_carbon_kg
            assert terms["wet_mass_kg_per_principal_kg"] == material.mass_kg
            assert terms["incoming_water_kg"] == 0
            _assert_compiled_atom_balance(terms)
            rows = {
                (row.input["name"], row.input["unit"], tuple(row.input["categories"])): row
                for row in node.biosphere()
            }
            assert len(rows) == len(list(node.biosphere()))
            assert all(
                row["uncertainty type"] == 0 and row["loc"] == row["amount"]
                for row in rows.values()
            )
            assert rows["Heat, waste", "megajoule", _AIR]["amount"] == fsum(
                (material.carbon_heat_mj, material.sulfur_heat_mj)
            )
            assert (
                rows["Non-hazardous waste disposed", "kilogram", _INDICATOR]["amount"]
                == material.mass_kg
            )
            assert (
                rows["Water", "cubic meter", _AIR]["amount"]
                == terms["air_compounds_kg"]["Water"] / 1000
            )
            zinc = material.elements_kg["Zn"]
            assert rows["Zinc II", "kilogram", _SOIL]["amount"] == pytest.approx(
                zinc * 992.5600513870927 / 1000
            )
            assert rows["Zinc II", "kilogram", _AIR]["amount"] == pytest.approx(
                zinc * (1 - 992.5600513870927 / 1000)
            )
            assert all(
                rows["Zinc II", "kilogram", compartment]["amount"] > 0
                for compartment in (_AIR, _SOIL)
            )
            principal = material.principal_carbon_kg[origin]
            total = material.elements_kg["C"]
            fossil = material.auxiliary_carbon_kg.get("fossil", 0) + (
                principal if origin == "fossil" else 0
            )
            for name, _ in _GASES:
                gas = terms["air_compounds_kg"][name]
                assert rows[f"{name}, fossil", "kilogram", _AIR]["amount"] == pytest.approx(
                    gas * fossil / total
                )
                assert rows[f"{name}, non-fossil", "kilogram", _AIR]["amount"] == pytest.approx(
                    gas * (1 - fossil / total), abs=1e-15
                )
            lca = LCA({node: 1})
            lca.lci()
            assert cast(NDArray[np.float64], lca.supply_array)[
                lca.dicts.activity[node.id]
            ] == pytest.approx(1)
            for row in rows.values():
                assert lca.inventory[
                    lca.dicts.biosphere[row.input.id], lca.dicts.activity[node.id]
                ] == pytest.approx(row["amount"], rel=2e-7, abs=0)

        # The SBR material tests establish the principal carbon density.
        # This fixed density isolates route selection from manufacturing supplies.
        # Auxiliary carbon remains outside the MFA principal carbon input.
        density = 0.8969229843817300
        routes = ProviderRoutes(
            authority.routes,
            identity=authority.identity,
            interface_database=frontier.database_name,
            material_carbon=MaterialCarbon(
                "sbr_route_carbon",
                (("Others", "non_tyre_elastomers"),),
                (
                    cast(
                        tuple[tuple[float, float, float], ...],
                        tuple(tuple(density if i == j else 0 for j in range(3)) for i in range(3)),
                    ),
                ),
            ),
        )
        spec = next(
            row for row in BURDEN_SPECS if row.demand_type == "mismanaged_open_burning_treatment"
        )
        rows = authority.routes.loc[
            authority.routes.material.eq("Others")
            & authority.routes.constituent.eq("non_tyre_elastomers")
            & authority.routes.provider_demand_type.eq(spec.demand_type)
        ]
        assert len(rows) == 3
        assert set(rows.provider_service_id) == expected_ids
        assert (rows.share == 1).all() and (rows.throughput_multiplier == 1).all()
        demands = _origin_demands(
            spec,
            scenario="sbr_burning_fixture",
            year=year,
            region="EUR",
            material="Others",
            constituent="non_tyre_elastomers",
            product="polymer",
            parent_amount_mt=2.5 / MT_TO_KG,
            allocation=1,
            carbon_mt=2.5 * density * np.array((0.2, 0.3, 0.5)) / MT_TO_KG,
        )
        demand_frame = pd.DataFrame(demands)
        ordinary = _activation_ledger(
            demand_frame, routes, provider_service_database=frontier.database_name
        )
        candidate = _optimization_activation_ledger(demand_frame, routes)
        pd.testing.assert_series_equal(
            ordinary.groupby("provider_service_id").amount.sum(),
            candidate.groupby("provider_service_id").amount.sum(),
        )
        assert set(ordinary.provider_service_id) == expected_ids
        assert fsum(ordinary.amount) == pytest.approx(2.5)
        activation = ordinary.rename(
            columns={
                "demand_type": "life_cycle_stage",
                "provider_service_database": "provider_database",
                "provider_service_code": "provider_code",
            }
        )
        activation["provider_anchor_year"] = year
        diagnostic = direct_treatment_carbon_flows(bd, activation)
        assert set(diagnostic.origin_mapping_status) == {"actual_sbr_principal_origin"}
        for row in ordinary.itertuples():
            terms = traces[row.provider_service_id]
            selected = diagnostic.loc[diagnostic.provider_code.eq(row.provider_service_code)]
            assert set(selected.mfa_carbon_origin) == {row.ultimate_carbon_origin}
            fraction = fsum(terms["principal_carbon_kg"].values()) / terms["input_elements_kg"]["C"]
            for name, carbon_fraction in _GASES:
                actual = selected.loc[
                    selected.carbon_species.eq(name.lower()), "carbon_amount"
                ].sum()
                full = row.amount * terms["air_compounds_kg"][name] * carbon_fraction
                assert actual == pytest.approx(full * fraction, rel=3e-14)
                assert actual < full
        assert before == {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("original", "future")
            for node in bd.Database(database)
        }
