"""Connect retired compound 22 to the complete landfill treatment chain.

Synthetic original and dated inventories exercise production and uncertainty.
Separate tests verify phase chemistry; this test checks quantities from one
principal polymer demand through compound treatment, LCI and carbon reporting.
"""

import json
from copy import deepcopy
from math import fsum
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from sbr_landfill_fixtures import OWNERS, biosphere_inventory, native_inventory
from scipy.sparse import csr_matrix

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

_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
_CARBON_ROLES = ("landfill", "wastewater_treatment", "sludge_incineration", "sludge_landfarming")
_CARBON_FRACTIONS = {"Carbon dioxide": 12 / 44, "Methane": 12 / 16, "Carbon monoxide": 12 / 28}


@pytest.mark.parametrize("year", (2025, 2050), ids=("original", "future"))
def test_retired_compound_uses_complete_landfill_and_principal_carbon_owners(tmp_path, year):
    """Connect one principal polymer demand to eleven treatment processes for each origin."""
    authority = prepared_provider_authority()
    expected_roots = {f"sbr_compound_22_{origin}_landfill" for origin in _ORIGINS}
    child_ids = {f"sbr_filled_{origin}_{role}" for origin in _ORIGINS for role in OWNERS}
    services = {
        row["provider_service_id"]: row
        for row in cast(list[dict[str, Any]], authority.catalogue["services"])
        if row["provider_service_id"] in expected_roots | child_ids
    }
    assert set(services) == expected_roots | child_ids
    # Retain formulation and source references while compiling the 36 receiving services.
    source_only_ids = {
        "others_esbr_compounding_conversion",
        "others_esbr_compounding_conversion__biogenic",
        "others_esbr_compounding_conversion__atmospheric_dac",
        "others_esbr_filter_bag_delivery",
        "others_esbr_filter_bag_fabrication",
        "sbr_captured_mixer_dust_incineration",
        "sbr_captured_mixer_dust_ash_treatment",
    }
    source_only = [
        {**row, "foreground_service": False, "frontier_service": False}
        for row in cast(list[dict[str, Any]], authority.catalogue["services"])
        if row["provider_service_id"] in source_only_ids
    ]
    provider = PreparedProviderAuthority(
        identity="sbr_retired_compound_fixture",
        interface_database="sbr_landfill_interface",
        catalogue={
            **authority.catalogue,
            "services": [*services.values(), *source_only],
            "routes": [],
        },
        routes=pd.DataFrame(),
        method_authorities=authority.method_authorities,
    )
    names = BrightwayNames(tmp_path / "brightway", "sbr_retired", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere_inventory(), searchable=False)
        for database in ("original", "future"):
            bd.Database(database).write(
                native_inventory(database, future=database == "future"), searchable=False
            )
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

        # The admitted principal-carbon density selects three MFA-origin demands; fixed auxiliaries
        # retain their supplied quantities.
        density = 0.8969229843817300
        routes = ProviderRoutes(
            authority.routes,
            identity=authority.identity,
            interface_database=frontier.database_name,
            material_carbon=MaterialCarbon(
                "retired_sbr_carbon",
                (("Others", "non_tyre_elastomers"),),
                (
                    tuple(
                        cast(
                            tuple[float, float, float],
                            tuple(density if i == j else 0 for j in range(3)),
                        )
                        for i in range(3)
                    ),
                ),
            ),
        )
        spec = next(row for row in BURDEN_SPECS if row.demand_type == "landfill_treatment")
        selected_routes = authority.routes.loc[
            authority.routes.material.eq("Others")
            & authority.routes.constituent.eq("non_tyre_elastomers")
            & authority.routes.provider_demand_type.eq(spec.demand_type)
        ]
        assert set(selected_routes.provider_service_id) == expected_roots
        assert len(selected_routes) == 3
        assert (selected_routes.share == 1).all()
        assert (selected_routes.throughput_multiplier == 1).all()
        demands = pd.DataFrame(
            _origin_demands(
                spec,
                scenario="sbr_landfill_fixture",
                year=year,
                region="EUR",
                material="Others",
                constituent="non_tyre_elastomers",
                product="polymer",
                parent_amount_mt=2.5 / MT_TO_KG,
                allocation=1,
                carbon_mt=2.5 * density * np.array((0.2, 0.3, 0.5)) / MT_TO_KG,
            )
        )
        ordinary = _activation_ledger(
            demands, routes, provider_service_database=frontier.database_name
        )
        candidate = _optimization_activation_ledger(demands, routes)
        pd.testing.assert_series_equal(
            ordinary.groupby("provider_service_id").amount.sum(),
            candidate.groupby("provider_service_id").amount.sum(),
        )
        assert set(ordinary.provider_service_id) == expected_roots
        assert (ordinary.amount > 0).all() and fsum(ordinary.amount) == pytest.approx(2.5)
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

        for origin_index, origin in enumerate(_ORIGINS):
            sid = f"sbr_compound_22_{origin}_landfill"
            root = bd.get_node(database=frontier.database_name, code=provider_service_code(sid))
            nodes = {
                role: bd.get_node(
                    database=frontier.database_name,
                    code=provider_service_code(f"sbr_filled_{origin}_{role}"),
                )
                for role in OWNERS
            }
            mass = _sbr_compound_material(origin, principal_kg=1).mass_kg
            assert root.rp_exchange()["amount"] == 1
            assert [(row.input.key, row["amount"]) for row in root.technosphere()] == [
                (nodes["landfill"].key, -mass)
            ]
            assert list(root.biosphere()) == []
            # The only new exchange is the complete waste function. No 136 km
            # manufacturing-scrap haul or extra factor of 19 belongs at this end-of-life gate.
            lca = LCA({root: 1})
            lca.lci()
            levels = {
                role: cast(NDArray[np.float64], lca.supply_array)[lca.dicts.activity[node.id]]
                for role, node in nodes.items()
            }
            assert cast(NDArray[np.float64], lca.supply_array)[
                lca.dicts.activity[root.id]
            ] == pytest.approx(1)
            assert levels["landfill"] == pytest.approx(mass / (4 if year == 2025 else 9))
            for role, node in nodes.items():
                native = bd.get_node(database=dated, code=OWNERS[role][0])
                assert node.rp_exchange()["amount"] == native.rp_exchange()["amount"] < 0
                assert node["location"] == OWNERS[role][3]
                assert levels[role] > 0
            complete_inventory = cast(csr_matrix, lca.inventory).copy()
            lca.lci(demand={nodes["landfill"].id: -mass})
            np.testing.assert_allclose(
                cast(csr_matrix, lca.inventory).toarray(),
                complete_inventory.toarray(),
                rtol=2e-7,
                atol=1e-14,
            )

            site = nodes["landfill"]
            native_site = bd.get_node(database=dated, code="site")
            # All independent site supports remain on their native reference,
            # including a signed retirement and their complete upstream carbon.
            for code in (
                "site_equipment",
                "site_admin_a",
                "site_admin_b",
                "site_heat",
                "site_retirement",
            ):
                source = next(
                    row for row in native_site.technosphere() if row.input["code"] == code
                )
                target = next(
                    row for row in site.technosphere() if row.input.key == source.input.key
                )
                for field in (
                    "amount",
                    "uncertainty type",
                    "loc",
                    "scale",
                    "negative",
                    "minimum",
                    "maximum",
                    "allocation",
                    "comment",
                ):
                    assert target[field] == source[field]
                supplier = target.input
                expected_level = (
                    target["amount"] * levels["landfill"] / supplier.rp_exchange()["amount"]
                )
                assert cast(NDArray[np.float64], lca.supply_array)[
                    lca.dicts.activity[supplier.id]
                ] == pytest.approx(expected_level, rel=2e-7)
                flow = next(iter(supplier.biosphere())).input
                # activity() stores the fixture's .013 kg/unit control on its
                # nonunit production reference, including negative retirement.
                assert lca.inventory[
                    lca.dicts.biosphere[flow.id], lca.dicts.activity[supplier.id]
                ] == pytest.approx(
                    0.013 * abs(supplier.rp_exchange()["amount"]) * expected_level, rel=2e-7
                )

            # Only four direct receiving owners contribute to principal gas
            # diagnostics. Their scaling is independently checked against the
            # solved graph, including every negative/nonunit child reference.
            owners = json.loads(root["direct_treatment_carbon_owners_json"])
            assert {owner["owner_code"] for owner in owners} == {
                nodes[role]["code"] for role in _CARBON_ROLES
            }
            assert len(owners) == 4
            selected = diagnostic.loc[diagnostic.provider_code.eq(root["code"])]
            assert set(selected.mfa_carbon_origin) == {origin}
            assert set(selected.owner_code) == {nodes[role]["code"] for role in _CARBON_ROLES}
            demand = ordinary.loc[ordinary.provider_service_id.eq(sid), "amount"].sum()
            for role in _CARBON_ROLES:
                node = nodes[role]
                owner = next(row for row in owners if row["owner_code"] == node["code"])
                assert owner["owner_scale_per_provider_unit"] / node.rp_exchange()[
                    "amount"
                ] == pytest.approx(levels[role], rel=2e-7)
                principal_carbon = []
                full_carbon = []
                for name, carbon_fraction in _CARBON_FRACTIONS.items():
                    gas_rows = [
                        row for row in node.biosphere() if row.input["name"].startswith(name + ",")
                    ]
                    actual = fsum(
                        row["amount"] * row["principal_carbon_origin_fractions"][origin_index]
                        for row in gas_rows
                    )
                    full = fsum(row["amount"] for row in gas_rows)
                    expected = demand * levels[role] * actual * carbon_fraction
                    reported = selected.loc[
                        selected.owner_code.eq(node["code"])
                        & selected.carbon_species.eq(name.lower()),
                        "carbon_amount",
                    ].sum()
                    assert reported == pytest.approx(expected, rel=2e-7, abs=1e-16)
                    principal_carbon.append(actual * carbon_fraction)
                    full_carbon.append(full * carbon_fraction)
                    for row in gas_rows:
                        # Repeated flow rows are additive. Principal fractions select reporting
                        # attribution.
                        emitted = fsum(
                            r["amount"] for r in gas_rows if r.input.key == row.input.key
                        )
                        assert lca.inventory[
                            lca.dicts.biosphere[row.input.id], lca.dicts.activity[node.id]
                        ] == pytest.approx(levels[role] * emitted, rel=2e-7, abs=1e-16)
                assert 0 < fsum(principal_carbon) < fsum(full_carbon)
        assert before == {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("original", "future")
            for node in bd.Database(database)
        }
