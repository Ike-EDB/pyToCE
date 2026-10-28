"""Exercise the real TPS treatment frontier and its complete native treatment supplies.

The graph uses unchanged packaged service declarations and actual formulations,
with small invented original/future inventories. These tests protect activation,
signed graph normalization, source provenance and shared auxiliary supply; the
individual load equations remain covered at their producing owners. Relative
3e-7 accommodates bw_processing float32 exchanges in solved inventories.
"""

import json
from math import fsum

import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from tps_frontier_fixtures import (
    ACTIVE_SERVICES,
    ORIGINS,
    TPS_SUFFIXES,
    biosphere_inventory,
    flow_key,
    frontier_catalogue,
    native_inventory,
)
from tps_site_fixtures import ORIGINAL_LEACHATE, SITE_PRODUCTION

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PROVIDER_INTERFACE_DATABASE,
    PreparedProviderAuthority,
    _method_authority_rows,
    _route_table,
    _service_source_closures,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.tps_residue_load import (
    _tps_residue_load,
)
from pytoce.run_lca.calculation.carbon import (
    carbon_consistency_table,
    direct_treatment_carbon_flows,
)


@pytest.fixture(scope="module", params=(False, True), ids=("original", "future"))
def frontier(request, tmp_path_factory):
    """Compile every actual treatment owner once in a dedicated Brightway project."""
    future = request.param
    catalogue = frontier_catalogue()
    root = tmp_path_factory.mktemp("tps_frontier")
    names = BrightwayNames(root / "brightway", "tps_frontier", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere_inventory(), searchable=False)
        bd.Database("original").write(native_inventory("original", future=False), searchable=False)
        native = "future" if future else "original"
        if future:
            bd.Database(native).write(native_inventory(native, future=True), searchable=False)
        authority = PreparedProviderAuthority(
            identity="invented_tps_frontier",
            interface_database=PROVIDER_INTERFACE_DATABASE,
            catalogue=catalogue,
            routes=_route_table(catalogue),
            method_authorities=_method_authority_rows(catalogue),
        )
        result = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            ordinary_database=native,
            technology_database=native,
            provider=authority,
            material_recipe=physical_material_recipe(),
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )
        yield bd, catalogue, result, native, future


@pytest.fixture(scope="module", params=ORIGINS)
def solved(request, frontier):
    """Solve one kg principal TPS through the public preparation frontier output."""
    bd, catalogue, result, native, future = frontier
    origin = request.param
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    residue = _tps_residue_load(services, f"tps_{origin}")
    principal = bd.get_node(
        database=result.database_name,
        code=provider_service_code(f"tps_residue_treatment_{origin}"),
    )
    lca = LCA({principal: 1})
    lca.lci()
    return bd, catalogue, result, native, future, origin, residue, lca


def _production(node):
    """Read the persisted signed output for a native or generated service."""
    return next(iter(node.production()))["amount"]


def _inputs(node):
    """Group supply exchanges and retain repeated native portfolios."""
    values = {}
    for row in node.technosphere():
        values.setdefault(row.input.key, []).append(row["amount"])
    return {key: fsum(amounts) for key, amounts in values.items()}


def test_frontier_activates_real_treatment_services_and_finite_source_closure(frontier):
    """Connect all actual TPS services while retaining formulations only as sources."""
    bd, catalogue, result, _, _ = frontier
    written = {node["provider_service_id"] for node in bd.Database(result.database_name)}
    assert written == ACTIVE_SERVICES
    assert {row["constituent"] for row in catalogue["routes"]} == {
        "other_biodegradable",
        "starch_blends",
    }
    assert {row["virgin_feedstock_route"] for row in catalogue["routes"]} == set(ORIGINS)
    assert not {f"tps_{origin}" for origin in ORIGINS} & written
    assert "plastic_residue_haul" not in written
    closure = _service_source_closures(catalogue)
    for origin in ORIGINS:
        # Ash references its incinerator composition and retains formulation and forming provenance.
        for suffix in ("bottom_ash_market", "residual_ash_treatment"):
            ids = set(closure[f"tps_{origin}_{suffix}"])
            assert {
                "sp_ivl_2000_13_component_landfill",
                "doka_wastetools_2021_original_models",
            } <= ids
        route_rows = [
            row
            for row in result.semantic_rows
            if row.get("ledger_entry_type") == "route_mapping"
            and row["provider_service_id"] == f"tps_residue_treatment_{origin}"
        ]
        assert len(route_rows) == 2
        assert all(row["ultimate_carbon_origin"] == "all" for row in route_rows)


def test_principal_demand_reaches_signed_site_water_and_actual_terminals_once(solved):
    """Keep source hydraulics, treated fractions and source-only ash links distinct."""
    bd, _, result, native, future, origin, residue, lca = solved
    provider = result.database_name

    def service(suffix):
        """Read this formulation's persisted conditional treatment node."""
        return bd.get_node(database=provider, code=provider_service_code(f"tps_{origin}_{suffix}"))

    def level(node):
        """Read the real solved activity level through the public LCA mapping."""
        return lca.supply_array[lca.dicts.activity[node.id]]

    principal = bd.get_node(
        database=provider, code=provider_service_code(f"tps_residue_treatment_{origin}")
    )
    site, collection, plant = [
        service(k) for k in ("landfill", "leachate_collection", "wastewater_treatment")
    ]
    q = residue.wet_mass_kg
    assert _production(principal) == 1
    assert _inputs(principal) == {site.key: pytest.approx(-q, rel=2e-12, abs=0)}
    # Invented source 0.006 m3/kg wet is retained despite dated donor 0.009.
    volume = ORIGINAL_LEACHATE * q
    treated = 0.6 if future else 0.8  # Invented complete collection-market shares.
    assert _production(site) == SITE_PRODUCTION[int(future)]
    for node, demand in ((site, q), (collection, volume), (plant, treated * volume)):
        assert level(node) == pytest.approx(demand / -_production(node), rel=3e-7, abs=0)
    assert _inputs(site)[collection.key] == pytest.approx(
        _production(site) * ORIGINAL_LEACHATE, rel=2e-12, abs=0
    )
    for source_code in ("site:site", "site:collection", "site:plant", "thermal:sludge"):
        assert level(bd.get_node(database=native, code=source_code)) == 0
    for suffix in TPS_SUFFIXES:
        assert level(service(suffix)) > 0
    for other in set(ORIGINS) - {origin}:
        for suffix in TPS_SUFFIXES:
            other_node = bd.get_node(
                database=provider, code=provider_service_code(f"tps_{other}_{suffix}")
            )
            assert level(other_node) == 0
    # Waste markets supply transport; terminals receive their parent market demand. Formula/source
    # references supply provenance.
    for parent, child in (
        (plant, service("sludge_incineration_market")),
        (plant, service("sludge_landfarming_market")),
        (service("sludge_incineration_market"), service("sludge_incineration")),
        (service("sludge_landfarming_market"), service("sludge_landfarming")),
        (service("sludge_incineration"), service("bottom_ash_market")),
        (service("sludge_incineration"), service("residual_ash_market")),
        (service("bottom_ash_market"), service("bottom_ash_treatment")),
        (service("residual_ash_market"), service("residual_ash_treatment")),
    ):
        assert _inputs(parent)[child.key] < 0
        assert level(child) == pytest.approx(
            level(parent) * _inputs(parent)[child.key] / _production(child), rel=3e-7, abs=0
        )
    for suffix in (
        "bottom_ash_market",
        "residual_ash_market",
        "bottom_ash_treatment",
        "residual_ash_treatment",
    ):
        assert service("sludge_incineration").key not in _inputs(service(suffix))
    owners = json.loads(principal["direct_treatment_carbon_owners_json"])
    expected_owners = {
        service(suffix).key
        for suffix in (
            "landfill",
            "wastewater_treatment",
            "sludge_incineration",
            "sludge_landfarming",
        )
    }
    assert len(owners) == 4
    assert {(row["owner_database"], row["owner_code"]) for row in owners} == expected_owners
    for row in owners:
        owner = bd.get_node(database=row["owner_database"], code=row["owner_code"])
        # Signed incoming demand divided by native production matches the matrix solution.
        assert row["owner_scale_per_provider_unit"] / _production(owner) == pytest.approx(
            level(owner), rel=3e-7, abs=0
        )
        assert row["origin_mapping_status"] == "actual_tps_principal_origin"
    trace = next(
        row
        for row in result.technology_rows
        if row.get("provider_service_id") == f"tps_{origin}_landfill"
        and row.get("component_id") == "TPS_actual_residue_landfill"
    )
    assert json.loads(trace["project_terms_json"])["leachate_m3_per_principal_kg"] == pytest.approx(
        volume, rel=2e-12, abs=0
    )


def test_native_catalysts_preserve_supplier_shares_and_auxiliary_origins(solved):
    """Check TiO2 shares, positive V2O5 demand and positive aggregate fossil CO2."""
    bd, _, result, native, future, origin, _, lca = solved
    thermal = bd.get_node(
        database=result.database_name,
        code=provider_service_code(f"tps_{origin}_sludge_incineration"),
    )
    inputs = _inputs(thermal)
    oxide_mix = 0.7 if future else 0.25
    assert inputs[native, "thermal:TiO2_a"] / fsum(
        inputs[native, f"thermal:TiO2_{suffix}"] for suffix in ("a", "b")
    ) == pytest.approx(oxide_mix, rel=2e-12)
    assert inputs[native, "thermal:V2O5_a"] > 0
    fossil_flow = bd.get_node(
        database="biosphere", code=flow_key("Carbon dioxide, fossil", "kilogram", ("air",))[1]
    )
    assert float(lca.inventory[lca.dicts.biosphere[fossil_flow.id], :].sum()) > 0


def test_tps_diagnostic_follows_actual_gas_origins_and_excludes_auxiliaries(solved):
    """Read persisted per-exchange origin shares."""
    bd, _, result, _, future, origin, residue, _ = solved
    principal_carbon = {
        name: fsum(
            pool.input_carbon_kg
            for pool in residue.carbon
            if pool.principal and pool.origin == name
        )
        for name in ORIGINS
    }
    total_carbon = fsum(principal_carbon.values())
    # A 2.5-kg demand is attributed by principal inlet carbon, as the
    # foreground owner does. The diagnostic restores that demand before using
    # the different methane formation rates of glucan and glycerol.
    demand = 2.5
    rows = [
        {
            "year": 2050,
            "region": "EUR",
            "life_cycle_stage": "landfill_treatment",
            "material": "BioPlastics",
            "constituent": "starch_blends",
            "product": "polymer",
            "virgin_feedstock_route": origin,
            "good": "all",
            "material_cycle_source": "virgin_feedstock",
            "ultimate_carbon_origin": name,
            "lca_fate": "landfill",
            "fate_origin": "remind_mfa",
            "provider_database": result.database_name,
            "provider_code": provider_service_code(f"tps_residue_treatment_{origin}"),
            "provider_anchor_year": 2050 if future else 2025,
            "amount": demand * carbon / total_carbon,
            "mfa_carbon_amount_kg": demand * carbon,
        }
        for name, carbon in principal_carbon.items()
        if carbon > 0
    ]
    activations = pd.DataFrame(rows)
    flows = direct_treatment_carbon_flows(bd, activations)
    compact = carbon_consistency_table(bd, activations)
    assert set(flows.mfa_carbon_origin) == {
        name for name, carbon in principal_carbon.items() if carbon > 0
    }
    assert set(flows.owner_code) == {
        provider_service_code(f"tps_{origin}_{suffix}")
        for suffix in (
            "landfill",
            "wastewater_treatment",
            "sludge_incineration",
            "sludge_landfarming",
        )
    }
    assert flows.provider_demand_amount.to_numpy() == pytest.approx(demand)
    assert compact.provider_demand_amount.to_numpy() == pytest.approx(demand)
    assert not flows.duplicated(["owner_code", "flow_code", "mfa_carbon_origin"]).any()
    expected_compact = flows.groupby("mfa_carbon_origin").carbon_amount.agg(fsum)
    for row in compact.itertuples():
        assert row.native_direct_selected_carbon_kg == pytest.approx(
            expected_compact[row.mfa_carbon_origin]
        )
        assert row.mfa_treatment_input_carbon_kg == pytest.approx(
            demand * principal_carbon[row.mfa_carbon_origin]
        )
    # Selected principal methane follows each class' formed methane C, even
    # though DAC and biogenic primer use the same non-fossil native flow.
    methane = (
        flows.loc[
            flows.owner_code.eq(provider_service_code(f"tps_{origin}_landfill"))
            & flows.carbon_species.eq("methane")
        ]
        .groupby("mfa_carbon_origin")
        .carbon_amount.agg(fsum)
    )
    formed = {
        name: fsum(
            pool.methane_carbon_kg
            for pool in residue.carbon
            if pool.principal and pool.origin == name
        )
        for name in methane.index
    }
    assert (methane / fsum(methane)).to_numpy() == pytest.approx(
        [formed[name] / fsum(formed.values()) for name in methane.index], rel=2e-12
    )
    if origin != "biogenic":
        assert methane["biogenic"] / fsum(methane) != pytest.approx(
            principal_carbon["biogenic"] / total_carbon, rel=1e-4, abs=0
        )
    if origin == "atmospheric_dac":
        assert set(flows.flow_carbon_origin) == {"non_fossil"}
    # Multiple identical activation fragments represent the same demand and
    # must preserve output values, coordinates, units, ordering and dtypes.
    split = pd.concat(
        [
            activations.assign(
                amount=activations.amount / 2,
                mfa_carbon_amount_kg=activations.mfa_carbon_amount_kg / 2,
            )
        ]
        * 2,
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(direct_treatment_carbon_flows(bd, split), flows)
    pd.testing.assert_frame_equal(carbon_consistency_table(bd, split), compact)
