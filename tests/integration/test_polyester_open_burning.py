"""Test polyester burning and manufacturing carbon origins.

The synthetic inventories exercise the Brightway interface against fixed
PBAT/PBS/PTT material, heat and phase expectations. Separate tests cover
the common burning equations and historical MFA.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.resources import files
from math import fsum
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from material_carbon_fixtures import material_carbon_sources
from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _service_source_closures,
    prepared_provider_authority,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.material_carbon import (
    prepare_material_carbon,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.run_lca.calculation.burdens import BURDEN_SPECS, MT_TO_KG, _origin_demands
from pytoce.run_lca.calculation.carbon import direct_treatment_carbon_flows
from tests.unit.prepare_backgrounds.pa_open_burning_fixtures import (
    _AIR,
    _INDICATOR,
    _SOIL,
    open_burning_biosphere,
    open_burning_native_inventory,
)

_ORIGINS: tuple[str, ...] = ("fossil", "biogenic", "atmospheric_dac")
_CONSTITUENTS = {"pbat": "pbat", "pbs": "pbs", "ptt": "other_bio_based"}
_FAMILIES = tuple(_CONSTITUENTS)
_SERVICE_IDS = tuple(
    f"{family}_open_burning_{origin}" for family in _FAMILIES for origin in _ORIGINS
)
_GASES = (
    ("carbon dioxide", "Carbon dioxide", 12 / 44),
    ("carbon monoxide", "Carbon monoxide", 12 / 28),
    ("methane", "Methane", 12 / 16),
)


@dataclass(frozen=True)
class _BurningReference:
    """Retained material and phase expectations per kg finished principal product."""

    elements: dict[str, float]
    selected_carbon: float
    fixed_fossil_carbon: float
    heat: float
    oxygen: float
    water: float
    gas_amounts: dict[str, float]
    emitted_elements: tuple[tuple[str, float, float], ...]


# All three one-kg products include their nominal TBOT-derived residue. Only PBAT
# contains the dry acid reference. PBS/PBAT organic atoms use CIAAW weights;
# PTT backbone and all TBOT atoms keep their separate nominal conventions.
_REFERENCES = {
    "pbat": _BurningReference(
        elements={
            "C": 0.625628439584006,
            "H": 0.06813828642886272,
            "O": 0.3047718721343507,
            "N": 0.0013057965706675544,
            "Ti": 9.487058823529411e-05,
            "P": 6.0734693877551016e-05,
        },
        selected_carbon=0.6207700759259209,
        fixed_fossil_carbon=0.004858363658085175,
        heat=26.761344484014707,
        oxygen=1.8536785137013265,
        water=0.5975616014358498,
        gas_amounts={
            "carbon dioxide": 2.1954078824896066,
            "carbon monoxide": 0.0386,
            "methane": 0.0059,
        },
        emitted_elements=(
            ("Titanium ion", 1.0059442695482492e-06, 9.386464396574586e-05),
            ("Phosphorus", 4.1728888985912354e-07, 6.031740498769189e-05),
        ),
    ),
    "pbs": _BurningReference(
        elements={
            "C": 0.5580762328035366,
            "H": 0.07030086823520178,
            "O": 0.3714295151679802,
            "Ti": 0.00019338379328137184,
        },
        selected_carbon=0.5573026976304111,
        fixed_fossil_carbon=0.0007735351731254874,
        heat=24.068851230521723,
        oxygen=1.6232461345147176,
        water=0.6170248376929014,
        gas_amounts={
            "carbon dioxide": 1.948958651280649,
            "carbon monoxide": 0.0386,
            "methane": 0.0059,
        },
        emitted_elements=(("Titanium ion", 2.0505124116278e-06, 0.00019133328086974404),),
    ),
    "ptt": _BurningReference(
        elements={
            "C": 0.640741030268418,
            "H": 0.04857057478266387,
            "O": 0.31062219887048675,
            "Ti": 0.00006619607843137255,
        },
        selected_carbon=0.6404762459546925,
        fixed_fossil_carbon=0.0002647843137254902,
        heat=24.777399293726027,
        oxygen=1.7302891780973741,
        water=0.4214521966200603,
        gas_amounts={
            "carbon dioxide": 2.2505428149108786,
            "carbon monoxide": 0.0386,
            "methane": 0.0059,
        },
        emitted_elements=(("Titanium ion", 0.0000007018989446914097, 0.00006549417948668114),),
    ),
}


@dataclass(frozen=True)
class _CompiledPolyesters:
    """Nine persisted origin services and the actual manufactured-carbon table."""

    bd: Any
    context: _RecipeContext
    authority: PreparedProviderAuthority
    carbon: pd.DataFrame
    year: int


@pytest.fixture(scope="module", params=(2025, 2050), ids=("original", "future"))
def compiled_polyesters(
    request: pytest.FixtureRequest, tmp_path_factory
) -> Iterator[_CompiledPolyesters]:
    """Use real admission, material construction, compilation and Brightway persistence."""
    year = int(request.param)
    authority = prepared_provider_authority()
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], authority.catalogue["services"])
    }
    root = tmp_path_factory.mktemp("polyester_burning")
    names = BrightwayNames(root / "brightway", "polyester_burning", "biosphere", "original")
    carbon_native, carbon_flows = material_carbon_sources("original", "biosphere")
    biosphere = open_burning_biosphere("biosphere")
    biosphere.update(carbon_flows)
    # The synthetic MIXED donor has zero Ti/P; the biosphere supplies separate titanium endpoints.
    for compartment in (_AIR, _SOIL):
        biosphere["biosphere", f"titanium_{compartment[0]}"] = {
            "name": "Titanium ion",
            "unit": "kilogram",
            "categories": compartment,
            "type": "emission",
        }
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere, searchable=False)
        for database in ("original", "future"):
            native = open_burning_native_inventory(
                database, original=database == "original", biosphere_database="biosphere"
            )
            donor = native[database, "burning"]
            donor["exchanges"][0]["amount"] = -3.0 if database == "original" else -7.0
            donor["exchanges"] = [
                row
                for row in donor["exchanges"]
                if row["type"] == "production" or biosphere[row["input"]]["name"] != "Phosphorus"
            ]
            if database == "original":
                native.update(carbon_native)
            bd.Database(database).write(native, searchable=False)
        carbon = pd.DataFrame(prepare_material_carbon("original", authority.catalogue))
        ordinary = _ActivityResolver(bd, "original" if year == 2025 else "future")
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=year,
            provider_database="polyester_frontier",
            source=_ActivityResolver(bd, "original"),
            ordinary=ordinary,
            technology=ordinary,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=authority.catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in authority.method_authorities
            },
            complete_source_ids=_service_source_closures(authority.catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        data = {
            (context.provider_database, provider_service_code(service_id)): _service_dataset(
                context, services[service_id], services, context.provider_database
            )
            for service_id in _SERVICE_IDS
        }
        bd.Database(context.provider_database).write(data, searchable=False)
        yield _CompiledPolyesters(bd, context, authority, carbon, year)


@pytest.mark.parametrize("family", _FAMILIES)
def test_polyester_burning_routes_replace_the_polymer_proxy(family: str) -> None:
    """Three ultimate-origin routes retain one full process per principal kg."""
    raw = json.loads(
        files("pytoce.prepare_backgrounds.data.provider_recipes")
        .joinpath("provider_catalogue.json")
        .read_text(encoding="utf-8")
    )
    for catalogue in (raw, prepared_provider_authority().catalogue):
        services = cast(list[dict[str, Any]], catalogue["services"])
        selected = {
            row["provider_service_id"]: row
            for row in services
            if row["provider_service_id"].startswith(f"{family}_open_burning_")
        }
        assert set(selected) == {f"{family}_open_burning_{origin}" for origin in _ORIGINS}
        routes = [
            row
            for row in cast(list[dict[str, Any]], catalogue["routes"])
            if row["material"] == "BioPlastics"
            and row["constituent"] == _CONSTITUENTS[family]
            and row["provider_demand_type"] == "mismanaged_open_burning_treatment"
        ]
        assert len(routes) == 3
        assert {row["ultimate_carbon_origin"] for row in routes} == set(_ORIGINS)
        for route in routes:
            origin = route["ultimate_carbon_origin"]
            assert route["provider_service_id"] == f"{family}_open_burning_{origin}"
            assert route["virgin_feedstock_route"] == "all"
            assert route["share"] == 1
            assert route["throughput_multiplier"] == 1
            construction = selected[route["provider_service_id"]]["construction"]
            assert construction["kind"] == "polyester_open_burning"
            assert construction["formulation_service_id"] == f"{family}_biogenic"
            assert construction["ultimate_carbon_origin"] == origin
            assert construction["source"]["reference_product"] == "waste plastic, mixture"


@pytest.mark.parametrize("origin", _ORIGINS)
@pytest.mark.parametrize("family", _FAMILIES)
def test_polyester_compiler_binds_complete_material_and_homogeneous_origin(
    compiled_polyesters: _CompiledPolyesters, family: str, origin: str
) -> None:
    """Dated source controls cannot rescale actual material, heat, Ti/P or principal gas."""
    compiled = compiled_polyesters
    reference = _REFERENCES[family]
    service_id = f"{family}_open_burning_{origin}"
    provider = compiled.bd.get_node(
        database=compiled.context.provider_database, code=provider_service_code(service_id)
    )
    assert provider.rp_exchange()["amount"] == 1
    assert provider["unit"] == "kilogram"
    assert list(provider.technosphere()) == []
    for database, production in (("original", -3), ("future", -7)):
        donor = compiled.bd.get_node(database=database, code="burning")
        assert donor.rp_exchange()["amount"] == production
        assert {exchange.input["name"] for exchange in donor.biosphere()}.isdisjoint(
            {"Titanium ion", "Phosphorus"}
        )
    trace = next(
        row
        for row in compiled.context.semantic_rows
        if row["provider_service_id"] == service_id
        and row["ledger_entry_type"] == "service_construction"
    )
    terms = json.loads(str(trace["project_terms_json"]))
    assert terms["original_source_key"] == ["original", "burning"]
    assert terms["ultimate_carbon_origin"] == origin
    assert terms["material_service_id"] == f"{family}_biogenic"
    assert terms["input_elements_kg"] == pytest.approx(reference.elements, rel=2e-12)
    assert fsum(terms["input_elements_kg"].values()) == pytest.approx(1)
    assert terms["wet_mass_kg_per_principal_kg"] == 1
    assert terms["incoming_water_kg"] == 0
    assert terms["principal_carbon_kg"] == pytest.approx({origin: reference.elements["C"]})
    assert terms["auxiliary_carbon_kg"] == {}
    assert terms["soil_compounds_kg"] == {}
    assert terms["gross_heat_potential_mj"] == pytest.approx(reference.heat, rel=2e-12)
    assert terms["atmospheric_oxygen_kg"] == pytest.approx(reference.oxygen, rel=2e-12)
    rows = {
        (str(row.input["name"]), str(row.input["unit"]), tuple(row.input["categories"])): row
        for row in provider.biosphere()
    }
    assert len(rows) == len(list(provider.biosphere()))
    assert all(
        row["uncertainty type"] == 0 and row["loc"] == row["amount"] for row in rows.values()
    )
    assert all(row["amount"] >= 0 for row in rows.values())
    assert not any("chlor" in name.casefold() or "dioxin" in name.casefold() for name, _, _ in rows)
    assert ("PCB", "kilogram", _AIR) not in rows
    assert ("Carbon dioxide, fossil", "kilogram", ("air",)) not in rows
    assert rows["Heat, waste", "megajoule", _AIR]["amount"] == pytest.approx(reference.heat)
    assert rows["Non-hazardous waste disposed", "kilogram", _INDICATOR]["amount"] == 1
    assert rows["Water", "cubic meter", _AIR]["amount"] == pytest.approx(reference.water / 1000)
    assert {name for name, _, _ in rows} & {"Titanium ion", "Phosphorus"} == {
        name for name, _, _ in reference.emitted_elements
    }
    if family in {"pbs", "ptt"}:
        # Thermal NOx/N2O still use atmospheric nitrogen. Formula-defined
        # source-fuel N/P and complete-inventory nitrogen flows use separate quantities.
        assert terms["input_elements_kg"].get("N", 0) == 0
        assert terms["input_elements_kg"].get("P", 0) == 0
        assert terms["air_compounds_kg"]["Nitrogen"] == 0
        assert terms["atmospheric_nitrogen_kg"] > 0
    for name, air, soil in reference.emitted_elements:
        assert rows[name, "kilogram", _AIR]["amount"] == pytest.approx(air, rel=2e-12)
        assert rows[name, "kilogram", _SOIL]["amount"] == pytest.approx(soil, rel=2e-12)
    for species, name, _ in _GASES:
        fossil = rows[f"{name}, fossil", "kilogram", _AIR]
        nonfossil = rows[f"{name}, non-fossil", "kilogram", _AIR]
        active, inactive = (fossil, nonfossil) if origin == "fossil" else (nonfossil, fossil)
        assert active["amount"] == pytest.approx(reference.gas_amounts[species], rel=2e-12)
        assert inactive["amount"] == 0
        assert active["principal_carbon_origin_fractions"] == tuple(
            float(name == origin) for name in _ORIGINS
        )


@pytest.mark.parametrize("family", _FAMILIES)
def test_manufactured_and_mixed_polyester_carbon_enters_burning_once(
    compiled_polyesters: _CompiledPolyesters, family: str
) -> None:
    """Retain fossil residue and PTT primer carbon when assigning treatment origins."""
    compiled = compiled_polyesters
    reference = _REFERENCES[family]
    manufactured = compiled.carbon.loc[
        compiled.carbon.constituent.eq(_CONSTITUENTS[family])
    ].set_index("virgin_feedstock_route")
    columns = [f"{origin}_carbon" for origin in _ORIGINS]
    vectors = manufactured.loc[list(_ORIGINS), columns].to_numpy(dtype=float)
    if family == "ptt":
        # The reference retains PTA/PDO carbon shares 8/11 and 3/11 and fossil TBOT.
        # Glucose carries 25/738 biogenic primer carbon: 0.050 g/L dextrin in
        # 1.476 g/L dry glucan, with common carbohydrate mixing and retention.
        expected = np.array(
            (
                (0.6348238439606325, 0.005917186307785407, 0),
                (0.0002647843137254902, 0.6404762459546925, 0),
                (0.0002647843137254902, 0.005917186307785407, 0.6345590596469071),
            )
        )
    else:
        expected = np.eye(3) * reference.selected_carbon
        expected[:, 0] += reference.fixed_fossil_carbon
    np.testing.assert_allclose(vectors, expected, rtol=2e-12, atol=0)
    spec = next(
        row for row in BURDEN_SPECS if row.demand_type == "mismanaged_open_burning_treatment"
    )
    routes = compiled.authority.routes.loc[
        compiled.authority.routes.constituent.eq(_CONSTITUENTS[family])
        & compiled.authority.routes.provider_demand_type.eq("mismanaged_open_burning_treatment")
    ].set_index("ultimate_carbon_origin")
    demand = 2.5
    # The second input combines retained articles from three manufacturing routes.
    for shares in ((0.0, 1.0, 0.0), (0.2, 0.3, 0.5)):
        carbon = np.asarray(shares) @ vectors
        demands = _origin_demands(
            spec,
            scenario="polyester_origin_fixture",
            year=2050,
            region="EUR",
            material="BioPlastics",
            constituent=_CONSTITUENTS[family],
            product="polymer",
            parent_amount_mt=demand / MT_TO_KG,
            allocation=1.0,
            carbon_mt=demand * carbon / MT_TO_KG,
        )
        activations = pd.DataFrame(demands).rename(
            columns={"demand_type": "life_cycle_stage", "amount_kg": "amount"}
        )
        activations["provider_database"] = compiled.context.provider_database
        activations["provider_code"] = [
            provider_service_code(str(routes.loc[origin, "provider_service_id"]))
            for origin in activations.ultimate_carbon_origin
        ]
        activations["provider_anchor_year"] = compiled.year
        assert fsum(activations.amount) == pytest.approx(demand)
        flows = direct_treatment_carbon_flows(compiled.bd, activations)
        assert set(flows.origin_mapping_status) == {f"actual_{family}_principal_origin"}
        assert not flows.duplicated(["provider_code", "flow_code", "mfa_carbon_origin"]).any()
        assert set(flows.flow_unit) == {"kilogram"}
        assert set(flows.carbon_species) == {row[0] for row in _GASES}
        for origin, amount in zip(_ORIGINS, carbon, strict=True):
            selected = flows.loc[flows.mfa_carbon_origin.eq(origin)]
            if amount == 0:
                assert selected.empty
                continue
            # Polyester providers are homogeneous by carbon origin, so each
            # attributed demand retains its allocated treatment mass.
            physical = demand * amount / fsum(carbon)
            assert selected.provider_demand_amount.to_numpy() == pytest.approx(physical)
            for species, _, fraction in _GASES:
                gas = reference.gas_amounts[species]
                assert fsum(selected.loc[selected.carbon_species.eq(species), "carbon_amount"]) == (
                    pytest.approx(physical * gas * fraction, rel=2e-12)
                )
        for species, _, fraction in _GASES:
            gas = reference.gas_amounts[species]
            assert fsum(flows.loc[flows.carbon_species.eq(species), "carbon_amount"]) == (
                pytest.approx(demand * gas * fraction, rel=2e-12)
            )
