"""Compile complete historical TPS burning and read its actual principal carbon.

The two source graphs are small invented inventories with real native metadata.
The packaged formulations and compiler are unchanged. These tests cover their
quantity/source binding and carbon-reader contract; the full burning equations
and historical MFA routing are tested at their separate scientific owners.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from math import fsum
from typing import Any, cast

import pandas as pd
import pytest
from material_recipe_fixtures import physical_material_recipe
from pa_open_burning_fixtures import _AIR, _INDICATOR, _SOIL
from tps_open_burning_fixtures import (
    ORIGINS,
    SERVICE_IDS,
    tps_burning_biosphere,
    tps_burning_native_inventory,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.tps_residue_load import (
    _tps_material_load,
)
from pytoce.run_lca.calculation.carbon import direct_treatment_carbon_flows

# Independent accepted actual-formulation references, per kg principal. They
# deliberately retain the full co-delivered water/reagent and gross-heat basis.
_REFERENCES = {
    "biogenic": (1.0810810810810811, 18.28313679335153, 1.1382358908115457),
    "fossil": (1.085738567914896, 18.396303521524004, 1.146334691284502),
    "atmospheric_dac": (1.085738567914896, 18.396303521524004, 1.146334691284502),
}
_PRINCIPAL_C = {
    "biogenic": {"biogenic": 0.43008615722273447},
    "fossil": {"fossil": 0.4190546373358607, "biogenic": 0.010969573299714026},
    "atmospheric_dac": {
        "atmospheric_dac": 0.4190546373358607,
        "biogenic": 0.010969573299714026,
    },
}


@dataclass(frozen=True)
class _CompiledBurning:
    """One real selected project with persisted burning services and their provenance."""

    bd: Any
    context: _RecipeContext
    services: dict[str, dict[str, object]]
    database: str
    year: int


@pytest.fixture(scope="module", params=(False, True), ids=("original", "future"))
def compiled_burning(
    request: pytest.FixtureRequest, tmp_path_factory
) -> Iterator[_CompiledBurning]:
    """Compile all three services once using the actual preparation context and writer."""
    future = bool(request.param)
    authority = prepared_provider_authority()
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], authority.catalogue["services"])
    }
    root = tmp_path_factory.mktemp("tps_burning")
    names = BrightwayNames(root / "brightway", "tps_burning", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database(names.biosphere_database).write(
            tps_burning_biosphere(names.biosphere_database), searchable=False
        )
        for database in ("original", "future"):
            bd.Database(database).write(
                tps_burning_native_inventory(
                    database,
                    original=database == "original",
                    biosphere_database=names.biosphere_database,
                ),
                searchable=False,
            )
        year = 2050 if future else 2025
        ordinary = _ActivityResolver(bd, "future" if future else "original")
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=year,
            provider_database="tps_frontier",
            source=_ActivityResolver(bd, "original"),
            ordinary=ordinary,
            technology=ordinary,
            biosphere=_FlowResolver(bd, names.biosphere_database),
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
        # Full catalog entries remain quantity/provenance inputs. Only the
        # three affected operations are constructed at this real compiler seam.
        data = {
            (context.provider_database, provider_service_code(service_id)): _service_dataset(
                context, services[service_id], services, context.provider_database
            )
            for service_id in SERVICE_IDS
        }
        bd.Database(context.provider_database).write(data, searchable=False)
        yield _CompiledBurning(bd, context, services, context.provider_database, year)


def test_tps_burning_catalogue_replaces_only_the_two_native_proxy_routes() -> None:
    """Select three providers for the six formulation routes."""
    authority = prepared_provider_authority()
    services = cast(list[dict[str, Any]], authority.catalogue["services"])
    assert {
        row["provider_service_id"]
        for row in services
        if row["construction"]["kind"] == "tps_open_burning"
    } == set(SERVICE_IDS)
    assert not any(
        row["provider_service_id"].startswith("tps_open_burning_")
        and row["provider_service_id"] not in SERVICE_IDS
        for row in services
    )
    routes = authority.routes.loc[
        authority.routes.provider_demand_type.eq("mismanaged_open_burning_treatment")
        & authority.routes.material.eq("BioPlastics")
        & authority.routes.constituent.isin(("other_biodegradable", "starch_blends"))
    ]
    assert len(routes) == 6
    assert set(zip(routes.constituent, routes.virgin_feedstock_route, strict=True)) == {
        (constituent, origin)
        for constituent in ("other_biodegradable", "starch_blends")
        for origin in ORIGINS
    }
    assert set(routes.ultimate_carbon_origin) == {"all"}
    assert routes.share.to_numpy() == pytest.approx(1)
    assert routes.throughput_multiplier.to_numpy() == pytest.approx(1)
    for row in routes.itertuples():
        assert row.provider_service_id == f"tps_open_burning_{row.virgin_feedstock_route}"
    assert any(row["provider_service_id"] == "mismanaged_open_burning_mixed" for row in services)


@pytest.mark.parametrize("origin", ORIGINS)
def test_tps_burning_compiler_binds_actual_load_and_principal_carbon(
    compiled_burning: _CompiledBurning, origin: str
) -> None:
    """Keep complete outputs while the reader selects principal origins once."""
    compiled = compiled_burning
    context, bd = compiled.context, compiled.bd
    service_id = f"tps_open_burning_{origin}"
    key = compiled.database, provider_service_code(service_id)
    provider = bd.get_node(database=key[0], code=key[1])
    assert provider.rp_exchange()["amount"] == 1
    assert provider["unit"] == "kilogram"
    assert list(provider.technosphere()) == []
    assert bd.get_node(database="original", code="burning").rp_exchange()["amount"] == -3
    assert bd.get_node(database="future", code="burning").rp_exchange()["amount"] == -7
    assert {
        exchange.input["name"]
        for exchange in bd.get_node(database="original", code="burning").biosphere()
    }.isdisjoint({"Magnesium", "Phosphorus"})

    trace = next(
        row
        for row in context.semantic_rows
        if row["provider_service_id"] == service_id
        and row["ledger_entry_type"] == "service_construction"
    )
    terms = json.loads(str(trace["project_terms_json"]))
    inlet = _tps_material_load(compiled.services, f"tps_{origin}")
    wet, gross, oxygen = _REFERENCES[origin]
    assert terms["original_source_key"] == ["original", "burning"]
    assert terms["formulation_service_id"] == f"tps_{origin}"
    assert terms["wet_mass_kg_per_principal_kg"] == pytest.approx(wet, rel=2e-12)
    assert terms["gross_heat_potential_mj"] == pytest.approx(gross, rel=2e-12)
    assert terms["atmospheric_oxygen_kg"] == pytest.approx(oxygen, rel=2e-12)
    assert terms["input_elements_kg"] == inlet.elements_kg
    assert terms["incoming_water_kg"] == inlet.water_kg
    principal = _PRINCIPAL_C[origin]
    assert terms["principal_carbon_kg"] == pytest.approx(
        {name: principal.get(name, 0.0) for name in ORIGINS}, rel=2e-12
    )
    complete_sources = json.loads(str(trace["complete_source_authority_ids_json"]))
    assert "surendren_2024_thermoplastic_starch" in complete_sources
    assert "doka_2003_waste_incineration" in complete_sources

    rows = {
        (str(row.input["name"]), str(row.input["unit"]), tuple(row.input["categories"])): row
        for row in provider.biosphere()
    }
    assert len(rows) == len(list(provider.biosphere()))
    assert all(
        row["uncertainty type"] == 0 and row["loc"] == row["amount"] for row in rows.values()
    )
    assert all(row["amount"] >= 0 for row in rows.values())
    assert ("PCB", "kilogram", _AIR) not in rows
    assert ("Carbon dioxide, fossil", "kilogram", ("air",)) not in rows
    assert rows["Heat, waste", "megajoule", _AIR]["amount"] == pytest.approx(gross)
    assert rows["Non-hazardous waste disposed", "kilogram", _INDICATOR]["amount"] == pytest.approx(
        wet
    )
    assert rows["Oxygen", "kilogram", ("natural resource", "in air")]["amount"] == pytest.approx(
        oxygen
    )
    pm_names = (
        "Particulate Matter, > 10 um",
        "Particulate Matter, > 2.5 um and < 10um",
        "Particulate Matter, < 2.5 um",
    )
    pm = [float(rows[name, "kilogram", _AIR]["amount"]) for name in pm_names]
    if origin != "biogenic":
        assert [amount / fsum(pm) for amount in pm] == pytest.approx((0.20, 0.28, 0.52))
        for name in ("Magnesium", "Zinc II", "Phosphorus", "Sodium I"):
            assert rows[name, "kilogram", _AIR]["amount"] > 0
            assert rows[name, "kilogram", _SOIL]["amount"] > 0
        assert rows["Hydrochloric acid", "kilogram", _AIR]["amount"] > 0
        assert rows["Polychlorinated biphenyls", "kilogram", _AIR]["amount"] > 0
    else:
        assert fsum(pm) == 0
        assert terms["auxiliary_carbon_kg"] == {name: 0 for name in ORIGINS}

    # The 2.5-kg demand is split by inlet principal origin exactly as
    # the foreground owner does; output principal fractions exclude auxiliaries.
    demand = 2.5
    activations = pd.DataFrame(
        [
            {
                "year": 2050,
                "region": "EUR",
                "life_cycle_stage": "mismanaged_open_burning_treatment",
                "material": "BioPlastics",
                "constituent": "starch_blends",
                "product": "polymer",
                "virgin_feedstock_route": origin,
                "good": "all",
                "material_cycle_source": "virgin_feedstock",
                "ultimate_carbon_origin": name,
                "lca_fate": "open_burning",
                "fate_origin": "remind_mfa",
                "provider_database": key[0],
                "provider_code": key[1],
                "provider_anchor_year": compiled.year,
                "amount": demand * carbon / fsum(principal.values()),
                "mfa_carbon_amount_kg": demand * carbon,
            }
            for name, carbon in principal.items()
        ]
    )
    flows = direct_treatment_carbon_flows(bd, activations)
    assert set(flows.owner_code) == {key[1]}
    assert set(flows.mfa_carbon_origin) == set(principal)
    assert set(flows.flow_unit) == {"kilogram"}
    assert set(flows.origin_mapping_status) == {"actual_tps_principal_origin"}
    assert flows.provider_demand_amount.to_numpy() == pytest.approx(demand)
    assert not flows.duplicated(["flow_code", "mfa_carbon_origin"]).any()
    assert set(flows.carbon_species) == {"carbon dioxide", "carbon monoxide", "methane"}
    all_carbon = inlet.elements_kg["C"]
    # Fixed-formulation burning uses one burnable C origin pool. This independent
    # contraction checks the *whole* gas quantity, including the non-fossil pool
    # shared by DAC principal, biogenic primer and retained protein.
    for species, native_name, carbon_fraction in (
        ("carbon dioxide", "Carbon dioxide", 12 / 44),
        ("carbon monoxide", "Carbon monoxide", 12 / 28),
        ("methane", "Methane", 12 / 16),
    ):
        native_amount = fsum(
            float(rows[f"{native_name}, {label}", "kilogram", _AIR]["amount"])
            for label in ("fossil", "non-fossil")
        )
        actual = flows.loc[flows.carbon_species.eq(species)].set_index("mfa_carbon_origin")
        for name, carbon in principal.items():
            assert actual.loc[name, "carbon_amount"] == pytest.approx(
                demand * native_amount * carbon_fraction * carbon / all_carbon, rel=2e-12
            )
        if origin != "biogenic":
            assert fsum(actual.carbon_amount) < demand * native_amount * carbon_fraction
    if origin == "atmospheric_dac":
        assert rows["Carbon dioxide, fossil", "kilogram", _AIR]["amount"] > 0
        assert set(flows.flow_carbon_origin) == {"non_fossil"}
