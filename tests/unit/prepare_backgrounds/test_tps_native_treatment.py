"""Keep complete native treatment proxies on each historical TPS wet basis."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from math import fsum, log
from typing import Any, cast

import pandas as pd
import pytest
from material_recipe_fixtures import physical_material_recipe

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
from pytoce.run_lca.calculation.carbon import direct_treatment_carbon_flows

_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
_AIR = ("air", "urban air close to ground")
_HIGH = ("air", "non-urban air or from high stacks")
_SPECIES = ("Carbon dioxide", "Methane", "Carbon monoxide")
# Independent accepted formulation values per kg principal TPS.
_WET = {
    "biogenic": 1.0810810810810811,
    "fossil": 1.085738567914896,
    "atmospheric_dac": 1.085738567914896,
}
_PRINCIPAL = {
    "biogenic": {"biogenic": 0.43008615722273447},
    "fossil": {"fossil": 0.4190546373358607, "biogenic": 0.010969573299714026},
    "atmospheric_dac": {
        "atmospheric_dac": 0.4190546373358607,
        "biogenic": 0.010969573299714026,
    },
}
_AUXILIARY_C = 0.0022853051847661816


def _biosphere(database: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Provide exact native gas pairs in both treatment compartments and ammonia."""
    descriptors = [
        (f"{species}, {origin}", compartment)
        for species in _SPECIES
        for origin in ("fossil", "non-fossil")
        for compartment in (_AIR, _HIGH)
    ] + [("Ammonia", _AIR)]
    return {
        (database, f"flow_{index}"): {
            "name": name,
            "unit": "kilogram",
            "categories": categories,
            "type": "emission",
        }
        for index, (name, categories) in enumerate(descriptors)
    }


def _native_inventory(
    database: str,
    sources: list[dict[str, object]],
    biosphere: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Supply invented complete operations with dated and nonunit native quantities."""
    future = database == "future"
    basis, dated = (3.0, 7.0) if future else (2.0, 1.0)
    flows = {(row["name"], row["categories"]): key for key, row in biosphere.items()}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for product in ("electricity", "ash"):
        key = database, product
        result[key] = {
            "name": product,
            "reference product": product,
            "unit": "kilogram",
            "location": "GLO",
            "type": "process",
            "exchanges": [{"input": key, "amount": 1.0, "type": "production"}],
        }
    for index, source in enumerate(sources):
        key = database, f"treatment_{index}"
        municipal = "municipal" in str(source["name"])
        waste_compartment = _AIR if municipal else _HIGH
        utility_compartment = _HIGH if municipal else _AIR
        rows: list[dict[str, Any]] = [
            {"input": key, "amount": -basis, "type": "production"},
        ]
        # The source coefficients are deliberate fixture controls. Duplicate CO2
        # rows must remain distinct; explicit zero CO and signed ash must survive.
        entries = [
            ("power", (database, "electricity"), "technosphere", 0.7),
            ("ash", (database, "ash"), "technosphere", -0.06),
            ("ammonia", flows["Ammonia", _AIR], "biosphere", 0.02),
            ("waste_co2_a", flows["Carbon dioxide, fossil", waste_compartment], "biosphere", 1.2),
            ("waste_co2_b", flows["Carbon dioxide, fossil", waste_compartment], "biosphere", 0.03),
            ("waste_ch4", flows["Methane, fossil", waste_compartment], "biosphere", 0.02),
            (
                "utility_co2",
                flows["Carbon dioxide, fossil", utility_compartment],
                "biosphere",
                0.11,
            ),
            ("utility_ch4", flows["Methane, fossil", utility_compartment], "biosphere", 0.13),
            (
                "utility_co",
                flows["Carbon monoxide, fossil", utility_compartment],
                "biosphere",
                0.17,
            ),
            (
                "utility_biogenic",
                flows["Carbon dioxide, non-fossil", waste_compartment],
                "biosphere",
                0.19,
            ),
        ]
        if municipal:
            entries.append(
                ("waste_co", flows["Carbon monoxide, fossil", waste_compartment], "biosphere", 0.0)
            )
        for label, input_key, kind, value in entries:
            amount = value * basis * dated
            row: dict[str, Any] = {
                "input": input_key,
                "type": kind,
                "amount": amount,
                "fixture_row": label,
                "comment": f"Retained source {label}",
                "uncertainty type": 2 if amount > 0 else 0,
                # Independent lognormal location deliberately differs from log(amount).
                "loc": log(amount) + 0.07 if amount > 0 else amount,
                "minimum": min(amount * 0.5, amount * 1.5),
                "maximum": max(amount * 0.5, amount * 1.5),
            }
            if amount > 0:
                row["scale"] = 0.4
            if kind == "biosphere":
                flow = biosphere[input_key]
                row.update(
                    name=flow["name"],
                    unit=flow["unit"],
                    categories=flow["categories"],
                    flow=input_key[1],
                )
                if label.startswith("waste_"):
                    row["properties"] = {
                        "carbon content, fossil": {"amount": 0.5, "unit": "dimensionless"},
                        "carbon content, non-fossil": {"amount": 0.0, "unit": "dimensionless"},
                    }
            rows.append(row)
        result[key] = {
            "name": source["name"],
            "reference product": source["reference_product"],
            "unit": source["unit"],
            "location": source["exact_location"],
            "type": "process",
            "exchanges": rows,
        }
    return result


@dataclass(frozen=True)
class _CompiledTPS:
    """A real disposable source/project and its compiled TPS native proxies."""

    bd: Any
    context: _RecipeContext
    services: list[dict[str, object]]
    sources: list[dict[str, object]]


@pytest.fixture(scope="module", params=("original", "future"))
def compiled_tps(request: pytest.FixtureRequest, tmp_path_factory) -> Iterator[_CompiledTPS]:
    """Compile declared treatment services from synthetic Brightway inventories."""
    authority = prepared_provider_authority()
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, Any]], authority.catalogue["services"])
    }
    selected = [
        row for row in services.values() if row["construction"]["kind"] == "tps_native_treatment"
    ]
    assert len(selected) == 18  # Three formulations, one municipal and five dump classes.
    sources: list[dict[str, object]] = []
    for service in selected:
        source = service["construction"]["source"]
        if source not in sources:
            sources.append(source)
    root = tmp_path_factory.mktemp(f"tps_native_{request.param}")
    names = BrightwayNames(root / "brightway", "tps_native", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        biosphere = _biosphere(names.biosphere_database)
        bd.Database(names.biosphere_database).write(biosphere, searchable=False)
        for database in ("original", "future"):
            bd.Database(database).write(
                _native_inventory(database, sources, biosphere), searchable=False
            )
        ordinary = _ActivityResolver(bd, request.param)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2025 if request.param == "original" else 2050,
            provider_database="tps_native_frontier",
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
        bd.Database(context.provider_database).write(
            {
                (
                    context.provider_database,
                    provider_service_code(str(service["provider_service_id"])),
                ): _service_dataset(context, service, services, context.provider_database)
                for service in selected
            },
            searchable=False,
        )
        yield _CompiledTPS(bd, context, selected, sources)


def test_tps_native_proxies_retain_every_source_row(compiled_tps: _CompiledTPS) -> None:
    """Scale complete dated inventories once and retain source metadata and duplicate rows."""
    bd, context = compiled_tps.bd, compiled_tps.context
    for service in compiled_tps.services:
        construction = cast(dict[str, Any], service["construction"])
        origin = construction["formulation_service_id"].removeprefix("tps_")
        source_index = compiled_tps.sources.index(construction["source"])
        source = bd.get_node(
            database=context.ordinary.database_name, code=f"treatment_{source_index}"
        )
        target = bd.get_node(
            database=context.provider_database,
            code=provider_service_code(cast(str, service["provider_service_id"])),
        )
        assert target.rp_exchange()["amount"] == 1.0
        multiplier = _WET[origin] / abs(source.rp_exchange()["amount"])
        generated = [row for row in target.exchanges() if row["type"] != "production"]
        native = [row for row in source.exchanges() if row["type"] != "production"]
        assert {row["fixture_row"] for row in generated} == {row["fixture_row"] for row in native}
        for row in native:
            assert row["minimum"] <= row["amount"] <= row["maximum"]
            pieces = [item for item in generated if item["fixture_row"] == row["fixture_row"]]
            assert fsum(item["amount"] for item in pieces) == pytest.approx(
                row["amount"] * multiplier
            )
            assert fsum(item["minimum"] for item in pieces) == pytest.approx(
                row["minimum"] * multiplier
            )
            assert fsum(item["maximum"] for item in pieces) == pytest.approx(
                row["maximum"] * multiplier
            )
            waste = row["fixture_row"].startswith("waste_")
            assert len(pieces) == (2 if waste and origin != "biogenic" else 1)
            for piece in pieces:
                assert piece["minimum"] <= piece["amount"] <= piece["maximum"]
                assert piece["comment"] == row["comment"]
                assert piece["uncertainty type"] == row["uncertainty type"]
                if row["uncertainty type"] == 2:
                    assert piece["scale"] == row["scale"]
                    assert piece["loc"] == pytest.approx(
                        row["loc"] + log(piece["amount"] / row["amount"])
                    )
                else:
                    assert piece["loc"] == pytest.approx(piece["amount"])
                if not waste:
                    assert piece["input"] == row["input"]
                    if row["fixture_row"].startswith("utility_"):
                        assert piece["principal_carbon_origin_fractions"] == (0.0, 0.0, 0.0)
                else:
                    assert piece.input["categories"] == row.input["categories"]
                    assert piece["flow"] == piece.input.key[1]
                    non_fossil = "non-fossil" in piece.input["name"]
                    assert piece["properties"]["carbon content, non-fossil"]["amount"] == (
                        0.5 if non_fossil else 0.0
                    )
                    assert piece["properties"]["carbon content, fossil"]["amount"] == (
                        0.0 if non_fossil else 0.5
                    )
        trace = next(
            row
            for row in context.technology_rows
            if row["provider_service_id"] == service["provider_service_id"]
        )
        terms = json.loads(cast(str, trace["project_terms_json"]))
        assert terms["wet_mass_kg_per_principal_kg"] == pytest.approx(_WET[origin])
        assert terms["native_source_key"] == list(source.key)
        assert terms["native_inventory_multiplier"] == pytest.approx(multiplier)
        assert "unchanged source yield" in terms["emission_basis"]
        assert json.loads(cast(str, trace["removed_technosphere_json"])) == []


def test_tps_proxy_diagnostic_excludes_auxiliaries_and_unrelated_gases(
    compiled_tps: _CompiledTPS,
) -> None:
    """Read actual origin fractions and preserve native proxy-gas quantities."""
    bd, context = compiled_tps.bd, compiled_tps.context
    for service in compiled_tps.services:
        construction = cast(dict[str, Any], service["construction"])
        origin = construction["formulation_service_id"].removeprefix("tps_")
        principal = _PRINCIPAL[origin]
        total_carbon = fsum(principal.values()) + (0.0 if origin == "biogenic" else _AUXILIARY_C)
        key = (
            context.provider_database,
            provider_service_code(cast(str, service["provider_service_id"])),
        )
        municipal = "municipal" in str(construction["source"]["name"])
        stage = "incineration_treatment" if municipal else "mismanaged_open_dump_treatment"
        demand = 2.5
        activations = pd.DataFrame(
            [
                {
                    "year": 2050,
                    "region": "EUR",
                    "life_cycle_stage": stage,
                    "material": "BioPlastics",
                    "constituent": "starch_blends",
                    "product": "polymer",
                    "virgin_feedstock_route": origin,
                    "good": "all",
                    "material_cycle_source": "virgin_feedstock",
                    "ultimate_carbon_origin": name,
                    "lca_fate": "incineration" if municipal else "open_dump",
                    "fate_origin": "remind_mfa",
                    "provider_database": key[0],
                    "provider_code": key[1],
                    "provider_anchor_year": context.year,
                    "amount": demand * carbon / fsum(principal.values()),
                    "mfa_carbon_amount_kg": demand * carbon,
                }
                for name, carbon in principal.items()
            ]
        )
        flows = direct_treatment_carbon_flows(bd, activations)
        assert set(flows.mfa_carbon_origin) == set(principal)
        assert set(flows.owner_code) == {key[1]}
        assert flows.provider_demand_amount.to_numpy() == pytest.approx(demand)
        # Invented per-wet-kg source gas factors: two CO2 rows sum to 1.23 kg;
        # CH4 is 0.02 kg. Future source coefficients are seven times these values.
        dated = 1.0 if context.year == 2025 else 7.0
        for species, gas_kg, carbon_fraction in (
            ("carbon dioxide", 1.23, 12 / 44),
            ("methane", 0.02, 12 / 16),
        ):
            actual = flows.loc[flows.carbon_species.eq(species)].set_index("mfa_carbon_origin")
            for name, carbon in principal.items():
                assert actual.loc[name, "carbon_amount"] == pytest.approx(
                    demand
                    * _WET[origin]
                    * dated
                    * gas_kg
                    * carbon_fraction
                    * carbon
                    / total_carbon,
                )
        assert set(flows.carbon_species) == {"carbon dioxide", "methane"}
