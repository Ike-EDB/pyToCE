"""Check PHB conditioning and residue transport through provider compilation and LCI."""

import json
from math import fsum
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from recovery_delivery_fixtures import recovery_biosphere, recovery_native_inventory

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


@pytest.mark.parametrize(
    ("year", "market_production"), [(2025, -2.0), (2030, -2.0), (2025, 0.0), (2025, 1.0)]
)
def test_phb_recovery_replaces_energy_and_hauls_only_actual_residue(
    tmp_path: Path, year: int, market_production: float
) -> None:
    """Preserve signed source references and future freight with disposal counted once.

    Synthetic inputs have production amounts other than one and nonzero material
    burdens, distinguishing reference normalization from extra yield or disposal.
    Invalid external waste reference amounts are also supplied through Brightway
    and rejected before they can change the material-balance calculation.
    """
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    parent_id = "mechanical_recycling_phb"
    support_id = "recycling_pet_finishing_support"
    haul_id = "plastic_residue_haul"
    freight_id = "recycling_pet_pellet_delivery"
    selected = {parent_id, support_id, haul_id, freight_id}
    services = {
        row["provider_service_id"]: row
        for row in catalogue["services"]
        if row["provider_service_id"] in selected
    }
    assert {
        row["constituent"] for row in catalogue["routes"] if row["provider_service_id"] == parent_id
    } == {"pha"}
    provider = PreparedProviderAuthority(
        identity="invented_phb_recovery",
        interface_database="interface",
        catalogue={**catalogue, "services": list(services.values()), "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="phb",
        biosphere_database="biosphere",
        source_database="original",
    )
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(recovery_biosphere(), searchable=False)
        bd.Database("original").write(recovery_native_inventory("original"), searchable=False)
        data = recovery_native_inventory("native")

        def add_source(code: str, selector: dict, production: float, exchanges: list) -> None:
            """Install one invented source using the selected native descriptor."""
            data[("native", code)] = {
                "name": selector["name"],
                "reference product": selector["reference_product"],
                "unit": selector["unit"],
                "location": selector["exact_location"],
                "type": "process",
                "exchanges": [
                    {"input": ("native", code), "type": "production", "amount": production},
                    *exchanges,
                ],
            }

        for service_id, production in (
            (support_id, 2.0),
            (freight_id, 5.0),
            (haul_id, market_production),
        ):
            construction = services[service_id]["construction"]
            rules = (
                construction["omit_disposal"]
                if service_id == haul_id
                else [rule["selector"] for rule in construction["omit_technosphere"]]
            )
            exchanges = []
            for i, selector in enumerate(rules):
                code = f"{service_id}_omitted_{i}"
                if isinstance(selector["name"], list):
                    key = ("native", "electricity")
                elif service_id == freight_id:
                    key = ("native", support_id)
                else:
                    add_source(
                        code,
                        {**selector, "exact_location": selector["location"]},
                        1.0,
                        [{"input": ("biosphere", "co2"), "type": "biosphere", "amount": 100.0}],
                    )
                    key = ("native", code)
                exchanges.append({"input": key, "type": "technosphere", "amount": 0.9})
            if service_id == haul_id:
                descriptor = {
                    "name": "market for transport, freight, lorry, unspecified"
                    if year == 2025
                    else "market for transport, freight, lorry",
                    "reference_product": "transport, freight, lorry, diesel, unspecified"
                    if year == 2025
                    else "transport, freight, lorry",
                    "unit": "ton kilometer",
                    "exact_location": "RER" if year == 2025 else "EUR",
                }
                add_source(
                    "residue_lorry",
                    descriptor,
                    1.0,
                    [{"input": ("biosphere", "co2"), "type": "biosphere", "amount": 7.0}],
                )
                exchanges.append(
                    {
                        "input": ("native", "residue_lorry"),
                        "type": "technosphere",
                        "amount": 0.31,
                        "uncertainty type": 2,
                        "loc": -1.12,
                        "scale": 0.2,
                        "comment": "Invented haul quantity with a distinct lognormal location.",
                    }
                )
                # A native market may also contain a direct elementary exchange.
                # Retain it in the LCI and report it as biosphere provenance.
                exchanges.append(
                    {"input": ("biosphere", "co2"), "type": "biosphere", "amount": 0.004}
                )
            else:
                exchanges.extend(
                    [
                        {"input": ("native", "tap_water"), "type": "technosphere", "amount": 0.05},
                        {
                            "input": ("native", "wastewater_market"),
                            "type": "technosphere",
                            "amount": -0.00005,
                        },
                        {"input": ("native", "capital"), "type": "technosphere", "amount": 0.002},
                        {"input": ("biosphere", "co2"), "type": "biosphere", "amount": 0.004},
                    ]
                )
            add_source(service_id, construction["base"]["selector"], production, exchanges)
        bd.Database("native").write(data, searchable=False)

        def compile_frontier():
            """Compile PHB recovery services from the synthetic inventories."""
            return prepare_provider_frontier(
                bd,
                background_scenario="remind_ssp2_npi",
                year=year,
                ordinary_database="native",
                technology_database="native",
                material_recipe=physical_material_recipe(),
                provider=provider,
                resolver_cache=_NativeResolverCache("original", "biosphere"),
            )

        if market_production >= 0:
            with pytest.raises(ValueError, match="finite negative waste-input reference"):
                compile_frontier()
            return
        frontier = compile_frontier()
        parent = bd.get_node(database=frontier.database_name, code=provider_service_code(parent_id))
        # Independent source-fixture basis: finishing ratio=1.2 and dust=.002/kg pellets.
        pellets = 1 / 1.2
        residue = 1 - fsum((pellets, pellets * 0.002))
        demands = {exchange.input.key: exchange["amount"] for exchange in parent.technosphere()}
        assert demands[(frontier.database_name, provider_service_code(support_id))] == pellets
        assert demands[(frontier.database_name, provider_service_code(haul_id))] == residue
        assert demands[
            (frontier.database_name, provider_service_code(freight_id))
        ] == pytest.approx(pellets, rel=2e-14)
        assert demands[("native", "electricity")] == pytest.approx(0.35 * pellets, rel=2e-14)
        haul = bd.get_node(database=frontier.database_name, code=provider_service_code(haul_id))
        original_haul = bd.get_node(database="native", code=haul_id)
        haul_trace = next(
            row
            for row in frontier.technology_rows
            if row["provider_service_id"] == haul_id
            and row.get("component_id") == "plastic_residue_haul"
        )
        retained_technosphere = json.loads(cast(str, haul_trace["retained_technosphere_json"]))
        retained_biosphere = json.loads(cast(str, haul_trace["retained_biosphere_json"]))
        assert {row["exchange_type"] for row in retained_technosphere} == {"technosphere"}
        assert [(row["exchange_type"], row["amount"]) for row in retained_biosphere] == [
            ("biosphere", 0.004)
        ]
        (direct_emission,) = tuple(haul.biosphere())
        assert direct_emission["amount"] == 0.004
        (actual_haul,) = tuple(haul.technosphere())
        source_haul = next(
            ex for ex in original_haul.technosphere() if ex.input.key[1] == "residue_lorry"
        )
        assert {key: value for key, value in dict(actual_haul).items() if key != "output"} == {
            key: value for key, value in dict(source_haul).items() if key != "output"
        }
        (haul_production,) = tuple(haul.production())
        assert haul_production["amount"] == 2
        calculation = LCA({parent: 1.0})
        calculation.lci()
        lorry = bd.get_node(database="native", code="residue_lorry")
        assert cast(NDArray[np.float64], calculation.supply_array)[
            calculation.dicts.activity[lorry.id]
        ] == pytest.approx(float(np.float32(residue)) / 2 * float(np.float32(0.31)), rel=2e-14)
        for i in range(3):
            disposal = bd.get_node(database="native", code=f"{haul_id}_omitted_{i}")
            assert (
                cast(NDArray[np.float64], calculation.supply_array)[
                    calculation.dicts.activity[disposal.id]
                ]
                == 0
            )
