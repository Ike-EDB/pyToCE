"""Synthetic feeds retain complete procurement and their distinct quantity bases."""

from math import fsum, log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from provider_recipe_fixtures import _process

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

# Source-qualified supplier identities; weights and support inventories below
# are invented. No licensed inventory or private source capture is a test input.
_GAS_SUPPLIERS = (
    (
        "biomethane production, high pressure from synthetic gas, wood, fluidised technology",
        "CH",
        0.11,
    ),
    (
        "biomethane production, high pressure from synthetic gas, wood, fixed bed technology",
        "CH",
        0.17,
    ),
    ("biogas purification to biomethane by amino washing", "CH", 0.19),
    ("biogas purification to biomethane by membrane technique", "CH", 0.23),
    ("biogas purification to biomethane by pressure swing adsorption", "CH", 0.30),
)
_STARCH_SUPPLIERS = (
    ("citric acid production", "RER", 0.000002),
    ("citric acid production", "RoW", 0.000003),
    ("maize starch production", "DE", 0.17),
    ("citric acid production", "CN", 0.000005),
    ("citric acid production", "RNA", 0.000007),
    ("maize starch production", "RoW", 0.829983),
)
# Use NIST methane at 288.15 K and 0.101325 MPa with the technical-feed beta.
_METHANE_KG_PER_M3 = 0.67983
_AMYLOSE_FRACTION = 0.8549445155163381
_GAS_INPUTS_M3 = (0.010684597448073627, 1.5978844223854431)
_QUANTITY_FIELDS = (
    "amount",
    "uncertainty type",
    "loc",
    "scale",
    "minimum",
    "maximum",
    "negative",
    "allocation",
    "source_record",
)


def _dated(amount: float, future: bool) -> float:
    """Represent the source's original versus stored float32 precision."""
    return float(np.float32(amount)) if future else amount


@pytest.mark.parametrize(("year", "ordinary"), ((2025, "native"), (2050, "dated")))
@pytest.mark.parametrize(
    ("family", "origin"),
    (("gas", "atmospheric_dac"), ("starch", "fossil"), ("starch", "atmospheric_dac")),
)
def test_synthetic_feed_procurement_preserves_native_records_and_actual_basis(
    tmp_path: Path, year: int, ordinary: str, family: str, origin: str
) -> None:
    """Test recipient, delivery and unit-conversion declarations.

    Synthetic upstream suppliers have nonunit production and distinct
    emissions. The gas calculation retains its standard-volume conversion,
    so a missing or repeated kg/m3 conversion changes the result. The starch
    recipient retains its dry-amylose adjustment.
    """
    gas = family == "gas"
    catalogue = load_provider_catalogue()
    services = {s["provider_service_id"]: s for s in catalogue["services"]}
    root_id = "acetylene_atmospheric_dac" if gas else f"glucose_{origin}"
    delivered_id = (
        "methane_delivered_atmospheric_dac_standard_volume" if gas else f"starch_delivered_{origin}"
    )
    supply_id = "methane_atmospheric_dac" if gas else f"starch_{origin}"
    volume_id = "methane_atmospheric_dac_standard_volume"
    selected = [services[root_id], services[delivered_id]]
    if gas:
        selected.append(services[volume_id])
    selected.append(
        {
            **services[supply_id],
            "construction": {
                "kind": "native_activity",
                "source_role": "native_ecoinvent_provider",
                "selector": {
                    "name": f"admitted {supply_id}",
                    "reference_product": services[supply_id]["reference_product"],
                    "unit": "kilogram",
                    "exact_location": services[supply_id]["location"],
                },
                "amount": 1.0,
            },
        }
    )
    provider = PreparedProviderAuthority(
        identity=f"invented_{family}_{origin}_procurement",
        interface_database="synthetic_feed_interface",
        catalogue={**catalogue, "services": selected, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    suppliers = _GAS_SUPPLIERS if gas else _STARCH_SUPPLIERS
    market_product = "biomethane, high pressure" if gas else "maize starch"
    market_unit = "cubic meter" if gas else "kilogram"
    # Three network operating supplies plus four native emissions, versus four
    # starch freight supplies. Their values are deliberately small inventions.
    support = (
        (
            ("electricity", 0.02, "kilowatt hour"),
            ("heat", 0.3, "megajoule"),
            ("pipeline", 0.0004, "kilometer"),
        )
        if gas
        else (
            ("van", 0.012, "ton kilometer"),
            ("road", 0.27, "ton kilometer"),
            ("sea", 0.41, "ton kilometer"),
            ("rail", 0.13, "ton kilometer"),
        )
    )
    network_air = (
        (("nonfossil_co2", 0.003), ("h2s", 0.00002), ("methane", 0.0007), ("nitrogen", 0.00004))
        if gas
        else ()
    )
    markers = (
        "manufacture",
        "unreplaced",
        "utility",
        "waste",
        "direct",
        *(code for code, _, _ in support),
    )

    def inventory(database: str) -> dict[tuple[str, str], dict[str, Any]]:
        """Build invented source-shaped markets and independently traceable supports."""
        future = database == "dated"
        root_basis, market_basis, factory_basis = (3.0, 7.0, 5.0) if future else (2.0, 4.0, 2.0)

        def exchange(
            code: str, amount: float, tag: str, kind: str = "technosphere"
        ) -> dict[str, Any]:
            """Expose valid source distributions and unrelated allocation provenance."""
            row = {
                "input": ("biosphere" if kind == "biosphere" else database, code),
                "amount": amount,
                "type": kind,
                "source_record": tag,
                "activity": f"invented source UUID for {code}",
                "allocation": "already allocated native reference",
                "uncertainty type": 0 if future else 2,
                "loc": amount if future else log(abs(amount)) + 0.03,
            }
            if not future:
                row.update(
                    scale=0.23,
                    minimum=abs(amount) * 0.8,
                    maximum=abs(amount) * 1.2,
                    negative=amount < 0,
                )
            return row

        records = {
            (database, "selected_supply"): _process(
                database,
                "selected_supply",
                f"admitted {supply_id}",
                services[supply_id]["reference_product"],
                location=services[supply_id]["location"],
                production_amount=factory_basis,
                exchanges=[
                    exchange("manufacture", 13.0 if future else 7.0, "manufacture", "biosphere")
                ],
            )
        }
        product_rows = []
        for index, (name, location, weight) in enumerate(suppliers):
            code = f"producer_{index}"
            records[database, code] = _process(
                database,
                code,
                name,
                market_product,
                location=location,
                exchanges=[exchange("unreplaced", 11.0, code, "biosphere")],
            )
            records[database, code]["unit"] = market_unit
            product_rows.append(exchange(code, _dated(weight, future) * market_basis, code))
        for code, _, unit in support:
            records[database, code] = _process(
                database,
                code,
                f"native {code} supply",
                f"native {code} supply",
                location=("NEU" if gas else "World") if future else ("CH" if gas else "GLO"),
                exchanges=[exchange(code, 0.25, code, "biosphere")],
            )
            records[database, code]["unit"] = unit
        market_rows = [
            *product_rows,
            *(
                exchange(code, _dated(amount, future) * market_basis, code)
                for code, amount, _ in support
            ),
            *(
                exchange(code, _dated(amount, future) * market_basis, code, "biosphere")
                for code, amount in network_air
            ),
        ]
        records[database, "market"] = _process(
            database,
            "market",
            f"market for {market_product}",
            market_product,
            location="CH" if gas else "GLO",
            production_amount=market_basis,
            exchanges=market_rows,
        )
        records[database, "market"]["unit"] = market_unit
        for code, production, emission in (("utility", 1.0, 0.7), ("waste", -2.0, 0.6)):
            records[database, code] = _process(
                database,
                code,
                f"native recipient {code}",
                f"native recipient {code}",
                production_amount=production,
                exchanges=[exchange(code, emission, code, "biosphere")],
            )
        if gas:
            root_rows = []
            for index, amount in enumerate(_GAS_INPUTS_M3):
                code = f"original_gas_{index}"
                records[database, code] = _process(
                    database,
                    code,
                    f"market{' group' if index else ''} for natural gas, high pressure",
                    "natural gas, high pressure",
                    location="EUR"
                    if future
                    else ("CH" if index == 0 else "Europe without Switzerland"),
                    exchanges=[exchange("unreplaced", 11.0, code, "biosphere")],
                )
                records[database, code]["unit"] = "cubic meter"
                root_rows.append(exchange(code, _dated(amount, future) * root_basis, code))
            root_rows.append(
                exchange("fossil_co2", 0.017 * root_basis, "reaction_co2", "biosphere")
            )
        else:
            root_rows = [exchange("market", _dated(0.9, future) * root_basis, "starch_feed")]
        root_rows.extend(
            (
                exchange("utility", 0.11 * root_basis, "utility"),
                exchange("waste", -0.031 * root_basis, "waste"),
                exchange("direct", 0.013 * root_basis, "direct", "biosphere"),
            )
        )
        base = services[root_id]["construction"]["base"]["selector"]
        records[database, "recipient"] = _process(
            database,
            "recipient",
            base["name"],
            base["reference_product"],
            location=base["exact_location"],
            production_amount=root_basis,
            exchanges=root_rows,
        )
        for code in ("market", "recipient"):
            production = records[database, code]["exchanges"][0]
            production.update(
                exchange(code, production["amount"], f"production_{code}", "production")
            )
        for record in records.values():
            for row in record["exchanges"]:
                if row["type"] not in {"production", "technosphere"}:
                    continue
                source = records[row["input"]]
                product = source["reference product"]
                row.update(name=source["name"] if future else product, unit=source["unit"])
                if future:
                    row.update(product=product, location=source["location"])
                elif product in {market_product, "natural gas, high pressure"}:
                    row.update(
                        flow="invented source-grade flow",
                        **{"CAS number": "invented source CAS"},
                        classifications={"CPC": "invented donor grade"},
                        properties={
                            "carbon content": {"amount": 0.71, "unit": "dimensionless"},
                            "water in wet mass": {"amount": 0.04, "unit": "kg"},
                        },
                    )
        return records

    names = BrightwayNames(tmp_path / "brightway", "synthetic_feed_delivery", "biosphere", "native")
    with selected_brightway_project(names) as bd:
        flows = {
            ("biosphere", code): {
                "name": f"invented {code} witness",
                "unit": "kilogram",
                "categories": ("air",),
                "type": "emission",
            }
            for code in markers
        }
        for code, name in (
            ("fossil_co2", "Carbon dioxide, fossil"),
            ("nonfossil_co2", "Carbon dioxide, non-fossil"),
            ("h2s", "Hydrogen sulfide"),
            ("methane", "Methane, non-fossil"),
            ("nitrogen", "Nitrogen"),
        ):
            flows["biosphere", code] = {
                "name": name,
                "unit": "kilogram",
                "categories": ("air",),
                "type": "emission",
            }
        bd.Database("biosphere").write(flows, searchable=False)
        for database in ("native", "dated"):
            bd.Database(database).write(inventory(database), searchable=False)
        before = {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("native", "dated")
            for node in bd.Database(database)
        }
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=ordinary,
            technology_database="dated",
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("native", "biosphere"),
        )
        for code, service_id in (("market", delivered_id), ("recipient", root_id)):
            source = bd.get_node(database=ordinary, code=code)
            native_rows = {row["source_record"]: row for row in source.exchanges()}
            result = bd.get_node(
                database=frontier.database_name, code=provider_service_code(service_id)
            )
            rows = {row["source_record"]: row for row in result.exchanges()}
            assert set(rows) == set(native_rows)
            for tag, row in rows.items():
                original = native_rows[tag]
                factor = 1 / _AMYLOSE_FRACTION if tag == "starch_feed" else 1.0
                for field in _QUANTITY_FIELDS:
                    if field not in original:
                        assert field not in row
                    elif factor != 1 and field in {"amount", "minimum", "maximum", "loc"}:
                        expected = original[field] * factor
                        if field == "loc" and ordinary == "native":
                            expected = original[field] + log(factor)
                        assert row[field] == pytest.approx(expected)
                    else:
                        assert row[field] == original[field]
                target = None
                if row["type"] == "production":
                    target = service_id
                elif code == "market" and tag.startswith("producer_"):
                    target = volume_id if gas else supply_id
                elif tag.startswith("original_gas_") or tag == "starch_feed":
                    target = delivered_id
                if target:
                    assert row.input.key == (frontier.database_name, provider_service_code(target))
                    assert row["name"] == services[target]["reference_product"]
                    assert row["unit"] == services[target]["unit"]
                    assert "activity" not in row
                    if ordinary == "dated":
                        assert row["product"] == services[target]["reference_product"]
                        assert row["location"] == services[target]["location"]
                    else:
                        assert "product" not in row and "location" not in row
                    if "properties" in original:
                        for field in ("properties", "flow", "CAS number", "classifications"):
                            assert field not in row
                elif tag == "reaction_co2":
                    assert row.input.key == ("biosphere", "nonfossil_co2")
                else:
                    assert {k: v for k, v in dict(row).items() if k != "output"} == {
                        k: v for k, v in dict(original).items() if k != "output"
                    }
            if code == "market":
                assert len(list(result.technosphere())) == len(suppliers) + len(support)

        future = ordinary == "dated"
        demand = (
            fsum(_dated(q, future) for q in _GAS_INPUTS_M3)
            if gas
            else _dated(0.9, future) / _AMYLOSE_FRACTION
        )
        supplied = demand * fsum(_dated(q, future) for _, _, q in suppliers)
        kg_demand = supplied * _METHANE_KG_PER_M3 if gas else supplied
        root = bd.get_node(database=frontier.database_name, code=provider_service_code(root_id))
        market = bd.get_node(
            database=frontier.database_name, code=provider_service_code(delivered_id)
        )
        calculation = LCA({root: 1.0})
        calculation.lci()
        assert calculation.supply_array[calculation.dicts.activity[market.id]] == pytest.approx(
            demand / market.rp_exchange()["amount"]
        )
        factory = bd.get_node(database=ordinary, code="selected_supply")
        assert calculation.supply_array[calculation.dicts.activity[factory.id]] == pytest.approx(
            kg_demand / (5 if future else 2)
        )
        if gas:
            volume = bd.get_node(
                database=frontier.database_name, code=provider_service_code(volume_id)
            )
            assert calculation.supply_array[calculation.dicts.activity[volume.id]] == pytest.approx(
                supplied
            )
        expected = {
            "manufacture": kg_demand * (13 / 5 if future else 7 / 2),
            "unreplaced": 0.0,
            "utility": 0.11 * 0.7,
            "waste": 0.031 * 0.6 / 2,
            "direct": 0.013,
        }
        expected.update(
            {code: demand * _dated(amount, future) * 0.25 for code, amount, _ in support}
        )
        expected.update({code: demand * _dated(amount, future) for code, amount in network_air})
        if gas:
            expected["fossil_co2"] = 0.0
            expected["nonfossil_co2"] += 0.017
        for code, amount in expected.items():
            flow = bd.get_node(database="biosphere", code=code)
            actual = float(calculation.inventory[calculation.dicts.biosphere[flow.id], :].sum())
            assert actual == pytest.approx(amount, rel=3e-7)
        assert before == {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("native", "dated")
            for node in bd.Database(database)
        }
