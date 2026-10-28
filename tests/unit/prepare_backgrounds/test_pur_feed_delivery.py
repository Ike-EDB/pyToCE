"""PUR procurement preserves markets, formulations and composition-specific SAN metadata."""

from math import fsum, log
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
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

_ORIGINS = ("biogenic", "atmospheric_dac")
_RER_FREIGHT = (("barge", 0.0219), ("road", 0.1713), ("rail", 0.0438))
_ROW_FREIGHT = (
    ("road_za", 0.00806991183069717),
    ("road_br", 0.05154414670619652),
    ("road_row", 0.39078594146310636),
    ("sea", 0.5248),
    ("rail_za", 0.005303271752946537),
    ("rail_row", 0.10428848437824675),
    ("rail_cn", 0.07559239115070607),
    ("rail_us", 0.08257258958116194),
    ("rail_in", 0.0209432631369387),
)
# Source-qualified product shares and freight are represented by synthetic inventories.
_MARKETS = {
    "polyether_polyols_long_chain_delivered_rer": {
        "product": "polyether polyols, long chain",
        "location": "RER",
        "manufacturer": "polyether_polyols_long_chain",
        "suppliers": (("polyether polyols production, long chain", "RER", 1.0),),
        "freight": _RER_FREIGHT,
        "spread": 0.4125530268947254,
    },
    "polyether_polyols_long_chain_delivered_row": {
        "product": "polyether polyols, long chain",
        "location": "RoW",
        "manufacturer": "polyether_polyols_long_chain",
        "suppliers": (("polyether polyols production, long chain", "RoW", 1.0),),
        "freight": _ROW_FREIGHT,
        "spread": 0.41231056256176607,
    },
    "polyether_polyols_short_chain_delivered": {
        "product": "polyether polyols, short chain",
        "location": "RER",
        "manufacturer": "polyether_polyols_short_chain",
        "suppliers": (("polyether polyols production, short chain", "RER", 1.0),),
        "freight": _RER_FREIGHT,
        "spread": 0.4125530268947254,
    },
    "san_delivered": {
        "product": "styrene-acrylonitrile copolymer",
        "location": "GLO",
        "manufacturer": "san",
        "suppliers": (
            ("styrene-acrylonitrile copolymer production", "RoW", 0.669407648360697),
            ("styrene-acrylonitrile copolymer production", "RER", 0.330592351639303),
        ),
        "freight": (("road", 0.4504), ("sea", 0.5248), ("rail", 0.2887)),
        "spread": 0.44944410108488464,
    },
    "methylene_diphenyl_diisocyanate_delivered_rer": {
        "product": "methylene diphenyl diisocyanate",
        "location": "RER",
        "manufacturer": "methylene_diphenyl_diisocyanate",
        "suppliers": (("methylene diphenyl diisocyanate production", "RER", 1.0),),
        "freight": _RER_FREIGHT,
        "spread": 0.41472882706655445,
    },
    "methylene_diphenyl_diisocyanate_delivered_row": {
        "product": "methylene diphenyl diisocyanate",
        "location": "RoW",
        "manufacturer": "methylene_diphenyl_diisocyanate",
        "suppliers": (("methylene diphenyl diisocyanate production", "RoW", 1.0),),
        "freight": _ROW_FREIGHT,
        "spread": 0.44944410108488464,
    },
}
# Actual per-kg native purchases; future fixtures retain float32 precision.
_RECIPIENTS = {
    "pur_flexible": {
        "polyether_polyols_long_chain_delivered_rer": 0.74794631,
        "san_delivered": 0.00891573,
        "tdi": 0.2781273,
    },
    "pur_rigid": {
        "polyether_polyols_short_chain_delivered": 0.386,
        "methylene_diphenyl_diisocyanate_delivered_rer": 0.616,
    },
    "pur_adhesive": {
        "methylene_diphenyl_diisocyanate_delivered_row": 0.423333711468,
        "methylene_diphenyl_diisocyanate_delivered_rer": 0.10320075553199999,
        "polyether_polyols_long_chain_delivered_rer": 0.08972609931600001,
        "polyether_polyols_long_chain_delivered_row": 0.36806012168400004,
    },
}
_QUANTITIES = (
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


def _dated_amount(amount: float, database: str) -> float:
    """Keep source float32 precision, including the nonunit future SAN share sum."""
    return float(np.float32(amount)) if database == "dated" else amount


def _freight_code(market: str, mode: str, database: str) -> str:
    """Preserve distinct future native rows that share suppliers."""
    return f"{market}_{mode.split('_')[0] if database == 'dated' else mode}"


@pytest.mark.parametrize(("year", "ordinary"), ((2025, "native"), (2050, "dated")))
def test_pur_markets_preserve_geographic_inputs_and_selected_san_grade(
    tmp_path: Path, year: int, ordinary: str
) -> None:
    """Test procurement and metadata for all six PUR recipes.

    Synthetic upstream manufacturers have distinct emissions and nonunit
    outputs, making missing or duplicate manufacture and freight visible in
    the calculated inventories. Separate tests cover PUR treatment chemistry.
    """
    catalogue = load_provider_catalogue()
    services = {
        s["provider_service_id"]: s for s in cast(list[dict[str, Any]], catalogue["services"])
    }
    root_ids = tuple(f"{stem}_{origin}" for stem in _RECIPIENTS for origin in _ORIGINS)
    delivery_ids = tuple(f"{stem}_{origin}" for stem in _MARKETS for origin in _ORIGINS)
    supplied_ids = tuple(
        sorted(
            {
                *(
                    f"{data['manufacturer']}_{origin}"
                    for data in _MARKETS.values()
                    for origin in _ORIGINS
                ),
                *(f"toluene_diisocyanate_{origin}" for origin in _ORIGINS),
            }
        )
    )
    selected = [services[key] for key in (*root_ids, *delivery_ids)]
    for key in supplied_ids:
        selected.append(
            {
                **services[key],
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "selector": {
                        "name": f"admitted {key}",
                        "reference_product": services[key]["reference_product"],
                        "unit": services[key]["unit"],
                        "exact_location": services[key]["location"],
                    },
                    "amount": 1.0,
                },
            }
        )
    provider = PreparedProviderAuthority(
        identity="invented_complete_pur_procurement",
        interface_database="pur_interface",
        catalogue={**catalogue, "services": selected, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    freight_codes = tuple(
        dict.fromkeys(
            _freight_code(market, mode, database)
            for market, data in _MARKETS.items()
            for mode, _ in data["freight"]
            for database in ("native", "dated")
        )
    )
    markers = (
        *supplied_ids,
        *freight_codes,
        "unreplaced",
        "utility",
        "additive",
        "factory",
        "waste",
        "direct",
    )

    def inventory(database: str) -> dict[tuple[str, str], dict[str, Any]]:
        """Create source-shaped product rows and independent complete native supports."""
        future = database == "dated"
        root_basis, market_basis, factory_basis = (3.0, 7.0, 5.0) if future else (2.0, 4.0, 2.0)

        def exchange(
            code: str, amount: float, record: str, kind: str = "technosphere", spread: float = 0.19
        ) -> dict[str, Any]:
            """Expose native signs, distribution and unrelated allocation metadata."""
            row = {
                "input": ("biosphere" if kind == "biosphere" else database, code),
                "amount": amount,
                "type": kind,
                "source_record": record,
                "allocation": "invented native allocated record",
                "activity": f"native producer {code}",
                "uncertainty type": 0 if future else 2,
                "loc": amount if future else log(abs(amount)) + 0.03,
            }
            if not future:
                row.update(
                    scale=spread,
                    minimum=abs(amount) * 0.8,
                    maximum=abs(amount) * 1.2,
                    negative=amount < 0,
                )
            return row

        records = {}
        for index, key in enumerate(supplied_ids, start=1):
            records[database, key] = _process(
                database,
                key,
                f"admitted {key}",
                services[key]["reference_product"],
                location=services[key]["location"],
                production_amount=factory_basis,
                exchanges=[exchange(key, index * (3.0 if future else 1.0), key, "biosphere")],
            )
        for market, data in _MARKETS.items():
            inputs = []
            for index, (name, location, share) in enumerate(data["suppliers"]):
                code = f"{market}_producer_{index}"
                records[database, code] = _process(
                    database,
                    code,
                    name,
                    data["product"],
                    location=location,
                    exchanges=[exchange("unreplaced", 17.0, code, "biosphere")],
                )
                inputs.append(exchange(code, _dated_amount(share, database) * market_basis, code))
            for mode, dose in data["freight"]:
                code = _freight_code(market, mode, database)
                if data["location"] == "RER":
                    location = "EUR" if future and mode != "barge" else "RER"
                elif future:
                    location = "World"
                elif "_" in mode:
                    location = {
                        "za": "ZA",
                        "br": "BR",
                        "row": "RoW",
                        "cn": "CN",
                        "us": "US",
                        "in": "IN",
                    }[mode.split("_")[1]]
                else:
                    location = "GLO"
                records[database, code] = _process(
                    database,
                    code,
                    f"native freight {mode.split('_')[0]}",
                    f"freight {mode.split('_')[0]}",
                    location=location,
                    exchanges=[exchange(code, 0.25, code, "biosphere")],
                )
                records[database, code]["unit"] = "ton kilometer"
                inputs.append(
                    exchange(
                        code,
                        _dated_amount(dose, database) * market_basis,
                        f"{market}_{mode}",
                        spread=data["spread"],
                    )
                )
            records[database, market] = _process(
                database,
                market,
                f"market for {data['product']}",
                data["product"],
                location=data["location"],
                production_amount=market_basis,
                exchanges=inputs,
            )
        # Onsite TDI is a direct one-to-one manufacturer input in the flexible PUR recipe.
        records[database, "tdi_producer"] = _process(
            database,
            "tdi_producer",
            "toluene diisocyanate production",
            "toluene diisocyanate",
            exchanges=[exchange("unreplaced", 17.0, "tdi_producer", "biosphere")],
        )
        records[database, "tdi"] = _process(
            database,
            "tdi",
            "market for toluene diisocyanate",
            "toluene diisocyanate",
            exchanges=[exchange("tdi_producer", 1.0, "tdi_product")],
        )
        for code, emission, production in (
            ("utility", 0.7, 1.0),
            ("additive", 0.4, 1.0),
            ("factory", 0.2, 1.0),
            ("waste", 0.6, -2.0),
        ):
            records[database, code] = _process(
                database,
                code,
                f"native PUR {code}",
                f"native PUR {code}",
                production_amount=production,
                exchanges=[exchange(code, emission, code, "biosphere")],
            )
        for stem, feeds in _RECIPIENTS.items():
            base = services[f"{stem}_biogenic"]["construction"]["base"]["selector"]
            inputs = [
                exchange(market, _dated_amount(dose, database) * root_basis, market)
                for market, dose in feeds.items()
            ]
            inputs.extend(
                exchange(code, dose * root_basis, code)
                for code, dose in (
                    ("utility", 0.11),
                    ("additive", 0.02),
                    ("factory", 0.003),
                    ("waste", -0.031),
                )
            )
            inputs.append(exchange("direct", 0.017 * root_basis, "direct", "biosphere"))
            if stem == "pur_flexible":
                inputs.append(
                    exchange("fossil_co2", 0.013 * root_basis, "reaction_co2", "biosphere")
                )
            records[database, stem] = _process(
                database,
                stem,
                base["name"],
                base["reference_product"],
                location=base["exact_location"],
                production_amount=root_basis,
                exchanges=inputs,
            )
        for code in (*_MARKETS, *_RECIPIENTS):
            production = records[database, code]["exchanges"][0]
            production.update(
                exchange(code, production["amount"], f"production_{code}", "production")
            )
        for record in records.values():
            for row in record["exchanges"]:
                if row["type"] not in {"production", "technosphere"}:
                    continue
                supplier = records[row["input"]]
                product = supplier["reference product"]
                row.update(name=supplier["name"] if future else product, unit=supplier["unit"])
                if future:
                    row.update(product=product, location=supplier["location"])
                elif product == "styrene-acrylonitrile copolymer":
                    # Actual native 27.5%-AN descriptors differ from the selected
                    # 75:25 SAN. Valid generic identity and dry/water fields survive.
                    row.update(
                        flow="invented generic SAN flow", classifications={"CPC": "generic SAN"}
                    )
                    row["properties"] = {
                        "carbon allocation": {"amount": 0.856023222060958, "unit": "kg"},
                        "carbon content": {"amount": 0.856023222060958, "unit": "dimensionless"},
                        "carbon content, fossil": {
                            "amount": 0.856023222060958,
                            "unit": "dimensionless",
                            "comment": "27.5 wt% acrylonitrile",
                        },
                        "carbon content, non-fossil": {"amount": 0.0, "unit": "dimensionless"},
                        "heating value, net": {
                            "amount": 32.54,
                            "unit": "MJ",
                            "comment": "Source empirical formula C8.3H8.3N0.1",
                        },
                        "dry mass": {"amount": 1.0, "unit": "kg"},
                        "wet mass": {"amount": 1.0, "unit": "kg"},
                        "water content": {"amount": 0.0, "unit": "dimensionless"},
                    }
        return records

    names = BrightwayNames(tmp_path / "brightway", "pur_feed_delivery", "biosphere", "native")
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
        ):
            flows["biosphere", code] = {
                "name": name,
                "unit": "kilogram",
                "categories": ("air", "urban air close to ground"),
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
        # Compare complete source records at both market and recipient owners.
        # Separate input and output SAN witnesses exercise all four target IDs.
        for stem in (*_MARKETS, *_RECIPIENTS):
            source = bd.get_node(database=ordinary, code=stem)
            native_rows = {row["source_record"]: row for row in source.exchanges()}
            for origin in _ORIGINS:
                service_id = f"{stem}_{origin}"
                result = bd.get_node(
                    database=frontier.database_name, code=provider_service_code(service_id)
                )
                rows = {row["source_record"]: row for row in result.exchanges()}
                assert set(rows) == set(native_rows)
                for key, row in rows.items():
                    original = native_rows[key]
                    assert {k: row[k] for k in _QUANTITIES if k in row} == {
                        k: original[k] for k in _QUANTITIES if k in original
                    }
                    changed_target = None
                    if row["type"] == "production":
                        changed_target = service_id
                    elif stem in _MARKETS and "_producer_" in key:
                        changed_target = f"{_MARKETS[stem]['manufacturer']}_{origin}"
                    elif stem in _RECIPIENTS and key in _RECIPIENTS[stem]:
                        changed_target = (
                            f"toluene_diisocyanate_{origin}" if key == "tdi" else f"{key}_{origin}"
                        )
                    if changed_target:
                        assert row.input.key == (
                            frontier.database_name,
                            provider_service_code(changed_target),
                        )
                        target = services[changed_target]
                        assert row["name"] == target["reference_product"]
                        assert row["unit"] == target["unit"]
                        assert "activity" not in row
                        if ordinary == "dated":
                            assert row["product"] == target["reference_product"]
                            assert row["location"] == target["location"]
                            assert "properties" not in row
                        else:
                            assert "product" not in row and "location" not in row
                        if (
                            original.input["reference product"] == "styrene-acrylonitrile copolymer"
                            and ordinary == "native"
                        ):
                            assert row["properties"] == {
                                k: v
                                for k, v in original["properties"].items()
                                if k in {"dry mass", "wet mass", "water content"}
                            }
                            assert row["flow"] == original["flow"]
                            assert row["classifications"] == original["classifications"]
                            assert "chemical formula" not in row
                    elif key == "reaction_co2":
                        assert row.input.key == ("biosphere", "nonfossil_co2")
                    else:
                        assert {k: v for k, v in dict(row).items() if k != "output"} == {
                            k: v for k, v in dict(original).items() if k != "output"
                        }
                if stem in _MARKETS:
                    data = _MARKETS[stem]
                    assert len(list(result.technosphere())) == len(data["suppliers"]) + len(
                        data["freight"]
                    )

        for stem, feeds in _RECIPIENTS.items():
            for origin in _ORIGINS:
                root = bd.get_node(
                    database=frontier.database_name, code=provider_service_code(f"{stem}_{origin}")
                )
                root_rows = {row["source_record"]: row for row in root.technosphere()}
                expected = {
                    code: 0.0
                    for code in (
                        *supplied_ids,
                        "unreplaced",
                        "utility",
                        "additive",
                        "factory",
                        "waste",
                        "direct",
                    )
                }
                expected.update(
                    {
                        _freight_code(market, mode, ordinary): 0.0
                        for market, data in _MARKETS.items()
                        for mode, _ in data["freight"]
                    }
                )
                expected.update(
                    utility=0.11 * 0.7,
                    additive=0.02 * 0.4,
                    factory=0.003 * 0.2,
                    waste=0.031 * 0.6 / 2,
                    direct=0.017,
                )
                manufacture = dict.fromkeys(supplied_ids, 0.0)
                freight = {}
                for market, dose in feeds.items():
                    dose = _dated_amount(dose, ordinary)
                    assert root_rows[market]["amount"] / root.rp_exchange()[
                        "amount"
                    ] == pytest.approx(dose)
                    if market == "tdi":
                        manufacture[f"toluene_diisocyanate_{origin}"] += dose
                        continue
                    data = _MARKETS[market]
                    manufacture[f"{data['manufacturer']}_{origin}"] += dose * fsum(
                        _dated_amount(q, ordinary) for _, _, q in data["suppliers"]
                    )
                    for mode, amount in data["freight"]:
                        code = _freight_code(market, mode, ordinary)
                        freight[code] = freight.get(code, 0.0) + dose * _dated_amount(
                            amount, ordinary
                        )
                        expected[code] += dose * _dated_amount(amount, ordinary) * 0.25
                calculation = LCA({root: 1.0})
                calculation.lci()
                assert cast(NDArray[np.float64], calculation.supply_array)[
                    calculation.dicts.activity[root.id]
                ] == pytest.approx(1 / root.rp_exchange()["amount"])
                for market, dose in feeds.items():
                    if market != "tdi":
                        node = root_rows[market].input
                        assert cast(NDArray[np.float64], calculation.supply_array)[
                            calculation.dicts.activity[node.id]
                        ] == pytest.approx(
                            _dated_amount(dose, ordinary) / node.rp_exchange()["amount"]
                        )
                for code, demand in freight.items():
                    node = bd.get_node(database=ordinary, code=code)
                    assert cast(NDArray[np.float64], calculation.supply_array)[
                        calculation.dicts.activity[node.id]
                    ] == pytest.approx(demand)
                for index, key in enumerate(supplied_ids, start=1):
                    expected[key] = (
                        manufacture[key] * index * (3 / 5 if ordinary == "dated" else 1 / 2)
                    )
                for code, amount in expected.items():
                    flow = bd.get_node(database="biosphere", code=code)
                    actual = float(
                        calculation.inventory[calculation.dicts.biosphere[flow.id], :].sum()
                    )
                    assert actual == pytest.approx(amount, rel=3e-7)
                co2 = bd.get_node(database="biosphere", code="nonfossil_co2")
                assert float(
                    calculation.inventory[calculation.dicts.biosphere[co2.id], :].sum()
                ) == pytest.approx(0.013 if stem == "pur_flexible" else 0.0)
        assert before == {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("native", "dated")
            for node in bd.Database(database)
        }
