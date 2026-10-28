"""UPR preserves native PG demand while purchasing its complete dated delivery."""

from math import log
from pathlib import Path
from typing import Any

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


@pytest.mark.parametrize(("year", "ordinary"), ((2025, "native"), (2050, "dated")))
def test_upr_native_feed_purchases_complete_pg_delivery_once(
    tmp_path: Path, year: int, ordinary: str
) -> None:
    """Solve two origin roots with independent feed boundaries and whole PG markets.

    The actual UPR and PG-delivery declarations remain intact. Independently
    tested delivered feeds and PG manufacture stop at native-activity fixture
    supplies; their dated emissions expose incorrect origin/date substitutions.
    This test changes no formulation, manufacture or retention method.
    """
    catalogue = load_provider_catalogue()
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    origins = ("biogenic", "atmospheric_dac")
    selected = [
        services[service_id]
        for origin in origins
        for service_id in (
            f"unsaturated_polyester_resin_{origin}",
            f"propylene_glycol_delivered_{origin}",
        )
    ]
    feeds = {"maleic_anhydride": 0.16, "phthalic_anhydride": 0.24, "styrene": 0.4}
    supplied = {
        origin: {
            **{label: f"{label}_delivered_{origin}" for label in feeds},
            "propylene_glycol": f"propylene_glycol_{origin}",
        }
        for origin in origins
    }
    supplied_ids = tuple(
        service_id for origin in origins for service_id in supplied[origin].values()
    )
    for service_id in supplied_ids:
        selected.append(
            {
                **services[service_id],
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "selector": {
                        "name": f"admitted {service_id}",
                        "reference_product": services[service_id]["reference_product"],
                        "unit": services[service_id]["unit"],
                        "exact_location": "RER",
                    },
                    "amount": 1.0,
                },
            }
        )
    provider = PreparedProviderAuthority(
        identity="invented_upr_pg_delivery",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    freight = {"lorry": 0.0219, "rail": 0.1713, "ship": 0.0438}

    def inventory(database: str) -> dict[tuple[str, str], dict[str, Any]]:
        """Provide native UPR/PG rows on different source outputs and linked freight."""
        future = database == "dated"
        root_basis, market_basis = (3.0, 7.0) if future else (1.0, 4.0)
        manufacturer_basis = 4.0 if future else 2.0

        def exchange(code: str, amount: float, kind: str = "technosphere") -> dict[str, Any]:
            """Preserve an invented native lognormal record with nontrivial location."""
            return {
                "input": ("biosphere" if kind == "biosphere" else database, code),
                "amount": amount,
                "type": kind,
                "uncertainty type": 2,
                "loc": log(abs(amount)) + 0.03,
                "scale": 0.2262741699796952,
                "minimum": abs(amount) * 0.8,
                "maximum": abs(amount) * 1.2,
                "negative": amount < 0,
                "allocation": "invented native allocation",
            }

        records = {}
        for index, service_id in enumerate(supplied_ids, start=1):
            records[database, service_id] = _process(
                database,
                service_id,
                f"admitted {service_id}",
                services[service_id]["reference_product"],
                production_amount=manufacturer_basis,
                exchanges=[exchange(service_id, index * (3 if future else 1), "biosphere")],
            )
        for label in feeds:
            product = label.replace("_", " ")
            records[database, label] = _process(
                database,
                label,
                f"market for {product}",
                product,
                location="GLO" if label == "maleic_anhydride" else "RER",
                exchanges=[exchange("unreplaced_feed", 16, "biosphere")],
            )
        records[database, "pg_producer"] = _process(
            database,
            "pg_producer",
            "propylene glycol production, liquid",
            "propylene glycol, liquid",
            exchanges=[exchange("unreplaced_feed", 16, "biosphere")],
        )
        for index, mode in enumerate(freight, start=1):
            code = f"freight_{mode}"
            records[database, code] = _process(
                database,
                code,
                f"transport, freight, {mode}",
                f"transport, freight, {mode}",
                location="GLO",
                exchanges=[exchange(code, index / 4, "biosphere")],
            )
            records[database, code]["unit"] = "ton kilometer"
        records[database, "pg_market"] = _process(
            database,
            "pg_market",
            "market for propylene glycol, liquid",
            "propylene glycol, liquid",
            production_amount=market_basis,
            exchanges=[
                exchange("pg_producer", market_basis),
                *(
                    exchange(f"freight_{mode}", coefficient * market_basis)
                    for mode, coefficient in freight.items()
                ),
            ],
        )
        records[database, "utility"] = _process(
            database,
            "utility",
            "native UPR utility",
            "native UPR utility",
            exchanges=[exchange("utility", 0.7, "biosphere")],
        )
        records[database, "upr"] = _process(
            database,
            "upr",
            "orthophthalic acid based unsaturated polyester resin production",
            "orthophthalic acid based unsaturated polyester resin",
            production_amount=root_basis,
            exchanges=[
                *(exchange(label, amount * root_basis) for label, amount in feeds.items()),
                exchange("pg_market", 0.25 * root_basis),
                exchange("utility", 0.11 * root_basis),
                exchange("direct_upr", 0.017 * root_basis, "biosphere"),
            ],
        )
        for code in ("upr", "pg_market"):
            production = records[database, code]["exchanges"][0]
            production.update(exchange(code, production["amount"], "production"))
        return records

    names = BrightwayNames(tmp_path / "brightway", "upr_feed_delivery", "biosphere", "native")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": f"invented {code} inventory witness",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
                for code in (
                    *supplied_ids,
                    *(f"freight_{mode}" for mode in freight),
                    "unreplaced_feed",
                    "utility",
                    "direct_upr",
                )
            },
            searchable=False,
        )
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
        native_root = bd.get_node(database=ordinary, code="upr")
        native_market = bd.get_node(database=ordinary, code="pg_market")
        root_basis = native_root.rp_exchange()["amount"]
        market_basis = native_market.rp_exchange()["amount"]
        quantity_fields = (
            "amount",
            "uncertainty type",
            "loc",
            "scale",
            "minimum",
            "maximum",
            "negative",
            "allocation",
        )
        native_pg = next(
            row for row in native_root.technosphere() if row.input.key[1] == "pg_market"
        )
        for origin in origins:
            root = bd.get_node(
                database=frontier.database_name,
                code=provider_service_code(f"unsaturated_polyester_resin_{origin}"),
            )
            delivery = bd.get_node(
                database=frontier.database_name,
                code=provider_service_code(f"propylene_glycol_delivered_{origin}"),
            )
            assert {field: root.rp_exchange()[field] for field in quantity_fields} == {
                field: native_root.rp_exchange()[field] for field in quantity_fields
            }
            assert {field: delivery.rp_exchange()[field] for field in quantity_fields} == {
                field: native_market.rp_exchange()[field] for field in quantity_fields
            }
            pg = next(row for row in root.technosphere() if row.input.key == delivery.key)
            assert {field: pg[field] for field in quantity_fields} == {
                field: native_pg[field] for field in quantity_fields
            }
            assert pg["amount"] / root_basis == pytest.approx(0.25)
            root_feeds = {row.input.key: row for row in root.technosphere()}
            assert set(root_feeds) == {
                delivery.key,
                (ordinary, "utility"),
                *(
                    (frontier.database_name, provider_service_code(supplied[origin][label]))
                    for label in feeds
                ),
            }
            for label, dose in feeds.items():
                actual = root_feeds[
                    frontier.database_name, provider_service_code(supplied[origin][label])
                ]
                native = next(
                    row for row in native_root.technosphere() if row.input.key[1] == label
                )
                assert {field: actual[field] for field in quantity_fields} == {
                    field: native[field] for field in quantity_fields
                }
                assert actual["amount"] / root_basis == pytest.approx(dose)
            # The whole market retains every field in its three freight records
            # except the generated output owner; only the manufacturer is replaced.
            for actual in delivery.technosphere():
                if actual.input.key[1].startswith("freight_"):
                    native = next(
                        row
                        for row in native_market.technosphere()
                        if row.input.key == actual.input.key
                    )
                    assert {
                        key: value for key, value in dict(actual).items() if key != "output"
                    } == {key: value for key, value in dict(native).items() if key != "output"}
                else:
                    assert actual.input.key == (
                        frontier.database_name,
                        provider_service_code(supplied[origin]["propylene_glycol"]),
                    )
                    assert actual["amount"] == market_basis
            calculation = LCA({root: 1.0})
            calculation.lci()
            assert calculation.supply_array[calculation.dicts.activity[root.id]] == pytest.approx(
                1 / root_basis
            )
            assert calculation.supply_array[
                calculation.dicts.activity[delivery.id]
            ] == pytest.approx(0.25 / market_basis)
            expected = {"unreplaced_feed": 0.0, "utility": 0.11 * 0.7, "direct_upr": 0.017}
            for index, service_id in enumerate(supplied_ids, start=1):
                doses = {**feeds, "propylene_glycol": 0.25}
                dose = next(
                    (
                        doses[label]
                        for label, selected_id in supplied[origin].items()
                        if selected_id == service_id
                    ),
                    0.0,
                )
                expected[service_id] = dose * index * (3 / 4 if ordinary == "dated" else 1 / 2)
            for index, (mode, coefficient) in enumerate(freight.items(), start=1):
                node = bd.get_node(database=ordinary, code=f"freight_{mode}")
                assert calculation.supply_array[
                    calculation.dicts.activity[node.id]
                ] == pytest.approx(0.25 * coefficient)
                expected[f"freight_{mode}"] = 0.25 * coefficient * index / 4
            for code, amount in expected.items():
                flow = bd.get_node(database="biosphere", code=code)
                actual = float(calculation.inventory[calculation.dicts.biosphere[flow.id], :].sum())
                assert actual == pytest.approx(amount, rel=2e-7)
        assert before == {
            node.key: (dict(node), [dict(row) for row in node.exchanges()])
            for database in ("native", "dated")
            for node in bd.Database(database)
        }
