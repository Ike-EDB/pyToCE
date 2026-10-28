"""The actual PTT root purchases its full PTA market once at the adopted dose."""

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


@pytest.mark.parametrize("origin", ("biogenic", "atmospheric_dac"))
@pytest.mark.parametrize(("year", "ordinary"), ((2025, "native"), (2050, "dated")))
def test_ptt_root_scales_complete_pta_delivery_without_rescaling_other_inputs(
    tmp_path: Path, origin: str, year: int, ordinary: str
) -> None:
    """Compile the retained PTT recipe with complete PTA market delivery.

    Synthetic Brightway activities represent the PTT operating inputs, PDO,
    TBOT, PTA manufacture and finished resin delivery in this fixture. Their
    distinct emissions and nonunit outputs expose missing or duplicate inputs.
    The complete inventories are checked separately in their respective tests.
    This test checks the PTA market quantity purchased per kilogram of PTT.
    The PET market test checks every retained exchange and its metadata.
    """
    catalogue = load_provider_catalogue()
    services = {
        row["provider_service_id"]: row for row in cast(list[dict[str, Any]], catalogue["services"])
    }
    root_id = f"ptt_{origin}"
    delivery_id = f"purified_terephthalic_acid_delivered_{origin}"
    manufacturer_id = f"purified_terephthalic_acid_{origin}"
    independent = {
        "ptt_source_input_operation": 1.0,
        f"propanediol_{origin}": 0.37,
        "titanium_butoxide_delivered": 0.0004688888888888889,
        "recycling_pet_pellet_delivery": 1.0,
    }
    supplied_ids = (*independent, manufacturer_id)
    selected = [services[root_id], services[delivery_id]]
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
        identity="invented_ptt_feed_delivery",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )

    def inventory(database: str, factor: float) -> dict[tuple[str, str], dict[str, Any]]:
        """Build the complete GLO PTA market and independently emitting root inputs."""

        def row(code: str, amount: float, kind: str = "technosphere") -> dict[str, Any]:
            """Keep invented source quantities and meaningful native uncertainty."""
            return {
                "input": ("biosphere" if kind == "biosphere" else database, code),
                "amount": amount,
                "type": kind,
                "uncertainty type": 2,
                "loc": log(amount),
                "scale": 0.125,
            }

        records = {}
        for index, service_id in enumerate(supplied_ids, start=1):
            record = _process(
                database,
                service_id,
                f"admitted {service_id}",
                services[service_id]["reference_product"],
                production_amount=2 * factor,
                exchanges=[row(service_id, index * factor, "biosphere")],
            )
            record["unit"] = services[service_id]["unit"]
            records[database, service_id] = record
        market_inputs = []
        for index, location in enumerate(("RER", "CA-QC", "RoW")):
            code = f"pta_producer_{index}"
            records[database, code] = _process(
                database,
                code,
                "purified terephthalic acid production",
                "purified terephthalic acid",
                location=location,
                exchanges=[row("unreplaced_fossil", 16, "biosphere")],
            )
            # Retain the source's three supplier positions and a dated total
            # unequal to output. PET separately exercises the tiny native row.
            quantity = (0.5, 1.0, 3.0 if database == "native" else 3.5)[index]
            market_inputs.append(row(code, factor * quantity))
        for index, mode in enumerate(("lorry", "container ship", "rail"), start=1):
            code = f"freight_{index}"
            records[database, code] = _process(
                database,
                code,
                f"transport, freight, {mode}",
                f"transport, freight, {mode}",
                location="GLO",
                exchanges=[row(code, index / 4, "biosphere")],
            )
            records[database, code]["unit"] = "ton kilometer"
            market_inputs.append(row(code, factor * index / 64))
        records[database, "pta_market"] = _process(
            database,
            "pta_market",
            "market for purified terephthalic acid",
            "purified terephthalic acid",
            location="GLO",
            production_amount=4 * factor,
            exchanges=market_inputs,
        )
        return records

    names = BrightwayNames(tmp_path / "brightway", "ptt_feed_delivery", "biosphere", "native")
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
                    "freight_1",
                    "freight_2",
                    "freight_3",
                    "unreplaced_fossil",
                )
            },
            searchable=False,
        )
        for database, factor in (("native", 1.0), ("dated", 2.0)):
            bd.Database(database).write(inventory(database, factor), searchable=False)
        native_market = bd.get_node(database=ordinary, code="pta_market")
        source_rows = [dict(row) for row in native_market.exchanges()]
        market_output = next(row["amount"] for row in source_rows if row["type"] == "production")
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
        root = bd.get_node(database=frontier.database_name, code=provider_service_code(root_id))
        delivery = bd.get_node(
            database=frontier.database_name, code=provider_service_code(delivery_id)
        )
        # The root purchases 0.8 kg of delivered PTA per kg of PTT;
        # every independent dose retains the same one-kg PTT reference.
        assert next(iter(root.production()))["amount"] == 1.0
        assert {row.input.key: row["amount"] for row in root.technosphere()} == {
            delivery.key: 0.8,
            **{(ordinary, service_id): dose for service_id, dose in independent.items()},
        }
        calculation = LCA({root: 1.0})
        calculation.lci()
        expected_markers = {
            service_id: dose * (index + 1) / 2
            for index, (service_id, dose) in enumerate(independent.items())
        }
        producer_weight = 0.0
        for row in source_rows:
            if row["type"] != "technosphere":
                continue
            weighted = 0.8 * row["amount"] / market_output
            code = row["input"][1]
            if code.startswith("pta_producer_"):
                producer_weight += weighted
            else:
                freight = bd.get_node(database=ordinary, code=code)
                assert cast(NDArray[np.float64], calculation.supply_array)[
                    calculation.dicts.activity[freight.id]
                ] == (pytest.approx(weighted))
                emission = next(iter(freight.biosphere()))["amount"]
                expected_markers[code] = weighted * emission
        expected_markers[manufacturer_id] = producer_weight * len(supplied_ids) / 2
        expected_markers["unreplaced_fossil"] = 0.0
        assert producer_weight == pytest.approx(0.8 * (4.5 if ordinary == "native" else 5.0) / 4)
        for code, expected in expected_markers.items():
            flow = bd.get_node(database="biosphere", code=code)
            observed = float(calculation.inventory[calculation.dicts.biosphere[flow.id], :].sum())
            assert observed == pytest.approx(expected)
        # Root weights multiply complete normalized market rows.
        assert next(iter(delivery.production()))["amount"] == market_output
        assert fsum(row["amount"] for row in delivery.technosphere()) == pytest.approx(
            fsum(row["amount"] for row in source_rows if row["type"] == "technosphere")
        )
        assert [dict(row) for row in native_market.exchanges()] == source_rows
