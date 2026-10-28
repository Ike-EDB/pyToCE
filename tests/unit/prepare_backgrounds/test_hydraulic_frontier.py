"""Retain complete dated hydraulic receiving and preserve native data."""

from dataclasses import replace
from pathlib import Path

import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from water_treatment_fixtures import water_biosphere, water_native_inventory

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    prepared_provider_authority,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)


@pytest.mark.parametrize("original", [True, False], ids=["original", "future"])
def test_hydraulic_frontier_preserves_native_records_and_signed_receiving(
    tmp_path: Path, original: bool
) -> None:
    """Persist the real hydraulic pair and solve both signed nonunit functions.

    The small native inputs include different dated electricity suppliers.
    Their metadata and exchanges remain unchanged when the separate package
    frontier is written. No study node is appended to either native database.
    """
    full = prepared_provider_authority()
    service_ids = ("hydraulic_water_collection_row", "hydraulic_water_treatment_row")
    provider = replace(
        full,
        catalogue={
            **full.catalogue,
            "services": [
                row
                for row in full.catalogue["services"]
                if row["provider_service_id"] in service_ids
            ],
            "routes": [],
        },
        routes=full.routes.iloc[:0],
    )
    names = BrightwayNames(tmp_path / "brightway", "hydraulic_frontier", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(water_biosphere(), searchable=False)
        bd.Database("original").write(
            water_native_inventory(full.catalogue, "original", original=True), searchable=False
        )
        native_database = "original" if original else "future"
        if not original:
            bd.Database("future").write(
                water_native_inventory(full.catalogue, "future", original=False), searchable=False
            )
        before = {
            node.key: (dict(node), [(row.id, dict(row)) for row in node.exchanges()])
            for database in {"original", native_database}
            for node in bd.Database(database)
        }
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2025 if original else 2050,
            ordinary_database=native_database,
            technology_database=native_database,
            provider=provider,
            material_recipe=physical_material_recipe(),
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )
        collection, plant = (
            bd.get_node(database=frontier.database_name, code=provider_service_code(service))
            for service in service_ids
        )
        assert collection.rp_exchange()["amount"] == -2.0
        assert plant.rp_exchange()["amount"] == -4.0
        assert (
            next(row["amount"] for row in collection.technosphere() if row.input == plant) == -0.5
        )
        calculation = LCA({collection: -0.001})
        calculation.lci()
        # The invented collection treats 0.5/2 of one m3. Brightway applies
        # the child's four-m3 production exactly once to that demand.
        for node, expected in ((collection, 0.001 / 2), (plant, 0.001 * 0.25 / 4)):
            assert calculation.supply_array[calculation.dicts.activity[node.id]] == pytest.approx(
                expected
            )
        assert {
            node.key: (dict(node), [(row.id, dict(row)) for row in node.exchanges()])
            for database in {"original", native_database}
            for node in bd.Database(database)
        } == before
