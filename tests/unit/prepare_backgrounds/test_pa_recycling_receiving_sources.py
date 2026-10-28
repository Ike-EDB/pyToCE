"""Admit real completed recycling graphs with invented original and dated rows."""

import json
from dataclasses import asdict
from math import fsum, log
from pathlib import Path
from typing import Any, cast

import pytest
from material_recipe_fixtures import physical_material_recipe
from pa_recycling_receiving_fixtures import (
    RECYCLING_SERVICES,
    recycling_biosphere,
    recycling_native_inventory,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    pa_recycling_receiving_sources as recycling,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_source_closures,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


@pytest.mark.parametrize("future", (False, True))
def test_recycling_admission_preserves_complete_operations_and_distinct_physical_bases(
    tmp_path: Path, future: bool
) -> None:
    """Use actual source compilation, persisted rows and detached original calibration."""
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    selected = set(RECYCLING_SERVICES)
    services = {
        item["provider_service_id"]: item
        for item in catalogue["services"]
        if item["provider_service_id"] in selected
    }
    names = BrightwayNames(tmp_path / "brightway", "pa_recycling", "biosphere", "original")
    database = "future" if future else "original"
    with selected_brightway_project(names) as bd:
        biosphere = recycling_biosphere()
        bd.Database("biosphere").write(biosphere, searchable=False)
        bd.Database("original").write(
            recycling_native_inventory(catalogue, "original", future=False), searchable=False
        )
        if future:
            bd.Database("future").write(
                recycling_native_inventory(catalogue, "future", future=True), searchable=False
            )
        native = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=native,
            technology=native,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        original = recycling.admit_pa_recycling_receiving_reference(context)
        compiled = {
            ("providers", provider_service_code(service_id)): _service_dataset(
                context, service, services, "providers"
            )
            for service_id, service in services.items()
        }
        bd.Database("providers").write(compiled, searchable=False)
        admitted = recycling.admit_pa_recycling_receiving_sources(context)
        # One complete cargo-delivery endpoint retains the fixture's nonunit
        # production and freight distribution inside its prepared graph.
        assert admitted.pellet_delivery.key == (
            "providers",
            provider_service_code("recycling_pet_pellet_delivery"),
        )
        delivery = bd.get_node(
            database=admitted.pellet_delivery.key[0], code=admitted.pellet_delivery.key[1]
        )
        (production,) = delivery.production()
        assert production["amount"] == 4
        (freight,) = delivery.technosphere()
        assert freight.input.key == (database, "freight")
        assert freight["amount"] == pytest.approx(0.12)
        assert freight["loc"] == pytest.approx(log(0.12) + 0.02)
        assert freight["scale"] == 0.13
        original_payload = json.loads(json.dumps(asdict(original), allow_nan=False))
        dated_payload = json.loads(json.dumps(asdict(admitted), allow_nan=False))

    reference = recycling.recycling_reference_from_payload(original_payload)
    sources = recycling.recycling_sources_from_payload(dated_payload)
    assert reference == original
    assert sources == admitted
    assert reference.preparation.key == ("original", "recycling_weee_preparation")
    # The fixture's 3.2 kg dry feed per two reference kg gives a normalized basis of 1.6.
    assert reference.dry_feed_kg_per_reference_kg == 3.2 / 2
    assert reference.wash_water_kg_per_reference_kg == pytest.approx(4.8)
    assert (
        reference.wash_water_kg_per_reference_kg / reference.dry_feed_kg_per_reference_kg
        == pytest.approx(3.0)
    )
    production = 5 if future else 2
    assert sources.preparation.production_amount == production
    assert sources.finishing.production_amount == production
    assert sources.preparation_pm_rows == (5, 6)
    assert sources.finishing_pm_rows == (4,)
    assert sources.preparation_water_rows == (1, 2)
    assert sources.preparation_wastewater_rows == (3,)
    assert len(sources.preparation.rows) == 7
    assert len(sources.finishing.rows) == 5
    for operation in (sources.preparation, sources.finishing):
        for row in operation.rows:
            assert row.endpoint.key[0] == (
                database if isinstance(row.endpoint, PaReceivingActivity) else "biosphere"
            )
            assert row.metadata["loc"] == pytest.approx(log(abs(row.amount)) + 0.02)
            assert row.metadata["scale"] == 0.13
            assert row.metadata["negative"] == (row.amount < 0)
    assert fsum(
        sources.preparation.rows[i].amount for i in sources.preparation_water_rows
    ) == pytest.approx(6 if future else 4.8)
    assert [
        sources.preparation.rows[i].amount for i in sources.preparation_pm_rows
    ] == pytest.approx([0.00024, 0.00016])
    assert sources.finishing.rows[sources.finishing_pm_rows[0]].amount == 0.002
    cooling = sources.finishing.rows[2]
    assert isinstance(cooling.endpoint, PaReceivingActivity)
    assert cooling.endpoint.reference_product == "wastewater from PET pelletising"
    assert cooling.amount == pytest.approx(-0.0004 / 1000)
    assert sources.evaporated_water == PaReceivingFlow(
        ("biosphere", "water"), "Water", "cubic meter", ("air",)
    )
