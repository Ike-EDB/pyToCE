"""Actual PA quantities keep their process and general-anchor basis."""

import json
from dataclasses import replace
from math import fsum
from pathlib import Path

import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from infiltration_fixtures import resolved_infiltration_provider
from pa_event_receiving_fixtures import mixed_pa_events as mixed_pa_events
from pa_event_receiving_fixtures import pa_event_receiving_boundary

from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from pytoce.run_lca.calculation.burdens import BurdenDemands
from pytoce.run_lca.calculation.pa_landfill_receiving import PaLandfillReceiving
from pytoce.run_lca.calculation.pa_projection import (
    group_pa_receiving_amounts,
    group_pa_receiving_quantities,
    iter_pa_projection,
    iter_pa_projection_requests,
    pa_source_weights,
)
from pytoce.run_lca.calculation.pa_receiving import PaReceivingAmount


@pytest.mark.parametrize(
    "year,expected",
    (
        (2019, ((2025, 1.0),)),
        (2025, ((2025, 1.0),)),
        (2026, ((2025, 0.8), (2030, 0.2))),
        (2029, ((2025, 0.2), (2030, 0.8))),
        (2030, ((2030, 1.0),)),
        (2042, ((2040, 0.6), (2045, 0.4))),
        (2100, ((2100, 1.0),)),
    ),
)
def test_actual_receiving_uses_general_background_years(year, expected):
    """Use original 2025 through 2025, then the matching future interval once."""
    actual = pa_source_weights(year, tuple(range(2025, 2101, 5)))
    assert [anchor for anchor, _ in actual] == [anchor for anchor, _ in expected]
    assert [share for _, share in actual] == pytest.approx([share for _, share in expected])
    assert fsum(share for _, share in actual) == 1.0


def test_receiving_groups_physical_endpoints_without_splitting_origins_or_sources():
    """Keep actual mixed emissions together and distinct gas functions separate.

    These small synthetic records use kg CO2 and m3 water. Metadata describes
    two source authorities. The grouped quantity reports the sum of dated inputs.
    """
    original = PaReceivingActivity(("original", "source"), "original", "gas", "kilogram", "RoW")
    dated = PaReceivingActivity(("future", "source"), "dated", "gas", "megajoule", "RoW")
    carbon = PaReceivingFlow(
        ("biosphere", "co2"), "Carbon dioxide, non-fossil", "kilogram", ("air",)
    )
    water = PaReceivingFlow(("biosphere", "water"), "Water", "cubic meter", ("water",))
    supply = PaReceivingActivity(
        ("future", "power"), "power", "electricity", "kilowatt hour", "RoW"
    )
    rows = (
        PaReceivingAmount("chp", original, carbon, 2.0, {"loc": 2.0}, (0.0, 1.0, 0.0)),
        PaReceivingAmount("chp", dated, carbon, 3.0, {"loc": 3.0}, (0.0, 0.0, 1.0)),
        PaReceivingAmount("chp", original, carbon, 1.0, {"loc": 1.0}, (0.0, 0.0, 0.0)),
        PaReceivingAmount("flare", dated, carbon, 4.0, {"loc": 4.0}, (0.0, 0.0, 1.0)),
        PaReceivingAmount("chp", dated, water, 0.002, {}),
        PaReceivingAmount("chp", dated, water, 0.003, {}),
        PaReceivingAmount("chp", dated, supply, 0.0, {}),
        PaReceivingAmount("cancellation", original, supply, 1e16, {}),
        PaReceivingAmount("cancellation", dated, supply, 1.0, {}),
        PaReceivingAmount("cancellation", original, supply, -1e16, {}),
    )
    quantities = tuple(group_pa_receiving_quantities(rows))
    attributed = tuple(group_pa_receiving_amounts(rows))
    assert [(row.receiving_function, row.endpoint, row.amount) for row in quantities] == [
        (row.receiving_function, row.endpoint, row.amount) for row in attributed
    ]
    assert [(row.receiving_function, row.endpoint.key) for row in quantities] == [
        ("cancellation", supply.key),
        ("chp", carbon.key),
        ("chp", water.key),
        ("chp", supply.key),
        ("flare", carbon.key),
    ]
    groups = {(row.receiving_function, row.endpoint.key): row for row in attributed}
    assert len(groups) == 5
    assert groups["cancellation", supply.key].amount == 1.0
    mixed = groups["chp", carbon.key]
    assert mixed.amount == 6.0
    assert mixed.principal_amounts == (0.0, 2.0, 3.0)
    assert json.loads(mixed.source_records_json) == [
        {
            "database": "future",
            "code": "source",
            "name": "dated",
            "location": "RoW",
            "unit": "megajoule",
        },
        {
            "database": "original",
            "code": "source",
            "name": "original",
            "location": "RoW",
            "unit": "kilogram",
        },
    ]
    assert groups["flare", carbon.key].amount == 4.0
    assert groups["chp", water.key].amount == 0.005
    assert groups["chp", water.key].endpoint.unit == "cubic meter"
    assert groups["chp", supply.key].amount == 0.0
    assert groups["chp", supply.key].principal_amounts == (0.0, 0.0, 0.0)


def test_actual_mixed_receiving_precedes_source_interpolation(
    tmp_path: Path, mixed_pa_events: BurdenDemands
) -> None:
    """Use complete source graphs, then interpolate actual detached PA receiving.

    This validation fixture has original 2025 and dated 2050 endpoints. Shifting
    the admitted activity to 2035 tests the projection boundary only;
    no stock/cohort calculation is inferred from that relabeled validation year.
    """
    records = {}
    for original in (True, False):
        with pa_event_receiving_boundary(tmp_path / str(original), original=original) as boundary:
            records[2025 if original else 2050] = boundary.reference, boundary.sources
    reference = records[2025][0]
    sources = {year: record[1] for year, record in records.items()}
    provider = resolved_infiltration_provider(tmp_path / "class_source")
    dumping = provider.routes.loc[
        provider.routes["provider_demand_type"] == "mismanaged_open_dump_treatment"
    ]
    shares = dict(zip(dumping["infiltration_class"], dumping["infiltration_share"], strict=True))
    parent = next(
        event
        for event in mixed_pa_events.pa_disposal_events
        if event.demand_type == "landfill_treatment" and event.constituent == "polyamide_6_resin"
    )
    event = replace(parent, year=2035)
    rows = tuple(
        iter_pa_projection(
            spinning_events=(),
            recycling_events=(),
            disposal_events=(event,),
            reference=reference,
            sources_by_year=sources,
            matrix_by_year={2025: "original_matrix", 2050: "future_matrix"},
            dump_shares=shares,
        )
    )
    assert len(rows) == 2
    assert [row.source_anchor_year for row in rows] == [2025, 2050]
    assert [row.temporal_share for row in rows] == [0.6, 0.4]
    assert [row.source_database for row in rows] == ["original", "future"]
    assert [row.matrix_database for row in rows] == ["original_matrix", "future_matrix"]
    for row in rows:
        assert row.event == event
        assert row.infiltration_class == ""
        assert isinstance(row.receiving, PaLandfillReceiving)
        operation = sources[row.source_anchor_year].landfill.landfill
        supply = next(
            item
            for item in operation.rows
            if isinstance(item.endpoint, PaReceivingActivity)
            and item.endpoint.reference_product == "electricity, low voltage"
        )
        purchased = next(
            item
            for item in row.receiving.amounts
            if item.function == "pa_sanitary_landfill" and item.endpoint == supply.endpoint
        )
        # The source's specific operating meter applies to actual wet waste,
        # before the 0.6/0.4 temporal shares. No interpolated source is built.
        assert purchased.amount == pytest.approx(supply.amount * row.receiving.load.wet_mass_kg)
    assert isinstance(rows[0].receiving, PaLandfillReceiving)
    assert isinstance(rows[1].receiving, PaLandfillReceiving)
    assert rows[0].receiving.load == rows[1].receiving.load
    assert rows[0].receiving.amounts != rows[1].receiving.amounts

    # Activity records interleave anchor dates, interpolation, and a duplicate occurrence. Original
    # tuple positions identify each event through filtering.
    burning = next(
        item
        for item in mixed_pa_events.pa_disposal_events
        if item.demand_type == "mismanaged_open_burning_treatment"
    )
    dump = next(
        item
        for item in mixed_pa_events.pa_disposal_events
        if item.demand_type == "mismanaged_open_dump_treatment"
    )
    disposal = (
        replace(dump, year=2050),
        replace(burning, year=2025),
        event,
        replace(dump, year=2035),
        replace(dump, year=2035),
    )
    requests = tuple(
        iter_pa_projection_requests(
            spinning_events=(),
            recycling_events=(),
            disposal_events=disposal,
            reference=reference,
            sources_by_year=sources,
            matrix_by_year={2025: "original_matrix", 2050: "future_matrix"},
            dump_shares=shares,
        )
    )
    complete = tuple(
        iter_pa_projection(
            spinning_events=(),
            recycling_events=(),
            disposal_events=disposal,
            reference=reference,
            sources_by_year=sources,
            matrix_by_year={2025: "original_matrix", 2050: "future_matrix"},
            dump_shares=shares,
        )
    )
    assert [
        (
            row.receiving_request.event,
            row.receiving_request.infiltration_class,
            row.source_anchor_year,
            row.source_database,
            row.matrix_database,
            row.temporal_share,
        )
        for row in requests
    ] == [
        (
            row.event,
            row.infiltration_class,
            row.source_anchor_year,
            row.source_database,
            row.matrix_database,
            row.temporal_share,
        )
        for row in complete
    ]
    assert [row.receiving_request.disposal_index for row in requests] == [
        1,
        2,
        *([3] * 5),
        *([4] * 5),
        *([0] * 5),
        2,
        *([3] * 5),
        *([4] * 5),
    ]
