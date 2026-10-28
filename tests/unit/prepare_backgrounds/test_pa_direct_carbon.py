"""Actual receiving gases retain principal origin and their source basis."""

import json
from collections import defaultdict
from dataclasses import replace
from math import fsum
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from infiltration_fixtures import resolved_infiltration_provider
from pa_event_receiving_fixtures import mixed_pa_events as mixed_pa_events
from pa_event_receiving_fixtures import pa_event_receiving_boundary

from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    sources_from_payload,
    sources_to_payload,
)
from pytoce.run_lca.calculation.burdens import BurdenDemands
from pytoce.run_lca.calculation.carbon import pa_direct_carbon_flows
from pytoce.run_lca.calculation.pa_projection import iter_pa_projection
from pytoce.run_lca.calculation.pa_receiving import PaRecyclingEvent, PaSpinningEvent


def test_actual_direct_carbon_preserves_mixed_origins_and_dated_function_quantities(
    tmp_path: Path, mixed_pa_events: BurdenDemands
) -> None:
    """Use real cohorts and complete native graphs, then report after store closure.

    One latest event per grade and duty preserves actual inherited material and
    mixed principal origins. As in the projection boundary test, relocating
    these loads to 2035 tests the 2025/2050 source interpolation.
    """
    references, sources = {}, {}
    for original in (True, False):
        with pa_event_receiving_boundary(tmp_path / str(original), original=original) as boundary:
            anchor = 2025 if original else 2050
            references[anchor] = boundary.reference
            # The prepared source owns both converter provenance descriptors.
            # JSON decoding preserves them for receiving after project closure.
            payload = json.loads(json.dumps(sources_to_payload(boundary.sources), allow_nan=False))
            sources[anchor] = sources_from_payload(payload)
            assert sources[anchor] == boundary.sources
            assert set(sources[anchor].spinning_providers) == {"pa6", "pa66"}
    provider = resolved_infiltration_provider(tmp_path / "classes")
    dumping = provider.routes.loc[
        provider.routes["provider_demand_type"].eq("mismanaged_open_dump_treatment")
    ]
    shares = dict(zip(dumping["infiltration_class"], dumping["infiltration_share"], strict=True))
    # Input records follow increasing activity year; coordinate replacement retains the latest load.
    spun = {event.constituent: event for event in mixed_pa_events.pa_spinning_events}
    recycled = {event.constituent: event for event in mixed_pa_events.pa_recycling_events}
    disposed = {
        (event.constituent, event.demand_type): event
        for event in mixed_pa_events.pa_disposal_events
    }
    projected = tuple(
        iter_pa_projection(
            spinning_events=tuple(replace(event, year=2035) for event in spun.values()),
            recycling_events=tuple(replace(event, year=2035) for event in recycled.values()),
            disposal_events=tuple(replace(event, year=2035) for event in disposed.values()),
            reference=references[2025],
            sources_by_year=sources,
            matrix_by_year={2025: "original_matrix", 2050: "future_matrix"},
            dump_shares=shares,
        )
    )
    reported = pa_direct_carbon_flows(iter(projected))
    assert not reported.empty
    assert set(reported["constituent"]) == {"polyamide_6_resin", "polyamide_66_resin"}
    assert set(reported["carbon_species"]) == {"carbon dioxide", "methane", "carbon monoxide"}
    assert set(reported["mfa_carbon_origin"]) == {"fossil", "biogenic", "atmospheric_dac"}
    assert set(reported["source_anchor_year"]) == {2025, 2050}
    assert set(reported["source_database"]) == {"original", "future"}
    assert set(reported["matrix_database"]) == {"original_matrix", "future_matrix"}
    assert set(reported["year"]) == {2035}
    assert set(reported["h12_region"]) == {"EUR"}
    assert set(reported["material"]) == {"Fibres"}
    assert set(reported["product"]) == {"polymer"}
    assert set(reported["good"]) == {"all"}
    assert set(reported["virgin_feedstock_route"]) == {""}
    assert set(reported["source_role"]) == {"modelled_direct_pa_receiving"}
    assert set(reported["mfa_carbon_boundary"]) == {"actual_pa_principal_receiving_outputs"}
    assert set(reported["flow_unit"]) == {"kilogram"}
    assert set(reported["carbon_unit"]) == {"kilogram carbon"}
    assert (
        reported[
            [
                "provider_database",
                "provider_code",
                "provider_anchor_year",
                "provider_demand_amount",
                "owner_database",
                "owner_code",
                "owner_name",
                "owner_index",
                "owner_scale_per_provider_unit",
                "flow_amount_per_provider_unit",
            ]
        ]
        .isna()
        .all()
        .all()
    )

    # Independently sum actual native rows rather than using the grouping
    # owner under test. No inlet-origin share or extra provider demand enters.
    factors = {"Carbon dioxide": 12 / 44, "Methane": 12 / 16, "Carbon monoxide": 12 / 28}
    origins = ("fossil", "biogenic", "atmospheric_dac")
    nonprincipal_carbon = []
    seen_rows = 0
    for projection in projected:
        event = projection.event
        if isinstance(event, PaSpinningEvent):
            stage = "processing_conversion"
            grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
            converter = sources[projection.source_anchor_year].spinning_providers[grade]
            assert any(row.source == converter for row in projection.receiving.amounts)
        elif isinstance(event, PaRecyclingEvent):
            stage = "mechanical_recycling_operation"
        else:
            stage = event.demand_type
        expected: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
        total: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        authorities: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
        for row in projection.receiving.amounts:
            flow = row.endpoint
            if not isinstance(flow, PaReceivingFlow) or flow.unit != "kilogram":
                continue
            species = flow.name.split(",", 1)[0]
            if species not in factors:
                continue
            key = (row.function, *flow.key)
            total[key].append(row.amount)
            authorities[key].add(row.source.key)
            fractions = row.principal_carbon_origin_fractions
            if fractions is not None:
                for origin, fraction in zip(origins, fractions, strict=True):
                    expected[(*key, origin)].append(row.amount * fraction)
            nonprincipal_carbon.append(
                row.amount
                * (1 - fsum(fractions or ()))
                * factors[species]
                * projection.temporal_share
            )
        actual = reported.loc[
            reported["constituent"].eq(event.constituent)
            & reported["life_cycle_stage"].eq(stage)
            & reported["source_anchor_year"].eq(projection.source_anchor_year)
            & reported["infiltration_class"].eq(projection.infiltration_class)
        ]
        expected = {key: values for key, values in expected.items() if fsum(values) != 0}
        assert len(actual) == len(expected)
        seen_rows += len(actual)
        for row in actual.itertuples(index=False):
            key = (
                cast(str, row.receiving_function),
                cast(str, row.flow_database),
                cast(str, row.flow_code),
            )
            principal = fsum(expected[(*key, cast(str, row.mfa_carbon_origin))])
            whole = fsum(total[key])
            assert row.flow_amount == pytest.approx(principal * projection.temporal_share)
            assert row.receiving_flow_amount == pytest.approx(whole * projection.temporal_share)
            assert row.principal_carbon_fraction == pytest.approx(principal / whole)
            assert row.carbon_amount == pytest.approx(
                cast(float, row.flow_amount) * factors[cast(str, row.flow_name).split(",", 1)[0]]
            )
            assert row.temporal_share == {2025: 0.6, 2050: 0.4}[cast(int, row.source_anchor_year)]
            assert row.life_cycle_stage == stage
            assert row.physical_event_basis
            assert {
                (source["database"], source["code"])
                for source in json.loads(cast(str, row.source_records_json))
            } == authorities[key]
    assert seen_rows == len(reported)
    assert fsum(nonprincipal_carbon) > 0
    assert reported["receiving_function"].eq("pa_maintenance_incineration").any()
    # One non-fossil endpoint can carry principal biogenic and DAC carbon with separate origin
    # attribution.
    nonfossil = reported.loc[reported["flow_carbon_origin"].eq("non_fossil")]
    assert {"biogenic", "atmospheric_dac"} <= set(nonfossil["mfa_carbon_origin"])
    assert nonfossil["principal_carbon_fraction"].lt(1).any()
    assert set(reported.loc[reported["lca_fate"].eq("open_dump"), "infiltration_class"]) == set(
        shares
    )
    # Temporal scaling changes quantities and preserves identity and output fractions.
    unweighted = pa_direct_carbon_flows(replace(row, temporal_share=1.0) for row in projected)
    pd.testing.assert_series_equal(
        reported["principal_carbon_fraction"], unweighted["principal_carbon_fraction"]
    )
    assert reported["carbon_amount"].to_numpy() == pytest.approx(
        unweighted["carbon_amount"].to_numpy() * reported["temporal_share"].to_numpy()
    )


def test_no_pa_events_has_no_modelled_principal_gas_rows() -> None:
    """An empty PA event set produces an empty PA receiving inventory."""
    reported = pa_direct_carbon_flows(iter(()))
    assert reported.empty
    assert {"carbon_amount", "source_records_json", "receiving_function"} <= set(reported.columns)
