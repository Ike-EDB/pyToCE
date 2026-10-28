"""Route actual mixed PA cohorts through complete detached receiving functions."""

from collections import Counter
from math import fsum
from pathlib import Path
from typing import cast

import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from infiltration_fixtures import infiltration_source_data, resolved_infiltration_provider
from pa_event_receiving_fixtures import mixed_pa_events as mixed_pa_events
from pa_event_receiving_fixtures import pa_event_receiving_boundary

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    prepared_provider_authority,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from pytoce.prepare_backgrounds.preparation.providers.route_authorities import (
    resolve_open_dump_routes,
)
from pytoce.run_lca.calculation.burdens import BurdenDemands
from pytoce.run_lca.calculation.pa_dump_receiving import PaDumpReceiving
from pytoce.run_lca.calculation.pa_event_receiving import (
    iter_pa_receiving,
    iter_pa_receiving_requests,
)
from pytoce.run_lca.calculation.pa_landfill_receiving import PaLandfillReceiving
from pytoce.run_lca.calculation.pa_municipal_receiving import PaMunicipalReceiving
from pytoce.run_lca.calculation.pa_open_burning_receiving import PaOpenBurningReceiving
from pytoce.run_lca.calculation.pa_receiving import (
    PaDisposalEvent,
    PaRecyclingEvent,
    PaSpinningEvent,
)
from pytoce.run_lca.calculation.pa_recycling_receiving import PaRecyclingReceiving
from pytoce.run_lca.calculation.pa_spinning_receiving import PaSpinningReceiving


@pytest.fixture(scope="module")
def receiving_anchors(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[bool, tuple[PaReceivingReference, PaReceivingSources, dict[str, PaReceivingActivity]]]:
    """Admit both complete source graphs once, then return portable records only."""
    records = {}
    for original in (True, False):
        with pa_event_receiving_boundary(
            tmp_path_factory.mktemp("actual_pa_original" if original else "actual_pa_future"),
            original=original,
        ) as boundary:
            reference, sources = boundary.reference, boundary.sources
            spinning = {}
            for grade in ("pa6", "pa66"):
                service = boundary.services[f"fibres_{grade}_conversion"]
                # The selected prepared identity identifies the operating source.
                spinning[grade] = PaReceivingActivity(
                    (
                        boundary.context.provider_database,
                        provider_service_code(str(service["provider_service_id"])),
                    ),
                    str(service["name"]),
                    str(service["reference_product"]),
                    str(service["unit"]),
                    str(service["location"]),
                )
        records[original] = reference, sources, spinning
    return records


@pytest.mark.parametrize("original", (True, False))
def test_actual_pa_events_reach_each_complete_receiving_on_their_physical_basis(
    tmp_path: Path,
    mixed_pa_events: BurdenDemands,
    receiving_anchors: dict[
        bool, tuple[PaReceivingReference, PaReceivingSources, dict[str, PaReceivingActivity]]
    ],
    original: bool,
) -> None:
    """Dispatch both grades and all actual fates after closing the source project.

    The complete original/future native graph is shared with its source-admission
    test. The five source shares are resolved through the real original market
    owner. This case tests event routing, whole-mixture quantities and source
    custody; chemical equations remain tested at their individual receivers.
    """
    provider = resolved_infiltration_provider(tmp_path / "class_source")
    dumping = provider.routes.loc[
        provider.routes["provider_demand_type"] == "mismanaged_open_dump_treatment"
    ]
    shares = dict(zip(dumping["infiltration_class"], dumping["infiltration_share"], strict=True))
    reference, sources, spinning = receiving_anchors[original]
    # The context is now closed. Every result is evaluated from portable native
    # records; activity dates stay 2019--2021 at either source anchor.
    outputs = tuple(
        iter_pa_receiving(
            spinning_events=mixed_pa_events.pa_spinning_events,
            recycling_events=mixed_pa_events.pa_recycling_events,
            disposal_events=mixed_pa_events.pa_disposal_events,
            reference=reference,
            sources=sources,
            spinning_providers=spinning,
            dump_shares=shares,
        )
    )
    expected_types = {
        "incineration_treatment": PaMunicipalReceiving,
        "mechanical_recycling_reject_incineration_treatment": PaMunicipalReceiving,
        "landfill_treatment": PaLandfillReceiving,
        "mismanaged_open_burning_treatment": PaOpenBurningReceiving,
        "mismanaged_open_dump_treatment": PaDumpReceiving,
    }
    assert {e.demand_type for e in mixed_pa_events.pa_disposal_events} == set(expected_types)
    assert {e.constituent for e, _, _ in outputs} == {
        "polyamide_6_resin",
        "polyamide_66_resin",
    }
    counts = Counter((type(event), getattr(event, "demand_type", "")) for event, _, _ in outputs)
    assert counts[PaSpinningEvent, ""] == len(mixed_pa_events.pa_spinning_events)
    assert counts[PaRecyclingEvent, ""] == len(mixed_pa_events.pa_recycling_events)
    for demand in expected_types:
        assert counts[PaDisposalEvent, demand] == sum(
            event.demand_type == demand for event in mixed_pa_events.pa_disposal_events
        ) * (5 if demand == "mismanaged_open_dump_treatment" else 1)
    identities = [
        (
            type(event),
            event.year,
            event.region,
            event.constituent,
            getattr(event, "demand_type", ""),
            infiltration,
        )
        for event, infiltration, _ in outputs
    ]
    assert len(set(identities)) == len(identities)
    for event, infiltration, result in outputs:
        assert event.region == "EUR"
        assert event.year in (2019, 2020, 2021)
        functions = {row.function for row in result.amounts}
        grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
        finish = _pa_finish_reference(grade)
        retained = fsum(finish.retained.compounds_kg.values())
        if isinstance(event, PaSpinningEvent):
            assert isinstance(result, PaSpinningReceiving)
            assert event in mixed_pa_events.pa_spinning_events
            assert result.water.maintenance_principal_kg == fsum(
                event.maintenance_principal_by_origin_kg
            )
            assert result.water.maintenance_embedded_dose_kg == event.maintenance_finish_dose_kg
            assert {row.source for row in result.amounts if row.function == "pa_spinning_site"} == {
                spinning[grade]
            }
            assert {
                "pa_spinning_site",
                "pa_maintenance_incineration",
                "pa_swiss_wastewater",
            } <= functions
        elif isinstance(event, PaRecyclingEvent):
            assert isinstance(result, PaRecyclingReceiving)
            assert event in mixed_pa_events.pa_recycling_events
            principal = fsum(event.principal_by_origin_kg)
            assert result.load.incoming_dry_kg == pytest.approx(
                principal + retained * (principal + event.embedded_finish_dose_kg)
            )
            assert {
                "pa_recycling_preparation",
                "pa_recycling_finishing",
                "pa_swiss_wastewater",
            } <= functions
        else:
            assert isinstance(result, expected_types[event.demand_type])
            wet = (
                fsum(event.principal_by_origin_kg)
                + retained * event.total_finish_dose_kg
                + event.water_kg
            )
            if isinstance(result, PaMunicipalReceiving):
                assert result.incineration.material.wet_mass_kg == pytest.approx(wet)
                assert result.incineration.material.water_kg == event.water_kg
                assert "pa_municipal_incineration" in functions
            elif isinstance(result, PaLandfillReceiving):
                assert result.load.wet_mass_kg == pytest.approx(wet)
                assert result.load.water_kg == event.water_kg
                assert {"pa_sanitary_landfill", "pa_sanitary_wastewater"} <= functions
            elif isinstance(result, PaOpenBurningReceiving):
                assert result.load.wet_mass_kg == pytest.approx(wet)
                assert result.load.incoming_water_kg == event.water_kg
                assert all(isinstance(row.endpoint, PaReceivingFlow) for row in result.amounts)
                assert {row.source for row in result.amounts} == {reference.open_burning.activity}
            elif isinstance(result, PaDumpReceiving):
                assert result.load.wet_mass_kg == pytest.approx(wet)
                assert result.load.water_kg == event.water_kg
                assert functions == {"pa_open_dump"}
                assert all(isinstance(row.endpoint, PaReceivingFlow) for row in result.amounts)
                assert {row.source for row in result.amounts} == {
                    next(
                        site.placement.activity
                        for site in reference.dump.sites
                        if site.infiltration_class == infiltration
                    )
                }
            if event.demand_type != "mismanaged_open_dump_treatment":
                assert event in mixed_pa_events.pa_disposal_events
        if not isinstance(result, PaDumpReceiving):
            assert infiltration == ""
        assert not any(row.endpoint in spinning.values() for row in result.amounts)
    for parent in mixed_pa_events.pa_disposal_events:
        if parent.demand_type != "mismanaged_open_dump_treatment":
            continue
        children = [
            (event, infiltration, result)
            for event, infiltration, result in outputs
            if isinstance(event, PaDisposalEvent)
            and isinstance(result, PaDumpReceiving)
            and (event.year, event.region, event.constituent)
            == (parent.year, parent.region, parent.constituent)
        ]
        assert [infiltration for _, infiltration, _ in children] == [
            site.infiltration_class for site in reference.dump.sites
        ]
        for child, infiltration, _ in children:
            share = shares[infiltration]
            assert child.principal_by_origin_kg == pytest.approx(
                tuple(value * share for value in parent.principal_by_origin_kg)
            )
            assert child.total_finish_dose_kg == pytest.approx(parent.total_finish_dose_kg * share)
            assert child.water_kg == pytest.approx(parent.water_kg * share)
        assert tuple(
            fsum(child.principal_by_origin_kg[index] for child, _, _ in children)
            for index in range(3)
        ) == pytest.approx(parent.principal_by_origin_kg)
        assert fsum(child.total_finish_dose_kg for child, _, _ in children) == pytest.approx(
            parent.total_finish_dose_kg
        )
        assert fsum(child.water_kg for child, _, _ in children) == pytest.approx(parent.water_kg)


def test_zero_native_dump_class_is_an_absent_stream(
    tmp_path: Path,
    mixed_pa_events: BurdenDemands,
    receiving_anchors: dict[
        bool, tuple[PaReceivingReference, PaReceivingSources, dict[str, PaReceivingActivity]]
    ],
) -> None:
    """Admit a real five-row source portfolio with one explicit zero input.

    This is a valid external market boundary.
    The prior real MFA events and complete detached receivers are reused; only
    the selected market's class amounts differ from the first test's source.
    """
    provider = prepared_provider_authority()
    names = BrightwayNames(tmp_path / "brightway", "zero_class", "biosphere", "class_source")
    records = infiltration_source_data(names.source_database)
    market = records[names.source_database, "pe_waste_market"]
    # The existing fixture's first of five native treatment inputs is
    # zero; its declared activity still exists. Other four inputs remain intact.
    exchanges = cast(list[dict[str, object]], market["exchanges"])
    exchanges[1]["amount"] = 0.0
    with selected_brightway_project(names) as bd:
        bd.Database(names.source_database).write(records, searchable=False)
        routes, _ = resolve_open_dump_routes(names.source_database, provider.routes)
    dumping = routes.loc[routes["provider_demand_type"] == "mismanaged_open_dump_treatment"]
    shares = dict(zip(dumping["infiltration_class"], dumping["infiltration_share"], strict=True))
    assert len(shares) == 5
    assert sum(value == 0 for value in shares.values()) == 1
    reference, sources, spinning = receiving_anchors[True]
    events = tuple(
        event
        for event in mixed_pa_events.pa_disposal_events
        if event.demand_type == "mismanaged_open_dump_treatment"
    )
    outputs = tuple(
        iter_pa_receiving(
            spinning_events=(),
            recycling_events=(),
            disposal_events=events,
            reference=reference,
            sources=sources,
            spinning_providers=spinning,
            dump_shares=shares,
        )
    )
    assert len(outputs) == 4 * len(events)
    assert all(shares[infiltration] > 0 for _, infiltration, _ in outputs)
    requests = tuple(
        iter_pa_receiving_requests(
            spinning_events=(),
            recycling_events=(),
            disposal_events=tuple(enumerate(events)),
            reference=reference,
            dump_shares=shares,
        )
    )
    assert [(request.event, request.infiltration_class) for request in requests] == [
        (event, infiltration) for event, infiltration, _ in outputs
    ]
    assert [request.disposal_index for request in requests] == [
        index for index in range(len(events)) for _ in range(4)
    ]
    for parent in events:
        children = [
            event
            for event, _, _ in outputs
            if isinstance(event, PaDisposalEvent) and event.constituent == parent.constituent
        ]
        assert tuple(
            fsum(event.principal_by_origin_kg[i] for event in children) for i in range(3)
        ) == pytest.approx(parent.principal_by_origin_kg)
        assert fsum(event.total_finish_dose_kg for event in children) == pytest.approx(
            parent.total_finish_dose_kg
        )
        assert fsum(event.water_kg for event in children) == pytest.approx(parent.water_kg)
