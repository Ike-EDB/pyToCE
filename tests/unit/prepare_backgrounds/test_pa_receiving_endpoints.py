"""Cover PA receiving on the finite source axes with replaced children supplied separately."""

from pathlib import Path

import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from infiltration_fixtures import resolved_infiltration_provider
from pa_event_receiving_fixtures import mixed_pa_events as mixed_pa_events
from pa_event_receiving_fixtures import pa_event_receiving_boundary

from pytoce.prepare_backgrounds.preparation.providers.catalogue import provider_service_code
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from pytoce.run_lca.calculation.burdens import BurdenDemands
from pytoce.run_lca.calculation.pa_event_receiving import iter_pa_receiving
from pytoce.run_lca.calculation.pa_receiving_endpoints import pa_receiving_endpoints


@pytest.mark.parametrize("original", (True, False))
def test_finite_endpoints_cover_actual_receivers_and_keep_local_replacement_boundaries(
    tmp_path: Path,
    mixed_pa_events: BurdenDemands,
    original: bool,
) -> None:
    """Use the complete admitted graph and genuine routed cohorts after store closure.

    Endpoint enumeration is independent of the test events: these witness the
    actual outputs, while named supplier checks also protect conditional equipment
    and forbid adding calibration fuels merely because they appear in a source.
    """

    with pa_event_receiving_boundary(tmp_path / "native", original=original) as boundary:
        reference, sources = boundary.reference, boundary.sources
        spinning = {}
        for grade in ("pa6", "pa66"):
            service = boundary.services[f"fibres_{grade}_conversion"]
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
    suppliers, flows = pa_receiving_endpoints(reference, sources)
    supplier_keys = {endpoint.key for endpoint in suppliers}
    flow_keys = {endpoint.key for endpoint in flows}
    assert [endpoint.key for endpoint in suppliers] == sorted(supplier_keys)
    assert [endpoint.key for endpoint in flows] == sorted(flow_keys)
    assert all(endpoint.unit for endpoint in (*suppliers, *flows))

    provider = resolved_infiltration_provider(tmp_path / "dump_source")
    dumping = provider.routes.loc[
        provider.routes["provider_demand_type"].eq("mismanaged_open_dump_treatment")
    ]
    shares = dict(zip(dumping["infiltration_class"], dumping["infiltration_share"], strict=True))
    actual_native, actual_flows = set(), set()
    for _, _, received in iter_pa_receiving(
        spinning_events=mixed_pa_events.pa_spinning_events,
        recycling_events=mixed_pa_events.pa_recycling_events,
        disposal_events=mixed_pa_events.pa_disposal_events,
        reference=reference,
        sources=sources,
        spinning_providers=spinning,
        dump_shares=shares,
    ):
        for amount in received.amounts:
            if isinstance(amount.endpoint, PaReceivingActivity):
                actual_native.add(amount.endpoint.key)
            else:
                actual_flows.add(amount.endpoint.key)
    assert actual_native <= supplier_keys
    assert actual_flows <= flow_keys

    # Complete suppliers preserve native units and signed waste treatment demand.
    required = (
        sources.pam,
        sources.water.pam,
        sources.landfill.pam,
        sources.municipal.pam,
    )
    assert {endpoint.key for endpoint in required} <= supplier_keys
    for product in (
        "metalliferous hydroxide sludge",
        "titanium dioxide",
        "chromium",
        "chromium trioxide, flakes",
    ):
        assert product in {endpoint.reference_product for endpoint in suppliers}
    assert reference.gas.chp.fuel.key not in supplier_keys
    assert reference.gas.furnace.fuel.key not in supplier_keys
    assert (
        not {
            sources.hwi.activity.key,
            sources.fixed.activity.key,
            sources.gas.chp.activity.key,
            sources.gas.furnace.activity.key,
        }
        & supplier_keys
    )
    assert not {endpoint.key for endpoint in spinning.values()} & supplier_keys

    recycling = sources.recycling
    preparation_children = {
        recycling.preparation.rows[index].endpoint.key
        for index in recycling.preparation_wastewater_rows
    }
    finishing_water = {
        row.endpoint.key
        for row in recycling.finishing.rows
        if isinstance(row.endpoint, PaReceivingActivity)
        and (
            row.endpoint.reference_product.startswith("water")
            or row.endpoint.reference_product.startswith("wastewater")
        )
    }
    assert finishing_water
    assert finishing_water <= supplier_keys
    assert not (preparation_children - finishing_water) & supplier_keys
    assert recycling.pellet_delivery.key in supplier_keys
    for dated in (sources.water, sources.landfill, sources.municipal):
        for market, terminal in (
            (dated.bottom_ash_market, dated.bottom_ash_treatment),
            (dated.residual_ash_market, dated.residual_ash_treatment),
        ):
            assert terminal.activity.key not in supplier_keys
            # Every actual external haul and terminal support remains available.
            assert {
                row.endpoint.key for row in market.rows if row.endpoint.key != terminal.activity.key
            } <= supplier_keys
            assert {row.endpoint.key for row in terminal.rows} <= supplier_keys

    # A characterized inventory cannot discover these absent-coefficient or
    # alternate-compartment outputs. Admission supplies them independently.
    declared_flows = (
        *reference.open_burning.biosphere,
        *reference.dump.biosphere,
        *sources.added_biosphere,
        *sources.gas.biosphere,
        *sources.water.biosphere,
        *sources.landfill.biosphere,
        *sources.municipal.biosphere,
    )
    assert {flow.key for flow in declared_flows} <= flow_keys
    for calibrated in (reference.water, reference.landfill):
        for operation in (
            calibrated.sludge.operation,
            calibrated.landfarming.operation,
            calibrated.bottom_ash.operation,
            calibrated.residual_ash.operation,
        ):
            assert {
                row.endpoint.key
                for row in operation.rows
                if isinstance(row.endpoint, PaReceivingFlow)
            } <= flow_keys
    assert {
        row.endpoint.key for site in reference.dump.sites for row in site.placement.rows
    } <= flow_keys
