"""Actual PA receiving enters the existing native solve with its own units."""

import json
from dataclasses import asdict, replace
from math import fsum
from pathlib import Path
from typing import Any, cast

import bw2calc as bc
import pandas as pd
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from infiltration_fixtures import resolved_infiltration_provider
from pa_event_receiving_fixtures import mixed_pa_events as mixed_pa_events
from pa_event_receiving_fixtures import pa_event_receiving_boundary

from pytoce.prepare_backgrounds.preparation.providers.catalogue import provider_service_code
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
)
from pytoce.run_lca.calculation.burdens import BurdenDemands
from pytoce.run_lca.calculation.contribution_accumulation import (
    ContributionMethod,
    create_contribution_state,
)
from pytoce.run_lca.calculation.direct_contributions import (
    add_direct_contributions,
    admit_direct_characterization_factors,
)
from pytoce.run_lca.calculation.pa_event_receiving import iter_pa_receiving
from pytoce.run_lca.calculation.pa_inventory import (
    build_pa_receiving_inventory,
    pa_receiving_physical_events,
)
from pytoce.run_lca.calculation.pa_projection import PaProjectedEvent
from pytoce.run_lca.calculation.timex import (
    PrincipalTimeline,
    assemble_principal_inventory,
    build_provider_inventory_bank,
)


@pytest.mark.parametrize("original", (True, False))
def test_actual_pa_native_inventory_matches_independent_brightway_calculation(
    tmp_path: Path, mixed_pa_events: BurdenDemands, original: bool
) -> None:
    """Compare qB+e to a public Brightway solve for the full actual receiving graph.

    The admitted MFA mixtures are shifted to validation years on each side of
    the 2025--2050 interval. Synthetic characterization factors preserve mixed
    native flow units and signs.
    """
    provider = resolved_infiltration_provider(tmp_path / "class_source")
    dumping = provider.routes.loc[
        provider.routes["provider_demand_type"] == "mismanaged_open_dump_treatment"
    ]
    shares = dict(zip(dumping["infiltration_class"], dumping["infiltration_share"], strict=True))
    roots = {"EUR": ("foreground", "root_eur")}
    with pa_event_receiving_boundary(tmp_path / "native", original=original) as boundary:
        bd = boundary.bd
        spinning = {}
        for grade in ("pa6", "pa66"):
            service = boundary.services[f"fibres_{grade}_conversion"]
            spinning[grade] = PaReceivingActivity(
                ("frontier", provider_service_code(str(service["provider_service_id"]))),
                str(service["name"]),
                str(service["reference_product"]),
                str(service["unit"]),
                str(service["location"]),
            )
        receiving = tuple(
            iter_pa_receiving(
                spinning_events=tuple(
                    replace(event, year=event.year + 16)
                    for event in mixed_pa_events.pa_spinning_events
                ),
                recycling_events=tuple(
                    replace(event, year=event.year + 16)
                    for event in mixed_pa_events.pa_recycling_events
                ),
                disposal_events=tuple(
                    replace(event, year=event.year + 16)
                    for event in mixed_pa_events.pa_disposal_events
                ),
                reference=boundary.reference,
                sources=boundary.sources,
                spinning_providers=spinning,
                dump_shares=shares,
            )
        )
        projected = tuple(
            PaProjectedEvent(
                event,
                infiltration,
                2025 if original else 2050,
                boundary.sources.general_background_database,
                "frontier",
                (2050 - event.year) / 25 if original else (event.year - 2025) / 25,
                result,
            )
            for event, infiltration, result in receiving
        )
        actual = build_pa_receiving_inventory(bd, projected, roots)
        physical = pa_receiving_physical_events(projected)
        physical_path = tmp_path / "pa_receiving_physical_events.parquet"
        physical.to_parquet(physical_path, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(physical_path), physical)
        assert len(physical) == len(projected)
        for row, event in zip(physical.itertuples(index=False), projected, strict=True):
            assert json.loads(str(row.physical_event_json)) == json.loads(
                json.dumps(asdict(event.event))
            )
            assert json.loads(str(row.physical_receiving_json))
            # Inventory rows are written once for the parent and its nested water/sludge records.
            assert '"amounts":' not in str(row.physical_receiving_json)
            assert row.temporal_share == event.temporal_share
        assert pa_receiving_physical_events(()).columns.equals(physical.columns)
        assert {"kilogram", "cubic meter", "kilowatt hour"} <= set(actual.records.endpoint_unit)
        assert set(actual.records.endpoint_kind) == {"native_supplier", "elementary_flow"}
        assert (
            actual.records.amount
            == actual.records.unweighted_amount * actual.records.temporal_share
        ).all()
        assert not actual.activations.virgin_feedstock_route.any()
        assert not actual.activations.ultimate_carbon_origin.any()
        assert set(actual.activations.matrix_database) == {"frontier"}
        assert "frontier" in set(actual.activations.provider_database)
        assert boundary.sources.general_background_database in set(
            actual.activations.provider_database
        )
        # Unsplit functions retain the real common reporting stages, including
        # conversion/washing and both managed and mismanaged terminal routes.
        assert {
            "processing_conversion",
            "mechanical_recycling_operation",
            "landfill_treatment",
        } <= set(actual.records.life_cycle_stage)
        assert set(actual.records.infiltration_class) - {""} == set(shares)
        empty = build_pa_receiving_inventory(bd, (), roots)
        assert empty.activations.empty and empty.direct_inventory.empty and empty.records.empty
        assert empty.records.columns.equals(actual.records.columns)

        method = ContributionMethod(
            "validation", "1", "mixed", ("validation", "mixed"), "test impact unit"
        )
        native_flows = sorted(bd.Database("biosphere"), key=lambda flow: flow.key)
        factors = {
            int(flow.id): (3.0 if index % 2 == 0 else -2.0)
            for index, flow in enumerate(native_flows)
        }
        bd.Method(method.method_tuple).register(unit=method.unit)
        bd.Method(method.method_tuple).write(
            [(flow.key, factors[int(flow.id)]) for flow in native_flows]
        )
        state = create_contribution_state(
            bd, {"EUR": actual.activations}, roots, (method,), (2035, 2036, 2037)
        )
        admit_direct_characterization_factors(state, sorted(set(actual.direct_inventory.flow_id)))
        add_direct_contributions(state, actual.direct_contributions)
        bank = build_provider_inventory_bank(bd, actual.activations, state)
        principal = assemble_principal_inventory(
            PrincipalTimeline(
                pd.DataFrame(),
                actual.activations,
                0.0,
                actual.direct_inventory[["year", "flow_id", "amount"]],
            ),
            bank,
        )
        observed = fsum(
            float(cast(Any, row).amount) * factors[int(cast(Any, row).flow_id)]
            for row in principal.inventory.itertuples(index=False)
        )
        demands: dict[int, list[float]] = {}
        direct_scores = []
        for event, _, result in receiving:
            temporal_share = (2050 - event.year) / 25 if original else (event.year - 2025) / 25
            # Calculate the reference from raw receiving rows.
            for row in result.amounts:
                node = bd.get_node(database=row.endpoint.key[0], code=row.endpoint.key[1])
                quantity = row.amount * temporal_share
                if isinstance(row.endpoint, PaReceivingActivity):
                    demands.setdefault(int(node.rp_exchange().input.id), []).append(quantity)
                else:
                    direct_scores.append(quantity * factors[int(node.id)])
        reference = bc.LCA(
            {key: fsum(values) for key, values in demands.items()}, method.method_tuple
        )
        reference.lci()
        reference.lcia()
        expected = fsum((float(reference.score), *direct_scores))
        assert observed == pytest.approx(expected, rel=2e-7)
        assert fsum(value[0] for value in state.annual_totals.values()) == pytest.approx(
            expected, rel=2e-7
        )
        # Norms sum the absolute contributions at their individual contributor scale.
        assert fsum(value[1] for value in state.annual_totals.values()) >= abs(expected)
