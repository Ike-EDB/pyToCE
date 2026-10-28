"""Preserve repaired annual paths across compact feedstock-family transport.

These typed receipts test transport and reconstruction. The MFA tests check path admission.
"""

from dataclasses import replace
from fractions import Fraction
from math import nextafter

import pytest
from candidate_worker_fixtures import _infeasible_result, _result_and_acc_metrics
from test_candidate_repair import _decision, _family
from test_candidate_repair import family_planner as family_planner

from pytoce.run_optimization.calculation.candidate_authority import (
    physical_evaluation_identity,
)
from pytoce.run_optimization.calculation.candidate_repair import (
    repair_reference_from_result,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    _demand_stock_multipliers,
)
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    demand_path_departure,
    demand_repair_neighbors,
)
from pytoce.run_optimization.calculation.interventions import DemandLifetimeDecision
from pytoce.run_optimization.calculation.worker_protocol import (
    candidate_worker_result_identity,
    decode_candidate_repair_reference,
    encode_candidate_repair_reference,
)


@pytest.mark.parametrize("status", ("local_pool_feasible", "repair_unresolved"))
def test_compact_reference_preserves_realized_path_and_sibling_assignments(
    family_planner, status
) -> None:
    """Keep chosen rational paths, failed projections and each member's own origins."""
    planner = family_planner
    linear = DemandLifetimeDecision(50, 50)
    requested = planner.plan_intervention(demand_lifetime=linear, circularity=_decision())
    # Use the declared finite geometry to generate nonlinear test paths.
    proposal = next(
        item
        for item in demand_repair_neighbors(linear, failure_year=2020)
        if item.demand_fraction[1] < linear.demand_fraction[1]
    )
    pool = demand_repair_neighbors(proposal, failure_year=2020)
    rejected_controls = {_demand_stock_multipliers(linear), _demand_stock_multipliers(proposal)}
    distinct = tuple(
        item for item in pool if _demand_stock_multipliers(item) not in rejected_controls
    )
    admitted = status == "local_pool_feasible"
    chosen = (
        next(item for item in distinct if item.demand_fraction[1] < proposal.demand_fraction[1])
        if admitted
        else proposal
    )
    attempted = distinct.index(chosen) + 1 if admitted else len(distinct)
    realized = planner.plan_intervention(demand_lifetime=chosen, circularity=_decision())
    failed = _infeasible_result(intervention=realized)
    linear_diagnostic = failed.physical.diagnostic
    assert linear_diagnostic is not None
    # A stock boundary can remain rejected after binary64 LP transport. Retain
    # its exact diagnostic separately from the originally rejected linear path.
    projected_diagnostic = replace(
        linear_diagnostic,
        observed=nextafter(1.0, 0.0),
        bound=1.0,
        shortfall=1.0 - nextafter(1.0, 0.0),
        coordinates=(("year", 2020), ("region", "EUR"), ("good", "Buildings & Construction")),
    )
    result = _result_and_acc_metrics(intervention=realized)[0] if admitted else failed
    if not admitted:
        physical = replace(result.physical, diagnostic=projected_diagnostic)
        result = replace(
            result,
            physical=replace(
                physical, physical_candidate_identity=physical_evaluation_identity(physical)
            ),
        )
    result = replace(
        result,
        requested_intervention_identity=requested.identity,
        repair=replace(
            result.repair,
            status=status,
            stock_lp_count=1,
            mfa_evaluation_count=2 + attempted,
            pool_candidate_count=len(pool),
            pool_evaluation_count=attempted,
            demand_departure_pp=demand_path_departure(chosen) if admitted else None,
            linear_diagnostic=linear_diagnostic,
            stock_projection_diagnostic=projected_diagnostic,
        ),
    )
    result = replace(result, identity=candidate_worker_result_identity(result))
    family = _family(planner, requested)
    assert family is not None
    reference = repair_reference_from_result(result, family)
    transported = decode_candidate_repair_reference(encode_candidate_repair_reference(reference))
    assert transported == reference
    assert transported.requested_intervention_identity == requested.identity
    assert transported.worker_result_identity == result.identity
    assert transported.status == status
    assert transported.pool_candidate_count == len(pool)
    assert transported.linear_diagnostic == linear_diagnostic
    assert transported.stock_projection_diagnostic == projected_diagnostic
    assert transported.physical_diagnostic == (None if admitted else projected_diagnostic)
    assert transported.demand_departure_pp == (demand_path_departure(chosen) if admitted else None)
    assert transported.demand_lifetime == chosen
    assert chosen.demand_fraction != linear.demand_fraction
    assert all(isinstance(value, Fraction) for value in transported.demand_lifetime.demand_fraction)
    assert chosen.lifetime_fraction == linear.lifetime_fraction
    assert chosen.demand_fraction[31:] == linear.demand_fraction[31:]

    sibling_controls = _decision(25, 25)
    sibling_request = planner.plan_intervention(
        demand_lifetime=linear, circularity=sibling_controls
    )
    assert _family(planner, sibling_request) == family
    sibling = planner.plan_intervention(
        demand_lifetime=transported.demand_lifetime, circularity=sibling_controls
    )

    def nonfeedstock(records):
        """Select all controls whose exact assignments the family shares."""
        return tuple(
            row for row in records if row[0] not in {"bio_production_rate", "daccu_production_rate"}
        )

    def feedstock(records):
        """Select the member-specific carbon-origin assignments."""
        return tuple(
            row for row in records if row[0] in {"bio_production_rate", "daccu_production_rate"}
        )

    assert nonfeedstock(sibling.assignment_records) == nonfeedstock(realized.assignment_records)
    assert nonfeedstock(sibling.assignment_records) != nonfeedstock(
        sibling_request.assignment_records
    )
    assert feedstock(sibling.assignment_records) == feedstock(sibling_request.assignment_records)
    assert feedstock(sibling.assignment_records) != feedstock(realized.assignment_records)
    assert sibling.identity not in (realized.identity, sibling_request.identity)
