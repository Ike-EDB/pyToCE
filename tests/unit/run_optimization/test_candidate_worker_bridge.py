"""Derive candidate metrics from saved scores and admitted allocated carrying capacities."""

from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from candidate_worker_fixtures import (
    _infeasible_result,
    _planner,
    _result,
    _result_and_acc_metrics,
    _worker_contexts,
)
from intervention_fixtures import _circularity_parent, _parent_lifetimes, _parent_stock
from response_projection_fixtures import _allocated_cc_selection

from pytoce.run_lca.calculation.foreground import optimization_activation_delta
from pytoce.run_optimization.blocks import OPTIMIZATION_BLOCK_ORDER
from pytoce.run_optimization.calculation.candidate_acc import prepare_candidate_acc_context
from pytoce.run_optimization.calculation.candidate_projection import (
    _optimization_scores,
)
from pytoce.run_optimization.calculation.candidate_search import InterventionCandidate
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestBridge,
)
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    demand_path_departure,
    demand_repair_neighbors,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.request_search import (
    CandidateWorkerEvaluator,
    build_candidate_worker_metric_scopes,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    CandidateWorkerMetricScope,
    _selected_candidate_result,
    candidate_worker_result_identity,
    encode_candidate_worker_result,
)
from pytoce.run_optimization.runtime.candidate_results import CandidateResultStore
from pytoce.run_optimization.runtime.scheduler import CandidateProcessPool


def test_repaired_path_reuses_one_endpoint_result_across_blocks(
    tmp_path: Path,
) -> None:
    """Requested endpoint aliases retain one saved realized demand representative."""
    planner = _planner()
    parent = planner.intervention_parent
    planner = replace(
        planner,
        intervention_parent=replace(
            parent,
            demand_lifetime_parent=replace(
                parent.demand_lifetime_parent,
                stock_targets=_parent_stock(),
                lifetimes=_parent_lifetimes(),
            ),
            circularity_parent=replace(parent.circularity_parent, table=_circularity_parent()),
        ),
    )
    requested = DemandLifetimeDecision(50, 50)
    pool = demand_repair_neighbors(requested, failure_year=2020)
    realized = next(
        choice for choice in pool if choice.demand_fraction[1] < requested.demand_fraction[1]
    )
    attempts = pool.index(realized) + 1
    original_diagnostic = _infeasible_result().physical.diagnostic
    assert original_diagnostic is not None
    # A coherent synthetic receipt for a strict stock-boundary rejection:
    # the LP's returned linear path was already tested, so its reused
    # outcome adds no MFA call. Earlier pool paths cannot fix the first year.
    diagnostic = replace(
        original_diagnostic,
        observed=np.nextafter(1.0, 0.0),
        bound=1.0,
        shortfall=1.0 - np.nextafter(1.0, 0.0),
        coordinates=(("year", 2020), ("region", "EUR"), ("good", "Buildings & Construction")),
    )
    candidate = InterventionCandidate(
        "demand_lifetime_frontier", requested, CircularityDecision.parent()
    )
    requested_intervention = planner.plan_intervention(
        demand_lifetime=requested, circularity=candidate.circularity
    )
    realized_intervention = planner.plan_intervention(
        demand_lifetime=realized, circularity=candidate.circularity
    )
    assert requested_intervention.identity != realized_intervention.identity
    result = _result_and_acc_metrics(intervention=realized_intervention)[0]
    result = replace(
        result,
        requested_intervention_identity=requested_intervention.identity,
        repair=replace(
            result.repair,
            status="local_pool_feasible",
            mfa_evaluation_count=1 + attempts,
            stock_lp_count=1,
            pool_candidate_count=len(pool),
            pool_evaluation_count=attempts,
            demand_departure_pp=demand_path_departure(realized),
            linear_diagnostic=diagnostic,
            stock_projection_diagnostic=diagnostic,
        ),
        metric_scopes=build_candidate_worker_metric_scopes(
            ("demand_lifetime_frontier", "integrated_frontier")
        ),
    )
    result = replace(result, identity=candidate_worker_result_identity(result))
    alias = replace(candidate, optimization_block="integrated_frontier")
    context, scoring = _worker_contexts()
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="request::annual-reuse",
        worker_authority_identity=planner.worker_authority_identity,
    ) as store:
        # The request evaluator restores a completed fixture receipt directly.
        producer = CandidateWorkerRequestBridge(
            planner,
            result.metric_scopes,
            result_store=store,
            worker_scientific_authority_identity=result.worker_scientific_authority_identity,
            allocated_cc_context=context,
            search_scoring=scoring,
        )
        task = producer.build_task(requested_intervention, result.metric_scopes)
        store.save(task, encode_candidate_worker_result(result))
        reader = CandidateWorkerRequestBridge(
            planner,
            result.metric_scopes,
            result_store=store,
            worker_scientific_authority_identity=result.worker_scientific_authority_identity,
            allocated_cc_context=context,
            search_scoring=scoring,
        )
        with CandidateProcessPool(
            worker_initializer_payload=b"not used for completed results"
        ) as pool:
            evaluator = CandidateWorkerEvaluator(
                planner=planner,
                bridge=reader,
                scheduler=pool,
                result_store=store,
            )
            original = evaluator((candidate,))
            reused = evaluator((alias,))
            assert original[0].state_metrics
            assert reused[0].state_metrics
            assert all(
                metric.optimization_block == "integrated_frontier"
                for metric in reused[0].state_metrics
            )
            selected_candidates = (candidate, alias)
            pool.close()
            selected = evaluator.selected(
                selected_candidates,
                selection_identities={
                    item.identity: ("state::annual-reuse",) for item in selected_candidates
                },
            )
            assert pool._executor is None
        assert len(tuple(store.directory.glob("[0-9]*"))) == 1
        assert len(evaluator.result_summary_identities) == 1
    assert all(
        item.selected_result.result is selected[0].selected_result.result for item in selected
    )
    assert all(item.selected_result.demand_lifetime == realized for item in selected)
    assert all(item.candidate.demand_lifetime == requested for item in selected)
    assert selected[0].selected_result.result.identity == result.identity


@pytest.mark.parametrize("feasible", (True, False))
def test_bridge_derives_complete_search_states(tmp_path: Path, feasible: bool) -> None:
    """Generate the complete state set from one MFA outcome and trusted aCC inputs."""
    result = _result() if feasible else _infeasible_result()
    planner = _planner()
    context, scoring = _worker_contexts()
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="request::fixture",
        worker_authority_identity=planner.worker_authority_identity,
    ) as store:
        bridge = CandidateWorkerRequestBridge(
            planner,
            result.metric_scopes,
            result_store=store,
            worker_scientific_authority_identity=result.worker_scientific_authority_identity,
            allocated_cc_context=context,
            search_scoring=scoring,
        )
        task = bridge.build_task(result.intervention, result.metric_scopes)
        saved = store.save(task, encode_candidate_worker_result(result))
        summary = bridge.read_saved_result(saved)
    assert summary.physical.status == result.physical.status
    assert summary.physical.diagnostic == result.physical.diagnostic
    if feasible:
        observed = [
            (entry.optimization_block, entry.background_scenario, entry.allocated_cc_case)
            for entry in summary.member_metrics
        ]
        assert len(observed) == len(set(observed))
        assert set(observed) == set(
            product(
                (scope.optimization_block for scope in result.metric_scopes),
                (
                    member.background_scenario
                    for member in scoring.optimization_parent.member_parents
                ),
                context.allocated_cc_cases,
            )
        )
        for entry in summary.member_metrics:
            assert entry.all_region_joint_compliant
            assert entry.z_worst == 0.5
            assert entry.binding_slack == 0.5
            assert entry.compliant_region_count == 12
            assert entry.compliant_coordinate_count == 984
    else:
        assert summary.member_metrics == ()
        assert summary.scores is None


@pytest.mark.parametrize("unit_score", (2.0, -2.0))
def test_root_metrics_follow_real_provider_scores_and_external_capacity(
    tmp_path: Path, unit_score: float
) -> None:
    """Check annual/cumulative ratios across states, including negative impacts."""
    _, scoring = _worker_contexts(unit_score=unit_score)
    parent = scoring.physical_parent.activations
    candidate = pd.concat(
        [parent, parent.assign(year=2050, physical_event_year=2050, amount=3.0)], ignore_index=True
    )
    delta = optimization_activation_delta(parent, candidate)
    result = _result_and_acc_metrics(activation_delta=delta)[0]
    scores = _optimization_scores(result.physical, scoring, delta, scoring.pa_parent)
    scopes = tuple(
        CandidateWorkerMetricScope(block, "metrics::" + block) for block in OPTIMIZATION_BLOCK_ORDER
    )
    result = replace(result, scores=scores, metric_scopes=scopes)
    result = replace(result, identity=candidate_worker_result_identity(result))
    context = prepare_candidate_acc_context(
        _allocated_cc_selection(
            method_compatibility_identity=scoring.optimization_projection.method_compatibility_identity,
            climate_compatibility_identity=scoring.climate_projection.method_compatibility_identity,
        )
    )
    planner = replace(_planner(), allocated_cc_cases=context.allocated_cc_cases)
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="request::oracle",
        worker_authority_identity=planner.worker_authority_identity,
    ) as store:
        bridge = CandidateWorkerRequestBridge(
            planner,
            scopes,
            result_store=store,
            worker_scientific_authority_identity=result.worker_scientific_authority_identity,
            allocated_cc_context=context,
            search_scoring=scoring,
        )
        task = bridge.build_task(result.intervention, scopes)
        saved = store.save(task, encode_candidate_worker_result(result))
        summary = bridge.read_saved_result(saved)
        selected = _selected_candidate_result(
            bridge.load_candidate(task.candidate_identity),
            summary,
            selection_identities=("selection::oracle",),
        )
    assert selected.result.scores is not None
    np.testing.assert_array_equal(selected.result.scores.annual_scores, scores.annual_scores)
    observed = [
        (entry.optimization_block, entry.background_scenario, entry.allocated_cc_case)
        for entry in summary.member_metrics
    ]
    assert len(observed) == len(set(observed)) == 24
    assert set(observed) == set(
        product(OPTIMIZATION_BLOCK_ORDER, scores.backgrounds, context.allocated_cc_cases)
    )
    for entry in summary.member_metrics:
        noncompliant = unit_score > 0 and entry.allocated_cc_case == "lower_values"
        assert entry.compliant_coordinate_count == (982 if noncompliant else 984)
        assert entry.compliant_region_count == (11 if noncompliant else 12)
        assert entry.compliant_region_impact_group_count == (22 if noncompliant else 24)
        assert entry.fully_compliant_impact_target_count == (0 if noncompliant else 2)
        assert entry.all_region_joint_compliant == (not noncompliant)
        expected_worst = (2.0 if noncompliant else 1.0) if unit_score > 0 else 0.0
        assert entry.z_worst == expected_worst
