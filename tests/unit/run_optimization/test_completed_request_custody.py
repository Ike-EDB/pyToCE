"""Restore completed requests with their cached calculations and aliases."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from candidate_worker_fixtures import (
    _infeasible_result,
    _planner,
    _result_and_acc_metrics,
    _worker_contexts,
)
from intervention_fixtures import _circularity_parent, _parent_lifetimes, _parent_stock

from pytoce.run_lca.calculation.foreground import optimization_activation_delta
from pytoce.run_optimization.blocks import OPTIMIZATION_BLOCK_ORDER
from pytoce.run_optimization.calculation.candidate_projection import _optimization_scores
from pytoce.run_optimization.calculation.candidate_search import InterventionCandidate
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerPlanner,
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
    CompletedCandidateSearch,
    _run_candidate_search,
    _timing_candidate,
    build_candidate_worker_metric_scopes,
    run_optimization_search,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    candidate_worker_result_identity,
    encode_candidate_worker_result,
)
from pytoce.run_optimization.runtime.candidate_results import CandidateResultStore
from pytoce.run_optimization.runtime.completed_search import (
    decode_completed_search,
    encode_completed_search,
)
from pytoce.run_optimization.runtime.scheduler import CandidateProcessPool


def _trajectory_planner() -> CandidateWorkerPlanner:
    """Attach the small actual trajectory-owner tables to the worker fixture."""
    planner = _planner()
    parent = planner.intervention_parent
    return replace(
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


def test_cached_parent_request_round_trip_appends_stage_and_preserves_hidden_blocks(
    tmp_path: Path,
) -> None:
    """A compatible extension keeps actual stages, shared tasks and selected bytes."""
    planner = _trajectory_planner()
    context, scoring = _worker_contexts()
    context = replace(
        context,
        identity="acc-context::generous-parent",
        annual_allocated_cc=np.full_like(context.annual_allocated_cc, 10.0),
        cumulative_allocated_cc=np.full_like(context.cumulative_allocated_cc, 820.0),
    )
    scopes = build_candidate_worker_metric_scopes(("integrated_frontier",))
    intervention = planner.plan_intervention(
        demand_lifetime=DemandLifetimeDecision(0, 0),
        circularity=CircularityDecision.parent(),
    )
    parent_activations = scoring.physical_parent.activations
    delta = optimization_activation_delta(parent_activations, parent_activations)
    result = _result_and_acc_metrics(intervention=intervention, activation_delta=delta)[0]
    result = replace(
        result,
        scores=_optimization_scores(result.physical, scoring, delta, scoring.pa_parent),
        metric_scopes=scopes,
    )
    result = replace(result, identity=candidate_worker_result_identity(result))
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="input::terminal-parent",
        worker_authority_identity=planner.worker_authority_identity,
    ) as store:
        producer = CandidateWorkerRequestBridge(
            planner,
            scopes,
            result_store=store,
            allocated_cc_context=context,
            search_scoring=scoring,
            worker_scientific_authority_identity=result.worker_scientific_authority_identity,
        )
        saved_task = producer.build_task(intervention, scopes)
        store.save(saved_task, encode_candidate_worker_result(result))
        saved_payload = store.directory / "00000000" / "payload.bin"
        original_bytes = saved_payload.read_bytes()
        original_mtime = saved_payload.stat().st_mtime_ns
        fresh_bridge = CandidateWorkerRequestBridge(
            planner,
            scopes,
            result_store=store,
            allocated_cc_context=context,
            search_scoring=scoring,
            worker_scientific_authority_identity=result.worker_scientific_authority_identity,
        )
        with CandidateProcessPool(
            worker_initializer_payload=b"cached fixture has no worker"
        ) as pool:
            first = run_optimization_search(
                request_identity="request::five-point",
                planner=planner,
                bridge=fresh_bridge,
                scheduler=pool,
                result_store=store,
                requested_blocks=("integrated_frontier",),
                candidate_input_identity="input::terminal-parent",
                initial_grid_step_pct=100,
                endpoint_refinement_steps_pct=(10, 5),
            )
            assert pool._executor is None
            assert pool.computed_candidate_identities == frozenset()

        completed = first.completed_search
        assert completed is not None
        encoded = json.dumps(encode_completed_search(completed), allow_nan=False, sort_keys=True)
        restored = decode_completed_search(json.loads(encoded))
        assert restored == completed
        assert restored.executed_blocks == OPTIMIZATION_BLOCK_ORDER
        assert tuple(stage.step_pct for stage in restored.stages) == (100, 10, 5)
        assert len(restored.task_records) == len(restored.worker_summaries) == 1
        assert len(restored.candidate_records) == 3
        assert len({record[1] for record in restored.candidate_records}) == 1
        assert tuple(item.optimization_block for item in restored.stages[-1].single_blocks) == (
            "demand_lifetime_frontier",
            "circularity_feedstock_anchor",
        )

        resumed_bridge = CandidateWorkerRequestBridge(
            planner,
            scopes,
            result_store=store,
            allocated_cc_context=context,
            search_scoring=scoring,
            worker_scientific_authority_identity=result.worker_scientific_authority_identity,
        )
        with CandidateProcessPool(
            worker_initializer_payload=b"restored fixture has no worker"
        ) as pool:
            extended = run_optimization_search(
                request_identity="request::one-point",
                planner=planner,
                bridge=resumed_bridge,
                scheduler=pool,
                result_store=store,
                requested_blocks=("integrated_frontier",),
                candidate_input_identity="input::terminal-parent",
                initial_grid_step_pct=100,
                endpoint_refinement_steps_pct=(10, 5, 1),
                continuation_parent_identity="product::five-point",
                completed_search=restored,
            )
            assert pool._executor is None
            assert pool.computed_candidate_identities == frozenset()

        assert extended.completed_search is not None
        assert extended.completed_search.stages[:-1] == completed.stages
        assert extended.completed_search.stages[-1].step_pct == 1
        assert extended.completed_search.task_records == completed.task_records
        assert extended.completed_search.worker_summaries == completed.worker_summaries
        assert extended.completed_search.candidate_records == completed.candidate_records
        assert extended.worker_result_set_identity == first.worker_result_set_identity
        assert extended.search.identity != first.search.identity
        assert extended.search.continuation_parent_identity == "product::five-point"
        assert (
            json.dumps(encode_completed_search(completed), allow_nan=False, sort_keys=True)
            == encoded
        )
        for products in (first, extended):
            assert products.request_coordinate_count == 3
            assert products.request_unique_physical_candidate_count == 1
            assert products.request_physical_candidate_deduplication_count == 2
            assert products.newly_computed_worker_result_count == 0
            assert products.reused_worker_result_count == 1
            assert products.incremental_mfa_evaluation_count == 0
            assert products.incremental_stock_lp_count == 0
            assert products.incremental_physical_repair_elapsed_seconds == 0.0
            assert products.incremental_environmental_evaluation_elapsed_seconds == 0.0
            assert products.selected_timing == ()
            assert len(products.selected_2050) == 1
            selected = products.selected_2050[0]
            assert selected.selected_result.result.identity == result.identity
            assert selected.selected_result.summary.worker_result_identity == result.identity
            assert len(selected.selection_identities) == 4
            assert selected.selected_result.result.scores is not None and result.scores is not None
            np.testing.assert_array_equal(
                selected.selected_result.result.scores.annual_scores, result.scores.annual_scores
            )
        assert saved_payload.read_bytes() == original_bytes
        assert saved_payload.stat().st_mtime_ns == original_mtime
        assert len(tuple(store.directory.glob("[0-9]*/payload.bin"))) == 1


def test_restored_repaired_request_and_delayed_result_keep_custody_and_next_sequence(
    tmp_path: Path,
) -> None:
    """Keep the parent selection and additional diagnostic endpoint/timing records.

    A completed parent-selected request remains unchanged. Separately
    admitted diagnostic receipts exercise restoration of requested/repaired
    identities and delayed paths; they are not public optimizer selections or
    newly computed MFA feasibility claims.
    """
    planner = _trajectory_planner()
    context, scoring = _worker_contexts()
    scopes = build_candidate_worker_metric_scopes(("demand_lifetime_frontier",))
    parent = InterventionCandidate(
        "demand_lifetime_frontier", DemandLifetimeDecision(0, 0), CircularityDecision.parent()
    )
    requested = replace(parent, demand_lifetime=DemandLifetimeDecision(50, 50))
    pool_paths = demand_repair_neighbors(requested.demand_lifetime, failure_year=2020)
    realized = next(
        item
        for item in pool_paths
        if item.demand_fraction[1] < requested.demand_lifetime.demand_fraction[1]
    )
    timing = _timing_candidate(requested, target_year=2100)
    assert timing is not None
    next_endpoint = replace(requested, demand_lifetime=DemandLifetimeDecision(40, 50))
    candidates = (parent, requested, timing, next_endpoint)
    interventions = tuple(
        planner.plan_intervention(
            demand_lifetime=item.demand_lifetime, circularity=item.circularity
        )
        for item in candidates
    )
    realized_intervention = planner.plan_intervention(
        demand_lifetime=realized, circularity=requested.circularity
    )
    parent_activations = scoring.physical_parent.activations
    delta = optimization_activation_delta(parent_activations, parent_activations)
    parent_result = _result_and_acc_metrics(intervention=interventions[0], activation_delta=delta)[
        0
    ]
    parent_result = replace(
        parent_result,
        scores=_optimization_scores(parent_result.physical, scoring, delta, scoring.pa_parent),
        metric_scopes=scopes,
    )
    repaired = _result_and_acc_metrics(intervention=realized_intervention)[0]
    diagnostic = _infeasible_result().physical.diagnostic
    assert diagnostic is not None
    diagnostic = replace(
        diagnostic,
        observed=np.nextafter(1.0, 0.0),
        bound=1.0,
        shortfall=1.0 - np.nextafter(1.0, 0.0),
        coordinates=(("year", 2020), ("region", "EUR"), ("good", "Buildings & Construction")),
    )
    repaired = replace(
        repaired,
        requested_intervention_identity=interventions[1].identity,
        metric_scopes=scopes,
        repair=replace(
            repaired.repair,
            status="local_pool_feasible",
            physical_repair_elapsed_seconds=2.25,
            environmental_evaluation_elapsed_seconds=3.5,
            mfa_evaluation_count=2 + pool_paths.index(realized),
            stock_lp_count=1,
            pool_candidate_count=len(pool_paths),
            pool_evaluation_count=1 + pool_paths.index(realized),
            demand_departure_pp=demand_path_departure(realized),
            linear_diagnostic=diagnostic,
            stock_projection_diagnostic=diagnostic,
        ),
    )
    results = tuple(
        replace(item, identity=candidate_worker_result_identity(item))
        for item in (
            parent_result,
            repaired,
            *(
                replace(_result_and_acc_metrics(intervention=item)[0], metric_scopes=scopes)
                for item in interventions[2:]
            ),
        )
    )
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="input::repaired-stage",
        worker_authority_identity=planner.worker_authority_identity,
    ) as store:
        producer = CandidateWorkerRequestBridge(
            planner,
            scopes,
            result_store=store,
            allocated_cc_context=context,
            search_scoring=scoring,
            worker_scientific_authority_identity=results[0].worker_scientific_authority_identity,
        )
        for intervention, result in zip(interventions, results, strict=True):
            store.save(
                producer.build_task(intervention, scopes), encode_candidate_worker_result(result)
            )
        original_payloads = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in store.directory.glob("[0-9]*/payload.bin")
        }
        bridge = CandidateWorkerRequestBridge(
            planner,
            scopes,
            result_store=store,
            allocated_cc_context=context,
            search_scoring=scoring,
            worker_scientific_authority_identity=results[0].worker_scientific_authority_identity,
        )
        with CandidateProcessPool(worker_initializer_payload=b"cached repaired fixtures") as pool:
            evaluator = CandidateWorkerEvaluator(
                planner=planner, bridge=bridge, scheduler=pool, result_store=store
            )
            search = _run_candidate_search(
                request_identity="request::repaired-stage",
                requested_blocks=("demand_lifetime_frontier",),
                evaluator=evaluator,
                candidate_input_identity="input::repaired-stage",
                initial_grid_step_pct=50,
                endpoint_refinement_steps_pct=(10,),
            )
            stage = search.completed_stages[-1].single_blocks[0]
            assert stage.selection.selected_candidate_identities == (parent.identity,)
            repaired_evaluation = evaluator((requested,))[0]
            evaluator((timing,))
            completed = CompletedCandidateSearch(
                initial_grid_step_pct=50,
                endpoint_refinement_steps_pct=(10,),
                executed_blocks=("demand_lifetime_frontier",),
                stages=search.completed_stages,
                candidate_records=evaluator.candidate_records,
                worker_summaries=evaluator.worker_summaries,
                task_records=evaluator.task_records,
            )
            original_result_set = evaluator.worker_result_set_identity()
            assert evaluator.newly_computed_repairs == ()
            assert pool._executor is None

        restored = decode_completed_search(
            json.loads(json.dumps(encode_completed_search(completed)))
        )
        assert restored == completed
        assert tuple(item.step_pct for item in restored.stages) == (50, 10)
        assert len(restored.task_records) == 3
        assert results[1].intervention.identity != results[1].requested_intervention_identity
        restored_bridge = CandidateWorkerRequestBridge(
            planner,
            scopes,
            result_store=store,
            allocated_cc_context=context,
            search_scoring=scoring,
            worker_scientific_authority_identity=results[0].worker_scientific_authority_identity,
        )
        with CandidateProcessPool(worker_initializer_payload=b"restored repaired fixtures") as pool:
            resumed = CandidateWorkerEvaluator(
                planner=planner, bridge=restored_bridge, scheduler=pool, result_store=store
            )
            resumed.restore_completed_search(restored)
            assert resumed.worker_result_set_identity() == original_result_set
            reused = resumed((requested, timing))
            assert resumed.task_records == restored.task_records
            assert reused[0] == repaired_evaluation
            repaired_payload = restored_bridge.load_candidate(interventions[1].identity)
            delayed_payload = restored_bridge.load_candidate(interventions[2].identity)
            assert repaired_payload.requested_intervention_identity == interventions[1].identity
            assert (
                repaired_payload.intervention.demand_lifetime_evaluation_projection.decision
                == realized
            )
            assert repaired_payload.identity == results[1].identity
            assert delayed_payload.identity == results[2].identity
            assert (
                delayed_payload.intervention.demand_lifetime_evaluation_projection.decision
                == timing.demand_lifetime
            )
            assert restored.stages[0].single_blocks[0].selection == stage.selection
            appended = resumed((next_endpoint,))
            assert appended[0].candidate == next_endpoint
            assert resumed.task_records[:-1] == restored.task_records
            assert resumed.task_records[-1].sequence == 3
            assert resumed.task_records[-1].candidate_identity == interventions[3].identity
            assert resumed.worker_summaries[:3] == restored.worker_summaries
            assert resumed.worker_summaries[-1].worker_result_identity == results[3].identity
            assert resumed.newly_computed_repairs == ()
            assert sum(item.stock_lp_count for item in resumed.repair_outcomes) == 1
            assert (
                sum(item.physical_repair_elapsed_seconds for item in resumed.repair_outcomes)
                == 2.25
            )
            assert (
                sum(
                    item.environmental_evaluation_elapsed_seconds
                    for item in resumed.repair_outcomes
                )
                == 3.5
            )
            assert pool.computed_candidate_identities == frozenset()
            assert pool._executor is None
        assert {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in store.directory.glob("[0-9]*/payload.bin")
        } == original_payloads
