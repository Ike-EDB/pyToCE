"""Real candidate execution, MFA feasibility, cache reuse and selected reporting."""

from dataclasses import replace
from math import fsum
from multiprocessing.context import SpawnProcess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_optimization.calculation.candidate_search import (
    CandidateRefinement,
    CandidateStateCheck,
    InterventionCandidate,
    select_candidate_states,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestBridge,
    initialize_candidate_worker,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.request_search import (
    CandidateWorkerEvaluator,
    RequestSearchResult,
    SearchBlockResult,
    SearchExecutionProducts,
    build_candidate_worker_metric_scopes,
)
from pytoce.run_optimization.calculation.results import (
    _selected_reporting_states,
    expand_detailed_acc_metrics,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    CandidateWorkerTask,
    decode_candidate_worker_result,
    encode_candidate_worker_task,
)
from pytoce.run_optimization.runtime.candidate_results import CandidateResultStore
from pytoce.run_optimization.runtime.scheduler import CandidateProcessPool
from tests.integration.candidate_mfa_fixtures import (
    _WORKER_FOREGROUND,
    _prepare_worker_workspace,
)


@pytest.mark.scientific_integration
def test_real_candidates_agree_across_processes_and_preserve_physical_feasibility(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare three-worker science with serial results and release workers before loading."""

    workspace = tmp_path / "workspace"
    initializer, context = _prepare_worker_workspace(
        workspace,
        remind_mfa_source_root,
        include_generous_case=True,
        include_public_baseline_results=False,
        background_scenarios=("remind_ssp2_npi", "remind_ssp2_pkbudg650"),
    )
    candidate = InterventionCandidate(
        "demand_lifetime_frontier",
        DemandLifetimeDecision(0, 0),
        CircularityDecision.parent(),
    )
    changed = InterventionCandidate(
        "circularity_feedstock_anchor",
        DemandLifetimeDecision(0, 0),
        CircularityDecision(
            collection_pct=65,
            mechanical_recycling_pct=20,
            chemical_recycling_pct=10,
            incineration_pct=40,
            landfill_pct=30,
            fossil_pct=95,
            biogenic_pct=5,
            daccu_pct=0,
        ),
    )
    infeasible = InterventionCandidate(
        "demand_lifetime_frontier",
        DemandLifetimeDecision(90, 0),
        CircularityDecision.parent(),
    )
    candidates = (candidate, changed, infeasible)
    stock_reference = InterventionCandidate(
        "demand_lifetime_frontier",
        DemandLifetimeDecision(25, 100),
        CircularityDecision.parent(),
    )
    integrated_reference = replace(stock_reference, optimization_block="integrated_frontier")
    worker_processes: list[SpawnProcess] = []
    loading_selected = False
    start = SpawnProcess.start

    def observe_start(process: SpawnProcess) -> None:
        """Observe worker processes throughout calculation and reporting."""
        assert not loading_selected, "Loading selected results must not restart candidate workers"
        start(process)
        worker_processes.append(process)

    monkeypatch.setattr(SpawnProcess, "start", observe_start)
    cache = workspace / "results" / _WORKER_FOREGROUND / "optimization_results/cache_fixture"
    with CandidateResultStore.open(
        cache,
        candidate_input_identity="request::spawned-worker-fixture",
        worker_authority_identity=context.planner.worker_authority_identity,
    ) as result_store:
        bridge = CandidateWorkerRequestBridge(
            context.planner,
            build_candidate_worker_metric_scopes(
                ("demand_lifetime_frontier", "circularity_feedstock_anchor", "integrated_frontier")
            ),
            result_store=result_store,
            allocated_cc_context=context.allocated_cc_context,
            search_scoring=context.search_scoring,
            worker_scientific_authority_identity=context.worker_scientific_authority_identity,
        )
        with CandidateProcessPool(
            worker_initializer_payload=initializer, candidate_workers=3
        ) as scheduler:
            evaluator = CandidateWorkerEvaluator(
                planner=context.planner,
                bridge=bridge,
                scheduler=scheduler,
                result_store=result_store,
            )
            first = evaluator(candidates)
            second = evaluator(candidates)
            state_identities = tuple(item.state_identity for item in first[0].state_metrics)
            # The two search blocks share one reference. Admit its
            # actual failure before requesting alternative annual demand paths.
            saved_before = set(result_store.directory.glob("[0-9]*"))
            references = (stock_reference, integrated_reference)
            stock_failures = evaluator(references)
            assert len(stock_failures) == 2
            assert len({item.physical_candidate_identity for item in stock_failures}) == 1
            saved_after = set(result_store.directory.glob("[0-9]*"))
            assert len(saved_after - saved_before) == 1
            stock_intervention = context.planner.plan_intervention(
                demand_lifetime=stock_reference.demand_lifetime,
                circularity=stock_reference.circularity,
            )
            saved_repair = bridge.load_candidate(stock_intervention.identity)
            assert saved_repair.repair.linear_diagnostic is not None
            assert (
                saved_repair.repair.linear_diagnostic.code
                == "optimization_stock_below_surviving_cohort"
            )
            assert saved_repair.repair.stock_lp_count == 1
            assert saved_repair.repair.mfa_evaluation_count <= 178
            assert (
                saved_repair.intervention.demand_lifetime_evaluation_projection.decision.lifetime_fraction
                == stock_reference.demand_lifetime.lifetime_fraction
            )
            assert saved_repair.requested_intervention_identity == stock_intervention.identity
            assert evaluator(references) == stock_failures
            assert set(result_store.directory.glob("[0-9]*")) == saved_after
            assert len(worker_processes) == 3
            assert len({process.pid for process in worker_processes}) == 3
            assert all(process.is_alive() for process in worker_processes)
            scheduler.close()
            assert all(not process.is_alive() for process in worker_processes)
            loading_selected = True
            selected = evaluator.selected(
                (candidate,),
                selection_identities={candidate.identity: state_identities},
            )
        worker = initialize_candidate_worker(initializer)
        results = []
        for item in candidates:
            intervention = context.planner.plan_intervention(
                demand_lifetime=item.demand_lifetime, circularity=item.circularity
            )
            spawned = bridge.load_candidate(intervention.identity)
            task = CandidateWorkerTask(intervention, spawned.metric_scopes)
            serial = decode_candidate_worker_result(worker(encode_candidate_worker_task(task)))
            assert serial.identity == spawned.identity
            assert serial.repair.physical_repair_elapsed_seconds >= 0.0
            assert serial.repair.environmental_evaluation_elapsed_seconds >= 0.0
            assert (
                replace(
                    serial.repair,
                    physical_repair_elapsed_seconds=spawned.repair.physical_repair_elapsed_seconds,
                    environmental_evaluation_elapsed_seconds=spawned.repair.environmental_evaluation_elapsed_seconds,
                )
                == spawned.repair
            )
            if serial.scores is not None:
                assert spawned.scores is not None
                np.testing.assert_array_equal(
                    serial.scores.annual_scores, spawned.scores.annual_scores
                )
            results.append(spawned)

    assert first == second
    assert len(first) == 3
    assert first[0].candidate == candidate
    assert first[0].state_metrics
    assert all(item.region_count == len(H12_REGIONS) for item in first[0].state_metrics)

    parent_result, changed_result, failed_result = results
    assert parent_result.intervention.uses_exact_parent
    assert not changed_result.intervention.uses_exact_parent
    delta = changed_result.physical.activation_delta
    assert delta is not None and not delta.empty
    assert delta.loc[delta.year.eq(2019), "delta_amount_kg"].eq(0).all()
    assert delta.loc[delta.year.gt(2019), "delta_amount_kg"].ne(0).any()
    assert failed_result.physical.diagnostic is not None
    assert failed_result.physical.diagnostic.code == "optimization_stock_below_surviving_cohort"
    assert first[2].diagnostic == failed_result.physical.diagnostic
    assert failed_result.scores is None
    assert failed_result.feedstock_requirements is None
    assert first[2].state_metrics == ()
    assert (
        first[2].physical_candidate_identity == failed_result.physical.physical_candidate_identity
    )
    selection = select_candidate_states(first[:1])
    block = SearchBlockResult(
        optimization_block="demand_lifetime_frontier",
        refinement=CandidateRefinement(
            identity="refinement::exact-parent-fixture",
            optimization_block="demand_lifetime_frontier",
            evaluation_count=1,
            reused_candidate_count=1,
            selection=selection,
            state_verification=tuple(
                CandidateStateCheck(
                    state_identity=item.state_identity,
                    candidate_identity=candidate.identity,
                    selected_is_exact_parent=True,
                    selected_has_non_grid_baseline_component=False,
                    declared_neighborhood_verified=False,
                    selection_stable=True,
                )
                for item in first[0].state_metrics
            ),
            passes=(),
            final_neighbor_candidate_identities=(),
        ),
        selected_candidates=(candidate,),
        selected_states=first[0].state_metrics,
    )
    execution = SearchExecutionProducts(
        search=RequestSearchResult(
            identity="search::exact-parent-fixture",
            block_results=(block,),
        ),
        selected_2050=selected,
        selected_timing=(),
        physical_repair_elapsed_seconds=0.0,
        environmental_evaluation_elapsed_seconds=0.0,
        mfa_evaluation_count=0,
        stock_lp_count=0,
        environmental_evaluation_count=0,
        repair_status_counts=(),
        worker_result_set_identity=evaluator.worker_result_set_identity(),
        request_coordinate_count=5,
        request_unique_physical_candidate_count=4,
        request_physical_candidate_deduplication_count=1,
    )
    detailed = expand_detailed_acc_metrics(
        (selected[0].selected_result.result,),
        context,
    )
    raw = workspace / "data_raw/allocated_cc/results/worker_result"
    annual = pd.read_csv(raw / "annual_allocated_cc.csv")
    cumulative = pd.read_csv(raw / "cumulative_gwp100_allocated_cc.csv")
    scores = parent_result.scores
    assert scores is not None
    coordinates = pd.MultiIndex.from_product(
        (
            scores.regions,
            scores.years,
            [item.impact_category_id for item in scores.annual_categories],
        ),
        names=("region", "year", "impact_category_id"),
    )
    parent_metrics = detailed[parent_result.physical.physical_candidate_identity]
    assert {(item.background_scenario, item.allocated_cc_case) for item in parent_metrics} == {
        (background, case)
        for background in scores.backgrounds
        for case in ("direct_case", "generous_case")
    }
    for item in parent_metrics:
        background_index = scores.backgrounds.index(item.background_scenario)
        annual_capacity = (
            annual.loc[annual.allocated_cc_case.eq(item.allocated_cc_case)]
            .set_index(list(coordinates.names))
            .allocated_cc.reindex(coordinates)
            .to_numpy()
            .reshape(scores.annual_scores.shape[1:])
        )
        cumulative_capacity = (
            cumulative.loc[cumulative.allocated_cc_case.eq(item.allocated_cc_case)]
            .set_index("region")
            .cumulative_allocated_cc.reindex(scores.regions)
            .to_numpy()
        )
        expected_cumulative = np.array(
            [fsum(row) for row in scores.annual_climate_scores[background_index]]
        )
        np.testing.assert_array_equal(
            item.metrics.annual_asr,
            scores.annual_scores[background_index] / annual_capacity,
        )
        np.testing.assert_array_equal(
            item.metrics.cumulative_gwp100_asr, expected_cumulative / cumulative_capacity
        )

    reporting = _selected_reporting_states(execution, detailed)

    assert reporting
    assert all(item.selected_is_exact_parent for item in reporting)
    assert all(
        item.physical_candidate_identity == item.timing_2100_physical_candidate_identity
        for item in reporting
    )
