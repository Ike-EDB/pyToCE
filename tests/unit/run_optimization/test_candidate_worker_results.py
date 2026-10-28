"""Read complete saved candidates and bind selected results to admitted summaries."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from candidate_worker_fixtures import (
    _infeasible_result,
    _intervention,
    _pa_scores,
    _planner,
    _result,
    _result_and_acc_metrics,
    _worker_contexts,
)

from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestBridge,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    _selected_candidate_result,
    candidate_worker_result_identity,
    decode_candidate_worker_summary,
    encode_candidate_worker_result,
    encode_candidate_worker_summary,
)
from pytoce.run_optimization.runtime.candidate_results import (
    CandidateResultStore,
    CandidateTaskRecord,
)


def test_selected_candidate_is_loaded_from_its_saved_result(tmp_path: Path) -> None:
    """Load saved selections and preserve their worker result and metric summary."""

    result = replace(_result(), pa_scores=_pa_scores(active=True))
    result = replace(result, identity=candidate_worker_result_identity(result))
    planner = _planner()
    acc_context, scoring = _worker_contexts()
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="request::fixture",
        worker_authority_identity=planner.worker_authority_identity,
    ) as store:
        bridge = CandidateWorkerRequestBridge(
            planner,
            result.metric_scopes,
            allocated_cc_context=acc_context,
            search_scoring=scoring,
            result_store=store,
            worker_scientific_authority_identity=(result.worker_scientific_authority_identity),
        )
        task = bridge.build_task(result.intervention, result.metric_scopes)
        saved = store.save(task, encode_candidate_worker_result(result))
        summary = bridge.read_saved_result(saved)
        selected = _selected_candidate_result(
            bridge.load_candidate(task.candidate_identity),
            summary,
            selection_identities=("selection::b", "selection::a"),
        )

        assert summary.worker_result_identity == result.identity
        assert selected.result.identity == result.identity
        assert selected.result.pa_scores is not None and result.pa_scores is not None
        np.testing.assert_array_equal(
            selected.result.pa_scores.annual_scores, result.pa_scores.annual_scores
        )
        np.testing.assert_array_equal(
            selected.result.pa_scores.annual_contribution_norms,
            result.pa_scores.annual_contribution_norms,
        )
        pd.testing.assert_frame_equal(
            selected.result.pa_scores.stage_scores, result.pa_scores.stage_scores
        )
        assert selected.selection_identities == ("selection::a", "selection::b")
        assert bridge.finalize_result_set({task.candidate_identity: summary}).startswith(
            "pytoce_optimization_candidate_worker_result_set_v1::"
        )


@pytest.mark.parametrize("infeasible", (False, True))
def test_fresh_bridge_rebuilds_the_saved_candidate_task(tmp_path: Path, infeasible: bool) -> None:
    """Recreate the task identity before reading a result in a later process."""

    result = _infeasible_result() if infeasible else _result()
    first_planner = _planner()
    acc_context, scoring = _worker_contexts()
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="request::fixture",
        worker_authority_identity=first_planner.worker_authority_identity,
    ) as first_store:
        first_bridge = CandidateWorkerRequestBridge(
            first_planner,
            result.metric_scopes,
            allocated_cc_context=acc_context,
            search_scoring=scoring,
            result_store=first_store,
            worker_scientific_authority_identity=(result.worker_scientific_authority_identity),
        )
        first_task = first_bridge.build_task(result.intervention, result.metric_scopes)
        first_store.save(first_task, encode_candidate_worker_result(result))

    saved_bytes = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / ".candidate_results").rglob("*")
        if path.is_file()
    }

    second_planner = _planner()
    acc_context, scoring = _worker_contexts()
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="request::fixture",
        worker_authority_identity=second_planner.worker_authority_identity,
    ) as second_store:
        second_bridge = CandidateWorkerRequestBridge(
            second_planner,
            result.metric_scopes,
            allocated_cc_context=acc_context,
            search_scoring=scoring,
            result_store=second_store,
            worker_scientific_authority_identity=(result.worker_scientific_authority_identity),
        )
        second_task = second_bridge.build_task(result.intervention, result.metric_scopes)
        saved = second_store.read_record(second_task.record)

        assert second_task.record == first_task.record
        assert saved is not None
        summary = second_bridge.read_saved_result(saved)
        assert summary.worker_result_identity == result.identity
        selected = _selected_candidate_result(
            second_bridge.load_candidate(second_task.candidate_identity),
            summary,
            selection_identities=("selection::2050",),
        )
        assert selected.result.physical.status == result.physical.status
        assert selected.result.physical.diagnostic == result.physical.diagnostic
        assert selected.result.identity == result.identity
        assert (selected.result.scores is None) is infeasible
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in (tmp_path / ".candidate_results").rglob("*")
        if path.is_file()
    } == saved_bytes


def test_completed_stage_restores_tasks_summaries_and_result_order(tmp_path: Path) -> None:
    """Restore task order and compact summaries before loading a selected saved result."""

    results = (_result(), _infeasible_result(intervention=_intervention(target_year=2100)))
    planner = _planner()
    acc_context, scoring = _worker_contexts()
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="request::fixture",
        worker_authority_identity=planner.worker_authority_identity,
    ) as store:
        bridge = CandidateWorkerRequestBridge(
            planner,
            results[0].metric_scopes,
            allocated_cc_context=acc_context,
            search_scoring=scoring,
            result_store=store,
            worker_scientific_authority_identity=results[0].worker_scientific_authority_identity,
        )
        summaries = {}
        for result in results:
            task = bridge.build_task(result.intervention, result.metric_scopes)
            saved = store.save(task, encode_candidate_worker_result(result))
            summaries[task.candidate_identity] = bridge.read_saved_result(saved)
        completed_state = json.loads(
            json.dumps(
                {
                    "task_records": [asdict(record) for record in bridge.task_records],
                    "summaries": {
                        identity: encode_candidate_worker_summary(summary)
                        for identity, summary in summaries.items()
                    },
                },
                allow_nan=False,
            )
        )
        original_result_set = bridge.finalize_result_set(summaries)
        original_tasks = bridge.task_records

        resumed = CandidateWorkerRequestBridge(
            planner,
            results[0].metric_scopes,
            allocated_cc_context=acc_context,
            search_scoring=scoring,
            result_store=store,
            worker_scientific_authority_identity=results[0].worker_scientific_authority_identity,
        )
        resumed.restore_task_records(
            tuple(CandidateTaskRecord(**record) for record in completed_state["task_records"])
        )
        restored_summaries = {
            identity: decode_candidate_worker_summary(record)
            for identity, record in completed_state["summaries"].items()
        }
        assert resumed.task_records == original_tasks
        assert restored_summaries == summaries
        assert resumed.finalize_result_set(restored_summaries) == original_result_set
        selected = resumed.load_candidate(results[0].intervention.identity)
        assert selected.identity == results[0].identity
        assert selected.scores is not None and results[0].scores is not None
        np.testing.assert_array_equal(
            selected.scores.annual_scores, results[0].scores.annual_scores
        )

        next_result = _result_and_acc_metrics(intervention=_intervention(target_year=2060))[0]
        continuous_task = bridge.build_task(next_result.intervention, next_result.metric_scopes)
        resumed_task = resumed.build_task(next_result.intervention, next_result.metric_scopes)
        assert resumed_task == continuous_task
        assert resumed_task.sequence == len(original_tasks)
        assert resumed.task_records == bridge.task_records
