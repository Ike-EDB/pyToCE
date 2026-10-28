"""Real late-cohort PA receiving through serial/spawned workers and saved results."""

from dataclasses import replace
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import pytest
from pa_worker_receiving_fixtures import prepare_pa_worker_workspace

from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestBridge,
    CandidateWorkerRequestContext,
    initialize_candidate_worker,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.request_search import (
    build_candidate_worker_metric_scopes,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    decode_candidate_worker_result,
    encode_candidate_worker_result,
)
from pytoce.run_optimization.runtime.candidate_results import CandidateResultStore
from pytoce.run_optimization.runtime.scheduler import CandidateProcessPool


@pytest.mark.scientific_integration
def test_actual_pa_parent_agrees_in_serial_spawn_and_cache(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> None:
    """Keep the actual PA mixture, score arrays and stage records through the real worker."""
    workspace = tmp_path / "workspace"
    initializer, context = prepare_pa_worker_workspace(workspace, remind_mfa_source_root)
    assert_pa_worker_equivalence(workspace, initializer, context)


def assert_pa_worker_equivalence(
    workspace: Path, initializer: bytes, context: CandidateWorkerRequestContext
) -> None:
    """Exercise the complete spawned, cached and serial contract from an actual parent."""
    started = monotonic()
    parent = context.physical_parent
    assert {
        (event.year, event.region, event.constituent) for event in parent.pa_spinning_events
    } == {
        (2100, "EUR", "polyamide_6_resin"),
        (2100, "EUR", "polyamide_66_resin"),
    }
    assert all(event.finished_principal_kg > 0 for event in parent.pa_spinning_events)
    assert {
        (event.year, event.region)
        for event in (*parent.pa_recycling_events, *parent.pa_disposal_events)
    } <= {(2100, "EUR")}
    pa_parent = context.search_scoring.pa_parent
    assert np.count_nonzero(pa_parent.annual_scores) > 0
    assert np.count_nonzero(pa_parent.annual_contribution_norms) > 0
    assert not pa_parent.stage_scores.empty
    assert set(pa_parent.stage_scores.region) == {"EUR"}
    assert set(pa_parent.stage_scores.year) == {2100}
    assert np.count_nonzero(pa_parent.annual_scores[:, :, :-1, :]) == 0
    assert (pa_parent.annual_contribution_norms[:, H12_REGIONS.index("EUR"), -1, :] > 0).all()

    scopes = build_candidate_worker_metric_scopes(("demand_lifetime_frontier",))
    intervention = context.planner.plan_intervention(
        demand_lifetime=DemandLifetimeDecision(0, 0), circularity=CircularityDecision.parent()
    )
    assert intervention.uses_exact_parent
    with CandidateResultStore.open(
        workspace / "results/pa_worker_cache",
        candidate_input_identity="request::actual-pa-exact-parent-fixture",
        worker_authority_identity=context.planner.worker_authority_identity,
    ) as store:
        bridge = CandidateWorkerRequestBridge(
            context.planner,
            scopes,
            result_store=store,
            allocated_cc_context=context.allocated_cc_context,
            search_scoring=context.search_scoring,
            worker_scientific_authority_identity=context.worker_scientific_authority_identity,
        )
        task = bridge.build_task(intervention, scopes)
        # The pool's single child closes before creating the serial session;
        # no two initialized scientific worker sessions coexist in this test.
        with CandidateProcessPool(worker_initializer_payload=initializer) as pool:
            first = pool.evaluate_batch(
                (task,), result_store=store, read_result=bridge.read_saved_result
            )
            saved = store.read_record(task.record)
            assert saved is not None
            second = pool.evaluate_batch(
                (task,), result_store=store, read_result=bridge.read_saved_result
            )
            assert store.read_record(task.record) == saved
        print(
            f"PA worker fixture: spawned task and cache complete ({monotonic() - started:.2f}s)",
            flush=True,
        )
        assert first == second
        serial = initialize_candidate_worker(initializer)(task.payload)
        result = decode_candidate_worker_result(serial)
        reused = bridge.load_candidate(task.candidate_identity)
        # Elapsed times describe independent executions. All other decoded
        # fields must retain their exact scientific and serialized content.
        for receipt in (result.repair, reused.repair):
            timings = (
                receipt.physical_repair_elapsed_seconds,
                receipt.environmental_evaluation_elapsed_seconds,
            )
            assert np.isfinite(timings).all() and min(timings) >= 0.0
        comparable = replace(
            result,
            repair=replace(
                result.repair,
                physical_repair_elapsed_seconds=reused.repair.physical_repair_elapsed_seconds,
                environmental_evaluation_elapsed_seconds=(
                    reused.repair.environmental_evaluation_elapsed_seconds
                ),
            ),
        )
        assert encode_candidate_worker_result(comparable) == saved

    assert first[0].worker_result_identity == result.identity == reused.identity
    assert (
        result.worker_scientific_authority_identity == context.worker_scientific_authority_identity
    )
    assert result.physical.pa_spinning_events == parent.pa_spinning_events
    assert result.physical.pa_recycling_events == parent.pa_recycling_events
    assert result.physical.pa_disposal_events == parent.pa_disposal_events
    assert result.physical.activation_delta is not None and result.physical.activation_delta.empty
    assert result.pa_scores is not None and reused.pa_scores is not None
    np.testing.assert_array_equal(result.pa_scores.annual_scores, pa_parent.annual_scores)
    np.testing.assert_array_equal(
        result.pa_scores.annual_contribution_norms, pa_parent.annual_contribution_norms
    )
    pd.testing.assert_frame_equal(result.pa_scores.stage_scores, pa_parent.stage_scores)
    assert result.scores is not None
    np.testing.assert_array_equal(
        result.scores.annual_scores, context.search_scoring.optimization_baseline_scores
    )
    np.testing.assert_array_equal(
        result.scores.annual_climate_scores, context.search_scoring.climate_baseline_scores[..., 0]
    )
    print(
        f"PA worker fixture: serial equivalence complete ({monotonic() - started:.2f}s)", flush=True
    )
