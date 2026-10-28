"""Round-trip completed-stage records."""

import json
from dataclasses import replace

import pytest
from completed_search_fixtures import make_completed_search

from pytoce.run_optimization.calculation.integrated_design import (
    INTEGRATED_DESIGN_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.integrated_search import (
    INTEGRATED_SEARCH_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.interventions import DemandLifetimeDecision
from pytoce.run_optimization.runtime.completed_search import (
    COMPLETED_SEARCH_SCHEMA,
    decode_completed_search,
    encode_completed_search,
)


@pytest.mark.parametrize("parent_compliant", (True, False))
def test_completed_joint_search_roundtrip_preserves_every_stage_and_compact_result(
    parent_compliant: bool,
) -> None:
    """JSON preserves branches, routes, diagnostics, pruning and shared MFA identities."""
    completed = make_completed_search(parent_compliant=parent_compliant)
    completed = replace(
        completed,
        task_records=(
            replace(completed.task_records[0], repair_family_identity="repair-family::fixture"),
            *completed.task_records[1:],
        ),
    )
    encoded = encode_completed_search(completed)
    payload = json.dumps(encoded, allow_nan=False, sort_keys=True)
    restored = decode_completed_search(json.loads(payload))
    assert restored == completed
    assert restored.task_records[0].repair_family_identity == "repair-family::fixture"
    assert all(item.repair_family_identity is None for item in restored.task_records[1:])
    assert encode_completed_search(restored) == encoded
    assert encoded["schema_version"] == COMPLETED_SEARCH_SCHEMA
    assert encoded["integrated_design_method"] == INTEGRATED_DESIGN_METHOD_IDENTITY
    assert encoded["integrated_search_method"] == INTEGRATED_SEARCH_METHOD_IDENTITY
    assert encoded["single_block_stage_policies"] == [[10, False], [5, True]]
    assert len(encoded["joint_stages"]) == len(completed.stages) == 3
    assert len(encoded["pair_covers"]) == 1
    assert len(encoded["worker_summaries"]) == len(completed.worker_summaries)
    assert len(encoded["candidate_records"]) == len(completed.candidate_records)
    assert len(encoded["task_records"]) == len(completed.task_records)
    all_candidate_objects = {
        value.candidate.identity: value.candidate
        for stage in completed.stages
        for state in stage.single_blocks
        for value in state.evaluations
    }
    all_passes = {
        value
        for stage in completed.stages
        for state in stage.single_blocks
        for value in state.passes
    }
    for stage in completed.stages:
        assert stage.integrated is not None
        all_candidate_objects.update(
            {value.candidate.identity: value.candidate for value in stage.integrated.evaluations}
        )
        for joint in stage.integrated.completed_stages:
            all_candidate_objects.update(
                {value.identity: value for value in joint.design.candidates}
            )
            all_passes.update(poll.refinement_pass for poll in joint.polls)
    assert len(encoded["candidates"]) == len(all_candidate_objects)
    assert len(encoded["refinement_passes"]) == len(all_passes)
    assert len(encoded["demand_lifetime_decisions"]) == len(
        {item.demand_lifetime for item in all_candidate_objects.values()}
    )
    assert all(
        item.demand_lifetime
        == DemandLifetimeDecision(
            item.demand_lifetime.demand_reduction_pct,
            item.demand_lifetime.lifetime_extension_pct,
            item.target_year,
        )
        for item in all_candidate_objects.values()
    )
    last = restored.stages[-1].integrated
    assert last is not None
    assert last.selection is last.completed_stages[-1].selection
    assert last.terminal_branches is last.completed_stages[-1].terminal_branches
    if parent_compliant:
        assert all(stage.parent_shortcut for stage in last.completed_stages)
        assert len(completed.worker_summaries) == 1
        assert len(encoded["candidates"]) > len(completed.candidate_records)
    else:
        assert any(stage.design.routes for stage in last.completed_stages)
        assert any(stage.polls for stage in last.completed_stages)
        assert any(item.physical.diagnostic is not None for item in restored.worker_summaries)
        assert any(item.scores is not None for item in restored.worker_summaries)
        assert last.completed_stages[-1].local_polling_complete
    for earlier, later in zip(restored.stages, restored.stages[1:], strict=False):
        assert earlier.integrated is not None and later.integrated is not None
        assert earlier.integrated.completed_stages[0] is later.integrated.completed_stages[0]


@pytest.mark.parametrize("steps", ((10,), (10, 5, 1)))
def test_single_block_completed_search_needs_no_joint_catalog(steps: tuple[int, ...]) -> None:
    """Standalone I requests retain their own completed states and schedule semantics."""
    completed = make_completed_search(
        steps=steps,
        requested_blocks=("demand_lifetime_frontier",),
        parent_compliant=False,
    )
    encoded = encode_completed_search(completed)
    restored = decode_completed_search(json.loads(json.dumps(encoded, allow_nan=False)))
    assert restored == completed
    assert encoded["joint_stages"] == encoded["pair_covers"] == []
    assert all(stage.integrated is None for stage in restored.stages)
    assert encoded["single_block_stage_policies"] == [
        [step, step <= 5 or step == steps[-1]] for step in steps
    ]
    assert restored.executed_blocks == ("demand_lifetime_frontier",)


def test_prefix_extension_keeps_completed_stage_bytes_and_shared_definitions() -> None:
    """A compatible one-point schedule leaves each existing completed design unchanged."""
    five = make_completed_search()
    one = make_completed_search(steps=(10, 5, 1))
    encoded_five = encode_completed_search(five)
    encoded_one = encode_completed_search(one)
    restored_five = decode_completed_search(json.loads(json.dumps(encoded_five)))
    restored_one = decode_completed_search(json.loads(json.dumps(encoded_one)))
    assert restored_one.stages[: len(five.stages)] == restored_five.stages
    assert len(encoded_five["joint_stages"]) == 3
    assert len(encoded_one["joint_stages"]) == 4
    assert encoded_one["joint_stages"][:3] == encoded_five["joint_stages"]
    assert encoded_one["worker_summaries"] == encoded_five["worker_summaries"]
    assert encoded_one["candidates"] == encoded_five["candidates"]
