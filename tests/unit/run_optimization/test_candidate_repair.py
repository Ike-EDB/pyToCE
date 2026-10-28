"""Feedstock-family eligibility, repair-result transport and candidate identities."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from candidate_worker_fixtures import _planner, _result_and_acc_metrics, _worker_contexts
from intervention_fixtures import _circularity_parent, _parent_lifetimes, _parent_stock

from pytoce.run_mfa.scenario.scenario_contract import ScenarioEvaluation
from pytoce.run_optimization.calculation.candidate_repair import (
    candidate_repair_family_identity,
    prepare_repair_family_context,
    repair_reference_from_result,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerPlanner,
    CandidateWorkerRequestBridge,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
    _assignment_frame,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    CandidateWorkerMetricScope,
    decode_candidate_repair_reference,
    encode_candidate_repair_reference,
    encode_candidate_worker_result,
)
from pytoce.run_optimization.runtime.candidate_results import (
    CandidateResultStore,
)


@pytest.fixture
def family_planner() -> CandidateWorkerPlanner:
    """Use the canonical annual trajectory owners with a complete small parent."""
    planner = _planner()
    stock, lifetime, circularity = _parent_stock(), _parent_lifetimes(), _circularity_parent()
    frames = [
        _assignment_frame(stock, parameter_id="stock_projection", values=stock.stock_projection),
        _assignment_frame(lifetime, parameter_id="lifetime_mean", values=lifetime.lifetime_mean),
    ]
    for name in (
        "collection_rate",
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
        "bio_production_rate",
        "daccu_production_rate",
    ):
        frames.append(_assignment_frame(circularity, parameter_id=name, values=circularity[name]))
    compiled = pd.concat(frames, ignore_index=True).assign(value_origin="remind_mfa_default")
    scenario = ScenarioEvaluation("fixture", "scenario::fixture", "SSP2", compiled, (), "table")
    parent = planner.intervention_parent
    return replace(
        planner,
        intervention_parent=replace(
            parent,
            demand_lifetime_parent=replace(
                parent.demand_lifetime_parent, stock_targets=stock, lifetimes=lifetime
            ),
            circularity_parent=replace(parent.circularity_parent, table=circularity),
        ),
        repair_family_context=prepare_repair_family_context(scenario),
    )


def _decision(bio: int = 20, dac: int = 10) -> CircularityDecision:
    """Select one treatment path with separately variable virgin origins."""
    return CircularityDecision(60, 20, 10, 30, 40, 100 - bio - dac, bio, dac)


def _family(planner: CandidateWorkerPlanner, intervention) -> str | None:
    """Calculate the same scientific family identity used by coordinator and worker."""
    return candidate_repair_family_identity(
        intervention,
        planner.repair_family_context,
        worker_authority_identity=planner.worker_authority_identity,
    )


def test_repair_family_binds_annual_physics_and_separates_feedstock(family_planner) -> None:
    """Include the parent, timing, and fixed controls in the repair reuse identity."""
    planner = family_planner
    demand = DemandLifetimeDecision(20, 20)
    first = planner.plan_intervention(demand_lifetime=demand, circularity=_decision())
    family = _family(planner, first)
    assert family is not None
    for bio, dac in ((0, 0), (100, 0), (0, 100), (25, 25)):
        member = planner.plan_intervention(demand_lifetime=demand, circularity=_decision(bio, dac))
        assert _family(planner, member) == family
    for control in (
        replace(_decision(), collection_pct=80),
        replace(_decision(), mechanical_recycling_pct=10, landfill_pct=50),
        replace(_decision(), target_year=2100),
    ):
        other = planner.plan_intervention(demand_lifetime=demand, circularity=control)
        assert _family(planner, other) != family
    parent = planner.plan_intervention(
        demand_lifetime=demand, circularity=CircularityDecision.parent()
    )
    assert _family(planner, parent) is None
    shaped = replace(
        demand, demand_fraction=(demand.demand_fraction[0],) * 2 + demand.demand_fraction[2:]
    )
    assert (
        _family(planner, planner.plan_intervention(demand_lifetime=shaped, circularity=_decision()))
        != family
    )
    assert (
        _family(replace(planner, worker_authority_identity="another::authority"), first) != family
    )


@pytest.mark.parametrize("values", ((101, 0), (-1, 0), (60, 60)))
def test_unadmitted_feedstock_does_not_inherit_a_repair(family_planner, values) -> None:
    """An unsupported partition keeps original admission and its exception precedence."""
    member = family_planner.plan_intervention(
        demand_lifetime=DemandLifetimeDecision(20, 20), circularity=_decision(*values)
    )
    assert _family(family_planner, member) is None


def test_explicit_unchanged_default_requires_its_own_execution_mode(family_planner) -> None:
    """Explicit feedstock alone cannot equate raw source defaults with rewritten controls."""
    planner = family_planner
    table = planner.intervention_parent.circularity_parent.table.copy()
    table["collection_rate"] = 0.6
    planner = replace(
        planner,
        intervention_parent=replace(
            planner.intervention_parent,
            circularity_parent=replace(planner.intervention_parent.circularity_parent, table=table),
        ),
    )
    first = planner.plan_intervention(
        demand_lifetime=DemandLifetimeDecision(0, 0), circularity=_decision()
    )
    parent_values = {row[:-1]: float(row[-1]) for row in first.assignment_records}
    context = replace(planner.repair_family_context, parent_values=parent_values)
    planner = replace(planner, repair_family_context=context)
    assert _family(planner, first) is None
    changed = planner.plan_intervention(
        demand_lifetime=DemandLifetimeDecision(0, 0), circularity=_decision(25, 25)
    )
    assert _family(planner, changed) is None
    # An authored non-feedstock parent already uses translated source updates;
    # its zero-demand case has only one path, but overlay diagnostic modes differ.
    planner = replace(
        planner, repair_family_context=replace(context, parent_nonfeedstock_authored=True)
    )
    assert _family(planner, first) is not None
    assert _family(planner, first) != _family(planner, changed)


def test_cached_basis_releases_siblings_and_survives_individual_result_custody(
    tmp_path: Path, family_planner
) -> None:
    """Use only requested cached results, and retain provenance before stage completion."""
    planner = family_planner
    scopes = (CandidateWorkerMetricScope("integrated_frontier", "metrics::fixture"),)
    context, scoring = _worker_contexts()
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="family::test",
        worker_authority_identity=planner.worker_authority_identity,
    ) as store:
        bridge = CandidateWorkerRequestBridge(
            planner,
            scopes,
            result_store=store,
            worker_scientific_authority_identity="scientific::fixture",
            allocated_cc_context=context,
            search_scoring=scoring,
        )
        interventions = tuple(
            planner.plan_intervention(
                demand_lifetime=DemandLifetimeDecision(20, 20), circularity=_decision(bio, dac)
            )
            for bio, dac in ((20, 10), (25, 25), (100, 0))
        )
        tasks = tuple(bridge.build_task(item, scopes) for item in interventions)
        categorical = bridge.build_task(
            planner.plan_intervention(
                demand_lifetime=DemandLifetimeDecision(0, 0),
                circularity=CircularityDecision.parent(),
            ),
            scopes,
        )
        assert bridge.prepare_dispatch(categorical).task == categorical
        assert bridge.prepare_dispatch(categorical).repair_reference is None
        assert bridge.prepare_dispatch(tasks[1]) is None
        assert bridge.prepare_dispatch(tasks[0]).repair_reference is None
        result = _result_and_acc_metrics(intervention=interventions[0])[0]
        payload = encode_candidate_worker_result(result)
        store.save(tasks[0], payload)
        bridge.read_saved_result(payload)
        dispatch = bridge.prepare_dispatch(tasks[1])
        reference = decode_candidate_repair_reference(dispatch.repair_reference)
        assert reference == repair_reference_from_result(result, tasks[0].repair_family_identity)
        assert encode_candidate_repair_reference(reference) == dispatch.repair_reference
        member_payload = encode_candidate_worker_result(
            _result_and_acc_metrics(intervention=interventions[1])[0]
        )
        store.save(tasks[1], member_payload, repair_provenance=dispatch.repair_provenance)
        manifest = json.loads((store.directory / "00000001/manifest.json").read_bytes())
        assert manifest["repair_reference"]["worker_result_identity"] == result.identity
        assert (
            manifest["repair_reference"]["requested_intervention_identity"]
            == interventions[0].identity
        )
        assert store.read_record(tasks[1].record) == member_payload
        # A completed stage supplies records; only its first family payload is
        # needed when the next requested sibling arrives in a new process.
        restored = CandidateWorkerRequestBridge(
            planner,
            scopes,
            result_store=store,
            worker_scientific_authority_identity="scientific::fixture",
            allocated_cc_context=context,
            search_scoring=scoring,
        )
        restored.restore_task_records((tasks[0].record,))
        future = restored.build_task(interventions[2], scopes)
        assert restored.prepare_dispatch(future).repair_reference == dispatch.repair_reference


@pytest.mark.scientific_integration
def test_real_worker_reuses_repair_without_sharing_feedstock_science(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
    record_property,
) -> None:
    """Compare independently admitted members with compact reuse through real MFA and LCI."""
    from pytoce.run_mfa.scenario.scenario_contract import ScenarioFeasibilityError
    from pytoce.run_optimization.calculation.candidate_worker import (
        CandidateEvaluationWorker,
    )
    from pytoce.run_optimization.calculation.worker_protocol import (
        CandidateWorkerTask,
        decode_candidate_worker_result,
        encode_candidate_worker_task,
    )
    from pytoce.run_optimization.runtime import scheduler
    from tests.integration.candidate_mfa_fixtures import _prepare_worker_workspace
    from tests.unit.run_optimization.test_candidate_worker_initialization import (
        _assert_same_candidate_result,
    )

    initializer, context = _prepare_worker_workspace(
        tmp_path / "workspace",
        remind_mfa_source_root,
        include_generous_case=True,
        include_public_baseline_results=False,
    )
    scheduler._initialize_worker(initializer)
    initialized = scheduler._WORKER
    assert isinstance(initialized, CandidateEvaluationWorker)
    worker = scheduler._evaluate_in_worker
    planner = context.planner
    scopes = (CandidateWorkerMetricScope("integrated_frontier", "metrics::fixture"),)
    for demand in (DemandLifetimeDecision(0, 0), DemandLifetimeDecision(90, 0)):
        first_controls = CircularityDecision(65, 20, 10, 40, 30, 95, 5, 0)
        second_controls = replace(first_controls, fossil_pct=50, biogenic_pct=25, daccu_pct=25)
        first = planner.plan_intervention(demand_lifetime=demand, circularity=first_controls)
        second = planner.plan_intervention(demand_lifetime=demand, circularity=second_controls)
        family = _family(planner, first)
        assert family is not None and _family(planner, second) == family
        original = decode_candidate_worker_result(
            worker(encode_candidate_worker_task(CandidateWorkerTask(first, scopes)))
        )
        second_payload = encode_candidate_worker_task(CandidateWorkerTask(second, scopes))
        independently_evaluated = decode_candidate_worker_result(worker(second_payload))
        reference = encode_candidate_repair_reference(
            repair_reference_from_result(original, family)
        )
        reused = decode_candidate_worker_result(worker(second_payload, reference))
        record_property(
            f"repair_{demand.demand_reduction_pct}_{demand.lifetime_extension_pct}",
            json.dumps(
                {
                    name: {
                        "status": result.repair.status,
                        "mfa": result.repair.mfa_evaluation_count,
                        "lp": result.repair.stock_lp_count,
                        "pool_size": result.repair.pool_candidate_count,
                        "pool_evaluations": result.repair.pool_evaluation_count,
                    }
                    for name, result in (
                        ("basis", original),
                        ("independent_member", independently_evaluated),
                        ("reused_member", reused),
                    )
                }
            ),
        )
        assert reused.identity == independently_evaluated.identity
        assert reused.repair.stock_lp_count == reused.repair.pool_evaluation_count == 0
        assert reused.repair.mfa_evaluation_count == int(reused.scores is not None)
        # Work metadata differs by design; every scientific field/table/array
        # remains compared through the established complete-result assertion.
        normalized = replace(
            reused,
            repair=replace(
                reused.repair,
                mfa_evaluation_count=independently_evaluated.repair.mfa_evaluation_count,
                stock_lp_count=independently_evaluated.repair.stock_lp_count,
                pool_evaluation_count=independently_evaluated.repair.pool_evaluation_count,
            ),
        )
        _assert_same_candidate_result(normalized, independently_evaluated)
        if reused.scores is not None:
            assert (
                reused.feedstock_requirements_identity != original.feedstock_requirements_identity
            )
        else:
            assert reused.repair.status == "stock_infeasible_fixed_controls"
            # The rejected endpoint must propagate the original feasibility failure.
            with pytest.raises(ScenarioFeasibilityError) as failure:
                initialized._session.calculate_physical(second)
            assert failure.value.diagnostic == independently_evaluated.physical.diagnostic
        # Reject incompatible scientific references during transport/admission.
        wrong = replace(repair_reference_from_result(original, family), family_identity="different")
        with pytest.raises(ValueError, match="different controls"):
            worker(second_payload, encode_candidate_repair_reference(wrong))
