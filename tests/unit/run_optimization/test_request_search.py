from dataclasses import replace
from pathlib import Path
from typing import cast

from candidate_worker_fixtures import (
    _initializer,
    _result,
    _summary,
    _worker_contexts,
)
from candidate_worker_fixtures import (
    _planner as _worker_planner,
)
from intervention_fixtures import (
    _circularity_parent,
    _parent_lifetimes,
    _parent_stock,
    _shaped_decision,
)
from request_search_fixtures import _compliant_evaluation, _planner

from pytoce.run_optimization.calculation.candidate_metrics import (
    CandidateMetricSummary,
)
from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    CompetitiveCandidateBatch,
    InterventionCandidate,
    advance_candidate_search,
    finish_candidate_search,
    initialize_candidate_search,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestBridge,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.regional_metrics import BindingConstraint
from pytoce.run_optimization.calculation.request_search import (
    CANDIDATE_SEARCH_CONFIGURATION_METHOD_IDENTITY,
    CandidateWorkerEvaluator,
    _candidate_evaluation,
    _compatible_metric_scopes,
    _execution_blocks,
    _run_candidate_search,
    _search_block_result,
    _selected_timing_candidates,
    _timing_candidate,
    build_candidate_input_identity,
    build_candidate_worker_metric_scopes,
    build_optimization_request_identity,
    search_configuration_identity,
)
from pytoce.run_optimization.runtime.candidate_results import CandidateResultStore
from pytoce.run_optimization.runtime.scheduler import CandidateProcessPool


def test_search_configuration_identity_records_the_fixed_method() -> None:
    assert search_configuration_identity((10, 5)).startswith(
        CANDIDATE_SEARCH_CONFIGURATION_METHOD_IDENTITY + "::"
    )
    assert len({search_configuration_identity(steps) for steps in ((10, 5), (5,), (10, 5, 1))}) == 3


def test_candidate_reuse_identity_binds_science_and_not_outer_request_labels() -> None:
    """Only equal MFA/scoring/metric inputs may share completed endpoint calculations."""
    planner = _worker_planner()
    scopes = build_candidate_worker_metric_scopes(("integrated_frontier",))
    identity = build_candidate_input_identity(
        planner=planner,
        worker_scientific_authority_identity="science::one",
        search_scoring_identity="scoring::one",
        metric_scopes=scopes,
    )
    alternatives = (
        (replace(planner, identity="planner::other"), "science::one", "scoring::one", scopes),
        (
            replace(planner, worker_authority_identity="worker::other"),
            "science::one",
            "scoring::one",
            scopes,
        ),
        (planner, "science::other", "scoring::one", scopes),
        (planner, "science::one", "scoring::other", scopes),
        (planner, "science::one", "scoring::one", scopes[:1]),
        (
            planner,
            "science::one",
            "scoring::one",
            (replace(scopes[0], metric_registry_identity="metrics::other"), *scopes[1:]),
        ),
    )
    assert all(
        build_candidate_input_identity(
            planner=other_planner,
            worker_scientific_authority_identity=scientific,
            search_scoring_identity=scoring,
            metric_scopes=other_scopes,
        )
        != identity
        for other_planner, scientific, scoring, other_scopes in alternatives
    )
    # Search labels remain distinct while the evaluation-input authority is the
    # same object used by the cache. No MFA calculation is needed for either.
    requests = {
        build_optimization_request_identity(
            foreground_scenario="parent",
            allocated_cc_result="capacity",
            optimization_lcia_selection="ef31",
            search_scoring_identity="scoring::one",
            readable_request_selection=selection,
            planner=planner,
            requested_blocks=("integrated_frontier",),
        )
        for selection in ("five_point_product", "finer_product")
    }
    assert len(requests) == 2
    assert identity == build_candidate_input_identity(
        planner=planner,
        worker_scientific_authority_identity="science::one",
        search_scoring_identity="scoring::one",
        metric_scopes=scopes,
    )


def test_nonintegrated_request_executes_only_its_requested_blocks() -> None:
    """Run the requested single-group search."""

    assert _execution_blocks(("demand_lifetime_frontier",)) == ("demand_lifetime_frontier",)
    assert _execution_blocks(("circularity_feedstock_anchor",)) == ("circularity_feedstock_anchor",)


def test_timing_candidate_retimes_active_circularity_without_activating_demand() -> None:
    """Move only the active circularity target for fixed-year reporting."""

    source = InterventionCandidate(
        "circularity_feedstock_anchor",
        DemandLifetimeDecision(0, 0, 2050),
        CircularityDecision(60, 30, 20, 30, 20, 70, 20, 10, 2050),
    )

    observed = _timing_candidate(source, target_year=2100)

    assert observed is not None
    assert observed.demand_lifetime == DemandLifetimeDecision(0, 0, 2100)
    assert observed.circularity == CircularityDecision(60, 30, 20, 30, 20, 70, 20, 10, 2100)


def test_integrated_request_executes_all_scientific_blocks_once() -> None:
    scopes = build_candidate_worker_metric_scopes(("integrated_frontier",))

    assert tuple(scope.optimization_block for scope in scopes) == (
        "demand_lifetime_frontier",
        "circularity_feedstock_anchor",
        "integrated_frontier",
    )
    assert len({scope.metric_registry_identity for scope in scopes}) == 3


def test_competitive_planning_creates_no_task_or_summary_for_pruned_candidates(
    tmp_path: Path,
) -> None:
    """Prune larger interventions before evaluating or saving a candidate result."""

    planner = _worker_planner()
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
    candidates = tuple(
        InterventionCandidate(
            "demand_lifetime_frontier",
            DemandLifetimeDecision(reduction, 0),
            CircularityDecision.parent(),
        )
        for reduction in (10, 20)
    )
    context, scoring = _worker_contexts()
    scopes = build_candidate_worker_metric_scopes(("demand_lifetime_frontier",))
    with CandidateResultStore.open(
        tmp_path,
        candidate_input_identity="request::pruned",
        worker_authority_identity=planner.worker_authority_identity,
    ) as store:
        bridge = CandidateWorkerRequestBridge(
            planner,
            scopes,
            result_store=store,
            worker_scientific_authority_identity=_result().worker_scientific_authority_identity,
            allocated_cc_context=context,
            search_scoring=scoring,
        )
        with CandidateProcessPool(
            worker_initializer_payload=_initializer(workspace=str(tmp_path))
        ) as pool:
            evaluator = CandidateWorkerEvaluator(
                planner=planner,
                bridge=bridge,
                scheduler=pool,
                result_store=store,
            )
            batch = evaluator.evaluate_competitive(candidates, 0.0)

        # Demand and lifetime have equal weights: 10% and 20% demand reductions
        # alone exceed a compliant unchanged parent's zero intervention distance.
        assert batch.candidate_distances == (
            (candidates[0].identity, 0.05),
            (candidates[1].identity, 0.1),
        )
        assert batch.evaluations == ()
        assert evaluator.candidate_records == ()
        assert evaluator.result_summary_identities == ()
        assert tuple(store.directory.glob("[0-9]*/manifest.json")) == ()


def test_block_result_retains_a_selected_competitive_evaluation() -> None:
    class CompetitiveEvaluator:
        def __init__(self) -> None:
            self.retained: set[str] = set()

        def __call__(
            self,
            candidates: tuple[InterventionCandidate, ...],
        ) -> tuple[CandidateEvaluation, ...]:
            return tuple(_compliant_evaluation(candidate) for candidate in candidates)

        def evaluate_competitive(
            self,
            candidates: tuple[InterventionCandidate, ...],
            distance_ceiling: float,
        ) -> CompetitiveCandidateBatch:
            evaluations = tuple(_compliant_evaluation(candidate) for candidate in candidates)
            distances = tuple(
                (item.candidate.identity, item.intervention_distance) for item in evaluations
            )
            retained = tuple(
                item for item in evaluations if item.intervention_distance <= distance_ceiling
            )
            self.retained.update(item.candidate.identity for item in retained)
            return CompetitiveCandidateBatch(distances, retained)

    evaluator = CompetitiveEvaluator()
    initial = InterventionCandidate(
        "demand_lifetime_frontier",
        DemandLifetimeDecision(10, 0),
        CircularityDecision.parent(),
    )

    state = initialize_candidate_search(
        optimization_block="demand_lifetime_frontier",
        initial_candidates=(initial,),
        evaluate=evaluator,
    )
    state = advance_candidate_search(
        state,
        step_pct=10,
        unpruned=False,
        evaluate=evaluator,
        evaluate_competitive=evaluator.evaluate_competitive,
    )
    result = _search_block_result(
        "demand_lifetime_frontier",
        finish_candidate_search(state, endpoint_refinement_steps_pct=(10, 5)),
        state.evaluations,
    )

    assert result.selected_candidates == (
        InterventionCandidate(
            "demand_lifetime_frontier",
            DemandLifetimeDecision(0, 0),
            CircularityDecision.parent(),
        ),
    )
    assert result.selected_candidates[0].identity in evaluator.retained
    assert result.selected_states[0].candidate_identity == result.selected_candidates[0].identity


def test_metric_scopes_follow_the_intervention_families_in_an_integrated_request() -> None:
    scopes = build_candidate_worker_metric_scopes(("integrated_frontier",))
    demand_lifetime = InterventionCandidate(
        "demand_lifetime_frontier",
        DemandLifetimeDecision(25, 25),
        CircularityDecision.parent(),
    )
    circularity_feedstock = InterventionCandidate(
        "circularity_feedstock_anchor",
        DemandLifetimeDecision(0, 0),
        CircularityDecision(50, 25, 25, 25, 25, 50, 25, 25),
    )
    integrated = InterventionCandidate(
        "integrated_frontier",
        DemandLifetimeDecision(25, 25),
        CircularityDecision(50, 25, 25, 25, 25, 50, 25, 25),
    )
    parent = InterventionCandidate(
        "integrated_frontier",
        DemandLifetimeDecision(0, 0),
        CircularityDecision.parent(),
    )

    assert tuple(
        item.optimization_block for item in _compatible_metric_scopes(demand_lifetime, scopes)
    ) == ("demand_lifetime_frontier", "integrated_frontier")
    assert tuple(
        item.optimization_block for item in _compatible_metric_scopes(circularity_feedstock, scopes)
    ) == ("circularity_feedstock_anchor", "integrated_frontier")
    assert tuple(
        item.optimization_block for item in _compatible_metric_scopes(integrated, scopes)
    ) == ("integrated_frontier",)
    assert _compatible_metric_scopes(parent, scopes) == scopes


def test_request_identity_changes_only_with_an_explicit_parent_or_selection() -> None:
    arguments = {
        "foreground_scenario": "frozen_2019_default_baseline",
        "allocated_cc_result": "pyaesa_h12",
        "optimization_lcia_selection": "ef31",
        "search_scoring_identity": "search-scoring",
        "readable_request_selection": (
            "blocks_demand_lifetime_circularity_feedstock_integrated__cases_all"
        ),
        "planner": _planner(),
        "requested_blocks": ("integrated_frontier",),
    }
    identity = build_optimization_request_identity(**arguments)  # type: ignore[arg-type]
    repeated = build_optimization_request_identity(**arguments)  # type: ignore[arg-type]
    changed = build_optimization_request_identity(
        **{**arguments, "search_scoring_identity": "different-scoring"}  # type: ignore[arg-type]
    )

    assert repeated == identity
    assert changed != identity
    for alternative in (
        {"initial_grid_step_pct": 50},
        {"endpoint_refinement_steps_pct": (10, 5, 1)},
        {"continuation_parent_identity": "prior::completed"},
    ):
        assert (
            build_optimization_request_identity(
                **{**arguments, **alternative}  # type: ignore[arg-type]
            )
            != identity
        )


def test_timing_recalculation_changes_only_active_target_completion() -> None:
    """Build one later trajectory per endpoint, retaining all selection roles."""
    parent = InterventionCandidate(
        "demand_lifetime_frontier",
        DemandLifetimeDecision(0, 0),
        CircularityDecision.parent(),
    )
    active = InterventionCandidate(
        "demand_lifetime_frontier",
        _shaped_decision(25, 50),
        CircularityDecision.parent(),
    )

    delayed, roles = _selected_timing_candidates(
        (parent, active),
        selection_identities={
            parent.identity: ("parent_state",),
            active.identity: ("state_z", "state_a"),
        },
    )

    assert len(delayed) == 1
    assert tuple(item.identity for item in delayed) == tuple(sorted(roles))
    assert {item.target_year for item in delayed} == {2100}
    for item in delayed:
        assert item.optimization_block == active.optimization_block
        assert item.demand_lifetime == DemandLifetimeDecision(25, 50, item.target_year)
        assert item.circularity == CircularityDecision.parent(target_year=item.target_year)
        assert roles[item.identity] == ["state_a", "state_z"]


def test_candidate_evaluation_projects_one_real_joint_state() -> None:
    result = _result()
    summary = _summary(result)
    candidate = InterventionCandidate(
        "integrated_frontier",
        result.intervention.demand_lifetime_evaluation_projection.decision,
        result.intervention.circularity_evaluation_projection.decision,
    )
    entry = CandidateMetricSummary(
        identity="entry::joint",
        physical_candidate_identity=result.physical.physical_candidate_identity,
        intervention_authority_identity=result.intervention.identity,
        physical_parent_identity=result.intervention.physical_parent_identity,
        distance_inputs_identity=result.intervention.distance_inputs.identity,
        demand_lifetime_parent_identity=result.intervention.demand_lifetime_parent_identity,
        demand_lifetime_trajectory_identity=(
            result.intervention.demand_lifetime_trajectory_identity
        ),
        circularity_trajectory_identity=result.intervention.circularity_trajectory_identity,
        target_year=2050,
        demand_reduction_fraction=0.25,
        lifetime_extension_fraction=0.5,
        collection_distance=0.1,
        treatment_share_distance=0.2,
        feedstock_share_distance=0.3,
        evaluation_identity="evaluation::joint",
        metric_identity="metric::joint",
        background_scenario="remind_ssp2_ndc",
        allocated_cc_case="p50",
        allocated_cc_context_identity="acc::context",
        optimization_item_identity="state::integrated::ndc::p50",
        comparison_scope_identity="scope::joint",
        optimization_block="integrated_frontier",
        is_zero_intervention_parent=False,
        intervention_distance_identity="distance::joint",
        intervention_distance=0.31,
        regions=("EUR",),
        all_region_joint_compliant=True,
        fully_compliant_regions=("EUR",),
        fully_compliant_impact_targets=("annual::ef31::climate_change",),
        compliant_region_count=1,
        fully_compliant_impact_target_count=1,
        compliant_region_impact_group_count=1,
        compliant_coordinate_count=82,
        region_count=1,
        impact_target_count=1,
        region_impact_group_count=1,
        coordinate_count=82,
        z_worst=0.9,
        binding_slack=1.0 - 0.9,
        maximum_positive_exceedance=0.0,
        balanced_mean_positive_exceedance=0.0,
        binding_constraints=(
            BindingConstraint("EUR", "annual", "ef31", "climate_change", 2050, 0.9),
        ),
    )
    summary = replace(summary, member_metrics=(entry,))

    evaluation = _candidate_evaluation(candidate, summary)
    assert len(evaluation.state_metrics) == 1
    state = evaluation.state_metrics[0]
    assert state.state_identity == entry.optimization_item_identity
    assert state.background_scenario == "remind_ssp2_ndc"
    assert state.binding_slack == 1.0 - 0.9
    assert state.binding_coordinates[0][:5] == (
        "EUR",
        "annual",
        "ef31",
        "climate_change",
        2050,
    )


def test_standalone_blocks_do_not_execute_unrequested_search_groups() -> None:
    """The terminal parent bounds each single-block request to its own search."""

    class ParentEvaluator:
        """Supply synthetic compliant metrics to the production search and ranking owners."""

        candidate_records = ()
        result_summary_identities = ()

        def __call__(
            self, candidates: tuple[InterventionCandidate, ...]
        ) -> tuple[CandidateEvaluation, ...]:
            """Return the known compliant parent to the real search and ranking owners."""
            return tuple(_compliant_evaluation(candidate) for candidate in candidates)

    for block in ("demand_lifetime_frontier", "circularity_feedstock_anchor"):
        result = _run_candidate_search(
            request_identity="standalone::" + block,
            requested_blocks=(block,),  # type: ignore[arg-type]
            evaluator=cast(CandidateWorkerEvaluator, ParentEvaluator()),
            candidate_input_identity="inputs::same",
            initial_grid_step_pct=50,
        )
        assert len(result.block_results) == 1
        assert result.block_results[0].optimization_block == block
        assert result.block_results[0].selected_candidates[0].is_exact_parent
