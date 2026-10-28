"""Selected candidate states shared by public result-table owner tests."""

from dataclasses import replace

import pandas as pd

from pytoce.prepare_backgrounds.preparation.premise import ACTIVE_BACKGROUND_SCENARIOS
from pytoce.run_lca.calculation.foreground import (
    OPTIMIZATION_ACTIVATION_COLUMNS,
    OPTIMIZATION_ACTIVATION_DELTA_COLUMNS,
)
from pytoce.run_lca.calculation.pa_receiving import (
    PaDisposalEvent,
    PaRecyclingEvent,
    PaSpinningEvent,
)
from pytoce.run_lca.calculation.providers import PreparedProviderProduct
from pytoce.run_optimization.calculation.candidate_acc import (
    CandidateACCMetrics,
    calculate_candidate_acc_metrics,
    prepare_candidate_acc_context,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    _optimization_scores,
    build_response_parent_score_projection,
    prepare_search_scoring,
)
from pytoce.run_optimization.calculation.candidate_search import (
    CandidateRefinement,
    CandidateSelection,
    CandidateStateCheck,
    InterventionCandidate,
    SelectedCandidateState,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestContext,
    candidate_worker_result_summary,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
)
from pytoce.run_optimization.calculation.pa_candidate_projection import (
    build_pa_candidate_projection,
    calculate_pa_candidate_projection,
)
from pytoce.run_optimization.calculation.request_search import (
    RequestSearchResult,
    SearchBlockResult,
    SearchExecutionProducts,
    SelectedCandidateEvaluation,
    _candidate_evaluation,
)
from pytoce.run_optimization.calculation.response_bank import (
    project_optimization_response_bank,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    LoadedOptimizationResponseBank,
)
from pytoce.run_optimization.calculation.selected_reporting import (
    BaselineLCAResult,
    BaselineLCAResultRecord,
)
from pytoce.run_optimization.calculation.worker_inputs import _initializer_with_pa_parent
from pytoce.run_optimization.calculation.worker_protocol import (
    CandidateWorkerResultSummary,
    _selected_candidate_result,
    candidate_worker_result_identity,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256
from tests.unit.run_optimization.candidate_worker_fixtures import (
    _infeasible_result,
    _initializer,
    _intervention,
    _planner,
    _result,
    _result_and_acc_metrics,
    _summary,
    _worker_contexts,
)
from tests.unit.run_optimization.intervention_fixtures import _circularity_parent, _shaped_decision
from tests.unit.run_optimization.response_projection_fixtures import (
    _allocated_cc_selection,
    _response_projection,
)


def _integrated_candidate(*, target_year: int = 2050) -> InterventionCandidate:
    """Return one joint demand, lifetime, and circularity candidate."""

    return InterventionCandidate(
        "integrated_frontier",
        _shaped_decision(
            25,
            50,
            target_year,
        ),
        CircularityDecision(10, 40, 20, 30, 10, 70, 20, 10, target_year),
    )


def _selected(candidate: InterventionCandidate) -> SelectedCandidateEvaluation:
    """Wrap a complete worker result as a selected candidate."""

    result = _result()
    summary = _summary(result)
    selected_result = _selected_candidate_result(
        result,
        summary,
        selection_identities=("integrated_frontier::member::p50::ef31::climate::all",),
    )
    return SelectedCandidateEvaluation(
        candidate, selected_result.selection_identities, selected_result
    )


def _selected_acc_reconstruction_fixture() -> tuple[
    SelectedCandidateEvaluation,
    CandidateWorkerRequestContext,
    tuple[CandidateACCMetrics, ...],
]:
    """Return selected compact metrics and their exact reconstruction context."""

    selected = _selected(_integrated_candidate())
    result = selected.selected_result.result
    acc_context, scoring = _worker_contexts()
    planner = _planner()
    parent = planner.intervention_parent
    planner = replace(
        planner,
        intervention_parent=replace(
            parent,
            circularity_parent=replace(parent.circularity_parent, table=_circularity_parent()),
        ),
    )
    context = CandidateWorkerRequestContext(
        planner=planner,
        allocated_cc_context=acc_context,
        search_scoring=scoring,
        result_lcia_selection_identity="result-selection",
        result_parent=scoring.optimization_parent,
        result_projection=_response_projection("result"),
        result_pa_projection=scoring.pa_projection,
        result_pa_parent=scoring.pa_parent,
        worker_scientific_authority_identity=result.worker_scientific_authority_identity,
        worker_initializer_payload=_initializer_with_pa_parent(_initializer(), scoring.pa_parent),
    )
    scope = result.metric_scopes[0]
    detailed = calculate_candidate_acc_metrics(
        result.scores,
        acc_context,
        environmental_parent_members=scoring.optimization_parent.member_parents,
        scoring_context_identity=scoring.identity,
        intervention_authority_identity=result.intervention.identity,
        distance_inputs=result.intervention.distance_inputs,
        is_zero_intervention_parent=result.intervention.uses_exact_parent,
        optimization_block=scope.optimization_block,
        metric_registry_identity=scope.metric_registry_identity,
    )
    return selected, context, detailed


def _search_products(
    selected_2050: SelectedCandidateEvaluation,
    timing: tuple[SelectedCandidateEvaluation, ...],
    *,
    summary: CandidateWorkerResultSummary | None = None,
) -> SearchExecutionProducts:
    """Represent one completed selection whose metric rows come from the score owner."""
    if summary is None:
        summary = _summary(selected_2050.selected_result.result)
    evaluation = _candidate_evaluation(selected_2050.candidate, summary)
    metric = evaluation.state_metrics[0]
    selection = SelectedCandidateState(
        state_identity=metric.state_identity,
        candidate_identity=metric.candidate_identity,
        status="intervention_compliant"
        if metric.all_region_joint_compliant
        else "selected_noncompliant",
    )
    verification = CandidateStateCheck(
        state_identity=metric.state_identity,
        candidate_identity=metric.candidate_identity,
        selected_is_exact_parent=False,
        selected_has_non_grid_baseline_component=False,
        declared_neighborhood_verified=True,
        selection_stable=True,
    )
    refinement = CandidateRefinement(
        identity="refinement::fixture",
        optimization_block="integrated_frontier",
        evaluation_count=1,
        reused_candidate_count=0,
        selection=CandidateSelection(
            identity="selection::fixture",
            states=(selection,),
            selected_candidate_identities=(metric.candidate_identity,),
        ),
        state_verification=(verification,),
        passes=(),
        final_neighbor_candidate_identities=(),
    )
    block = SearchBlockResult(
        optimization_block="integrated_frontier",
        refinement=refinement,
        selected_candidates=(selected_2050.candidate,),
        selected_states=(metric,),
    )
    return SearchExecutionProducts(
        search=RequestSearchResult(identity="search::fixture", block_results=(block,)),
        selected_2050=(selected_2050,),
        selected_timing=timing,
        physical_repair_elapsed_seconds=0.0,
        environmental_evaluation_elapsed_seconds=0.0,
        mfa_evaluation_count=0,
        stock_lp_count=0,
        environmental_evaluation_count=0,
        repair_status_counts=(),
        worker_result_set_identity="worker-results::fixture",
        request_coordinate_count=1 + len(timing),
        request_unique_physical_candidate_count=1 + len(timing),
        request_physical_candidate_deduplication_count=0,
    )


def selected_pa_reporting_fixture(
    bank: LoadedOptimizationResponseBank,
    providers: PreparedProviderProduct,
    dump_shares: dict[str, float],
    *,
    spinning: tuple[PaSpinningEvent, ...],
    recycling: tuple[PaRecyclingEvent, ...],
    disposal: tuple[PaDisposalEvent, ...],
) -> tuple[SearchExecutionProducts, CandidateWorkerRequestContext, tuple[BaselineLCAResult, ...]]:
    """Combine PA search and reporting calculations.

    The four invented background members deliberately share the same admitted
    native graph. Search stores only the fixture EF climate method; reporting
    adds the independently characterized PB fixture method. The parent MFA result
    has no PA or ordinary activations, and the changed candidate has the supplied
    real MFA-derived PA events. Only selected reporting is exercised here, not
    a new optimization search or the infeasible delayed-path admission owner.
    """
    backgrounds = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)

    def for_each_member(frame: pd.DataFrame) -> pd.DataFrame:
        """Reuse one computed invented supply chain under each background scenario key."""
        return pd.concat(
            [frame.assign(background_scenario=member) for member in backgrounds],
            ignore_index=True,
        )

    bank = replace(
        bank,
        identity="bank::actual-native-four-member-fixture",
        provider_unit_responses=for_each_member(bank.provider_unit_responses),
        native_unit_responses=for_each_member(bank.native_unit_responses),
        timex_year_weights=for_each_member(bank.timex_year_weights),
    )
    search_registry = bank.method_registry.loc[bank.method_registry.method_id.eq("ef31")]
    search = project_optimization_response_bank(bank, search_registry, role="optimization")
    climate = project_optimization_response_bank(bank, search_registry, role="mandatory_climate")
    reporting = project_optimization_response_bank(bank, bank.method_registry, role="result")
    member_providers = {member: providers for member in backgrounds}
    search_pa = build_pa_candidate_projection(
        search, member_providers, dump_shares, prepared_receiving_identity="receiving::fixture"
    )
    reporting_pa = build_pa_candidate_projection(
        reporting, member_providers, dump_shares, prepared_receiving_identity="receiving::fixture"
    )
    search_pa_parent = calculate_pa_candidate_projection(
        search_pa, spinning_events=(), recycling_events=(), disposal_events=()
    )
    reporting_pa_parent = calculate_pa_candidate_projection(
        reporting_pa, spinning_events=(), recycling_events=(), disposal_events=()
    )
    _unused_selected, context, _unused_metrics = _selected_acc_reconstruction_fixture()
    physical_parent = replace(
        context.physical_parent,
        provider_route_identity=bank.provider_route_identity,
        activations=pd.DataFrame(columns=OPTIMIZATION_ACTIVATION_COLUMNS),
    )
    optimization_parent = build_response_parent_score_projection(
        physical_parent,
        search,
        pa_parent=search_pa_parent,
        pa_method_registry=search_pa.method_registry,
    )
    scoring = prepare_search_scoring(
        physical_parent=physical_parent,
        optimization_parent=optimization_parent,
        climate_parent=build_response_parent_score_projection(
            physical_parent,
            climate,
            pa_parent=search_pa_parent,
            pa_method_registry=search_pa.method_registry,
        ),
        optimization_projection=search,
        climate_projection=climate,
        pa_projection=search_pa,
        pa_parent=search_pa_parent,
    )
    context = replace(
        context,
        allocated_cc_context=prepare_candidate_acc_context(
            _allocated_cc_selection(
                method_compatibility_identity=scoring.optimization_projection.method_compatibility_identity,
                climate_compatibility_identity=scoring.climate_projection.method_compatibility_identity,
            )
        ),
        search_scoring=scoring,
        result_parent=build_response_parent_score_projection(
            physical_parent,
            reporting,
            pa_parent=reporting_pa_parent,
            pa_method_registry=reporting_pa.method_registry,
        ),
        result_projection=reporting,
        result_pa_projection=reporting_pa,
        result_pa_parent=reporting_pa_parent,
        result_lcia_selection_identity="result-selection::ef31-pb-fixture",
    )
    delta = pd.DataFrame(columns=OPTIMIZATION_ACTIVATION_DELTA_COLUMNS)
    cached_pa = calculate_pa_candidate_projection(
        search_pa, spinning_events=spinning, recycling_events=recycling, disposal_events=disposal
    )
    worker, _unused_metric = _result_and_acc_metrics(
        activation_delta=delta,
        pa_spinning_events=spinning,
        pa_recycling_events=recycling,
        pa_disposal_events=disposal,
        pa_scores=cached_pa,
    )
    # Replace the compact worker fixture's example scores with the actual
    # native/direct calculation, then derive the search metrics from those values.
    worker = replace(
        worker, scores=_optimization_scores(worker.physical, scoring, delta, cached_pa)
    )
    worker = replace(worker, identity=candidate_worker_result_identity(worker))
    summary = candidate_worker_result_summary(
        worker, allocated_cc_context=context.allocated_cc_context, search_scoring=scoring
    )
    candidate = _integrated_candidate()
    state = _candidate_evaluation(candidate, summary).state_metrics[0]
    selected = _selected_candidate_result(
        worker,
        summary,
        selection_identities=(state.state_identity,),
    )
    evaluation = SelectedCandidateEvaluation(candidate, selected.selection_identities, selected)
    timing = []
    for year in (2100,):
        delayed = _infeasible_result(intervention=_intervention(target_year=year))
        delayed_summary = candidate_worker_result_summary(
            delayed, allocated_cc_context=context.allocated_cc_context, search_scoring=scoring
        )
        delayed_candidate = _integrated_candidate(target_year=year)
        delayed_selected = _selected_candidate_result(
            delayed,
            delayed_summary,
            selection_identities=(state.state_identity,),
        )
        timing.append(
            SelectedCandidateEvaluation(
                delayed_candidate, delayed_selected.selection_identities, delayed_selected
            )
        )
    products = _search_products(evaluation, tuple(timing), summary=summary)
    # Empty parent inventories have no stage burdens. Retain the real stage
    # schema so the reporting writer must add the complete candidate PA stream.
    annual_parent = reporting_pa_parent.stage_scores.drop(
        columns=["characterization_background_scenario", "factor_identity"]
    )
    cumulative_parent = annual_parent.copy()
    parents = tuple(
        BaselineLCAResult(
            record=BaselineLCAResultRecord(
                foreground_scenario="frozen_2019_default_baseline",
                background_scenario=member,
                result_identity=f"empty-baseline::{member}",
                result_manifest_sha256="b" * 64,
                annual_life_cycle_stage_scores_identity=canonical_table_sha256(annual_parent),
                life_cycle_stage_scores_identity=canonical_table_sha256(cumulative_parent),
            ),
            annual_life_cycle_stage_scores=annual_parent,
            life_cycle_stage_scores=cumulative_parent,
        )
        for member in backgrounds
    )
    return products, context, parents
