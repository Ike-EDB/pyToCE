"""Calculate selected optimization results from the required baseline LCA results."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import fsum
from typing import TYPE_CHECKING, Final, cast

import pandas as pd

from pytoce.run_optimization.calculation.candidate_acc import (
    CandidateACCMetrics,
    calculate_candidate_acc_metrics,
)
from pytoce.run_optimization.calculation.candidate_evaluation import (
    SelectedCandidateCharacterization,
    characterize_selected_physical_candidates,
)
from pytoce.run_optimization.calculation.candidate_search import (
    CANDIDATE_REFINEMENT_METHOD_IDENTITY,
    CANDIDATE_SELECTION_METHOD_IDENTITY,
    CandidateStateCheck,
    CandidateStateMetrics,
    SelectedCandidateState,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestContext,
    CandidateWorkerResult,
)
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    STOCK_PROPOSAL_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.integrated_search import (
    INTEGRATED_SEARCH_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.interventions import (
    build_circularity_trajectories,
)
from pytoce.run_optimization.calculation.pa_candidate_projection import (
    PaCandidateScores,
    calculate_pa_candidate_projection,
)
from pytoce.run_optimization.calculation.regional_metrics import (
    RegionalMetrics,
    transgressing_constraints,
)
from pytoce.run_optimization.calculation.request_search import (
    SearchExecutionProducts,
    SelectedCandidateEvaluation,
)
from pytoce.run_optimization.calculation.selected_reporting import (
    BaselineLCAResult,
    SelectedReportingState,
)
from pytoce.run_optimization.calculation.selected_stage_reporting import (
    stream_selected_stage_reporting,
)
from pytoce.run_optimization.runtime.completed_search import encode_completed_search
from pytoce.run_optimization.runtime.result_store import (
    StagedTable,
    TableStreamWriter,
)
from pytoce.shared.runtime.filesystem import (
    canonical_json_bytes,
    canonical_table_sha256,
    payload_sha256,
)

if TYPE_CHECKING:
    from pytoce.run_mfa.model.optimization_parents import OptimizationCircularityParent
    from pytoce.run_optimization.calculation.candidate_projection import (
        OptimizationScores,
    )

SELECTED_RESULTS_METHOD_IDENTITY: Final = "pytoce_selected_results_v17"


@dataclass(frozen=True, slots=True)
class SelectedResultTables:
    """Complete selected-result tables and identity metadata."""

    identity: str
    method_identity: str
    parent_identities: Mapping[str, str]
    tables: Mapping[str, pd.DataFrame | StagedTable]
    summary: Mapping[str, object]


def _selected_result_identity(
    method_identity: str,
    parent_identities: Mapping[str, str],
    summary: Mapping[str, object],
) -> str:
    """Identify selected result tables from their method, parents, and content summary."""

    return (
        method_identity
        + "::"
        + payload_sha256(
            {
                "method_identity": method_identity,
                "parent_identities": dict(sorted(parent_identities.items())),
                "summary": summary,
            }
        )
    )


def build_selected_result_tables(
    products: SearchExecutionProducts,
    *,
    request_context: CandidateWorkerRequestContext,
    baseline_lca_results: Sequence[BaselineLCAResult],
    stage_writer: TableStreamWriter,
    stage_progress: Callable[[int, int], None],
) -> SelectedResultTables:
    """Report selected 2050 results and the MFA outcomes of their delayed trajectories."""

    selected = (*products.selected_2050, *products.selected_timing)
    feasible = tuple(
        item
        for item in selected
        if item.selected_result.result.physical.status == "physical_feasible"
    )
    selected_acc_metrics = expand_detailed_acc_metrics(
        tuple(item.selected_result.result for item in feasible),
        request_context,
    )
    return _build_result_tables(
        selected,
        feasible_evaluations=feasible,
        states=_selected_reporting_states(products, selected_acc_metrics),
        execution=products,
        worker_result_set_identity=products.worker_result_set_identity,
        request_context=request_context,
        baseline_lca_results=baseline_lca_results,
        selected_acc_metrics=selected_acc_metrics,
        method_identity=SELECTED_RESULTS_METHOD_IDENTITY,
        stage_writer=stage_writer,
        stage_progress=stage_progress,
    )


def expand_detailed_acc_metrics(
    evaluations: Sequence[CandidateWorkerResult],
    request_context: CandidateWorkerRequestContext,
) -> dict[str, tuple[CandidateACCMetrics, ...]]:
    """Reconstruct selected sustainability metrics using supplied allocated carrying capacities."""

    outputs: dict[str, tuple[CandidateACCMetrics, ...]] = {}
    scoring = request_context.search_scoring
    for result in evaluations:
        physical = result.physical.physical_candidate_identity
        prior = outputs.get(physical)
        if prior is not None:
            continue
        scores = cast("OptimizationScores", result.scores)
        reconstructed: list[CandidateACCMetrics] = []
        for scope in result.metric_scopes:
            reconstructed.extend(
                calculate_candidate_acc_metrics(
                    scores,
                    request_context.allocated_cc_context,
                    environmental_parent_members=(scoring.optimization_parent.member_parents),
                    scoring_context_identity=scoring.identity,
                    intervention_authority_identity=result.intervention.identity,
                    distance_inputs=result.intervention.distance_inputs,
                    is_zero_intervention_parent=result.intervention.uses_exact_parent,
                    optimization_block=scope.optimization_block,
                    metric_registry_identity=scope.metric_registry_identity,
                )
            )
        outputs[physical] = tuple(reconstructed)
    return outputs


def _selected_reporting_states(
    products: SearchExecutionProducts,
    selected_acc_metrics: Mapping[str, tuple[CandidateACCMetrics, ...]],
) -> tuple[SelectedReportingState, ...]:
    """Build reporting states for selected candidates at 2050 and 2100."""
    selection_by_identity = {
        item.state_identity: item
        for block in products.search.block_results
        for item in block.refinement.selection.states
    }
    verification_by_identity = {
        item.state_identity: item
        for block in products.search.block_results
        for item in block.refinement.state_verification
    }
    candidates_by_identity = {item.candidate.identity: item for item in products.selected_2050}
    timing_by_state_year: dict[tuple[str, int], SelectedCandidateEvaluation] = {}
    for timing in products.selected_timing:
        year = timing.candidate.target_year
        for state_identity in timing.selection_identities:
            timing_by_state_year[(state_identity, year)] = timing
    states: list[SelectedReportingState] = []
    for block in products.search.block_results:
        for metric in block.selected_states:
            selected = candidates_by_identity[metric.candidate_identity]
            if selected.candidate.is_exact_parent:
                timing_2100 = selected
            else:
                timing_2100 = timing_by_state_year[(metric.state_identity, 2100)]
            states.append(
                _reporting_state(
                    metric,
                    selection_by_identity[metric.state_identity],
                    verification_by_identity[metric.state_identity],
                    selected,
                    timing_2100,
                    selected_acc_metrics,
                )
            )
    return tuple(states)


def _reporting_state(
    metric: CandidateStateMetrics,
    selection: SelectedCandidateState,
    verification: CandidateStateCheck,
    selected: SelectedCandidateEvaluation,
    timing_2100: SelectedCandidateEvaluation,
    selected_acc_metrics: Mapping[str, tuple[CandidateACCMetrics, ...]],
) -> SelectedReportingState:
    """Combine one selected candidate with its timing and compliance evidence."""
    regional_metrics = _selected_state_regional_metrics(
        selected,
        metric,
        selected_acc_metrics,
    )
    transgressions = transgressing_constraints(regional_metrics)
    circularity = selected.candidate.circularity.percentages()
    circularity_values = circularity or (None,) * 8
    return SelectedReportingState(
        physical_candidate_identity=(
            selected.selected_result.result.physical.physical_candidate_identity
        ),
        state_identity=metric.state_identity,
        selection_candidate_identity=metric.candidate_identity,
        timing_2100_candidate_identity=timing_2100.candidate.identity,
        timing_2100_physical_candidate_identity=(
            timing_2100.selected_result.result.physical.physical_candidate_identity
        ),
        optimization_block=metric.optimization_block,
        selection_background_scenario=metric.background_scenario,
        allocated_cc_case=metric.allocated_cc_case,
        selection_status=selection.status,
        all_region_joint_compliant=metric.all_region_joint_compliant,
        intervention_distance=metric.intervention_distance,
        binding_slack=metric.binding_slack,
        compliant_region_count=metric.compliant_region_count,
        fully_compliant_impact_target_count=(metric.fully_compliant_impact_target_count),
        compliant_region_impact_group_count=(metric.compliant_region_impact_group_count),
        compliant_coordinate_count=metric.compliant_coordinate_count,
        region_count=metric.region_count,
        impact_target_count=metric.impact_target_count,
        region_impact_group_count=metric.region_impact_group_count,
        coordinate_count=metric.coordinate_count,
        worst_asr=metric.worst_asr,
        maximum_positive_exceedance=metric.maximum_positive_exceedance,
        balanced_mean_positive_exceedance=(metric.balanced_mean_positive_exceedance),
        fully_compliant_regions=metric.fully_compliant_regions,
        fully_compliant_impact_targets=metric.fully_compliant_impact_targets,
        binding_coordinates=metric.binding_coordinates,
        transgressing_coordinates=tuple(
            (
                item.region,
                item.constraint_type,
                item.optimization_lcia_method,
                item.impact_category_id,
                item.year,
                item.asr,
            )
            for item in transgressions
        ),
        selected_is_exact_parent=verification.selected_is_exact_parent,
        selected_has_non_grid_baseline_component=(
            verification.selected_has_non_grid_baseline_component
        ),
        declared_neighborhood_verified=(verification.declared_neighborhood_verified),
        selection_stable=verification.selection_stable,
        demand_reduction_pct=selected.candidate.demand_lifetime.demand_reduction_pct,
        lifetime_extension_pct=selected.candidate.demand_lifetime.lifetime_extension_pct,
        collection_pct=circularity_values[0],
        mechanical_recycling_pct=circularity_values[1],
        chemical_recycling_pct=circularity_values[2],
        incineration_pct=circularity_values[3],
        landfill_pct=circularity_values[4],
        fossil_feedstock_pct=circularity_values[5],
        biogenic_feedstock_pct=circularity_values[6],
        daccu_feedstock_pct=circularity_values[7],
        selection_target_year=2050,
    )


def _selected_state_regional_metrics(
    selected: SelectedCandidateEvaluation,
    metric: CandidateStateMetrics,
    selected_acc_metrics: Mapping[str, tuple[CandidateACCMetrics, ...]],
) -> RegionalMetrics:
    """Index regional metrics for each selected state and background member."""
    physical = selected.selected_result.result.physical.physical_candidate_identity
    return next(
        item.metrics
        for item in selected_acc_metrics[physical]
        if item.metrics.optimization_item_identity == metric.state_identity
    )


def _build_result_tables(
    selected: Sequence[SelectedCandidateEvaluation],
    *,
    feasible_evaluations: Sequence[SelectedCandidateEvaluation],
    states: Sequence[SelectedReportingState],
    execution: SearchExecutionProducts,
    worker_result_set_identity: str,
    request_context: CandidateWorkerRequestContext,
    baseline_lca_results: Sequence[BaselineLCAResult],
    selected_acc_metrics: dict[str, tuple[CandidateACCMetrics, ...]],
    method_identity: str,
    stage_writer: TableStreamWriter,
    stage_progress: Callable[[int, int], None],
) -> SelectedResultTables:
    """Build every table for the selected optimization result.

    Unique feasible candidates are characterized once for the reporting
    LCIA methods. All requested outcomes retain their coordinates and paths;
    environmental and sustainability tables contain the trajectories admitted by the MFA. The
    function builds selection values, annual and cumulative allocated
    carrying capacities and absolute sustainability ratios, annual LCIA results,
    stage-attributed LCIA results, and feedstock requirements by product,
    quantity, and technology route. The stage table is streamed to
    ``stage_writer``; other tables remain in memory. Their table hashes and all
    MFA, LCA, response, search, and worker input identities determine the result
    identity.
    """
    evaluations = tuple(selected)
    selected_states = tuple(states)
    physical_index = {
        item.selected_result.result.physical.physical_candidate_identity: item
        for item in feasible_evaluations
    }
    unique_evaluations = tuple(physical_index.values())
    pa_candidates: dict[str, PaCandidateScores] = {}
    for item in unique_evaluations:
        result = item.selected_result.result
        physical = result.physical
        if (
            physical.physical_state_identity
            == request_context.physical_parent.physical_state_identity
        ):
            pa_scores = request_context.result_pa_parent
        elif request_context.result_pa_projection is request_context.search_scoring.pa_projection:
            pa_scores = cast(PaCandidateScores, result.pa_scores)
        else:
            pa_scores = calculate_pa_candidate_projection(
                request_context.result_pa_projection,
                spinning_events=physical.pa_spinning_events,
                recycling_events=physical.pa_recycling_events,
                disposal_events=physical.pa_disposal_events,
            )
        pa_candidates[physical.physical_candidate_identity] = pa_scores
    characterizations = characterize_selected_physical_candidates(
        tuple(
            (
                item.selected_result.identity,
                item.selected_result.result.physical,
                pa_candidates[item.selected_result.result.physical.physical_candidate_identity],
            )
            for item in unique_evaluations
        ),
        result_lcia_selection_identity=request_context.result_lcia_selection_identity,
        physical_parent=request_context.physical_parent,
        result_parent=request_context.result_parent,
        result_projection=request_context.result_projection,
        pa_parent=request_context.result_pa_parent,
        pa_method_registry=request_context.result_pa_projection.method_registry,
    )
    coordinates = _coordinates_table(evaluations)
    acc = build_acc_table(
        {
            item.selected_result.result.physical.physical_candidate_identity: (
                item.candidate.target_year
            )
            for item in unique_evaluations
        },
        selected_acc_metrics,
    )
    selected_acc_metrics.clear()
    environmental = _environmental_table(characterizations, physical_index)
    del characterizations
    stream_selected_stage_reporting(
        selected_evaluations=tuple(item.selected_result for item in unique_evaluations),
        states=selected_states,
        baseline_lca_results=baseline_lca_results,
        response_projection=request_context.result_projection,
        pa_parent=request_context.result_pa_parent,
        pa_candidates=pa_candidates,
        sink=stage_writer,
        progress=stage_progress,
    )
    stage = stage_writer.finish()
    tables: dict[str, pd.DataFrame | StagedTable] = {
        "selected_coordinates.parquet": coordinates,
        "selected_annual_interventions.parquet": _annual_interventions_table(
            evaluations, request_context.planner.intervention_parent.circularity_parent
        ),
        "selection_roles.parquet": _states_table(selected_states),
        "selected_annual_acc_asr.parquet": acc,
        "selected_environmental_scores.parquet": environmental,
        "selected_life_cycle_stage_scores.parquet": stage,
        "selected_feedstock_requirements.parquet": build_feedstock_requirements_table(
            tuple(
                (
                    item.candidate.target_year,
                    item.selected_result.result,
                )
                for item in unique_evaluations
            )
        ),
    }
    table_identities = {
        name: (
            table.canonical_sha256
            if isinstance(table, StagedTable)
            else canonical_table_sha256(table)
        )
        for name, table in tables.items()
    }
    parents = {
        "candidate_search": execution.search.identity,
        "worker_result_set": worker_result_set_identity,
        "result_lcia_selection": request_context.result_lcia_selection_identity,
        "result_projection": request_context.result_projection.projection_identity,
        "result_response_bank": request_context.result_projection.response_bank_identity,
        **{
            f"baseline_lca_{parent.record.background_scenario}": parent.record.identity
            for parent in baseline_lca_results
        },
    }
    if execution.search.continuation_parent_identity is not None:
        parents["optimization_continuation"] = execution.search.continuation_parent_identity
    summary = _result_summary(
        execution,
        selected_coordinate_count=len(evaluations),
        selected_unique_physical_candidate_count=len(
            {
                item.selected_result.result.physical.physical_candidate_identity
                for item in evaluations
            }
        ),
        states=selected_states,
        target_years=tuple(sorted({item.candidate.target_year for item in evaluations})),
        table_identities=table_identities,
    )
    identity = _selected_result_identity(method_identity, parents, summary)
    return SelectedResultTables(identity, method_identity, parents, tables, summary)


def _result_summary(
    execution: SearchExecutionProducts,
    *,
    selected_coordinate_count: int,
    selected_unique_physical_candidate_count: int,
    states: Sequence[SelectedReportingState],
    target_years: tuple[int, ...],
    table_identities: Mapping[str, str],
) -> dict[str, object]:
    """Summarize candidate counts, refinement steps, and result table identities.

    The summary records request-wide candidate deduplication, selections by
    optimization block, pass counts at each endpoint
    step and proposal family, declared neighborhood checks, target years, and
    the hash of every selected-result table. The summary describes the evaluated
    candidates and declared neighborhood checks.
    """

    refinement_summary = [
        {
            "optimization_block": block.optimization_block,
            "evaluation_count": block.refinement.evaluation_count,
            "reused_candidate_count": block.refinement.reused_candidate_count,
            "unsuccessful_path_search_count": block.refinement.unsuccessful_path_search_count,
            "selected_candidate_count": len(block.selected_candidates),
            "refinement_pass_counts": [
                {
                    "endpoint_step_pct": step_pct,
                    "proposal_family": (
                        "coordinated_and_guided"
                        if coordinated and block.optimization_block == "integrated_frontier"
                        else "scalar_and_coordinated"
                        if coordinated
                        else "scalar"
                        if block.optimization_block == "demand_lifetime_frontier"
                        else "complete_lines"
                    ),
                    "pass_count": sum(
                        item.step_pct == step_pct and item.coordinated == coordinated
                        for item in block.refinement.passes
                    ),
                }
                for step_pct in execution.search.endpoint_refinement_steps_pct
                for coordinated in (False, True)
                if not coordinated or block.optimization_block != "circularity_feedstock_anchor"
            ],
            "final_neighbor_count": len(block.refinement.final_neighbor_candidate_identities),
        }
        for block in execution.search.block_results
    ]
    return {
        "selected_coordinate_count": selected_coordinate_count,
        "selected_unique_physical_candidate_count": selected_unique_physical_candidate_count,
        "state_count": len(states),
        "declared_neighborhood_verified_state_count": sum(
            item.declared_neighborhood_verified for item in states
        ),
        "parent_terminal_state_count": sum(
            item.selection_status == "baseline_compliant" for item in states
        ),
        "not_declared_neighborhood_verified_state_count": sum(
            not item.declared_neighborhood_verified
            and item.selection_status != "baseline_compliant"
            for item in states
        ),
        "request_coordinate_count": execution.request_coordinate_count,
        "request_physical_repair_elapsed_seconds": execution.physical_repair_elapsed_seconds,
        "request_environmental_evaluation_elapsed_seconds": (
            execution.environmental_evaluation_elapsed_seconds
        ),
        "request_mfa_evaluation_count": execution.mfa_evaluation_count,
        "request_stock_lp_count": execution.stock_lp_count,
        "request_environmental_evaluation_count": execution.environmental_evaluation_count,
        "work_accounting": {
            "request_totals": "embodied work in all unique completed candidate results",
            "elapsed_values": "summed worker effort, not wall-clock time",
            "newly_computed_worker_result_count": execution.newly_computed_worker_result_count,
            "reused_worker_result_count": execution.reused_worker_result_count,
            "incremental_physical_repair_elapsed_seconds": (
                execution.incremental_physical_repair_elapsed_seconds
            ),
            "incremental_environmental_evaluation_elapsed_seconds": (
                execution.incremental_environmental_evaluation_elapsed_seconds
            ),
            "incremental_mfa_evaluation_count": execution.incremental_mfa_evaluation_count,
            "incremental_stock_lp_count": execution.incremental_stock_lp_count,
        },
        "request_repair_status_counts": dict(execution.repair_status_counts),
        "request_unique_physical_candidate_count": (
            execution.request_unique_physical_candidate_count
        ),
        "request_physical_candidate_deduplication_count": (
            execution.request_physical_candidate_deduplication_count
        ),
        "refinement": refinement_summary,
        "target_years": list(target_years),
        "search_method": "deterministic_candidate_search",
        "refinement_method": CANDIDATE_REFINEMENT_METHOD_IDENTITY,
        "integrated_search_method": INTEGRATED_SEARCH_METHOD_IDENTITY,
        "selection_method": CANDIDATE_SELECTION_METHOD_IDENTITY,
        "candidate_input_identity": execution.search.candidate_input_identity,
        "continuation_parent_identity": execution.search.continuation_parent_identity,
        "initial_grid_step_pct": execution.search.initial_grid_step_pct,
        "endpoint_refinement_steps_pct": list(execution.search.endpoint_refinement_steps_pct),
        "endpoint_resolution_pct": execution.search.endpoint_refinement_steps_pct[-1],
        "annual_control_resolution_pct": 1,
        "annual_control_reference": "linear_first_demand_only_repair",
        "annual_control_years": list(range(2019, 2101)),
        "conditional_stock_proposal_method": STOCK_PROPOSAL_METHOD_IDENTITY,
        "global_optimum_claim": False,
        "completed_search": (
            None
            if execution.completed_search is None
            else encode_completed_search(execution.completed_search)
        ),
        "table_identities": table_identities,
    }


def _coordinates_table(evaluations: Sequence[SelectedCandidateEvaluation]) -> pd.DataFrame:
    """Format selected intervention percentages and target years for user output."""
    names = (
        "collection_pct",
        "mechanical_recycling_pct",
        "chemical_recycling_pct",
        "incineration_pct",
        "landfill_pct",
        "fossil_pct",
        "biogenic_pct",
        "daccu_pct",
    )
    rows: list[dict[str, object]] = []
    for item in evaluations:
        candidate = item.candidate
        circularity = candidate.circularity.percentages()
        diagnostic = item.selected_result.result.physical.diagnostic
        repair = item.selected_result.result.repair
        row: dict[str, object] = {
            "candidate_identity": candidate.identity,
            "physical_candidate_identity": (
                item.selected_result.result.physical.physical_candidate_identity
            ),
            "selected_result_identity": item.selected_result.identity,
            "physical_status": item.selected_result.result.physical.status,
            "requested_intervention_identity": (
                item.selected_result.result.requested_intervention_identity
            ),
            "realized_intervention_identity": item.selected_result.result.intervention.identity,
            "repair_status": repair.status,
            "mfa_evaluation_count": repair.mfa_evaluation_count,
            "stock_lp_count": repair.stock_lp_count,
            "pool_candidate_count": repair.pool_candidate_count,
            "pool_evaluation_count": repair.pool_evaluation_count,
            "demand_departure_pp": (
                None if repair.demand_departure_pp is None else float(repair.demand_departure_pp)
            ),
            "diagnostic_code": None if diagnostic is None else diagnostic.code,
            "optimization_block": candidate.optimization_block,
            "target_year": candidate.target_year,
            "demand_reduction_pct": candidate.demand_lifetime.demand_reduction_pct,
            "lifetime_extension_pct": candidate.demand_lifetime.lifetime_extension_pct,
        }
        row.update(zip(names, circularity or (pd.NA,) * len(names), strict=True))
        if circularity is None:
            row["total_recycling_pct"] = pd.NA
            row["mechanical_fraction_of_recycling"] = pd.NA
        else:
            total = circularity[1] + circularity[2]
            row["total_recycling_pct"] = total
            row["mechanical_fraction_of_recycling"] = (
                pd.NA if total == 0 else circularity[1] / total
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(
        ["target_year", "optimization_block", "candidate_identity"],
        kind="stable",
        ignore_index=True,
    )


def _annual_interventions_table(
    evaluations: Sequence[SelectedCandidateEvaluation],
    circularity_parent: "OptimizationCircularityParent",
) -> pd.DataFrame:
    """Report common demand/lifetime fractions and regional circularity fractions.

    Demand and lifetime fractions cover 2019--2100. Regional circularity covers
    2020--2100: active controls already include the linear transition increment in
    2020, while inactive controls retain the parent trajectory. Unsuccessful timings
    identify the rejected path retained by the worker.
    """
    rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    for item in evaluations:
        decision = item.selected_result.demand_lifetime
        for year_i, year in enumerate(range(2019, 2101)):
            for parameter_id, values in (
                ("demand_reduction_fraction", decision.demand_fraction),
                ("lifetime_extension_fraction", decision.lifetime_fraction),
            ):
                rows.append(
                    {
                        "candidate_identity": item.candidate.identity,
                        "year": year,
                        "region": None,
                        "parameter_id": parameter_id,
                        "value": float(values[year_i]),
                    }
                )
        circularity = build_circularity_trajectories(
            parent=circularity_parent.table,
            parent_identity=circularity_parent.identity,
            decision=item.candidate.circularity,
        )
        annual = circularity.annual_values.melt(
            id_vars=["year", "region"], var_name="parameter_id", value_name="value"
        )
        annual.insert(0, "candidate_identity", item.candidate.identity)
        frames.append(annual)
    return pd.concat([pd.DataFrame.from_records(rows), *frames], ignore_index=True).sort_values(
        ["candidate_identity", "year", "region", "parameter_id"],
        kind="stable",
        na_position="first",
        ignore_index=True,
    )


def _states_table(
    states: Sequence[SelectedReportingState],
) -> pd.DataFrame:
    """Format selected-state compliance and ranking evidence for user output."""
    return pd.DataFrame.from_records(
        [
            {
                "selected_state_identity": state.identity,
                "state_identity": state.state_identity,
                "selection_candidate_identity": (state.selection_candidate_identity),
                "physical_candidate_identity": state.physical_candidate_identity,
                "timing_2100_candidate_identity": (state.timing_2100_candidate_identity),
                "timing_2100_physical_candidate_identity": (
                    state.timing_2100_physical_candidate_identity
                ),
                "optimization_block": state.optimization_block,
                "selection_background_scenario": (state.selection_background_scenario),
                "allocated_cc_case": state.allocated_cc_case,
                "selection_status": state.selection_status,
                "all_region_joint_compliant": state.all_region_joint_compliant,
                "intervention_distance": state.intervention_distance,
                "binding_slack": state.binding_slack,
                "compliant_region_count": state.compliant_region_count,
                "fully_compliant_impact_target_count": (state.fully_compliant_impact_target_count),
                "compliant_region_impact_group_count": (state.compliant_region_impact_group_count),
                "compliant_coordinate_count": state.compliant_coordinate_count,
                "region_count": state.region_count,
                "impact_target_count": state.impact_target_count,
                "region_impact_group_count": state.region_impact_group_count,
                "coordinate_count": state.coordinate_count,
                "worst_asr": state.worst_asr,
                "maximum_positive_exceedance": state.maximum_positive_exceedance,
                "balanced_mean_positive_exceedance": (state.balanced_mean_positive_exceedance),
                "fully_compliant_regions_json": canonical_json_bytes(
                    list(state.fully_compliant_regions)
                ).decode("utf-8"),
                "fully_compliant_impact_targets_json": canonical_json_bytes(
                    list(state.fully_compliant_impact_targets)
                ).decode("utf-8"),
                "binding_coordinates_json": canonical_json_bytes(
                    [list(item) for item in state.binding_coordinates]
                ).decode("utf-8"),
                "transgressing_coordinates_json": canonical_json_bytes(
                    [list(item) for item in state.transgressing_coordinates]
                ).decode("utf-8"),
                "selected_is_exact_parent": state.selected_is_exact_parent,
                "selected_has_non_grid_baseline_component": (
                    state.selected_has_non_grid_baseline_component
                ),
                "declared_neighborhood_verified": (state.declared_neighborhood_verified),
                "selection_stable": state.selection_stable,
                "demand_reduction_pct": state.demand_reduction_pct,
                "lifetime_extension_pct": state.lifetime_extension_pct,
                "collection_pct": state.collection_pct,
                "mechanical_recycling_pct": state.mechanical_recycling_pct,
                "chemical_recycling_pct": state.chemical_recycling_pct,
                "incineration_pct": state.incineration_pct,
                "landfill_pct": state.landfill_pct,
                "fossil_feedstock_pct": state.fossil_feedstock_pct,
                "biogenic_feedstock_pct": state.biogenic_feedstock_pct,
                "daccu_feedstock_pct": state.daccu_feedstock_pct,
                "selection_target_year": state.selection_target_year,
            }
            for state in states
        ]
    ).sort_values(
        ["state_identity", "physical_candidate_identity"],
        kind="stable",
        ignore_index=True,
    )


def build_acc_table(
    target_year_by_physical_candidate: Mapping[str, int],
    selected_acc_metrics: Mapping[str, tuple[CandidateACCMetrics, ...]],
) -> pd.DataFrame:
    """Build allocated carrying capacity and ASR rows for the selected backgrounds."""

    frames: list[pd.DataFrame] = []
    target_years = dict(target_year_by_physical_candidate)
    for physical in sorted(target_years):
        target_year = target_years[physical]
        recorded: set[tuple[str, str]] = set()
        for result in selected_acc_metrics[physical]:
            key = (result.background_scenario, result.allocated_cc_case)
            if key in recorded:
                continue
            recorded.add(key)
            frames.append(_acc_metric_table(physical, target_year, result))
    frame = pd.concat(frames, ignore_index=True)
    frames.clear()
    frame["compliant"] = frame["asr"].le(1.0)
    return frame.sort_values(
        [
            "physical_candidate_identity",
            "target_year",
            "background_scenario",
            "projection_kind",
            "allocated_cc_case",
            "geography",
            "period_kind",
            "year",
            "optimization_lcia_method",
            "impact_category_id",
        ],
        kind="stable",
        ignore_index=True,
    )


def _acc_metric_table(
    physical: str,
    target_year: int,
    result: CandidateACCMetrics,
) -> pd.DataFrame:
    """Build allocated carrying capacity and ASR rows for each candidate, background, and label.

    Regional annual rows use ``[region, year, LCIA category]`` coordinates;
    cumulative climate rows use one ``[region]`` value for 2019--2100. Global
    rows contain the corresponding ratios after summing all 12 regional numerators
    and denominators. LCIA results retain native category units, allocated
    carrying capacities are positive in the same units.
    """

    metrics = result.metrics
    rows: list[dict[str, object]] = []
    for region_index, region in enumerate(metrics.regions):
        for category_index, category in enumerate(metrics.annual_categories):
            for year_index, year in enumerate(metrics.years):
                controlled_for_selection = target_year == 2050 and year >= 2020
                rows.append(
                    {
                        "physical_candidate_identity": physical,
                        "target_year": target_year,
                        "background_scenario": result.background_scenario,
                        "projection_kind": "member",
                        "allocated_cc_case": result.allocated_cc_case,
                        "geography": region,
                        "period_kind": "annual",
                        "year": year,
                        "optimization_lcia_method": category.optimization_lcia_method,
                        "impact_category_id": category.impact_category_id,
                        "controlled_for_selection": controlled_for_selection,
                        "selection_compliant": (
                            bool(
                                metrics.annual_asr[region_index, year_index, category_index] <= 1.0
                            )
                            if controlled_for_selection
                            else None
                        ),
                        "score": float(
                            metrics.annual_scores[region_index, year_index, category_index]
                        ),
                        "allocated_cc": float(
                            metrics.annual_allocated_cc[region_index, year_index, category_index]
                        ),
                        "asr": float(metrics.annual_asr[region_index, year_index, category_index]),
                    }
                )
        rows.append(
            {
                "physical_candidate_identity": physical,
                "target_year": target_year,
                "background_scenario": result.background_scenario,
                "projection_kind": "member",
                "allocated_cc_case": result.allocated_cc_case,
                "geography": region,
                "period_kind": "cumulative_2019_2100",
                "year": 2100,
                "optimization_lcia_method": "ef31",
                "impact_category_id": "climate_change",
                "controlled_for_selection": target_year == 2050,
                "selection_compliant": (
                    bool(metrics.cumulative_gwp100_asr[region_index] <= 1.0)
                    if target_year == 2050
                    else None
                ),
                "score": float(metrics.cumulative_gwp100_scores[region_index]),
                "allocated_cc": float(metrics.cumulative_gwp100_allocated_cc[region_index]),
                "asr": float(metrics.cumulative_gwp100_asr[region_index]),
            }
        )
    global_metrics = metrics.global_metrics
    for category_index, category in enumerate(metrics.annual_categories):
        for year_index, year in enumerate(metrics.years):
            rows.append(
                {
                    "physical_candidate_identity": physical,
                    "target_year": target_year,
                    "background_scenario": result.background_scenario,
                    "projection_kind": "member",
                    "allocated_cc_case": result.allocated_cc_case,
                    "geography": "global",
                    "period_kind": "annual",
                    "year": year,
                    "optimization_lcia_method": category.optimization_lcia_method,
                    "impact_category_id": category.impact_category_id,
                    "controlled_for_selection": False,
                    "selection_compliant": None,
                    "score": float(global_metrics.annual_scores[year_index, category_index]),
                    "allocated_cc": float(
                        global_metrics.annual_allocated_cc[year_index, category_index]
                    ),
                    "asr": float(global_metrics.annual_asr[year_index, category_index]),
                }
            )
    rows.append(
        {
            "physical_candidate_identity": physical,
            "target_year": target_year,
            "background_scenario": result.background_scenario,
            "projection_kind": "member",
            "allocated_cc_case": result.allocated_cc_case,
            "geography": "global",
            "period_kind": "cumulative_2019_2100",
            "year": 2100,
            "optimization_lcia_method": "ef31",
            "impact_category_id": "climate_change",
            "controlled_for_selection": False,
            "selection_compliant": None,
            "score": float(global_metrics.cumulative_gwp100_score),
            "allocated_cc": float(global_metrics.cumulative_gwp100_allocated_cc),
            "asr": float(global_metrics.cumulative_gwp100_asr),
        }
    )
    return pd.DataFrame.from_records(rows)


def _environmental_table(
    characterizations: Sequence[SelectedCandidateCharacterization],
    physical_index: Mapping[str, SelectedCandidateEvaluation],
) -> pd.DataFrame:
    """Format annual regional and global environmental results for selected states."""
    frames = [
        _environmental_candidate_table(
            characterization,
            physical_index[
                characterization.scores.physical_candidate_identity
            ].candidate.target_year,
        )
        for characterization in characterizations
    ]
    frame = pd.concat(frames, ignore_index=True)
    frames.clear()
    return frame.sort_values(
        [
            "physical_candidate_identity",
            "target_year",
            "background_scenario",
            "region",
            "year",
            "method_id",
            "impact_category_id",
        ],
        kind="stable",
        ignore_index=True,
    )


def _environmental_candidate_table(
    characterization: SelectedCandidateCharacterization,
    target_year: int,
) -> pd.DataFrame:
    """Build annual regional and global LCIA rows for one selected candidate.

    Regional rows span ``[background, world region, year, LCIA category]`` and
    contain LCIA results in native method units. ``contribution_norm`` is the sum
    of the absolute values of the characterized contributions, in the LCIA
    category's unit. Global rows stably sum the 12 regional values at the same
    background, year, and method coordinates. ``target_year`` records when the
    selected intervention percentages are complete.
    """

    scores = characterization.scores
    methods = tuple(scores.method_registry.itertuples(index=False))
    rows: list[dict[str, object]] = []
    for background_index, background in enumerate(scores.backgrounds):
        for region_index, region in enumerate(scores.regions):
            for year_index, year in enumerate(scores.years):
                for method_index, method in enumerate(methods):
                    rows.append(
                        {
                            "physical_candidate_identity": scores.physical_candidate_identity,
                            "target_year": target_year,
                            "background_scenario": background,
                            "region": region,
                            "year": year,
                            "method_id": str(method.method_id),
                            "method_version": str(method.method_version),
                            "impact_category_id": str(method.impact_category_id),
                            "impact_category_tuple_json": str(method.impact_category_tuple_json),
                            "unit": str(method.unit),
                            "factor_identity": str(method.factor_identity),
                            "score": float(
                                scores.annual_scores[
                                    background_index, region_index, year_index, method_index
                                ]
                            ),
                            "contribution_norm": float(
                                scores.annual_contribution_norms[
                                    background_index, region_index, year_index, method_index
                                ]
                            ),
                        }
                    )
    regional = pd.DataFrame.from_records(rows)
    values = ["score", "contribution_norm"]
    keys = [column for column in regional.columns if column not in (*values, "region")]

    def stable_sum(values: pd.Series) -> float:
        """Return a sum computed with ``math.fsum``."""
        return fsum(float(value) for value in values)

    global_rows = (
        regional.sort_values([*keys, "region"], kind="stable")
        .groupby(keys, observed=True, sort=True, dropna=False)[values]
        .agg(stable_sum)
        .reset_index()
    )
    global_rows["region"] = "global"
    return pd.concat((regional, global_rows), ignore_index=True)


def build_feedstock_requirements_table(
    evaluations: Sequence[tuple[int, CandidateWorkerResult]],
) -> pd.DataFrame:
    """Attach selected candidate identities to their feedstock requirements."""

    frames: list[pd.DataFrame] = []
    for target_year, result in evaluations:
        feedstock = cast(pd.DataFrame, result.feedstock_requirements).copy()
        feedstock.insert(0, "target_year", target_year)
        feedstock.insert(
            0,
            "physical_candidate_identity",
            result.physical.physical_candidate_identity,
        )
        frames.append(feedstock)
    return pd.concat(frames, ignore_index=True).sort_values(
        [
            "physical_candidate_identity",
            "target_year",
            "year",
            "region",
            "material",
            "constituent",
            "product",
            "quantity",
            "virgin_feedstock_route",
        ],
        kind="stable",
        ignore_index=True,
    )
