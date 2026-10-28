"""Prepare allocated carrying capacity arrays and evaluate candidate sustainability metrics."""

from dataclasses import dataclass

from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.candidate_authority import (
    CandidateEvaluationError,
    CandidateParentMember,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    OptimizationScores,
    SearchScoringContext,
)
from pytoce.run_optimization.calculation.candidate_projection_arrays import (
    FloatArray,
    _array_identity,
    _readonly,
)
from pytoce.run_optimization.calculation.interventions import (
    INTERVENTION_DISTANCE_INPUT_METHOD_IDENTITY,
    InterventionDistanceInputs,
)
from pytoce.run_optimization.calculation.regional_metrics import (
    INTERVENTION_DISTANCE_METHOD_IDENTITY,
    REGIONAL_METRIC_METHOD_IDENTITY,
    AnnualCategory,
    RegionalMetrics,
    calculate_regional_metrics,
    intervention_distance,
)
from pytoce.run_optimization.input.allocated_cc_contract import AllocatedCCSelection
from pytoce.shared.runtime.filesystem import payload_sha256

_ACC_CONTEXT_IDENTITY_PREFIX = "pytoce_optimization_acc_context_v2::"
_ACC_ITEM_IDENTITY_PREFIX = "pytoce_optimization_acc_item_v4::"
_ACC_IDENTITY_PREFIX = "pytoce_optimization_acc_evaluation_v9::"
_COMPARISON_SCOPE_IDENTITY_PREFIX = "pytoce_optimization_comparison_scope_v3::"


@dataclass(frozen=True, slots=True)
class CandidateACCMetrics:
    """One explicit background/case-bound aCC metric result."""

    background_scenario: str
    allocated_cc_case: str
    allocated_cc_context_identity: str
    acc_evaluation_identity: str
    metrics: RegionalMetrics


@dataclass(frozen=True, slots=True)
class CandidateACCContext:
    """Store allocated carrying capacity arrays used by one optimization request.

    Attributes:
        identity: Content identifier calculated from the selected cases, methods, coordinates,
            and arrays.
        allocated_cc_selection_identity: Content identifier of the selected aCC rows.
        method_compatibility_identity: Identity matching annual aCC and EF score methods.
        cumulative_climate_compatibility_identity: Identity matching the cumulative
            climate aCC and score method.
        allocated_cc_cases: Selected case names in array order.
        regions: 12 world regions in array order.
        years: Calendar years 2019 through 2100 in array order.
        annual_categories: EF impact categories in array order.
        annual_allocated_cc: Positive allocated carrying capacities with axes
            ``[case, region, year, impact category]``, in native LCIA units for
            each assessment year.
        cumulative_allocated_cc: Positive allocated carrying capacities for
            climate change over 2019--2100, with axes ``[case, region]`` in kg CO2-eq.
        annual_case_table_identities: Table identity for each selected annual case.
        cumulative_case_table_identities: Table identity for each cumulative case.
    """

    identity: str
    allocated_cc_selection_identity: str
    method_compatibility_identity: str
    cumulative_climate_compatibility_identity: str
    allocated_cc_cases: tuple[str, ...]
    regions: tuple[str, ...]
    years: tuple[int, ...]
    annual_categories: tuple[AnnualCategory, ...]
    annual_allocated_cc: FloatArray
    cumulative_allocated_cc: FloatArray
    annual_case_table_identities: tuple[str, ...]
    cumulative_case_table_identities: tuple[str, ...]


def prepare_candidate_acc_context(selection: AllocatedCCSelection) -> CandidateACCContext:
    """Convert one aCC selection to read-only arrays."""

    region_index = [selection.regions.index(region) for region in H12_REGIONS]
    annual_allocated_cc = _readonly(selection.annual_allocated_cc[:, region_index, :, :])
    cumulative_allocated_cc = _readonly(selection.cumulative_allocated_cc[:, region_index])
    annual_allocated_cc_identity = _array_identity(annual_allocated_cc)
    cumulative_allocated_cc_identity = _array_identity(cumulative_allocated_cc)
    annual_categories = tuple(
        AnnualCategory(method, category) for method, category in selection.annual_categories
    )
    payload = {
        "allocated_cc_result_identity": selection.allocated_cc_result_identity,
        "allocated_cc_selection_identity": selection.allocated_cc_selection_identity,
        "optimization_lcia_selection_identity": (selection.optimization_lcia_selection_identity),
        "method_registry_identity": selection.method_registry_identity,
        "cumulative_climate_method_registry_identity": (
            selection.cumulative_climate_method_registry_identity
        ),
        "method_compatibility_identity": selection.method_compatibility_identity,
        "cumulative_climate_compatibility_identity": (
            selection.cumulative_climate_compatibility_identity
        ),
        "allocated_cc_cases": list(selection.allocated_cc_cases),
        "regions": list(H12_REGIONS),
        "years": list(selection.years),
        "annual_categories": [
            {
                "optimization_lcia_method": category.optimization_lcia_method,
                "impact_category_id": category.impact_category_id,
            }
            for category in annual_categories
        ],
        "annual_allocated_cc_identity": annual_allocated_cc_identity,
        "cumulative_allocated_cc_identity": cumulative_allocated_cc_identity,
        "annual_case_table_identities": list(selection.annual_case_table_identities),
        "cumulative_case_table_identities": list(selection.cumulative_case_table_identities),
    }
    return CandidateACCContext(
        identity=_ACC_CONTEXT_IDENTITY_PREFIX + payload_sha256(payload),
        allocated_cc_selection_identity=selection.allocated_cc_selection_identity,
        method_compatibility_identity=selection.method_compatibility_identity,
        cumulative_climate_compatibility_identity=(
            selection.cumulative_climate_compatibility_identity
        ),
        allocated_cc_cases=selection.allocated_cc_cases,
        regions=H12_REGIONS,
        years=selection.years,
        annual_categories=annual_categories,
        annual_allocated_cc=annual_allocated_cc,
        cumulative_allocated_cc=cumulative_allocated_cc,
        annual_case_table_identities=selection.annual_case_table_identities,
        cumulative_case_table_identities=selection.cumulative_case_table_identities,
    )


def _assert_acc_scoring_compatibility(
    context: CandidateACCContext,
    scoring: SearchScoringContext,
) -> None:
    """Match the aCC and score methods when a request session is created."""

    if (
        context.method_compatibility_identity
        != scoring.optimization_projection.method_compatibility_identity
        or context.cumulative_climate_compatibility_identity
        != scoring.climate_projection.method_compatibility_identity
    ):
        raise CandidateEvaluationError(
            "The allocated carrying capacity result and the selected LCA methods do not match"
        )


def _acc_item_identity(
    allocated_cc_context_identity: str,
    *,
    scoring_context_identity: str,
    optimization_block: OptimizationBlock,
    background_scenario: str,
    allocated_cc_case: str,
    metric_registry_identity: str,
) -> str:
    """Identify one candidate-independent metric item."""

    payload = {
        "intervention_distance_input_method_identity": (
            INTERVENTION_DISTANCE_INPUT_METHOD_IDENTITY
        ),
        "intervention_distance_method_identity": INTERVENTION_DISTANCE_METHOD_IDENTITY,
        "scoring_context_identity": scoring_context_identity,
        "allocated_cc_context_identity": allocated_cc_context_identity,
        "optimization_block": optimization_block,
        "background_scenario": background_scenario,
        "allocated_cc_case": allocated_cc_case,
        "metric_registry_identity": metric_registry_identity,
    }
    return _ACC_ITEM_IDENTITY_PREFIX + payload_sha256(payload)


def _acc_comparison_scope_identity(
    context: CandidateACCContext,
    scores: OptimizationScores,
    *,
    environmental_parent_member: CandidateParentMember,
    scoring_context_identity: str,
    distance_inputs: InterventionDistanceInputs,
    background_scenario: str,
    allocated_cc_case_index: int,
) -> str:
    """Collect the common inputs used to compare candidate metrics."""

    payload = {
        "method_identity": "pytoce_optimization_comparison_scope_v3",
        "physical_parent_identity": distance_inputs.physical_parent_identity,
        "demand_lifetime_parent_identity": distance_inputs.demand_lifetime_parent_identity,
        "scoring_context_identity": scoring_context_identity,
        "background_scenario": background_scenario,
        "environmental_parent_member": {
            "parent_score_identity": environmental_parent_member.parent_score_identity,
            "projected_annual_scores_identity": (
                environmental_parent_member.projected_annual_scores_identity
            ),
        },
        "allocated_cc_context_identity": context.identity,
        "allocated_cc_selection_identity": context.allocated_cc_selection_identity,
        "allocated_cc_case": context.allocated_cc_cases[allocated_cc_case_index],
        "annual_case_table_identity": context.annual_case_table_identities[allocated_cc_case_index],
        "cumulative_case_table_identity": context.cumulative_case_table_identities[
            allocated_cc_case_index
        ],
        "optimization_method_registry_identity": scores.optimization_method_registry_identity,
        "cumulative_climate_method_registry_identity": (
            scores.cumulative_climate_method_registry_identity
        ),
        "regions": list(scores.regions),
        "years": list(scores.years),
        "annual_categories": [
            [item.optimization_lcia_method, item.impact_category_id]
            for item in scores.annual_categories
        ],
        "intervention_distance_input_method_identity": (
            INTERVENTION_DISTANCE_INPUT_METHOD_IDENTITY
        ),
        "intervention_distance_method_identity": INTERVENTION_DISTANCE_METHOD_IDENTITY,
        "target_year": distance_inputs.target_year,
        "direct_family_method_identity": INTERVENTION_DISTANCE_INPUT_METHOD_IDENTITY,
    }
    return _COMPARISON_SCOPE_IDENTITY_PREFIX + payload_sha256(payload)


def calculate_candidate_acc_metrics(
    scores: OptimizationScores,
    context: CandidateACCContext,
    *,
    environmental_parent_members: tuple[CandidateParentMember, ...],
    scoring_context_identity: str,
    intervention_authority_identity: str,
    distance_inputs: InterventionDistanceInputs,
    is_zero_intervention_parent: bool,
    optimization_block: OptimizationBlock,
    metric_registry_identity: str,
) -> tuple[CandidateACCMetrics, ...]:
    """Calculate sustainability metrics and intervention distance for each selected
    background and allocated carrying capacity case."""

    distance = intervention_distance(
        physical_candidate_identity=scores.physical_candidate_identity,
        intervention_authority_identity=intervention_authority_identity,
        distance_inputs=distance_inputs,
        optimization_block=optimization_block,
    )
    outputs: list[CandidateACCMetrics] = []
    for background_i, background in enumerate(scores.backgrounds):
        member = next(
            item for item in environmental_parent_members if item.background_scenario == background
        )
        for case_i, case in enumerate(context.allocated_cc_cases):
            comparison_scope_identity = _acc_comparison_scope_identity(
                context,
                scores,
                environmental_parent_member=member,
                scoring_context_identity=scoring_context_identity,
                distance_inputs=distance_inputs,
                background_scenario=background,
                allocated_cc_case_index=case_i,
            )
            item_identity = _acc_item_identity(
                context.identity,
                scoring_context_identity=scoring_context_identity,
                optimization_block=optimization_block,
                background_scenario=background,
                allocated_cc_case=case,
                metric_registry_identity=metric_registry_identity,
            )
            item_payload = {
                "regional_metric_method_identity": REGIONAL_METRIC_METHOD_IDENTITY,
                "optimization_item_identity": item_identity,
                "physical_candidate_identity": scores.physical_candidate_identity,
                "intervention_authority_identity": intervention_authority_identity,
                "optimization_block": optimization_block,
                "intervention_distance_identity": distance.identity,
                "environmental_member_identity": scores.member_identities[background_i],
                "score_identity": scores.score_identity,
                "allocated_cc_context_identity": context.identity,
                "allocated_cc_case": case,
                "annual_case_table_identity": context.annual_case_table_identities[case_i],
                "cumulative_case_table_identity": (
                    context.cumulative_case_table_identities[case_i]
                ),
                "metric_registry_identity": metric_registry_identity,
            }
            evaluation_identity = _ACC_IDENTITY_PREFIX + payload_sha256(item_payload)
            metrics = calculate_regional_metrics(
                evaluation_identity=evaluation_identity,
                background_scenario=background,
                allocated_cc_case=case,
                allocated_cc_context_identity=context.identity,
                physical_candidate_identity=scores.physical_candidate_identity,
                intervention_authority_identity=intervention_authority_identity,
                optimization_item_identity=item_identity,
                optimization_block=optimization_block,
                is_zero_intervention_parent=is_zero_intervention_parent,
                intervention_distance=distance,
                optimization_lcia_method_registry_identity=(
                    scores.optimization_method_registry_identity
                ),
                cumulative_climate_method_registry_identity=(
                    scores.cumulative_climate_method_registry_identity
                ),
                comparison_scope_identity=comparison_scope_identity,
                annual_scores=scores.annual_scores[background_i],
                annual_allocated_cc=context.annual_allocated_cc[case_i],
                annual_budget_compatible_gwp100_scores=(scores.annual_climate_scores[background_i]),
                cumulative_gwp100_allocated_cc=context.cumulative_allocated_cc[case_i],
                regions=scores.regions,
                years=scores.years,
                annual_categories=scores.annual_categories,
            )
            outputs.append(
                CandidateACCMetrics(
                    background_scenario=background,
                    allocated_cc_case=case,
                    allocated_cc_context_identity=context.identity,
                    acc_evaluation_identity=evaluation_identity,
                    metrics=metrics,
                )
            )
    return tuple(outputs)
