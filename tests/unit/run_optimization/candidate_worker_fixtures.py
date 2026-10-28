"""Focused candidate worker contracts."""

import json
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd

from pytoce.prepare_backgrounds.preparation.premise import ACTIVE_BACKGROUND_SCENARIOS
from pytoce.run_lca.calculation.foreground import (
    H12_REGIONS,
    OPTIMIZATION_ACTIVATION_DELTA_COLUMNS,
)
from pytoce.run_lca.calculation.pa_receiving import (
    PaDisposalEvent,
    PaRecyclingEvent,
    PaSpinningEvent,
)
from pytoce.run_mfa.model.materials import load_material_composition
from pytoce.run_mfa.model.mfa import (
    MFA_PHYSICAL_STATE_SCHEMA,
)
from pytoce.run_mfa.model.optimization_parents import (
    OPTIMIZATION_CIRCULARITY_PARENT_METHOD_IDENTITY,
    OPTIMIZATION_DEMAND_LIFETIME_PARENT_METHOD_IDENTITY,
    OptimizationCircularityParent,
    OptimizationDemandLifetimeParent,
)
from pytoce.run_mfa.scenario.scenario_contract import OptimizationPhysicalInfeasibility
from pytoce.run_optimization.calculation.candidate_acc import (
    CandidateACCContext,
    CandidateACCMetrics,
    calculate_candidate_acc_metrics,
)
from pytoce.run_optimization.calculation.candidate_authority import (
    _PHYSICAL_IDENTITY_PREFIX,
    CandidateInterventionAuthority,
    CandidateInterventionParent,
    PhysicalEvaluation,
    _candidate_intervention_identity,
    _canonical_assignment_table,
    _execution_foreground_scenario,
    _immutable_assignment_records,
    physical_evaluation_identity,
)
from pytoce.run_optimization.calculation.candidate_evaluation import (
    _activation_identity,
    feedstock_requirements_identity,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    OptimizationScores,
    SearchScoringContext,
    _environmental_identity,
    build_response_parent_score_projection,
    identify_optimization_scores,
    prepare_search_scoring,
)
from pytoce.run_optimization.calculation.candidate_repair import (
    CandidateRepairFamilyContext,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerPlanner,
    candidate_worker_result_summary,
)
from pytoce.run_optimization.calculation.interventions import (
    INTERVENTION_ASSIGNMENT_COLUMNS,
    CircularityDecision,
    DemandLifetimeEvaluationProjection,
    InterventionDistanceInputs,
    build_circularity_evaluation_projection,
    demand_lifetime_evaluation_projection_identity,
    intervention_distance_inputs_identity,
)
from pytoce.run_optimization.calculation.pa_candidate_projection import PaCandidateScores
from pytoce.run_optimization.calculation.regional_metrics import (
    AnnualCategory,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    CANDIDATE_WORKER_INITIALIZER_SCHEMA,
    CandidateWorkerMetricScope,
    CandidateWorkerResult,
    CandidateWorkerTask,
    DemandRepairOutcome,
    candidate_worker_result_identity,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256, payload_sha256
from tests.unit.run_optimization.candidate_characterization_fixtures import _empty_pa_scoring
from tests.unit.run_optimization.intervention_fixtures import _shaped_decision
from tests.unit.run_optimization.response_projection_fixtures import (
    _response_parent,
    _response_projection,
)

ACC_CONTEXT_IDENTITY = "acc-context::fixture"


def _intervention(*, target_year: int = 2050) -> CandidateInterventionAuthority:
    """Build one intervention whose stored identities match its values."""

    demand_decision = _shaped_decision(
        demand_reduction_pct=25,
        lifetime_extension_pct=50,
        target_year=target_year,
    )
    circularity_decision = CircularityDecision(
        collection_pct=10,
        mechanical_recycling_pct=40,
        chemical_recycling_pct=20,
        incineration_pct=30,
        landfill_pct=10,
        fossil_pct=70,
        biogenic_pct=20,
        daccu_pct=10,
        target_year=target_year,
    )
    demand_trajectory = f"demand-trajectory::fixture::{target_year}"
    circularity_trajectory = f"circularity-trajectory::fixture::{target_year}"
    distance = InterventionDistanceInputs(
        identity="",
        physical_parent_identity="parent::fixture",
        demand_lifetime_parent_identity="demand-parent::fixture",
        demand_lifetime_trajectory_identity=demand_trajectory,
        circularity_trajectory_identity=circularity_trajectory,
        target_year=target_year,
        demand_reduction_fraction=0.25,
        lifetime_extension_fraction=0.5,
        collection_distance=0.1,
        treatment_share_distance=0.2,
        feedstock_share_distance=0.3,
    )
    distance = replace(distance, identity=intervention_distance_inputs_identity(distance))
    assignments = _canonical_assignment_table(
        pd.DataFrame.from_records(
            [
                (
                    "demand_multiplier",
                    2030,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0.75,
                )
            ],
            columns=INTERVENTION_ASSIGNMENT_COLUMNS,
        )
    )
    assignments_identity = canonical_table_sha256(assignments)
    intervention_identity = _candidate_intervention_identity(
        physical_parent_identity="parent::fixture",
        parent_foreground_scenario_identity="scenario::fixture",
        demand_lifetime_parent_identity="demand-parent::fixture",
        circularity_parent_identity="circularity-parent::fixture",
        demand_lifetime_trajectory_identity=demand_trajectory,
        circularity_trajectory_identity=circularity_trajectory,
        combined_assignments_identity=assignments_identity,
        uses_exact_parent=False,
    )
    demand_projection = DemandLifetimeEvaluationProjection(
        identity="",
        decision=demand_decision,
        demand_lifetime_trajectory_identity=demand_trajectory,
        physical_parent_identity="parent::fixture",
        demand_lifetime_parent_identity="demand-parent::fixture",
    )
    demand_projection = replace(
        demand_projection,
        identity=demand_lifetime_evaluation_projection_identity(demand_projection),
    )
    return CandidateInterventionAuthority(
        identity=intervention_identity,
        physical_parent_identity="parent::fixture",
        parent_foreground_scenario_identity="scenario::fixture",
        demand_lifetime_parent_identity="demand-parent::fixture",
        circularity_parent_identity="circularity-parent::fixture",
        demand_lifetime_trajectory_identity=demand_trajectory,
        circularity_trajectory_identity=circularity_trajectory,
        combined_assignments_identity=assignments_identity,
        demand_lifetime_evaluation_projection=demand_projection,
        circularity_evaluation_projection=build_circularity_evaluation_projection(
            proposal_decision=circularity_decision,
            circularity_trajectory_identity=circularity_trajectory,
            intervention_authority_identity=intervention_identity,
            physical_parent_identity="parent::fixture",
            demand_lifetime_parent_identity="demand-parent::fixture",
            distance_inputs_identity=distance.identity,
        ),
        distance_inputs=distance,
        uses_exact_parent=False,
        execution_foreground_scenario=_execution_foreground_scenario(
            parent_foreground_scenario="remind_mfa_default",
            intervention_identity=intervention_identity,
            uses_exact_parent=False,
        ),
        assignment_records=_immutable_assignment_records(assignments),
    )


def _task() -> CandidateWorkerTask:
    return CandidateWorkerTask(
        intervention=_intervention(),
        metric_scopes=(
            CandidateWorkerMetricScope(
                optimization_block="integrated_frontier",
                metric_registry_identity="metrics::fixture",
            ),
        ),
    )


def _initializer(**changes: object) -> bytes:
    payload: dict[str, object] = {
        "schema_version": CANDIDATE_WORKER_INITIALIZER_SCHEMA,
        "workspace": "C:/workspace",
        "foreground_scenario": "remind_mfa_default",
        "compiled_scenario_identity": "scenario::fixture",
        "compiled_scenario_manifest_sha256": "a" * 64,
        "mfa_result_identity": "mfa::fixture",
        "mfa_result_manifest_sha256": "b" * 64,
        "candidate_mfa_session_identity": "candidate-mfa::fixture",
        "response_method_union": "ef31_pb_lcia",
        "response_bank_identity": "response::fixture",
        "response_bank_manifest_sha256": "e" * 64,
        "allocated_cc_result": "pyaesa_h12",
        "allocated_cc_result_identity": "acc::fixture",
        "allocated_cc_case": ["central"],
    }
    payload.update(changes)
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _worker_contexts(
    *, unit_score: float = 2.0
) -> tuple[CandidateACCContext, SearchScoringContext]:
    """Prepare real score parents/projections and a complete positive one-case aCC fixture."""
    parent = replace(_response_parent(), identity="parent::fixture")
    projection = _response_projection("optimization", unit_score=unit_score)
    climate = _response_projection("mandatory_climate", unit_score=unit_score)
    pa_projection, pa_parent = _empty_pa_scoring(projection.method_registry)
    scoring = prepare_search_scoring(
        physical_parent=parent,
        optimization_parent=build_response_parent_score_projection(
            parent,
            projection,
            pa_parent=pa_parent,
            pa_method_registry=pa_projection.method_registry,
        ),
        climate_parent=build_response_parent_score_projection(
            parent, climate, pa_parent=pa_parent, pa_method_registry=pa_projection.method_registry
        ),
        optimization_projection=projection,
        climate_projection=climate,
        pa_projection=pa_projection,
        pa_parent=pa_parent,
    )
    years = tuple(range(2019, 2101))
    context = CandidateACCContext(
        identity=ACC_CONTEXT_IDENTITY,
        allocated_cc_selection_identity="acc-selection::fixture",
        method_compatibility_identity=scoring.optimization_projection.method_compatibility_identity,
        cumulative_climate_compatibility_identity=scoring.climate_projection.method_compatibility_identity,
        allocated_cc_cases=("p50",),
        regions=H12_REGIONS,
        years=years,
        annual_categories=(AnnualCategory("ef31", "climate_change"),),
        annual_allocated_cc=np.ones((1, len(H12_REGIONS), len(years), 1)),
        cumulative_allocated_cc=np.full((1, len(H12_REGIONS)), float(len(years))),
        annual_case_table_identities=("annual-table::fixture",),
        cumulative_case_table_identities=("cumulative-table::fixture",),
    )
    return context, scoring


def _summary(result: CandidateWorkerResult):
    """Derive compact fixture metrics through the actual request-summary owner."""
    context, scoring = _worker_contexts()
    return candidate_worker_result_summary(
        result, allocated_cc_context=context, search_scoring=scoring
    )


@lru_cache(maxsize=1)
def _feedstock_requirements() -> tuple[pd.DataFrame, str]:
    """Share a complete zero-containing report; corruption tests copy before changing it."""
    coordinates = [
        (item.material, item.constituent, product_name, quantity)
        for item in load_material_composition().constituents
        for product_name, quantity in (
            (
                ("ethylene", "virgin_monomer_consumption"),
                ("propylene", "virgin_monomer_consumption"),
            )
            if item.chemical_polymer is not None
            else (("polymer", "new_polymer_output"),)
        )
    ]
    keys = [
        "year",
        "region",
        "material",
        "constituent",
        "product",
        "quantity",
        "virgin_feedstock_route",
    ]
    frame = pd.DataFrame.from_records(
        [
            (year, region, *coordinate, route)
            for year, region, coordinate, route in product(
                range(2019, 2101),
                H12_REGIONS,
                coordinates,
                ("fossil", "biogenic", "atmospheric_dac"),
            )
        ],
        columns=keys,
    )
    frame["amount"] = 0.0
    frame["unit"] = "Mt"
    frame = frame.astype({key: "int64" if key == "year" else "string" for key in (*keys, "unit")})
    frame.loc[
        frame["year"].eq(2030)
        & frame["region"].eq("EUR")
        & frame["material"].eq("HDPE")
        & frame["product"].eq("ethylene")
        & frame["virgin_feedstock_route"].eq("fossil"),
        "amount",
    ] = 1.0
    frame = frame.sort_values(keys, kind="stable", ignore_index=True)
    return frame, feedstock_requirements_identity(frame)


def _pa_scores(*, active: bool) -> PaCandidateScores:
    """Represent the full-union serialization contract.

    These deterministic LCIA quantities isolate binary/Arrow preservation; the
    independent two-anchor PA projection test owns native scientific equivalence.
    The active record uses two methods and two years, while the non-PA case
    retains the identical typed stage schema and valid zero annual arrays.
    """
    backgrounds = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)
    scores = np.zeros((len(backgrounds), len(H12_REGIONS), 82, 2), dtype=np.float64)
    norms = np.zeros_like(scores)
    rows = []
    if active:
        for background_index, background in enumerate(backgrounds):
            for year, method_index, score, norm in (
                (2025, 0, -0.75, 2.25),
                (2025, 1, 0.125, 1.125),
                (2050, 0, 2.5, 3.5),
                (2050, 1, -0.25, 1.25),
            ):
                coordinate = (background_index, H12_REGIONS.index("EUR"), year - 2019, method_index)
                scores[coordinate], norms[coordinate] = score, norm
                rows.append(
                    {
                        "characterization_background_scenario": background,
                        "region": "EUR",
                        "year": year,
                        "method_id": f"test_{method_index}",
                        "method_version": "fixture",
                        "impact_category_id": "category",
                        "impact_category_tuple_json": json.dumps(["test", str(method_index)]),
                        "unit": "test impact unit",
                        "factor_identity": str(method_index) * 64,
                        "life_cycle_stage": "incineration_treatment",
                        "stage_group": "managed_terminal_end_of_life",
                        "cutoff_cycle_role": "waste_producer_terminal_treatment",
                        "material_cycle_source": "not_applicable",
                        "physical_event_basis": "year collected end-of-life waste is incinerated",
                        "physical_event_year": year,
                        "score": score,
                        "contribution_norm": norm,
                    }
                )
    columns = (
        "characterization_background_scenario",
        "region",
        "year",
        "method_id",
        "method_version",
        "impact_category_id",
        "impact_category_tuple_json",
        "unit",
        "factor_identity",
        "life_cycle_stage",
        "stage_group",
        "cutoff_cycle_role",
        "material_cycle_source",
        "physical_event_basis",
        "physical_event_year",
        "score",
        "contribution_norm",
    )
    stage = pd.DataFrame.from_records(rows, columns=columns).astype(
        {
            column: "int64"
            if column in ("year", "physical_event_year")
            else "float64"
            if column in ("score", "contribution_norm")
            else "string"
            for column in columns
        }
    )
    scores.setflags(write=False)
    norms.setflags(write=False)
    return PaCandidateScores(scores, norms, stage)


def _result_and_acc_metrics(
    *,
    activation_delta: pd.DataFrame | None = None,
    intervention: CandidateInterventionAuthority | None = None,
    pa_spinning_events: tuple[PaSpinningEvent, ...] = (),
    pa_recycling_events: tuple[PaRecyclingEvent, ...] = (),
    pa_disposal_events: tuple[PaDisposalEvent, ...] = (),
    pa_scores: PaCandidateScores | None = None,
) -> tuple[CandidateWorkerResult, CandidateACCMetrics]:
    """Build one internally consistent worker result and its aCC metrics."""

    intervention = _intervention() if intervention is None else intervention
    physical_state_identity = MFA_PHYSICAL_STATE_SCHEMA + "::" + "c" * 64
    activation = (
        pd.DataFrame.from_records(
            [
                (
                    2030,
                    "EUR",
                    "production",
                    "service::one",
                    "service_one",
                    "material_production",
                    "primary_material_cycle",
                    "primary_resin",
                    "year material enters system",
                    2030,
                    1.0,
                    2.5,
                    1.5,
                ),
                (
                    2050,
                    "USA",
                    "collection",
                    "service::two",
                    "service_two",
                    "collection",
                    "post_use_collection",
                    "post_use_plastic",
                    "year material is collected",
                    2050,
                    3.0,
                    1.0,
                    -2.0,
                ),
            ],
            columns=OPTIMIZATION_ACTIVATION_DELTA_COLUMNS,
        )
        if activation_delta is None
        else activation_delta.loc[:, list(OPTIMIZATION_ACTIVATION_DELTA_COLUMNS)].copy(deep=True)
    )
    activation_identity = _activation_identity(
        activation, pa_spinning_events, pa_recycling_events, pa_disposal_events
    )
    physical_request_identity = "physical-request::fixture"
    physical_identity = _PHYSICAL_IDENTITY_PREFIX + payload_sha256(
        {
            "physical_request_identity": physical_request_identity,
            "physical_parent_identity": "parent::fixture",
            "intervention_authority_identity": intervention.identity,
            "status": "physical_feasible",
            "physical_state_identity": physical_state_identity,
            "activation_identity": activation_identity,
        }
    )
    physical = PhysicalEvaluation(
        physical_request_identity=physical_request_identity,
        physical_candidate_identity=physical_identity,
        physical_parent_identity="parent::fixture",
        intervention_authority_identity=intervention.identity,
        status="physical_feasible",
        diagnostic=None,
        physical_state_identity=physical_state_identity,
        activation_identity=activation_identity,
        activation_delta=activation,
        pa_spinning_events=pa_spinning_events,
        pa_recycling_events=pa_recycling_events,
        pa_disposal_events=pa_disposal_events,
    )
    backgrounds = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)
    years = tuple(range(2019, 2101))
    base = np.full((len(backgrounds), len(H12_REGIONS), len(years), 1), 0.5)
    climate = base[..., 0].copy()
    context, scoring = _worker_contexts()
    scores = identify_optimization_scores(
        OptimizationScores(
            physical_candidate_identity=physical.physical_candidate_identity,
            environmental_identity=_environmental_identity(physical, scoring)[0],
            member_identities=(),
            backgrounds=backgrounds,
            regions=H12_REGIONS,
            years=years,
            annual_categories=(AnnualCategory("ef31", "climate_change"),),
            optimization_method_registry_identity=scoring.optimization_projection.method_registry_identity,
            cumulative_climate_method_registry_identity=scoring.climate_projection.method_registry_identity,
            annual_scores=base,
            annual_climate_scores=climate,
            annual_scores_identity="",
            annual_climate_scores_identity="",
            score_identity="",
        )
    )
    scope = CandidateWorkerMetricScope("integrated_frontier", "metric-registry::fixture")
    acc_metrics = calculate_candidate_acc_metrics(
        scores,
        context,
        environmental_parent_members=scoring.optimization_parent.member_parents,
        scoring_context_identity=scoring.identity,
        intervention_authority_identity=intervention.identity,
        distance_inputs=intervention.distance_inputs,
        is_zero_intervention_parent=intervention.uses_exact_parent,
        optimization_block=scope.optimization_block,
        metric_registry_identity=scope.metric_registry_identity,
    )
    compact, compact_identity = _feedstock_requirements()
    result = CandidateWorkerResult(
        identity="",
        worker_authority_identity="worker::fixture",
        worker_scientific_authority_identity="worker-science::fixture",
        requested_intervention_identity=intervention.identity,
        repair=DemandRepairOutcome(
            "linear_feasible",
            0.0,
            0.0,
            int(not intervention.uses_exact_parent),
            0,
            0,
            0,
            Fraction(0),
            None,
            None,
        ),
        intervention=intervention,
        physical=physical,
        scores=scores,
        pa_scores=_pa_scores(active=False) if pa_scores is None else pa_scores,
        metric_scopes=(scope,),
        feedstock_requirements=compact,
        feedstock_requirements_identity=compact_identity,
    )
    return replace(result, identity=candidate_worker_result_identity(result)), acc_metrics[0]


def _result() -> CandidateWorkerResult:
    """Return one internally consistent candidate worker result."""

    return _result_and_acc_metrics()[0]


def _infeasible_result(
    *,
    intervention: CandidateInterventionAuthority | None = None,
) -> CandidateWorkerResult:
    """Represent rejection by the fixed-lifetime stock LP after linear-path failure."""
    feasible = _result_and_acc_metrics(intervention=intervention)[0]
    diagnostic = OptimizationPhysicalInfeasibility(
        code="optimization_stock_below_surviving_cohort",
        equation="target_stock >= surviving_existing_stock",
        observed=0.75,
        bound=1.0,
        tolerance=0.0,
        shortfall=0.25,
        status="stock_target_below_surviving_cohort",
        count=1,
        coordinates=(("year", 2050), ("region", "EUR"), ("good", "Buildings")),
        details=(("unit", "Mt"),),
    )
    physical = replace(
        feasible.physical,
        physical_candidate_identity="",
        status="physical_infeasible",
        diagnostic=diagnostic,
        physical_state_identity=None,
        activation_identity=None,
        activation_delta=None,
    )
    physical = replace(physical, physical_candidate_identity=physical_evaluation_identity(physical))
    result = replace(
        feasible,
        physical=physical,
        repair=DemandRepairOutcome(
            "stock_infeasible_fixed_controls", 0.0, 0.0, 1, 1, 0, 0, None, diagnostic, None
        ),
        scores=None,
        pa_scores=None,
        feedstock_requirements=None,
        feedstock_requirements_identity=None,
    )
    return replace(result, identity=candidate_worker_result_identity(result))


def _planner() -> CandidateWorkerPlanner:
    """Return a planner that owns the candidate worker fixture."""

    result = _result()
    assert result.physical.physical_state_identity is not None
    demand_parent = OptimizationDemandLifetimeParent(
        identity=result.intervention.demand_lifetime_parent_identity,
        method_identity=OPTIMIZATION_DEMAND_LIFETIME_PARENT_METHOD_IDENTITY,
        parent_foreground_scenario_identity=result.intervention.parent_foreground_scenario_identity,
        physical_state_identity=result.physical.physical_state_identity,
        stock_target_identity="stock-target::fixture",
        lifetime_identity="lifetime::fixture",
        stock_targets=pd.DataFrame(),
        lifetimes=pd.DataFrame(),
    )
    circularity_parent = OptimizationCircularityParent(
        identity=result.intervention.circularity_parent_identity,
        method_identity=OPTIMIZATION_CIRCULARITY_PARENT_METHOD_IDENTITY,
        physical_state_identity=result.physical.physical_state_identity,
        foreground_scenario="fixture",
        table=pd.DataFrame(),
    )
    parent = CandidateInterventionParent(
        identity="intervention-parent::fixture",
        physical_parent_identity=result.physical.physical_parent_identity,
        foreground_scenario="fixture",
        foreground_scenario_identity=result.intervention.parent_foreground_scenario_identity,
        demand_lifetime_parent=demand_parent,
        circularity_parent=circularity_parent,
    )
    return CandidateWorkerPlanner(
        identity="planner::fixture",
        worker_authority_identity=result.worker_authority_identity,
        allocated_cc_cases=("p50",),
        intervention_parent=parent,
        repair_family_context=CandidateRepairFamilyContext({}, False, False),
    )
