"""Bind candidate baseline data and identify interventions."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, TypeAlias

import numpy as np
import pandas as pd

from pytoce.run_lca.calculation.foreground import (
    OPTIMIZATION_ACTIVATION_METHOD_IDENTITY,
    OptimizationActivationContext,
)
from pytoce.run_lca.calculation.pa_receiving import (
    PaDisposalEvent,
    PaRecyclingEvent,
    PaSpinningEvent,
)
from pytoce.run_lca.calculation.providers import ProviderRoutes
from pytoce.run_mfa.model.mfa import (
    MFACandidateEvaluation,
    MFAEvaluation,
)
from pytoce.run_mfa.model.optimization_parents import (
    OptimizationCircularityParent,
    OptimizationDemandLifetimeParent,
    optimization_circularity_parent,
    optimization_demand_lifetime_parent,
)
from pytoce.run_mfa.scenario.parent_relative import (
    OPTIMIZATION_PARENT_OVERLAY_METHOD_IDENTITY,
)
from pytoce.run_mfa.scenario.scenario_contract import (
    OptimizationPhysicalInfeasibility,
    ScenarioEvaluation,
    ScenarioFeasibilityError,
)
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    DEMAND_REPAIR_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.interventions import (
    CIRCULARITY_METHOD_IDENTITY,
    DEMAND_LIFETIME_METHOD_IDENTITY,
    INTERVENTION_ASSIGNMENT_COLUMNS,
    CircularityEvaluationProjection,
    CircularityTrajectories,
    DemandLifetimeEvaluationProjection,
    DemandLifetimeTrajectories,
    InterventionDistanceInputs,
    build_circularity_evaluation_projection,
    build_demand_lifetime_evaluation_projection,
    build_intervention_distance_inputs,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256, payload_sha256

CANDIDATE_EVALUATION_METHOD_IDENTITY = "pytoce_optimization_candidate_evaluation_v56"
_PARENT_IDENTITY_PREFIX = "pytoce_optimization_parent_environment_v5::"
_INTERVENTION_PARENT_IDENTITY_PREFIX = "pytoce_optimization_intervention_parent_v1::"
_INTERVENTION_IDENTITY_PREFIX = "pytoce_optimization_intervention_request_v4::"
_PHYSICAL_IDENTITY_PREFIX = "pytoce_optimization_physical_candidate_v5::"
_ACTIVATION_IDENTITY_PREFIX = "pytoce_optimization_activation_delta_v2::"
_METHOD_SIGNATURE_COLUMNS = (
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "unit",
    "factor_identity",
)

PhysicalStatus: TypeAlias = Literal[
    "physical_feasible",
    "physical_infeasible",
]


class CandidateEvaluationError(ValueError):
    """A candidate input or calculation is invalid."""


class CandidateMFASession(Protocol):
    """MFA calculation boundary used while evaluating optimization candidates."""

    @property
    def authority_identity(self) -> str:
        """Return the identity of the MFA session used for candidate calculations."""
        ...

    def evaluate(
        self,
        *,
        foreground_scenario: str,
        compiled: pd.DataFrame,
        progress: Callable[[str, int | None], None] | None = None,
    ) -> MFACandidateEvaluation:
        """Evaluate one compiled foreground scenario for an optimization candidate."""
        ...


@dataclass(frozen=True, slots=True)
class CandidateParentMember:
    """Identify baseline LCIA scores for one REMIND background scenario."""

    background_scenario: str
    parent_score_identity: str
    projected_annual_scores_identity: str


@dataclass(frozen=True, slots=True)
class CandidateParentAuthority:
    """Baseline MFA inputs and provider activations for candidate calculations.

    Attributes:
        identity: Content identifier calculated from all baseline MFA and activation inputs.
        foreground_scenario_identity: Content identifier of the compiled baseline scenario.
        mfa_result_identity: Content identifier of the completed baseline MFA result.
        physical_state_identity: Identity of the admitted method-bound MFA state.
        provider_route_identity: Content identifier of the LCA provider-route table.
        activations: Baseline provider activations by region, year, stage, and service.
        pa_spinning_events: Complete baseline PA manufacturing mixtures.
        pa_recycling_events: Baseline PA washing, recovery and airborne exits.
        pa_disposal_events: Baseline wet fiber and dry reject receiving mixtures.
        demand_lifetime_parent: Baseline demand, stock, and lifetime tables.
        circularity_parent: Baseline collection, treatment, and feedstock shares.
    """

    identity: str
    foreground_scenario_identity: str
    mfa_result_identity: str
    physical_state_identity: str
    provider_route_identity: str
    activations: pd.DataFrame
    pa_spinning_events: tuple[PaSpinningEvent, ...]
    pa_recycling_events: tuple[PaRecyclingEvent, ...]
    pa_disposal_events: tuple[PaDisposalEvent, ...]
    demand_lifetime_parent: OptimizationDemandLifetimeParent
    circularity_parent: OptimizationCircularityParent


@dataclass(frozen=True, slots=True)
class CandidateInterventionParent:
    """Store baseline tables used to construct candidate trajectories."""

    identity: str
    physical_parent_identity: str
    foreground_scenario: str
    foreground_scenario_identity: str
    demand_lifetime_parent: OptimizationDemandLifetimeParent
    circularity_parent: OptimizationCircularityParent


@dataclass(frozen=True, slots=True)
class CandidateInterventionAuthority:
    """Store validated candidate trajectories and their input identities.

    Attributes:
        identity: Content identifier of the complete intervention request.
        physical_parent_identity: Identifier binding baseline MFA inputs and provider
            activations.
        parent_foreground_scenario_identity: Content identifier of the baseline scenario.
        demand_lifetime_parent_identity: Content identifier of the baseline demand and lifetime
            tables.
        circularity_parent_identity: Content identifier of the baseline circularity tables.
        demand_lifetime_trajectory_identity: Content identifier of the changed demand and
            lifetime values.
        circularity_trajectory_identity: Content identifier of the changed circularity values.
        combined_assignments_identity: Content identifier of all annual parameter assignments.
        demand_lifetime_evaluation_projection: Demand and lifetime values used in MFA.
        circularity_evaluation_projection: Circularity values used in MFA.
        distance_inputs: Baseline and candidate values used for intervention distance.
        uses_exact_parent: Whether both intervention groups request the exact parent.
        execution_foreground_scenario: Scenario name used for the candidate calculation.
        assignment_records: Immutable annual parameter rows passed to the scenario compiler.
    """

    identity: str
    physical_parent_identity: str
    parent_foreground_scenario_identity: str
    demand_lifetime_parent_identity: str
    circularity_parent_identity: str
    demand_lifetime_trajectory_identity: str
    circularity_trajectory_identity: str
    combined_assignments_identity: str
    demand_lifetime_evaluation_projection: DemandLifetimeEvaluationProjection
    circularity_evaluation_projection: CircularityEvaluationProjection
    distance_inputs: InterventionDistanceInputs
    uses_exact_parent: bool
    execution_foreground_scenario: str
    assignment_records: tuple[tuple[str | int | float | None, ...], ...]


@dataclass(frozen=True, slots=True)
class PhysicalEvaluation:
    """Store the MFA outcome of one candidate MFA calculation.

    Attributes:
        physical_request_identity: Identifier binding baseline, intervention, MFA session,
            method and provider-activation inputs.
        physical_candidate_identity: Identifier of this feasible outcome or recorded
            MFA rejection, bound to its request.
        physical_parent_identity: Identifier binding baseline MFA inputs and provider
            activations.
        intervention_authority_identity: Content identifier of the candidate trajectories.
        status: Whether the candidate satisfies the MFA feasibility checks.
        diagnostic: Complete direct-owner failure, or ``None`` when feasible.
        physical_state_identity: MFA state identity, or ``None`` when infeasible.
        activation_identity: Candidate provider-activation identity, when feasible.
        activation_delta: Candidate minus baseline provider activations, when feasible.
        pa_spinning_events: Whole candidate PA manufacturing quantities; empty if infeasible.
        pa_recycling_events: Actual PA wash/recovery/dust inputs and exits; empty if infeasible.
        pa_disposal_events: Actual wet fiber/dry reject receiving mixtures; empty if infeasible.
    """

    physical_request_identity: str
    physical_candidate_identity: str
    physical_parent_identity: str
    intervention_authority_identity: str
    status: PhysicalStatus
    diagnostic: OptimizationPhysicalInfeasibility | None
    physical_state_identity: str | None
    activation_identity: str | None
    activation_delta: pd.DataFrame | None
    pa_spinning_events: tuple[PaSpinningEvent, ...]
    pa_recycling_events: tuple[PaRecyclingEvent, ...]
    pa_disposal_events: tuple[PaDisposalEvent, ...]


def physical_evaluation_identity(physical: PhysicalEvaluation) -> str:
    """Bind a feasible MFA state or a recognized infeasibility to its request."""
    payload: dict[str, object] = {
        "physical_request_identity": physical.physical_request_identity,
        "physical_parent_identity": physical.physical_parent_identity,
        "intervention_authority_identity": physical.intervention_authority_identity,
        "status": physical.status,
    }
    if physical.status == "physical_feasible":
        payload.update(
            physical_state_identity=physical.physical_state_identity,
            activation_identity=physical.activation_identity,
        )
    else:
        payload["diagnostic"] = None if physical.diagnostic is None else asdict(physical.diagnostic)
    return _PHYSICAL_IDENTITY_PREFIX + payload_sha256(payload)


def _signatures(frame: pd.DataFrame) -> tuple[tuple[str, ...], ...]:
    """Extract immutable LCIA method signatures in table order."""
    return tuple(
        tuple(str(getattr(row, column)) for column in _METHOD_SIGNATURE_COLUMNS)
        for row in frame.itertuples(index=False)
    )


def _canonical_activations(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort the provider activation table by all columns."""
    return frame.sort_values(
        list(frame.columns),
        kind="stable",
        na_position="first",
        ignore_index=True,
    )


def _execution_foreground_scenario(
    *,
    parent_foreground_scenario: str,
    intervention_identity: str,
    uses_exact_parent: bool,
) -> str:
    """Return deterministic execution metadata outside scientific identity."""

    value = (
        parent_foreground_scenario
        if uses_exact_parent
        else f"optimization_candidate_{intervention_identity.rsplit('::', 1)[-1]}"
    )
    return value


def _candidate_intervention_identity(
    *,
    physical_parent_identity: str,
    parent_foreground_scenario_identity: str,
    demand_lifetime_parent_identity: str,
    circularity_parent_identity: str,
    demand_lifetime_trajectory_identity: str,
    circularity_trajectory_identity: str,
    combined_assignments_identity: str,
    uses_exact_parent: bool,
) -> str:
    """Return the scientific identity of one intervention."""

    payload = {
        "candidate_evaluation_method_identity": CANDIDATE_EVALUATION_METHOD_IDENTITY,
        "demand_repair_method_identity": DEMAND_REPAIR_METHOD_IDENTITY,
        "demand_lifetime_method_identity": DEMAND_LIFETIME_METHOD_IDENTITY,
        "circularity_method_identity": CIRCULARITY_METHOD_IDENTITY,
        "optimization_parent_overlay_method_identity": (
            OPTIMIZATION_PARENT_OVERLAY_METHOD_IDENTITY
        ),
        "physical_parent_identity": physical_parent_identity,
        "parent_foreground_scenario_identity": parent_foreground_scenario_identity,
        "demand_lifetime_parent_identity": demand_lifetime_parent_identity,
        "circularity_parent_identity": circularity_parent_identity,
        "demand_lifetime_trajectory_identity": demand_lifetime_trajectory_identity,
        "circularity_trajectory_identity": circularity_trajectory_identity,
        "combined_assignments_identity": combined_assignments_identity,
        "uses_exact_parent": uses_exact_parent,
    }
    return _INTERVENTION_IDENTITY_PREFIX + payload_sha256(payload)


def _immutable_assignment_records(
    assignments: pd.DataFrame,
) -> tuple[tuple[str | int | float | None, ...], ...]:
    """Convert intervention assignments to immutable scalar records."""

    return tuple(
        tuple(
            None
            if pd.isna(value)
            else int(value)
            if column == "year"
            else float(value)
            if column == "resolved_value"
            else str(value)
            for column, value in zip(assignments.columns, row, strict=True)
        )
        for row in assignments.itertuples(index=False, name=None)
    )


def _canonical_assignment_table(assignments: pd.DataFrame) -> pd.DataFrame:
    """Normalize a newly constructed assignment table in place."""

    result = assignments
    result["parameter_id"] = result["parameter_id"].astype("string")
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(np.int64)
    for column in INTERVENTION_ASSIGNMENT_COLUMNS[2:-1]:
        result[column] = result[column].astype("string")
    result["resolved_value"] = pd.to_numeric(result["resolved_value"], errors="raise").astype(
        np.float64
    )
    return result


def _intervention_assignments(
    intervention: CandidateInterventionAuthority,
) -> pd.DataFrame:
    """Materialize the compiler assignments stored with an intervention."""

    return _canonical_assignment_table(
        pd.DataFrame.from_records(
            intervention.assignment_records,
            columns=INTERVENTION_ASSIGNMENT_COLUMNS,
        )
    )


def build_candidate_intervention_parent(
    parent: ScenarioEvaluation,
    *,
    physical_parent: CandidateParentAuthority,
) -> CandidateInterventionParent:
    """Bind the owned baseline tables to one candidate intervention parent."""

    compiled_table_identity = canonical_table_sha256(parent.compiled_parameters)
    normalized_change_points_identity = payload_sha256(parent.normalized_change_points)
    demand_parent = physical_parent.demand_lifetime_parent
    circularity_parent = physical_parent.circularity_parent
    payload = {
        "method_identity": "pytoce_optimization_intervention_parent_v1",
        "physical_parent_identity": physical_parent.identity,
        "foreground_scenario": parent.foreground_scenario,
        "foreground_scenario_identity": parent.foreground_scenario_identity,
        "compiled_table_identity": compiled_table_identity,
        "normalized_change_points_identity": normalized_change_points_identity,
        "demand_lifetime_parent_identity": demand_parent.identity,
        "circularity_parent_identity": circularity_parent.identity,
    }
    result = CandidateInterventionParent(
        identity=_INTERVENTION_PARENT_IDENTITY_PREFIX + payload_sha256(payload),
        physical_parent_identity=physical_parent.identity,
        foreground_scenario=parent.foreground_scenario,
        foreground_scenario_identity=parent.foreground_scenario_identity,
        demand_lifetime_parent=demand_parent,
        circularity_parent=circularity_parent,
    )
    return result


def build_candidate_intervention_from_parent(
    parent: CandidateInterventionParent,
    *,
    demand_lifetime: DemandLifetimeTrajectories,
    circularity: CircularityTrajectories,
) -> CandidateInterventionAuthority:
    """Bind typed demand, lifetime, and circularity trajectories to their parent."""

    demand_parent = parent.demand_lifetime_parent
    circularity_parent = parent.circularity_parent
    demand_assignments = demand_lifetime.assignments
    circularity_assignments = circularity.assignments
    if demand_assignments.empty and circularity_assignments.empty:
        combined = demand_assignments
    else:
        combined = pd.concat(
            [frame for frame in (demand_assignments, circularity_assignments) if not frame.empty],
            ignore_index=True,
        )
    combined = combined.sort_values(
        list(combined.columns), kind="stable", na_position="first", ignore_index=True
    )
    combined = _canonical_assignment_table(combined)
    uses_exact_parent = demand_lifetime.uses_exact_parent and circularity.uses_exact_parent
    combined_assignments_identity = canonical_table_sha256(combined)
    distance_inputs = build_intervention_distance_inputs(
        physical_parent_identity=parent.physical_parent_identity,
        demand_lifetime_parent_identity=demand_parent.identity,
        demand_lifetime=demand_lifetime,
        circularity=circularity,
        parent_circularity=circularity_parent.table,
    )
    identity = _candidate_intervention_identity(
        physical_parent_identity=parent.physical_parent_identity,
        parent_foreground_scenario_identity=parent.foreground_scenario_identity,
        demand_lifetime_parent_identity=demand_parent.identity,
        circularity_parent_identity=circularity_parent.identity,
        demand_lifetime_trajectory_identity=demand_lifetime.identity,
        circularity_trajectory_identity=circularity.identity,
        combined_assignments_identity=combined_assignments_identity,
        uses_exact_parent=uses_exact_parent,
    )
    execution_foreground_scenario = _execution_foreground_scenario(
        parent_foreground_scenario=parent.foreground_scenario,
        intervention_identity=identity,
        uses_exact_parent=uses_exact_parent,
    )
    circularity_evaluation_projection = build_circularity_evaluation_projection(
        proposal_decision=circularity.decision,
        circularity_trajectory_identity=circularity.identity,
        intervention_authority_identity=identity,
        physical_parent_identity=parent.physical_parent_identity,
        demand_lifetime_parent_identity=demand_parent.identity,
        distance_inputs_identity=distance_inputs.identity,
    )
    demand_lifetime_evaluation_projection = build_demand_lifetime_evaluation_projection(
        demand_lifetime=demand_lifetime,
        physical_parent_identity=parent.physical_parent_identity,
        demand_lifetime_parent_identity=demand_parent.identity,
    )
    return CandidateInterventionAuthority(
        identity=identity,
        physical_parent_identity=parent.physical_parent_identity,
        parent_foreground_scenario_identity=parent.foreground_scenario_identity,
        demand_lifetime_parent_identity=demand_parent.identity,
        circularity_parent_identity=circularity_parent.identity,
        demand_lifetime_trajectory_identity=demand_lifetime.identity,
        circularity_trajectory_identity=circularity.identity,
        combined_assignments_identity=combined_assignments_identity,
        demand_lifetime_evaluation_projection=demand_lifetime_evaluation_projection,
        circularity_evaluation_projection=circularity_evaluation_projection,
        distance_inputs=distance_inputs,
        uses_exact_parent=uses_exact_parent,
        execution_foreground_scenario=execution_foreground_scenario,
        assignment_records=_immutable_assignment_records(combined),
    )


def build_candidate_parent_authority(
    parent_scenario: ScenarioEvaluation,
    *,
    mfa_result_identity: str,
    mfa_result_manifest_sha256: str,
    parent_mfa: MFAEvaluation,
    candidate_mfa_session_identity: str,
    provider_routes: ProviderRoutes,
    activation_context: OptimizationActivationContext,
) -> CandidateParentAuthority:
    """Build baseline optimization inputs and bind their provider activations."""

    identities = {
        "foreground_scenario_identity": parent_scenario.foreground_scenario_identity,
        "compiled_table_identity": canonical_table_sha256(parent_scenario.compiled_parameters),
        "normalized_change_points_identity": payload_sha256(
            parent_scenario.normalized_change_points
        ),
        "mfa_result_identity": mfa_result_identity,
        "mfa_result_manifest_sha256": mfa_result_manifest_sha256,
        "candidate_mfa_session_identity": candidate_mfa_session_identity,
        "physical_state_identity": parent_mfa.physical_state_identity,
        "provider_route_identity": provider_routes.identity,
        "provider_route_table_identity": canonical_table_sha256(provider_routes.routes),
        "material_carbon_identity": provider_routes.material_carbon.identity,
        "material_carbon_constituents": provider_routes.material_carbon.constituents,
        "material_carbon_fractions": provider_routes.material_carbon.fractions,
        "provider_interface_database": provider_routes.interface_database,
    }
    activation = _canonical_activations(activation_context.activations)
    activation_identity = canonical_table_sha256(activation)
    activation_context_identity = activation_context.identity
    try:
        demand_lifetime_parent = optimization_demand_lifetime_parent(
            parent_mfa.derived_outputs,
            physical_state_identity=parent_mfa.physical_state_identity,
            parent=parent_scenario,
        )
        circularity_parent = optimization_circularity_parent(
            parent_mfa.model_flows,
            parent_mfa.material_realization,
            physical_state_identity=parent_mfa.physical_state_identity,
            parent=parent_scenario,
        )
    except ScenarioFeasibilityError as exc:
        raise CandidateEvaluationError(
            "The MFA result cannot be represented by the optimization intervention inputs"
        ) from exc
    payload = {
        **identities,
        "activation_method_identity": OPTIMIZATION_ACTIVATION_METHOD_IDENTITY,
        "activation_identity": activation_identity,
        "activation_context_identity": activation_context_identity,
        "demand_lifetime_parent_identity": demand_lifetime_parent.identity,
        "circularity_parent_identity": circularity_parent.identity,
    }
    return CandidateParentAuthority(
        identity=_PARENT_IDENTITY_PREFIX + payload_sha256(payload),
        foreground_scenario_identity=parent_scenario.foreground_scenario_identity,
        mfa_result_identity=mfa_result_identity,
        physical_state_identity=parent_mfa.physical_state_identity,
        provider_route_identity=provider_routes.identity,
        activations=activation,
        pa_spinning_events=activation_context.pa_spinning_events,
        pa_recycling_events=activation_context.pa_recycling_events,
        pa_disposal_events=activation_context.pa_disposal_events,
        demand_lifetime_parent=demand_lifetime_parent,
        circularity_parent=circularity_parent,
    )
