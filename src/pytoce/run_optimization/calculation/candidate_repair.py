"""Identify feedstock requests that share one finite demand repair.

Only the repair disposition is shared. Every feasible feedstock member still
receives its own complete MFA and environmental calculation. Eligibility binds
the actual annual non-feedstock assignments and their execution mode, rather
than treating equal endpoint percentages as equal MFA inputs.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import cast

from pytoce.run_mfa.scenario.rules import OPTIMIZATION_PARENT_OVERLAY_ORIGIN
from pytoce.run_mfa.scenario.scenario_contract import ScenarioEvaluation
from pytoce.shared.runtime.filesystem import payload_sha256

from .candidate_authority import (
    CandidateInterventionAuthority,
    _immutable_assignment_records,
)
from .demand_lifetime_search import DEMAND_REPAIR_METHOD_IDENTITY
from .interventions import INTERVENTION_ASSIGNMENT_COLUMNS
from .worker_protocol import CandidateRepairReference, CandidateWorkerResult

_FEEDSTOCK = frozenset({"bio_production_rate", "daccu_production_rate"})
_ASSIGNED_PARAMETERS = _FEEDSTOCK | {
    "stock_projection",
    "lifetime_mean",
    "collection_rate",
    "mechanical_recycling_rate",
    "chemical_recycling_rate",
    "incineration_rate",
}
_AUTHORED = frozenset({"user_foreground_scenario", OPTIMIZATION_PARENT_OVERLAY_ORIGIN})
_FAMILY_SCHEMA = "pytoce_feedstock_independent_repair_family_schema_v1"
AssignmentKey = tuple[str | int | float | None, ...]


@dataclass(frozen=True, slots=True)
class CandidateRepairFamilyContext:
    """Read-only parent controls and execution metadata within one worker authority."""

    parent_values: Mapping[AssignmentKey, float]
    parent_nonfeedstock_authored: bool
    parent_optimization_overlay: bool


def prepare_repair_family_context(parent: ScenarioEvaluation) -> CandidateRepairFamilyContext:
    """Retain admitted parent values once."""
    compiled = parent.compiled_parameters
    records = _immutable_assignment_records(
        compiled.loc[
            compiled["parameter_id"].isin(_ASSIGNED_PARAMETERS),
            list(INTERVENTION_ASSIGNMENT_COLUMNS),
        ]
    )
    origins = compiled["value_origin"]
    return CandidateRepairFamilyContext(
        {row[:-1]: cast(float, row[-1]) for row in records},
        bool((origins.isin(_AUTHORED) & ~compiled["parameter_id"].isin(_FEEDSTOCK)).any()),
        bool(origins.eq(OPTIMIZATION_PARENT_OVERLAY_ORIGIN).any()),
    )


def candidate_repair_family_identity(
    intervention: CandidateInterventionAuthority,
    context: CandidateRepairFamilyContext,
    *,
    worker_authority_identity: str,
) -> str | None:
    """Prove equal non-feedstock execution, or leave the original path responsible.

    A fixed changed control forces both source updates and optimization-overlay
    diagnostics throughout the repair pool. Demand assignments at and after T
    are fixed; earlier demand can be repaired. With no demand freedom, an
    already-authored parent also proves the source-update mode for its sole path.
    The exact parent, default path, and explicit path retain their separate identities.

    Rate checks here establish reuse eligibility: an
    ineligible member follows ordinary admission with its original error order.
    """
    circularity = intervention.circularity_evaluation_projection.decision
    if circularity.uses_exact_parent:
        return None
    decision = intervention.demand_lifetime_evaluation_projection.decision
    nonfeedstock: list[AssignmentKey] = []
    rates: dict[AssignmentKey, dict[str, float]] = {}
    fixed_changed = False
    any_changed = False
    for row in intervention.assignment_records:
        parameter = str(row[0])
        value = cast(float, row[-1])
        parent = context.parent_values.get(row[:-1])
        if parent is None:
            return None
        changed = value != parent
        any_changed |= changed
        if parameter in _FEEDSTOCK:
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                return None
            rates.setdefault(row[1:-1], {})[parameter] = value
        else:
            nonfeedstock.append(row)
            fixed_changed |= changed and (
                parameter != "stock_projection" or cast(int, row[1]) >= decision.target_year
            )
    if not rates or any(
        len(values) != 2 or values["bio_production_rate"] + values["daccu_production_rate"] > 1.0
        for values in rates.values()
    ):
        return None
    if not fixed_changed and not (
        decision.demand_reduction_pct == 0 and context.parent_nonfeedstock_authored
    ):
        return None
    return (
        _FAMILY_SCHEMA
        + "::"
        + payload_sha256(
            {
                "schema_version": _FAMILY_SCHEMA,
                "repair_method_identity": DEMAND_REPAIR_METHOD_IDENTITY,
                "worker_authority_identity": worker_authority_identity,
                "physical_parent_identity": intervention.physical_parent_identity,
                "parent_foreground_scenario_identity": (
                    intervention.parent_foreground_scenario_identity
                ),
                "demand_lifetime_parent_identity": intervention.demand_lifetime_parent_identity,
                "circularity_parent_identity": intervention.circularity_parent_identity,
                "requested_demand_lifetime": decision.coordinates(),
                "circularity_target_year": circularity.target_year,
                "nonfeedstock_assignments": nonfeedstock,
                "source_updates": True,
                "optimization_overlay": context.parent_optimization_overlay or any_changed,
            }
        )
    )


def repair_reference_from_result(
    result: CandidateWorkerResult, family_identity: str
) -> CandidateRepairReference:
    """Extract the chosen path and feasibility record from one completed requested result."""
    return CandidateRepairReference(
        family_identity=family_identity,
        requested_intervention_identity=result.requested_intervention_identity,
        worker_result_identity=result.identity,
        demand_lifetime=result.intervention.demand_lifetime_evaluation_projection.decision,
        status=result.repair.status,
        pool_candidate_count=result.repair.pool_candidate_count,
        demand_departure_pp=result.repair.demand_departure_pp,
        linear_diagnostic=result.repair.linear_diagnostic,
        stock_projection_diagnostic=result.repair.stock_projection_diagnostic,
        physical_diagnostic=result.physical.diagnostic,
    )
