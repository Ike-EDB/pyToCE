"""Translate optimization percentages into prospective MFA assignments.

The optimization uses integer percentage targets. Demand reduction scales the
parent's in-use stock targets, and lifetime extension scales its mean lifetimes,
relative to their values in each year. Collection, treatment, and virgin
feedstock shares interpolate each region's contemporaneous parent values toward
a common target.
"""

from dataclasses import dataclass, replace
from fractions import Fraction
from math import fsum
from typing import Final, cast

import numpy as np
import pandas as pd

from pytoce.shared.runtime.filesystem import (
    canonical_table_sha256,
    payload_sha256,
)

DEMAND_LIFETIME_METHOD_IDENTITY: Final = "pytoce_optimization_demand_lifetime_intervention_v5"
CIRCULARITY_METHOD_IDENTITY: Final = "pytoce_optimization_circularity_intervention_v4"
CIRCULARITY_PROPOSAL_METHOD_IDENTITY: Final = "pytoce_optimization_circularity_proposal_v3"
CIRCULARITY_EVALUATION_PROJECTION_METHOD_IDENTITY: Final = (
    "pytoce_optimization_circularity_evaluation_projection_v3"
)
DEMAND_LIFETIME_EVALUATION_PROJECTION_METHOD_IDENTITY: Final = (
    "pytoce_optimization_demand_lifetime_evaluation_projection_v4"
)
INTERVENTION_DISTANCE_INPUT_METHOD_IDENTITY: Final = "pytoce_optimization_distance_inputs_v6"
_TRAJECTORY_IDENTITY_PREFIX = "pytoce_optimization_demand_lifetime_trajectory_v5::"
_CIRCULARITY_IDENTITY_PREFIX = "pytoce_optimization_circularity_trajectory_v4::"
_CIRCULARITY_PROPOSAL_IDENTITY_PREFIX = "pytoce_optimization_circularity_proposal_v3::"
_CIRCULARITY_EVALUATION_PROJECTION_IDENTITY_PREFIX = (
    "pytoce_optimization_circularity_evaluation_projection_v3::"
)
_DEMAND_LIFETIME_EVALUATION_PROJECTION_IDENTITY_PREFIX = (
    "pytoce_optimization_demand_lifetime_evaluation_projection_v4::"
)
_DISTANCE_INPUT_IDENTITY_PREFIX = "pytoce_optimization_distance_inputs_v6::"
_TREATMENT_PARAMETERS = (
    "mechanical_recycling_rate",
    "chemical_recycling_rate",
    "incineration_rate",
    "landfill_share",
)
_FEEDSTOCK_PARAMETERS = (
    "fossil_share",
    "bio_production_rate",
    "daccu_production_rate",
)
_COORDINATE_COLUMNS = ("region", "material", "good", "process", "route")
INTERVENTION_ASSIGNMENT_COLUMNS: Final = (
    "parameter_id",
    "year",
    *_COORDINATE_COLUMNS,
    "resolved_value",
)


AnnualControl = tuple[Fraction, ...]
_CONTROL_YEARS = tuple(range(2019, 2101))


@dataclass(frozen=True, slots=True)
class DemandLifetimeDecision:
    """Endpoints and exact common annual fractions for 2019 through 2100.

    Rational values preserve exact annual control coordinates and seed/solver
    increment values.
    Assignment/reporting owners convert to float; lifetime spread remains derived
    by the scenario owner. Omitted arrays construct the exact linear reference.
    """

    demand_reduction_pct: int
    lifetime_extension_pct: int
    target_year: int = 2050
    demand_fraction: AnnualControl = ()
    lifetime_fraction: AnnualControl = ()

    def __post_init__(self) -> None:
        """Construct the linear reference when complete annual values are omitted."""
        for field, endpoint in (
            ("demand_fraction", self.demand_reduction_pct),
            ("lifetime_fraction", self.lifetime_extension_pct),
        ):
            if not getattr(self, field):
                object.__setattr__(self, field, _reference_fractions(endpoint, self.target_year))

    def coordinates(
        self,
    ) -> tuple[int, int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        """Return exact annual rational coordinates for decision identity and transport."""
        return (
            self.demand_reduction_pct,
            self.lifetime_extension_pct,
            self.target_year,
            tuple((value.numerator, value.denominator) for value in self.demand_fraction),
            tuple((value.numerator, value.denominator) for value in self.lifetime_fraction),
        )

    @property
    def uses_exact_parent(self) -> bool:
        """Report whether demand and lifetime remain at their baseline values."""
        return self.demand_reduction_pct == 0 and self.lifetime_extension_pct == 0


@dataclass(frozen=True, slots=True)
class DemandLifetimeTrajectories:
    """Store annual demand and lifetime values and their baseline-based identity."""

    identity: str
    parent_stock_target_identity: str
    parent_lifetime_identity: str
    decision: DemandLifetimeDecision
    assignments: pd.DataFrame
    uses_exact_parent: bool


@dataclass(frozen=True, slots=True)
class CircularityDecision:
    """Direct collection, treatment-share, and feedstock-share targets.

    All target fields are ``None`` when the baseline is unchanged. An active
    decision supplies every target; the selected shares must form complete compositions.
    """

    collection_pct: int | None = None
    mechanical_recycling_pct: int | None = None
    chemical_recycling_pct: int | None = None
    incineration_pct: int | None = None
    landfill_pct: int | None = None
    fossil_pct: int | None = None
    biogenic_pct: int | None = None
    daccu_pct: int | None = None
    target_year: int = 2050

    @classmethod
    def parent(cls, *, target_year: int = 2050) -> "CircularityDecision":
        """Create a decision that leaves every baseline percentage unchanged."""
        return cls(target_year=target_year)

    @property
    def uses_exact_parent(self) -> bool:
        """Report whether collection, treatment, and feedstock remain unchanged."""
        return self.collection_pct is None

    @property
    def treatment_percentages(self) -> tuple[int, int, int, int]:
        """Return the active decision's four treatment percentages."""
        return cast(
            tuple[int, int, int, int],
            (
                self.mechanical_recycling_pct,
                self.chemical_recycling_pct,
                self.incineration_pct,
                self.landfill_pct,
            ),
        )

    @property
    def feedstock_percentages(self) -> tuple[int, int, int]:
        """Return the active decision's fossil, biogenic, and DACCU percentages."""
        return cast(tuple[int, int, int], (self.fossil_pct, self.biogenic_pct, self.daccu_pct))

    def percentages(self) -> tuple[int, int, int, int, int, int, int, int] | None:
        """Return the eight circularity percentages in documented field order."""
        if self.uses_exact_parent:
            return None
        return cast(
            tuple[int, int, int, int, int, int, int, int],
            (
                self.collection_pct,
                self.mechanical_recycling_pct,
                self.chemical_recycling_pct,
                self.incineration_pct,
                self.landfill_pct,
                self.fossil_pct,
                self.biogenic_pct,
                self.daccu_pct,
            ),
        )


@dataclass(frozen=True, slots=True)
class CircularityTrajectories:
    """Store complete regional fractions and editable assignments for 2020--2100.

    ``annual_values`` retains all eight controls, including landfill and fossil
    shares. An unchanged trajectory references the admitted parent table directly.
    ``assignments`` contains only the six controls accepted by the scenario owner.
    """

    identity: str
    parent_identity: str
    decision: CircularityDecision
    annual_values: pd.DataFrame
    assignments: pd.DataFrame
    uses_exact_parent: bool


@dataclass(frozen=True, slots=True)
class InterventionDistanceInputs:
    """Five intervention changes used to compare candidates.

    ``identity`` binds the record to the baseline MFA result, the baseline
    demand and lifetime table, and the two calculated trajectories. The four
    corresponding identity fields record those inputs. ``target_year`` is the
    year by which the changes are complete. Demand reduction and lifetime
    extension are fractions of their baseline values; collection is the mean
    absolute change in regional shares; treatment and feedstock use mean total
    variation across their complete share vectors. All five distances are
    fractions from zero to one.
    """

    identity: str
    physical_parent_identity: str
    demand_lifetime_parent_identity: str
    demand_lifetime_trajectory_identity: str
    circularity_trajectory_identity: str
    target_year: int
    demand_reduction_fraction: float
    lifetime_extension_fraction: float
    collection_distance: float
    treatment_share_distance: float
    feedstock_share_distance: float


@dataclass(frozen=True, slots=True)
class CircularityEvaluationProjection:
    """Connect circularity targets to their candidate calculation."""

    identity: str
    proposal_identity: str
    decision: CircularityDecision
    circularity_trajectory_identity: str
    intervention_authority_identity: str
    physical_parent_identity: str
    demand_lifetime_parent_identity: str
    distance_inputs_identity: str


@dataclass(frozen=True, slots=True)
class DemandLifetimeEvaluationProjection:
    """Link one direct demand and lifetime target to its evaluation."""

    identity: str
    decision: DemandLifetimeDecision
    demand_lifetime_trajectory_identity: str
    physical_parent_identity: str
    demand_lifetime_parent_identity: str


def build_demand_lifetime_evaluation_projection(
    *,
    demand_lifetime: DemandLifetimeTrajectories,
    physical_parent_identity: str,
    demand_lifetime_parent_identity: str,
) -> DemandLifetimeEvaluationProjection:
    """Bind an existing demand/lifetime trajectory and its parents to candidate evaluation."""
    projection = DemandLifetimeEvaluationProjection(
        identity="",
        decision=demand_lifetime.decision,
        demand_lifetime_trajectory_identity=demand_lifetime.identity,
        physical_parent_identity=physical_parent_identity,
        demand_lifetime_parent_identity=demand_lifetime_parent_identity,
    )
    return replace(projection, identity=demand_lifetime_evaluation_projection_identity(projection))


def demand_lifetime_evaluation_projection_identity(
    projection: DemandLifetimeEvaluationProjection,
) -> str:
    """Identify the demand and lifetime inputs used for candidate evaluation."""
    return _DEMAND_LIFETIME_EVALUATION_PROJECTION_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": DEMAND_LIFETIME_EVALUATION_PROJECTION_METHOD_IDENTITY,
            "demand_lifetime_method_identity": DEMAND_LIFETIME_METHOD_IDENTITY,
            "decision": list(projection.decision.coordinates()),
            "demand_lifetime_trajectory_identity": projection.demand_lifetime_trajectory_identity,
            "physical_parent_identity": projection.physical_parent_identity,
            "demand_lifetime_parent_identity": projection.demand_lifetime_parent_identity,
        }
    )


def circularity_decision_coordinates(decision: CircularityDecision) -> tuple[int | None, ...]:
    """Return the canonical direct circularity target coordinates."""

    percentages = decision.percentages()
    if percentages is None:
        return (None, None, None, None, None, None, None, None, decision.target_year)
    return (*percentages, decision.target_year)


def circularity_proposal_identity(
    decision: CircularityDecision,
    *,
    circularity_trajectory_identity: str,
) -> str:
    """Identify one circularity decision and its compiled trajectory."""
    return _CIRCULARITY_PROPOSAL_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": CIRCULARITY_PROPOSAL_METHOD_IDENTITY,
            "circularity_method_identity": CIRCULARITY_METHOD_IDENTITY,
            "circularity_trajectory_identity": circularity_trajectory_identity,
            "decision": list(circularity_decision_coordinates(decision)),
        }
    )


def build_circularity_evaluation_projection(
    *,
    proposal_decision: CircularityDecision,
    circularity_trajectory_identity: str,
    intervention_authority_identity: str,
    physical_parent_identity: str,
    demand_lifetime_parent_identity: str,
    distance_inputs_identity: str,
) -> CircularityEvaluationProjection:
    """Bind a circularity decision and existing trajectory to candidate evaluation."""
    proposal_identity = circularity_proposal_identity(
        proposal_decision,
        circularity_trajectory_identity=circularity_trajectory_identity,
    )
    projection = CircularityEvaluationProjection(
        identity="",
        proposal_identity=proposal_identity,
        decision=proposal_decision,
        circularity_trajectory_identity=circularity_trajectory_identity,
        intervention_authority_identity=intervention_authority_identity,
        physical_parent_identity=physical_parent_identity,
        demand_lifetime_parent_identity=demand_lifetime_parent_identity,
        distance_inputs_identity=distance_inputs_identity,
    )
    return replace(projection, identity=circularity_evaluation_projection_identity(projection))


def circularity_evaluation_projection_identity(
    projection: CircularityEvaluationProjection,
) -> str:
    """Identify the circularity inputs used for candidate evaluation."""
    return _CIRCULARITY_EVALUATION_PROJECTION_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": CIRCULARITY_EVALUATION_PROJECTION_METHOD_IDENTITY,
            "proposal_identity": projection.proposal_identity,
            "circularity_trajectory_identity": projection.circularity_trajectory_identity,
            "intervention_authority_identity": projection.intervention_authority_identity,
            "physical_parent_identity": projection.physical_parent_identity,
            "demand_lifetime_parent_identity": projection.demand_lifetime_parent_identity,
            "distance_inputs_identity": projection.distance_inputs_identity,
        }
    )


def intervention_distance_inputs_identity(inputs: InterventionDistanceInputs) -> str:
    """Identify the five normalized changes used to rank interventions."""
    return _DISTANCE_INPUT_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": INTERVENTION_DISTANCE_INPUT_METHOD_IDENTITY,
            "physical_parent_identity": inputs.physical_parent_identity,
            "demand_lifetime_parent_identity": inputs.demand_lifetime_parent_identity,
            "demand_lifetime_trajectory_identity": inputs.demand_lifetime_trajectory_identity,
            "circularity_trajectory_identity": inputs.circularity_trajectory_identity,
            "target_year": inputs.target_year,
            "demand_reduction_fraction": inputs.demand_reduction_fraction,
            "lifetime_extension_fraction": inputs.lifetime_extension_fraction,
            "collection_distance": inputs.collection_distance,
            "treatment_share_distance": inputs.treatment_share_distance,
            "feedstock_share_distance": inputs.feedstock_share_distance,
        }
    )


def build_intervention_distance_inputs(
    *,
    physical_parent_identity: str,
    demand_lifetime_parent_identity: str,
    demand_lifetime: DemandLifetimeTrajectories,
    circularity: CircularityTrajectories,
    parent_circularity: pd.DataFrame,
) -> InterventionDistanceInputs:
    """Measure intervention endpoints relative to the parent in the target year.

    A varying parent can share a candidate's endpoint while their earlier
    annual values differ. Endpoint distance uses target-year values; annual paths
    are recorded separately.
    """

    target_year = (
        circularity.decision.target_year
        if demand_lifetime.uses_exact_parent and not circularity.uses_exact_parent
        else demand_lifetime.decision.target_year
    )
    demand_distance = demand_lifetime.decision.demand_reduction_pct / 100.0
    lifetime_distance = demand_lifetime.decision.lifetime_extension_pct / 100.0
    collection_distance = 0.0
    treatment_distance = 0.0
    feedstock_distance = 0.0
    if not circularity.uses_exact_parent:
        parent = parent_circularity
        # Compare endpoint shares with contemporaneous parent shares. Constant
        # regional parent shares give the same reference in every completion year.
        endpoint = parent.loc[parent["year"].eq(target_year)].sort_values("region", kind="stable")
        collection_target = cast(int, circularity.decision.collection_pct) / 100.0
        collection_distance = _mean_absolute_change(
            np.full(len(endpoint), collection_target, dtype=np.float64),
            endpoint["collection_rate"].to_numpy(dtype=np.float64),
        )
        treatment_target = (
            np.asarray(
                circularity.decision.treatment_percentages,
                dtype=np.float64,
            )
            / 100.0
        )
        treatment_parent = endpoint.loc[:, list(_TREATMENT_PARAMETERS)].to_numpy(dtype=np.float64)
        treatment_distance = _mean_total_variation(treatment_parent, treatment_target)
        feedstock_target = (
            np.asarray(
                circularity.decision.feedstock_percentages,
                dtype=np.float64,
            )
            / 100.0
        )
        feedstock_parent = endpoint.loc[:, list(_FEEDSTOCK_PARAMETERS)].to_numpy(dtype=np.float64)
        feedstock_distance = _mean_total_variation(feedstock_parent, feedstock_target)
    result = InterventionDistanceInputs(
        identity="",
        physical_parent_identity=physical_parent_identity,
        demand_lifetime_parent_identity=demand_lifetime_parent_identity,
        demand_lifetime_trajectory_identity=demand_lifetime.identity,
        circularity_trajectory_identity=circularity.identity,
        target_year=target_year,
        demand_reduction_fraction=demand_distance,
        lifetime_extension_fraction=lifetime_distance,
        collection_distance=collection_distance,
        treatment_share_distance=treatment_distance,
        feedstock_share_distance=feedstock_distance,
    )
    return replace(result, identity=intervention_distance_inputs_identity(result))


def _mean_absolute_change(candidate: np.ndarray, parent: np.ndarray) -> float:
    """Average the absolute elementwise change from baseline to candidate."""
    values = np.abs(candidate - parent).ravel(order="C")
    return fsum(float(value) for value in values) / int(values.size)


def _mean_total_variation(parent: np.ndarray, target: np.ndarray) -> float:
    """Average the total-variation distance between rows of shares."""
    values = 0.5 * np.abs(parent - target).sum(axis=1)
    return fsum(float(value) for value in values) / int(values.size)


def transition_weight(year: int, target_year: int) -> float:
    """Return exact linear reference progress, zero through 2019 and one after completion."""
    span = target_year - 2019
    return min(max(year - 2019, 0), span) / span


def _reference_fractions(endpoint: int, target_year: int) -> AnnualControl:
    """Retain the exact existing reference floats as reversible rational controls."""
    return tuple(
        Fraction.from_float((endpoint / 100.0) * transition_weight(year, target_year))
        for year in _CONTROL_YEARS
    )


def build_demand_lifetime_trajectories(
    *,
    parent_stock_targets: pd.DataFrame,
    parent_lifetimes: pd.DataFrame,
    parent_stock_target_identity: str,
    parent_lifetime_identity: str,
    decision: DemandLifetimeDecision,
) -> DemandLifetimeTrajectories:
    """Scale parent in-use stock targets and mean lifetimes by the decision's annual fractions."""

    stock = parent_stock_targets
    lifetimes = parent_lifetimes
    if decision.uses_exact_parent:
        return DemandLifetimeTrajectories(
            identity=_TRAJECTORY_IDENTITY_PREFIX
            + payload_sha256(
                {
                    "method_identity": DEMAND_LIFETIME_METHOD_IDENTITY,
                    "parent_stock_target_identity": parent_stock_target_identity,
                    "parent_lifetime_identity": parent_lifetime_identity,
                    "uses_exact_parent": True,
                }
            ),
            parent_stock_target_identity=parent_stock_target_identity,
            parent_lifetime_identity=parent_lifetime_identity,
            decision=decision,
            assignments=pd.DataFrame(columns=INTERVENTION_ASSIGNMENT_COLUMNS),
            uses_exact_parent=True,
        )

    demand_fraction = (
        stock["year"]
        .map(lambda year: float(decision.demand_fraction[int(year) - 2019]))
        .to_numpy(dtype=np.float64)
    )
    lifetime_fraction = (
        lifetimes["year"]
        .map(lambda year: float(decision.lifetime_fraction[int(year) - 2019]))
        .to_numpy(dtype=np.float64)
    )
    stock_parent = stock["stock_projection"].to_numpy(dtype=np.float64)
    lifetime_parent = lifetimes["lifetime_mean"].to_numpy(dtype=np.float64)
    stock_values = stock_parent * (1.0 - demand_fraction)
    lifetime_values = lifetime_parent * (1.0 + lifetime_fraction)
    frames: list[pd.DataFrame] = []
    if decision.demand_reduction_pct:
        frames.append(
            _assignment_frame(
                stock.loc[:, ["year", "region", "good"]],
                parameter_id="stock_projection",
                values=stock_values,
            )
        )
    if decision.lifetime_extension_pct:
        frames.append(
            _assignment_frame(
                lifetimes.loc[:, ["year", "good"]],
                parameter_id="lifetime_mean",
                values=lifetime_values,
            )
        )
    assignments = _canonical_assignments(frames)
    identity = _TRAJECTORY_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": DEMAND_LIFETIME_METHOD_IDENTITY,
            "parent_stock_target_identity": parent_stock_target_identity,
            "parent_lifetime_identity": parent_lifetime_identity,
            "annual_demand_fraction": [float(value) for value in decision.demand_fraction],
            "annual_lifetime_fraction": [float(value) for value in decision.lifetime_fraction],
            "target_year": decision.target_year,
            "assignments_table_identity": canonical_table_sha256(assignments),
            "uses_exact_parent": False,
        }
    )
    return DemandLifetimeTrajectories(
        identity=identity,
        parent_stock_target_identity=parent_stock_target_identity,
        parent_lifetime_identity=parent_lifetime_identity,
        decision=decision,
        assignments=assignments,
        uses_exact_parent=False,
    )


def build_circularity_trajectories(
    *,
    parent: pd.DataFrame,
    parent_identity: str,
    decision: CircularityDecision,
) -> CircularityTrajectories:
    """Interpolate each year's regional parent shares toward the common targets."""

    if decision.uses_exact_parent:
        return CircularityTrajectories(
            identity=_CIRCULARITY_IDENTITY_PREFIX
            + payload_sha256(
                {
                    "method_identity": CIRCULARITY_METHOD_IDENTITY,
                    "parent_identity": parent_identity,
                    "uses_exact_parent": True,
                }
            ),
            parent_identity=parent_identity,
            decision=decision,
            annual_values=parent,
            assignments=pd.DataFrame(columns=INTERVENTION_ASSIGNMENT_COLUMNS),
            uses_exact_parent=True,
        )

    values = parent
    weight = (
        values["year"]
        .map(lambda value: transition_weight(int(value), decision.target_year))
        .to_numpy(dtype=np.float64)[:, None]
    )
    collection_parent = values[["collection_rate"]].to_numpy(dtype=np.float64)
    collection_target = np.full_like(
        collection_parent,
        cast(int, decision.collection_pct) / 100.0,
    )
    treatment_parent = values.loc[:, list(_TREATMENT_PARAMETERS)].to_numpy(dtype=np.float64)
    treatment_target = np.broadcast_to(
        np.asarray(
            decision.treatment_percentages,
            dtype=np.float64,
        )
        / 100.0,
        treatment_parent.shape,
    )
    feedstock_parent = values.loc[:, list(_FEEDSTOCK_PARAMETERS)].to_numpy(dtype=np.float64)
    feedstock_target = np.broadcast_to(
        np.asarray(
            decision.feedstock_percentages,
            dtype=np.float64,
        )
        / 100.0,
        feedstock_parent.shape,
    )
    collection = collection_parent + weight * (collection_target - collection_parent)
    treatment = treatment_parent + weight * (treatment_target - treatment_parent)
    feedstock = feedstock_parent + weight * (feedstock_target - feedstock_parent)
    annual_values = values.loc[:, ["year", "region"]].assign(collection_rate=collection[:, 0])
    annual_values[list(_TREATMENT_PARAMETERS)] = treatment
    annual_values[list(_FEEDSTOCK_PARAMETERS)] = feedstock
    # The scenario/MFA owners derive landfill and fossil from the three and two
    # editable upstream shares respectively. Reporting retains all eight paths
    # from the same interpolation, including when an MFA evaluation fails.
    frames = [
        _assignment_frame(
            annual_values.loc[:, ["year", "region"]],
            parameter_id=parameter_id,
            values=annual_values[parameter_id].to_numpy(dtype=np.float64),
        )
        for parameter_id in (
            "collection_rate",
            *_TREATMENT_PARAMETERS[:3],
            *_FEEDSTOCK_PARAMETERS[1:],
        )
    ]
    assignments = _canonical_assignments(frames)
    identity = _CIRCULARITY_IDENTITY_PREFIX + payload_sha256(
        {
            "method_identity": CIRCULARITY_METHOD_IDENTITY,
            "parent_identity": parent_identity,
            "decision": list(circularity_decision_coordinates(decision)),
            "assignments_table_identity": canonical_table_sha256(assignments),
            "uses_exact_parent": False,
        }
    )
    return CircularityTrajectories(
        identity=identity,
        parent_identity=parent_identity,
        decision=decision,
        annual_values=annual_values,
        assignments=assignments,
        uses_exact_parent=False,
    )


def _assignment_frame(
    coordinates: pd.DataFrame,
    *,
    parameter_id: str,
    values: np.ndarray,
) -> pd.DataFrame:
    """Format resolved values as canonical scenario assignment rows."""
    result = coordinates.copy()
    result.insert(0, "parameter_id", parameter_id)
    for column in _COORDINATE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["resolved_value"] = values
    return cast(pd.DataFrame, result.loc[:, list(INTERVENTION_ASSIGNMENT_COLUMNS)])


def _canonical_assignments(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine and deterministically order scenario assignment tables."""
    return pd.concat(frames, ignore_index=True).sort_values(
        ["parameter_id", "year", *_COORDINATE_COLUMNS],
        kind="stable",
        na_position="first",
        ignore_index=True,
    )
