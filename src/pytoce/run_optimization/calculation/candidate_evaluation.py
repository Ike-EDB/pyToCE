"""Evaluate one optimization candidate and characterize selected candidates.

One MFA calculation is shared across REMIND backgrounds, allocated carrying
capacity cases, and reporting methods. Environmental scoring updates baseline
scores using precomputed provider unit responses, candidate activation changes
and actual PA receiving quantities.
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from math import fsum
from typing import Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.run_lca.calculation.custody import (
    CohortWeightsByYear,
    prepare_carbon_cohort_weights,
    virgin_route_shares,
)
from pytoce.run_lca.calculation.foreground import (
    H12_REGIONS,
    OPTIMIZATION_ACTIVATION_METHOD_IDENTITY,
    OptimizationActivationContext,
    compile_optimization_activations_from_history,
    optimization_activation_delta,
)
from pytoce.run_lca.calculation.pa_receiving import (
    PaDisposalEvent,
    PaRecyclingEvent,
    PaSpinningEvent,
)
from pytoce.run_lca.calculation.providers import ProviderRoutes
from pytoce.run_mfa.model.cohorts import OPTIMIZATION_COHORT_FEASIBILITY_CODES
from pytoce.run_mfa.model.materials import load_material_composition
from pytoce.run_mfa.model.mfa import (
    OPTIMIZATION_CANDIDATE_MFA_METHOD_IDENTITY,
    OPTIMIZATION_MFA_FEASIBILITY_CODES,
    MFACandidateEvaluation,
    MFAEvaluation,
)
from pytoce.run_mfa.scenario.parent_relative import (
    OPTIMIZATION_PARENT_OVERLAY_METHOD_IDENTITY,
    compile_parent_relative_scenario_assignments,
)
from pytoce.run_mfa.scenario.scenario_contract import (
    OptimizationPhysicalInfeasibility,
    ScenarioEvaluation,
    ScenarioFeasibilityError,
)
from pytoce.run_optimization.calculation.candidate_authority import (
    _ACTIVATION_IDENTITY_PREFIX,
    CANDIDATE_EVALUATION_METHOD_IDENTITY,
    CandidateInterventionAuthority,
    CandidateMFASession,
    CandidateParentAuthority,
    PhysicalEvaluation,
    _intervention_assignments,
    physical_evaluation_identity,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    OptimizationScores,
    ResultScores,
    _optimization_scores,
    _pa_method_indices,
    prepare_search_scoring,
)
from pytoce.run_optimization.calculation.candidate_projection_arrays import (
    CandidateParentScoreProjection,
    _array_identity,
    _dated_projection,
    _DatedProjection,
    _projection_baseline,
    _readonly,
    _score_arrays,
)
from pytoce.run_optimization.calculation.interventions import (
    CIRCULARITY_METHOD_IDENTITY,
    DEMAND_LIFETIME_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.pa_candidate_projection import (
    PaCandidateProjection,
    PaCandidateScores,
    calculate_pa_candidate_projection,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    ResponseBankProjection,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256, payload_sha256

_PHYSICAL_REQUEST_IDENTITY_PREFIX = "pytoce_optimization_physical_request_v4::"
_RESULT_IDENTITY_PREFIX = "pytoce_optimization_result_scores_v5::"
_RESULT_SCORE_IDENTITY_PREFIX = "pytoce_optimization_result_score_record_v1::"
_SELECTED_CHARACTERIZATION_PREFIX = "pytoce_optimization_selected_characterization_v1::"
_FEEDSTOCK_REQUIREMENTS_IDENTITY_PREFIX = "pytoce_optimization_feedstock_requirements_schema_v1::"

_YEARS = tuple(range(2019, 2101))
_METHOD_SIGNATURE_COLUMNS = (
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "unit",
    "factor_identity",
)
_FEEDSTOCK_REQUIREMENTS_COLUMNS = (
    "year",
    "region",
    "material",
    "constituent",
    "product",
    "quantity",
    "virgin_feedstock_route",
    "amount",
    "unit",
)

_COHORT_WEIGHT_CACHE_SIZE = 2
_PHYSICAL_INFEASIBILITY_CODES = (
    OPTIMIZATION_COHORT_FEASIBILITY_CODES | OPTIMIZATION_MFA_FEASIBILITY_CODES
)

FloatArray: TypeAlias = NDArray[np.float64]
PhysicalStatus: TypeAlias = Literal[
    "physical_feasible",
    "physical_infeasible",
]


@dataclass(frozen=True, slots=True)
class AdmittedCandidateCalculation:
    """A candidate request with an admitted MFA evaluation or reuse of the exact parent."""

    intervention: CandidateInterventionAuthority
    evaluation: MFACandidateEvaluation | None


@dataclass(frozen=True, slots=True)
class FeasibleCandidateCalculation:
    """A feasible MFA result and the inputs needed for scoring and reporting."""

    physical: PhysicalEvaluation
    activation_delta: pd.DataFrame
    feedstock_requirements: pd.DataFrame


@dataclass(frozen=True, slots=True)
class InfeasibleCandidateCalculation:
    """An infeasible MFA outcome with its MFA constraint diagnostic."""

    physical: PhysicalEvaluation


CandidatePhysicalCalculation: TypeAlias = (
    FeasibleCandidateCalculation | InfeasibleCandidateCalculation
)


def _stable_sum(values: pd.Series) -> float:
    """Sum floating-point values with ``math.fsum``."""

    return fsum(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class SelectedCandidateCharacterization:
    """Reporting LCIA results for one selected original evaluation."""

    identity: str
    selected_result_identity: str
    result_lcia_selection_identity: str
    scores: ResultScores


@dataclass(frozen=True, slots=True)
class _SelectedCharacterizationContext:
    """Result projection and baseline LCIA results for selected candidates."""

    physical_parent_identity: str
    result_parent_identity: str
    projection: _DatedProjection
    baseline_scores: FloatArray
    baseline_norms: FloatArray
    pa_parent: PaCandidateScores
    pa_indices: np.ndarray


def _selected_feedstock_requirements(
    evaluation: MFAEvaluation | MFACandidateEvaluation,
) -> pd.DataFrame:
    """Report virgin monomer inputs or whole-polymer outputs by feed route.

    Eligible constituents use their admitted gross virgin ethylene/propylene
    requirements; other constituents use domestic plus exported new polymer.
    Values are Mt of the named product.
    Feedstock route selects the production technology.
    Nominal polymer-equivalent values supply only the already admitted ratios.
    Unlike products and quantities must remain distinct in downstream reporting.
    """
    eligible = {
        (item.material, item.constituent)
        for item in load_material_composition().constituents
        if item.chemical_polymer is not None
    }
    realization = evaluation.material_realization
    coordinates = pd.MultiIndex.from_frame(
        realization[["destination_material", "destination_constituent"]]
    )
    selected = realization.loc[
        realization["year"].between(2019, 2100)
        & (
            (realization["flow_name"].eq("virgin_monomer_consumption") & coordinates.isin(eligible))
            | (
                realization["flow_name"].isin(("new_polymer_domestic", "new_polymer_export"))
                & ~coordinates.isin(eligible)
            )
        ),
        [
            "year",
            "region",
            "destination_material",
            "destination_constituent",
            "product",
            "flow_name",
            "amount",
            "unit",
        ],
    ].rename(
        columns={
            "destination_material": "material",
            "destination_constituent": "constituent",
            "flow_name": "quantity",
        }
    )
    selected["quantity"] = selected["quantity"].replace(
        {
            "new_polymer_domestic": "new_polymer_output",
            "new_polymer_export": "new_polymer_output",
        }
    )
    keys = ["year", "region", "material", "constituent", "product", "quantity", "unit"]
    grouped = (
        selected.groupby(keys, observed=True, sort=True)["amount"].agg(_stable_sum).reset_index()
    )
    shares = virgin_route_shares(evaluation.nominal_feedstock_equivalents)
    rows = [
        (
            item.year,
            item.region,
            item.material,
            item.constituent,
            item.product,
            item.quantity,
            route,
            float(cast(float, item.amount)) * float(fraction),
            item.unit,
        )
        for item in grouped.itertuples(index=False)
        for route, fraction in zip(
            ("fossil", "biogenic", "atmospheric_dac"),
            shares[cast(tuple[int, str, str], (item.year, item.region, item.material))],
            strict=True,
        )
    ]
    result = pd.DataFrame.from_records(rows, columns=_FEEDSTOCK_REQUIREMENTS_COLUMNS)
    result["year"] = result["year"].astype(np.int64)
    result["amount"] = result["amount"].astype(np.float64)
    for column in (
        "region",
        "material",
        "constituent",
        "product",
        "quantity",
        "virgin_feedstock_route",
        "unit",
    ):
        result[column] = result[column].astype("string")
    return result.sort_values(
        list(_FEEDSTOCK_REQUIREMENTS_COLUMNS[:-2]), kind="stable", ignore_index=True
    )


def feedstock_requirements_identity(frame: pd.DataFrame) -> str:
    """Identify feedstock requirements from one candidate MFA calculation."""
    return _FEEDSTOCK_REQUIREMENTS_IDENTITY_PREFIX + canonical_table_sha256(frame)


def _prepare_selected_characterization_context(
    *,
    physical_parent: CandidateParentAuthority,
    result_parent: CandidateParentScoreProjection,
    result_projection: ResponseBankProjection,
    pa_parent: PaCandidateScores,
    pa_method_registry: pd.DataFrame,
) -> _SelectedCharacterizationContext:
    """Prepare the candidate-invariant selected-result inputs once."""

    projection = _dated_projection(result_projection)
    baseline_scores, baseline_norms = _projection_baseline(
        result_parent,
        projection,
    )
    return _SelectedCharacterizationContext(
        physical_parent_identity=physical_parent.identity,
        result_parent_identity=result_parent.identity,
        projection=projection,
        baseline_scores=baseline_scores,
        baseline_norms=baseline_norms,
        pa_parent=pa_parent,
        pa_indices=_pa_method_indices(pa_method_registry, projection.method_registry),
    )


def _characterize_prepared_selected_candidate(
    physical: PhysicalEvaluation,
    *,
    activation_delta: pd.DataFrame,
    context: _SelectedCharacterizationContext,
    pa_scores: PaCandidateScores,
) -> ResultScores:
    """Apply one delta to the result-scoring context."""

    projection = context.projection
    scores, norms = _score_arrays(
        context.baseline_scores,
        context.baseline_norms,
        projection,
        activation_delta,
        pa_parent_scores=context.pa_parent.annual_scores[..., context.pa_indices],
        pa_parent_norms=context.pa_parent.annual_contribution_norms[..., context.pa_indices],
        pa_candidate_scores=pa_scores.annual_scores[..., context.pa_indices],
        pa_candidate_norms=pa_scores.annual_contribution_norms[..., context.pa_indices],
    )
    identity = _RESULT_IDENTITY_PREFIX + payload_sha256(
        {
            "physical_candidate_identity": physical.physical_candidate_identity,
            "activation_identity": physical.activation_identity,
            "parent_authority_identity": context.physical_parent_identity,
            "result_parent_scores_identity": context.result_parent_identity,
            "result_projection_identity": projection.projection_identity,
            "result_method_registry_identity": projection.method_registry_identity,
        }
    )
    scores = _readonly(scores)
    norms = _readonly(norms)
    scores_identity = _array_identity(scores)
    norms_identity = _array_identity(norms)
    score_payload = {
        "result_scoring_identity": identity,
        "physical_candidate_identity": physical.physical_candidate_identity,
        "result_projection_identity": projection.projection_identity,
        "result_method_registry_identity": projection.method_registry_identity,
        "result_parent_scores_identity": context.result_parent_identity,
        "activation_identity": physical.activation_identity,
        "backgrounds": list(projection.backgrounds),
        "regions": list(H12_REGIONS),
        "years": list(_YEARS),
        "method_registry_identity": projection.method_registry_identity,
        "annual_scores_identity": scores_identity,
        "annual_contribution_norms_identity": norms_identity,
    }
    return ResultScores(
        physical_candidate_identity=physical.physical_candidate_identity,
        result_scoring_identity=identity,
        backgrounds=projection.backgrounds,
        regions=H12_REGIONS,
        years=_YEARS,
        method_registry=projection.method_registry,
        annual_scores=scores,
        annual_contribution_norms=norms,
        score_identity=(_RESULT_SCORE_IDENTITY_PREFIX + payload_sha256(score_payload)),
    )


def _activation_identity(
    delta: pd.DataFrame,
    pa_spinning_events: tuple[PaSpinningEvent, ...],
    pa_recycling_events: tuple[PaRecyclingEvent, ...],
    pa_disposal_events: tuple[PaDisposalEvent, ...],
) -> str:
    """Bind provider differences and whole PA mixtures used by receiving calculations."""
    return _ACTIVATION_IDENTITY_PREFIX + payload_sha256(
        {
            "activation_delta": canonical_table_sha256(delta),
            "pa_spinning_events": [asdict(event) for event in pa_spinning_events],
            "pa_recycling_events": [asdict(event) for event in pa_recycling_events],
            "pa_disposal_events": [asdict(event) for event in pa_disposal_events],
        }
    )


class CandidateEvaluationSession:
    """Evaluate candidates once and reuse exact environmental descendants."""

    def __init__(
        self,
        *,
        mfa_session: CandidateMFASession,
        material_recipe: EcoinventMaterialRecipe,
        parent_scenario: ScenarioEvaluation,
        parent_mfa: MFAEvaluation,
        parent_authority: CandidateParentAuthority,
        optimization_parent_scores: CandidateParentScoreProjection,
        cumulative_climate_parent_scores: CandidateParentScoreProjection,
        provider_routes: ProviderRoutes,
        optimization_activations: OptimizationActivationContext,
        optimization_projection: ResponseBankProjection,
        cumulative_climate_projection: ResponseBankProjection,
        pa_projection: PaCandidateProjection,
        pa_parent: PaCandidateScores,
    ) -> None:
        """Bind the MFA model and response projections for one search."""
        self._scoring = prepare_search_scoring(
            physical_parent=parent_authority,
            optimization_parent=optimization_parent_scores,
            climate_parent=cumulative_climate_parent_scores,
            optimization_projection=optimization_projection,
            climate_projection=cumulative_climate_projection,
            pa_projection=pa_projection,
            pa_parent=pa_parent,
        )
        self._mfa_session = mfa_session
        self._material_recipe = material_recipe
        self._parent_scenario = parent_scenario
        self._parent = parent_authority
        self._routes = provider_routes
        self._parent_activations = parent_authority.activations
        self._optimization_activations = OptimizationActivationContext(
            identity=optimization_activations.identity,
            activations=parent_authority.activations,
            carbon_history=optimization_activations.carbon_history,
            pa_spinning_events=optimization_activations.pa_spinning_events,
            pa_recycling_events=optimization_activations.pa_recycling_events,
            pa_disposal_events=optimization_activations.pa_disposal_events,
        )
        self._parent_feedstock_requirements = _selected_feedstock_requirements(parent_mfa)
        self._cohort_weights: OrderedDict[str, CohortWeightsByYear] = OrderedDict()

    def _physical_request_identity(
        self,
        intervention: CandidateInterventionAuthority,
    ) -> str:
        """Identify one MFA calculation from its baseline and intervention inputs."""
        payload = {
            "method_identity": CANDIDATE_EVALUATION_METHOD_IDENTITY,
            "optimization_parent_overlay_method_identity": (
                OPTIMIZATION_PARENT_OVERLAY_METHOD_IDENTITY
            ),
            "candidate_mfa_method_identity": OPTIMIZATION_CANDIDATE_MFA_METHOD_IDENTITY,
            "candidate_mfa_session_identity": self._mfa_session.authority_identity,
            "parent_authority_identity": self._parent.identity,
            "parent_foreground_scenario_identity": self._parent.foreground_scenario_identity,
            "parent_mfa_result_identity": self._parent.mfa_result_identity,
            "intervention_authority_identity": intervention.identity,
            "demand_lifetime_method_identity": DEMAND_LIFETIME_METHOD_IDENTITY,
            "circularity_method_identity": CIRCULARITY_METHOD_IDENTITY,
            "activation_method_identity": OPTIMIZATION_ACTIVATION_METHOD_IDENTITY,
            "activation_context_identity": self._optimization_activations.identity,
            "provider_route_identity": self._routes.identity,
        }
        return _PHYSICAL_REQUEST_IDENTITY_PREFIX + payload_sha256(payload)

    def calculate_physical(
        self, intervention: CandidateInterventionAuthority
    ) -> AdmittedCandidateCalculation:
        """Evaluate candidate MFA or reuse the exact parent; propagate failures.

        A feedstock member with a proven admitted demand path uses this strict
        operation. Any unexpected MFA or numerical failure stops execution;
        the worker reports that failure directly.
        """
        if intervention.uses_exact_parent:
            return AdmittedCandidateCalculation(intervention, None)
        scenario = compile_parent_relative_scenario_assignments(
            self._parent_scenario,
            intervention.execution_foreground_scenario,
            _intervention_assignments(intervention),
        )
        evaluation = self._mfa_session.evaluate(
            foreground_scenario=scenario.foreground_scenario,
            compiled=scenario.compiled_parameters,
        )
        return AdmittedCandidateCalculation(intervention, evaluation)

    def admit_physical(
        self, intervention: CandidateInterventionAuthority
    ) -> AdmittedCandidateCalculation | InfeasibleCandidateCalculation:
        """Classify recognized failures during an ordinary candidate search."""
        try:
            return self.calculate_physical(intervention)
        except ScenarioFeasibilityError as exc:
            if exc.code not in _PHYSICAL_INFEASIBILITY_CODES:
                raise
            return self.reuse_physical_rejection(
                intervention, cast(OptimizationPhysicalInfeasibility, exc.diagnostic)
            )

    def reuse_physical_rejection(
        self,
        intervention: CandidateInterventionAuthority,
        diagnostic: OptimizationPhysicalInfeasibility,
    ) -> InfeasibleCandidateCalculation:
        """Bind an owned MFA rejection to the actual member's request identity.

        The worker may supply a completed family diagnostic only after proving
        equal non-feedstock execution. Environmental scores and MFA identities are
        recorded for admitted requests.
        """
        physical = PhysicalEvaluation(
            physical_request_identity=self._physical_request_identity(intervention),
            physical_candidate_identity="",
            physical_parent_identity=self._parent.identity,
            intervention_authority_identity=intervention.identity,
            status="physical_infeasible",
            diagnostic=diagnostic,
            physical_state_identity=None,
            activation_identity=None,
            activation_delta=None,
            pa_spinning_events=(),
            pa_recycling_events=(),
            pa_disposal_events=(),
        )
        return InfeasibleCandidateCalculation(
            replace(physical, physical_candidate_identity=physical_evaluation_identity(physical))
        )

    def complete_physical(
        self, admitted: AdmittedCandidateCalculation
    ) -> FeasibleCandidateCalculation:
        """Build activations and selected feedstock once from an already admitted MFA."""
        intervention = admitted.intervention
        evaluation = admitted.evaluation
        physical_request_identity = self._physical_request_identity(intervention)
        if evaluation is None:
            delta = optimization_activation_delta(
                self._parent_activations, self._parent_activations
            )
            pa_events = self._optimization_activations.pa_spinning_events
            activation_identity = _activation_identity(
                delta,
                pa_events,
                self._optimization_activations.pa_recycling_events,
                self._optimization_activations.pa_disposal_events,
            )
            result = PhysicalEvaluation(
                physical_request_identity=physical_request_identity,
                physical_candidate_identity="",
                physical_parent_identity=self._parent.identity,
                intervention_authority_identity=intervention.identity,
                status="physical_feasible",
                diagnostic=None,
                physical_state_identity=self._parent.physical_state_identity,
                activation_identity=activation_identity,
                activation_delta=delta,
                pa_spinning_events=pa_events,
                pa_recycling_events=self._optimization_activations.pa_recycling_events,
                pa_disposal_events=self._optimization_activations.pa_disposal_events,
            )
            result = replace(
                result, physical_candidate_identity=physical_evaluation_identity(result)
            )
            return FeasibleCandidateCalculation(
                physical=result,
                activation_delta=delta,
                feedstock_requirements=self._parent_feedstock_requirements,
            )
        cached_cohort_weights = self._cohort_weights.get(
            intervention.demand_lifetime_trajectory_identity
        )
        if cached_cohort_weights is None:
            cached_cohort_weights = prepare_carbon_cohort_weights(
                evaluation.cohort_state,
                evaluation.cohort_material_identity,
            )
            self._cohort_weights[intervention.demand_lifetime_trajectory_identity] = (
                cached_cohort_weights
            )
        active_cohort_weights = cached_cohort_weights
        self._cohort_weights.move_to_end(intervention.demand_lifetime_trajectory_identity)
        if len(self._cohort_weights) > _COHORT_WEIGHT_CACHE_SIZE:
            self._cohort_weights.popitem(last=False)
        compiled = compile_optimization_activations_from_history(
            evaluation,
            self._routes,
            self._material_recipe,
            self._optimization_activations,
            cohort_weights_by_year=active_cohort_weights,
        )
        delta = optimization_activation_delta(self._parent_activations, compiled.activations)
        feedstock_requirements = _selected_feedstock_requirements(evaluation)
        activation_identity = _activation_identity(
            delta,
            compiled.pa_spinning_events,
            compiled.pa_recycling_events,
            compiled.pa_disposal_events,
        )
        physical = PhysicalEvaluation(
            physical_request_identity=physical_request_identity,
            physical_candidate_identity="",
            physical_parent_identity=self._parent.identity,
            intervention_authority_identity=intervention.identity,
            status="physical_feasible",
            diagnostic=None,
            physical_state_identity=evaluation.physical_state_identity,
            activation_identity=activation_identity,
            activation_delta=delta,
            pa_spinning_events=compiled.pa_spinning_events,
            pa_recycling_events=compiled.pa_recycling_events,
            pa_disposal_events=compiled.pa_disposal_events,
        )
        physical = replace(
            physical, physical_candidate_identity=physical_evaluation_identity(physical)
        )

        return FeasibleCandidateCalculation(
            physical=physical,
            activation_delta=delta,
            feedstock_requirements=feedstock_requirements,
        )

    def score_optimization(
        self, calculation: FeasibleCandidateCalculation
    ) -> tuple[OptimizationScores, PaCandidateScores]:
        """Calculate actual receiving for EF search, including its climate category."""
        physical = calculation.physical
        if physical.physical_state_identity == self._parent.physical_state_identity:
            pa_scores = self._scoring.pa_parent
        else:
            pa_scores = calculate_pa_candidate_projection(
                self._scoring.pa_projection,
                spinning_events=physical.pa_spinning_events,
                recycling_events=physical.pa_recycling_events,
                disposal_events=physical.pa_disposal_events,
            )
        return (
            _optimization_scores(physical, self._scoring, calculation.activation_delta, pa_scores),
            pa_scores,
        )


def characterize_selected_physical_candidates(
    candidates: Sequence[tuple[str, PhysicalEvaluation, PaCandidateScores]],
    *,
    result_lcia_selection_identity: str,
    physical_parent: CandidateParentAuthority,
    result_parent: CandidateParentScoreProjection,
    result_projection: ResponseBankProjection,
    pa_parent: PaCandidateScores,
    pa_method_registry: pd.DataFrame,
) -> tuple[SelectedCandidateCharacterization, ...]:
    """Characterize the supplied selected states for reporting only."""

    outputs: list[SelectedCandidateCharacterization] = []
    context: _SelectedCharacterizationContext | None = None
    for selected_result_identity, physical, pa_scores in sorted(
        candidates,
        key=lambda item: item[1].physical_candidate_identity,
    ):
        activation_delta = cast(pd.DataFrame, physical.activation_delta)
        if context is None:
            context = _prepare_selected_characterization_context(
                physical_parent=physical_parent,
                result_parent=result_parent,
                result_projection=result_projection,
                pa_parent=pa_parent,
                pa_method_registry=pa_method_registry,
            )
        scores = _characterize_prepared_selected_candidate(
            physical,
            activation_delta=activation_delta,
            context=context,
            pa_scores=pa_scores,
        )
        identity = _SELECTED_CHARACTERIZATION_PREFIX + payload_sha256(
            {
                "selected_result_identity": selected_result_identity,
                "result_lcia_selection_identity": result_lcia_selection_identity,
                "result_scoring_identity": scores.result_scoring_identity,
                "result_score_identity": scores.score_identity,
            }
        )
        outputs.append(
            SelectedCandidateCharacterization(
                identity=identity,
                selected_result_identity=selected_result_identity,
                result_lcia_selection_identity=result_lcia_selection_identity,
                scores=scores,
            )
        )
    return tuple(outputs)
