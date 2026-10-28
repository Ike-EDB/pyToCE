"""Dated response projections and candidate LCIA calculations."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_optimization.calculation.candidate_authority import (
    _METHOD_SIGNATURE_COLUMNS,
    CandidateParentAuthority,
    CandidateParentMember,
    PhysicalEvaluation,
)
from pytoce.run_optimization.calculation.candidate_projection_arrays import (
    _BASELINE_COLUMNS,
    _YEARS,
    CandidateParentScoreProjection,
    FloatArray,
    _array_identity,
    _baseline_service_quantities,
    _dated_projection,
    _DatedProjection,
    _projection_baseline,
    _readonly,
    _score_arrays,
    _stable_service_response,
)
from pytoce.run_optimization.calculation.pa_candidate_projection import (
    PaCandidateProjection,
    PaCandidateScores,
)
from pytoce.run_optimization.calculation.regional_metrics import AnnualCategory
from pytoce.run_optimization.calculation.response_bank_contract import (
    ResponseBankProjection,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256, payload_sha256

_ENVIRONMENTAL_IDENTITY_PREFIX = "pytoce_optimization_environmental_scores_v5::"
_MEMBER_IDENTITY_PREFIX = "pytoce_optimization_member_scores_v4::"
_SCORE_IDENTITY_PREFIX = "pytoce_optimization_score_v1::"
_SEARCH_CONTEXT_IDENTITY_PREFIX = "pytoce_optimization_search_scoring_v6::"


@dataclass(frozen=True, slots=True)
class OptimizationScores:
    """Store the LCIA results used to select one candidate.

    Attributes:
        physical_candidate_identity: Content identifier of the candidate MFA result.
        environmental_identity: Content identifier calculated from its baseline and response
            inputs.
        member_identities: Result identity for each REMIND background scenario.
        backgrounds: Selected REMIND background scenarios in array order.
        regions: 12 world regions in array order.
        years: Calendar years 2019 through 2100 in array order.
        annual_categories: Selected EF impact categories in array order.
        optimization_method_registry_identity: Content identifier of the selected EF methods.
        cumulative_climate_method_registry_identity: Content identifier of the climate method.
        annual_scores: EF LCIA results with axes
            ``[background, region, year, impact category]`` in native LCIA units.
        annual_climate_scores: Climate-change LCIA results with axes
            ``[background, region, year]`` in kg CO2-eq.
        annual_scores_identity: Byte-level identity of ``annual_scores``.
        annual_climate_scores_identity: Byte-level identity of ``annual_climate_scores``.
        score_identity: Content identifier calculated from the coordinates, methods, and LCIA
            result arrays.
    """

    physical_candidate_identity: str
    environmental_identity: str
    member_identities: tuple[str, ...]
    backgrounds: tuple[str, ...]
    regions: tuple[str, ...]
    years: tuple[int, ...]
    annual_categories: tuple[AnnualCategory, ...]
    optimization_method_registry_identity: str
    cumulative_climate_method_registry_identity: str
    annual_scores: FloatArray
    annual_climate_scores: FloatArray
    annual_scores_identity: str
    annual_climate_scores_identity: str
    score_identity: str


@dataclass(frozen=True, slots=True)
class ResultScores:
    """Reporting-only characterization of one already selected MFA candidate."""

    physical_candidate_identity: str
    result_scoring_identity: str
    backgrounds: tuple[str, ...]
    regions: tuple[str, ...]
    years: tuple[int, ...]
    method_registry: pd.DataFrame
    annual_scores: FloatArray
    annual_contribution_norms: FloatArray
    score_identity: str


@dataclass(frozen=True, slots=True)
class SearchScoringContext:
    """Store baseline LCIA-result arrays and provider responses for a search."""

    identity: str
    physical_parent: CandidateParentAuthority
    optimization_parent: CandidateParentScoreProjection
    climate_parent: CandidateParentScoreProjection
    optimization_projection: "_DatedProjection"
    climate_projection: "_DatedProjection"
    optimization_baseline_scores: FloatArray
    climate_baseline_scores: FloatArray
    optimization_baseline_norms: FloatArray
    climate_baseline_norms: FloatArray
    pa_projection: PaCandidateProjection
    pa_parent: PaCandidateScores
    optimization_pa_indices: np.ndarray
    climate_pa_indices: np.ndarray


def _pa_method_indices(full_registry: pd.DataFrame, selected_registry: pd.DataFrame) -> np.ndarray:
    """Select PA array columns by the complete LCIA method signature."""
    columns = list(_METHOD_SIGNATURE_COLUMNS)
    full = pd.MultiIndex.from_frame(full_registry.loc[:, columns])
    selected = pd.MultiIndex.from_frame(selected_registry.loc[:, columns])
    return full.get_indexer(selected)


def build_response_parent_score_projection(
    physical_parent: CandidateParentAuthority,
    projection: ResponseBankProjection,
    *,
    pa_parent: PaCandidateScores,
    pa_method_registry: pd.DataFrame,
) -> CandidateParentScoreProjection:
    """Derive baseline LCIA results from activations and unit responses.

    Evaluate baseline provider activations once with the same dated response
    array used for candidate differences.
    """

    dated = _dated_projection(projection)
    amounts = _baseline_service_quantities(physical_parent, dated.services)
    indices = _pa_method_indices(pa_method_registry, dated.method_registry)
    scores = _readonly(
        _stable_service_response(amounts, dated.scores) + pa_parent.annual_scores[..., indices]
    )
    norms = _readonly(
        _stable_service_response(amounts, dated.norms)
        + pa_parent.annual_contribution_norms[..., indices]
    )
    methods = dated.method_registry.reset_index(drop=True)
    coordinates = pd.MultiIndex.from_product(
        (dated.backgrounds, H12_REGIONS, _YEARS),
        names=("background_scenario", "region", "year"),
    ).to_frame(index=False)
    coordinates["_join"] = 1
    method_coordinates = methods.loc[:, list(_METHOD_SIGNATURE_COLUMNS)].copy()
    method_coordinates["_join"] = 1
    regional = coordinates.merge(method_coordinates, on="_join", how="inner").drop(columns="_join")
    regional["score"] = scores.reshape(-1)
    regional["contribution_norm"] = norms.reshape(-1)
    annual = regional.loc[:, list(_BASELINE_COLUMNS)]
    annual = annual.sort_values(
        ["background_scenario", "region", "year", *_METHOD_SIGNATURE_COLUMNS],
        kind="stable",
        ignore_index=True,
    )
    for column in ("background_scenario", "region", *_METHOD_SIGNATURE_COLUMNS):
        annual[column] = annual[column].astype("string")
    annual["year"] = annual["year"].astype(np.int64)
    annual[["score", "contribution_norm"]] = annual[["score", "contribution_norm"]].astype(
        np.float64
    )
    member_records: list[CandidateParentMember] = []
    for background in dated.backgrounds:
        member_table = annual.loc[annual["background_scenario"].eq(background)].reset_index(
            drop=True
        )
        member_table_identity = canonical_table_sha256(member_table)
        member_records.append(
            CandidateParentMember(
                background_scenario=background,
                parent_score_identity=(
                    "pytoce_response_parent_member_v3::"
                    + payload_sha256(
                        {
                            "background_scenario": background,
                            "physical_parent_identity": physical_parent.identity,
                            "response_projection_identity": projection.projection_identity,
                            "provider_route_identity": projection.provider_route_identity,
                            "provider_service_registry_identity": (
                                projection.provider_service_registry_identity
                            ),
                            "timex_year_weights_identity": (projection.timex_year_weights_identity),
                        }
                    )
                ),
                projected_annual_scores_identity=member_table_identity,
            )
        )
    members = tuple(member_records)
    annual_identity = canonical_table_sha256(annual)
    registry_identity = canonical_table_sha256(methods)
    payload = {
        "role": projection.role,
        "physical_parent_identity": physical_parent.identity,
        "annual_scores_identity": annual_identity,
        "method_registry_identity": registry_identity,
        "member_parents": [
            {
                "background_scenario": parent.background_scenario,
                "parent_score_identity": parent.parent_score_identity,
                "projected_annual_scores_identity": parent.projected_annual_scores_identity,
            }
            for parent in members
        ],
    }
    return CandidateParentScoreProjection(
        role=projection.role,
        identity=(
            f"pytoce_optimization_parent_{projection.role}_scores_v3::" + payload_sha256(payload)
        ),
        physical_parent_identity=physical_parent.identity,
        annual_scores_identity=annual_identity,
        method_registry_identity=registry_identity,
        member_parents=members,
        annual_scores=annual,
        method_registry=methods,
    )


def prepare_search_scoring(
    *,
    physical_parent: CandidateParentAuthority,
    optimization_parent: CandidateParentScoreProjection,
    climate_parent: CandidateParentScoreProjection,
    optimization_projection: ResponseBankProjection,
    climate_projection: ResponseBankProjection,
    pa_projection: PaCandidateProjection,
    pa_parent: PaCandidateScores,
) -> SearchScoringContext:
    """Prepare scoring inputs and read-only LCIA arrays for the selected backgrounds."""

    optimization = _dated_projection(optimization_projection)
    climate = _dated_projection(climate_projection)
    optimization_baseline_scores, optimization_baseline_norms = _projection_baseline(
        optimization_parent,
        optimization,
    )
    climate_baseline_scores, climate_baseline_norms = _projection_baseline(
        climate_parent,
        climate,
    )
    optimization_baseline_scores = _readonly(optimization_baseline_scores)
    climate_baseline_scores = _readonly(climate_baseline_scores)
    optimization_baseline_norms = _readonly(optimization_baseline_norms)
    climate_baseline_norms = _readonly(climate_baseline_norms)
    optimization_scores_identity = _array_identity(optimization_baseline_scores)
    climate_scores_identity = _array_identity(climate_baseline_scores)
    payload = {
        "pa_response_projection_identity": pa_projection.response_projection_identity,
        "prepared_receiving_identity": pa_projection.prepared_receiving_identity,
        "physical_parent_identity": physical_parent.identity,
        "optimization_parent_identity": optimization_parent.identity,
        "climate_parent_identity": climate_parent.identity,
        "optimization_dated_projection_identity": optimization.identity,
        "climate_dated_projection_identity": climate.identity,
        "optimization_baseline_scores_identity": optimization_scores_identity,
        "climate_baseline_scores_identity": climate_scores_identity,
    }
    return SearchScoringContext(
        identity=_SEARCH_CONTEXT_IDENTITY_PREFIX + payload_sha256(payload),
        physical_parent=physical_parent,
        optimization_parent=optimization_parent,
        climate_parent=climate_parent,
        optimization_projection=optimization,
        climate_projection=climate,
        optimization_baseline_scores=optimization_baseline_scores,
        climate_baseline_scores=climate_baseline_scores,
        optimization_baseline_norms=optimization_baseline_norms,
        climate_baseline_norms=climate_baseline_norms,
        pa_projection=pa_projection,
        pa_parent=pa_parent,
        optimization_pa_indices=_pa_method_indices(
            pa_projection.method_registry, optimization.method_registry
        ),
        climate_pa_indices=_pa_method_indices(
            pa_projection.method_registry, climate.method_registry
        ),
    )


def _environmental_identity(
    physical: PhysicalEvaluation,
    context: SearchScoringContext,
) -> tuple[str, dict[str, object]]:
    """Identify LCIA results from their MFA result and method inputs."""
    payload: dict[str, object] = {
        "physical_candidate_identity": physical.physical_candidate_identity,
        "intervention_authority_identity": physical.intervention_authority_identity,
        "physical_parent_identity": physical.physical_parent_identity,
        "activation_identity": physical.activation_identity,
        "search_scoring_context_identity": context.identity,
        "parent_authority_identity": context.physical_parent.identity,
        "optimization_parent_scores_identity": context.optimization_parent.identity,
        "cumulative_climate_parent_scores_identity": context.climate_parent.identity,
        "optimization_projection_identity": context.optimization_projection.projection_identity,
        "cumulative_climate_projection_identity": context.climate_projection.projection_identity,
    }
    return _ENVIRONMENTAL_IDENTITY_PREFIX + payload_sha256(payload), payload


def _score_optimization_candidate(
    physical: PhysicalEvaluation,
    context: SearchScoringContext,
    activation_delta: pd.DataFrame,
    environmental_identity: str,
    pa_scores: PaCandidateScores,
) -> OptimizationScores:
    """Calculate candidate LCIA results from the baseline and response tables."""

    # Search and selected reporting use the same complete-parent replacement
    # calculation. Actual PA receiving also contributes when no fixed service
    # changes, with its search-method arrays selected by the admitted signatures.
    annual_scores, _ = _score_arrays(
        context.optimization_baseline_scores,
        context.optimization_baseline_norms,
        context.optimization_projection,
        activation_delta,
        pa_parent_scores=context.pa_parent.annual_scores[..., context.optimization_pa_indices],
        pa_parent_norms=(
            context.pa_parent.annual_contribution_norms[..., context.optimization_pa_indices]
        ),
        pa_candidate_scores=pa_scores.annual_scores[..., context.optimization_pa_indices],
        pa_candidate_norms=(
            pa_scores.annual_contribution_norms[..., context.optimization_pa_indices]
        ),
    )
    climate_scores, _ = _score_arrays(
        context.climate_baseline_scores,
        context.climate_baseline_norms,
        context.climate_projection,
        activation_delta,
        pa_parent_scores=context.pa_parent.annual_scores[..., context.climate_pa_indices],
        pa_parent_norms=(
            context.pa_parent.annual_contribution_norms[..., context.climate_pa_indices]
        ),
        pa_candidate_scores=pa_scores.annual_scores[..., context.climate_pa_indices],
        pa_candidate_norms=pa_scores.annual_contribution_norms[..., context.climate_pa_indices],
    )
    categories = tuple(
        AnnualCategory(
            optimization_lcia_method=str(row.lcia_method),
            impact_category_id=str(row.impact_category_id),
        )
        for row in context.optimization_projection.method_registry.itertuples(index=False)
    )
    annual_scores = _readonly(annual_scores)
    annual_climate_scores = _readonly(climate_scores[..., 0])
    scores = OptimizationScores(
        physical_candidate_identity=physical.physical_candidate_identity,
        environmental_identity=environmental_identity,
        member_identities=(),
        backgrounds=context.optimization_projection.backgrounds,
        regions=H12_REGIONS,
        years=_YEARS,
        annual_categories=categories,
        optimization_method_registry_identity=(
            context.optimization_projection.method_registry_identity
        ),
        cumulative_climate_method_registry_identity=(
            context.climate_projection.method_registry_identity
        ),
        annual_scores=annual_scores,
        annual_climate_scores=annual_climate_scores,
        annual_scores_identity="",
        annual_climate_scores_identity="",
        score_identity="",
    )
    return identify_optimization_scores(scores)


def identify_optimization_scores(scores: OptimizationScores) -> OptimizationScores:
    """Bind native score bytes, coordinates and methods through one identity calculation."""

    array_identities = {
        "annual_scores_identity": _array_identity(scores.annual_scores),
        "annual_climate_scores_identity": _array_identity(scores.annual_climate_scores),
    }
    score_payload = {
        "environmental_identity": scores.environmental_identity,
        "physical_candidate_identity": scores.physical_candidate_identity,
        "backgrounds": list(scores.backgrounds),
        "regions": list(scores.regions),
        "years": list(scores.years),
        "annual_categories": [
            {
                "optimization_lcia_method": item.optimization_lcia_method,
                "impact_category_id": item.impact_category_id,
            }
            for item in scores.annual_categories
        ],
        "optimization_method_registry_identity": (scores.optimization_method_registry_identity),
        "cumulative_climate_method_registry_identity": (
            scores.cumulative_climate_method_registry_identity
        ),
        **array_identities,
    }
    score_identity = _SCORE_IDENTITY_PREFIX + payload_sha256(score_payload)
    members = tuple(
        _MEMBER_IDENTITY_PREFIX
        + payload_sha256(
            {
                "physical_candidate_identity": scores.physical_candidate_identity,
                "score_identity": score_identity,
                "background_scenario": background,
            }
        )
        for background in scores.backgrounds
    )
    return replace(
        scores, **array_identities, score_identity=score_identity, member_identities=members
    )


def _optimization_scores(
    physical: PhysicalEvaluation,
    context: SearchScoringContext,
    activation_delta: pd.DataFrame,
    pa_scores: PaCandidateScores,
) -> OptimizationScores:
    """Calculate optimization LCIA results for one feasible MFA candidate."""

    environmental_identity, _payload = _environmental_identity(physical, context)
    return _score_optimization_candidate(
        physical,
        context,
        activation_delta,
        environmental_identity,
        pa_scores,
    )
