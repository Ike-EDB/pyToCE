"""Calculate optimization ASRs from LCIA results and allocated carrying capacities.

Annual impact categories and cumulative EF climate change are assessed
separately. Global ASRs use separate stable sums of numerators and denominators
across the 12 world regions.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Literal, TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.interventions import (
    InterventionDistanceInputs,
)
from pytoce.shared.assessment.asr import (
    GlobalMetrics,
    _finite_fsum,
    calculate_asr,
)
from pytoce.shared.runtime.filesystem import payload_sha256

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
Scalar = TypeVar("Scalar", bound=np.generic)
ConstraintType: TypeAlias = Literal["annual", "cumulative_gwp100"]
INTERVENTION_DISTANCE_METHOD_IDENTITY = "pytoce_optimization_intervention_distance_v7"
REGIONAL_METRIC_METHOD_IDENTITY = "pytoce_optimization_regional_metrics_v12"
_METRIC_IDENTITY_PREFIX = "pytoce_optimization_regional_metric_result_v12::"
_INTERVENTION_DISTANCE_IDENTITY_PREFIX = "pytoce_optimization_intervention_distance_v7::"
_CONTROL_START_YEAR = 2020
_CUMULATIVE_METHOD = "ef31"
_CUMULATIVE_CATEGORY = "climate_change"


@dataclass(frozen=True, slots=True)
class AnnualCategory:
    """Identify one annual LCIA impact category used by optimization."""

    optimization_lcia_method: str
    impact_category_id: str


@dataclass(frozen=True, slots=True)
class BindingConstraint:
    """One exact annual or cumulative regional ASR coordinate."""

    region: str
    constraint_type: ConstraintType
    optimization_lcia_method: str
    impact_category_id: str
    year: int | None
    asr: float


@dataclass(frozen=True, slots=True)
class InterventionDistance:
    """Mean intervention distance, as a fraction, for one search block.

    Demand and lifetime each contribute one value to their two-item mean.
    Collection, treatment shares, and feedstock shares contribute to their
    three-item mean. The combined block uses the mean of all five values.
    """

    identity: str
    physical_candidate_identity: str
    intervention_authority_identity: str
    distance_inputs_identity: str
    physical_parent_identity: str
    demand_lifetime_parent_identity: str
    demand_lifetime_trajectory_identity: str
    circularity_trajectory_identity: str
    optimization_block: OptimizationBlock
    target_year: int
    demand_reduction_fraction: float
    lifetime_extension_fraction: float
    collection_distance: float
    treatment_share_distance: float
    feedstock_share_distance: float
    demand_lifetime_distance: float
    circularity_distance: float
    total_distance: float


@dataclass(frozen=True, slots=True)
class RegionalMetrics:
    """Annual and cumulative metrics for one candidate and aCC case.

    Allocated carrying capacity (aCC) is the positive denominator of the
    absolute sustainability ratio (ASR). Annual LCIA-result, aCC, and
    ASR arrays have axes ``[region, year, impact category]``. Cumulative GWP100
    arrays have axis ``[region]`` after summing the annual climate LCIA result.

    Attributes:
        evaluation_identity: Content identifier calculated from the candidate,
            background scenario, aCC case, LCIA result, and aCC values.
        metric_identity: Content identifier calculated from all values in this record.
        background_scenario: REMIND member used to calculate the LCIA results.
        allocated_cc_case: Readable label for a supplied input. Its source
            documentation defines the allocation method and statistic.
        allocated_cc_context_identity: Content identifier calculated from the selected
            aCC cases, LCIA methods, coordinates, and values.
        physical_candidate_identity: Content identifier of the candidate MFA calculation.
        intervention_authority_identity: Content identifier calculated from the candidate
            trajectories and baseline MFA result.
        optimization_item_identity: Content identifier calculated from the optimization
            block, background scenario, aCC case, LCIA inputs, and metric definitions.
        optimization_block: Intervention family searched for this candidate.
        is_zero_intervention_parent: Whether every intervention value retains the
            exact baseline state.
        intervention_distance: Intervention changes and their combined distance from the
            baseline for this optimization block.
        optimization_lcia_method_registry_identity: Content identifier of the ordered
            annual LCIA methods used for optimization.
        cumulative_climate_method_registry_identity: Content identifier of the EF climate
            method used for the cumulative GWP100 constraint.
        comparison_scope_identity: Content identifier calculated from the baseline MFA
            result, aCC and LCIA inputs, regions, years, and target year.
        regions: The 12 world regions in array order, forming the first axis
            of the arrays of flows.
        years: Ordered calendar years 2019--2100 forming the annual array axis.
        annual_categories: Ordered annual LCIA method and impact category pairs
            forming the last annual array axis.
        annual_scores: Annual LCIA results with axes
            ``[region, year, impact category]`` and each impact category's native unit.
        annual_allocated_cc: Positive aCC denominators with the same axes and
            category-specific native units as ``annual_scores``.
        annual_asr: Ratio ``annual_scores / annual_allocated_cc``
            with the same axes.
        annual_peak_asr: Maximum annual ASR over 2020--2100 with axes
            ``[region, impact category]``.
        annual_budget_compatible_gwp100_scores: Annual EF climate-change GWP100
            LCIA results with axes ``[region, year]`` in the native EF climate
            unit, used only for the cumulative constraint.
        cumulative_gwp100_scores: Stable sum of each region's annual climate LCIA
            results over 2019--2100, with axis ``[region]``.
        cumulative_gwp100_allocated_cc: Positive regional aCC denominators for EF
            GWP100 over 2019--2100, with axis ``[region]`` and the same native
            climate unit as ``cumulative_gwp100_scores``.
        cumulative_gwp100_asr: Ratio of the regional cumulative
            LCIA result to its cumulative aCC, with axis ``[region]``.
        regional_worst_asr: Maximum of each region's annual 2020--2100 ASRs and
            cumulative climate ASR, with axis ``[region]``.
        regional_balanced_mean_positive_exceedance: Mean positive ASR exceedance
            for each region, weighting every annual target and the cumulative
            target equally after averaging annual exceedance over 2020--2100.
        annual_coordinate_compliant: Boolean mask ``annual_asr <= 1`` over the
            controlled axes ``[region, year 2020--2100, impact category]``.
        annual_target_compliant: Boolean mask with axes
            ``[region, impact category]`` indicating compliance in every controlled
            year.
        cumulative_target_compliant: Boolean mask with axis ``[region]`` for the
            cumulative GWP100 constraint.
        regional_compliant: Boolean mask with axis ``[region]`` requiring every
            annual target and the cumulative target to be compliant.
        all_region_joint_compliant: Whether every world region is compliant for
            every required target.
        fully_compliant_regions: World regions compliant for every annual target
            over 2020--2100 and for cumulative GWP100 over 2019--2100.
        fully_compliant_impact_targets: Annual method/impact category targets compliant
            in every region, plus the cumulative climate target when applicable.
        compliant_region_count: Number of fully compliant world regions.
        fully_compliant_impact_target_count: Number of impact targets compliant
            in every world region.
        compliant_region_impact_group_count: Number of region and impact-target
            groups that are fully compliant over their applicable period.
        compliant_coordinate_count: Number of compliant annual region-year-target
            coordinates plus compliant regional cumulative climate coordinates.
        region_count: Total number of world regions assessed.
        impact_target_count: Number of annual LCIA impact categories plus one cumulative
            climate target.
        region_impact_group_count: Product of ``region_count`` and
            ``impact_target_count``.
        coordinate_count: Total number of annual 2020--2100 coordinates plus one
            cumulative climate coordinate per region.
        z_worst: Largest exact regional ASR across the controlled annual and
            cumulative constraints.
        binding_slack: ``1 - z_worst`` when all regions are jointly compliant;
            otherwise ``None``.
        maximum_positive_exceedance: Largest positive exceedance,
            ``max(z_worst - 1, 0)``.
        balanced_mean_positive_exceedance: Mean of
            ``regional_balanced_mean_positive_exceedance`` across the 12 world regions.
        worst_regions: World regions whose ``regional_worst_asr`` equals ``z_worst``.
        binding_constraints: Exact controlled regional annual or cumulative
            coordinates whose ASR equals ``z_worst``.
        global_metrics: Global LCIA results and aCC values obtained by separate stable
            sums of the twelve regional numerators and denominators before each
            ASR is calculated.

    Notes:
        Annual and cumulative aCC denominators are positive.
        Annual compliance covers 2020--2100, while 2019 remains
        reported and contributes to cumulative GWP100 for 2019--2100. The metric
        identity changes with any coordinate, input or method identity, array
        value, compliance summary, binding constraint, or global metric.
    """

    evaluation_identity: str
    metric_identity: str
    background_scenario: str
    allocated_cc_case: str
    allocated_cc_context_identity: str
    physical_candidate_identity: str
    intervention_authority_identity: str
    optimization_item_identity: str
    optimization_block: OptimizationBlock
    is_zero_intervention_parent: bool
    intervention_distance: InterventionDistance
    optimization_lcia_method_registry_identity: str
    cumulative_climate_method_registry_identity: str
    comparison_scope_identity: str
    regions: tuple[str, ...]
    years: tuple[int, ...]
    annual_categories: tuple[AnnualCategory, ...]
    annual_scores: FloatArray
    annual_allocated_cc: FloatArray
    annual_asr: FloatArray
    annual_peak_asr: FloatArray
    annual_budget_compatible_gwp100_scores: FloatArray
    cumulative_gwp100_scores: FloatArray
    cumulative_gwp100_allocated_cc: FloatArray
    cumulative_gwp100_asr: FloatArray
    regional_worst_asr: FloatArray
    regional_balanced_mean_positive_exceedance: FloatArray
    annual_coordinate_compliant: BoolArray
    annual_target_compliant: BoolArray
    cumulative_target_compliant: BoolArray
    regional_compliant: BoolArray
    all_region_joint_compliant: bool
    fully_compliant_regions: tuple[str, ...]
    fully_compliant_impact_targets: tuple[str, ...]
    compliant_region_count: int
    fully_compliant_impact_target_count: int
    compliant_region_impact_group_count: int
    compliant_coordinate_count: int
    region_count: int
    impact_target_count: int
    region_impact_group_count: int
    coordinate_count: int
    z_worst: float
    binding_slack: float | None
    maximum_positive_exceedance: float
    balanced_mean_positive_exceedance: float
    worst_regions: tuple[str, ...]
    binding_constraints: tuple[BindingConstraint, ...]
    global_metrics: GlobalMetrics


def intervention_distance(
    *,
    physical_candidate_identity: str,
    intervention_authority_identity: str,
    distance_inputs: InterventionDistanceInputs,
    optimization_block: OptimizationBlock,
) -> InterventionDistance:
    """Calculate the mean intervention distance as a fraction.

    Demand and lifetime are averaged for ``demand_lifetime_frontier``;
    collection, treatment, and feedstock changes are averaged for
    ``circularity_feedstock_anchor``; ``integrated_frontier`` averages all five.
    Each component is a fraction between zero and one. Zero
    denotes unchanged endpoints in the active groups. Earlier annual values are
    recorded in the candidate trajectory. Demand and lifetime fractions are relative to
    their parent values, while collection, treatment and feedstock distances
    compare target-year shares. The allowed search range can be narrower; demand
    reduction stops at 0.9.
    """

    demand_components = (
        distance_inputs.demand_reduction_fraction,
        distance_inputs.lifetime_extension_fraction,
    )
    circularity_components = (
        distance_inputs.collection_distance,
        distance_inputs.treatment_share_distance,
        distance_inputs.feedstock_share_distance,
    )
    demand_lifetime = _finite_fsum(demand_components, name="demand_lifetime_distance") / 2.0
    circularity = _finite_fsum(circularity_components, name="circularity_distance") / 3.0
    total = (
        demand_lifetime
        if optimization_block == "demand_lifetime_frontier"
        else circularity
        if optimization_block == "circularity_feedstock_anchor"
        else _finite_fsum(
            (*demand_components, *circularity_components),
            name="integrated_distance",
        )
        / 5.0
    )
    payload = {
        "method_identity": INTERVENTION_DISTANCE_METHOD_IDENTITY,
        "physical_candidate_identity": physical_candidate_identity,
        "intervention_authority_identity": intervention_authority_identity,
        "distance_inputs_identity": distance_inputs.identity,
        "physical_parent_identity": distance_inputs.physical_parent_identity,
        "demand_lifetime_parent_identity": distance_inputs.demand_lifetime_parent_identity,
        "demand_lifetime_trajectory_identity": (
            distance_inputs.demand_lifetime_trajectory_identity
        ),
        "circularity_trajectory_identity": distance_inputs.circularity_trajectory_identity,
        "optimization_block": optimization_block,
        "distance_rule": "mean_active_intervention_distance",
        "target_year": distance_inputs.target_year,
        "demand_reduction_fraction": distance_inputs.demand_reduction_fraction,
        "lifetime_extension_fraction": distance_inputs.lifetime_extension_fraction,
        "collection_distance": distance_inputs.collection_distance,
        "treatment_share_distance": distance_inputs.treatment_share_distance,
        "feedstock_share_distance": distance_inputs.feedstock_share_distance,
        "demand_lifetime_distance": demand_lifetime,
        "circularity_distance": circularity,
        "total_distance": total,
    }
    return InterventionDistance(
        identity=_INTERVENTION_DISTANCE_IDENTITY_PREFIX + payload_sha256(payload),
        physical_candidate_identity=physical_candidate_identity,
        intervention_authority_identity=intervention_authority_identity,
        distance_inputs_identity=distance_inputs.identity,
        physical_parent_identity=distance_inputs.physical_parent_identity,
        demand_lifetime_parent_identity=distance_inputs.demand_lifetime_parent_identity,
        demand_lifetime_trajectory_identity=(distance_inputs.demand_lifetime_trajectory_identity),
        circularity_trajectory_identity=distance_inputs.circularity_trajectory_identity,
        optimization_block=optimization_block,
        target_year=distance_inputs.target_year,
        demand_reduction_fraction=distance_inputs.demand_reduction_fraction,
        lifetime_extension_fraction=distance_inputs.lifetime_extension_fraction,
        collection_distance=distance_inputs.collection_distance,
        treatment_share_distance=distance_inputs.treatment_share_distance,
        feedstock_share_distance=distance_inputs.feedstock_share_distance,
        demand_lifetime_distance=demand_lifetime,
        circularity_distance=circularity,
        total_distance=total,
    )


def calculate_regional_metrics(
    *,
    evaluation_identity: str,
    background_scenario: str,
    allocated_cc_case: str,
    allocated_cc_context_identity: str,
    physical_candidate_identity: str,
    intervention_authority_identity: str,
    optimization_item_identity: str,
    optimization_block: OptimizationBlock,
    is_zero_intervention_parent: bool,
    intervention_distance: InterventionDistance,
    optimization_lcia_method_registry_identity: str,
    cumulative_climate_method_registry_identity: str,
    comparison_scope_identity: str,
    annual_scores: FloatArray,
    annual_allocated_cc: FloatArray,
    annual_budget_compatible_gwp100_scores: FloatArray,
    cumulative_gwp100_allocated_cc: FloatArray,
    regions: Sequence[str],
    years: Sequence[int],
    annual_categories: Sequence[AnnualCategory],
) -> RegionalMetrics:
    """Calculate annual and independent cumulative aCC metrics.

    Arrays use ``[region, year, annual EF category]`` order. The separate climate
    LCIA result uses ``[region, year]`` order and supplies the standard EF all-GHG
    GWP100 numerator for the cumulative constraint. Annual ASR equals the regional
    LCIA result divided by the positive allocated carrying capacity for the same
    region, year, and category. Cumulative climate ASR uses the cumulative GWP100
    LCIA result and its matching cumulative allocated carrying capacity.

    Global numerators and allocated carrying capacities are each summed once
    across all 12 world regions with stable summation before their ratio is
    calculated. Optimization metrics use the 16 EF v3.1 categories and cumulative GWP100.

    Keep the immutable score and aCC input slices supplied by their producers.
    Only newly calculated metric arrays need their own immutable storage.
    """

    identities = {
        "evaluation_identity": evaluation_identity,
        "background_scenario": background_scenario,
        "allocated_cc_case": allocated_cc_case,
        "allocated_cc_context_identity": allocated_cc_context_identity,
        "physical_candidate_identity": physical_candidate_identity,
        "intervention_authority_identity": intervention_authority_identity,
        "optimization_item_identity": optimization_item_identity,
        "optimization_lcia_method_registry_identity": (optimization_lcia_method_registry_identity),
        "cumulative_climate_method_registry_identity": (
            cumulative_climate_method_registry_identity
        ),
        "comparison_scope_identity": comparison_scope_identity,
    }
    region_axis = tuple(regions)
    year_axis = tuple(years)
    categories = tuple(annual_categories)
    scores = annual_scores
    annual_acc = annual_allocated_cc
    climate_scores = annual_budget_compatible_gwp100_scores
    cumulative_acc = cumulative_gwp100_allocated_cc

    control_start = year_axis.index(_CONTROL_START_YEAR)
    assessment = calculate_asr(
        annual_scores=scores,
        annual_allocated_cc=annual_acc,
        annual_budget_compatible_gwp100_scores=climate_scores,
        cumulative_gwp100_allocated_cc=cumulative_acc,
        control_start=control_start,
    )
    annual_asr = assessment.annual_asr
    annual_peak = np.max(annual_asr[:, control_start:, :], axis=1)
    cumulative_scores = assessment.cumulative_gwp100_scores
    cumulative_asr = assessment.cumulative_gwp100_asr
    controlled_annual_asr = annual_asr[:, control_start:, :]
    annual_coordinate_compliant = controlled_annual_asr <= 1.0
    annual_compliant = np.all(annual_coordinate_compliant, axis=1)
    cumulative_compliant = cumulative_asr <= 1.0
    regional_compliant = np.all(annual_compliant, axis=1) & cumulative_compliant
    regional_worst = np.maximum(np.max(annual_peak, axis=1), cumulative_asr)
    regional_exceedance = np.fromiter(
        (
            (
                _finite_fsum(
                    (
                        *(
                            _finite_fsum(
                                (
                                    max(0.0, float(value) - 1.0)
                                    for value in controlled_annual_asr[
                                        region_index, :, category_index
                                    ]
                                ),
                                name="annual_category_mean_positive_exceedance",
                            )
                            / controlled_annual_asr.shape[1]
                            for category_index in range(len(categories))
                        ),
                        max(0.0, float(cumulative_asr[region_index]) - 1.0),
                    ),
                    name="regional_balanced_mean_positive_exceedance",
                )
            )
            / (len(categories) + 1)
            for region_index in range(len(region_axis))
        ),
        dtype=np.float64,
        count=len(region_axis),
    )

    annual_target_identities = tuple(
        f"annual::{category.optimization_lcia_method}::{category.impact_category_id}"
        for category in categories
    )
    cumulative_target_identity = f"cumulative_gwp100::{_CUMULATIVE_METHOD}::{_CUMULATIVE_CATEGORY}"
    fully_compliant_regions = tuple(
        region_axis[index] for index in np.flatnonzero(regional_compliant).tolist()
    )
    fully_compliant_impact_targets = tuple(
        identity
        for identity, compliant in zip(
            (*annual_target_identities, cumulative_target_identity),
            (*np.all(annual_compliant, axis=0).tolist(), bool(np.all(cumulative_compliant))),
            strict=True,
        )
        if compliant
    )
    compliant_region_count = int(np.count_nonzero(regional_compliant))
    fully_compliant_impact_target_count = len(fully_compliant_impact_targets)
    compliant_region_impact_group_count = int(
        np.count_nonzero(annual_compliant) + np.count_nonzero(cumulative_compliant)
    )
    compliant_coordinate_count = int(
        np.count_nonzero(annual_coordinate_compliant) + np.count_nonzero(cumulative_compliant)
    )
    region_count = len(region_axis)
    impact_target_count = len(categories) + 1
    region_impact_group_count = region_count * impact_target_count
    coordinate_count = region_count * (controlled_annual_asr.shape[1] * len(categories) + 1)
    all_region_joint_compliant = bool(np.all(regional_compliant))
    z_worst = float(np.max(regional_worst))
    binding_slack = 1.0 - z_worst if all_region_joint_compliant else None
    maximum_positive_exceedance = max(z_worst - 1.0, 0.0)
    balanced_mean_exceedance = _finite_fsum(
        (float(value) for value in regional_exceedance),
        name="balanced_mean_positive_exceedance",
    ) / len(region_axis)
    worst_regions = tuple(
        region_axis[index] for index in np.flatnonzero(regional_worst == z_worst).tolist()
    )
    bindings = _binding_constraints(
        regions=region_axis,
        years=year_axis,
        categories=categories,
        annual_asr=annual_asr,
        cumulative_asr=cumulative_asr,
        worst_asr=z_worst,
        control_start=control_start,
    )
    global_metrics = assessment.global_metrics
    result = RegionalMetrics(
        evaluation_identity=identities["evaluation_identity"],
        metric_identity="",
        background_scenario=identities["background_scenario"],
        allocated_cc_case=identities["allocated_cc_case"],
        allocated_cc_context_identity=identities["allocated_cc_context_identity"],
        physical_candidate_identity=identities["physical_candidate_identity"],
        intervention_authority_identity=identities["intervention_authority_identity"],
        optimization_item_identity=identities["optimization_item_identity"],
        optimization_block=optimization_block,
        is_zero_intervention_parent=is_zero_intervention_parent,
        intervention_distance=intervention_distance,
        optimization_lcia_method_registry_identity=(
            identities["optimization_lcia_method_registry_identity"]
        ),
        cumulative_climate_method_registry_identity=(
            identities["cumulative_climate_method_registry_identity"]
        ),
        comparison_scope_identity=identities["comparison_scope_identity"],
        regions=region_axis,
        years=year_axis,
        annual_categories=categories,
        annual_scores=scores,
        annual_allocated_cc=annual_acc,
        annual_asr=_readonly(annual_asr),
        annual_peak_asr=_readonly(annual_peak),
        annual_budget_compatible_gwp100_scores=climate_scores,
        cumulative_gwp100_scores=_readonly(cumulative_scores),
        cumulative_gwp100_allocated_cc=cumulative_acc,
        cumulative_gwp100_asr=_readonly(cumulative_asr),
        regional_worst_asr=_readonly(regional_worst),
        regional_balanced_mean_positive_exceedance=_readonly(regional_exceedance),
        annual_coordinate_compliant=_readonly(annual_coordinate_compliant),
        annual_target_compliant=_readonly(annual_compliant),
        cumulative_target_compliant=_readonly(cumulative_compliant),
        regional_compliant=_readonly(regional_compliant),
        all_region_joint_compliant=all_region_joint_compliant,
        fully_compliant_regions=fully_compliant_regions,
        fully_compliant_impact_targets=fully_compliant_impact_targets,
        compliant_region_count=compliant_region_count,
        fully_compliant_impact_target_count=fully_compliant_impact_target_count,
        compliant_region_impact_group_count=compliant_region_impact_group_count,
        compliant_coordinate_count=compliant_coordinate_count,
        region_count=region_count,
        impact_target_count=impact_target_count,
        region_impact_group_count=region_impact_group_count,
        coordinate_count=coordinate_count,
        z_worst=z_worst,
        binding_slack=binding_slack,
        maximum_positive_exceedance=maximum_positive_exceedance,
        balanced_mean_positive_exceedance=balanced_mean_exceedance,
        worst_regions=worst_regions,
        binding_constraints=bindings,
        global_metrics=global_metrics,
    )
    return replace(result, metric_identity=_metric_identity(result))


def _metric_identity(metrics: RegionalMetrics) -> str:
    """Hash the complete coordinates, arrays, compliance values, and bindings."""
    arrays = {
        field: _array_identity(getattr(metrics, field))
        for field in (
            "annual_scores",
            "annual_allocated_cc",
            "annual_asr",
            "annual_peak_asr",
            "annual_budget_compatible_gwp100_scores",
            "cumulative_gwp100_scores",
            "cumulative_gwp100_allocated_cc",
            "cumulative_gwp100_asr",
            "regional_worst_asr",
            "regional_balanced_mean_positive_exceedance",
            "annual_coordinate_compliant",
            "annual_target_compliant",
            "cumulative_target_compliant",
            "regional_compliant",
        )
    }
    global_arrays = {
        field: _array_identity(getattr(metrics.global_metrics, field))
        for field in (
            "annual_scores",
            "annual_allocated_cc",
            "annual_asr",
            "annual_peak_asr",
            "annual_budget_compatible_gwp100_scores",
        )
    }
    payload = {
        "method_identity": REGIONAL_METRIC_METHOD_IDENTITY,
        "evaluation_identity": metrics.evaluation_identity,
        "background_scenario": metrics.background_scenario,
        "allocated_cc_case": metrics.allocated_cc_case,
        "allocated_cc_context_identity": metrics.allocated_cc_context_identity,
        "physical_candidate_identity": metrics.physical_candidate_identity,
        "intervention_authority_identity": metrics.intervention_authority_identity,
        "optimization_item_identity": metrics.optimization_item_identity,
        "optimization_block": metrics.optimization_block,
        "is_zero_intervention_parent": metrics.is_zero_intervention_parent,
        "intervention_distance": {
            "identity": metrics.intervention_distance.identity,
            "physical_candidate_identity": (
                metrics.intervention_distance.physical_candidate_identity
            ),
            "intervention_authority_identity": (
                metrics.intervention_distance.intervention_authority_identity
            ),
            "distance_inputs_identity": metrics.intervention_distance.distance_inputs_identity,
            "physical_parent_identity": metrics.intervention_distance.physical_parent_identity,
            "demand_lifetime_parent_identity": (
                metrics.intervention_distance.demand_lifetime_parent_identity
            ),
            "demand_lifetime_trajectory_identity": (
                metrics.intervention_distance.demand_lifetime_trajectory_identity
            ),
            "circularity_trajectory_identity": (
                metrics.intervention_distance.circularity_trajectory_identity
            ),
            "optimization_block": metrics.intervention_distance.optimization_block,
            "target_year": metrics.intervention_distance.target_year,
            "demand_reduction_fraction": (metrics.intervention_distance.demand_reduction_fraction),
            "lifetime_extension_fraction": (
                metrics.intervention_distance.lifetime_extension_fraction
            ),
            "collection_distance": metrics.intervention_distance.collection_distance,
            "treatment_share_distance": (metrics.intervention_distance.treatment_share_distance),
            "feedstock_share_distance": (metrics.intervention_distance.feedstock_share_distance),
            "demand_lifetime_distance": (metrics.intervention_distance.demand_lifetime_distance),
            "circularity_distance": metrics.intervention_distance.circularity_distance,
            "total_distance": metrics.intervention_distance.total_distance,
        },
        "optimization_lcia_method_registry_identity": (
            metrics.optimization_lcia_method_registry_identity
        ),
        "cumulative_climate_method_registry_identity": (
            metrics.cumulative_climate_method_registry_identity
        ),
        "comparison_scope_identity": metrics.comparison_scope_identity,
        "regions": list(metrics.regions),
        "years": list(metrics.years),
        "annual_categories": [
            [item.optimization_lcia_method, item.impact_category_id]
            for item in metrics.annual_categories
        ],
        "arrays": arrays,
        "all_region_joint_compliant": metrics.all_region_joint_compliant,
        "fully_compliant_regions": list(metrics.fully_compliant_regions),
        "fully_compliant_impact_targets": list(metrics.fully_compliant_impact_targets),
        "counts": {
            "compliant_region_count": metrics.compliant_region_count,
            "fully_compliant_impact_target_count": (metrics.fully_compliant_impact_target_count),
            "compliant_region_impact_group_count": (metrics.compliant_region_impact_group_count),
            "compliant_coordinate_count": metrics.compliant_coordinate_count,
            "region_count": metrics.region_count,
            "impact_target_count": metrics.impact_target_count,
            "region_impact_group_count": metrics.region_impact_group_count,
            "coordinate_count": metrics.coordinate_count,
        },
        "z_worst": metrics.z_worst,
        "binding_slack": metrics.binding_slack,
        "maximum_positive_exceedance": metrics.maximum_positive_exceedance,
        "balanced_mean_positive_exceedance": (metrics.balanced_mean_positive_exceedance),
        "worst_regions": list(metrics.worst_regions),
        "binding_constraints": [
            [
                item.region,
                item.constraint_type,
                item.optimization_lcia_method,
                item.impact_category_id,
                item.year,
                item.asr,
            ]
            for item in metrics.binding_constraints
        ],
        "global_arrays": global_arrays,
        "global_scalars": {
            "cumulative_gwp100_score": metrics.global_metrics.cumulative_gwp100_score,
            "cumulative_gwp100_allocated_cc": (
                metrics.global_metrics.cumulative_gwp100_allocated_cc
            ),
            "cumulative_gwp100_asr": metrics.global_metrics.cumulative_gwp100_asr,
            "worst_asr": metrics.global_metrics.worst_asr,
            "mean_positive_exceedance": metrics.global_metrics.mean_positive_exceedance,
            "compliant": metrics.global_metrics.compliant,
        },
    }
    return _METRIC_IDENTITY_PREFIX + payload_sha256(payload)


def _array_identity(values: NDArray[np.generic]) -> dict[str, object]:
    """Describe an array by shape, dtype, and the hash of its contiguous bytes."""
    contiguous = np.ascontiguousarray(values)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "bytes_sha256": sha256(contiguous.tobytes()).hexdigest(),
    }


def _readonly(values: NDArray[Scalar]) -> NDArray[Scalar]:
    """Copy an array into an immutable bytes-backed view."""
    contiguous = np.ascontiguousarray(values)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
    return cast(NDArray[Scalar], result)


def _binding_constraints(
    *,
    regions: tuple[str, ...],
    years: tuple[int, ...],
    categories: tuple[AnnualCategory, ...],
    annual_asr: FloatArray,
    cumulative_asr: FloatArray,
    worst_asr: float,
    control_start: int,
) -> tuple[BindingConstraint, ...]:
    """List annual and cumulative regional coordinates equal to the worst ASR."""
    bindings: list[BindingConstraint] = []
    for region_index, region in enumerate(regions):
        for category_index, category in enumerate(categories):
            for year_index in range(control_start, len(years)):
                asr = float(annual_asr[region_index, year_index, category_index])
                if asr == worst_asr:
                    bindings.append(
                        BindingConstraint(
                            region=region,
                            constraint_type="annual",
                            optimization_lcia_method=(category.optimization_lcia_method),
                            impact_category_id=category.impact_category_id,
                            year=years[year_index],
                            asr=asr,
                        )
                    )
        if float(cumulative_asr[region_index]) == worst_asr:
            bindings.append(
                BindingConstraint(
                    region=region,
                    constraint_type="cumulative_gwp100",
                    optimization_lcia_method=_CUMULATIVE_METHOD,
                    impact_category_id=_CUMULATIVE_CATEGORY,
                    year=None,
                    asr=worst_asr,
                )
            )
    return tuple(bindings)


def transgressing_constraints(metrics: RegionalMetrics) -> tuple[BindingConstraint, ...]:
    """Return every controlled regional coordinate whose exact ASR exceeds one."""

    control_start = 1
    transgressions: list[BindingConstraint] = []
    for region_index, region in enumerate(metrics.regions):
        for category_index, category in enumerate(metrics.annual_categories):
            for year_index in range(control_start, len(metrics.years)):
                asr = float(metrics.annual_asr[region_index, year_index, category_index])
                if asr > 1.0:
                    transgressions.append(
                        BindingConstraint(
                            region=region,
                            constraint_type="annual",
                            optimization_lcia_method=category.optimization_lcia_method,
                            impact_category_id=category.impact_category_id,
                            year=metrics.years[year_index],
                            asr=asr,
                        )
                    )
        cumulative_asr = float(metrics.cumulative_gwp100_asr[region_index])
        if cumulative_asr > 1.0:
            transgressions.append(
                BindingConstraint(
                    region=region,
                    constraint_type="cumulative_gwp100",
                    optimization_lcia_method=_CUMULATIVE_METHOD,
                    impact_category_id=_CUMULATIVE_CATEGORY,
                    year=None,
                    asr=cumulative_asr,
                )
            )
    return tuple(transgressions)
