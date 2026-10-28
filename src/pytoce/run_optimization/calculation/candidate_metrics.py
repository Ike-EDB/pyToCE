"""Compact regional metric records and their identities."""

from dataclasses import dataclass, replace

from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.shared.runtime.filesystem import payload_sha256

from .regional_metrics import (
    BindingConstraint,
    RegionalMetrics,
)

_METRIC_ENTRY_IDENTITY_PREFIX = "pytoce_optimization_metric_entry_v11::"


@dataclass(frozen=True, slots=True)
class CandidateMetricSummary:
    """Values used to compare one candidate during optimization.

    The record keeps intervention fractions, the selected background and
    allocated carrying capacity (aCC) case, counts of compliant regional coordinates,
    the worst absolute sustainability ratio (ASR), and the constraints that
    determine candidate ranking. Large annual arrays remain in the calculated
    regional metric.

    Attributes:
        identity: Content identifier calculated from every other field in this record.
        physical_candidate_identity: Content identifier of the candidate MFA calculation.
        intervention_authority_identity: Content identifier calculated from the
            candidate trajectories and baseline MFA result.
        physical_parent_identity: Content identifier combining the baseline MFA
            inputs and provider activations.
        distance_inputs_identity: Content identifier calculated from the five
            intervention changes used for ranking.
        demand_lifetime_parent_identity: Content identifier of the baseline demand,
            lifetime, and stock-target inputs.
        demand_lifetime_trajectory_identity: Content identifier of the candidate's annual
            demand and lifetime values.
        circularity_trajectory_identity: Content identifier of the candidate's annual
            collection, treatment, and feedstock shares.
        target_year: Calendar year in which the requested intervention values
            are reached for selection.
        demand_reduction_fraction: Demand reduction from baseline as a fraction.
        lifetime_extension_fraction: Product-lifetime extension from baseline as
            a fraction.
        collection_distance: Mean absolute change in regional collection share
            relative to the parent in the target year, as a fraction.
        treatment_share_distance: Mean total-variation distance between baseline
            and candidate treatment shares, expressed as a fraction.
        feedstock_share_distance: Mean total-variation distance between baseline
            and candidate virgin-feedstock shares, expressed as a fraction.
        evaluation_identity: Content identifier calculated from the candidate,
            background scenario, aCC case, LCIA result, and aCC values.
        metric_identity: Content identifier calculated from the complete regional metric.
        background_scenario: REMIND member used for the candidate's LCIA results.
        allocated_cc_case: Readable label for a supplied input. Its source
            documentation defines the allocation method and statistic.
        allocated_cc_context_identity: Content identifier calculated from the selected
            aCC cases, methods, coordinates, and values.
        optimization_item_identity: Content identifier calculated from the optimization
            block, background scenario, aCC case, LCIA inputs, and metric definitions.
        comparison_scope_identity: Content identifier calculated from the baseline MFA
            result, aCC and LCIA inputs, regions, years, and target year.
        optimization_block: Intervention family searched for this candidate.
        is_zero_intervention_parent: Whether every intervention value retains the
            exact baseline state.
        intervention_distance_identity: Content identifier calculated from the
            intervention changes and comparison rule for the active optimization block.
        intervention_distance: Equal-family mean distance from the
            baseline across the intervention groups active in the search block.
        regions: The 12 world regions in compliance-evaluation order.
        all_region_joint_compliant: Whether every controlled annual coordinate
            and every regional cumulative climate target has ``ASR <= 1``.
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
        region_impact_group_count: Total number of region and impact-target groups.
        coordinate_count: Total number of controlled annual coordinates plus
            regional cumulative climate coordinates.
        z_worst: Largest exact regional ASR across annual 2020--2100 coordinates
            and cumulative 2019--2100 climate constraints.
        binding_slack: ``1 - z_worst`` when the candidate is jointly compliant;
            otherwise ``None``.
        maximum_positive_exceedance: Largest positive exceedance,
            ``max(z_worst - 1, 0)``.
        balanced_mean_positive_exceedance: Mean positive ASR exceedance balanced
            equally across regions and impact targets, with annual exceedance
            first averaged over 2020--2100.
        binding_constraints: Exact regional annual or cumulative coordinates
            whose ASR equals ``z_worst``.

    Notes:
        Every aCC denominator is positive. The summary identity changes when any
        listed scientific coordinate, provenance identity, ranking value, or
        binding constraint changes.
    """

    identity: str
    physical_candidate_identity: str
    intervention_authority_identity: str
    physical_parent_identity: str
    distance_inputs_identity: str
    demand_lifetime_parent_identity: str
    demand_lifetime_trajectory_identity: str
    circularity_trajectory_identity: str
    target_year: int
    demand_reduction_fraction: float
    lifetime_extension_fraction: float
    collection_distance: float
    treatment_share_distance: float
    feedstock_share_distance: float
    evaluation_identity: str
    metric_identity: str
    background_scenario: str
    allocated_cc_case: str
    allocated_cc_context_identity: str
    optimization_item_identity: str
    comparison_scope_identity: str
    optimization_block: OptimizationBlock
    is_zero_intervention_parent: bool
    intervention_distance_identity: str
    intervention_distance: float
    regions: tuple[str, ...]
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
    binding_constraints: tuple[BindingConstraint, ...]


def candidate_metric_summary(metrics: RegionalMetrics) -> CandidateMetricSummary:
    """Project one regional metric into the compact search representation."""

    entry = CandidateMetricSummary(
        identity="",
        physical_candidate_identity=metrics.physical_candidate_identity,
        intervention_authority_identity=metrics.intervention_authority_identity,
        physical_parent_identity=metrics.intervention_distance.physical_parent_identity,
        distance_inputs_identity=metrics.intervention_distance.distance_inputs_identity,
        demand_lifetime_parent_identity=(
            metrics.intervention_distance.demand_lifetime_parent_identity
        ),
        demand_lifetime_trajectory_identity=(
            metrics.intervention_distance.demand_lifetime_trajectory_identity
        ),
        circularity_trajectory_identity=(
            metrics.intervention_distance.circularity_trajectory_identity
        ),
        target_year=metrics.intervention_distance.target_year,
        demand_reduction_fraction=metrics.intervention_distance.demand_reduction_fraction,
        lifetime_extension_fraction=metrics.intervention_distance.lifetime_extension_fraction,
        collection_distance=metrics.intervention_distance.collection_distance,
        treatment_share_distance=metrics.intervention_distance.treatment_share_distance,
        feedstock_share_distance=metrics.intervention_distance.feedstock_share_distance,
        evaluation_identity=metrics.evaluation_identity,
        metric_identity=metrics.metric_identity,
        background_scenario=metrics.background_scenario,
        allocated_cc_case=metrics.allocated_cc_case,
        allocated_cc_context_identity=metrics.allocated_cc_context_identity,
        optimization_item_identity=metrics.optimization_item_identity,
        comparison_scope_identity=metrics.comparison_scope_identity,
        optimization_block=metrics.optimization_block,
        is_zero_intervention_parent=metrics.is_zero_intervention_parent,
        intervention_distance_identity=metrics.intervention_distance.identity,
        intervention_distance=metrics.intervention_distance.total_distance,
        regions=metrics.regions,
        all_region_joint_compliant=metrics.all_region_joint_compliant,
        fully_compliant_regions=metrics.fully_compliant_regions,
        fully_compliant_impact_targets=metrics.fully_compliant_impact_targets,
        compliant_region_count=metrics.compliant_region_count,
        fully_compliant_impact_target_count=(metrics.fully_compliant_impact_target_count),
        compliant_region_impact_group_count=(metrics.compliant_region_impact_group_count),
        compliant_coordinate_count=metrics.compliant_coordinate_count,
        region_count=metrics.region_count,
        impact_target_count=metrics.impact_target_count,
        region_impact_group_count=metrics.region_impact_group_count,
        coordinate_count=metrics.coordinate_count,
        z_worst=metrics.z_worst,
        binding_slack=metrics.binding_slack,
        maximum_positive_exceedance=metrics.maximum_positive_exceedance,
        balanced_mean_positive_exceedance=(metrics.balanced_mean_positive_exceedance),
        binding_constraints=metrics.binding_constraints,
    )
    return replace(entry, identity=_candidate_metric_summary_identity(entry))


def _candidate_metric_summary_identity(entry: CandidateMetricSummary) -> str:
    """Identify one stored metric row from its candidate and metric fields."""
    payload = {
        "physical_candidate_identity": entry.physical_candidate_identity,
        "intervention_authority_identity": entry.intervention_authority_identity,
        "physical_parent_identity": entry.physical_parent_identity,
        "distance_inputs_identity": entry.distance_inputs_identity,
        "demand_lifetime_parent_identity": entry.demand_lifetime_parent_identity,
        "demand_lifetime_trajectory_identity": entry.demand_lifetime_trajectory_identity,
        "circularity_trajectory_identity": entry.circularity_trajectory_identity,
        "target_year": entry.target_year,
        "demand_reduction_fraction": entry.demand_reduction_fraction,
        "lifetime_extension_fraction": entry.lifetime_extension_fraction,
        "collection_distance": entry.collection_distance,
        "treatment_share_distance": entry.treatment_share_distance,
        "feedstock_share_distance": entry.feedstock_share_distance,
        "evaluation_identity": entry.evaluation_identity,
        "metric_identity": entry.metric_identity,
        "background_scenario": entry.background_scenario,
        "allocated_cc_case": entry.allocated_cc_case,
        "allocated_cc_context_identity": entry.allocated_cc_context_identity,
        "optimization_item_identity": entry.optimization_item_identity,
        "comparison_scope_identity": entry.comparison_scope_identity,
        "optimization_block": entry.optimization_block,
        "is_zero_intervention_parent": entry.is_zero_intervention_parent,
        "intervention_distance_identity": entry.intervention_distance_identity,
        "intervention_distance": entry.intervention_distance,
        "regions": list(entry.regions),
        "all_region_joint_compliant": entry.all_region_joint_compliant,
        "fully_compliant_regions": list(entry.fully_compliant_regions),
        "fully_compliant_impact_targets": list(entry.fully_compliant_impact_targets),
        "counts": {
            "compliant_region_count": entry.compliant_region_count,
            "fully_compliant_impact_target_count": (entry.fully_compliant_impact_target_count),
            "compliant_region_impact_group_count": (entry.compliant_region_impact_group_count),
            "compliant_coordinate_count": entry.compliant_coordinate_count,
            "region_count": entry.region_count,
            "impact_target_count": entry.impact_target_count,
            "region_impact_group_count": entry.region_impact_group_count,
            "coordinate_count": entry.coordinate_count,
        },
        "z_worst": entry.z_worst,
        "binding_slack": entry.binding_slack,
        "maximum_positive_exceedance": entry.maximum_positive_exceedance,
        "balanced_mean_positive_exceedance": (entry.balanced_mean_positive_exceedance),
        "binding_constraints": [
            [
                item.region,
                item.constraint_type,
                item.optimization_lcia_method,
                item.impact_category_id,
                item.year,
                item.asr,
            ]
            for item in entry.binding_constraints
        ],
    }
    return _METRIC_ENTRY_IDENTITY_PREFIX + payload_sha256(payload)
