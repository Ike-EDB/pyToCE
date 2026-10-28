import numpy as np

from pytoce.prepare_backgrounds.preparation.premise import (
    ACTIVE_BACKGROUND_SCENARIOS,
)
from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_optimization.calculation.interventions import (
    InterventionDistanceInputs,
)
from pytoce.run_optimization.calculation.regional_metrics import (
    AnnualCategory,
    calculate_regional_metrics,
    intervention_distance,
)

YEARS = tuple(range(2019, 2101))
BACKGROUNDS = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)


def _inputs(
    *,
    demand: float = 0.0,
    lifetime: float = 0.0,
    collection: float = 0.0,
    treatment: float = 0.0,
    feedstock: float = 0.0,
) -> InterventionDistanceInputs:
    return InterventionDistanceInputs(
        identity="distance-inputs",
        physical_parent_identity="physical-parent",
        demand_lifetime_parent_identity="demand-parent",
        demand_lifetime_trajectory_identity="demand-trajectory",
        circularity_trajectory_identity="circularity-trajectory",
        target_year=2050,
        demand_reduction_fraction=demand,
        lifetime_extension_fraction=lifetime,
        collection_distance=collection,
        treatment_share_distance=treatment,
        feedstock_share_distance=feedstock,
    )


def _metric(
    background: str,
    score: float,
    *,
    annual_scores: np.ndarray | None = None,
    climate_scores: np.ndarray | None = None,
    cumulative_acc_value: float = 100.0,
):
    distance = intervention_distance(
        physical_candidate_identity="candidate",
        intervention_authority_identity="authority",
        distance_inputs=_inputs(demand=0.1, lifetime=0.2),
        optimization_block="demand_lifetime_frontier",
    )
    scores = (
        np.full((12, 82, 1), score, dtype=np.float64) if annual_scores is None else annual_scores
    )
    annual_acc = np.ones_like(scores)
    climate = (
        np.full((12, 82), score, dtype=np.float64) if climate_scores is None else climate_scores
    )
    cumulative_acc = np.full(12, cumulative_acc_value, dtype=np.float64)
    return calculate_regional_metrics(
        evaluation_identity=f"evaluation::{background}",
        background_scenario=background,
        allocated_cc_case="p50",
        allocated_cc_context_identity="allocated-context",
        physical_candidate_identity="candidate",
        intervention_authority_identity="authority",
        optimization_item_identity=f"item::{background}",
        optimization_block="demand_lifetime_frontier",
        is_zero_intervention_parent=False,
        intervention_distance=distance,
        optimization_lcia_method_registry_identity="optimization-methods",
        cumulative_climate_method_registry_identity="climate-methods",
        comparison_scope_identity=f"scope::{background}",
        annual_scores=scores,
        annual_allocated_cc=annual_acc,
        annual_budget_compatible_gwp100_scores=climate,
        cumulative_gwp100_allocated_cc=cumulative_acc,
        regions=H12_REGIONS,
        years=YEARS,
        annual_categories=(AnnualCategory("ef31", "climate_change"),),
    )
