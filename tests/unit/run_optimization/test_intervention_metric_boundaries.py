"""Tests for interventions, metrics, and selected reporting."""

import numpy as np
import pandas as pd
import pytest
from intervention_fixtures import _circularity_parent, _demand
from regional_metric_fixtures import BACKGROUNDS, _metric

from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
    build_circularity_trajectories,
    build_demand_lifetime_evaluation_projection,
    build_intervention_distance_inputs,
)
from pytoce.run_optimization.calculation.regional_metrics import (
    RegionalMetrics,
    transgressing_constraints,
)
from pytoce.run_optimization.calculation.selected_reporting import (
    _apply_adjustment,
    _stage_response_adjustments,
)
from pytoce.shared.assessment.asr import ASRError


def test_exact_circularity_parent_and_demand_projection_preserve_parent_identity() -> None:
    """Keep unchanged circularity empty while identity-binding active demand trajectories."""

    circularity = build_circularity_trajectories(
        parent=_circularity_parent(),
        parent_identity="circularity::accepted",
        decision=CircularityDecision.parent(target_year=2075),
    )
    assert circularity.uses_exact_parent
    assert circularity.assignments.empty

    demand = _demand(DemandLifetimeDecision(10, 20, 2075))
    distance = build_intervention_distance_inputs(
        physical_parent_identity="physical::accepted",
        demand_lifetime_parent_identity="demand::accepted",
        demand_lifetime=demand,
        circularity=circularity,
        parent_circularity=_circularity_parent(),
    )
    assert distance.collection_distance == 0.0
    assert distance.treatment_share_distance == 0.0
    assert distance.feedstock_share_distance == 0.0
    projection = build_demand_lifetime_evaluation_projection(
        demand_lifetime=demand,
        physical_parent_identity="physical::accepted",
        demand_lifetime_parent_identity="demand::accepted",
    )
    assert projection.decision == demand.decision
    assert projection.demand_lifetime_trajectory_identity == demand.identity
    assert projection.identity.startswith(
        "pytoce_optimization_demand_lifetime_evaluation_projection_v4::"
    )


@pytest.mark.parametrize(
    ("decision", "parameter_id"),
    [
        (DemandLifetimeDecision(10, 0), "stock_projection"),
        (DemandLifetimeDecision(0, 10), "lifetime_mean"),
    ],
)
def test_demand_and_lifetime_families_can_change_independently(
    decision: DemandLifetimeDecision,
    parameter_id: str,
) -> None:
    """Write only the scientific parameter family activated by a direct decision."""

    trajectories = _demand(decision)
    assert set(trajectories.assignments["parameter_id"]) == {parameter_id}


def test_metric_division_rejects_nonfinite_regional_asr() -> None:
    """Fail if extreme finite scientific inputs overflow either ASR definition."""

    annual_scores = np.ones((12, 82, 1), dtype=np.float64)
    annual_acc = np.full_like(annual_scores, 1.0e-320)
    with pytest.raises(ASRError, match="missing or infinite ASR"):
        _metric_with_allocations(
            annual_scores=annual_scores,
            annual_allocated_cc=annual_acc,
            climate_scores=np.zeros((12, 82), dtype=np.float64),
            cumulative_acc=np.ones(12, dtype=np.float64),
        )

    climate_scores = np.full((12, 82), 1.0e300, dtype=np.float64)
    with pytest.raises(ASRError, match="missing or infinite ASR"):
        _metric_with_allocations(
            annual_scores=np.zeros((12, 82, 1), dtype=np.float64),
            annual_allocated_cc=np.ones((12, 82, 1), dtype=np.float64),
            climate_scores=climate_scores,
            cumulative_acc=np.full(12, 1.0e-300, dtype=np.float64),
        )

    with pytest.raises(ASRError, match="regional_cumulative_gwp100_score must be finite"):
        _metric_with_allocations(
            annual_scores=np.zeros((12, 82, 1), dtype=np.float64),
            annual_allocated_cc=np.ones((12, 82, 1), dtype=np.float64),
            climate_scores=np.full((12, 82), 1.0e308, dtype=np.float64),
            cumulative_acc=np.ones(12, dtype=np.float64),
        )


def _metric_with_allocations(
    *,
    annual_scores: np.ndarray,
    annual_allocated_cc: np.ndarray,
    climate_scores: np.ndarray,
    cumulative_acc: np.ndarray,
) -> RegionalMetrics:
    """Call the canonical metric owner with explicit finite allocation arrays."""

    from pytoce.run_lca.calculation.foreground import H12_REGIONS
    from pytoce.run_optimization.calculation.interventions import (
        InterventionDistanceInputs,
    )
    from pytoce.run_optimization.calculation.regional_metrics import (
        AnnualCategory,
        calculate_regional_metrics,
        intervention_distance,
    )

    distance_inputs = InterventionDistanceInputs(
        identity="distance-inputs::fixture",
        physical_parent_identity="physical-parent::fixture",
        demand_lifetime_parent_identity="demand-parent::fixture",
        demand_lifetime_trajectory_identity="demand-trajectory::fixture",
        circularity_trajectory_identity="circularity-trajectory::fixture",
        target_year=2050,
        demand_reduction_fraction=0.1,
        lifetime_extension_fraction=0.2,
        collection_distance=0.0,
        treatment_share_distance=0.0,
        feedstock_share_distance=0.0,
    )
    distance = intervention_distance(
        physical_candidate_identity="candidate::fixture",
        intervention_authority_identity="intervention::fixture",
        distance_inputs=distance_inputs,
        optimization_block="demand_lifetime_frontier",
    )
    return calculate_regional_metrics(
        evaluation_identity="evaluation::fixture",
        background_scenario=BACKGROUNDS[0],
        allocated_cc_case="p50",
        allocated_cc_context_identity="acc-context::fixture",
        physical_candidate_identity="candidate::fixture",
        intervention_authority_identity="intervention::fixture",
        optimization_item_identity="item::fixture",
        optimization_block="demand_lifetime_frontier",
        is_zero_intervention_parent=False,
        intervention_distance=distance,
        optimization_lcia_method_registry_identity="methods::optimization",
        cumulative_climate_method_registry_identity="methods::climate",
        comparison_scope_identity="scope::fixture",
        annual_scores=annual_scores,
        annual_allocated_cc=annual_allocated_cc,
        annual_budget_compatible_gwp100_scores=climate_scores,
        cumulative_gwp100_allocated_cc=cumulative_acc,
        regions=H12_REGIONS,
        years=tuple(range(2019, 2101)),
        annual_categories=(AnnualCategory("ef31", "climate_change"),),
    )


def test_cumulative_transgression_is_reported_separately_from_annual_targets() -> None:
    """Retain the independent cumulative GWP100 constraint in failure evidence."""

    metrics = _metric(
        BACKGROUNDS[0],
        0.0,
        annual_scores=np.zeros((12, 82, 1), dtype=np.float64),
        climate_scores=np.ones((12, 82), dtype=np.float64),
        cumulative_acc_value=1.0,
    )
    constraints = transgressing_constraints(metrics)
    assert len(constraints) == 12
    assert {item.constraint_type for item in constraints} == {"cumulative_gwp100"}
    assert {item.asr for item in constraints} == {82.0}


def test_exact_parent_response_adjustment_preserves_baseline_scores() -> None:
    """Keep baseline totals unchanged when the selected change is empty."""

    stage = _stage_response_adjustments(
        pd.DataFrame(),
        pd.DataFrame(),
        background=BACKGROUNDS[0],
    )
    assert stage.empty

    keys = ("region", "year")
    parent = pd.DataFrame(
        {"region": ["EUR"], "year": [2050], "score": [-2.0], "contribution_norm": [3.0]}
    )
    empty_adjustment = pd.DataFrame(
        {
            "region": pd.Series(dtype="object"),
            "year": pd.Series(dtype="int64"),
            "score_adjustment": pd.Series(dtype="float64"),
            "norm_adjustment": pd.Series(dtype="float64"),
            "changed_parent_score": pd.Series(dtype="float64"),
            "changed_candidate_score": pd.Series(dtype="float64"),
            "changed_parent_norm": pd.Series(dtype="float64"),
            "changed_candidate_norm": pd.Series(dtype="float64"),
        }
    )
    observed = _apply_adjustment(parent, empty_adjustment, keys)
    pd.testing.assert_frame_equal(observed, parent, check_exact=True)
