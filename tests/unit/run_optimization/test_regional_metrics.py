"""Regional annual and cumulative optimization metric contracts."""

import pytest
from regional_metric_fixtures import (
    BACKGROUNDS,
    H12_REGIONS,
    _inputs,
    _metric,
    np,
)

from pytoce.run_optimization.calculation.regional_metrics import (
    intervention_distance,
    transgressing_constraints,
)


def test_intervention_distance_averages_each_active_group() -> None:
    demand_lifetime = intervention_distance(
        physical_candidate_identity="candidate-i",
        intervention_authority_identity="authority-i",
        distance_inputs=_inputs(demand=0.2, lifetime=0.6),
        optimization_block="demand_lifetime_frontier",
    )
    circularity_feedstock = intervention_distance(
        physical_candidate_identity="candidate-ii",
        intervention_authority_identity="authority-ii",
        distance_inputs=_inputs(collection=0.3, treatment=0.6, feedstock=0.9),
        optimization_block="circularity_feedstock_anchor",
    )
    integrated = intervention_distance(
        physical_candidate_identity="candidate-iii",
        intervention_authority_identity="authority-iii",
        distance_inputs=_inputs(
            demand=0.2,
            lifetime=0.4,
            collection=0.6,
            treatment=0.8,
            feedstock=1.0,
        ),
        optimization_block="integrated_frontier",
    )

    assert demand_lifetime.total_distance == pytest.approx(0.4)
    assert demand_lifetime.identity == (
        "pytoce_optimization_intervention_distance_v7::"
        "2665670ffa419c81450695c09c973cb431e7d5ae88cc30f44b5d394770dc736a"
    )
    assert circularity_feedstock.total_distance == pytest.approx(0.6)
    assert integrated.total_distance == pytest.approx(0.6)


def test_regional_metrics_calculate_the_expected_asr() -> None:
    metrics = _metric(BACKGROUNDS[0], -0.5)

    assert np.all(metrics.annual_asr == -0.5)
    assert np.all(metrics.cumulative_gwp100_scores == -41.0)
    assert metrics.compliant_region_count == 12
    assert metrics.z_worst == pytest.approx(-0.41)
    assert metrics.all_region_joint_compliant
    assert metrics.binding_slack == pytest.approx(1.41)


def test_joint_counts_and_balanced_severity_use_every_controlled_coordinate() -> None:
    scores = np.full((12, 82, 1), 0.5, dtype=np.float64)
    violation = np.nextafter(1.0, np.inf)
    scores[0, 1, 0] = violation
    climate = np.zeros((12, 82), dtype=np.float64)

    metrics = _metric(
        BACKGROUNDS[0],
        0.0,
        annual_scores=scores,
        climate_scores=climate,
    )

    exceedance = violation - 1.0
    assert not metrics.all_region_joint_compliant
    assert metrics.binding_slack is None
    assert metrics.compliant_region_count == 11
    assert metrics.fully_compliant_regions == H12_REGIONS[1:]
    assert metrics.fully_compliant_impact_target_count == 1
    assert metrics.compliant_region_impact_group_count == 23
    assert metrics.compliant_coordinate_count == 983
    assert metrics.region_count == 12
    assert metrics.impact_target_count == 2
    assert metrics.region_impact_group_count == 24
    assert metrics.coordinate_count == 984
    assert metrics.maximum_positive_exceedance == exceedance
    # One positive coordinate is averaged over 81 years, two targets and 12 regions.
    # All other exceedances are zero, so each sum retains that coordinate exactly.
    assert metrics.balanced_mean_positive_exceedance == exceedance / 81.0 / 2.0 / 12.0


def test_exact_one_is_compliant_and_2019_is_annual_report_only() -> None:
    scores = np.full((12, 82, 1), 0.5, dtype=np.float64)
    scores[:, 0, :] = 100.0
    scores[0, 1, 0] = 1.0
    climate = np.zeros((12, 82), dtype=np.float64)

    metrics = _metric(
        BACKGROUNDS[0],
        0.0,
        annual_scores=scores,
        climate_scores=climate,
    )

    assert metrics.all_region_joint_compliant
    assert metrics.binding_slack == 0.0
    assert metrics.compliant_coordinate_count == metrics.coordinate_count


def test_cumulative_climate_includes_2019() -> None:
    scores = np.zeros((12, 82, 1), dtype=np.float64)
    climate = np.zeros((12, 82), dtype=np.float64)
    climate[:, 0] = 2.0

    metrics = _metric(
        BACKGROUNDS[0],
        0.0,
        annual_scores=scores,
        climate_scores=climate,
        cumulative_acc_value=1.0,
    )

    assert np.all(metrics.cumulative_gwp100_scores == 2.0)
    assert not metrics.all_region_joint_compliant
    assert metrics.fully_compliant_impact_targets == ("annual::ef31::climate_change",)


def test_binding_and_transgression_evidence_use_the_joint_state_maximum() -> None:
    """Bind worst-ASR coordinates and retain every above-one transgression."""
    scores = np.full((12, 82, 1), 0.5, dtype=np.float64)
    scores[0, 0, 0] = 9.0
    scores[0, 1, 0] = 1.4
    scores[1, 1, 0] = 1.2
    climate = np.zeros((12, 82), dtype=np.float64)

    metrics = _metric(
        BACKGROUNDS[0],
        0.0,
        annual_scores=scores,
        climate_scores=climate,
    )

    assert tuple(
        (item.region, item.constraint_type, item.year, item.asr)
        for item in metrics.binding_constraints
    ) == ((H12_REGIONS[0], "annual", 2020, 1.4),)
    assert tuple(
        (item.region, item.constraint_type, item.year, item.asr)
        for item in transgressing_constraints(metrics)
    ) == (
        (H12_REGIONS[0], "annual", 2020, 1.4),
        (H12_REGIONS[1], "annual", 2020, 1.2),
    )
