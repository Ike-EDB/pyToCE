"""Scientific invariants shared by monitoring and target setting."""

from math import fsum

import numpy as np
import pytest

from pytoce.shared.assessment.asr import ASRError, calculate_asr


def test_stable_global_ratios_sum_burdens_and_capacities_separately() -> None:
    """Aggregate burdens and allocated carrying capacities separately before division."""
    scores = np.zeros((3, 82, 2), dtype=np.float64)
    scores[:, 1, 0] = [1e16, 1.0, -1e16]
    capacities = np.ones_like(scores)
    capacities[1] = 2.0
    climate = np.zeros((3, 82), dtype=np.float64)
    climate[0, :3] = [1e16, 1.0, -1e16]
    climate[1, 0] = -2.0
    result = calculate_asr(
        annual_scores=scores,
        annual_allocated_cc=capacities,
        annual_budget_compatible_gwp100_scores=climate,
        cumulative_gwp100_allocated_cc=np.array([1.0, 2.0, 1.0]),
        control_start=1,
    )
    assert result.global_metrics.annual_scores[1, 0] == 1.0
    assert result.global_metrics.annual_asr[1, 0] == 0.25
    assert result.cumulative_gwp100_scores.tolist() == [1.0, -2.0, 0.0]
    assert result.global_metrics.cumulative_gwp100_score == fsum([1.0, -2.0, 0.0])
    assert result.global_metrics.cumulative_gwp100_asr == -0.25
    assert not result.annual_asr.flags.writeable
    with pytest.raises(ValueError):
        result.annual_asr.setflags(write=True)


@pytest.mark.parametrize("failure", ["annual", "cumulative", "sum"])
def test_nonfinite_asr_and_overflow_are_explicit_errors(failure: str) -> None:
    """Reject ratios that overflow for extreme finite inputs."""
    annual = np.ones((1, 82, 1), dtype=np.float64)
    annual_acc = np.ones_like(annual)
    climate = np.ones((1, 82), dtype=np.float64)
    cumulative_acc = np.ones(1, dtype=np.float64)
    if failure == "annual":
        annual_acc.fill(1e-320)
    elif failure == "cumulative":
        cumulative_acc.fill(1e-320)
    else:
        climate.fill(1e308)
    with pytest.raises(ASRError):
        calculate_asr(
            annual_scores=annual,
            annual_allocated_cc=annual_acc,
            annual_budget_compatible_gwp100_scores=climate,
            cumulative_gwp100_allocated_cc=cumulative_acc,
            control_start=1,
        )
