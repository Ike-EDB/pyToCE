"""Compare environmental burdens with allocated carrying capacities.

Monitoring and target setting use the same annual and cumulative definitions.
Input owners admit matching, finite scores and positive allocated carrying
capacities before these calculations. Annual arrays use [region, year, category]
order; the independent climate numerator uses [region, year] order.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum
from typing import TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
Scalar = TypeVar("Scalar", bound=np.generic)


class ASRError(ValueError):
    """An ASR or stable sum cannot be represented as a finite value."""


@dataclass(frozen=True, slots=True)
class GlobalMetrics:
    """Global assessment values derived from the 12 world regions.

    Annual LCIA, allocated carrying capacity and ASR arrays have axes
    ``[year, impact category]``. Peak annual ASRs have only the impact category
    axis; annual climate scores have only the year axis. The cumulative GWP100
    fields describe 2019--2100; the cumulative ASR is the ratio of the cumulative
    score to its allocated carrying capacity. LCIA results retain native impact
    category units; allocated carrying capacities and ASR denominators are positive.

    ``mean_positive_exceedance`` averages positive exceedance for each impact
    category's peak annual ASR and the cumulative climate ASR. This global
    reporting summary is separate from the regionally balanced mean used in
    candidate ranking.
    """

    annual_scores: FloatArray
    annual_allocated_cc: FloatArray
    annual_asr: FloatArray
    annual_peak_asr: FloatArray
    annual_budget_compatible_gwp100_scores: FloatArray
    cumulative_gwp100_score: float
    cumulative_gwp100_allocated_cc: float
    cumulative_gwp100_asr: float
    worst_asr: float
    mean_positive_exceedance: float
    compliant: bool


@dataclass(frozen=True, slots=True)
class ASRValues:
    """Annual ratios and independent cumulative GWP100 assessment results."""

    annual_asr: FloatArray
    cumulative_gwp100_scores: FloatArray
    cumulative_gwp100_asr: FloatArray
    global_metrics: GlobalMetrics


def calculate_asr(
    *,
    annual_scores: FloatArray,
    annual_allocated_cc: FloatArray,
    annual_budget_compatible_gwp100_scores: FloatArray,
    cumulative_gwp100_allocated_cc: FloatArray,
    control_start: int,
) -> ASRValues:
    """Calculate annual ASRs and the independent cumulative climate ASR.

    Annual environmental burdens are divided by the allocated carrying
    capacities for the same region, year, and impact category. Cumulative
    GWP100 burdens are summed over the supplied period before division by
    the separately supplied cumulative allocated carrying capacity. Global ASRs
    divide separate stable sums of regional burdens and allocated carrying
    capacities.

    Args:
        annual_scores: Finite annual LCIA results in native category units.
        annual_allocated_cc: Matching positive annual allocated carrying capacities.
        annual_budget_compatible_gwp100_scores: Annual all-GHG GWP100 burdens
            in the cumulative allocated carrying capacity's native unit.
        cumulative_gwp100_allocated_cc: Positive cumulative allocated carrying
            capacities by region.
        control_start: First year-axis index included in annual compliance.
            pyToCE supplies 2019--2100 and assesses annual compliance from 2020.

    Returns:
        Immutable regional ratio arrays, cumulative burdens, and global metrics.

    Raises:
        ASRError: If summation overflows or a regional ratio is nonfinite.
    """
    scores = annual_scores
    annual_acc = annual_allocated_cc
    climate_scores = annual_budget_compatible_gwp100_scores
    cumulative_acc = cumulative_gwp100_allocated_cc
    with np.errstate(over="ignore", invalid="ignore"):
        annual_asr = scores / annual_acc
    if not np.isfinite(annual_asr).all():
        raise ASRError(
            "An annual LCIA result divided by its aCC produced a missing or infinite ASR. Check "
            "the "
            "positive annual aCC values in the selected input and use a new result name after "
            "correcting them."
        )
    cumulative_scores = np.fromiter(
        (
            _finite_fsum(
                (float(value) for value in row),
                name="regional_cumulative_gwp100_score",
            )
            for row in climate_scores
        ),
        dtype=np.float64,
        count=scores.shape[0],
    )
    with np.errstate(over="ignore", invalid="ignore"):
        cumulative_asr = cumulative_scores / cumulative_acc
    if not np.isfinite(cumulative_asr).all():
        raise ASRError(
            "A cumulative climate LCIA result divided by its aCC produced a missing or infinite "
            "ASR. Check the positive cumulative aCC values in the selected input and use a new "
            "result name after correcting them."
        )

    return ASRValues(
        annual_asr=_readonly(annual_asr),
        cumulative_gwp100_scores=_readonly(cumulative_scores),
        cumulative_gwp100_asr=_readonly(cumulative_asr),
        global_metrics=_global_metrics(
            annual_scores=scores,
            annual_allocated_cc=annual_acc,
            climate_scores=climate_scores,
            cumulative_acc=cumulative_acc,
            control_start=control_start,
            category_count=scores.shape[2],
        ),
    )


def _finite_fsum(values: Iterable[float], *, name: str) -> float:
    """Sum finite input values with ``fsum`` and report numeric overflow."""
    try:
        return fsum(values)
    except OverflowError as exc:
        raise ASRError(f"{name} must be finite") from exc


def _readonly(values: NDArray[Scalar]) -> NDArray[Scalar]:
    """Copy an array into an immutable bytes-backed view."""
    contiguous = np.ascontiguousarray(values)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
    return cast(NDArray[Scalar], result)


def _global_metrics(
    *,
    annual_scores: FloatArray,
    annual_allocated_cc: FloatArray,
    climate_scores: FloatArray,
    cumulative_acc: FloatArray,
    control_start: int,
    category_count: int,
) -> GlobalMetrics:
    """Calculate global LCIA results, allocated carrying capacity, and ASR.

    Regional numerators and denominators are summed separately across the
    12 world regions with ``math.fsum``. Annual and cumulative global ASR values are then
    calculated from those sums.
    Annual maxima use the configured control years; cumulative climate metrics
    cover the full 2019--2100 period.
    """
    year_count = annual_scores.shape[1]
    global_scores = np.empty((year_count, category_count), dtype=np.float64)
    global_acc = np.empty((year_count, category_count), dtype=np.float64)
    for year_index in range(year_count):
        for category_index in range(category_count):
            global_scores[year_index, category_index] = _finite_fsum(
                (float(value) for value in annual_scores[:, year_index, category_index]),
                name="global_annual_score",
            )
            global_acc[year_index, category_index] = _finite_fsum(
                (float(value) for value in annual_allocated_cc[:, year_index, category_index]),
                name="global_annual_allocated_cc",
            )
    global_asr = global_scores / global_acc
    global_peak = np.max(global_asr[control_start:, :], axis=0)
    global_climate_scores = np.fromiter(
        (
            _finite_fsum(
                (float(value) for value in climate_scores[:, year_index]),
                name="global_annual_gwp100_score",
            )
            for year_index in range(year_count)
        ),
        dtype=np.float64,
        count=year_count,
    )
    global_cumulative_score = _finite_fsum(
        (
            _finite_fsum(
                (float(value) for value in regional_values),
                name="regional_cumulative_gwp100_score",
            )
            for regional_values in climate_scores
        ),
        name="global_cumulative_gwp100_score",
    )
    global_cumulative_acc = _finite_fsum(
        (float(value) for value in cumulative_acc),
        name="global_cumulative_allocated_cc",
    )
    global_cumulative_asr = global_cumulative_score / global_cumulative_acc
    global_worst = max(float(np.max(global_peak)), global_cumulative_asr)
    global_exceedance = _finite_fsum(
        (
            *(max(0.0, float(value) - 1.0) for value in global_peak),
            max(0.0, global_cumulative_asr - 1.0),
        ),
        name="global_positive_exceedance",
    ) / (category_count + 1)
    return GlobalMetrics(
        annual_scores=_readonly(global_scores),
        annual_allocated_cc=_readonly(global_acc),
        annual_asr=_readonly(global_asr),
        annual_peak_asr=_readonly(global_peak),
        annual_budget_compatible_gwp100_scores=_readonly(global_climate_scores),
        cumulative_gwp100_score=global_cumulative_score,
        cumulative_gwp100_allocated_cc=global_cumulative_acc,
        cumulative_gwp100_asr=global_cumulative_asr,
        worst_asr=global_worst,
        mean_positive_exceedance=global_exceedance,
        compliant=global_worst <= 1.0,
    )
