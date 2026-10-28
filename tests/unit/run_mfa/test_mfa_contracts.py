"""Scalar scenario admission and scale-aware cohort stock targets."""

import numpy as np
import pandas as pd
import pytest

from pytoce.run_mfa.model.cohorts import continue_cohort_stock
from pytoce.run_mfa.remind_mfa.translation import NormalizedCohortState
from pytoce.run_mfa.scenario.scenario_contract import (
    ScenarioError,
    validate_foreground_scenario_name,
)


def test_foreground_name_requires_a_readable_filename_stem() -> None:
    """Reject a name that cannot identify an admitted scenario YAML file."""

    with pytest.raises(ScenarioError, match="lowercase snake_case"):
        validate_foreground_scenario_name("Not readable")


def _zero_cohort_state() -> NormalizedCohortState:
    """Return the exact annual axes for one zero-mass REMIND-MFA cohort state."""

    years = tuple(range(1950, 2101))
    count = len(years)
    return NormalizedCohortState(
        years=years,
        interval_years=np.ones(count),
        regions=("EUR",),
        goods=("Packaging",),
        materials=("PET",),
        elements=("C", "Other Elements"),
        stock_by_cohort=np.zeros((count, count, 1, 1)),
        outflow_by_cohort=np.zeros((count, count, 1, 1)),
        annual_stock=np.zeros((count, 1, 1)),
        annual_inflow=np.zeros((count, 1, 1)),
        annual_outflow=np.zeros((count, 1, 1)),
        material_shares=np.ones((1, 1, 1)),
        # Synthetic complementary fractions represent the admitted two-element composition.
        carbon_content=np.asarray([[0.6], [0.4]]),
        lifetime_mean=np.asarray([10.0]),
        lifetime_std=np.asarray([3.0]),
    )


def test_cohort_continuation_admits_large_valid_stock_with_local_roundoff() -> None:
    """Admit a large stock target within its arithmetic tolerance."""

    rows: list[dict[str, object]] = []
    for year in range(2020, 2101):
        rows.extend(
            [
                {
                    "parameter_id": "lifetime_mean",
                    "year": year,
                    "good": "Packaging",
                    "resolved_value": 10.0,
                },
                {
                    "parameter_id": "lifetime_std",
                    "year": year,
                    "good": "Packaging",
                    "resolved_value": 3.0,
                },
                {
                    "parameter_id": "plastic_family_share",
                    "year": year,
                    "region": "EUR",
                    "material": "PET",
                    "good": "Packaging",
                    "resolved_value": 1.0,
                },
            ]
        )

    baseline = _zero_cohort_state()
    result = continue_cohort_stock(
        baseline,
        pd.DataFrame.from_records(rows),
        stock_target=np.full((81, 1, 1), 100_000.0),
        optimization_overlay=False,
    )
    np.testing.assert_allclose(result.annual_stock[-81:], 100_000.0, rtol=1e-13, atol=0)
    np.testing.assert_array_equal(result.stock_by_cohort[:70], baseline.stock_by_cohort[:70])
    assert np.isfinite(result.annual_inflow).all()
    assert np.all(result.annual_inflow >= 0)
