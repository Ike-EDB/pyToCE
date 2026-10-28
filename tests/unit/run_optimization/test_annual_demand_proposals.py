"""Fixed-lifetime stock proposals at the real cohort and SciPy boundaries."""

from dataclasses import replace
from fractions import Fraction
from itertools import accumulate

import numpy as np
import pandas as pd
import pytest
from intervention_fixtures import _GOODS, _H12
from scipy import optimize

from pytoce.run_mfa.model.cohorts import (
    FixedLifetimeStockSystem,
    continue_cohort_stock,
    fixed_lifetime_stock_system,
)
from pytoce.run_mfa.remind_mfa.translation import NormalizedCohortState
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    AnnualDemandProposalError,
    initial_demand_lifetime_decisions,
    propose_annual_demand,
)
from pytoce.run_optimization.calculation.interventions import DemandLifetimeDecision


@pytest.mark.parametrize(
    ("step_pct", "expected_demand", "expected_lifetime"),
    ((20, (0, 20, 40, 60, 80, 90), (0, 20, 40, 60, 80, 100)), (50, (0, 50, 90), (0, 50, 100))),
)
def test_initial_grid_keeps_parent_and_both_endpoint_bounds(
    step_pct: int, expected_demand: tuple[int, ...], expected_lifetime: tuple[int, ...]
) -> None:
    """Retain the 90-percent demand bound for every requested grid spacing."""
    decisions = initial_demand_lifetime_decisions(target_year=2050, step_pct=step_pct)
    assert tuple(dict.fromkeys(item.demand_reduction_pct for item in decisions)) == expected_demand
    assert tuple(dict.fromkeys(item.lifetime_extension_pct for item in decisions)) == (
        expected_lifetime
    )
    assert len(decisions) == len(expected_demand) * len(expected_lifetime)
    assert decisions[0] == DemandLifetimeDecision(0, 0, 2050)
    assert decisions[-1] == DemandLifetimeDecision(90, 100, 2050)


def _stock_inputs(mean: float) -> tuple[NormalizedCohortState, pd.DataFrame]:
    """Supply all region/end-use stock controls for an invented empty historical stock."""
    years = tuple(range(1950, 2101))
    annual = np.zeros((len(years), len(_H12), len(_GOODS)))
    cohorts = np.zeros((len(years), *annual.shape))
    state = NormalizedCohortState(
        years=years,
        interval_years=np.ones(len(years)),
        regions=_H12,
        goods=_GOODS,
        materials=("HDPE",),
        elements=("C", "Other Elements"),
        stock_by_cohort=cohorts,
        outflow_by_cohort=cohorts,
        annual_stock=annual,
        annual_inflow=annual,
        annual_outflow=annual,
        material_shares=np.ones((len(_H12), 1, len(_GOODS))),
        carbon_content=np.array([[0.75], [0.25]]),
        lifetime_mean=np.full(len(_GOODS), mean),
        lifetime_std=np.full(len(_GOODS), mean / 10),
    )
    compiled = pd.DataFrame.from_records(
        (parameter, year, good, value)
        for year in range(2020, 2101)
        for good in _GOODS
        for parameter, value in (("lifetime_mean", mean), ("lifetime_std", mean / 10))
    )
    compiled.columns = ["parameter_id", "year", "good", "resolved_value"]
    return state, compiled


@pytest.fixture(scope="module")
def stock_system() -> FixedLifetimeStockSystem:
    """Share the immutable short-lifetime coefficient fixture across boundary checks."""
    return fixed_lifetime_stock_system(*_stock_inputs(1.0))


@pytest.mark.parametrize("target_year", (2050, 2075, 2100))
def test_real_solver_preserves_full_stock_equation_and_exact_transport(
    stock_system: FixedLifetimeStockSystem,
    monkeypatch: pytest.MonkeyPatch,
    target_year: int,
) -> None:
    """A real proposal is monotone, holds after its target and satisfies every annual row."""
    free_years = target_year - 2020
    parent = np.full(stock_system.inherited_stock.shape, 10.0)
    lifetime = DemandLifetimeDecision(0, 0).lifetime_fraction
    captured: dict[str, object] = {}
    actual_linprog = optimize.linprog

    def observed_solve(c: np.ndarray, **kwargs: object) -> optimize.OptimizeResult:
        """Observe the public solver's matrices and result."""
        result = actual_linprog(c, **kwargs)
        captured.update(kwargs, result=result)
        return result

    monkeypatch.setattr(optimize, "linprog", observed_solve)
    proposal = propose_annual_demand(
        stock_system,
        parent_stock=parent,
        demand_reduction_pct=50,
        lifetime_extension_pct=0,
        lifetime_fraction=lifetime,
        target_year=target_year,
    )
    assert proposal is not None
    demand = np.array([float(value) for value in proposal.demand_fraction])
    assert demand[0] == 0.0
    assert np.all(np.diff(demand) >= 0)
    np.testing.assert_array_equal(demand[free_years + 1 :], np.full(81 - free_years, 0.5))
    target = parent * (1 - demand[1:, None, None])
    # Independent dense solve of A @ closing = target - inherited, rather
    # than repeating the proposal owner's h/K elimination.
    closing = np.stack(
        [
            np.linalg.solve(
                stock_system.closing_survival[:, :, good],
                target[:, :, good] - stock_system.inherited_stock[:, :, good],
            )
            for good in range(len(stock_system.goods))
        ],
        axis=-1,
    )
    assert closing.shape == (81, 12, 7)
    assert closing.min() > 0.0
    solution = captured["result"]
    increments = solution.x[:free_years]
    assert proposal.demand_fraction[1 : free_years + 1] == tuple(
        accumulate(Fraction.from_float(float(value)) for value in increments)
    )
    coefficients = np.linalg.solve(
        stock_system.closing_survival[:, :, 0], np.diag(parent[:, 0, 0])[:, :free_years]
    )
    transported = captured["A_ub"][: parent.size, :free_years].reshape(81, 12, 7, free_years)[
        :, 0, 0
    ]
    np.testing.assert_allclose(
        coefficients @ np.cumsum(increments), transported @ increments, rtol=0, atol=1e-13
    )
    assert solution.fun == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(
        demand,
        [float(value) for value in DemandLifetimeDecision(50, 0, target_year).demand_fraction],
        rtol=0,
        atol=1e-14,
    )
    assert captured["A_ub"].shape == (parent.size + 2 * free_years + 1, 2 * free_years)
    # Each nonnegative increment is bounded by the final reduction, while
    # deviation variables retain their complete nonnegative epigraph domain.
    assert captured["bounds"] == [(0.0, 0.5)] * free_years + [(0.0, None)] * free_years


def test_increment_coordinates_admit_the_previously_nonmonotone_solver_case() -> None:
    """Propose monotone demand for a lifetime mean of ten years and standard
    deviation of one year.
    """
    system = fixed_lifetime_stock_system(*_stock_inputs(10.0))
    result = propose_annual_demand(
        system,
        parent_stock=np.full(system.inherited_stock.shape, 10.0),
        demand_reduction_pct=50,
        lifetime_extension_pct=0,
        lifetime_fraction=DemandLifetimeDecision(0, 0).lifetime_fraction,
        target_year=2050,
    )
    assert result is not None
    assert result.demand_fraction[0] == 0
    assert result.demand_fraction[-1] == Fraction(1, 2)
    assert all(a <= b for a, b in zip(result.demand_fraction, result.demand_fraction[1:]))
    assert all(value == Fraction(1, 2) for value in result.demand_fraction[31:])
    reference = np.array(
        [float(value) for value in DemandLifetimeDecision(50, 0).demand_fraction[1:]]
    )
    proposed = np.array([float(value) for value in result.demand_fraction[1:]])
    kernel = system.closing_survival[:, :, 0]
    reference_closing = np.linalg.solve(kernel, 10.0 * (1.0 - reference))
    assert reference_closing.min() < 0.0
    assert np.abs(proposed[:30] - reference[:30]).sum() > 0.0
    closing = np.linalg.solve(kernel, 10.0 * (1.0 - proposed))
    # Check the LP linear algebra. Full MFA subsequently applies its local stock roundoff
    # convention.
    np.testing.assert_allclose(kernel @ closing, 10.0 * (1.0 - proposed), rtol=0, atol=1e-13)
    assert closing.min() >= -1e-12


def test_original_proposal_passes_common_stock_roundoff_admission() -> None:
    """Admit the original active LP face under the shared local stock convention."""
    state, compiled = _stock_inputs(10.0)
    system = fixed_lifetime_stock_system(state, compiled)
    parent = np.full(system.inherited_stock.shape, 10.0)
    reference = DemandLifetimeDecision(50, 0)
    options = dict(
        parent_stock=parent,
        demand_reduction_pct=50,
        lifetime_extension_pct=0,
        lifetime_fraction=reference.lifetime_fraction,
        target_year=2050,
    )
    original = propose_annual_demand(system, **options)
    assert original is not None
    material = pd.DataFrame.from_records(
        ("plastic_family_share", year, region, "HDPE", good, 1.0)
        for year in system.years
        for region in system.regions
        for good in system.goods
    )
    material.columns = ["parameter_id", "year", "region", "material", "good", "resolved_value"]
    physical_controls = pd.concat([compiled, material], ignore_index=True)
    original_target = parent * (
        1 - np.array([float(value) for value in original.demand_fraction[1:]])[:, None, None]
    )
    accepted = continue_cohort_stock(
        state,
        physical_controls,
        stock_target=original_target,
        optimization_overlay=True,
        record_diagnostics=False,
    )
    assert accepted.annual_inflow[70:].min() >= 0
    assert original.demand_fraction[0] == 0
    assert original.demand_fraction[31:] == (Fraction(1, 2),) * 51
    assert original.lifetime_fraction == reference.lifetime_fraction
    assert all(a <= b for a, b in zip(original.demand_fraction, original.demand_fraction[1:]))


@pytest.mark.parametrize(
    ("pulse_year", "message"),
    (
        (2049, "exceeded its endpoint by"),
        (2048, "nonfinite or negative increment/deviation"),
    ),
)
def test_real_solver_tolerance_cannot_relax_exact_annual_controls(
    pulse_year: int, message: str
) -> None:
    """Reject real solver values that violate exact endpoint or monotonicity controls."""
    years = tuple(range(1950, 2101))
    annual = np.zeros((len(years), 1, 1))
    cohorts = np.zeros((len(years), *annual.shape))
    state = NormalizedCohortState(
        years=years,
        interval_years=np.ones(len(years)),
        regions=("EUR",),
        goods=("Packaging",),
        materials=("HDPE",),
        elements=("C",),
        stock_by_cohort=cohorts,
        outflow_by_cohort=cohorts,
        annual_stock=annual,
        annual_inflow=annual,
        annual_outflow=annual,
        material_shares=np.ones((1, 1, 1)),
        carbon_content=np.ones((1, 1)),
        lifetime_mean=np.array([1.0]),
        lifetime_std=np.array([0.2]),
    )
    systems: list[FixedLifetimeStockSystem] = []
    for mean in (1.0, 2.0):
        # Preserve Packaging's source coefficient of variation: mean 0.5, std 0.1 years.
        std = 0.1 * (mean / 0.5)
        compiled = pd.DataFrame.from_records(
            (
                (parameter, year, "Packaging", value)
                for year in range(2020, 2101)
                for parameter, value in (("lifetime_mean", mean), ("lifetime_std", std))
            ),
            columns=["parameter_id", "year", "good", "resolved_value"],
        )
        systems.append(fixed_lifetime_stock_system(state, compiled))
    baseline, candidate = systems
    pulse_i = candidate.years.index(pulse_year)
    next_i = pulse_i + 1
    rho_parent = baseline.closing_survival[next_i, pulse_i, 0]
    rho = candidate.closing_survival[next_i, pulse_i, 0]
    endpoint = 0.5
    # This fixed binary gap is below HiGHS's 1e-7 feasibility tolerance;
    # solver tolerance cannot authorize decreasing controls into the endpoint.
    delta = 2.0**-25
    parent = np.zeros((81, 1, 1))
    parent[pulse_i, 0, 0] = 1.0  # One Mt of new closing stock at the pulse year.
    parent[next_i, 0, 0] = rho * (1.0 - 2.0 * delta)
    parent[next_i + 1 :, 0, 0] = 1.0
    # The shorter-lifetime parent admits positive new stock after the pulse. Its
    # subsequent nondecreasing target also exceeds its surviving cohorts.
    assert 0.0 < rho_parent < parent[next_i, 0, 0] < 1.0
    assert not candidate.inherited_stock.any()
    # After pulse p, x_(p+1) = rho*((1-2*delta)*(1-d_(p+1)) - (1-d_p)). For p=2049, d2050=E=0.5
    # requires d2049 >= E+delta, exceeding E. For p=2048, q2049=d2049-d2048 must be <=
    # -2*delta*(1-d2049) <= -delta, violating nonnegative increments. These fixtures check the
    # cohort equations directly.
    actual_gap = (1.0 - parent[next_i, 0, 0] / rho) * (1.0 - endpoint)
    assert 0.0 < actual_gap < 1e-7
    with pytest.raises(AnnualDemandProposalError, match=message):
        propose_annual_demand(
            candidate,
            parent_stock=parent,
            demand_reduction_pct=50,
            lifetime_extension_pct=100,
            lifetime_fraction=(Fraction(0), *((Fraction(1),) * 81)),
            target_year=2050,
        )


def test_zero_closing_stock_is_feasible_and_late_year_infeasibility_is_not_omitted(
    stock_system: FixedLifetimeStockSystem,
) -> None:
    """Zero closing stock is admitted; a constraint in the final region/end-use/year still binds."""
    options = {
        "parent_stock": np.zeros(stock_system.inherited_stock.shape),
        "demand_reduction_pct": 50,
        "lifetime_extension_pct": 0,
        "lifetime_fraction": DemandLifetimeDecision(0, 0).lifetime_fraction,
        "target_year": 2050,
    }
    assert propose_annual_demand(stock_system, **options) is not None
    inherited = stock_system.inherited_stock.copy()
    inherited[-1, -1, -1] = 1.0
    assert (
        propose_annual_demand(replace(stock_system, inherited_stock=inherited), **options) is None
    )


@pytest.mark.parametrize("status", (1, 3, 4))
def test_solver_failure_remains_visible(
    stock_system: FixedLifetimeStockSystem, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """Documented non-infeasibility failures cannot silently remove an endpoint proposal."""

    def failed_solve(c: np.ndarray, **kwargs: object) -> optimize.OptimizeResult:
        """Return the documented public result shape for an interrupted/failed solver."""
        assert c.shape == (60,)
        assert kwargs["method"] == "highs-ds"
        return optimize.OptimizeResult(status=status, success=False, message="external failure")

    monkeypatch.setattr(optimize, "linprog", failed_solve)
    with pytest.raises(AnnualDemandProposalError, match=f"solver status {status}"):
        propose_annual_demand(
            stock_system,
            parent_stock=np.ones(stock_system.inherited_stock.shape),
            demand_reduction_pct=50,
            lifetime_extension_pct=0,
            lifetime_fraction=DemandLifetimeDecision(0, 0).lifetime_fraction,
            target_year=2050,
        )


@pytest.mark.parametrize(
    "message",
    (
        "(HiGHS Status 2: Model error)",
        "The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; "
        "primal_status is At lower/fixed bound)",
    ),
)
def test_status_two_distinguishes_infeasibility_from_model_error(
    stock_system: FixedLifetimeStockSystem,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    """SciPy 1.13.1's overloaded public status cannot hide a model-load failure."""

    def result(c: np.ndarray, **kwargs: object) -> optimize.OptimizeResult:
        """Supply the exact public result shape of either external status-two classification."""
        assert c.shape == (60,)
        assert kwargs["method"] == "highs-ds"
        return optimize.OptimizeResult(status=2, success=False, message=message)

    monkeypatch.setattr(optimize, "linprog", result)
    options = dict(
        parent_stock=np.ones(stock_system.inherited_stock.shape),
        demand_reduction_pct=50,
        lifetime_extension_pct=0,
        lifetime_fraction=DemandLifetimeDecision(0, 0).lifetime_fraction,
        target_year=2050,
    )
    if "Model error" in message:
        with pytest.raises(AnnualDemandProposalError, match="Model error"):
            propose_annual_demand(stock_system, **options)
    else:
        assert propose_annual_demand(stock_system, **options) is None
