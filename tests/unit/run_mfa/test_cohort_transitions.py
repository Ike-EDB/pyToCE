from dataclasses import replace
from fractions import Fraction

import flodym as fd
import numpy as np
import pandas as pd
import pytest
from scenario_fixtures import compile_scenario_payload
from scipy.stats import lognorm

from pytoce.run_mfa.model.cohorts import (
    accepted_default_cohort_result,
    conditional_survival_ratio,
    continue_cohort_stock,
    fixed_lifetime_stock_system,
    lognormal_log_survival,
)
from pytoce.run_mfa.remind_mfa.translation import NormalizedCohortState
from pytoce.run_mfa.scenario.scenario_contract import ScenarioFeasibilityError
from pytoce.run_optimization.calculation.interventions import (
    DemandLifetimeDecision,
    build_demand_lifetime_trajectories,
)


def _baseline(*, stock_scale: float = 1.0) -> NormalizedCohortState:
    """Supply balanced source cohorts and a default aging path after 2019."""
    years = tuple(range(1950, 2101))
    shape = (len(years), len(years), 1, 1)
    stock_by_cohort = np.zeros(shape)
    outflow_by_cohort = np.zeros(shape)
    legacy_i = years.index(2010)
    boundary_i = years.index(2019)
    stock_by_cohort[legacy_i : boundary_i + 1, legacy_i, 0, 0] = 10.0
    retired_i = years.index(2000)
    stock_by_cohort[retired_i:legacy_i, retired_i, 0, 0] = 5.0
    outflow_by_cohort[years.index(2010), retired_i, 0, 0] = 5.0
    sigma = np.sqrt(np.log1p(0.1**2))
    scale = 10.0 / np.exp(sigma**2 / 2)
    future_stock = (
        10.0
        * lognorm.sf(np.arange(10.5, 91.5), sigma, scale=scale)
        / lognorm.sf(9.5, sigma, scale=scale)
    )
    stock_by_cohort[boundary_i + 1 :, legacy_i, 0, 0] = future_stock
    outflow_by_cohort[boundary_i + 1 :, legacy_i, 0, 0] = -np.diff(
        np.concatenate(([10.0], future_stock))
    )
    annual_stock = stock_by_cohort.sum(axis=1)
    annual_inflow = np.zeros((len(years), 1, 1))
    annual_inflow[retired_i, 0, 0] = 5.0
    annual_inflow[legacy_i, 0, 0] = 10.0
    annual_outflow = outflow_by_cohort.sum(axis=1)
    material_shares = np.array([[[0.8], [0.2]]])
    carbon_content = np.array([[0.8, 0.6], [0.2, 0.4]])
    return NormalizedCohortState(
        years=years,
        interval_years=np.ones(len(years)),
        regions=("EUR",),
        goods=("Transportation",),
        materials=("HDPE", "BioPlastics"),
        elements=("C", "Other Elements"),
        stock_by_cohort=stock_by_cohort * stock_scale,
        outflow_by_cohort=outflow_by_cohort * stock_scale,
        annual_stock=annual_stock * stock_scale,
        annual_inflow=annual_inflow * stock_scale,
        annual_outflow=annual_outflow * stock_scale,
        material_shares=material_shares,
        carbon_content=carbon_content,
        lifetime_mean=np.array([10.0]),
        lifetime_std=np.array([1.0]),
    )


def _compiled(
    *,
    first_mean: float,
    second_mean: float | None = None,
    first_target: float = 20.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in range(2020, 2101):
        mean = second_mean if second_mean is not None and year >= 2050 else first_mean
        rows.extend(
            [
                {
                    "year": year,
                    "region": None,
                    "material": None,
                    "good": "Transportation",
                    "parameter_id": "lifetime_mean",
                    "resolved_value": mean,
                },
                {
                    "year": year,
                    "region": None,
                    "material": None,
                    "good": "Transportation",
                    "parameter_id": "lifetime_std",
                    "resolved_value": mean / 10.0,
                },
                {
                    "year": year,
                    "region": "EUR",
                    "material": None,
                    "good": "Transportation",
                    "parameter_id": "stock_projection",
                    "resolved_value": first_target,
                },
                {
                    "year": year,
                    "region": "EUR",
                    "material": "HDPE",
                    "good": "Transportation",
                    "parameter_id": "plastic_family_share",
                    "resolved_value": 0.8 if year < 2040 else 0.4,
                },
                {
                    "year": year,
                    "region": "EUR",
                    "material": "BioPlastics",
                    "good": "Transportation",
                    "parameter_id": "plastic_family_share",
                    "resolved_value": 0.2 if year < 2040 else 0.6,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_conditional_lognormal_reference_uses_only_boundary_survivors() -> None:
    boundary_stock = 7.25
    age_2019 = np.array([9.5])
    age_2020 = np.array([10.5])
    expected_stock = boundary_stock * np.exp(
        lognormal_log_survival(age_2020, 15.0, 2.0) - lognormal_log_survival(age_2019, 15.0, 2.0)
    )
    ratio = conditional_survival_ratio(
        age_2019,
        age_2020,
        mean=15.0,
        std=2.0,
    )

    np.testing.assert_array_equal(boundary_stock * ratio, expected_stock)
    np.testing.assert_allclose(
        boundary_stock - expected_stock,
        boundary_stock * (1.0 - ratio),
        rtol=1e-14,
        atol=0.0,
    )


@pytest.mark.parametrize("means", [(10.0, 15.0), (0.8, 1.0)])
def test_fixed_lifetime_system_reconstructs_complete_annual_closing_stock(
    means: tuple[float, float],
) -> None:
    """Proposal coefficients retain old cohorts and the real closing-stock basis."""
    source = _baseline()
    original = source.stock_by_cohort.copy()
    compiled = _compiled(first_mean=means[0], second_mean=means[1])
    system = fixed_lifetime_stock_system(source, compiled)
    result = continue_cohort_stock(
        source, compiled, optimization_overlay=True, record_diagnostics=False
    )
    first = source.years.index(2020)
    indices = np.arange(first, len(source.years))
    new_closing = result.stock_by_cohort[indices, indices]

    assert system.years == tuple(range(2020, 2101))
    assert system.regions == source.regions
    assert system.goods == source.goods
    np.testing.assert_array_equal(source.stock_by_cohort, original)
    np.testing.assert_array_equal(
        system.inherited_stock, result.stock_by_cohort[first:, :first].sum(axis=1)
    )
    np.testing.assert_array_equal(
        system.closing_survival[np.arange(81), np.arange(81)], np.ones((81, 1))
    )
    reconstructed = system.inherited_stock + np.einsum(
        "tcg,crg->trg", system.closing_survival, new_closing
    )
    # Cumulative coefficients and successive cohort propagation have different floating-point
    # operation orders. The MFA retains its exact sign check.
    np.testing.assert_allclose(reconstructed, result.annual_stock[first:], rtol=5e-15, atol=0)
    if means[0] < 1:
        # For short-lived products, same-year losses separate gross inflow from the proposal
        # system's closing-stock variable.
        assert np.any(result.annual_inflow[first:] > new_closing)


@pytest.mark.parametrize("stock_target", [20.0, 100_000.0])
def test_prospective_transitions_preserve_history_age_material_and_stock(
    stock_target: float,
) -> None:
    baseline = _baseline()
    progress: list[tuple[str, int | None]] = []

    default = continue_cohort_stock(
        baseline,
        _compiled(first_mean=10.0, first_target=stock_target),
        optimization_overlay=False,
    )
    longer_constant = continue_cohort_stock(
        baseline,
        _compiled(first_mean=15.0, first_target=stock_target),
        optimization_overlay=False,
    )
    longer = continue_cohort_stock(
        baseline,
        _compiled(first_mean=15.0, second_mean=8.0, first_target=stock_target),
        optimization_overlay=False,
        progress=lambda stage, year: progress.append((stage, year)),
    )
    compact = continue_cohort_stock(
        baseline,
        _compiled(first_mean=15.0, second_mean=8.0, first_target=stock_target),
        optimization_overlay=False,
        record_diagnostics=False,
    )
    for name in ("stock_by_cohort", "outflow_by_cohort", "annual_inflow", "cohort_material_shares"):
        np.testing.assert_array_equal(getattr(compact, name), getattr(longer, name))
    assert compact.transition_ledger.empty
    assert not longer.transition_ledger.empty
    assert progress == [("cohort_continuation", year) for year in range(2020, 2101, 10)]
    years = baseline.years
    event_2020 = years.index(2020)
    event_2050 = years.index(2050)
    legacy_2010 = years.index(2010)
    retired_2000 = years.index(2000)
    cohort_2040 = years.index(2040)

    np.testing.assert_array_equal(
        longer.stock_by_cohort[:event_2020],
        baseline.stock_by_cohort[:event_2020],
    )
    np.testing.assert_array_equal(
        longer.outflow_by_cohort[:event_2020],
        baseline.outflow_by_cohort[:event_2020],
    )
    assert (
        longer.outflow_by_cohort[event_2020, legacy_2010, 0, 0]
        < default.outflow_by_cohort[event_2020, legacy_2010, 0, 0]
    )
    assert not longer.stock_by_cohort[event_2020:, retired_2000].any()
    assert not longer.outflow_by_cohort[event_2020:, retired_2000].any()
    np.testing.assert_array_equal(
        longer.cohort_material_shares[legacy_2010, 0, :, 0],
        [0.8, 0.2],
    )
    np.testing.assert_array_equal(
        longer.cohort_material_shares[cohort_2040, 0, :, 0],
        [0.4, 0.6],
    )
    assert (
        longer.outflow_by_cohort[event_2050, cohort_2040, 0, 0]
        > longer_constant.outflow_by_cohort[event_2050, cohort_2040, 0, 0]
    )
    np.testing.assert_allclose(longer.annual_stock[event_2020:, 0, 0], stock_target)
    assert np.all(longer.annual_inflow[event_2020:] >= 0.0)
    assert longer.annual_stock[-1].sum() == pytest.approx(stock_target)


def test_nominal_elements_preserve_material_mass_and_frozen_history() -> None:
    """Keep prospective C/Other accounts on the exact stock and trade mass basis."""
    # ABS and PVC source fractions exercise C-dominant and Other-dominant
    # materials. Independent products can differ from their polymer total.
    fractions = np.array([[0.824528584806638, 0.38], [0.175471415193362, 0.62]])
    source = replace(_baseline(), materials=("ABS_ASA_SAN", "PVC"), carbon_content=fractions.copy())
    compiled = _compiled(first_mean=15.0).replace(
        {"material": {"HDPE": "ABS_ASA_SAN", "BioPlastics": "PVC"}}
    )
    results = (
        accepted_default_cohort_result(source),
        continue_cohort_stock(source, compiled, optimization_overlay=False),
    )
    boundary = source.years.index(2020)
    for result in results:
        for name in ("stock", "inflow", "outflow"):
            material = getattr(result, "material_" + name)
            elements = getattr(result, "carbon_" + name)
            original = np.einsum("trmg,em->termg", material, fractions)
            np.testing.assert_array_equal(elements[:boundary], original[:boundary])
            np.testing.assert_array_equal(
                elements[boundary:, 0, :, 0], original[boundary:, 0, :, 0]
            )
            np.testing.assert_array_equal(
                elements[boundary:, 1, :, 1], original[boundary:, 1, :, 1]
            )
            np.testing.assert_array_equal(elements[boundary:].sum(axis=1), material[boundary:])
            assert np.all(elements[boundary:] >= 0.0)
            # Use exact binary manufacturing quantities to check fully imported endpoints.
            for total, carbon, other in zip(
                material[boundary:].flat,
                elements[boundary:, 0].flat,
                elements[boundary:, 1].flat,
                strict=True,
            ):
                assert Fraction(float(carbon)) + Fraction(float(other)) == Fraction(float(total))
        np.testing.assert_array_equal(result.carbon_content, fractions)
    np.testing.assert_array_equal(source.carbon_content, fractions)


def test_unreachable_stock_target_fails_without_forced_disposal() -> None:
    with pytest.raises(
        ScenarioFeasibilityError,
        match="does not force surviving stock to disposal",
    ) as ordinary:
        continue_cohort_stock(
            _baseline(),
            _compiled(first_mean=15.0, first_target=0.0),
            optimization_overlay=False,
        )
    assert ordinary.value.code is None

    with pytest.raises(ScenarioFeasibilityError) as optimization:
        continue_cohort_stock(
            _baseline(),
            _compiled(first_mean=15.0, first_target=0.0),
            optimization_overlay=True,
        )
    assert optimization.value.code == "optimization_stock_below_surviving_cohort"
    assert optimization.value.diagnostic is not None
    assert optimization.value.diagnostic.equation == ("target_stock >= surviving_existing_stock")
    assert optimization.value.diagnostic.observed == 0.0
    assert optimization.value.diagnostic.bound is not None
    assert optimization.value.diagnostic.shortfall == optimization.value.diagnostic.bound
    assert dict(optimization.value.diagnostic.coordinates) == {
        "year": 2020,
        "region": "EUR",
        "good": "Transportation",
    }


@pytest.mark.parametrize(
    ("mean", "message"),
    (
        (1e-6, "finite, nonnegative inflow"),
        (1e-200, "finite, nonnegative inflow"),
        (1e-308, "conditional cohort survival is not representable"),
    ),
)
def test_short_finite_lifetime_reports_unrepresentable_cohort_calculation(
    mean: float, message: str
) -> None:
    """Positive admitted lifetimes can exceed annual floating-point representability."""
    scenario = compile_scenario_payload(
        {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": "short_lifetime",
            "ssp_scenario": "SSP2",
            "changes": [
                {
                    "parameter": "lifetime_mean",
                    "good": "Transportation",
                    "changes": [{"year": 2020, "mode": "absolute", "value": mean}],
                }
            ],
        }
    )
    compiled = scenario.compiled_parameters
    lifetimes = compiled.loc[
        compiled["parameter_id"].isin(("lifetime_mean", "lifetime_std"))
        & compiled["good"].eq("Transportation")
        & compiled["year"].ge(2020),
        "resolved_value",
    ].to_numpy(dtype=float)
    assert np.isfinite(lifetimes).all()
    assert (lifetimes > 0).all()
    with (
        np.errstate(divide="ignore", invalid="ignore", over="ignore"),
        pytest.raises(ScenarioFeasibilityError, match=message),
    ):
        continue_cohort_stock(
            _baseline(),
            compiled,
            stock_target=np.full((81, 1, 1), 20.0),
            optimization_overlay=False,
        )


def test_invalid_optimization_stock_target_has_typed_context() -> None:
    compiled = _compiled(first_mean=15.0)
    stock_target = np.full((81, 1, 1), float("nan"))

    with pytest.raises(ScenarioFeasibilityError) as error:
        continue_cohort_stock(
            _baseline(),
            compiled,
            optimization_overlay=True,
            stock_target=stock_target,
        )

    assert error.value.code == "optimization_stock_target_infeasible"
    assert error.value.diagnostic is not None
    assert error.value.diagnostic.status == "nonfinite_stock_target"
    assert error.value.diagnostic.observed is None
    assert error.value.diagnostic.bound == 0.0
    assert error.value.diagnostic.tolerance == 0.0
    assert error.value.diagnostic.shortfall is None
    assert error.value.diagnostic.count == 81
    assert dict(error.value.diagnostic.coordinates) == {
        "year": 2020,
        "region": "EUR",
        "good": "Transportation",
    }


def test_shortfall_beyond_local_roundoff_remains_infeasible() -> None:
    baseline = _baseline()
    surviving_2020 = (
        10.0
        * conditional_survival_ratio(
            np.array([9.5]),
            np.array([10.5]),
            mean=15.0,
            std=1.5,
        )[0]
    )

    with pytest.raises(
        ScenarioFeasibilityError,
        match="does not force surviving stock to disposal",
    ):
        continue_cohort_stock(
            baseline,
            _compiled(
                first_mean=15.0,
                first_target=surviving_2020 - 5e-12,
            ),
            optimization_overlay=False,
        )


def test_continuation_copies_the_accepted_state_until_its_change_year() -> None:
    baseline = _baseline()
    result = continue_cohort_stock(
        baseline,
        _compiled(first_mean=15.0),
        transition_start_year=2030,
        optimization_overlay=False,
    )
    transition_i = baseline.years.index(2030)

    np.testing.assert_array_equal(
        result.stock_by_cohort[:transition_i],
        baseline.stock_by_cohort[:transition_i],
    )
    np.testing.assert_array_equal(
        result.outflow_by_cohort[:transition_i],
        baseline.outflow_by_cohort[:transition_i],
    )
    np.testing.assert_array_equal(
        result.annual_inflow[:transition_i],
        baseline.annual_inflow[:transition_i],
    )
    assert result.transition_ledger["year"].min() == 2030


def test_delayed_demand_can_fall_below_survivors_after_parent_growth_stops() -> None:
    """Retiming can fail despite a feasible 2050 path, with fixed lifetime spread.

    This is a designed cohort counterexample.
    Native nonnegative inflows create young stock until 2050; replacement inflows
    then hold its level. The delayed reduction can outrun retirement after growth stops.
    """
    years = tuple(range(1950, 2101))
    good = "Buildings & Construction"
    dimensions = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name="Time", letter="t", items=list(years), dtype=int),
            fd.Dimension(name="Region", letter="r", items=["EUR"], dtype=str),
            fd.Dimension(name="Good", letter="g", items=[good], dtype=str),
        ]
    )
    lifetime = fd.LogNormalLifetime(
        dims=dimensions,
        mean=np.full(dimensions.shape, 35.0),
        std=np.full(dimensions.shape, 7.0),
    )
    growth = fd.InflowDrivenDSM(dims=dimensions, lifetime_model=lifetime)
    for year in range(2019, 2051):
        growth.inflow.values[years.index(year), 0, 0] = 1.3 ** (year - 2019)
    growth.compute()
    target = growth.stock.values.copy()
    target[years.index(2051) :] = target[years.index(2050)]
    parent = fd.StockDrivenDSM(dims=dimensions, lifetime_model=lifetime, solver="manual")
    parent.stock.values[...] = target
    parent.compute()
    assert np.all(parent.inflow.values >= 0)
    baseline = replace(
        _baseline(),
        goods=(good,),
        stock_by_cohort=parent.get_stock_by_cohort().copy(),
        outflow_by_cohort=parent.get_outflow_by_cohort().copy(),
        annual_stock=target,
        annual_inflow=parent.inflow.values.copy(),
        annual_outflow=parent.outflow.values.copy(),
        lifetime_mean=np.array([35.0]),
        lifetime_std=np.array([7.0]),
    )
    compiled = _compiled(first_mean=35.0).replace({"good": {"Transportation": good}})
    compiled.loc[compiled.parameter_id.eq("lifetime_std"), "resolved_value"] = 7.0
    for material, share in (("HDPE", 0.8), ("BioPlastics", 0.2)):
        compiled.loc[compiled.material.eq(material), "resolved_value"] = share
    stock_rows = compiled.parameter_id.eq("stock_projection")
    stock = compiled.loc[stock_rows, ["year", "region", "good"]].assign(
        stock_projection=target[70:, 0, 0]
    )
    lifetimes = stock.loc[:, ["year", "good"]].assign(lifetime_mean=35.0)
    for target_year in (2050, 2075, 2100):
        trajectory = build_demand_lifetime_trajectories(
            parent_stock_targets=stock,
            parent_lifetimes=lifetimes,
            parent_stock_target_identity="stock::native_growth_stop_fixture",
            parent_lifetime_identity="lifetime::fixed_mean35_std7_fixture",
            decision=DemandLifetimeDecision(10, 0, target_year=target_year),
        )
        compiled.loc[stock_rows, "resolved_value"] = (
            trajectory.assignments.resolved_value.to_numpy()
        )
        if target_year == 2050:
            selected = continue_cohort_stock(baseline, compiled, optimization_overlay=True)
            assert np.all(selected.annual_inflow[70:] >= 0)
            np.testing.assert_allclose(
                selected.annual_stock[70:, 0, 0], compiled.loc[stock_rows, "resolved_value"]
            )
            assert np.all(np.diff(compiled.loc[stock_rows, "resolved_value"]) >= 0)
        else:
            with pytest.raises(ScenarioFeasibilityError) as failure:
                continue_cohort_stock(baseline, compiled, optimization_overlay=True)
            assert failure.value.code == "optimization_stock_below_surviving_cohort"
            diagnostic = failure.value.diagnostic
            assert diagnostic is not None
            assert dict(diagnostic.coordinates) == {"year": 2051, "region": "EUR", "good": good}
            assert diagnostic.tolerance > 0
            assert diagnostic.shortfall > max(16.0, diagnostic.tolerance)


@pytest.mark.parametrize("stock_scale", [1e-12, 1.0, 1e6])
def test_stock_roundoff_retains_survivors_and_zero_new_inflow(stock_scale: float) -> None:
    """One-ULP target deficits retain balanced cohorts at tiny and large mass scales."""
    source = _baseline(stock_scale=stock_scale)
    first = source.years.index(2020)
    ratio = conditional_survival_ratio(np.array([9.5]), np.array([10.5]), mean=15.0, std=1.5)[0]
    surviving = source.annual_stock[first - 1, 0, 0] * ratio
    target = np.nextafter(surviving, -np.inf)
    compiled = _compiled(first_mean=15.0, first_target=target)
    ordinary = continue_cohort_stock(source, compiled, optimization_overlay=False)
    compact = continue_cohort_stock(
        source, compiled, optimization_overlay=True, record_diagnostics=False
    )
    for name in ("stock_by_cohort", "outflow_by_cohort", "annual_inflow"):
        np.testing.assert_array_equal(getattr(ordinary, name), getattr(compact, name))
        np.testing.assert_array_equal(
            getattr(ordinary, name)[:first], getattr(source, name)[:first]
        )
    assert ordinary.annual_inflow[first, 0, 0] == 0.0
    assert ordinary.stock_by_cohort[first, first, 0, 0] == 0.0
    assert ordinary.annual_stock[first, 0, 0] == surviving
    assert ordinary.annual_outflow[first, 0, 0] == source.annual_stock[first - 1, 0, 0] - surviving
    row = ordinary.transition_ledger.iloc[0]
    assert row.stock_target_residual == target - surviving < 0.0
    assert -row.stock_target_residual <= row.stock_target_roundoff_tolerance
    assert row.stock_target_roundoff_tolerance < 1e-12 * surviving
    assert compact.transition_ledger.empty


@pytest.mark.parametrize("stock_scale", [1e-12, 1.0, 1e6])
def test_stock_roundoff_does_not_hide_larger_local_shortfalls(stock_scale: float) -> None:
    """Reject a deficit beyond the floating-point arithmetic tolerance."""
    source = _baseline(stock_scale=stock_scale)
    first = source.years.index(2020)
    ratio = conditional_survival_ratio(np.array([9.5]), np.array([10.5]), mean=15.0, std=1.5)[0]
    surviving = source.annual_stock[first - 1, 0, 0] * ratio
    # Twice the first-year local bound remains a tiny but resolvable deficit.
    target = surviving * (1.0 - 4 * (first + 2) * np.finfo(float).eps / 2)
    with pytest.raises(ScenarioFeasibilityError) as failure:
        continue_cohort_stock(
            source, _compiled(first_mean=15.0, first_target=target), optimization_overlay=True
        )
    diagnostic = failure.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.code == "optimization_stock_below_surviving_cohort"
    assert diagnostic.shortfall is not None
    assert diagnostic.shortfall > diagnostic.tolerance > 0.0


def test_stock_roundoff_keeps_positive_residual_and_exact_zero() -> None:
    """Preserve small positive demand and the exact zero-stock solution."""
    source = _baseline()
    first = source.years.index(2020)
    ratio = conditional_survival_ratio(np.array([9.5]), np.array([10.5]), mean=15.0, std=1.5)[0]
    surviving = source.annual_stock[first - 1, 0, 0] * ratio
    target = np.nextafter(surviving, np.inf)
    result = continue_cohort_stock(
        source, _compiled(first_mean=15.0, first_target=target), optimization_overlay=True
    )
    assert result.stock_by_cohort[first, first, 0, 0] == target - surviving > 0.0
    assert result.annual_inflow[first, 0, 0] > 0.0
    empty = replace(
        source,
        stock_by_cohort=np.zeros_like(source.stock_by_cohort),
        outflow_by_cohort=np.zeros_like(source.outflow_by_cohort),
        annual_stock=np.zeros_like(source.annual_stock),
        annual_inflow=np.zeros_like(source.annual_inflow),
        annual_outflow=np.zeros_like(source.annual_outflow),
    )
    zero = continue_cohort_stock(
        empty, _compiled(first_mean=15.0, first_target=0.0), optimization_overlay=False
    )
    assert not zero.annual_inflow.any()
    assert not zero.annual_outflow.any()
    assert not zero.annual_stock.any()
    assert zero.transition_ledger.stock_target_residual.eq(0.0).all()
    assert zero.transition_ledger.stock_target_roundoff_tolerance.eq(0.0).all()
