"""Prospective stock/cohort continuation from the frozen end-2019 state."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import flodym as fd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from flodym.lifetime_models import UnevenTimeDim  # type: ignore[import-untyped]
from scipy.stats import lognorm  # type: ignore[import-untyped]

from pytoce.run_mfa.remind_mfa.translation import NormalizedCohortState
from pytoce.run_mfa.scenario.scenario_contract import (
    OptimizationPhysicalInfeasibility,
    ScenarioFeasibilityError,
)

OPTIMIZATION_COHORT_FEASIBILITY_CODES = frozenset(
    {
        "optimization_stock_target_infeasible",
        "optimization_stock_below_surviving_cohort",
    }
)

_BOUNDARY_YEAR = 2019
_FIRST_PROSPECTIVE_YEAR = 2020
_FINAL_YEAR = 2100


@dataclass(frozen=True, slots=True)
class FixedLifetimeStockSystem:
    """Relate prospective closing cohorts to targets at fixed annual lifetimes.

    ``inherited_stock`` is surviving end-2019 stock in Mt, with axes
    ``[year, region, good]``. ``closing_survival`` has axes
    ``[year, prospective_cohort, good]``. Its lower triangle maps each new
    closing cohort to later closing stocks; its diagonal is one. These are
    stock constraints for proposing demand paths.
    """

    years: tuple[int, ...]
    regions: tuple[str, ...]
    goods: tuple[str, ...]
    inherited_stock: np.ndarray
    closing_survival: np.ndarray


@dataclass(frozen=True, slots=True)
class ProspectiveCohortResult:
    """Store the in-use stock and cohort calculation from 1950 through 2100.

    Attributes:
        years: Calendar-year coordinate used by every time axis.
        regions: world region coordinate.
        goods: Plastic-using product groups.
        materials: Plastic material families.
        elements: Nominal REMIND bookkeeping labels C and Other Elements.
        stock_by_cohort: In-use stock in Mt with axes
            ``[year, cohort, region, good]``.
        outflow_by_cohort: End-of-life outflow in Mt with the same axes as
            ``stock_by_cohort``.
        annual_stock: Total in-use stock in Mt with axes ``[year, region, good]``.
        annual_inflow: Gross material inflow entering use, including the portion
            that leaves use within the same year, in Mt with axes
            ``[year, region, good]``.
        annual_outflow: Annual end-of-life outflow in Mt with axes
            ``[year, region, good]``.
        cohort_material_shares: Material shares with axes
            ``[cohort, region, material, good]``.
        material_stock: In-use material stock in Mt with axes
            ``[year, region, material, good]``.
        material_inflow: Material entering use in Mt with the same axes.
        material_outflow: Material leaving use in Mt with the same axes.
        carbon_stock: In-use nominal element mass in Mt with axes
            ``[year, element, region, material, good]``.
        carbon_inflow: Nominal element mass entering use in Mt with the same axes.
        carbon_outflow: Nominal element mass leaving use in Mt with the same axes.
        carbon_content: Nominal element mass fractions with axes ``[element, material]``.
        transition_ledger: Annual stock-transition checks, including opening,
            surviving, inflow, outflow, and closing stock values in Mt.
        material_identity_method: Method used to retain or update cohort material shares.
    """

    years: tuple[int, ...]
    regions: tuple[str, ...]
    goods: tuple[str, ...]
    materials: tuple[str, ...]
    elements: tuple[str, ...]
    stock_by_cohort: np.ndarray
    outflow_by_cohort: np.ndarray
    annual_stock: np.ndarray
    annual_inflow: np.ndarray
    annual_outflow: np.ndarray
    cohort_material_shares: np.ndarray
    material_stock: np.ndarray
    material_inflow: np.ndarray
    material_outflow: np.ndarray
    carbon_stock: np.ndarray
    carbon_inflow: np.ndarray
    carbon_outflow: np.ndarray
    carbon_content: np.ndarray
    transition_ledger: pd.DataFrame
    material_identity_method: str


def _nominal_element_masses(material_mass: np.ndarray, state: NormalizedCohortState) -> np.ndarray:
    """Partition prospective polymer Mt into its two nominal REMIND elements.

    Keep the larger fraction's product and derive the smaller by subtraction.
    The two stored components then sum exactly to the material amount, including
    at a fully imported demand boundary. These are nominal C/Other accounts.
    Historical projections through 2019 retain their original numerical values.
    """
    result = np.einsum("trmg,em->termg", material_mass, state.carbon_content)
    prospective = np.asarray(state.years) >= _FIRST_PROSPECTIVE_YEAR
    # The admitted source has exactly C and Other Elements. Retaining the larger
    # component makes the complementary subtraction exact (Sterbenz's lemma),
    # so downstream exact mass arithmetic uses the original polymer total.
    for material, larger in enumerate(np.argmax(state.carbon_content, axis=0)):
        smaller = 1 - larger
        result[prospective, smaller, :, material, :] = (
            material_mass[prospective, :, material, :] - result[prospective, larger, :, material, :]
        )
    return result


def accepted_default_cohort_result(
    state: NormalizedCohortState,
) -> ProspectiveCohortResult:
    """Represent the stored upstream cohort calculation."""

    cohort_material_shares = np.broadcast_to(
        state.material_shares[np.newaxis, ...],
        (len(state.years), *state.material_shares.shape),
    ).copy()
    material_stock = np.einsum(
        "tcrg,crmg->trmg",
        state.stock_by_cohort,
        cohort_material_shares,
    )
    material_outflow = np.einsum(
        "tcrg,crmg->trmg",
        state.outflow_by_cohort,
        cohort_material_shares,
    )
    material_inflow = np.einsum(
        "trg,trmg->trmg",
        state.annual_inflow,
        cohort_material_shares,
    )
    return ProspectiveCohortResult(
        years=state.years,
        regions=state.regions,
        goods=state.goods,
        materials=state.materials,
        elements=state.elements,
        stock_by_cohort=np.array(state.stock_by_cohort, copy=True),
        outflow_by_cohort=np.array(state.outflow_by_cohort, copy=True),
        annual_stock=np.array(state.annual_stock, copy=True),
        annual_inflow=np.array(state.annual_inflow, copy=True),
        annual_outflow=np.array(state.annual_outflow, copy=True),
        cohort_material_shares=cohort_material_shares,
        material_stock=material_stock,
        material_inflow=material_inflow,
        material_outflow=material_outflow,
        carbon_stock=_nominal_element_masses(material_stock, state),
        carbon_inflow=_nominal_element_masses(material_inflow, state),
        carbon_outflow=_nominal_element_masses(material_outflow, state),
        carbon_content=np.array(state.carbon_content, copy=True),
        transition_ledger=pd.DataFrame(
            columns=[
                "year",
                "region",
                "good",
                "opening_stock",
                "surviving_existing_stock",
                "new_inflow",
                "same_year_new_cohort_outflow",
                "total_outflow",
                "closing_stock",
                "stock_target",
                "stock_target_residual",
                "stock_target_roundoff_tolerance",
                "lifetime_mean",
                "lifetime_std",
                "unit",
                "transition_method",
            ]
        ),
        material_identity_method="accepted_default_material_identity",
    )


def flodym_time_bounds(years: tuple[int, ...]) -> np.ndarray:
    """Return FLODYM 0.8.1 time bounds for the exact ordered year coordinates."""

    dimension = fd.Dimension(
        name="Time",
        letter="t",
        items=list(years),
        dtype=int,
    )
    uneven = UnevenTimeDim(dim=dimension)
    return np.asarray(uneven.bounds, dtype=float)


def lognormal_log_survival(age: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Return log survival for age, mean and standard deviation measured in years."""

    # Squaring the spread avoids underflow from tiny lifetimes.
    relative_std = std / mean
    log_variance = math.log1p(relative_std * relative_std)
    log_scale = math.log(mean) - log_variance / 2
    shape = math.sqrt(log_variance)
    return np.asarray(
        lognorm.logsf(age, s=shape, loc=0, scale=math.exp(log_scale)),
        dtype=float,
    )


def conditional_survival_ratio(
    age_start: np.ndarray,
    age_end: np.ndarray,
    *,
    mean: float,
    std: float,
) -> np.ndarray:
    """Condition future survival on mass present at the transition boundary."""

    log_start = lognormal_log_survival(age_start, mean, std)
    log_end = lognormal_log_survival(age_end, mean, std)
    ratio = np.exp(log_end - log_start)
    # Both callers advance ordered annual ages. Finite log survival decreases
    # with age, so exp produces [0, 1]; tiny positive lifetimes can still make
    # the dependency's age/scale arithmetic unrepresentable.
    if not np.isfinite(ratio).all():
        raise ScenarioFeasibilityError(
            f"For lifetime_mean={mean:.17g} years and derived lifetime_std={std:.17g} years, "
            "conditional cohort survival is not representable. Revise lifetime_mean in "
            "the scenario YAML; the derived spread retains the source relative dispersion."
        )
    return ratio


def _input_cube(
    compiled: pd.DataFrame,
    parameter_id: str,
    *,
    years: tuple[int, ...],
    regions: tuple[str, ...] = (),
    materials: tuple[str, ...] = (),
    goods: tuple[str, ...] = (),
) -> np.ndarray:
    """Reshape one parameter table to the requested cohort coordinate cube."""
    selected = compiled.loc[compiled["parameter_id"].eq(parameter_id)]
    dimensions: list[str] = ["year"]
    coordinates: list[tuple[object, ...]] = [tuple(years)]
    for name, values in (
        ("region", regions),
        ("material", materials),
        ("good", goods),
    ):
        if values:
            dimensions.append(name)
            coordinates.append(tuple(values))
    expected = pd.MultiIndex.from_product(coordinates, names=dimensions)
    actual = selected.set_index(dimensions).reindex(expected)
    return (
        actual["resolved_value"]
        .to_numpy(dtype=float)
        .reshape(tuple(len(values) for values in coordinates))
    )


def fixed_lifetime_stock_system(
    state: NormalizedCohortState, compiled: pd.DataFrame
) -> FixedLifetimeStockSystem:
    """Derive annual stock constraints using the admitted lifetime schedule.

    For each region and good, ``A @ x = target - inherited_stock``, where
    ``A`` is ``closing_survival`` and ``x`` is new closing stock in Mt.
    The unit diagonal represents closing stock.
    The ordinary cohort owner still calculates inflow, outflow and MFA feasibility for any proposed
    path.

    Args:
        state: Admitted historical cohorts and ordered MFA coordinates.
        compiled: Already compiled annual controls. Mean and standard deviation
            are taken directly from their owner, preserving the original
            relative dispersion and its exact floating-point calculation.

    Returns:
        Surviving historical stocks and prospective survival coefficients for
        every modeled year from 2020 through 2100.
    """
    years = tuple(range(_FIRST_PROSPECTIVE_YEAR, _FINAL_YEAR + 1))
    first = state.years.index(_FIRST_PROSPECTIVE_YEAR)
    bounds = flodym_time_bounds(state.years)
    # FLODYM locates inflow at each interval's midpoint. Subsequent period
    # survival acts on the closing stock already present, including old cohorts.
    entry_time = 0.5 * (bounds[:-1] + bounds[1:])
    means = _input_cube(compiled, "lifetime_mean", years=years, goods=state.goods)
    spreads = _input_cube(compiled, "lifetime_std", years=years, goods=state.goods)
    remaining = state.stock_by_cohort[first - 1, :first].copy()
    inherited = np.zeros((len(years), len(state.regions), len(state.goods)))
    survival = np.zeros((len(years), len(years), len(state.goods)))
    for year_i, year in enumerate(years):
        event_i = state.years.index(year)
        for good_i in range(len(state.goods)):
            ratios = conditional_survival_ratio(
                bounds[event_i] - entry_time[:event_i],
                bounds[event_i + 1] - entry_time[:event_i],
                mean=float(means[year_i, good_i]),
                std=float(spreads[year_i, good_i]),
            )
            remaining[:, :, good_i] *= ratios[:first, np.newaxis]
            inherited[year_i, :, good_i] = remaining[:, :, good_i].sum(axis=0)
            if year_i:
                survival[year_i, :year_i, good_i] = (
                    survival[year_i - 1, :year_i, good_i] * ratios[first:]
                )
            # A unit of new closing stock is wholly present at its own close;
            # its same-year gross inflow and outflow remain the MFA owner's job.
            survival[year_i, year_i, good_i] = 1.0
    return FixedLifetimeStockSystem(years, state.regions, state.goods, inherited, survival)


def _solve_new_cohort_with_flodym(
    closing_stock: np.ndarray,
    *,
    year: int,
    regions: tuple[str, ...],
    goods: tuple[str, ...],
    lifetime_mean: np.ndarray,
    lifetime_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve one new-cohort inflow with FLODYM's public stock-driven model.

    FLODYM 0.8.1 requires at least three time coordinates to establish its
    public time bounds. The two preceding coordinates have exact zero stock,
    so the final coordinate is an isolated solve for the new cohort entering
    in ``year``. This retains FLODYM's own inflow-position convention and
    lower-triangular stock-driven solver.
    """

    time = fd.Dimension(
        name="Time",
        letter="t",
        items=[year - 2, year - 1, year],
        dtype=int,
    )
    region = fd.Dimension(name="Region", letter="r", items=list(regions), dtype=str)
    good = fd.Dimension(name="Good", letter="g", items=list(goods), dtype=str)
    dimensions = fd.DimensionSet(dim_list=[time, region, good])
    mean = np.broadcast_to(
        lifetime_mean[np.newaxis, np.newaxis, :],
        dimensions.shape,
    ).copy()
    std = np.broadcast_to(
        lifetime_std[np.newaxis, np.newaxis, :],
        dimensions.shape,
    ).copy()
    lifetime = fd.LogNormalLifetime(dims=dimensions, mean=mean, std=std)
    stock = fd.StockDrivenDSM(
        dims=dimensions,
        lifetime_model=lifetime,
        solver="manual",
    )
    # FLODYM exposes these arrays through generic stock objects in its type
    # annotations; the public 0.8.1 objects provide the documented ``values`` arrays.
    stock_values = cast(Any, stock.stock).values
    stock_values[:-1] = 0.0
    stock_values[-1] = closing_stock
    stock.compute()
    inflow = np.asarray(cast(Any, stock.inflow).values[-1], dtype=float).copy()
    same_year_outflow = np.asarray(cast(Any, stock.outflow).values[-1], dtype=float).copy()
    surviving = np.asarray(stock.get_stock_by_cohort()[-1, -1], dtype=float).copy()
    if (
        not np.isfinite(inflow).all()
        or not np.isfinite(same_year_outflow).all()
        or not np.isfinite(surviving).all()
        or np.any(inflow < 0)
        or np.any(same_year_outflow < 0)
    ):
        raise ScenarioFeasibilityError(
            "The stock and lifetime values do not produce a finite, nonnegative inflow "
            "that reaches the requested closing stock. Revise those scenario values; "
            "no MFA result was written."
        )
    return inflow, same_year_outflow, surviving


def continue_cohort_stock(
    state: NormalizedCohortState,
    compiled: pd.DataFrame,
    *,
    stock_target: np.ndarray | None = None,
    transition_start_year: int = _FIRST_PROSPECTIVE_YEAR,
    optimization_overlay: bool,
    record_diagnostics: bool = True,
    progress: Callable[[str, int | None], None] | None = None,
) -> ProspectiveCohortResult:
    """Continue exact end-2019 cohorts through prospective annual schedules.

    This function constructs the lower-triangular prospective survival system
    using FLODYM's public time bounds and inflow convention. It continues the
    accepted upstream history and applies each later lifetime rule to both
    surviving and new cohorts.

    Cohort stock and outflow use ``[year, cohort, region, good]``. Annual stock,
    inflow, and outflow use ``[year, region, good]``. Material stock and outflow
    use ``[year, region, material, good]``; carbon stock and outflow use
    ``[year, element, region, material, good]``. All amounts are in Mt of polymer
    or nominal element; only the C component of the nominal accounts is in Mt C.

    Args:
        state: Normalized REMIND-MFA cohort history and coordinate labels.
        compiled: Complete annual foreground parameter table.
        stock_target: Optional REMIND-MFA stock target calculated for population
            or GDP changes.
        transition_start_year: First year allowed to differ from the baseline.
        optimization_overlay: Whether infeasible inputs need an optimization
            diagnostic code.
        record_diagnostics: Include the ordinary annual transition ledger. Candidate calculations
        use the same cohort arrays and checks; set to False to omit the report.
        progress: Optional callback receiving the calculation stage and year.

    Returns:
        Annual inflow, stock, and outflow by material and cohort through 2100.
    """

    years = state.years
    future_years = tuple(range(_FIRST_PROSPECTIVE_YEAR, _FINAL_YEAR + 1))
    transition_i = years.index(transition_start_year)
    bounds = flodym_time_bounds(years)
    entry_time = 0.5 * (bounds[:-1] + bounds[1:])

    lifetime_mean = _input_cube(
        compiled,
        "lifetime_mean",
        years=future_years,
        goods=state.goods,
    )
    lifetime_std = _input_cube(
        compiled,
        "lifetime_std",
        years=future_years,
        goods=state.goods,
    )
    target = (
        _input_cube(
            compiled,
            "stock_projection",
            years=future_years,
            regions=state.regions,
            goods=state.goods,
        )
        if stock_target is None
        else np.asarray(stock_target, dtype=float)
    )
    if stock_target is not None:
        invalid_target = ~np.isfinite(target) | (target < 0.0)
        if invalid_target.any():
            year_i, region_i, good_i = np.argwhere(invalid_target)[0]
            raw_observed = float(target[year_i, region_i, good_i])
            observed = raw_observed if math.isfinite(raw_observed) else None
            code = "optimization_stock_target_infeasible" if optimization_overlay else None
            raise ScenarioFeasibilityError(
                f"stock_projection for {future_years[year_i]}, {state.regions[region_i]}, "
                f"{state.goods[good_i]} is {raw_observed!r}. Use a finite, nonnegative value "
                "in the scenario YAML.",
                code=code,
                diagnostic=(
                    None
                    if code is None
                    else OptimizationPhysicalInfeasibility(
                        code=code,
                        equation="stock_projection >= 0 and finite",
                        observed=observed,
                        bound=0.0,
                        tolerance=0.0,
                        shortfall=(None if observed is None else max(-observed, 0.0)),
                        status=(
                            "nonfinite_stock_target"
                            if observed is None
                            else "negative_stock_target"
                        ),
                        count=int(invalid_target.sum()),
                        coordinates=(
                            ("year", int(future_years[year_i])),
                            ("region", str(state.regions[region_i])),
                            ("good", str(state.goods[good_i])),
                        ),
                        details=(
                            ("observed_status", "nonfinite" if observed is None else "finite"),
                        ),
                    )
                ),
            )

    material_schedule = _input_cube(
        compiled,
        "plastic_family_share",
        years=future_years,
        regions=state.regions,
        materials=state.materials,
        goods=state.goods,
    )
    stock_by_cohort = np.zeros_like(state.stock_by_cohort)
    outflow_by_cohort = np.zeros_like(state.outflow_by_cohort)
    stock_by_cohort[:transition_i] = state.stock_by_cohort[:transition_i]
    outflow_by_cohort[:transition_i] = state.outflow_by_cohort[:transition_i]
    annual_stock = np.zeros_like(state.annual_stock)
    annual_inflow = np.zeros_like(state.annual_inflow)
    annual_outflow = np.zeros_like(state.annual_outflow)
    annual_stock[:transition_i] = state.annual_stock[:transition_i]
    annual_inflow[:transition_i] = state.annual_inflow[:transition_i]
    annual_outflow[:transition_i] = state.annual_outflow[:transition_i]

    cohort_material_shares = np.broadcast_to(
        state.material_shares[np.newaxis, ...],
        (len(years), *state.material_shares.shape),
    ).copy()
    cohort_material_shares[transition_i:] = material_schedule[
        transition_start_year - _FIRST_PROSPECTIVE_YEAR :
    ]

    ledger: list[dict[str, object]] = []
    for event_i in range(transition_i, len(years)):
        year = years[event_i]
        if progress is not None and (
            year == transition_start_year or year == _FINAL_YEAR or year % 10 == 0
        ):
            progress("cohort_continuation", year)
        schedule_i = year - _FIRST_PROSPECTIVE_YEAR
        opening = stock_by_cohort[event_i - 1, :event_i]
        existing_closing = np.zeros_like(opening)
        existing_outflow = np.zeros_like(opening)
        for good_i, good in enumerate(state.goods):
            mean = float(lifetime_mean[schedule_i, good_i])
            std = float(lifetime_std[schedule_i, good_i])
            ages_start = bounds[event_i] - entry_time[:event_i]
            ages_end = bounds[event_i + 1] - entry_time[:event_i]
            ratios = conditional_survival_ratio(
                ages_start,
                ages_end,
                mean=mean,
                std=std,
            )
            existing_closing[:, :, good_i] = opening[:, :, good_i] * ratios[:, np.newaxis]
            existing_outflow[:, :, good_i] = opening[:, :, good_i] - existing_closing[:, :, good_i]
        stock_by_cohort[event_i, :event_i] = existing_closing
        outflow_by_cohort[event_i, :event_i] = existing_outflow

        surviving = existing_closing.sum(axis=0)
        requested = target[schedule_i]
        residual = requested - surviving
        # Local binary64 convention for n nonnegative cohort products, their reduction and the
        # target difference. The allowance covers local arithmetic roundoff. Scale each coordinate
        # separately. Unit roundoff is eps/2.
        scaled_roundoff = (existing_closing.shape[0] + 2) * np.finfo(np.float64).eps / 2
        gamma = scaled_roundoff / (1.0 - scaled_roundoff)
        stock_tolerance = gamma * np.abs(requested) + gamma * np.abs(surviving)
        shortfall = residual < -stock_tolerance
        if np.any(shortfall):
            region_i, good_i = np.argwhere(shortfall)[0]
            requested_value = float(requested[region_i, good_i])
            surviving_value = float(surviving[region_i, good_i])
            code = "optimization_stock_below_surviving_cohort" if optimization_overlay else None
            raise ScenarioFeasibilityError(
                f"stock_projection for {state.regions[region_i]}, {state.goods[good_i]}, "
                f"{year} is {requested_value:.17g}, but {surviving_value:.17g} remains in "
                "use. Increase the stock target or revise the lifetime values; the package "
                "does not force surviving stock to disposal.",
                code=code,
                diagnostic=(
                    None
                    if code is None
                    else OptimizationPhysicalInfeasibility(
                        code=code,
                        equation="target_stock >= surviving_existing_stock",
                        observed=requested_value,
                        bound=surviving_value,
                        tolerance=float(stock_tolerance[region_i, good_i]),
                        shortfall=max(surviving_value - requested_value, 0.0),
                        status="stock_target_below_surviving_cohort",
                        count=int(shortfall.sum()),
                        coordinates=(
                            ("year", int(year)),
                            ("region", str(state.regions[region_i])),
                            ("good", str(state.goods[good_i])),
                        ),
                        details=(
                            ("requested_stock", requested_value),
                            ("surviving_stock", surviving_value),
                        ),
                    )
                ),
            )
        # An admitted negative residual is a numerical zero for NEW stock only.
        # Preserve every survivor and its outflow; the actual closing stock may
        # exceed its requested target within the recorded local roundoff bound.
        new_stock = np.where(residual < 0.0, 0.0, residual)
        inflow, same_year_outflow, new_closing = _solve_new_cohort_with_flodym(
            new_stock,
            year=year,
            regions=state.regions,
            goods=state.goods,
            lifetime_mean=lifetime_mean[schedule_i],
            lifetime_std=lifetime_std[schedule_i],
        )
        stock_by_cohort[event_i, event_i] = new_closing
        outflow_by_cohort[event_i, event_i] = same_year_outflow
        annual_stock[event_i] = stock_by_cohort[event_i].sum(axis=0)
        annual_inflow[event_i] = inflow
        annual_outflow[event_i] = outflow_by_cohort[event_i].sum(axis=0)
        if record_diagnostics:
            for region_i, region in enumerate(state.regions):
                for good_i, good in enumerate(state.goods):
                    ledger.append(
                        {
                            "year": year,
                            "region": region,
                            "good": good,
                            "opening_stock": float(opening[:, region_i, good_i].sum()),
                            "surviving_existing_stock": float(surviving[region_i, good_i]),
                            "new_inflow": float(inflow[region_i, good_i]),
                            "same_year_new_cohort_outflow": float(
                                outflow_by_cohort[event_i, event_i, region_i, good_i]
                            ),
                            "total_outflow": float(annual_outflow[event_i, region_i, good_i]),
                            "closing_stock": float(annual_stock[event_i, region_i, good_i]),
                            "stock_target": float(requested[region_i, good_i]),
                            "stock_target_residual": float(
                                requested[region_i, good_i]
                                - annual_stock[event_i, region_i, good_i]
                            ),
                            "stock_target_roundoff_tolerance": float(
                                stock_tolerance[region_i, good_i]
                            ),
                            "lifetime_mean": float(lifetime_mean[schedule_i, good_i]),
                            "lifetime_std": float(lifetime_std[schedule_i, good_i]),
                            "unit": "Mt",
                            "transition_method": "conditional_lognormal_frozen_history_v2",
                        }
                    )

    material_stock = np.einsum(
        "tcrg,crmg->trmg",
        stock_by_cohort,
        cohort_material_shares,
    )
    material_outflow = np.einsum(
        "tcrg,crmg->trmg",
        outflow_by_cohort,
        cohort_material_shares,
    )
    material_inflow = np.einsum(
        "trg,trmg->trmg",
        annual_inflow,
        cohort_material_shares,
    )
    carbon_stock = _nominal_element_masses(material_stock, state)
    carbon_outflow = _nominal_element_masses(material_outflow, state)
    carbon_inflow = _nominal_element_masses(material_inflow, state)
    return ProspectiveCohortResult(
        years=years,
        regions=state.regions,
        goods=state.goods,
        materials=state.materials,
        elements=state.elements,
        stock_by_cohort=stock_by_cohort,
        outflow_by_cohort=outflow_by_cohort,
        annual_stock=annual_stock,
        annual_inflow=annual_inflow,
        annual_outflow=annual_outflow,
        cohort_material_shares=cohort_material_shares,
        material_stock=material_stock,
        material_inflow=material_inflow,
        material_outflow=material_outflow,
        carbon_stock=carbon_stock,
        carbon_inflow=carbon_inflow,
        carbon_outflow=carbon_outflow,
        carbon_content=np.array(state.carbon_content, copy=True),
        transition_ledger=pd.DataFrame(ledger),
        material_identity_method="prospective_new_inflow_material_schedule",
    )
