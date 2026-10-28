"""Construct fixed-lifetime demand repairs and environmental endpoint neighbors."""

from bisect import bisect_left
from dataclasses import replace
from fractions import Fraction
from itertools import accumulate, product

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from pytoce.run_mfa.model.cohorts import FixedLifetimeStockSystem

from .interventions import (
    AnnualControl,
    DemandLifetimeDecision,
)

STOCK_PROPOSAL_METHOD_IDENTITY = "pytoce_annual_stock_proposal_v6"
DEMAND_REPAIR_METHOD_IDENTITY = "pytoce_linear_first_demand_repair_v3"


def initial_demand_lifetime_decisions(
    *, target_year: int, step_pct: int = 20
) -> tuple[DemandLifetimeDecision, ...]:
    """Construct linear grid starts, including 90% demand reduction."""
    demand_levels = tuple(dict.fromkeys((*range(0, 91, step_pct), 90)))
    return tuple(
        DemandLifetimeDecision(demand, lifetime, target_year)
        for demand, lifetime in product(demand_levels, range(0, 101, step_pct))
    )


class AnnualDemandProposalError(RuntimeError):
    """A conditional stock solve failed to produce a usable proposal or infeasibility status."""


def propose_annual_demand(
    system: FixedLifetimeStockSystem,
    *,
    parent_stock: np.ndarray,
    demand_reduction_pct: int,
    lifetime_extension_pct: int,
    lifetime_fraction: AnnualControl,
    target_year: int,
) -> DemandLifetimeDecision | None:
    """Minimize demand departure from the linear reference at fixed lifetime.

    Nonnegative annual increments and the nonnegative target-year remainder
    parameterize exactly the monotone demand domain. All 2020--2100 region/end-use
    constraints remain present. Annual epigraph variables describe absolute
    control deviations, as control deviations. Explicit solver infeasibility
    means no proposal in the submitted fixed-control formulation; its scope is that submitted
    formulation. Other
    failures leave the search incomplete. Full MFA
    admission follows every returned path, including a numerical stock boundary.
    """
    endpoint = demand_reduction_pct / 100.0
    free_years = target_year - 2020
    right = parent_stock - system.inherited_stock
    right[free_years:] = (1.0 - endpoint) * parent_stock[free_years:] - system.inherited_stock[
        free_years:
    ]
    derivative = np.zeros((*parent_stock.shape, free_years))
    for year_i in range(free_years):
        derivative[year_i, :, :, year_i] = parent_stock[year_i]
    h = np.zeros_like(right)
    k = np.zeros_like(derivative)
    for year_i in range(len(system.years)):
        survival = system.closing_survival[year_i, :year_i]
        h[year_i] = right[year_i] - np.einsum("cg,crg->rg", survival, h[:year_i])
        k[year_i] = derivative[year_i] - np.einsum("cg,crgd->rgd", survival, k[:year_i])
    physical_rows = parent_stock.size
    constraints = np.zeros((physical_rows + 1 + 2 * free_years, 2 * free_years))
    # d_j = sum(q_i for i <= j), so K C uses reverse column sums. Bound
    # increments directly instead of relying on tolerances in d_j <= d_(j+1).
    stock_matrix = np.cumsum(k[..., ::-1], axis=-1)[..., ::-1]
    constraints[:physical_rows, :free_years] = stock_matrix.reshape(physical_rows, free_years)
    constraints[physical_rows, :free_years] = 1.0
    cumulative_matrix = np.tril(np.ones((free_years, free_years)))
    deviations = physical_rows + 1
    constraints[deviations : deviations + free_years, :free_years] = cumulative_matrix
    constraints[deviations + free_years :, :free_years] = -cumulative_matrix
    constraints[deviations : deviations + free_years, free_years:] = -np.eye(free_years)
    constraints[deviations + free_years :, free_years:] = -np.eye(free_years)
    reference = np.array(
        [
            float(value)
            for value in DemandLifetimeDecision(
                demand_reduction_pct, 0, target_year
            ).demand_fraction[1 : free_years + 1]
        ]
    )
    bounds = np.concatenate((h.ravel(), [endpoint], reference, -reference))
    objective = np.concatenate((np.zeros(free_years), np.ones(free_years)))
    # Nonnegative increments summing to at most the endpoint already imply
    # q_i <= endpoint. State these equivalent bounds for HiGHS;
    # leave epigraph variables unbounded above and retain every original row.
    solution = optimize.linprog(
        objective,
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(0.0, endpoint)] * free_years + [(0.0, None)] * free_years,
        method="highs-ds",
    )
    # SciPy 1.13.1 also maps a HiGHS model error to status 2. Only its
    # explicit public infeasible message permits omitting this stock proposal.
    if solution.status == 2 and solution.message.startswith("The problem is infeasible. "):
        return None
    if solution.status != 0:
        raise AnnualDemandProposalError(
            f"Annual demand proposal at D{demand_reduction_pct}/L{lifetime_extension_pct} "
            f"did not complete: solver status {solution.status}: {solution.message}"
        )
    values = np.asarray(solution.x, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise AnnualDemandProposalError(
            "The annual demand solver returned a nonfinite or negative increment/deviation; "
            f"minimum={values.min()!r}. "
            "No rounding or repair was applied."
        )
    cumulative = tuple(
        accumulate(Fraction.from_float(float(value)) for value in values[:free_years])
    )
    exact_endpoint = Fraction.from_float(endpoint)
    remainder = exact_endpoint - cumulative[-1]
    if remainder < 0:
        raise AnnualDemandProposalError(
            f"The annual demand solver exceeded its endpoint by {float(-remainder)!r}; "
            "no rounding or repair was applied."
        )
    return DemandLifetimeDecision(
        demand_reduction_pct,
        lifetime_extension_pct,
        target_year,
        (Fraction(0), *cumulative, *((exact_endpoint,) * (2101 - target_year))),
        lifetime_fraction,
    )


def demand_path_departure(decision: DemandLifetimeDecision) -> Fraction:
    """Return mean absolute demand departure over 2020 to T-1, in percentage points."""
    reference = DemandLifetimeDecision(
        decision.demand_reduction_pct, decision.lifetime_extension_pct, decision.target_year
    )
    end = decision.target_year - 2019
    return (
        100
        * sum(
            (
                abs(value - expected)
                for value, expected in zip(
                    decision.demand_fraction[1:end], reference.demand_fraction[1:end], strict=True
                )
            ),
            Fraction(0),
        )
        / (end - 1)
    )


def endpoint_neighbors(
    decision: DemandLifetimeDecision, *, step_pct: int, coordinated: bool = False
) -> tuple[DemandLifetimeDecision, ...]:
    """Construct fresh linear paths at scalar and optional joint endpoint neighbors."""
    moves: tuple[tuple[int, int], ...] = (
        (-step_pct, 0),
        (step_pct, 0),
        (0, -step_pct),
        (0, step_pct),
    )
    if coordinated:
        moves += tuple(
            (left, right) for left in (-step_pct, step_pct) for right in (-step_pct, step_pct)
        )
    return tuple(
        sorted(
            (
                DemandLifetimeDecision(demand, lifetime, decision.target_year)
                for demand_step, lifetime_step in moves
                if 0 <= (demand := decision.demand_reduction_pct + demand_step) <= 90
                and 0 <= (lifetime := decision.lifetime_extension_pct + lifetime_step) <= 100
            ),
            key=lambda item: item.coordinates(),
        )
    )


def _move_demand(
    decision: DemandLifetimeDecision,
    levels: tuple[Fraction, ...],
    moves: tuple[tuple[int, int], ...],
) -> DemandLifetimeDecision | None:
    """Apply adjacent demand levels with minimal monotone control propagation."""
    values = decision.demand_fraction
    targets: dict[int, Fraction] = {}
    for index, direction in moves:
        position = bisect_left(levels, values[index]) + direction
        if not 0 <= position < len(levels):
            return None
        targets[index] = levels[position]
    end = decision.target_year - 2019
    points = sorted(((0, values[0]), *targets.items(), (end, values[end])))
    if any(left[1] > right[1] for left, right in zip(points, points[1:])):
        return None
    updated = list(values)
    for (left, lower), (right, upper) in zip(points, points[1:]):
        updated[left] = lower
        for index in range(left + 1, right):
            updated[index] = min(max(values[index], lower), upper)
        updated[right] = upper
    return replace(decision, demand_fraction=tuple(updated))


def demand_repair_neighbors(
    decision: DemandLifetimeDecision, *, failure_year: int
) -> tuple[DemandLifetimeDecision, ...]:
    """Order one immutable scalar/binding-year demand pool by exact departure.

    Other controls are unchanged. The pool has at most 2n + 4(n-1) members
    for n adjustable years. The declared pool remains fixed during evaluation.
    """
    reference = DemandLifetimeDecision(
        decision.demand_reduction_pct, decision.lifetime_extension_pct, decision.target_year
    )
    levels = tuple(
        sorted(
            {
                *(
                    Fraction.from_float(value / 100.0)
                    for value in range(decision.demand_reduction_pct + 1)
                ),
                *reference.demand_fraction,
                *decision.demand_fraction,
            }
        )
    )
    end = decision.target_year - 2019
    binding = min(failure_year, decision.target_year - 1) - 2019
    moves: list[tuple[tuple[int, int], ...]] = [
        ((index, direction),) for index, direction in product(range(1, end), (-1, 1))
    ]
    moves.extend(
        ((earlier, first), (binding, second))
        for earlier, first, second in product(range(1, binding), (-1, 1), (-1, 1))
    )
    choices = {
        changed for move in moves if (changed := _move_demand(decision, levels, move)) is not None
    }
    choices.discard(decision)
    return tuple(
        sorted(choices, key=lambda item: (demand_path_departure(item), item.coordinates()))
    )
