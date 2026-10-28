"""Allocate mechanical recycling jointly with chemical and receiving capacities.

All masses are exact rational Mt inside this annual material calculation. The
strictly convex projection changes the waste-proportional reference as little
as possible while preserving requested regional treatment quantities. Its objective
is the squared distance from the waste-proportional allocation.
"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class _MaterialLimit:
    """One inequality, coefficients times mechanical inputs >= minimum."""

    name: str
    coefficients: tuple[Fraction, ...]
    minimum: Fraction


class RecyclingCapacityError(ValueError):
    """A requested allocation contradicts the combined material-capacity limits."""

    def __init__(self, constraint: str, required: Fraction, maximum: Fraction) -> None:
        """Retain the exact incompatible bound for the public material diagnostic."""
        self.constraint = constraint
        self.required = required
        self.maximum = maximum
        super().__init__(constraint)


def _dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
    """Sum exact products, omitting zero and unit-coefficient multiplications.

    Most receiving-limit normals are signed unit rows. Their exact arithmetic
    needs only addition or subtraction; nonunit coefficients keep the same
    rational product. Consume every pair to retain strict dimension checking.
    """
    total = Fraction()
    for coefficient, value in zip(left, right, strict=True):
        if not coefficient or not value:
            continue
        if coefficient == 1:
            total += value
        elif coefficient == -1:
            total -= value
        else:
            total += coefficient * value
    return total


def _working_direction(
    normals: list[tuple[Fraction, ...]], incoming: tuple[Fraction, ...]
) -> tuple[list[Fraction], tuple[Fraction, ...]]:
    """Resolve a new limit along the current active constraints.

    Independent active normals make their Gram matrix positive definite. Exact
    elimination therefore has positive pivots; a zero pivot is a programming
    error. MFA infeasibility is reported by the allocation constraints. The identity objective
    Hessian needs no scientific-library solver or numerical tolerance.
    """
    rows = [[_dot(a, b) for b in normals] + [_dot(a, incoming)] for a in normals]
    count = len(rows)
    for column in range(count):
        pivot = rows[column][column]
        for row in range(column + 1, count):
            factor = rows[row][column] / pivot
            for position in range(column + 1, count + 1):
                rows[row][position] -= factor * rows[column][position]
    reduction = [Fraction() for _ in normals]
    for row in range(count - 1, -1, -1):
        reduction[row] = (
            rows[row][-1]
            - sum(rows[row][column] * reduction[column] for column in range(row + 1, count))
        ) / rows[row][row]
    direction = tuple(
        value - sum(row[j] * amount for row, amount in zip(normals, reduction, strict=True))
        for j, value in enumerate(incoming)
    )
    return reduction, direction


def _project_inputs(
    reference: tuple[Fraction, ...],
    total: Fraction,
    limits: tuple[_MaterialLimit, ...],
) -> tuple[Fraction, ...]:
    """Find the unique Euclidean projection or prove contradictory capacities.

    The Goldfarb-Idnani dual active-set steps specialize to an identity Hessian
    and one permanent total-input equality. Its multiplier is unrestricted;
    inequality multipliers remain nonnegative. Exact partial steps preserve
    the incoming multiplier while releasing an active limit. See Schreppel
    and Brembeck (2020), doi:10.3390/computation8040088, Section 2, Table 2.
    """
    count = len(reference)
    normals = [tuple(Fraction(1) for _ in reference)]
    multipliers = [(total - sum(reference)) / count]
    point = tuple(value + multipliers[0] for value in reference)
    while True:
        violated = next(
            (
                i
                for i, limit in enumerate(limits)
                if _dot(limit.coefficients, point) < limit.minimum
            ),
            None,
        )
        if violated is None:
            return point
        limit = limits[violated]
        incoming = limit.coefficients
        incoming_multiplier = Fraction()
        while True:
            reduction, direction = _working_direction(normals, incoming)
            squared = _dot(incoming, direction)
            steps: list[tuple[Fraction, int | None]] = []
            if squared:
                steps.append(((limit.minimum - _dot(incoming, point)) / squared, None))
            steps.extend(
                (multipliers[j] / amount, j)
                for j, amount in enumerate(reduction)
                if j > 0 and amount > 0
            )
            if not steps:
                # The incoming normal is a combination of active normals with
                # nonpositive inequality weights. Their bounds prove that its
                # required minimum exceeds every feasible value.
                raise RecyclingCapacityError(limit.name, limit.minimum, _dot(incoming, point))
            # Stable minimum selects the primal step on a tie, then the first
            # limiting active inequality in deterministic insertion order.
            step, removed = min(steps, key=lambda option: option[0])
            point = tuple(
                value + step * delta for value, delta in zip(point, direction, strict=True)
            )
            multipliers = [
                value - step * delta for value, delta in zip(multipliers, reduction, strict=True)
            ]
            incoming_multiplier += step
            if removed is None:
                normals.append(incoming)
                multipliers.append(incoming_multiplier)
                break
            del normals[removed], multipliers[removed]


def mechanical_allocation(
    *,
    waste: tuple[Fraction, ...],
    recovery: tuple[Fraction, ...],
    receiving_capacity: tuple[Fraction, ...],
    chemical_eligible: tuple[bool, ...],
    monomer_use: tuple[tuple[Fraction, ...], tuple[Fraction, ...]],
    monomer_headroom: tuple[Fraction, Fraction],
    mechanical_total: Fraction,
    chemical_total: Fraction,
) -> tuple[Fraction, ...]:
    """Preserve treatment totals while respecting usable-resin and monomer demand.

    ``recovery`` is usable dry resin per dry input; zero denotes a constituent
    excluded from this receiving function. Receiving capacities are gross
    local requirements after unavoidable net primary imports. Monomer headroom
    is unreduced new-polymer monomer demand minus chemical monomer supply.
    Source-dependent coefficients remain separate from this allocation rule.

    Chemical-compatible constituents are a subset of mechanically compatible
    ones. Their common chemical recipe makes remaining eligible waste a
    sufficient chemical-input constraint. A changed source-dependent chemical
    yield would require a different equation.
    """
    chemical_capacity = sum(
        (amount for amount, eligible in zip(waste, chemical_eligible, strict=True) if eligible),
        Fraction(),
    )
    if chemical_total > chemical_capacity:
        raise RecyclingCapacityError("chemical feed", chemical_total, chemical_capacity)
    for index, capacity in enumerate(receiving_capacity):
        if capacity < 0:
            raise RecyclingCapacityError(f"net primary imports {index}", -capacity, Fraction())
    for name, headroom in zip(
        ("ethylene absorption", "propylene absorption"), monomer_headroom, strict=True
    ):
        if headroom < 0:
            raise RecyclingCapacityError(name, -headroom, Fraction())
    selected = tuple(i for i, eta in enumerate(recovery) if eta > 0)
    if not selected:
        if mechanical_total > 0:
            raise RecyclingCapacityError("compatible mechanical feed", mechanical_total, Fraction())
        return tuple(Fraction() for _ in waste)
    reference_capacity = tuple(
        waste[i] * (1 - chemical_total / chemical_capacity)
        if chemical_eligible[i] and chemical_total > 0
        else waste[i]
        for i in selected
    )
    available = sum(reference_capacity, Fraction())
    if mechanical_total > available:
        raise RecyclingCapacityError("compatible mechanical feed", mechanical_total, available)
    reference = tuple(
        amount * mechanical_total / available if available else Fraction()
        for amount in reference_capacity
    )
    limits = []
    for position, index in enumerate(selected):
        row = tuple(Fraction(position == j) for j in range(len(selected)))
        upper = min(waste[index], receiving_capacity[index] / recovery[index])
        limits.extend(
            (
                _MaterialLimit(f"nonnegative mechanical input {index}", row, Fraction()),
                _MaterialLimit(f"waste/receiving capacity {index}", tuple(-v for v in row), -upper),
            )
        )
    for name, coefficients, headroom in zip(
        ("ethylene absorption", "propylene absorption"), monomer_use, monomer_headroom, strict=True
    ):
        limits.append(
            _MaterialLimit(name, tuple(-coefficients[i] * recovery[i] for i in selected), -headroom)
        )
    limits.append(
        _MaterialLimit(
            "reserved chemical feed",
            tuple(-Fraction(chemical_eligible[i]) for i in selected),
            chemical_total - chemical_capacity,
        )
    )
    allocation = _project_inputs(reference, mechanical_total, tuple(limits))
    amounts = [Fraction() for _ in waste]
    for index, amount in zip(selected, allocation, strict=True):
        amounts[index] = amount
    return tuple(amounts)
