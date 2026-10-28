"""Scenario errors, YAML admission, and compiled scenario records."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import pandas as pd
import yaml

OptimizationDiagnosticScalar: TypeAlias = str | int | float | bool | None
# Absolute input precision; shares are retained.
MATERIAL_SHARE_SUM_TOLERANCE = 1e-12
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_foreground_scenario_name(foreground_scenario: str) -> str:
    """Return a lowercase snake case scenario name."""

    if not _IDENTIFIER.fullmatch(foreground_scenario):
        raise ScenarioError("foreground_scenario must be a lowercase snake_case filename stem")
    return foreground_scenario


class ScenarioError(ValueError):
    """A foreground scenario is invalid or unavailable."""


class ScenarioConflictError(ScenarioError):
    """Foreground scenario rules conflict at the same coordinate."""


@dataclass(frozen=True, slots=True)
class OptimizationPhysicalInfeasibility:
    """Describe why a candidate foreground scenario is infeasible in the MFA.

    Attributes:
        code: Stable identifier for the failed condition.
        equation: Equation or relation that was checked.
        observed: Calculated value, or ``None`` when no scalar value applies.
        bound: Required bound, or ``None`` when the check is not a bound.
        tolerance: Numerical tolerance applied in the equation's unit.
        shortfall: Distance from feasibility, or ``None`` when not applicable.
        status: Direct description of the failed condition.
        count: Number of affected rows or coordinates.
        coordinates: Coordinate labels for the first or summarized failure.
        details: Additional scalar values needed to interpret the failure.
    """

    code: str
    equation: str
    observed: float | None
    bound: float | None
    tolerance: float
    shortfall: float | None
    status: str
    count: int
    coordinates: tuple[tuple[str, OptimizationDiagnosticScalar], ...]
    details: tuple[tuple[str, OptimizationDiagnosticScalar], ...] = ()


class ScenarioFeasibilityError(ScenarioError):
    """A requested foreground scenario is infeasible in the MFA."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: Mapping[str, object] | None = None,
        diagnostic: OptimizationPhysicalInfeasibility | None = None,
    ) -> None:
        """Attach a stable failure code and optional diagnostic to the error."""
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})
        self.diagnostic = diagnostic


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    """Build a mapping and reject duplicate keys."""
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "foreground scenario mapping keys must be text",
                key_node.start_mark,
            )
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


@dataclass(frozen=True, slots=True)
class ScenarioEvaluation:
    """Admitted scenario inputs and their existing canonical table identity.

    The compiler or completed-manifest loader supplies ``compiled_table_sha256``.
    Owners and consumers treat the table and normalized change points as read-only
    after admission. An overlay derives a new table and record while the parent's table and identity
    remain fixed.
    """

    foreground_scenario: str
    foreground_scenario_identity: str
    ssp_scenario: str
    compiled_parameters: pd.DataFrame
    normalized_change_points: tuple[dict[str, object], ...]
    compiled_table_sha256: str


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)
