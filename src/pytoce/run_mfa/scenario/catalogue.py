"""Foreground parameter resources and user-facing catalog contracts."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import PurePosixPath
from typing import Any, Literal, cast, overload

import pandas as pd

from .scenario_contract import ScenarioError

ParameterRole = Literal[
    "editable_input",
    "derived_input",
    "derived_output",
    "fixed_historical_boundary",
    "structural_configuration",
    "background_only_input",
    "unsupported",
]
ValueMode = Literal[
    "absolute",
    "retain_default",
    "multiplier_from_default",
    "delta_from_default",
    "retain_2019",
    "multiplier_from_2019",
    "delta_from_2019",
    "retain_2019_propensity_pretrade_collected",
    "retain_default_propensity_pretrade_collected",
    "absolute_parent_overlay",
]
_CATALOGUE_RESOURCE = "catalogues/foreground_parameter_catalogue.json"
_DEFAULT_INPUT_RESOURCE = (
    "workspace/data_raw/foreground_scenarios/default_inputs/default_input_values.parquet"
)
_COORDINATE_COLUMNS = ("region", "material", "good", "process", "route")


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    """Describe one foreground scenario parameter.

    Attributes:
        parameter_id: Name used in foreground scenario YAML files.
        upstream_name: Matching REMIND-MFA parameter, when one exists.
        role: Whether the value is editable, derived, fixed, structural, or unsupported.
        meaning: Plain-language scientific meaning.
        unit: Unit required for scenario values.
        dimensions: Ordered table dimensions, including year where applicable.
        coordinates: Allowed labels for each non-year dimension.
        value_modes: Allowed absolute, retained, multiplier, or difference forms.
        minimum: Smallest accepted value, or ``None`` when unbounded below.
        minimum_inclusive: Whether ``minimum`` itself is accepted.
        maximum: Largest accepted value, or ``None`` when unbounded above.
        continuous: Whether values between the listed bounds are accepted.
        driver_mode: How the parameter enters the MFA calculation.
        transition: How changes are applied through time.
        upstream_translation: How resolved scenario values enter their source or
            direct pyToCE calculation.
        recomputation_order: Calculations repeated after the parameter changes.
        coupled_equations: Equations that use the parameter.
        direct_dependencies: Inputs read directly when applying the parameter.
        invalidated_descendants: Derived values recalculated after a change.
        upstream_source: Named input or scientific-method source, when applicable.
        parameter_origin: Catalog provenance label for the parameter's input source.
        computation_origin: Catalog provenance label for the calculation owner.
        optimization_group: Optimization decision group, or ``None``.
        broad_selectors: Named selectors that may replace explicit coordinates.
    """

    parameter_id: str
    upstream_name: str | None
    role: ParameterRole
    meaning: str
    unit: str
    dimensions: tuple[str, ...]
    coordinates: Mapping[str, tuple[str, ...]]
    value_modes: tuple[ValueMode, ...]
    minimum: float | None
    minimum_inclusive: bool
    maximum: float | None
    continuous: bool
    driver_mode: str
    transition: str
    upstream_translation: str
    recomputation_order: tuple[str, ...]
    coupled_equations: tuple[str, ...]
    direct_dependencies: tuple[str, ...]
    invalidated_descendants: tuple[str, ...]
    upstream_source: str | None
    parameter_origin: str
    computation_origin: str
    optimization_group: str | None
    broad_selectors: tuple[str, ...]

    @property
    def editable(self) -> bool:
        """Return whether ordinary foreground scenarios may set this field."""

        return self.role == "editable_input"


def _resource(relative: str) -> Any:
    """Locate one packaged scenario catalog resource."""
    node = resources.files("pytoce.run_mfa.data")
    for part in PurePosixPath(relative).parts:
        node = node.joinpath(part)
    return node


def _catalogue_payload() -> dict[str, object]:
    """Read the packaged foreground parameter catalog."""
    payload = cast(
        dict[str, object],
        json.loads(_resource(_CATALOGUE_RESOURCE).read_text(encoding="utf-8")),
    )
    return payload


@lru_cache(maxsize=1)
def _waste_trade_authority() -> dict[tuple[str, str], tuple[float, float]]:
    """Read the packaged 2019 waste-trade propensities and source quantities."""
    resource = _resource("catalogues/waste_trade_2019_propensities.csv")
    with resource.open("rb") as handle:
        table = pd.read_csv(handle)
    result: dict[tuple[str, str], tuple[float, float]] = {}
    for row in table.itertuples(index=False):
        for parameter_id, source_column, propensity_column in (
            (
                "waste_imports",
                "source_input_waste_imports_mt_2019",
                "import_propensity_2019",
            ),
            (
                "waste_exports",
                "source_input_waste_exports_mt_2019",
                "export_propensity_2019",
            ),
        ):
            source = float(cast(Any, getattr(row, source_column)))
            propensity = float(cast(Any, getattr(row, propensity_column)))
            result[(parameter_id, str(row.region))] = (propensity, source)
    return result


def _definition(record: object, coordinate_sets: Mapping[str, object]) -> ParameterDefinition:
    """Convert one catalog record to a typed parameter definition.

    Coordinate labels are selected for the record's non-year dimensions. Units,
    bounds, value modes, upstream mapping, recomputation dependencies,
    optimization group, and allowed broad selectors retain their packaged definitions. The packaged
    catalog owns these definitions.
    """
    record = cast(dict[str, object], record)
    dimensions = tuple(str(value) for value in cast(Sequence[object], record["dimensions"]))
    coordinates = {
        dimension: tuple(str(value) for value in cast(Sequence[object], coordinate_sets[dimension]))
        for dimension in dimensions
        if dimension != "year"
    }
    return ParameterDefinition(
        parameter_id=str(record["parameter_id"]),
        upstream_name=(
            None if record.get("upstream_name") is None else str(record["upstream_name"])
        ),
        role=cast(ParameterRole, record["role"]),
        meaning=str(record["meaning"]),
        unit=str(record["unit"]),
        dimensions=dimensions,
        coordinates=coordinates,
        value_modes=tuple(
            cast(ValueMode, value)
            for value in cast(Sequence[object], record.get("value_modes", ()))
        ),
        minimum=(None if record.get("minimum") is None else float(cast(float, record["minimum"]))),
        minimum_inclusive=bool(record.get("minimum_inclusive", True)),
        maximum=(None if record.get("maximum") is None else float(cast(float, record["maximum"]))),
        continuous=bool(record.get("continuous", False)),
        driver_mode=str(record["driver_mode"]),
        transition=str(record["transition"]),
        upstream_translation=str(record["upstream_translation"]),
        recomputation_order=tuple(
            str(value) for value in cast(Sequence[object], record["recomputation_order"])
        ),
        coupled_equations=tuple(
            str(value) for value in cast(Sequence[object], record["coupled_equations"])
        ),
        direct_dependencies=tuple(
            str(value) for value in cast(Sequence[object], record["direct_dependencies"])
        ),
        invalidated_descendants=tuple(
            str(value) for value in cast(Sequence[object], record["invalidated_descendants"])
        ),
        upstream_source=(
            None if record.get("upstream_source") is None else str(record["upstream_source"])
        ),
        parameter_origin=str(record["parameter_origin"]),
        computation_origin=str(record["computation_origin"]),
        optimization_group=(
            None if record.get("optimization_group") is None else str(record["optimization_group"])
        ),
        broad_selectors=tuple(
            str(value) for value in cast(Sequence[object], record["broad_selectors"])
        ),
    )


@overload
def load_parameter_catalogue(parameter_id: None = None) -> tuple[ParameterDefinition, ...]:
    """Load all parameter definitions in stable catalog order."""
    ...


@overload
def load_parameter_catalogue(parameter_id: str) -> ParameterDefinition:
    """Load the single definition named by its exact parameter identifier."""
    ...


def load_parameter_catalogue(
    parameter_id: str | None = None,
) -> tuple[ParameterDefinition, ...] | ParameterDefinition:
    """Load the immutable parameter catalog.

    Args:
        parameter_id: Optional exact project-facing parameter identifier.

    Returns:
        All definitions in stable order, or one exact definition.

    Raises:
        ScenarioError: If the requested parameter is unknown.
    """

    payload = _catalogue_payload()
    coordinate_sets = cast(Mapping[str, object], payload["coordinate_sets"])
    definitions = tuple(
        _definition(record, coordinate_sets)
        for record in cast(Sequence[object], payload["parameters"])
    )
    if parameter_id is None:
        return definitions
    for definition in definitions:
        if definition.parameter_id == parameter_id:
            return definition
    raise ScenarioError(
        f"Unknown foreground parameter {parameter_id!r}. Choose from "
        f"{tuple(definition.parameter_id for definition in definitions)!r}."
    )


def _default_inputs() -> pd.DataFrame:
    """Read and order the packaged default scenario inputs."""
    with _resource(_DEFAULT_INPUT_RESOURCE).open("rb") as stream:
        frame = pd.read_parquet(stream)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype("int64")
    frame["default_value"] = pd.to_numeric(frame["default_value"], errors="raise").astype(float)
    return frame.sort_values(
        ["parameter_id", "year", *_COORDINATE_COLUMNS],
        kind="stable",
    ).reset_index(drop=True)
