"""Trajectory parsing, application, and checks available before MFA execution."""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import yaml

from .catalogue import (
    ParameterDefinition,
    ValueMode,
    _resource,
    _waste_trade_authority,
)
from .scenario_contract import (
    ScenarioConflictError,
    ScenarioError,
    _UniqueKeyLoader,
    validate_foreground_scenario_name,
)

TrajectoryTargetMode = Literal["absolute", "multiplier_from_2019", "delta_from_2019"]
_SCENARIO_SCHEMA = "pytoce_foreground_scenario_schema_v1"
_DEFAULT_SCENARIO_RESOURCE = (
    "workspace/data_raw/foreground_scenarios/scenarios/remind_mfa_default.yaml"
)
_DEFERRED_TRADE_MODE = "retain_2019_propensity_pretrade_collected"
_DEFAULT_DEFERRED_TRADE_MODE = "retain_default_propensity_pretrade_collected"
OPTIMIZATION_PARENT_OVERLAY_ORIGIN = "optimization_parent_overlay"
_OPTIMIZATION_PARENT_OVERLAY_MODE = "absolute_parent_overlay"
_COORDINATE_COLUMNS = ("region", "material", "good", "process", "route")
_COMPILED_COLUMNS = (
    "year",
    *_COORDINATE_COLUMNS,
    "parameter_id",
    "unit",
    "parameter_role",
    "resolved_value",
    "value_mode",
    "value_origin",
    "controlling_change_point_year",
    "default_value",
    "reference_value",
    "deferred_propensity",
    "deferred_denominator_id",
    "direct_dependencies",
    "invalidated_descendants",
)
_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "foreground_scenario", "ssp_scenario", "description", "changes"}
)
_PARAMETER_ENTRY_KEYS = frozenset(
    {
        "parameter",
        "parameter_group",
        "selector",
        "region",
        "regions",
        "material",
        "materials",
        "good",
        "goods",
        "process",
        "processes",
        "route",
        "routes",
        "changes",
        "trajectory",
    }
)
_CHANGE_POINT_KEYS = frozenset({"year", "mode", "value", "interpolation"})
_TRAJECTORY_KEYS = frozenset(
    {"mode", "target_year", "target_mode", "target_value", "target_values"}
)
_TRAJECTORY_TARGET_MODES = frozenset({"absolute", "multiplier_from_2019", "delta_from_2019"})
_TREATMENT_GROUP = "collected_waste_treatment_shares"
_FEEDSTOCK_GROUP = "virgin_feedstock_shares"
_TRAJECTORY_GROUP_MEMBERS: Mapping[str, tuple[str, ...]] = {
    _TREATMENT_GROUP: (
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
        "landfill_share",
    ),
    _FEEDSTOCK_GROUP: ("fossil_share", "bio_production_rate", "daccu_production_rate"),
}
_TRAJECTORY_GROUP_CONTROLS: Mapping[str, tuple[str, ...]] = {
    _TREATMENT_GROUP: (
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
    ),
    _FEEDSTOCK_GROUP: ("bio_production_rate", "daccu_production_rate"),
}
_COORDINATE_SELECTOR_KEYS = frozenset(
    key for dimension in _COORDINATE_COLUMNS for key in (dimension, f"{dimension}s")
)
_YEARS = tuple(range(2020, 2101))


@dataclass(frozen=True, slots=True)
class _Rule:
    """One dated parameter rule with explicit coordinates and a value mode."""

    parameter_id: str
    coordinate: tuple[object, ...]
    year: int
    mode: ValueMode
    value: float | None
    interpolation: str
    trajectory_target_year: int | None = None
    trajectory_target_mode: TrajectoryTargetMode | None = None


@dataclass(frozen=True, slots=True)
class _Trajectory:
    """One scalar or grouped trajectory from its 2019 value to a target year."""

    parameter_id: str | None
    parameter_group: str | None
    coordinate: tuple[object, ...]
    target_year: int
    target_mode: TrajectoryTargetMode
    target_value: float | None
    target_values: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class _ParsedRules:
    """Annual parameter rules and the trajectories from which they were calculated."""

    rules: tuple[_Rule, ...]
    trajectories: tuple[_Trajectory, ...]


def _validate_scenario_payload(
    value: object,
    *,
    expected_name: str,
) -> dict[str, object]:
    """Validate one parsed scenario file or in-memory scenario mapping."""

    if not isinstance(value, dict):
        raise ScenarioError("A foreground scenario must be a YAML mapping")
    if unknown := set(value).difference(_TOP_LEVEL_KEYS):
        raise ScenarioError(f"Unsupported top-level scenario fields: {sorted(unknown)}")
    schema = value.get("schema_version")
    if schema != _SCENARIO_SCHEMA:
        raise ScenarioError(
            f"Foreground scenario uses schema {schema!r}; expected {_SCENARIO_SCHEMA!r}. "
            "Create a new v1 scenario from the installed template."
        )
    foreground_scenario = value.get("foreground_scenario")
    if not isinstance(foreground_scenario, str):
        raise ScenarioError("foreground_scenario must be text")
    validate_foreground_scenario_name(foreground_scenario)
    if foreground_scenario != expected_name:
        raise ScenarioError("The readable filename stem must equal the foreground_scenario field")
    if value.get("ssp_scenario") != "SSP2":
        raise ScenarioError("The only accepted ssp_scenario is 'SSP2'")
    if "description" in value and not isinstance(value["description"], str):
        raise ScenarioError("Foreground scenario description must be text")
    changes = value.get("changes")
    if not isinstance(changes, list):
        raise ScenarioError("Foreground scenario changes must be a list")
    if foreground_scenario == "remind_mfa_default":
        packaged = yaml.load(
            _resource(_DEFAULT_SCENARIO_RESOURCE).read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
        if value != packaged:
            raise ScenarioError(
                "remind_mfa_default is reserved for the unmodified packaged default"
            )
    return cast(dict[str, object], value)


def _load_scenario(path: Path, expected_name: str) -> dict[str, object]:
    """Load the scenario from its owned source."""
    try:
        value = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ScenarioError(
            f"Foreground scenario {expected_name!r} at {path} cannot be read as UTF-8 YAML. "
            "Correct the file or replace it from data_raw/foreground_scenarios/templates, "
            "then run the workflow again."
        ) from exc
    return _validate_scenario_payload(value, expected_name=expected_name)


def _coordinate_values(
    entry: Mapping[str, object],
    definition: ParameterDefinition,
    *,
    mode: ValueMode,
) -> tuple[tuple[object, ...], ...]:
    """Expand one scenario selector to explicit parameter coordinates.

    Args:
        entry: Parsed scenario rule containing a named selector or explicit
            singular or plural coordinate fields.
        definition: Parameter dimensions and allowed labels.
        mode: Value mode, used to prevent an absolute stock or mass from being
            spread implicitly across all regions and goods.

    Returns:
        Coordinates in parameter-dimension order and stable selector order.

    Raises:
        ScenarioError: If a field, selector, coordinate, or selector combination
            is not accepted for this parameter.
        ScenarioConflictError: If singular and plural fields are combined or a
            coordinate appears more than once.
    """
    non_year = tuple(dimension for dimension in definition.dimensions if dimension != "year")
    provided_coordinates = _COORDINATE_SELECTOR_KEYS.intersection(entry)
    allowed_coordinates = {key for dimension in non_year for key in (dimension, f"{dimension}s")}
    if unsupported := provided_coordinates.difference(allowed_coordinates):
        raise ScenarioError(
            f"Parameter {definition.parameter_id!r} does not accept coordinate "
            f"fields {sorted(unsupported)}. Use only {sorted(allowed_coordinates)}."
        )
    selector = entry.get("selector")
    selected: dict[str, tuple[str, ...]]
    if selector is not None:
        selector_name = str(selector)
        if selector_name not in definition.broad_selectors:
            raise ScenarioError(
                f"Selector {selector_name!r} is not valid for {definition.parameter_id!r}. "
                f"Choose from {definition.broad_selectors!r}, or specify each coordinate."
            )
        selector_dimensions = {
            "uniform_all_h12": ("region",),
            "uniform_all_goods": ("good",),
            "uniform_all_h12_goods": ("region", "good"),
        }
        if selector_name == "uniform_all_h12_goods" and mode == "absolute":
            raise ScenarioError(
                "Absolute stock or mass values require coordinate-level regional "
                "allocation; uniform_all_h12_goods is relative-mode only"
            )
        selected = {
            dimension: definition.coordinates[dimension]
            for dimension in selector_dimensions[selector_name]
        }
        if provided_coordinates:
            raise ScenarioError(
                f"Selector {selector_name!r} cannot be combined with {sorted(provided_coordinates)}"
            )
    else:
        selected = {}
        for dimension in non_year:
            plural = f"{dimension}s"
            singular_value = entry.get(dimension)
            plural_value = entry.get(plural)
            if singular_value is not None and plural_value is not None:
                raise ScenarioConflictError(f"Specify either {dimension} or {plural}, not both")
            if plural_value is not None:
                if not isinstance(plural_value, list) or not plural_value:
                    raise ScenarioError(f"{plural} must be a non-empty list")
                values = tuple(str(value) for value in plural_value)
            elif singular_value is not None:
                values = (str(singular_value),)
            else:
                raise ScenarioError(
                    f"Parameter {definition.parameter_id!r} requires explicit {dimension}"
                )
            allowed = definition.coordinates[dimension]
            unknown = sorted(set(values).difference(allowed))
            if unknown:
                raise ScenarioError(
                    f"Parameter {definition.parameter_id!r} has unknown {dimension}: {unknown}. "
                    f"Choose from {allowed!r}."
                )
            if len(values) != len(set(values)):
                raise ScenarioConflictError(f"{dimension} selector contains duplicates")
            selected[dimension] = values
    coordinates: list[tuple[object, ...]] = [()]
    for dimension in non_year:
        values = selected[dimension]
        coordinates = [(*prefix, value) for prefix in coordinates for value in values]
    return tuple(coordinates)


def _parse_change(
    value: object,
    *,
    definition: ParameterDefinition,
) -> tuple[int, ValueMode, float | None, str]:
    """Parse one scenario change point for its parameter definition."""
    if not isinstance(value, dict):
        raise ScenarioError("Each scenario change point must be a mapping")
    if unknown := set(value).difference(_CHANGE_POINT_KEYS):
        raise ScenarioError(f"Unsupported change-point fields: {sorted(unknown)}")
    raw_year = value.get("year")
    if isinstance(raw_year, bool) or not isinstance(raw_year, int):
        raise ScenarioError("Every scenario change point requires an integer year")
    year = raw_year
    if year not in _YEARS:
        raise ScenarioError("Ordinary foreground change points must be in 2020–2100")
    mode = cast(ValueMode, value.get("mode"))
    if mode not in definition.value_modes:
        raise ScenarioError(
            f"Mode {mode!r} is not valid for {definition.parameter_id!r}. Choose from "
            f"{definition.value_modes!r}."
        )
    has_value = "value" in value
    raw = value.get("value")
    if mode in {
        "absolute",
        "multiplier_from_default",
        "delta_from_default",
        "multiplier_from_2019",
        "delta_from_2019",
    }:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ScenarioError(f"Mode {mode!r} requires a finite numeric value")
        try:
            numeric = float(raw)
        except OverflowError as exc:
            raise ScenarioError(f"Mode {mode!r} requires a finite numeric value") from exc
        if not math.isfinite(numeric):
            raise ScenarioError(f"Mode {mode!r} requires a finite numeric value")
    else:
        if has_value:
            raise ScenarioError(
                f"Mode {mode!r} does not accept value; omit the field rather than using null"
            )
        numeric = None
    interpolation = str(value.get("interpolation", "step"))
    if interpolation != "step":
        raise ScenarioError(
            "Ordinary change points are stepwise; use a 'linear_from_2019' "
            "trajectory for a linear target"
        )
    return year, mode, numeric, interpolation


def _reference_2019_lookup(
    defaults: pd.DataFrame,
    definitions: Mapping[str, ParameterDefinition],
) -> dict[tuple[str, tuple[object, ...]], float]:
    """Index the 2019 baseline values by their complete parameter coordinates."""
    lookup: dict[tuple[str, tuple[object, ...]], float] = {}
    for _, row in defaults.loc[defaults["year"].eq(2019)].iterrows():
        parameter_id = str(row["parameter_id"])
        definition = definitions[parameter_id]
        lookup[
            (
                parameter_id,
                _coordinate_key(cast(Mapping[str, object], row.to_dict()), definition),
            )
        ] = float(cast(float, row["default_value"]))
    return lookup


def _trajectory_year(value: object) -> int:
    """Read and validate one prospective trajectory year."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in _YEARS:
        raise ScenarioError("trajectory.target_year must be an integer from 2020 through 2100")
    return value


def _trajectory_numeric(value: object, *, field: str) -> float:
    """Read one finite numeric trajectory value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{field} must be a finite numeric value")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ScenarioError(f"{field} must be a finite numeric value") from exc
    if not math.isfinite(numeric):
        raise ScenarioError(f"{field} must be a finite numeric value")
    return numeric


def _trajectory_target(
    *,
    reference_2019: float,
    target_mode: TrajectoryTargetMode,
    target_value: float,
) -> float:
    """Resolve an absolute target from an absolute or 2019-relative value."""
    if target_mode == "absolute":
        return target_value
    if target_mode == "multiplier_from_2019":
        return reference_2019 * target_value
    return reference_2019 + target_value


def _trajectory_value(
    *,
    reference_2019: float,
    target: float,
    target_year: int,
    year: int,
) -> float:
    """Interpolate from the 2019 reference, holding the target after its year."""
    weight = min((year - 2019) / (target_year - 2019), 1.0)
    return reference_2019 + weight * (target - reference_2019)


def _rules(
    payload: Mapping[str, object],
    definitions: Mapping[str, ParameterDefinition],
    defaults: pd.DataFrame,
) -> _ParsedRules:
    """Parse foreground scenario changes into annual MFA parameter rules.

    An individual parameter accepts dated change points or a supported trajectory.
    Registered groups of share parameters accept only a ``linear_from_2019``
    trajectory with ``uniform_all_h12``. Selectors are expanded to exact parameter
    coordinates. Grouped shares are kept together so their annual sum can be
    checked, and a parameter cannot be set both by a group and by an individual
    entry.

    Args:
        payload: Scenario YAML after validation against the v1 schema.
        definitions: MFA parameter definitions indexed by parameter name.
        defaults: Complete annual parameter table for the baseline scenario.

    Returns:
        Annual rules and normalized scalar or grouped trajectory records ready
        to apply to the baseline table.

    Raises:
        ScenarioError: If an entry, parameter, selector, year, or value is invalid.
        ScenarioConflictError: If two entries set the same parameter coordinate
            and year, or mix grouped and individual changes for one parameter.
    """
    output: list[_Rule] = []
    trajectories: list[_Trajectory] = []
    seen: set[tuple[str, tuple[object, ...], int]] = set()
    entries = cast(Sequence[object], payload["changes"])
    group_claims: set[str] = set()
    individual_parameters: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ScenarioError("Every foreground change entry must be a mapping")
        if "parameter_group" in entry:
            group = str(entry["parameter_group"])
            members = _TRAJECTORY_GROUP_MEMBERS.get(group)
            if members is None:
                raise ScenarioError(
                    f"Unknown foreground parameter group {group!r}. Choose from "
                    f"{tuple(_TRAJECTORY_GROUP_MEMBERS)!r}."
                )
            group_claims.update(members)
        if "parameter" in entry:
            individual_parameters.add(str(entry["parameter"]))
    if overlap := group_claims.intersection(individual_parameters):
        raise ScenarioConflictError(
            f"A grouped trajectory cannot be combined with individual rules for {sorted(overlap)}"
        )

    references = _reference_2019_lookup(defaults, definitions)

    def append_rule(rule: _Rule) -> None:
        """Add a rule after rejecting duplicate parameter, coordinate and year combinations."""
        key = (rule.parameter_id, rule.coordinate, rule.year)
        if key in seen:
            raise ScenarioConflictError(
                "Duplicate or overlapping change point for "
                f"{rule.parameter_id!r}, coordinate={rule.coordinate!r}, year={rule.year}"
            )
        seen.add(key)
        output.append(rule)

    for raw_entry in entries:
        entry = cast(dict[str, object], raw_entry)
        if unknown := set(entry).difference(_PARAMETER_ENTRY_KEYS):
            raise ScenarioError(f"Unsupported parameter-entry fields: {sorted(unknown)}")
        has_parameter = "parameter" in entry
        has_group = "parameter_group" in entry
        if has_parameter == has_group:
            raise ScenarioError("Specify exactly one of 'parameter' or 'parameter_group'")
        has_changes = "changes" in entry
        has_trajectory = "trajectory" in entry
        if has_changes == has_trajectory:
            raise ScenarioError("Specify exactly one of 'changes' or 'trajectory'")

        if has_group:
            group = str(entry["parameter_group"])
            if set(entry).difference({"parameter_group", "selector", "trajectory"}):
                raise ScenarioError(
                    "Grouped trajectories accept only parameter_group, selector, and trajectory"
                )
            if entry.get("selector") != "uniform_all_h12":
                raise ScenarioError("Grouped trajectories require selector 'uniform_all_h12'")
            raw_trajectory = entry["trajectory"]
            if not isinstance(raw_trajectory, dict):
                raise ScenarioError("trajectory must be a mapping")
            if unknown := set(raw_trajectory).difference(_TRAJECTORY_KEYS):
                raise ScenarioError(f"Unsupported trajectory fields: {sorted(unknown)}")
            if raw_trajectory.get("mode") != "linear_from_2019":
                raise ScenarioError("trajectory.mode must be 'linear_from_2019'")
            if "target_mode" in raw_trajectory or "target_value" in raw_trajectory:
                raise ScenarioError("Grouped trajectories require explicit target_values")
            target_year = _trajectory_year(raw_trajectory.get("target_year"))
            raw_targets = raw_trajectory.get("target_values")
            if not isinstance(raw_targets, dict):
                raise ScenarioError("Grouped trajectories require target_values")
            members = _TRAJECTORY_GROUP_MEMBERS[group]
            if set(raw_targets) != set(members):
                missing = sorted(set(members).difference(raw_targets))
                extra = sorted(set(raw_targets).difference(members))
                raise ScenarioError(
                    f"{group} requires every exact member; missing={missing}, extra={extra}"
                )
            targets = tuple(
                (member, _trajectory_numeric(raw_targets[member], field=f"{group}.{member}"))
                for member in members
            )
            values = np.asarray([value for _member, value in targets], dtype=np.float64)
            if np.any(values < 0.0) or not np.isclose(
                float(values.sum()), 1.0, rtol=0.0, atol=1e-12
            ):
                raise ScenarioError(f"{group} target shares must be nonnegative and sum to one")
            target_lookup = dict(targets)
            controls = _TRAJECTORY_GROUP_CONTROLS[group]
            regions = definitions[controls[0]].coordinates["region"]
            for region in regions:
                region_coordinate = (region,)
                start_controls = {
                    parameter_id: references[(parameter_id, region_coordinate)]
                    for parameter_id in controls
                }
                trajectories.append(
                    _Trajectory(
                        parameter_id=None,
                        parameter_group=group,
                        coordinate=region_coordinate,
                        target_year=target_year,
                        target_mode="absolute",
                        target_value=None,
                        target_values=targets,
                    )
                )
                for parameter_id in controls:
                    reference = start_controls[parameter_id]
                    target = target_lookup[parameter_id]
                    for year in _YEARS:
                        append_rule(
                            _Rule(
                                parameter_id=parameter_id,
                                coordinate=region_coordinate,
                                year=year,
                                mode="absolute",
                                value=_trajectory_value(
                                    reference_2019=reference,
                                    target=target,
                                    target_year=target_year,
                                    year=year,
                                ),
                                interpolation="linear_from_2019",
                                trajectory_target_year=target_year,
                                trajectory_target_mode="absolute",
                            )
                        )
            continue

        parameter_id = str(entry["parameter"])
        definition = definitions.get(parameter_id)
        if definition is None:
            raise ScenarioError(
                f"Unknown foreground parameter {parameter_id!r}. Choose an editable parameter "
                "from data_raw/foreground_scenarios/default_inputs/parameter_catalogue.json."
            )
        if not definition.editable and parameter_id != "landfill_share":
            if parameter_id == "lifetime_std":
                raise ScenarioError(
                    "lifetime_std is derived from lifetime_mean to preserve each end-use "
                    "group's original coefficient of variation. Remove the lifetime_std entry "
                    "and change lifetime_mean only."
                )
            raise ScenarioError(
                f"Parameter {parameter_id!r} has role {definition.role!r} and is not editable"
            )
        if has_changes:
            changes = entry["changes"]
            if not isinstance(changes, list) or not changes:
                raise ScenarioError(f"Parameter {parameter_id!r} requires change points")
            for change in changes:
                year, mode, numeric, interpolation = _parse_change(change, definition=definition)
                coordinates = _coordinate_values(entry, definition, mode=mode)
                for coordinate in coordinates:
                    append_rule(
                        _Rule(
                            parameter_id=parameter_id,
                            coordinate=coordinate,
                            year=year,
                            mode=mode,
                            value=numeric,
                            interpolation=interpolation,
                        )
                    )
            continue

        if parameter_id in {"waste_imports", "waste_exports", "plastic_family_share"}:
            raise ScenarioError(f"Parameter {parameter_id!r} does not support trajectories")
        if not definition.editable:
            raise ScenarioError(
                "Parameter 'landfill_share' is calculated from the other treatment shares "
                "and cannot use an independent trajectory. Use "
                "parameter_group='collected_waste_treatment_shares' and provide all four "
                "treatment shares."
            )
        raw_trajectory = entry["trajectory"]
        if not isinstance(raw_trajectory, dict):
            raise ScenarioError("trajectory must be a mapping")
        if unknown := set(raw_trajectory).difference(_TRAJECTORY_KEYS):
            raise ScenarioError(f"Unsupported trajectory fields: {sorted(unknown)}")
        if raw_trajectory.get("mode") != "linear_from_2019":
            raise ScenarioError("trajectory.mode must be 'linear_from_2019'")
        if "target_values" in raw_trajectory:
            raise ScenarioError("Scalar trajectories require target_value")
        target_year = _trajectory_year(raw_trajectory.get("target_year"))
        target_mode = raw_trajectory.get("target_mode")
        if (
            not isinstance(target_mode, str)
            or target_mode not in _TRAJECTORY_TARGET_MODES
            or target_mode not in definition.value_modes
        ):
            accepted_target_modes = tuple(
                mode for mode in _TRAJECTORY_TARGET_MODES if mode in definition.value_modes
            )
            raise ScenarioError(
                f"trajectory.target_mode {target_mode!r} is not valid for {parameter_id!r}. "
                f"Choose from {accepted_target_modes!r}."
            )
        target_mode = cast(TrajectoryTargetMode, target_mode)
        target_value = _trajectory_numeric(
            raw_trajectory.get("target_value"), field="trajectory.target_value"
        )
        coordinates = _coordinate_values(entry, definition, mode=cast(ValueMode, target_mode))
        for coordinate in coordinates:
            reference = references[(parameter_id, coordinate)]
            target = _trajectory_target(
                reference_2019=reference,
                target_mode=target_mode,
                target_value=target_value,
            )
            trajectories.append(
                _Trajectory(
                    parameter_id=parameter_id,
                    parameter_group=None,
                    coordinate=coordinate,
                    target_year=target_year,
                    target_mode=target_mode,
                    target_value=target_value,
                    target_values=(),
                )
            )
            for year in _YEARS:
                append_rule(
                    _Rule(
                        parameter_id=parameter_id,
                        coordinate=coordinate,
                        year=year,
                        mode="absolute",
                        value=_trajectory_value(
                            reference_2019=reference,
                            target=target,
                            target_year=target_year,
                            year=year,
                        ),
                        interpolation="linear_from_2019",
                        trajectory_target_year=target_year,
                        trajectory_target_mode=target_mode,
                    )
                )

    rules = tuple(
        sorted(
            output,
            key=lambda rule: (
                rule.parameter_id,
                tuple(str(value) for value in rule.coordinate),
                rule.year,
            ),
        )
    )
    canonical_trajectories = tuple(
        sorted(
            trajectories,
            key=lambda trajectory: (
                trajectory.parameter_group or "",
                trajectory.parameter_id or "",
                tuple(str(value) for value in trajectory.coordinate),
            ),
        )
    )
    return _ParsedRules(rules=rules, trajectories=canonical_trajectories)


def _validate_driver_conflicts(rules: Sequence[_Rule]) -> None:
    """Reject simultaneous rules that assign the same trajectory driver."""
    changed = {rule.parameter_id for rule in rules}
    if "stock_projection" in changed and changed.intersection({"population", "gdppc"}):
        raise ScenarioConflictError(
            "stock_projection cannot be changed together with population or gdppc"
        )


def _rule_value(
    *,
    rule: _Rule,
    default_value: float,
    reference_2019: float,
) -> float:
    """Resolve one rule to its numeric value for a specific year."""
    if rule.mode == "absolute":
        return cast(float, rule.value)
    if rule.mode == "retain_default":
        return default_value
    if rule.mode == "multiplier_from_default":
        return cast(float, rule.value) * default_value
    if rule.mode == "delta_from_default":
        return default_value + cast(float, rule.value)
    if rule.mode == "retain_2019":
        return reference_2019
    if rule.mode == "multiplier_from_2019":
        return cast(float, rule.value) * reference_2019
    return reference_2019 + cast(float, rule.value)


def _coordinate_key(
    row: Mapping[str, object],
    definition: ParameterDefinition,
) -> tuple[object, ...]:
    """Read one table row's coordinates in parameter-definition order."""
    return tuple(row[dimension] for dimension in definition.dimensions if dimension != "year")


def _apply_rules(
    defaults: pd.DataFrame,
    definitions: Mapping[str, ParameterDefinition],
    rules: Sequence[_Rule],
) -> pd.DataFrame:
    """Apply validated change points and trajectories to annual default values.

    Args:
        defaults: Complete annual parameter table, including 2019 references.
        definitions: Parameter definitions indexed by parameter name.
        rules: Expanded rules with explicit coordinates and change years.

    Returns:
        Annual parameter table for 2020 through 2100. A rule applies until the
        next change point for the same parameter and coordinates. Absolute,
        retained, multiplier, and difference modes use either the annual
        default or the 2019 value specified by the parameter definition.
        ``linear_from_2019`` trajectories interpolate from the 2019 value to
        the requested value and retain that value afterward. Waste-trade propensity
        rules calculate traded mass after local collection is known.
    """
    editable = defaults.loc[defaults["year"].isin(_YEARS)].copy()
    editable["parameter_role"] = editable["parameter_id"].map(
        lambda value: definitions[str(value)].role
    )
    editable["resolved_value"] = editable["default_value"]
    editable["value_mode"] = "retain_default"
    editable["value_origin"] = "remind_mfa_default"
    editable["controlling_change_point_year"] = pd.array([pd.NA] * len(editable), dtype="Int64")
    editable["reference_value"] = editable["default_value"]
    editable["deferred_propensity"] = np.nan
    editable["deferred_denominator_id"] = None
    by_parameter_coordinate: dict[tuple[str, tuple[object, ...]], list[_Rule]] = defaultdict(list)
    for rule in rules:
        if rule.parameter_id != "landfill_share":
            by_parameter_coordinate[(rule.parameter_id, rule.coordinate)].append(rule)
    reference_lookup = _reference_2019_lookup(defaults, definitions)
    for (parameter_id, coordinate), coordinate_rules in by_parameter_coordinate.items():
        definition = definitions[parameter_id]
        non_year = tuple(dimension for dimension in definition.dimensions if dimension != "year")
        mask = editable["parameter_id"].eq(parameter_id)
        for dimension, coordinate_value in zip(non_year, coordinate, strict=True):
            mask &= pd.Series(
                editable[dimension].to_numpy(dtype=object) == coordinate_value,
                index=editable.index,
            )
        rows = editable.loc[mask].sort_values("year")
        reference_key = (parameter_id, coordinate)
        reference_2019 = reference_lookup[reference_key]
        ordered_rules = sorted(coordinate_rules, key=lambda item: item.year)
        # Each change point controls the years up to the next change for the
        # same parameter and coordinates.
        for position, rule in enumerate(ordered_rules):
            end = (
                ordered_rules[position + 1].year - 1 if position + 1 < len(ordered_rules) else 2100
            )
            years = list(range(rule.year, end + 1))
            target_index = rows.loc[rows["year"].isin(years)].index
            source_2019 = reference_2019
            if rule.mode in {_DEFERRED_TRADE_MODE, _DEFAULT_DEFERRED_TRADE_MODE}:
                resolved = [np.nan] * len(target_index)
                if rule.mode == _DEFERRED_TRADE_MODE:
                    propensity, source_2019 = _waste_trade_authority()[
                        (parameter_id, str(coordinate[0]))
                    ]
                    editable.loc[target_index, "deferred_propensity"] = propensity
                editable.loc[target_index, "deferred_denominator_id"] = (
                    "local_pretrade_collected_waste"
                )
            else:
                resolved = [
                    _rule_value(
                        rule=rule,
                        default_value=float(default_value),
                        reference_2019=reference_2019,
                    )
                    for default_value in editable.loc[target_index, "default_value"]
                ]
            editable.loc[target_index, "resolved_value"] = resolved
            editable.loc[target_index, "value_mode"] = (
                f"linear_from_2019_{rule.trajectory_target_mode}"
                if rule.interpolation == "linear_from_2019"
                else rule.mode
            )
            editable.loc[target_index, "value_origin"] = "user_foreground_scenario"
            editable.loc[target_index, "controlling_change_point_year"] = (
                rule.trajectory_target_year
                if rule.trajectory_target_year is not None
                else rule.year
            )
            if rule.mode in {_DEFERRED_TRADE_MODE, _DEFAULT_DEFERRED_TRADE_MODE}:
                editable.loc[target_index, "reference_value"] = source_2019
            else:
                editable.loc[target_index, "reference_value"] = (
                    reference_2019
                    if rule.mode.endswith("_2019") or rule.trajectory_target_year is not None
                    else editable.loc[target_index, "default_value"]
                )
    editable["direct_dependencies"] = editable["parameter_id"].map(
        lambda value: "|".join(definitions[str(value)].direct_dependencies)
    )
    editable["invalidated_descendants"] = editable["parameter_id"].map(
        lambda value: "|".join(definitions[str(value)].invalidated_descendants)
    )
    output = cast(pd.DataFrame, editable.loc[:, _COMPILED_COLUMNS])
    _derive_lifetime_std(output)
    return output.sort_values(
        ["parameter_id", "year", *_COORDINATE_COLUMNS], kind="stable"
    ).reset_index(drop=True)


def _derive_lifetime_std(compiled: pd.DataFrame) -> None:
    """Derive lifetime spread in years before either scenario identity is built.

    For each year and good, std = default_std * (mean / default_mean).
    The original defaults remain the reference even for a parent-relative
    optimization overlay, so each extension uses the same reference ratio.
    Unchanged means retain their exact default spread. When the derived value changes, its origin
    and change-year metadata follow the mean.
    """

    mean = compiled.loc[compiled["parameter_id"].eq("lifetime_mean")].set_index(["year", "good"])
    spread = compiled.loc[compiled["parameter_id"].eq("lifetime_std")]
    coordinates = pd.MultiIndex.from_frame(spread.loc[:, ["year", "good"]])
    mean = mean.reindex(coordinates)
    default_std = spread["default_value"].to_numpy(dtype=np.float64)
    default_mean = mean["default_value"].to_numpy(dtype=np.float64)
    resolved = default_std * (mean["resolved_value"].to_numpy(dtype=np.float64) / default_mean)
    changed = resolved != spread["resolved_value"].to_numpy(dtype=np.float64)
    indices = spread.index[changed]
    compiled.loc[indices, "resolved_value"] = resolved[changed]
    compiled.loc[indices, "value_mode"] = "derived_from_mean"
    compiled.loc[indices, "value_origin"] = mean["value_origin"].to_numpy()[changed]
    compiled.loc[indices, "controlling_change_point_year"] = mean[
        "controlling_change_point_year"
    ].to_numpy()[changed]
    compiled.loc[indices, "reference_value"] = (
        default_std * (mean["reference_value"].to_numpy(dtype=np.float64) / default_mean)
    )[changed]
