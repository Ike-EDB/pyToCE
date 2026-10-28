"""Compile prospective foreground scenarios into explicit annual inputs.

The compiler checks constraints that depend on scenario inputs. The MFA model
calculates stocks, inflows, outflows, discard, production, treatment, and the
other derived quantities.
"""

import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from pytoce.shared.runtime.filesystem import (
    atomic_write,
    canonical_json_bytes,
    payload_sha256,
    replace_directory,
)
from pytoce.shared.runtime.workspace_guides import write_workspace_guide

from .catalogue import (
    ParameterDefinition,
    _catalogue_payload,
    _default_inputs,
    load_parameter_catalogue,
)
from .parent_relative import (
    compiled_scenario_table_identity,
)
from .rules import (
    _apply_rules,
    _load_scenario,
    _ParsedRules,
    _rules,
    _validate_driver_conflicts,
)
from .scenario_constraints import (
    _validate_bounds,
    _validate_feedstock_partition,
    _validate_material_partitions,
    _validate_treatment_partition,
)
from .scenario_contract import (
    ScenarioError,
    ScenarioEvaluation,
)

_SCENARIO_SCHEMA = "pytoce_foreground_scenario_schema_v1"
_COMPILED_SCHEMA = "pytoce_compiled_foreground_scenario_schema_v1"
_WASTE_TRADE_PROPENSITY_SHA256 = "ced41733bcbc7d748978622b718ac8f51cda7d35be303666f0996d602dc70960"
_DEFERRED_TRADE_MODE = "retain_2019_propensity_pretrade_collected"
_DEFAULT_DEFERRED_TRADE_MODE = "retain_default_propensity_pretrade_collected"
_SCENARIOS_RELATIVE = Path("data_raw/foreground_scenarios/scenarios")
_COMPILED_RELATIVE = Path("data_processed/foreground_scenarios")


@dataclass(frozen=True, slots=True)
class CompiledForegroundScenario:
    """Immutable compiled input parent consumed by ``run_mfa``."""

    foreground_scenario: str
    foreground_scenario_identity: str
    ssp_scenario: str
    status: Literal["computed", "reused"]
    result_directory: Path
    manifest_path: Path
    compiled_parameters_path: Path
    manifest: dict[str, object]
    manifest_sha256: str

    @property
    def normalized_change_points(self) -> tuple[dict[str, object], ...]:
        """Return both persistent rules and authored trajectory targets for reporting."""
        return _normalized_change_points(self.manifest)


def _normalized_change_points(manifest: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Combine the two normalized control forms shared by file and in-memory compilation."""
    normalized = cast(Mapping[str, object], manifest["normalized_scenario"])
    return tuple(
        point
        for name in ("change_points", "trajectories")
        for point in cast(list[dict[str, object]], normalized[name])
    )


def _scenario_identity(
    normalized: Mapping[str, object], catalogue_identity: str, table_sha256: str
) -> str:
    """Bind normalized controls, the parameter catalog and the canonical input table."""
    payload = {
        "schema_version": _COMPILED_SCHEMA,
        "foreground_scenario": normalized["foreground_scenario"],
        "ssp_scenario": "SSP2",
        "normalized_scenario": normalized,
        "parameter_catalogue_identity": catalogue_identity,
        "compiled_table_sha256": table_sha256,
    }
    return f"{_COMPILED_SCHEMA}::{payload_sha256(payload)}"


def _scenario_path(root: Path, foreground_scenario: str) -> Path:
    """Resolve one named foreground scenario YAML inside the active workspace."""
    discovered = {
        path.stem: path for path in (root / _SCENARIOS_RELATIVE).glob("*.yaml") if path.is_file()
    }
    path = discovered.get(foreground_scenario)
    if path is None:
        raise ScenarioError(
            f"Foreground scenario {foreground_scenario!r} is not available in "
            f"{root / _SCENARIOS_RELATIVE}. Choose one of {sorted(discovered)} or add a "
            "valid v1 YAML scenario with the same filename stem."
        )
    return path


def _normalized_scenario(
    payload: Mapping[str, object],
    parsed: _ParsedRules,
    definitions: Mapping[str, ParameterDefinition],
) -> dict[str, object]:
    """Build the deterministic scenario content used in its identity.

    Args:
        payload: Validated top-level scenario values.
        parsed: Parsed change points and trajectories with explicit coordinates.
        definitions: Parameter definitions indexed by parameter name.

    Returns:
        The v1 schema name, foreground and SSP names, expanded change points,
        trajectories, and any waste-trade source identity required by the rules.
        Equivalent selector forms use the same canonical representation.
    """

    change_points: list[dict[str, object]] = []
    for rule in parsed.rules:
        if rule.trajectory_target_year is not None:
            continue
        definition = definitions[rule.parameter_id]
        dimensions = tuple(dimension for dimension in definition.dimensions if dimension != "year")
        point: dict[str, object] = {
            "parameter": rule.parameter_id,
            "year": rule.year,
            "mode": rule.mode,
            "interpolation": rule.interpolation,
        }
        for dimension, coordinate in zip(dimensions, rule.coordinate, strict=True):
            point[dimension] = coordinate
        if rule.value is not None:
            point["value"] = rule.value
        change_points.append(point)
    trajectories: list[dict[str, object]] = []
    for trajectory in parsed.trajectories:
        point = {
            "mode": "linear_from_2019",
            "target_year": trajectory.target_year,
            "target_mode": trajectory.target_mode,
        }
        if trajectory.parameter_id is not None:
            point["parameter"] = trajectory.parameter_id
            definition = definitions[trajectory.parameter_id]
            dimensions = tuple(
                dimension for dimension in definition.dimensions if dimension != "year"
            )
            for dimension, coordinate in zip(dimensions, trajectory.coordinate, strict=True):
                point[dimension] = coordinate
            point["target_value"] = trajectory.target_value
        else:
            point["parameter_group"] = trajectory.parameter_group
            point["region"] = trajectory.coordinate[0]
            point["target_values"] = dict(trajectory.target_values)
        trajectories.append(point)
    normalized: dict[str, object] = {
        "schema_version": _SCENARIO_SCHEMA,
        "foreground_scenario": payload["foreground_scenario"],
        "ssp_scenario": "SSP2",
        "change_points": change_points,
        "trajectories": trajectories,
    }
    if any(rule.mode == _DEFERRED_TRADE_MODE for rule in parsed.rules):
        normalized["waste_trade_propensity_authority_sha256"] = _WASTE_TRADE_PROPENSITY_SHA256
    if any(rule.mode == _DEFAULT_DEFERRED_TRADE_MODE for rule in parsed.rules):
        normalized["waste_trade_propensity_basis"] = "contemporaneous_remind_mfa_default"
    return normalized


def _manifest(
    *,
    payload: Mapping[str, object],
    catalogue: Mapping[str, object],
    compiled: pd.DataFrame,
    parsed: _ParsedRules,
    definitions: Mapping[str, ParameterDefinition],
    source_path: Path,
) -> dict[str, object]:
    """Build the compiled scenario manifest and its input identities."""
    normalized = _normalized_scenario(payload, parsed, definitions)
    table_sha256 = compiled_scenario_table_identity(compiled)
    identity = _scenario_identity(
        normalized, cast(str, catalogue["catalogue_identity"]), table_sha256
    )
    return {
        "schema_version": _COMPILED_SCHEMA,
        "identity": identity,
        "foreground_scenario": payload["foreground_scenario"],
        "foreground_scenario_identity": identity,
        "ssp_scenario": "SSP2",
        "source_filename": source_path.name,
        "parameter_catalogue_identity": catalogue["catalogue_identity"],
        "compiled_table": {
            "path": "compiled_annual_input_parameters.parquet",
            "canonical_sha256": table_sha256,
            "row_count": len(compiled),
        },
        "normalized_scenario": normalized,
        "complete": True,
    }


def _compile_payload(
    payload: Mapping[str, object],
    *,
    source_path: Path,
) -> tuple[ScenarioEvaluation, dict[str, object]]:
    """Compile an admitted scenario mapping into its annual inputs and manifest."""

    foreground_scenario = str(payload["foreground_scenario"])
    catalogue = _catalogue_payload()
    definitions_tuple = cast(tuple[ParameterDefinition, ...], load_parameter_catalogue())
    definitions = {definition.parameter_id: definition for definition in definitions_tuple}
    defaults = _default_inputs()
    parsed = _rules(payload, definitions, defaults)
    _validate_driver_conflicts(parsed.rules)
    compiled = _apply_rules(defaults, definitions, parsed.rules)
    _validate_bounds(compiled, definitions)
    _validate_feedstock_partition(compiled)
    _validate_treatment_partition(
        compiled,
        [rule for rule in parsed.rules if rule.parameter_id == "landfill_share"],
    )
    _validate_material_partitions(compiled, parsed.rules, definitions)
    manifest = _manifest(
        payload=payload,
        catalogue=catalogue,
        compiled=compiled,
        parsed=parsed,
        definitions=definitions,
        source_path=source_path,
    )
    compiled_table = cast(Mapping[str, object], manifest["compiled_table"])
    evaluation = ScenarioEvaluation(
        foreground_scenario=foreground_scenario,
        foreground_scenario_identity=str(manifest["identity"]),
        ssp_scenario="SSP2",
        compiled_parameters=compiled,
        normalized_change_points=_normalized_change_points(manifest),
        compiled_table_sha256=cast(str, compiled_table["canonical_sha256"]),
    )
    return evaluation, manifest


def compile_foreground_scenario(
    workspace: Path,
    foreground_scenario: str,
    *,
    refresh: bool = False,
) -> CompiledForegroundScenario:
    """Compile one runnable foreground scenario into explicit 2020–2100 inputs.

    This internal operation validates all constraints decidable from user
    inputs. MFA execution separately checks the feasibility of the resulting flows.
    """

    root = workspace
    result = root / _COMPILED_RELATIVE / foreground_scenario
    manifest_path = result / "scenario_manifest.json"
    table_path = result / "compiled_annual_input_parameters.parquet"
    if result.exists() or result.is_symlink():
        if result.is_symlink() or not result.is_dir():
            raise ScenarioError(
                f"The compiled scenario path {result} is not a package result directory. "
                "Keep it unchanged and create a new workspace."
            )
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ScenarioError(
                f"The compiled scenario directory {result} has no valid completion manifest. "
                "Keep it unchanged and create a new workspace."
            )
    if manifest_path.is_file():
        if not refresh:
            content = manifest_path.read_bytes()
            existing = json.loads(content)
            return CompiledForegroundScenario(
                foreground_scenario=foreground_scenario,
                foreground_scenario_identity=str(existing["identity"]),
                ssp_scenario="SSP2",
                status="reused",
                result_directory=result,
                manifest_path=manifest_path,
                compiled_parameters_path=table_path,
                manifest=existing,
                manifest_sha256=sha256(content).hexdigest(),
            )
        if any(
            path.name
            not in {
                "README.md",
                "compiled_annual_input_parameters.parquet",
                "scenario_manifest.json",
            }
            or path.is_symlink()
            or not path.is_file()
            for path in result.iterdir()
        ):
            raise ScenarioError(
                f"Cannot refresh {result} because it is not the completed package result for "
                f"foreground_scenario={foreground_scenario!r}. Keep it unchanged and create a "
                "new workspace."
            )

    source_path = _scenario_path(root, foreground_scenario)
    payload = _load_scenario(source_path, foreground_scenario)
    evaluation, manifest = _compile_payload(payload, source_path=source_path)
    compiled = evaluation.compiled_parameters
    identity = str(manifest["identity"])
    manifest_bytes = canonical_json_bytes(manifest)
    result.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{foreground_scenario}.", dir=result.parent))
    try:
        compiled.to_parquet(
            staging / "compiled_annual_input_parameters.parquet",
            index=False,
            compression="zstd",
        )
        write_workspace_guide(
            staging,
            "pytoce.run_mfa",
            "compiled_scenario",
        )
        atomic_write(staging / "scenario_manifest.json", manifest_bytes)
        write_workspace_guide(
            result.parent,
            "pytoce.run_mfa",
            "compiled_scenarios",
        )
        replace_directory(
            staging,
            result,
            replace_existing=refresh,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return CompiledForegroundScenario(
        foreground_scenario=foreground_scenario,
        foreground_scenario_identity=identity,
        ssp_scenario="SSP2",
        status="computed",
        result_directory=result,
        manifest_path=manifest_path,
        compiled_parameters_path=table_path,
        manifest=manifest,
        manifest_sha256=sha256(manifest_bytes).hexdigest(),
    )
