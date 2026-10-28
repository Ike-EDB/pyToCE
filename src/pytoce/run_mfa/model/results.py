"""Identify, reuse and publish complete MFA results at their output owner."""

import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import cast

import pandas as pd

from pytoce.shared.runtime.filesystem import (
    atomic_write,
    canonical_json_bytes,
    canonical_table_sha256,
    file_sha256,
    payload_sha256,
    replace_directory,
)
from pytoce.shared.runtime.manifests import ItemStatus
from pytoce.shared.runtime.workspace_guides import write_workspace_guide

from ..remind_mfa.execution import RemindMFAExecutionRequest
from ..scenario.compiler import CompiledForegroundScenario
from .flows import DIRECT_MFA_METHOD, NOMINAL_FEEDSTOCK_METHOD
from .materials import MATERIAL_REALIZATION_METHOD
from .mfa import MFAComputation

_RESULT_SCHEMA = "pytoce_mfa_result_schema_v1"
_COMPUTATION_CONTRACT = DIRECT_MFA_METHOD
_COHORT_CONTRACT = "conditional_lognormal_frozen_history_v2"
_RESULT_SCHEMA_RESOURCE = "mfa_result_schema_v1.json"


def _result_contract() -> tuple[dict[str, object], str]:
    """Describe MFA result tables, units, and scientific identities."""
    resource = files("pytoce.run_mfa.data.schemas").joinpath(_RESULT_SCHEMA_RESOURCE)
    content = resource.read_bytes()
    payload = json.loads(content.decode("utf-8"))
    return cast(dict[str, object], payload), sha256(content).hexdigest()


@dataclass(frozen=True, slots=True)
class MFAResultTarget:
    """One requested immutable MFA identity and destination."""

    identity: str
    identity_payload: dict[str, object]
    result_directory: Path


def selected_mfa_result(
    workspace: Path,
    foreground_scenario: str,
) -> Path | None:
    """Select a completed result by its manifest path.

    Same-name editable scenario changes continue to require explicit refresh.
    """
    path = workspace / "results" / foreground_scenario / "mfa_results" / "manifest.json"
    if not path.is_file():
        if path.parent.exists() or path.parent.is_symlink():
            raise ValueError(
                f"The MFA result directory {path.parent} has no valid completion manifest. "
                "Keep it unchanged and create a new workspace."
            )
        return None
    return path


def validate_mfa_result_for_refresh(target: MFAResultTarget) -> None:
    """Preserve unowned files and links when replacing an MFA result.

    The visible directory may contain only the owner's artifacts, fixed guides
    and their parent directories.
    """

    result = target.result_directory
    contract, _digest = _result_contract()
    expected_artifacts = {
        str(item["path"]) for item in cast(list[dict[str, object]], contract["artifacts"])
    }
    expected_files = expected_artifacts | {
        "README.md",
        "manifest.json",
        "sankey/README.md",
    }
    expected_directories = {
        parent.as_posix()
        for filename in expected_files
        for parent in Path(filename).parents
        if parent != Path(".")
    }
    entries = tuple(result.rglob("*"))
    observed_entries = {path.relative_to(result).as_posix() for path in entries}
    if (
        result.is_symlink()
        or not result.is_dir()
        or not observed_entries.issubset(expected_files | expected_directories)
        or any(
            path.is_symlink()
            or (path.relative_to(result).as_posix() in expected_files and not path.is_file())
            or (path.relative_to(result).as_posix() in expected_directories and not path.is_dir())
            for path in entries
        )
    ):
        raise ValueError(
            f"Cannot refresh {result} because it is not the completed package MFA result for "
            f"foreground_scenario={target.identity_payload['foreground_scenario']!r}. Keep it "
            "unchanged and create a new workspace."
        )


def _scientific_software() -> dict[str, str]:
    """Record exact versions of the scientific libraries used by MFA."""
    packages = (
        "pytoce",
        "flodym",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
    )
    resolved: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        resolved[package] = version(package)
    return resolved


def build_mfa_result_target(
    workspace: Path,
    compiled: CompiledForegroundScenario,
    request: RemindMFAExecutionRequest,
    *,
    material_recipe_identity: str,
    material_composition_identity: str,
) -> MFAResultTarget:
    """Return the deterministic identity and directory for one MFA product."""

    source_manifest_sha256 = file_sha256(request.source.source_manifest)
    _, result_contract_sha256 = _result_contract()
    identity_payload: dict[str, object] = {
        "schema_version": _RESULT_SCHEMA,
        "foreground_scenario": compiled.foreground_scenario,
        "foreground_scenario_identity": compiled.foreground_scenario_identity,
        "compiled_scenario_manifest_sha256": compiled.manifest_sha256,
        "parameter_catalogue_identity": compiled.manifest["parameter_catalogue_identity"],
        "ssp_scenario": "SSP2",
        "remind_mfa": request.identity.as_dict(),
        "remind_mfa_source_manifest_sha256": source_manifest_sha256,
        "cohort_computation": _COHORT_CONTRACT,
        "material_recipe_identity": material_recipe_identity,
        "material_composition_identity": material_composition_identity,
        "material_realization_method": MATERIAL_REALIZATION_METHOD,
        "nominal_feedstock_method": NOMINAL_FEEDSTOCK_METHOD,
        "mfa_computation_contract": _COMPUTATION_CONTRACT,
        "result_contract_sha256": result_contract_sha256,
        "scientific_software": _scientific_software(),
        "reporting_start_year": 2019,
        "reporting_end_year": 2100,
    }
    identity = f"{_RESULT_SCHEMA}::{payload_sha256(identity_payload)}"
    result_directory = workspace / "results" / compiled.foreground_scenario / "mfa_results"
    return MFAResultTarget(
        identity=identity,
        identity_payload=identity_payload,
        result_directory=result_directory,
    )


def _write_table(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    """Write one Parquet table and return its schema, row count and identities."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "canonical_sha256": canonical_table_sha256(frame),
        "artifact_sha256": file_sha256(path),
    }


def _tables(computation: MFAComputation) -> tuple[tuple[str, pd.DataFrame], ...]:
    """Pair each MFA result table with its relative output path."""
    return (
        ("resolved_annual_input_parameters.parquet", computation.resolved_inputs),
        ("derived_annual_outputs.parquet", computation.derived_outputs),
        ("flows.parquet", computation.model_flows),
        ("stocks.parquet", computation.model_stocks),
        ("material_realization.parquet", computation.material_realization),
        ("nominal_feedstock_equivalents.parquet", computation.nominal_feedstock_equivalents),
        ("cohort_state.parquet", computation.cohort_state),
        ("cohort_material_identity.parquet", computation.cohort_material_identity),
        ("cohort_composition.parquet", computation.cohort_composition),
        ("physical_reconciliation_ledger.parquet", computation.reconciliation_ledger),
        ("stock_balance.parquet", computation.stock_balance),
        ("transition_ledger.parquet", computation.transition_ledger),
        ("waste_trade_propensity_ledger.parquet", computation.waste_trade_propensity_ledger),
        ("sankey/edges.parquet", computation.sankey_edges),
        ("sankey/nodes.parquet", computation.sankey_nodes),
    )


def _manifest_payload(
    target: MFAResultTarget,
    artifacts: dict[str, dict[str, object]],
    computation: MFAComputation,
) -> dict[str, object]:
    """Build the completed MFA manifest from inputs, identities, and artifacts."""
    return {
        "schema_version": _RESULT_SCHEMA,
        "identity": target.identity,
        "mfa_result_identity": target.identity,
        "foreground_scenario": target.identity_payload["foreground_scenario"],
        "foreground_scenario_identity": target.identity_payload["foreground_scenario_identity"],
        "ssp_scenario": "SSP2",
        "identity_payload": target.identity_payload,
        "scientific_authorities": {
            "physical_state_identity": computation.physical_state_identity,
            "material_identity_method": computation.material_identity_method,
            "waste_trade_propensity_authority_identities": (
                sorted(
                    set(
                        computation.waste_trade_propensity_ledger[
                            "import_propensity_authority_identity"
                        ]
                        .dropna()
                        .astype(str)
                    )
                    | set(
                        computation.waste_trade_propensity_ledger[
                            "export_propensity_authority_identity"
                        ]
                        .dropna()
                        .astype(str)
                    )
                )
                if not computation.waste_trade_propensity_ledger.empty
                else []
            ),
        },
        "artifacts": artifacts,
        "complete": True,
    }


def _write_staging_result(
    staging: Path,
    target: MFAResultTarget,
    computation: MFAComputation,
) -> None:
    """Write every MFA table and then the completion manifest."""
    artifacts: dict[str, dict[str, object]] = {}
    for relative, frame in _tables(computation):
        artifacts[relative] = _write_table(staging / relative, frame)
        artifacts[relative]["path"] = relative
    for directory, guide_id in (
        (staging, "mfa_result"),
        (staging / "sankey", "sankey"),
    ):
        write_workspace_guide(
            directory,
            "pytoce.run_mfa",
            guide_id,
        )
    payload = _manifest_payload(target, artifacts, computation)
    atomic_write(staging / "manifest.json", canonical_json_bytes(payload))


def write_mfa_result(
    target: MFAResultTarget,
    computation: MFAComputation,
    *,
    replacing: bool,
) -> ItemStatus:
    """Save one new or refreshed MFA result."""

    result = target.result_directory
    result.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mfa_result.", dir=result.parent))
    try:
        _write_staging_result(staging, target, computation)
        write_workspace_guide(
            result.parent,
            "pytoce.run_mfa",
            "foreground_result",
        )
        replace_directory(staging, result, replace_existing=replacing)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return "updated" if replacing else "computed"
