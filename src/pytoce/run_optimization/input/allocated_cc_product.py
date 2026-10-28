"""Compile, identify, and save allocated carrying capacity results."""

import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from pytoce.shared.runtime.file_lock import FileLockError, writer_lock
from pytoce.shared.runtime.filesystem import (
    atomic_write,
    canonical_json_bytes,
    canonical_table_sha256,
    payload_sha256,
    replace_directory,
)

from .allocated_cc_contract import (
    _IDENTITY_PREFIX,
    _LCIA_SELECTION_PREFIX,
    _METHOD_COLUMNS,
    _METHOD_COMPATIBILITY_PREFIX,
    _PROCESSED_RELATIVE,
    _SELECTION_PREFIX,
    _YEARS,
    AllocatedCCError,
    AllocatedCCResult,
    FloatArray,
    _artifact,
    _method_authority,
    _readonly_array,
    validate_allocated_cc_frames,
)
from .allocated_cc_paths import (
    _ALLOCATED_CC_SCHEMA_VERSION,
    _allocated_cc_metadata,
    _read_csv,
    _readable_name,
    _secure_source_directory,
    _secure_source_file,
)

_PRODUCT_SCHEMA_VERSION = "pytoce_allocated_cc_product_schema_v1"


def _secure_processed_result_directory(workspace: Path, allocated_cc_result: str) -> Path:
    """Return a canonical processed-result path."""

    workspace_root = workspace.resolve(strict=True)
    data_processed = workspace_root / _PROCESSED_RELATIVE.parts[0]
    data_processed.mkdir(exist_ok=True)
    processed_root = data_processed.resolve(strict=True)
    if processed_root.parent != workspace_root or processed_root.name != "data_processed":
        raise AllocatedCCError("The data_processed directory escapes the workspace")

    allocated_root = processed_root / _PROCESSED_RELATIVE.parts[1]
    allocated_root.mkdir(exist_ok=True)
    allocated_root = allocated_root.resolve(strict=True)
    if allocated_root.parent != processed_root or allocated_root.name != "allocated_cc":
        raise AllocatedCCError("The processed allocated_cc directory escapes data_processed")

    result = allocated_root / allocated_cc_result
    if result.exists() or result.is_symlink():
        resolved = result.resolve(strict=True)
        if (
            not resolved.is_dir()
            or resolved.parent != allocated_root
            or resolved.name != allocated_cc_result
        ):
            raise AllocatedCCError("The processed allocated_cc result must be a real direct child")
        return resolved
    return result


def _selection_table_identities(
    annual: pd.DataFrame,
    cumulative: pd.DataFrame,
    methods: pd.DataFrame,
    cumulative_method: pd.DataFrame,
) -> dict[str, str]:
    """Hash the selected aCC tables and LCIA method registries."""
    return {
        "annual_table_identity": canonical_table_sha256(annual),
        "cumulative_gwp100_table_identity": canonical_table_sha256(cumulative),
        "method_registry_identity": canonical_table_sha256(methods),
        "cumulative_climate_method_registry_identity": canonical_table_sha256(cumulative_method),
        "method_compatibility_identity": optimization_lcia_method_compatibility_identity(methods),
        "cumulative_climate_compatibility_identity": (
            optimization_lcia_method_compatibility_identity(cumulative_method)
        ),
    }


def optimization_lcia_method_compatibility_identity(frame: pd.DataFrame) -> str:
    """Identify LCIA methods."""

    signature_columns = tuple(column for column in _METHOD_COLUMNS if column != "selection_order")
    signatures = frame.loc[:, list(signature_columns)].astype("string")
    signatures = signatures.sort_values(list(signature_columns), kind="stable", ignore_index=True)
    return _METHOD_COMPATIBILITY_PREFIX + payload_sha256(
        signatures.astype(str).to_dict(orient="records")
    )


def _selection_identities(
    *,
    result_identity: str,
    cases: tuple[str, ...],
    table_identities: Mapping[str, str],
) -> tuple[str, str]:
    """Identify the EF v3.1 aCC cases and selected tables."""
    method_payload = {"lcia_method": ["ef31"]}
    requested_identity = _LCIA_SELECTION_PREFIX + payload_sha256(method_payload)
    selection_identity = _SELECTION_PREFIX + payload_sha256(
        {
            "allocated_cc_result_identity": result_identity,
            "optimization_lcia_selection": method_payload,
            "allocated_cc_cases": list(cases),
            **table_identities,
        }
    )
    return requested_identity, selection_identity


def _compile_selection_arrays(
    annual: pd.DataFrame,
    cumulative: pd.DataFrame,
    methods: pd.DataFrame,
    cases: tuple[str, ...],
) -> tuple[
    FloatArray,
    FloatArray,
    tuple[tuple[str, str], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Compile canonical selected tables once for vectorized candidate division."""

    categories = tuple(
        (str(row.lcia_method), str(row.impact_category_id))
        for row in methods.itertuples(index=False)
    )
    annual_shape = (len(cases), 12, len(categories), len(_YEARS))
    annual_values = annual["allocated_cc"].to_numpy(dtype=np.float64).reshape(annual_shape)
    annual_values = _readonly_array(annual_values.transpose(0, 1, 3, 2))
    cumulative_values = _readonly_array(
        cumulative["cumulative_allocated_cc"].to_numpy(dtype=np.float64).reshape(len(cases), 12)
    )
    annual_case_identities = tuple(
        canonical_table_sha256(
            annual.loc[annual["allocated_cc_case"].eq(case)].reset_index(drop=True)
        )
        for case in cases
    )
    cumulative_case_identities = tuple(
        canonical_table_sha256(
            cumulative.loc[cumulative["allocated_cc_case"].eq(case)].reset_index(drop=True)
        )
        for case in cases
    )
    return (
        annual_values,
        cumulative_values,
        categories,
        annual_case_identities,
        cumulative_case_identities,
    )


def _existing_result(result_directory: Path) -> AllocatedCCResult | None:
    """Read the completed aCC result summary."""
    manifest_path = result_directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return AllocatedCCResult(
        status="reused",
        allocated_cc_result=manifest["allocated_cc_result"],
        allocated_cc_result_identity=manifest["allocated_cc_result_identity"],
        allocated_cc_cases=tuple(manifest["allocated_cc_cases"]),
        annual_impact_category_count=int(manifest["annual_impact_category_count"]),
        result_directory=result_directory,
        manifest_path=manifest_path,
    )


def compile_allocated_cc_result(
    workspace: Path,
    allocated_cc_result: str,
) -> AllocatedCCResult:
    """Compile one allocated carrying capacity result and write its manifest last.

    A completed processed result is reused. To change a completed aCC result, the user must
    choose a new readable name.
    """

    name = _readable_name(allocated_cc_result, field="allocated_cc_result")
    result_directory = _secure_processed_result_directory(workspace, name)
    existing = _existing_result(result_directory)
    if existing is not None:
        return existing
    lock_path = result_directory.parent / ".writer.lock"
    try:
        with writer_lock(lock_path):
            return _compile_allocated_cc_result(workspace, name, result_directory)
    except FileLockError as exc:
        raise AllocatedCCError(str(exc)) from exc


def _compile_allocated_cc_result(
    workspace: Path,
    name: str,
    result_directory: Path,
) -> AllocatedCCResult:
    """Validate and store one allocated carrying capacity result.

    Args:
        workspace: Active workspace containing the v1 YAML and CSV inputs.
        name: Validated ``allocated_cc_result`` directory name.
        result_directory: Direct processed directory reserved for this name.

    Returns:
        A computed or concurrently reused result record.

    Raises:
        AllocatedCCError: If an incomplete result already occupies the name, or
            the YAML, annual table, cumulative table, methods, units, cases,
            coordinates, values, or identities are invalid.

    Notes:
        Annual input rows cover ``[case, world region, year, EF impact category]`` and
        cumulative rows cover ``[case, world region]`` for climate change GWP100.
        Values must be positive and use the native LCIA units. Files are written
        to a directory beside the final path, the manifest is written last, and
        the completed directory is moved into place at a new result path.
    """

    existing = _existing_result(result_directory)
    if existing is not None:
        return existing

    if result_directory.exists():
        raise AllocatedCCError(
            f"The allocated carrying capacity result directory is incomplete: "
            f"{result_directory}. Keep it unchanged and select another allocated_cc_result, or "
            "create a new workspace and copy the v1 aCC input there."
        )

    source_directory = _secure_source_directory(workspace, name)
    metadata_path = _secure_source_file(
        source_directory,
        "allocated_cc_result.yaml",
        field="allocated_cc_result.yaml",
    )
    metadata, metadata_sha256 = _allocated_cc_metadata(metadata_path, name)
    annual_source_path = _secure_source_file(
        source_directory,
        metadata["annual_allocated_cc"],
        field="annual_allocated_cc",
    )
    cumulative_source_path = _secure_source_file(
        source_directory,
        metadata["cumulative_gwp100_allocated_cc"],
        field="cumulative_gwp100_allocated_cc",
    )
    annual_source, annual_sha256 = _read_csv(
        annual_source_path,
        role="annual_allocated_cc",
    )
    cumulative_source, cumulative_sha256 = _read_csv(
        cumulative_source_path,
        role="cumulative_gwp100_allocated_cc",
    )
    method_authority = _method_authority(workspace)
    annual, cumulative, cases = validate_allocated_cc_frames(
        annual_source,
        cumulative_source,
        method_authority,
    )
    source_files = dict(
        sorted(
            {
                "allocated_cc_result.yaml": metadata_sha256,
                str(metadata["annual_allocated_cc"]): annual_sha256,
                str(metadata["cumulative_gwp100_allocated_cc"]): cumulative_sha256,
            }.items()
        )
    )
    method_parent = {
        "prepared_background_shared_identity": method_authority.prepared_background_shared_identity,
        "prepared_background_shared_manifest_sha256": (
            method_authority.prepared_background_shared_manifest_sha256
        ),
        "prepared_lcia_authority_identity": method_authority.prepared_lcia_authority_identity,
        "annual_lcia_registry_identity": method_authority.identity,
        "cumulative_ef_climate_registry_identity": canonical_table_sha256(
            method_authority.cumulative_climate
        ),
    }
    identity_payload = {
        "schema_version": _ALLOCATED_CC_SCHEMA_VERSION,
        "allocated_cc_result": name,
        "description": str(metadata["description"]),
        "ssp_scenario": "SSP2",
        "lcia_method": ["ef31"],
        "lcia_method_identity": method_authority.selection_identity,
        "annual_table_identity": canonical_table_sha256(annual),
        "cumulative_table_identity": canonical_table_sha256(cumulative),
        "lcia_registry_identity": method_authority.identity,
        "cumulative_climate_registry_identity": canonical_table_sha256(
            method_authority.cumulative_climate
        ),
        "source_files": source_files,
    }
    result_identity = _IDENTITY_PREFIX + payload_sha256(identity_payload)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{name}.allocated_cc.",
            dir=result_directory.parent,
        )
    )
    try:
        frames = {
            "annual_allocated_cc.parquet": annual,
            "cumulative_gwp100_allocated_cc.parquet": cumulative,
            "lcia_method_registry.parquet": method_authority.frame,
            "cumulative_climate_method_registry.parquet": method_authority.cumulative_climate,
        }
        for filename, frame in frames.items():
            frame.to_parquet(staging / filename, index=False)
        artifacts = {
            filename: _artifact(staging / filename, frame) for filename, frame in frames.items()
        }
        manifest = {
            "schema_version": _PRODUCT_SCHEMA_VERSION,
            "complete": True,
            "allocated_cc_result": name,
            "allocated_cc_result_identity": result_identity,
            "allocated_cc_cases": list(cases),
            "lcia_method": ["ef31"],
            "lcia_method_identity": method_authority.selection_identity,
            "annual_impact_category_count": len(method_authority.frame),
            "description": str(metadata["description"]),
            "identity_payload": identity_payload,
            "method_authority_parent": method_parent,
            "source_files": source_files,
            "artifacts": artifacts,
        }
        atomic_write(staging / "manifest.json", canonical_json_bytes(manifest))
        replace_directory(staging, result_directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return AllocatedCCResult(
        status="computed",
        allocated_cc_result=name,
        allocated_cc_result_identity=result_identity,
        allocated_cc_cases=cases,
        annual_impact_category_count=len(method_authority.frame),
        result_directory=result_directory,
        manifest_path=result_directory / "manifest.json",
    )
