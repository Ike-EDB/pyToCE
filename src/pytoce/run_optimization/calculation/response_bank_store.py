"""Read and write reusable optimization response tables."""

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from pytoce.shared.runtime.file_lock import (
    FileLockError,
    acquire_writer_lock,
    release_writer_lock,
)
from pytoce.shared.runtime.filesystem import (
    atomic_write,
    canonical_json_bytes,
    canonical_table_sha256,
    file_sha256,
    payload_sha256,
)

from .response_bank_contract import (
    _ANCHOR_YEARS,
    _BACKGROUNDS,
    _IDENTITY_PREFIX,
    _OWNER_RELATIVE,
    RESPONSE_BANK_SCHEMA,
    OptimizationResponseBank,
    ResponseBankError,
    ResponseBankInputs,
)


def _artifact(
    directory: Path,
    *,
    name: str,
    frame: pd.DataFrame,
    scientific_identity: str,
) -> dict[str, object]:
    """Describe one newly written response table for the result manifest."""
    return {
        "path": name,
        "row_count": len(frame),
        "columns": [str(column) for column in frame.columns],
        "scientific_identity": scientific_identity,
        "artifact_sha256": file_sha256(directory / name),
    }


def _canonical_directory(path: Path, *, role: str) -> Path:
    """Require an existing directory without links, junctions, or path aliases."""
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not path.is_dir() or resolved != path:
        raise ResponseBankError(f"{role} cannot be a symlink or junction")
    return path


@contextmanager
def _owner_lock(owner: Path) -> Iterator[None]:
    """Hold the file lock while writing one response table directory."""

    lock_path = owner / ".response_bank.writer.lock"
    try:
        stream = acquire_writer_lock(lock_path)
    except FileLockError as exc:
        raise ResponseBankError(str(exc)) from exc
    try:
        yield
    finally:
        release_writer_lock(stream)


def _resolved_owner(workspace: Path) -> Path:
    """Create missing response-bank path components and return the directory."""
    root = workspace.resolve(strict=True)
    current = root
    for part in _OWNER_RELATIVE.parts:
        candidate = current / part
        candidate.mkdir(exist_ok=True)
        current = _canonical_directory(
            candidate,
            role=f"response-bank directory {part}",
        )
    return current


def load_optimization_response_bank(
    workspace: Path,
    response_method_union: str,
) -> OptimizationResponseBank | None:
    """Load a completed bank by its manifest."""

    directory = workspace / _OWNER_RELATIVE / response_method_union
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_bytes())
        return OptimizationResponseBank(
            status="reused",
            identity=str(manifest["response_bank_identity"]),
            response_method_union=response_method_union,
            result_directory=directory,
        )
    if directory.exists() or directory.is_symlink():
        raise ResponseBankError(
            f"The response bank directory {directory} has no completion manifest. "
            "Keep it unchanged and create a new workspace."
        )
    return None


def write_optimization_response_bank(
    owner: Path,
    *,
    response_method_union: str,
    method_union_registry: pd.DataFrame,
    mandatory_climate_method_registry: pd.DataFrame,
    provider_service_registry: pd.DataFrame,
    provider_unit_responses: pd.DataFrame,
    timex_year_weights: pd.DataFrame,
    native_unit_responses: pd.DataFrame,
    direct_characterization_factors: pd.DataFrame,
    source_inputs: ResponseBankInputs,
) -> OptimizationResponseBank:
    """Publish provider responses beneath the already prepared and locked owner.

    The caller supplies calculated provider, Timex and LCIA response tables.
    This function writes those tables.
    """

    name = response_method_union
    directory = owner / name
    if directory.exists() or directory.is_symlink():
        raise ResponseBankError(
            f"A response bank path already exists at {directory}. It was left unchanged. "
            "Use a new workspace before running run_optimization again."
        )
    methods = method_union_registry
    climate = mandatory_climate_method_registry
    services = provider_service_registry
    responses = provider_unit_responses
    weights = timex_year_weights
    response_identity = canonical_table_sha256(responses)
    weights_identity = canonical_table_sha256(weights)
    native_identity = canonical_table_sha256(native_unit_responses)
    direct_identity = canonical_table_sha256(direct_characterization_factors)

    scientific_payload: dict[str, object] = {
        "schema_version": RESPONSE_BANK_SCHEMA,
        "response_method_union": name,
        "background_scenarios": list(_BACKGROUNDS),
        "occurrence_years": [2019, 2100],
        "provider_anchor_years": list(_ANCHOR_YEARS),
        "method_union_registry": methods.to_dict(orient="records"),
        "method_union_registry_identity": canonical_table_sha256(methods),
        "mandatory_climate_registry_identity": canonical_table_sha256(climate),
        "provider_service_registry": services.to_dict(orient="records"),
        "provider_service_registry_identity": canonical_table_sha256(services),
        "provider_unit_responses_identity": response_identity,
        "timex_year_weights_identity": weights_identity,
        "native_unit_responses_identity": native_identity,
        "direct_characterization_factors_identity": direct_identity,
        "provider_route_identity": source_inputs.provider_route_identity,
        "prepared_background_shared_identity": (source_inputs.prepared_background_shared_identity),
        "prepared_background_shared_manifest_sha256": (
            source_inputs.prepared_background_shared_manifest_sha256
        ),
        "builder_identity": source_inputs.builder_identity,
        "member_parents": [
            {
                "background_scenario": parent.background_scenario,
                "prepared_background_identity": parent.prepared_background_identity,
                "prepared_background_manifest_sha256": (parent.prepared_background_manifest_sha256),
            }
            for parent in source_inputs.member_parents
        ],
    }
    scientific_digest = payload_sha256(scientific_payload)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{name}.{scientific_digest}.response_bank.",
            suffix=".staging",
            dir=owner,
        )
    )
    try:
        frames = {
            "provider_unit_responses.parquet": responses,
            "timex_year_weights.parquet": weights,
            "native_unit_responses.parquet": native_unit_responses,
            "direct_characterization_factors.parquet": direct_characterization_factors,
        }
        for filename, frame in frames.items():
            frame.to_parquet(staging / filename, index=False, compression="zstd")
        artifacts = {
            filename: _artifact(
                staging, name=filename, frame=frames[filename], scientific_identity=identity
            )
            for filename, identity in (
                ("provider_unit_responses.parquet", response_identity),
                ("timex_year_weights.parquet", weights_identity),
                ("native_unit_responses.parquet", native_identity),
                ("direct_characterization_factors.parquet", direct_identity),
            )
        }
        identity = _IDENTITY_PREFIX + scientific_digest
        manifest = {
            "schema_version": RESPONSE_BANK_SCHEMA,
            "complete": True,
            "response_method_union": name,
            "response_bank_identity": identity,
            "identity_payload": scientific_payload,
            "artifacts": artifacts,
        }
        atomic_write(staging / "manifest.json", canonical_json_bytes(manifest))
        os.replace(staging, directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return OptimizationResponseBank(
        status="computed",
        identity=identity,
        response_method_union=name,
        result_directory=directory,
    )
