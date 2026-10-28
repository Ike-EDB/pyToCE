"""premise 2.4.6 scenario identities and transformation execution.

Only public premise 2.4.6 callables are used. The worker process isolates
premise's import-time cache directory, report directory, and Brightway project
selection for one immutable source identity.
"""

import json
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Final, cast

from pytoce.shared.runtime.file_lock import writer_lock
from pytoce.shared.runtime.filesystem import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    payload_sha256,
)
from pytoce.shared.runtime.workspace_guides import write_workspace_guide

GENERAL_BACKGROUND_ANCHOR_YEARS: Final = (
    2030,
    2035,
    2040,
    2045,
    2050,
    2055,
    2060,
    2070,
    2080,
    2090,
    2100,
)
PROVIDER_TECHNOLOGY_ANCHOR_YEARS: Final = (2020, 2025)
PREMISE_UPDATE_SECTORS: Final = (
    "biomass",
    "electricity",
    "cement",
    "steel",
    "fuels",
    "renewable",
    "metals",
    "mining",
    "heat",
    "cdr",
    "battery",
    "cars",
    "two_wheelers",
    "trucks",
    "ships",
    "buses",
    "trains",
    "final energy",
    "emissions",
)
_IAM_RECORD: Final = "19049274"
_IAM_SCHEMA: Final = "pytoce_premise_iam_input_schema_v1"


@dataclass(frozen=True, slots=True)
class BackgroundScenario:
    """One accepted SSP2 premise background member."""

    background_scenario: str
    premise_model: str
    iam_model: str
    iam_scenario: str
    gmst: str
    iam_filename: str
    iam_file_size: int
    iam_file_sha256: str


BACKGROUND_SCENARIOS: Final = (
    BackgroundScenario(
        "remind_ssp2_ndc",
        "remind",
        "REMIND",
        "SSP2-NDC",
        "2.0-2.5 degC",
        "remind_SSP2-NDC.csv",
        13_201_932,
        "5b0d456e94bdacff6b4f9c4da8f88b522238039a31fe0bf39e3baabd6f0b16c0",
    ),
    BackgroundScenario(
        "remind_ssp2_npi",
        "remind",
        "REMIND",
        "SSP2-NPi",
        "3.2-3.5 degC",
        "remind_SSP2-NPi.csv",
        13_309_900,
        "3a9f77641a1cf2874e2d92a3494c86e90259299ffde34915a01a8bd2ec7ff898",
    ),
    BackgroundScenario(
        "remind_ssp2_pkbudg1000",
        "remind",
        "REMIND",
        "SSP2-PkBudg1000",
        "1.7-2.0 degC",
        "remind_SSP2-PkBudg1000.csv",
        13_533_412,
        "663a4fd70556223f78a534343f1b58dcea0868b6a6e43844fbc1cead7935c0b6",
    ),
    BackgroundScenario(
        "remind_ssp2_pkbudg650",
        "remind",
        "REMIND",
        "SSP2-PkBudg650",
        "<1.5 degC",
        "remind_SSP2-PkBudg650.csv",
        13_290_124,
        "159e09023c5e31ca5334e0712db5fd6da0a8ce1c9998a5d126d797a7403b617b",
    ),
    BackgroundScenario(
        "remind_eu_ssp2_ndc",
        "remind-eu",
        "REMIND-EU",
        "SSP2-NDC",
        "2.0-2.5 degC",
        "remind-eu_SSP2-NDC.csv",
        22_434_148,
        "c1681104dada18f5624e34754845f13e7c6b518a43736815d9c6f822d9b451b6",
    ),
    BackgroundScenario(
        "remind_eu_ssp2_npi",
        "remind-eu",
        "REMIND-EU",
        "SSP2-NPi",
        "3.2-3.5 degC",
        "remind-eu_SSP2-NPi.csv",
        22_776_972,
        "812cb12ec53aa7deb6865de904e5116c20989271dccdeca2d89e747fd5bbbd0e",
    ),
    BackgroundScenario(
        "remind_eu_ssp2_pkbudg1000",
        "remind-eu",
        "REMIND-EU",
        "SSP2-PkBudg1000",
        "1.7-2.0 degC",
        "remind-eu_SSP2-PkBudg1000.csv",
        23_195_384,
        "0b3c4be351d36714063940b4704afb64aa34178dfe0966a8c85767799d00961e",
    ),
    BackgroundScenario(
        "remind_eu_ssp2_pkbudg650",
        "remind-eu",
        "REMIND-EU",
        "SSP2-PkBudg650",
        "1.5-1.7 degC",
        "remind-eu_SSP2-PkBudg650.csv",
        22_888_356,
        "294bb9065bb0926276f3cdfa2b69d24dfbd17d2e7cb205e6dd4e8f2a6074ff31",
    ),
)

BACKGROUND_BY_ID: Final = {item.background_scenario: item for item in BACKGROUND_SCENARIOS}
ACTIVE_BACKGROUND_SCENARIOS: Final = tuple(
    item for item in BACKGROUND_SCENARIOS if item.premise_model == "remind"
)
ACTIVE_BACKGROUND_IDS: Final = frozenset(
    item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS
)
LEGACY_BACKGROUND_IDS: Final = frozenset(BACKGROUND_BY_ID).difference(ACTIVE_BACKGROUND_IDS)


@dataclass(frozen=True, slots=True)
class PremiseIAMInput:
    """Recorded encrypted IAM payload, file hash, and source provenance."""

    identity: str
    payload: Path
    manifest_sha256: str
    payload_sha256: str
    premise_authority_identity: str


def _installed_distribution_authority(
    distribution_name: str,
    *,
    expected_version: str,
) -> dict[str, object]:
    """Identify one installed dependency from its version and wheel record."""

    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"The required dependency {distribution_name} {expected_version} is not "
            "installed. Install pyToCE in a fresh supported Python "
            "environment."
        ) from exc
    if distribution.version != expected_version:
        raise RuntimeError(
            f"The installed {distribution_name} version is {distribution.version}; "
            f"version {expected_version} is required. Install pyToCE in a "
            "fresh supported Python environment."
        )
    record = distribution.read_text("RECORD")
    if record is None:
        raise RuntimeError(
            f"The installed {distribution_name} {expected_version} distribution has no "
            "wheel RECORD. Reinstall pyToCE in a fresh supported Python "
            "environment."
        )
    direct_url = distribution.read_text("direct_url.json")
    payload: dict[str, object] = {
        "schema_version": "pytoce_installed_distribution_schema_v1",
        "distribution": distribution_name,
        "version": distribution.version,
        "record_sha256": sha256(record.encode("utf-8")).hexdigest(),
        "direct_url_sha256": (
            None if direct_url is None else sha256(direct_url.encode("utf-8")).hexdigest()
        ),
    }
    return {
        **payload,
        "identity": "pytoce_installed_distribution_schema_v1::" + payload_sha256(payload),
    }


def installed_preparation_dependency_authorities() -> tuple[dict[str, object], ...]:
    """Identify preparation dependencies."""

    return tuple(
        _installed_distribution_authority(name, expected_version=version)
        for name, version in (
            ("bw2data", "4.7"),
            ("bw2io", "0.9.17"),
            ("premise", "2.4.6"),
        )
    )


def _iam_source_url(scenario: BackgroundScenario) -> str:
    """Construct the official source URL for one premise IAM file."""
    return f"https://zenodo.org/records/{_IAM_RECORD}/files/{scenario.iam_filename}"


def _iam_identity_payload(
    scenario: BackgroundScenario,
    *,
    premise_authority: dict[str, object],
) -> dict[str, object]:
    """Collect file and scenario fields used to identify one IAM input."""
    return {
        "schema_version": _IAM_SCHEMA,
        "background_scenario": scenario.background_scenario,
        "premise_model": scenario.premise_model,
        "iam_model": scenario.iam_model,
        "iam_scenario": scenario.iam_scenario,
        "source_url": _iam_source_url(scenario),
        "payload_name": scenario.iam_filename,
        "payload_size": scenario.iam_file_size,
        "payload_sha256": scenario.iam_file_sha256,
        "premise_authority_identity": premise_authority["identity"],
    }


def _completed_iam_input(
    root: Path,
    scenario: BackgroundScenario,
    *,
    premise_authority: dict[str, object],
) -> PremiseIAMInput | None:
    """Return the premise IAM file reference when its completion manifest exists."""
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return None
    identity_payload = _iam_identity_payload(
        scenario,
        premise_authority=premise_authority,
    )
    identity = f"{_IAM_SCHEMA}::{payload_sha256(identity_payload)}"
    payload = root / scenario.iam_filename
    return PremiseIAMInput(
        identity=identity,
        payload=payload,
        manifest_sha256=file_sha256(manifest),
        payload_sha256=scenario.iam_file_sha256,
        premise_authority_identity=str(premise_authority["identity"]),
    )


def _acquire_iam_input_unlocked(
    workspace: Path,
    scenario: BackgroundScenario,
    *,
    premise_authority: dict[str, object],
) -> PremiseIAMInput:
    """Download or reuse one official encrypted premise scenario file.

    The installed premise downloader writes to a hidden working directory. The
    package checks the official file size and hash before moving the directory
    into the workspace.
    """

    identity_payload = _iam_identity_payload(
        scenario,
        premise_authority=premise_authority,
    )
    identity = f"{_IAM_SCHEMA}::{payload_sha256(identity_payload)}"
    root = workspace / "data_raw" / "premise" / scenario.background_scenario
    existing = _completed_iam_input(
        root,
        scenario,
        premise_authority=premise_authority,
    )
    if existing is not None:
        return existing
    existing_error = (
        f"The premise input directory {root} has no completion manifest. It was left "
        "unchanged. Create a new workspace and run prepare_backgrounds there."
    )
    if root.exists() or root.is_symlink():
        raise RuntimeError(existing_error)

    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".iam_input.", dir=parent))
    try:
        from premise.scenario_downloader import download_csv  # type: ignore[import-untyped]

        downloaded = Path(
            download_csv(
                scenario.iam_filename,
                _iam_source_url(scenario),
                staging,
            )
        )
        expected = staging / scenario.iam_filename
        if (
            downloaded.resolve() != expected.resolve()
            or not expected.is_file()
            or expected.stat().st_size != scenario.iam_file_size
            or file_sha256(expected) != scenario.iam_file_sha256
        ):
            raise RuntimeError(
                f"The downloaded premise IAM file {expected} for "
                f"{scenario.background_scenario!r} does not match the official file. "
                "Run prepare_backgrounds again. If the same error occurs, report it."
            )
        artifact = {
            "path": scenario.iam_filename,
            "bytes": scenario.iam_file_size,
            "sha256": scenario.iam_file_sha256,
        }
        write_workspace_guide(
            staging,
            "pytoce.prepare_backgrounds",
            "premise_source",
        )
        write_workspace_guide(
            parent,
            "pytoce.prepare_backgrounds",
            "premise_sources",
        )
        atomic_write(
            staging / "manifest.json",
            canonical_json_bytes(
                {
                    "schema_version": _IAM_SCHEMA,
                    "identity": identity,
                    "identity_payload": identity_payload,
                    "artifact": artifact,
                    "complete": True,
                }
            ),
        )
        manifest_sha256 = file_sha256(staging / "manifest.json")
        os.replace(staging, root)
        return PremiseIAMInput(
            identity=identity,
            payload=root / scenario.iam_filename,
            manifest_sha256=manifest_sha256,
            payload_sha256=scenario.iam_file_sha256,
            premise_authority_identity=str(premise_authority["identity"]),
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def acquire_iam_input(
    workspace: Path,
    scenario: BackgroundScenario,
    *,
    premise_authority: dict[str, object],
) -> PremiseIAMInput:
    """Obtain one verified premise scenario file under the raw-data writer lock."""

    parent = workspace / "data_raw" / "premise"
    parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(parent / ".writer.lock"):
        return _acquire_iam_input_unlocked(
            workspace,
            scenario,
            premise_authority=premise_authority,
        )


def select_background_scenarios(
    requested: str | Sequence[str] = (
        "remind_ssp2_ndc",
        "remind_ssp2_npi",
        "remind_ssp2_pkbudg1000",
        "remind_ssp2_pkbudg650",
    ),
) -> tuple[BackgroundScenario, ...]:
    """Normalize, deduplicate, and order public background selections."""

    names: tuple[str, ...]
    if isinstance(requested, str):
        value = requested.strip()
        if not value:
            raise ValueError("background_scenario must be non-empty")
        names = (
            tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)
            if value == "all"
            else (value,)
        )
    else:
        try:
            raw = tuple(requested)
        except TypeError as exc:
            raise ValueError(
                "background_scenario must be one available name or a sequence of available names"
            ) from exc
        if not raw:
            raise ValueError("At least one background_scenario is required")
        if any(not isinstance(value, str) or not value.strip() for value in raw):
            raise ValueError("background_scenario entries must be non-empty text")
        if any(value.strip() == "all" for value in raw):
            if len(raw) != 1:
                raise ValueError("'all' cannot be combined with explicit backgrounds")
            names = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)
        else:
            names = tuple(value.strip() for value in raw)
    unknown = sorted(set(names).difference(BACKGROUND_BY_ID))
    if unknown:
        available = ", ".join(sorted(BACKGROUND_BY_ID))
        raise ValueError(
            "Unsupported background_scenario: "
            + ", ".join(repr(value) for value in unknown)
            + f". Choose from: {available}."
        )
    return tuple(BACKGROUND_BY_ID[name] for name in sorted(set(names)))


def database_name(
    scenario: BackgroundScenario,
    year: int,
    *,
    background_identity: str,
) -> str:
    """Return a snake-case premise database name tied to the source identity."""

    digest = background_identity.split("::", maxsplit=1)[-1][:12]
    return f"pytoce__{scenario.background_scenario}__{year}__{digest}"


def provider_technology_database_name(
    scenario: BackgroundScenario,
    year: int,
) -> str:
    """Return the database name for an early provider-only technology anchor."""

    return f"pytoce__{scenario.background_scenario}__provider_technology__{year}"


def premise_configuration() -> dict[str, object]:
    """Return the premise settings used in calculation and provenance.

    These deterministic calculations remove source and import uncertainty
    distributions.
    """

    return {
        "source_version": "3.12",
        "source_type": "brightway",
        "system_model": "cutoff",
        "use_cached_inventories": True,
        "use_cached_database": True,
        "external_scenarios": None,
        "quiet": False,
        "keep_imports_uncertainty": False,
        "keep_source_db_uncertainty": False,
        "gains_scenario": "CLE",
        "use_absolute_efficiency": False,
        "generate_reports": True,
        "update_sectors": list(PREMISE_UPDATE_SECTORS),
    }


def _premise_worker(
    *,
    result_connection: Connection,
    brightway_store: str,
    project_name: str,
    cache_root: str,
    report_root: str,
    iam_directory: str,
    iam_input_identity: str,
    iam_payload_sha256: str,
    premise_authority_identity: str,
    source_database: str,
    biosphere_database: str,
    scenario_payload: dict[str, str],
    years: tuple[int, ...],
    output_names: tuple[str, ...],
    premise_key: str | None,
) -> None:
    """Run one member in a fresh spawned process and return a compact result."""

    try:
        # premise 2.4.6 prints Unicode arrows while updating databases. Force
        # UTF-8 because a spawned Windows process can otherwise use cp1252.
        for stream in (sys.stdout, sys.stderr):
            cast(Any, stream).reconfigure(encoding="utf-8")
        store = Path(brightway_store)
        cache = Path(cache_root)
        reports = Path(report_root)
        cache.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)
        atomic_write(
            reports / "variables.yaml",
            f"USER_DATA_BASE_DIR: {json.dumps(str(cache))}\n".encode("utf-8"),
        )
        os.environ["BRIGHTWAY2_DIR"] = str(store)
        os.chdir(reports)

        import bw2data as bd  # type: ignore[import-untyped]
        import premise  # type: ignore[import-untyped]

        bd.projects.set_current(project_name)
        # premise 2.4.6 accepts the tested optional arguments through its public defaults; the
        # constructor's declared types cover a narrower argument mapping.
        configuration = premise_configuration()
        configuration.pop("update_sectors")
        database = cast(Any, premise.NewDatabase)(
            scenarios=[
                {
                    "model": scenario_payload["premise_model"],
                    "pathway": scenario_payload["iam_scenario"],
                    "year": year,
                    "filepath": iam_directory,
                }
                for year in years
            ],
            key=premise_key,
            source_db=source_database,
            biosphere_name=biosphere_database,
            **configuration,
        )
        database.update(list(PREMISE_UPDATE_SECTORS))
        database.write_db_to_brightway(name=list(output_names))
        database_rows: list[dict[str, object]] = []
        for year, name in zip(years, output_names):
            database_payload = {
                "schema_version": "pytoce_premise_database_schema_v1",
                "premise_authority_identity": premise_authority_identity,
                "iam_input_identity": iam_input_identity,
                "iam_payload_sha256": iam_payload_sha256,
                "source_database": source_database,
                "biosphere_database": biosphere_database,
                "background_scenario": scenario_payload["background_scenario"],
                "anchor_year": year,
                "premise_configuration": {
                    "model": scenario_payload["premise_model"],
                    "pathway": scenario_payload["iam_scenario"],
                    "year": year,
                    "source_db": source_database,
                    "biosphere_name": biosphere_database,
                    **premise_configuration(),
                },
                "writer_database_name": name,
            }
            metadata_row = dict(bd.databases[name])
            metadata_row.update(
                {
                    "pytoce_premise_identity": (
                        "pytoce_premise_database_schema_v1::" + payload_sha256(database_payload)
                    ),
                    "pytoce_premise_provenance": database_payload,
                }
            )
            bd.databases[name] = metadata_row
            database_rows.append(
                {
                    "database_name": name,
                    "anchor_year": year,
                    "premise_identity": metadata_row["pytoce_premise_identity"],
                    "premise_provenance": database_payload,
                }
            )
        result = {
            "ok": True,
            "premise_version": premise.__version__,
            "databases": list(output_names),
            "database_rows": database_rows,
        }
    except Exception as exc:
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": _redact_message(str(exc), premise_key),
        }
    try:
        result["schema_version"] = "pytoce_premise_result_schema_v1"
        result_connection.send(result)
    finally:
        result_connection.close()


def _redact_message(message: str, secret: str | None) -> str:
    """Remove premise credentials from an error message."""
    redacted = message.replace(secret, "<redacted>") if secret else message
    return re.sub(r"gAAAAA[A-Za-z0-9_-]+", "<redacted-fernet-token>", redacted)


def run_premise_member(
    *,
    brightway_store: Path,
    project_name: str,
    cache_root: Path,
    report_root: Path,
    iam_input: PremiseIAMInput,
    source_database: str,
    biosphere_database: str,
    scenario: BackgroundScenario,
    years: tuple[int, ...],
    output_names: tuple[str, ...],
    premise_key: str | None,
) -> dict[str, object]:
    """Create selected anchors using the IAM file verified during download.

    premise 2.4.6 reads its named file from the supplied directory. Reuse the
    package-owned file directly; the download step has verified its bytes.
    """

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_premise_worker,
        kwargs={
            "result_connection": sender,
            "brightway_store": str(brightway_store),
            "project_name": project_name,
            "cache_root": str(cache_root),
            "report_root": str(report_root),
            "iam_directory": str(iam_input.payload.parent),
            "iam_input_identity": iam_input.identity,
            "iam_payload_sha256": iam_input.payload_sha256,
            "premise_authority_identity": iam_input.premise_authority_identity,
            "source_database": source_database,
            "biosphere_database": biosphere_database,
            "scenario_payload": {
                "premise_model": scenario.premise_model,
                "background_scenario": scenario.background_scenario,
                "iam_scenario": scenario.iam_scenario,
            },
            "years": years,
            "output_names": output_names,
            "premise_key": premise_key,
        },
    )
    try:
        process.start()
        # Only the child retains a sender, so a crash signals EOF. Receive before
        # joining: a large result can fill the operating system's pipe buffer.
        sender.close()
        try:
            result = receiver.recv()
        except EOFError as exc:
            process.join()
            raise RuntimeError(
                f"Premise stopped with exit code {process.exitcode} before it returned the "
                f"databases for {scenario.background_scenario}."
            ) from exc
        process.join()
        if process.exitcode != 0 or not result.get("ok"):
            raise RuntimeError(
                f"Premise could not prepare {scenario.background_scenario}: "
                f"{result.get('error_type', 'error')}: {result.get('error', '')}."
            )
        return result
    finally:
        # Stop an interrupted writer before the caller releases the Brightway
        # lock or removes its preparation directory. is_alive() is also false
        # when process creation failed before the child started.
        if process.is_alive():
            process.terminate()
            process.join()
        receiver.close()
        sender.close()
        process.close()
