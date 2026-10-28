"""Prepare and save one readable background scenario."""

import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.shared.runtime.file_lock import acquire_writer_lock, release_writer_lock
from pytoce.shared.runtime.filesystem import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    payload_sha256,
    replace_directory,
)
from pytoce.shared.runtime.workspace_guides import write_workspace_guide

from ..ecoinvent import selected_brightway_project
from ..premise import (
    PROVIDER_TECHNOLOGY_ANCHOR_YEARS,
    BackgroundScenario,
    PremiseIAMInput,
    database_name,
    premise_configuration,
    provider_technology_database_name,
    run_premise_member,
)
from ..providers.catalogue import (
    PreparedProviderAuthority,
    _service_source_closures,
    provider_frontier_database_name,
)
from ..providers.frontier_database import ProviderFrontier, prepare_provider_frontier
from ..providers.native_resolution import _NativeResolverCache
from ..providers.pa_receiving_sources import (
    PaReceivingSources,
    admit_pa_receiving_sources,
    sources_to_payload,
)
from ..providers.recipe_context import _RecipeContext
from .authority_tables import MEMBER_SCHEMA, database_row, write_table
from .shared_preparation import SharedAuthority
from .source_authority import Sources, premise_transformation_method

PreparedBackgroundStatus = Literal["computed", "reused"]
_GMST_SOURCE = (
    "https://github.com/polca/premise/blob/46533fef6f860ffd506f62eb13fda29ba3c8404c/"
    "docs/introduction.rst#choosing-the-right-scenario"
)


@dataclass(frozen=True, slots=True)
class PreparedBackgroundResult:
    """One completed background scenario.

    Attributes:
        status: Whether this background scenario was computed or reused.
        background_scenario: Readable package name for the selected background scenario.
        ssp_scenario: Shared socioeconomic pathway, always ``SSP2``.
        result_directory: Directory containing the prepared background.
        manifest_path: Manifest recording the background inputs, files, and completion state.
    """

    status: PreparedBackgroundStatus
    background_scenario: str
    ssp_scenario: str
    result_directory: Path
    manifest_path: Path

    def as_dict(self) -> dict[str, object]:
        """Serialize the background scenario record with paths represented as text."""

        return {
            "status": self.status,
            "background_scenario": self.background_scenario,
            "ssp_scenario": self.ssp_scenario,
            "result_directory": str(self.result_directory),
            "manifest_path": str(self.manifest_path),
        }


@dataclass(frozen=True, slots=True)
class Target:
    """Identity and output paths for one prepared background scenario."""

    scenario: BackgroundScenario
    years: tuple[int, ...]
    background_identity: str
    background_payload: dict[str, object]
    iam_input: PremiseIAMInput
    identity: str
    identity_payload: dict[str, object]
    result_directory: Path


def selected_prepared_background(
    workspace: Path,
    background_scenario: str,
) -> PreparedBackgroundResult | None:
    """Return the saved completed-background reference, or None."""

    result = workspace / "data_processed" / "prepared_backgrounds" / background_scenario
    manifest_path = result / "manifest.json"
    if not manifest_path.is_file():
        return None
    return PreparedBackgroundResult(
        status="reused",
        background_scenario=background_scenario,
        ssp_scenario="SSP2",
        result_directory=result,
        manifest_path=manifest_path,
    )


def target(
    workspace: Path,
    sources: Sources,
    shared: SharedAuthority,
    iam_input: PremiseIAMInput,
    scenario: BackgroundScenario,
    *,
    years: tuple[int, ...],
) -> Target:
    """Resolve the identity and output path for one background scenario."""

    background_payload: dict[str, object] = {
        "schema_version": "pytoce_background_scenario_authority_schema_v1",
        "background_scenario": scenario.background_scenario,
        "ssp_scenario": "SSP2",
        "iam_model": scenario.iam_model,
        "iam_scenario": scenario.iam_scenario,
        "gmst": scenario.gmst,
        "gmst_source": _GMST_SOURCE,
        "source_identity": sources.source_identity,
        "source_identity_payload": sources.source_identity_payload,
        "dependency_authorities": list(sources.dependency_authorities),
        "premise_transformation_method": premise_transformation_method(),
        "iam_input_identity": iam_input.identity,
        "iam_payload_sha256": iam_input.payload_sha256,
        "iam_manifest_sha256": iam_input.manifest_sha256,
        "premise_authority_identity": sources.premise_authority_identity,
        "premise_configuration": premise_configuration(),
    }
    background_identity = "pytoce_background_scenario_authority_schema_v1::" + payload_sha256(
        background_payload
    )
    identity_payload: dict[str, object] = {
        "schema_version": MEMBER_SCHEMA,
        "background_scenario_identity": background_identity,
        "background_scenario_payload": background_payload,
        "anchors": list(years),
        "provider_technology_anchors": list(PROVIDER_TECHNOLOGY_ANCHOR_YEARS),
        "shared_authority_identity": shared.identity,
        "lcia_authority_identity": shared.lcia_identity,
        "provider_authority_identity": shared.provider_identity,
        "provider_timeline_identity": shared.provider_timeline_identity,
        "provider_timeline_database_by_year": dict(shared.provider_timeline_database_by_year),
        "background_preparation_method": sources.preparation_method,
    }
    identity = f"{MEMBER_SCHEMA}::{payload_sha256(identity_payload)}"
    return Target(
        scenario=scenario,
        years=years,
        background_identity=background_identity,
        background_payload=background_payload,
        iam_input=iam_input,
        identity=identity,
        identity_payload=identity_payload,
        result_directory=(
            workspace / "data_processed" / "prepared_backgrounds" / scenario.background_scenario
        ),
    )


def anchor_database_rows(
    shared: SharedAuthority,
    prepared_target: Target,
    premise_report: dict[str, object],
) -> tuple[dict[str, object], ...]:
    """Read the database records returned by the completed premise process."""

    records = cast(list[dict[str, object]], premise_report["database_rows"])
    by_year = {int(cast(Any, record["anchor_year"])): record for record in records}
    prepared_years = (*PROVIDER_TECHNOLOGY_ANCHOR_YEARS, *prepared_target.years)
    rows: list[dict[str, object]] = []
    with selected_brightway_project(shared.names) as bd:
        for year in prepared_years:
            record = by_year[year]
            name = (
                provider_technology_database_name(prepared_target.scenario, year)
                if year in PROVIDER_TECHNOLOGY_ANCHOR_YEARS
                else database_name(
                    prepared_target.scenario,
                    year,
                    background_identity=prepared_target.background_identity,
                )
            )
            metadata = bd.databases[name]
            rows.append(
                {
                    "database_name": name,
                    "database_role": (
                        "early_premise_only_provider_anchor"
                        if year in PROVIDER_TECHNOLOGY_ANCHOR_YEARS
                        else "premise_anchor"
                    ),
                    "activity_count": metadata.get("number"),
                    "depends_json": json.dumps(
                        sorted(str(value) for value in metadata.get("depends", [])),
                        separators=(",", ":"),
                    ),
                    "backend": str(metadata.get("backend", "")),
                    "premise_identity": record["premise_identity"],
                    "premise_provenance_json": json.dumps(
                        record["premise_provenance"],
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "background_scenario": (prepared_target.scenario.background_scenario),
                    "anchor_year": year,
                    "premise_version": "2.4.6",
                    "background_scenario_identity": prepared_target.background_identity,
                    "iam_input_identity": prepared_target.iam_input.identity,
                }
            )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class PreparedProviders:
    """Dated service tables and admitted PA receiving sources for one member."""

    tables: dict[str, tuple[dict[str, object], ...]]
    pa_receiving_sources_by_year: dict[int, PaReceivingSources]


def _prepare_pa_receiving_sources(
    bd: Any,
    *,
    shared: SharedAuthority,
    provider: PreparedProviderAuthority,
    material_recipe: EcoinventMaterialRecipe,
    background_scenario: str,
    year: int,
    ordinary_database: str,
    technology_database: str,
    provider_database: str,
    resolver_cache: _NativeResolverCache,
) -> PaReceivingSources:
    """Admit PA receiving inputs after writing this anchor's study providers.

    The selected project and resolver cache belong to member preparation. Native
    waste-treatment, catalyst and PAM supplies use the ordinary background year.
    The PA6 and PA66 spinning providers retain their study-database provenance.
    Admission uses the completed supplier and database records.
    """
    context = _RecipeContext(
        background_scenario=background_scenario,
        year=year,
        provider_database=provider_database,
        source=resolver_cache.activities(bd, shared.names.source_database),
        ordinary=resolver_cache.activities(bd, ordinary_database),
        technology=resolver_cache.activities(bd, technology_database),
        biosphere=resolver_cache.flows(bd),
        catalogue=provider.catalogue,
        material_recipe=material_recipe,
        method_authorities={
            str(row["method_authority_id"]): row for row in provider.method_authorities
        },
        complete_source_ids=_service_source_closures(provider.catalogue),
        semantic_rows=[],
        technology_rows=[],
        traced_services=set(),
    )
    return admit_pa_receiving_sources(context, shared.pa_receiving_reference)


def prepare_provider_tables(
    shared: SharedAuthority,
    material_recipe: EcoinventMaterialRecipe,
    provider: PreparedProviderAuthority,
    prepared_target: Target,
    anchor_rows: tuple[dict[str, object], ...],
    resolver_cache: _NativeResolverCache,
) -> PreparedProviders:
    """Build dated provider services for one background scenario.

    Provider technologies use premise 2020 and 2025 as early technology sources
    and the matching REMIND member from 2030 onward. Ordinary general background
    supply uses ecoinvent through 2025 and matching member databases thereafter.
    The returned tables record each provider database, temporal interpolation
    input, MFA route, functional unit, source, analog, and method limitation.
    PA receiving source records retain dated native quantities and endpoints for
    later calculation from the actual polymer and inherited finish amounts.
    """

    anchor_database = {
        int(cast(int, row["anchor_year"])): str(row["database_name"]) for row in anchor_rows
    }
    frontier_rows: list[dict[str, object]] = []
    semantic_rows: list[dict[str, object]] = []
    database_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    pa_sources: dict[int, PaReceivingSources] = {}
    prepared_years = (*PROVIDER_TECHNOLOGY_ANCHOR_YEARS, *prepared_target.years)
    with selected_brightway_project(shared.names) as bd:
        for year in prepared_years:
            ordinary_database = (
                shared.names.source_database
                if year in PROVIDER_TECHNOLOGY_ANCHOR_YEARS
                else anchor_database[year]
            )
            frontier: ProviderFrontier = prepare_provider_frontier(
                bd,
                background_scenario=prepared_target.scenario.background_scenario,
                year=year,
                ordinary_database=ordinary_database,
                technology_database=anchor_database[year],
                provider=provider,
                material_recipe=material_recipe,
                resolver_cache=resolver_cache,
            )
            frontier_rows.append(
                {
                    "background_scenario": (prepared_target.scenario.background_scenario),
                    "ssp_scenario": "SSP2",
                    "anchor_year": year,
                    "database_name": frontier.database_name,
                    "provider_level": "provider_service_frontier",
                    "provider_authority_identity": shared.provider_identity,
                    "provider_interface_database": shared.provider_interface_database,
                    "ordinary_database_name": ordinary_database,
                    "general_background_database": ordinary_database,
                    "general_background_anchor_year": (
                        2025 if year in PROVIDER_TECHNOLOGY_ANCHOR_YEARS else year
                    ),
                    "general_background_role": (
                        "ordinary_ecoinvent"
                        if year in PROVIDER_TECHNOLOGY_ANCHOR_YEARS
                        else "matching_member_premise_general_background"
                    ),
                    "selected_background_member": (prepared_target.scenario.background_scenario),
                    "technology_database_name": anchor_database[year],
                    "technology_source_role": (
                        "early_premise_only_provider_anchor"
                        if year in PROVIDER_TECHNOLOGY_ANCHOR_YEARS
                        else "matching_member_premise_anchor"
                    ),
                }
            )
            pa_sources[year] = _prepare_pa_receiving_sources(
                bd,
                shared=shared,
                provider=provider,
                material_recipe=material_recipe,
                background_scenario=prepared_target.scenario.background_scenario,
                year=year,
                ordinary_database=ordinary_database,
                technology_database=anchor_database[year],
                provider_database=frontier.database_name,
                resolver_cache=resolver_cache,
            )
            semantic_rows.extend(
                {**row, "provider_authority_identity": shared.provider_identity}
                for row in frontier.semantic_rows
            )
            semantic_rows.extend(
                {**row, "provider_authority_identity": shared.provider_identity}
                for row in frontier.technology_rows
            )
            database_rows.append(
                database_row(frontier.database_name, role="provider_service_frontier")
            )
            matrix_rows.append(
                {
                    "database_name": frontier.database_name,
                    "background_scenario": (prepared_target.scenario.background_scenario),
                    "anchor_year": year,
                }
            )

    return PreparedProviders(
        tables={
            "provider_technology_crosswalk.parquet": tuple(frontier_rows),
            "provider_semantic_ledger.parquet": tuple(semantic_rows),
            "provider_database_rows": tuple(database_rows),
            "provider_matrix_rows": tuple(matrix_rows),
        },
        pa_receiving_sources_by_year=pa_sources,
    )


def item_tables(
    shared: SharedAuthority,
    prepared_target: Target,
    anchor_rows: tuple[dict[str, object], ...],
    provider_tables: dict[str, tuple[dict[str, object], ...]],
) -> dict[str, tuple[dict[str, object], ...]]:
    """Build the descriptive tables for one prepared background scenario.

    The tables record the REMIND member and IAM input, premise anchor databases,
    LCIA methods, provider technology mapping, source and method details, and
    Brightway database and matrix status. Each row uses stable scalar or JSON
    fields for Parquet publication; completed package products are trusted on reuse.
    """

    background_rows: tuple[dict[str, object], ...] = (
        {
            "background_scenario": prepared_target.scenario.background_scenario,
            "ssp_scenario": "SSP2",
            "iam_model": prepared_target.scenario.iam_model,
            "iam_scenario": prepared_target.scenario.iam_scenario,
            "gmst": prepared_target.scenario.gmst,
            "gmst_source": _GMST_SOURCE,
            "premise_version": "2.4.6",
            "ecoinvent_version": "3.12",
            "system_model": "cutoff",
            "anchor_years_json": json.dumps(prepared_target.years, separators=(",", ":")),
            "source_identity": prepared_target.background_payload["source_identity"],
            "source_identity_payload_json": json.dumps(
                prepared_target.background_payload["source_identity_payload"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "dependency_authorities_json": json.dumps(
                prepared_target.background_payload["dependency_authorities"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "background_preparation_method_json": json.dumps(
                prepared_target.identity_payload["background_preparation_method"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "premise_transformation_method_json": json.dumps(
                prepared_target.background_payload["premise_transformation_method"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "background_scenario_identity": prepared_target.background_identity,
            "iam_input_identity": prepared_target.iam_input.identity,
            "iam_payload_sha256": prepared_target.iam_input.payload_sha256,
            "iam_manifest_sha256": prepared_target.iam_input.manifest_sha256,
            "premise_authority_identity": (prepared_target.iam_input.premise_authority_identity),
            "shared_authority_identity": shared.identity,
            "lcia_authority_identity": shared.lcia_identity,
            "provider_authority_identity": shared.provider_identity,
            "provider_timeline_identity": shared.provider_timeline_identity,
            "provider_timeline_database_by_year_json": json.dumps(
                shared.provider_timeline_database_by_year,
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    )
    methods: tuple[dict[str, object], ...] = tuple(
        {
            "method_id": method_id,
            "method_version": version,
            "lcia_authority_identity": shared.lcia_identity,
        }
        for method_id, version in (
            ("ef31", "3.1"),
            ("pb_lcia", "0.9.2"),
        )
    )
    anchor_databases = {
        2025: shared.names.source_database,
        **{
            int(cast(int, row["anchor_year"])): str(row["database_name"])
            for row in anchor_rows
            if row["database_role"] == "premise_anchor"
        },
    }
    available_anchors = tuple(sorted(anchor_databases))
    crosswalk: list[dict[str, object]] = []
    for stage_year in range(2019, max(prepared_target.years) + 1):
        if stage_year <= 2025:
            lower_year = upper_year = 2025
            numerator = 0
            denominator = 1
            role = "ecoinvent_2025_technical_anchor"
        elif stage_year in anchor_databases:
            lower_year = upper_year = stage_year
            numerator = 0
            denominator = 1
            role = "premise_anchor"
        else:
            lower_year = max(year for year in available_anchors if year < stage_year)
            upper_year = min(year for year in available_anchors if year > stage_year)
            numerator = stage_year - lower_year
            denominator = upper_year - lower_year
            role = "linear_ecoinvent_to_premise" if lower_year == 2025 else "linear_premise_anchors"
        crosswalk.append(
            {
                "background_scenario": (prepared_target.scenario.background_scenario),
                "ssp_scenario": "SSP2",
                "stage_year": stage_year,
                "lower_anchor_year": lower_year,
                "lower_database_name": anchor_databases[lower_year],
                "lower_weight": 1.0 - (numerator / denominator),
                "upper_anchor_year": upper_year,
                "upper_database_name": anchor_databases[upper_year],
                "upper_weight": numerator / denominator,
                "interpolation_fraction_numerator": numerator,
                "interpolation_fraction_denominator": denominator,
                "interpolation_role": role,
                "provider_level": "background_database_frontier",
                "native_provider_owner": "prepare_backgrounds",
                "coverage_start_year": 2019,
                "coverage_end_year": max(prepared_target.years),
            }
        )
    matrix_rows = tuple(
        {
            "database_name": row["database_name"],
            "background_scenario": prepared_target.scenario.background_scenario,
            "anchor_year": row["anchor_year"],
        }
        for row in anchor_rows
    )
    tables: dict[str, tuple[dict[str, object], ...]] = {
        "background_scenarios.parquet": background_rows,
        "lcia_methods.parquet": methods,
        "impact_categories.parquet": shared.impact_categories,
        "provider_crosswalk.parquet": tuple(crosswalk),
        "provider_technology_crosswalk.parquet": provider_tables[
            "provider_technology_crosswalk.parquet"
        ],
        "provider_semantic_ledger.parquet": provider_tables["provider_semantic_ledger.parquet"],
        "database_authority.parquet": (
            *shared.database_rows,
            *anchor_rows,
            *provider_tables["provider_database_rows"],
        ),
        "matrix_authority.parquet": (
            *shared.matrix_rows,
            *matrix_rows,
            *provider_tables["provider_matrix_rows"],
        ),
        "biosphere_authority.parquet": shared.biosphere_rows,
    }
    return tables


def flatten_premise_reports(
    staging: Path,
    report_root: Path,
    years: tuple[int, ...],
) -> tuple[dict[str, object], ...]:
    """Flatten official premise reports into readable year-labeled files."""

    scenario_reports = tuple((report_root / "export" / "scenario_report").glob("*.xlsx"))
    change_reports = tuple((report_root / "export" / "change reports").glob("*.xlsx"))
    if len(scenario_reports) != 1 or len(change_reports) != 1:
        raise RuntimeError(
            "Premise must write one workbook in export/scenario_report and one workbook in "
            "export/change reports. Create a new workspace and run prepare_backgrounds() again."
        )
    scenario_report = scenario_reports[0]
    change_report = change_reports[0]
    suffix = "_".join(str(year) for year in years)
    destinations = (
        staging / f"scenario_report_{suffix}.xlsx",
        staging / f"change_report_{suffix}.xlsx",
    )
    for source, destination in zip((scenario_report, change_report), destinations):
        shutil.move(source, destination)
    shutil.rmtree(report_root)
    return tuple(
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in destinations
    )


def write_prepared_background(
    prepared_target: Target,
    shared: SharedAuthority,
    staging: Path,
    tables: dict[str, tuple[dict[str, object], ...]],
    *,
    pa_receiving_sources_by_year: dict[int, PaReceivingSources],
    premise_report: dict[str, object],
    report_artifacts: tuple[dict[str, object], ...],
) -> None:
    """Write one prepared background, ending with its completion manifest."""

    artifacts = {name: write_table(staging / name, rows) for name, rows in tables.items()}
    pa_sources_bytes = canonical_json_bytes(
        [sources_to_payload(source) for _, source in sorted(pa_receiving_sources_by_year.items())]
    )
    atomic_write(staging / "pa_receiving_sources.json", pa_sources_bytes)
    pa_sources_identity = sha256(pa_sources_bytes).hexdigest()
    artifacts["pa_receiving_sources.json"] = {
        "path": "pa_receiving_sources.json",
        "artifact_sha256": pa_sources_identity,
        "scientific_identity": pa_sources_identity,
    }
    report = {
        "schema_version": "pytoce_background_preparation_report_schema_v1",
        "background_scenario": prepared_target.scenario.background_scenario,
        "anchors": list(prepared_target.years),
        "premise": {key: value for key, value in premise_report.items() if key != "database_rows"},
        "premise_reports": report_artifacts,
        "shared_authority_identity": shared.identity,
        "lcia_authority_identity": shared.lcia_identity,
        "provider_authority_identity": shared.provider_identity,
        "provider_timeline_identity": shared.provider_timeline_identity,
        "provider_timeline_database_by_year": shared.provider_timeline_database_by_year,
    }
    atomic_write(staging / "preparation_report.json", canonical_json_bytes(report))
    artifacts["preparation_report.json"] = {
        "path": "preparation_report.json",
        "artifact_sha256": file_sha256(staging / "preparation_report.json"),
    }
    manifest: dict[str, object] = {
        "schema_version": MEMBER_SCHEMA,
        "identity": prepared_target.identity,
        "identity_payload": prepared_target.identity_payload,
        "background_scenario": prepared_target.scenario.background_scenario,
        "ssp_scenario": "SSP2",
        "iam_model": prepared_target.scenario.iam_model,
        "iam_scenario": prepared_target.scenario.iam_scenario,
        "gmst": prepared_target.scenario.gmst,
        "gmst_source": _GMST_SOURCE,
        "premise_version": "2.4.6",
        "ecoinvent_version": "3.12",
        "system_model": "cutoff",
        "shared_authority_identity": shared.identity,
        "lcia_authority_identity": shared.lcia_identity,
        "provider_authority_identity": shared.provider_identity,
        "provider_timeline_identity": shared.provider_timeline_identity,
        "provider_timeline_database_by_year": shared.provider_timeline_database_by_year,
        "provider_interface_database": shared.provider_interface_database,
        "brightway_project": shared.names.project,
        "artifacts": artifacts,
        "premise_reports": report_artifacts,
        "complete": True,
    }
    write_workspace_guide(
        staging,
        "pytoce.prepare_backgrounds",
        "prepared_background",
    )
    atomic_write(staging / "manifest.json", canonical_json_bytes(manifest))


def item_report(
    prepared_target: Target,
    *,
    status: PreparedBackgroundStatus,
) -> PreparedBackgroundResult:
    """Build the report for one readable background scenario."""

    manifest = prepared_target.result_directory / "manifest.json"
    return PreparedBackgroundResult(
        status=status,
        background_scenario=prepared_target.scenario.background_scenario,
        ssp_scenario="SSP2",
        result_directory=prepared_target.result_directory,
        manifest_path=manifest,
    )


def _prepare_member_unlocked(
    workspace: Path,
    sources: Sources,
    shared: SharedAuthority,
    provider: PreparedProviderAuthority,
    iam_input: PremiseIAMInput,
    scenario: BackgroundScenario,
    *,
    years: tuple[int, ...],
    premise_key: str | None,
    progress: Callable[[str, str], None],
    provider_resolvers: _NativeResolverCache,
) -> PreparedBackgroundResult:
    """Prepare or reuse one REMIND background scenario.

    A completed manifest is reused. Otherwise the function first checks that none
    of the required premise or provider database names already exists, runs
    premise for the early provider anchors and requested background anchors,
    builds the provider services, and writes the result manifest last. Brightway
    changes cannot be rolled back reliably, so existing databases are preserved
    on failure. A later attempt refuses colliding names and requires a new workspace.

    Args:
        workspace: Active pyToCE workspace.
        sources: Recorded ecoinvent, premise and LCIA source authorities.
        shared: Completed background data shared by all REMIND scenarios.
        provider: Provider catalog compiled from package resources.
        iam_input: Recorded REMIND IAM file and file hash for this member.
        scenario: Background scenario and its premise identifiers.
        years: Premise anchor years requested from 2030 onward.
        premise_key: Decryption key passed to premise for its encrypted IAM files.
        progress: Callback receiving the member name and preparation stage.
        provider_resolvers: Activity-resolution cache shared across members.

    Returns:
        Newly computed or reused prepared-background result.
    """

    prepared_target = target(
        workspace,
        sources,
        shared,
        iam_input,
        scenario,
        years=years,
    )
    if (prepared_target.result_directory / "manifest.json").is_file():
        progress(scenario.background_scenario, "reuse")
        return item_report(prepared_target, status="reused")
    if prepared_target.result_directory.exists() or prepared_target.result_directory.is_symlink():
        raise RuntimeError(
            f"The prepared background directory has no completion manifest: "
            f"{prepared_target.result_directory}. Create a new workspace and run the "
            "workflows there; the existing directory has not been changed."
        )

    prepared_target.result_directory.parent.mkdir(parents=True, exist_ok=True)
    premise_years = (*PROVIDER_TECHNOLOGY_ANCHOR_YEARS, *years)
    premise_output_names = (
        *(
            provider_technology_database_name(scenario, year)
            for year in PROVIDER_TECHNOLOGY_ANCHOR_YEARS
        ),
        *(
            database_name(
                scenario,
                year,
                background_identity=prepared_target.background_identity,
            )
            for year in years
        ),
    )
    frontier_names = tuple(
        provider_frontier_database_name(
            scenario.background_scenario, year, interface_database=provider.interface_database
        )
        for year in premise_years
    )
    operation_database_names = (*premise_output_names, *frontier_names)
    with selected_brightway_project(shared.names) as bd:
        collisions = sorted(set(operation_database_names).intersection(bd.databases))
    if collisions:
        joined = ", ".join(repr(name) for name in collisions)
        raise RuntimeError(
            f"The Brightway project for {workspace} already contains {joined} needed for "
            f"background scenario {scenario.background_scenario!r}. Create a new workspace "
            "and run the workflows there; the existing project has not been changed."
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{scenario.background_scenario}.",
            dir=prepared_target.result_directory.parent,
        )
    )
    try:
        report_root = staging / ".premise_work"
        cache_root = report_root / "cache"
        progress(scenario.background_scenario, "premise_update")
        premise_report = run_premise_member(
            brightway_store=shared.names.store,
            project_name=shared.names.project,
            cache_root=cache_root,
            report_root=report_root,
            iam_input=iam_input,
            source_database=shared.names.source_database,
            biosphere_database=shared.names.biosphere_database,
            scenario=scenario,
            years=premise_years,
            output_names=premise_output_names,
            premise_key=premise_key,
        )
        report_artifacts = flatten_premise_reports(staging, report_root, premise_years)
        anchor_rows = anchor_database_rows(shared, prepared_target, premise_report)
        progress(scenario.background_scenario, "provider_services")
        provider_tables = prepare_provider_tables(
            shared,
            sources.material_recipe,
            provider,
            prepared_target,
            anchor_rows,
            resolver_cache=provider_resolvers,
        )
        write_prepared_background(
            prepared_target,
            shared,
            staging,
            item_tables(shared, prepared_target, anchor_rows, provider_tables.tables),
            pa_receiving_sources_by_year=provider_tables.pa_receiving_sources_by_year,
            premise_report=premise_report,
            report_artifacts=report_artifacts,
        )
        replace_directory(staging, prepared_target.result_directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return item_report(prepared_target, status="computed")


def prepare_member(
    workspace: Path,
    sources: Sources,
    shared: SharedAuthority,
    provider: PreparedProviderAuthority,
    iam_input: PremiseIAMInput,
    scenario: BackgroundScenario,
    *,
    years: tuple[int, ...],
    premise_key: str | None,
    progress: Callable[[str, str], None],
    provider_resolvers: _NativeResolverCache,
) -> PreparedBackgroundResult:
    """Prepare one background scenario under its Brightway project lock."""

    shared.names.store.mkdir(parents=True, exist_ok=True)
    stream = acquire_writer_lock(shared.names.writer_lock_path)
    try:
        return _prepare_member_unlocked(
            workspace,
            sources,
            shared,
            provider,
            iam_input,
            scenario,
            years=years,
            premise_key=premise_key,
            progress=progress,
            provider_resolvers=provider_resolvers,
        )
    finally:
        release_writer_lock(stream)
