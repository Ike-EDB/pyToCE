"""Synthetic Brightway processes and prepared inputs for bounded LCA integration tests."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pandas as pd
from material_recipe_fixtures import physical_material_recipe
from pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.premise import (
    BACKGROUND_BY_ID,
    GENERAL_BACKGROUND_ANCHOR_YEARS,
    PROVIDER_TECHNOLOGY_ANCHOR_YEARS,
    PremiseIAMInput,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    PreparedManifest,
    read_prepared_manifest,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.member_preparation import (
    item_tables,
    target,
    write_prepared_background,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.shared_preparation import (
    write_shared_data,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.source_authority import Sources
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    provider_frontier_database_name,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)

PROVIDER_TECHNICAL_YEARS = (
    *PROVIDER_TECHNOLOGY_ANCHOR_YEARS,
    *GENERAL_BACKGROUND_ANCHOR_YEARS,
)

_PROVIDER_AUTHORITY = "bounded_provider_authority"
_PROVIDER_SERVICE_ID = "bounded_provider_service"
_PROVIDER_SERVICE_CODE = provider_service_code(_PROVIDER_SERVICE_ID)


def _process(
    database: str,
    code: str,
    *,
    biosphere: tuple[str, str] | None = None,
    unmarked: tuple[str, str] | None = None,
    provider: tuple[str, str] | None = None,
    provider_stage_owner: str | None = None,
) -> dict[str, object]:
    """Return one synthetic Brightway process with optional external exchanges."""

    exchanges: list[dict[str, object]] = [
        {
            "input": (database, code),
            "amount": 1.0,
            "type": "production",
        }
    ]
    if biosphere is not None:
        exchanges.append(
            {
                "input": biosphere,
                "amount": 1.0,
                "type": "biosphere",
            }
        )
    if unmarked is not None:
        exchanges.append(
            {
                "input": unmarked,
                "amount": -0.5,
                "type": "biosphere",
            }
        )
    if provider is not None:
        exchanges.append(
            {
                "input": provider,
                "amount": 1.0,
                "type": "technosphere",
            }
        )
    result = {
        "name": "bounded provider",
        "reference product": "bounded provider service",
        "unit": "kilogram",
        "location": "GLO",
        "type": "process",
        "exchanges": exchanges,
    }
    if provider_stage_owner is not None:
        result["provider_stage_owner"] = provider_stage_owner
    return result


def write_prepared_inputs(
    workspace: Path,
    names: BrightwayNames,
    *,
    methods: pd.DataFrame,
    provider: PreparedProviderAuthority,
    carbon_rows: tuple[dict[str, object], ...],
    backgrounds: tuple[str, ...],
    receiving_reference: PaReceivingReference | None = None,
    receiving_sources_by_background: Mapping[str, Mapping[int, PaReceivingSources]] | None = None,
    technology_databases_by_background: Mapping[str, Mapping[int, str]] | None = None,
) -> tuple[PreparedManifest, dict[str, PreparedManifest]]:
    """Write synthetic prepared inputs using the shared and background table writers.

    Only input metadata and small tables are written. No licensed archive,
    premise transformation or Brightway database is read or constructed.
    Callers testing receiving calculations supply records from their synthetic
    original and future inventories; other tests use predefined synthetic records.
    Both paths write the same prepared input tables.
    """
    sources = Sources(
        ecoinvent_archive=workspace / "invented.7z",
        ef_workbook=workspace / "invented.xlsx",
        pb_archive=workspace / "invented.tar.gz",
        source_identity="sources::fixture",
        source_identity_payload={"schema_version": "invented_sources_schema_v1"},
        dependency_authorities=({"distribution": "premise", "version": "2.4.6"},),
        premise_authority={"distribution": "premise", "identity": "premise::fixture"},
        premise_authority_identity="premise::fixture",
        preparation_method={"identity": "preparation::fixture"},
        material_recipe=physical_material_recipe(),
        provider=provider,
    )
    prepared = workspace / "data_processed/prepared_backgrounds"
    prepared.mkdir(parents=True, exist_ok=True)
    shared = write_shared_data(
        workspace,
        sources,
        names,
        pa_receiving_reference=(
            pa_receiving_reference(names.source_database)
            if receiving_reference is None
            else receiving_reference
        ),
        provider=provider,
        material_carbon_rows=carbon_rows,
        impact_rows=cast(tuple[dict[str, object], ...], tuple(methods.to_dict("records"))),
        database_rows=({"database_name": names.source_database},),
        matrix_rows=({"database_name": names.source_database},),
        biosphere_rows=({"ecoinvent_biosphere_database": names.biosphere_database},),
        import_report={"fixture": "No native import or scientific calculation."},
    )
    members = {}
    for background in backgrounds:
        scenario = BACKGROUND_BY_ID[background]
        receiving_sources = (
            {
                year: pa_receiving_sources(
                    year,
                    background_scenario=scenario.background_scenario,
                    source_database=shared.names.source_database,
                )
                for year in PROVIDER_TECHNICAL_YEARS
            }
            if receiving_sources_by_background is None
            else receiving_sources_by_background[background]
        )
        general_databases = {
            year: (names.source_database if year <= 2025 else f"native_{background}_{year}")
            if receiving_sources_by_background is None
            else receiving_sources[year].general_background_database
            for year in PROVIDER_TECHNICAL_YEARS
        }
        technology_databases = (
            {year: f"native_{background}_{year}" for year in PROVIDER_TECHNICAL_YEARS}
            if technology_databases_by_background is None
            else technology_databases_by_background[background]
        )
        iam = PremiseIAMInput(
            identity=f"iam::{background}",
            payload=workspace / "invented_iam" / scenario.iam_filename,
            manifest_sha256="d" * 64,
            payload_sha256="e" * 64,
            premise_authority_identity=sources.premise_authority_identity,
        )
        prepared_target = target(
            workspace, sources, shared, iam, scenario, years=GENERAL_BACKGROUND_ANCHOR_YEARS
        )
        anchor_rows: tuple[dict[str, object], ...] = tuple(
            {
                "database_name": general_databases[year],
                "database_role": "premise_anchor",
                "anchor_year": year,
            }
            for year in GENERAL_BACKGROUND_ANCHOR_YEARS
        )
        frontier_rows: tuple[dict[str, object], ...] = tuple(
            {
                "background_scenario": background,
                "ssp_scenario": "SSP2",
                "anchor_year": year,
                "database_name": provider_frontier_database_name(
                    background, year, interface_database=provider.interface_database
                ),
                "provider_level": "provider_service_frontier",
                "provider_authority_identity": shared.provider_identity,
                "provider_interface_database": shared.provider_interface_database,
                "ordinary_database_name": general_databases[year],
                "general_background_database": general_databases[year],
                "general_background_anchor_year": 2025 if year <= 2025 else year,
                "general_background_role": "ordinary_ecoinvent"
                if year <= 2025
                else "matching_member_premise_general_background",
                "selected_background_member": background,
                "technology_database_name": technology_databases[year],
                "technology_source_role": "early_premise_only_provider_anchor"
                if year <= 2025
                else "matching_member_premise_anchor",
            }
            for year in (*PROVIDER_TECHNOLOGY_ANCHOR_YEARS, *GENERAL_BACKGROUND_ANCHOR_YEARS)
        )
        tables = item_tables(
            shared,
            prepared_target,
            anchor_rows,
            {
                "provider_technology_crosswalk.parquet": frontier_rows,
                "provider_semantic_ledger.parquet": (),
                "provider_database_rows": (),
                "provider_matrix_rows": (),
            },
        )
        directory = prepared / background
        directory.mkdir()
        write_prepared_background(
            prepared_target,
            shared,
            directory,
            tables,
            pa_receiving_sources_by_year=dict(receiving_sources),
            premise_report={"database_rows": []},
            report_artifacts=(),
        )
        members[background] = read_prepared_manifest(directory / "manifest.json")
    return read_prepared_manifest(prepared / "shared/manifest.json"), members
