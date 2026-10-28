"""Deterministic allocated carrying capacity products shared by owner tests."""

import json
from pathlib import Path

import pandas as pd
from infiltration_fixtures import resolved_infiltration_provider
from material_recipe_fixtures import physical_material_recipe
from pa_receiving_fixtures import pa_receiving_reference

from pytoce.prepare_backgrounds.preparation.ecoinvent import brightway_names
from pytoce.prepare_backgrounds.preparation.prepared_state.shared_preparation import (
    write_shared_data,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.source_authority import Sources
from pytoce.prepare_backgrounds.preparation.project_biosphere import (
    load_lcia_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
)
from pytoce.run_mfa.scenario.catalogue import (
    ParameterDefinition,
    load_parameter_catalogue,
)
from pytoce.run_optimization.input.allocated_cc_contract import _MethodAuthority


def _methods() -> pd.DataFrame:
    """Return the complete EF v3.1 method registry used by aCC fixtures."""

    specifications, _payload = load_lcia_catalogue()
    records = []
    for spec in (item for item in specifications if item.method_id == "ef31"):
        records.append(
            {
                "lcia_method": spec.method_id,
                "selection_order": spec.order,
                "method_id": spec.method_id,
                "method_version": spec.method_version,
                "impact_category_id": spec.impact_category_id,
                "impact_category_tuple_json": json.dumps(
                    list(spec.impact_category_tuple), separators=(",", ":")
                ),
                "unit": spec.unit,
                "factor_identity": f"factor::{spec.impact_category_id}",
            }
        )
    assert len(records) == 16
    return pd.DataFrame.from_records(records)


def _climate_method() -> pd.DataFrame:
    """Return the mandatory EF climate method row."""

    return _methods().loc[lambda frame: frame["impact_category_id"] == "climate_change"]


def _authority(methods: pd.DataFrame | None = None) -> _MethodAuthority:
    """Build the typed LCIA authority supplied by the prepared-background owner."""

    frame = _methods() if methods is None else methods
    return _MethodAuthority(
        frame=frame,
        cumulative_climate=frame.loc[frame["impact_category_id"] == "climate_change"],
        identity="method-authority::fixture",
        selection_identity="method-selection::fixture",
        prepared_background_shared_identity="prepared-background::fixture",
        prepared_background_shared_manifest_sha256="a" * 64,
        prepared_lcia_authority_identity="lcia-authority::fixture",
    )


def _regions() -> tuple[str, ...]:
    """Return the authoritative regional coordinates from the parameter catalog."""

    definition = load_parameter_catalogue("collection_rate")
    assert isinstance(definition, ParameterDefinition)
    return definition.coordinates["region"]


def _source_frames(
    cases: tuple[str, ...] = ("direct_case",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete annual and cumulative external aCC source tables."""

    methods = _methods()
    annual = pd.DataFrame.from_records(
        (
            {
                "allocated_cc_case": case,
                "lcia_method": method.lcia_method,
                "region": region,
                "impact_category_id": method.impact_category_id,
                "year": year,
                "allocated_cc": (
                    float(case_index + 1)
                    * float(region_index + 1)
                    * float(method.selection_order)
                    * float(year - 2018)
                ),
                "unit": method.unit,
                "cc_authority": "external_authority",
            }
            for case_index, case in enumerate(cases)
            for region_index, region in enumerate(_regions())
            for method in methods.itertuples(index=False)
            for year in range(2019, 2101)
        )
    )
    climate = _climate_method().iloc[0]
    cumulative = pd.DataFrame.from_records(
        (
            {
                "allocated_cc_case": case,
                "region": region,
                "impact_category_id": "climate_change",
                "period_start": 2019,
                "period_end": 2100,
                "cumulative_allocated_cc": (
                    float(case_index + 1) * float(region_index + 1) * 1000.0
                ),
                "unit": climate["unit"],
                "budget_metric": "ef31_climate_change_gwp100_all_ghg",
                "cc_authority": "external_authority",
            }
            for case_index, case in enumerate(cases)
            for region_index, region in enumerate(_regions())
        )
    )
    return annual, cumulative


def _write_shared_authority(
    workspace: Path,
    *,
    methods: pd.DataFrame | None = None,
    provider: PreparedProviderAuthority | None = None,
    carbon_rows: tuple[dict[str, object], ...] = (),
) -> None:
    """Persist final fixture authorities once through the real shared-product writer.

    Ordinary aCC tests consume only the complete LCIA rows. Candidate tests also
    supply their complete carbon authority before publication.
    """
    (workspace / "data_processed/prepared_backgrounds").mkdir(parents=True)
    provider = (
        resolved_infiltration_provider(workspace / "native_infiltration")
        if provider is None
        else provider
    )
    sources = Sources(
        ecoinvent_archive=workspace / "invented.7z",
        ef_workbook=workspace / "invented.xlsx",
        pb_archive=workspace / "invented.tar.gz",
        source_identity="sources::fixture",
        source_identity_payload={"schema_version": "invented_sources_schema_v1"},
        dependency_authorities=({"distribution": "premise", "version": "2.4.6"},),
        premise_authority={"distribution": "premise", "identity": "premise::2.4.6"},
        premise_authority_identity="premise::2.4.6",
        material_recipe=physical_material_recipe(),
        provider=provider,
        preparation_method={"identity": "preparation::fixture"},
    )
    frame = _methods() if methods is None else methods
    write_shared_data(
        workspace,
        sources,
        brightway_names(workspace, source_identity=sources.source_identity),
        pa_receiving_reference=pa_receiving_reference(
            brightway_names(workspace, source_identity=sources.source_identity).source_database
        ),
        provider=provider,
        material_carbon_rows=carbon_rows,
        impact_rows=tuple(frame.to_dict("records")),
        database_rows=({"database_name": "invented_native"},),
        matrix_rows=({"database_name": "invented_native"},),
        biosphere_rows=({"ecoinvent_biosphere_database": "invented_biosphere"},),
        import_report={"fixture": "No archives or full native databases are consumed."},
    )


def _write_raw_result(
    workspace: Path,
    name: str,
    *,
    reverse: bool = False,
    cases: tuple[str, ...] = ("direct_case",),
) -> Path:
    """Write one complete external aCC result bundle."""

    source = workspace / "data_raw/allocated_cc/results" / name
    source.mkdir(parents=True)
    annual, cumulative = _source_frames(cases)
    if reverse:
        annual = annual.iloc[::-1]
        cumulative = cumulative.iloc[::-1]
    annual.to_csv(source / "annual_allocated_cc.csv", index=False)
    cumulative.to_csv(source / "cumulative_gwp100_allocated_cc.csv", index=False)
    (source / "allocated_cc_result.yaml").write_text(
        "\n".join(
            (
                "schema_version: pytoce_allocated_cc_result_schema_v1",
                f"allocated_cc_result: {name}",
                "description: Direct external aCC fixture",
                "ssp_scenario: SSP2",
                "lcia_method:",
                "  - ef31",
                "annual_allocated_cc: annual_allocated_cc.csv",
                "cumulative_gwp100_allocated_cc: cumulative_gwp100_allocated_cc.csv",
                "",
            )
        ),
        encoding="utf-8",
    )
    return source
