"""Calculate LCA results and write their completion manifest last."""

import gc
import platform
import sys
import tempfile
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import pandas as pd

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
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

from .carbon import (
    CARBON_CONSISTENCY_METHOD_IDENTITY,
    DIRECT_TREATMENT_CARBON_FLOWS_IDENTITY,
    TREATMENT_STAGES,
    carbon_consistency_table,
    direct_treatment_carbon_flows,
    pa_direct_carbon_flows,
)
from .contribution_accumulation import (
    CONTRIBUTION_METHOD_IDENTITY,
    ContributionMethod,
    create_contribution_state,
)
from .contributions import (
    finalize_contributions,
)
from .direct_contributions import (
    DirectContribution,
    add_direct_contributions,
    admit_direct_characterization_factors,
)
from .flow_coverage import FLOW_OWNERSHIP_METHOD_IDENTITY
from .foreground import H12_REGIONS, ForegroundProduct
from .lcia import ImpactCategory, characterize_principal_inventory
from .pa_inventory import build_pa_receiving_inventory, pa_receiving_physical_events
from .pa_projection import iter_pa_projection
from .providers import PreparedProviderProduct
from .result_diagnostics import (
    MISMANAGED_ACTIVATION_COLUMNS,
    MISMANAGED_STAGES,
    mismanaged_result_tables,
    restore_activation_lineage,
    spool_activation_lineage,
)
from .stage_scores import (
    attach_contribution_norms,
)
from .timex import (
    GRAPH_TRAVERSAL,
    INTERPOLATION_TYPE,
    MAX_CALCULATIONS,
    PRINCIPAL_INVENTORY_METHOD,
    PROVIDER_SOLVE_BATCH_SIZE,
    STARTING_DATETIME,
    TEMPORAL_GROUPING,
    TRAVERSAL_CUTOFF,
    CharacterizationFactorBank,
    PrincipalTimeline,
    assemble_principal_inventory,
    build_principal_timelines,
    build_provider_inventory_bank,
    compact_principal_timelines,
    database_dates,
)

_RESULT_SCHEMA = "pytoce_lca_result_schema_v1"
_RESULT_ARTIFACTS = (
    "scores.parquet",
    "annual_scores.parquet",
    "annual_life_cycle_stage_scores.parquet",
    "life_cycle_stage_scores.parquet",
    "inventory_authority.parquet",
    "mechanical_recycling_loss.parquet",
    "mismanaged_waste_fate.parquet",
    "activation_lineage.parquet",
    "mismanaged_waste_fate_closure.parquet",
    "mfa_to_lca_flow_coverage.parquet",
    "carbon_consistency.csv",
    "direct_treatment_carbon_flows.parquet",
    "pa_receiving_inventory.parquet",
    "pa_receiving_physical_events.parquet",
    "contributions.parquet",
    "stage_contributions.parquet",
    "structural_paths.parquet",
    "structural_edges.parquet",
    "structural_attribution.parquet",
)
_CARBON_REPORTING_ACTIVATION_COLUMNS = (
    "year",
    "region",
    "life_cycle_stage",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "good",
    "material_cycle_source",
    "ultimate_carbon_origin",
    "lca_fate",
    "fate_origin",
    "provider_database",
    "provider_anchor_year",
    "provider_code",
    "amount",
    "mfa_carbon_amount_kg",
)


def _table(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    """Write a Parquet table and return its LCA result manifest record."""
    frame.to_parquet(path, index=False, compression="zstd")
    return {
        "path": path.name,
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "canonical_sha256": canonical_table_sha256(frame),
        "artifact_sha256": file_sha256(path),
    }


def _csv_table(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    """Write a CSV table and return its LCA result manifest record."""
    frame.to_csv(path, index=False, lineterminator="\n")
    return {
        "path": path.name,
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "canonical_sha256": canonical_table_sha256(frame),
        "artifact_sha256": file_sha256(path),
    }


def _spool_resolved_activations(
    activations: pd.DataFrame,
    target: Any,
) -> None:
    """Save resolved activations before releasing the provider calculation data."""

    activations.to_parquet(target, index=False)
    target.flush()


def _restore_resolved_activations(
    target: Any,
    *,
    columns: tuple[str, ...],
    life_cycle_stages: tuple[str, ...],
) -> pd.DataFrame:
    """Read the selected life cycle stages after the provider calculation ends."""

    target.seek(0)
    filters = [("life_cycle_stage", "in", list(life_cycle_stages))]
    return pd.read_parquet(
        target,
        columns=list(columns),
        filters=filters,
    )


def _scientific_software() -> dict[str, str]:
    """Record exact versions of the scientific libraries used by LCA."""
    return {
        "python": platform.python_version(),
        **{
            package: version(package)
            for package in (
                "pytoce",
                "bw_timex",
                "bw2data",
                "bw2calc",
                "bw_temporalis",
                "numpy",
                "pandas",
                "scipy",
            )
        },
    }


def _timex_arguments(root_count: int) -> dict[str, object]:
    """Return the exact public Timex invocation and its unit-root limits."""

    return {
        "starting_datetime": STARTING_DATETIME.isoformat(),
        "temporal_grouping": TEMPORAL_GROUPING,
        "interpolation_type": INTERPOLATION_TYPE,
        "root_count": root_count,
        "cutoff": TRAVERSAL_CUTOFF / root_count,
        "max_calc": MAX_CALCULATIONS * root_count,
        "unit_root_cutoff": TRAVERSAL_CUTOFF,
        "unit_root_max_calc": MAX_CALCULATIONS,
        "graph_traversal": GRAPH_TRAVERSAL,
        "traverse_background": False,
        "principal_inventory_method": PRINCIPAL_INVENTORY_METHOD,
        "provider_solve_batch_size": PROVIDER_SOLVE_BATCH_SIZE,
    }


def validate_lca_result_for_refresh(directory: Path) -> None:
    """Protect unrelated files and linked paths before replacing a completed result."""

    expected_files = {*_RESULT_ARTIFACTS, "README.md", "execution_report.json", "manifest.json"}
    if directory.is_symlink() or any(
        path.name not in expected_files or path.is_symlink() or not path.is_file()
        for path in directory.iterdir()
    ):
        raise ValueError(
            f"Cannot refresh {directory}: replacement requires a directory containing only "
            "ordinary LCA result files, with no links or additional entries. Keep it "
            "unchanged and use a new workspace."
        )


def write_lca_result(
    result_directory: Path,
    foreground_scenario: str,
    background_scenario: str,
    foreground: ForegroundProduct,
    methods: tuple[ImpactCategory, ...],
    years: tuple[int, ...],
    mfa_manifest: dict[str, object],
    mfa_manifest_sha256: str,
    background_manifest: dict[str, object],
    background_manifest_sha256: str,
    *,
    names: BrightwayNames,
    providers: PreparedProviderProduct,
    pa_dump_shares: Mapping[str, float],
    background_dimensions: dict[str, object],
    replacing: bool,
) -> ItemStatus:
    """Calculate and save one foreground/background LCA result.

    The function builds one dated foreground timeline for each of the 12 world
    regions, solves the corresponding Brightway inventories, and characterizes
    the same inventory with every selected LCIA method. Annual scores cover the
    activity years 2019--2100; global values are obtained from the completed
    regional values.

    Args:
        result_directory: Final directory for this LCA selection.
        foreground_scenario: Name of the completed foreground scenario.
        background_scenario: Name of the prepared background scenario.
        foreground: Compiled foreground database, activations, and MFA-to-LCA mappings.
        methods: LCIA methods used to characterize each inventory.
        years: Years selected for upstream contribution analysis.
        mfa_manifest: Manifest of the completed MFA result used as input.
        mfa_manifest_sha256: SHA-256 digest of that MFA manifest.
        background_manifest: Manifest of the prepared background used as input.
        background_manifest_sha256: SHA-256 digest of that background manifest.
        names: Admitted Brightway store, project and native database names.
        providers: Admitted provider and temporal database maps.
        pa_dump_shares: Original conditional infiltration weights already admitted
            by shared preparation, applied to complete PA dump mixtures once.
        background_dimensions: Admitted scenario, source and technology labels.
        replacing: Whether this calculation replaces the result at
            ``result_directory`` after successful completion.

    Returns:
        ``"computed"`` for a new result or ``"updated"`` when ``replacing`` is
        true, after all tables and the completion manifest have been written.
    """

    scores: list[pd.DataFrame] = []
    annual_scores: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    timex_arguments = _timex_arguments(len(H12_REGIONS))
    dates = database_dates(foreground.database_name, providers.timeline_database_by_year)
    flow_coverage = foreground.mfa_to_lca_flow_coverage.copy()
    flow_coverage["background_scenario"] = background_scenario
    roots: dict[str, tuple[str, str]] = dict(zip(H12_REGIONS, foreground.root_keys))
    result_directory.parent.mkdir(parents=True, exist_ok=True)
    write_workspace_guide(
        result_directory.parent,
        "pytoce.run_lca",
        "lca_results",
    )
    with tempfile.TemporaryDirectory(
        prefix=".lca_result.",
        dir=result_directory.parent,
        ignore_cleanup_errors=True,
    ) as runtime_temp:
        runtime_directory = Path(runtime_temp)
        with (
            selected_brightway_project(names) as bd,
            tempfile.TemporaryFile(suffix=".parquet") as activation_spool,
            tempfile.TemporaryFile(suffix=".parquet") as activation_lineage_spool,
        ):
            print(
                f"run_lca: {foreground_scenario}, {background_scenario}: preparing timelines",
                file=sys.stderr,
            )
            regional_roots = {
                region: bd.get_node(
                    database=roots[region][0],
                    code=roots[region][1],
                )
                for region in H12_REGIONS
            }
            timelines = build_principal_timelines(
                regional_roots,
                methods[0].method_tuple,
                dates,
                bd,
                foreground.activations,
                providers.database_by_year,
                providers.timeline_database_by_year,
            )
            for region in H12_REGIONS:
                print(
                    f"run_lca: {foreground_scenario}, {background_scenario}, {region}: "
                    "timeline prepared",
                    file=sys.stderr,
                )
            # Retain fixed kg-service diagnostics before actual native-unit purchases join the
            # inventory. Their PA provenance has its own explicit product.
            fixed_frames = [timeline.activations for timeline in timelines.values()]
            resolved_activations = pd.concat(
                [frame for frame in fixed_frames if not frame.empty] or fixed_frames[:1],
                ignore_index=True,
            )
            del fixed_frames
            _spool_resolved_activations(resolved_activations, activation_spool)
            del resolved_activations
            spool_activation_lineage(
                timelines,
                activation_lineage_spool,
                foreground_scenario=foreground_scenario,
                background_scenario=background_scenario,
            )
            pa_records: list[pd.DataFrame] = []
            pa_physical_records: list[pd.DataFrame] = []
            pa_carbon_records: list[pd.DataFrame] = []
            direct_rows: list[DirectContribution] = []
            # Keep complete receiving objects only for one world region at a time.
            # Source parameters are already admitted; candidate calculations use
            # this same pure projection before temporal weighting.
            for region in H12_REGIONS:
                projected = tuple(
                    iter_pa_projection(
                        spinning_events=tuple(
                            event
                            for event in foreground.pa_spinning_events
                            if event.region == region
                        ),
                        recycling_events=tuple(
                            event
                            for event in foreground.pa_recycling_events
                            if event.region == region
                        ),
                        disposal_events=tuple(
                            event
                            for event in foreground.pa_disposal_events
                            if event.region == region
                        ),
                        reference=providers.pa_receiving_reference,
                        sources_by_year=providers.pa_receiving_sources_by_year,
                        matrix_by_year=providers.database_by_year,
                        dump_shares=pa_dump_shares,
                    )
                )
                receiving = build_pa_receiving_inventory(bd, projected, roots)
                pa_records.append(receiving.records)
                pa_physical_records.append(pa_receiving_physical_events(projected))
                pa_carbon_records.append(pa_direct_carbon_flows(projected))
                direct_rows.extend(receiving.direct_contributions)
                timeline = timelines[region]
                timelines[region] = PrincipalTimeline(
                    timeline.timeline,
                    pd.concat((timeline.activations, receiving.activations), ignore_index=True)
                    if not receiving.activations.empty
                    else timeline.activations,
                    timeline.base_static_score,
                    receiving.direct_inventory[["year", "flow_id", "amount"]],
                )
                del projected, receiving
            contribution_state = create_contribution_state(
                bd,
                {region: timelines[region].activations for region in H12_REGIONS},
                cast(dict[str, tuple[str, str]], roots),
                tuple(
                    ContributionMethod(
                        method_id=method.method_id,
                        method_version=method.method_version,
                        impact_category_id=method.impact_category_id,
                        method_tuple=method.method_tuple,
                        unit=method.unit,
                    )
                    for method in methods
                ),
                years,
            )
            if direct_rows:
                admit_direct_characterization_factors(
                    contribution_state,
                    sorted({row.flow_id for row in direct_rows}),
                )
                add_direct_contributions(contribution_state, direct_rows)
            del direct_rows
            timelines = compact_principal_timelines(timelines)
            gc.collect()
            inventory_frames = [item.activations for item in timelines.values()]
            bank = build_provider_inventory_bank(
                bd,
                pd.concat(
                    [frame for frame in inventory_frames if not frame.empty]
                    or inventory_frames[:1],
                    ignore_index=True,
                ),
                contribution_state,
            )
            del inventory_frames
            execution_report = {
                "schema_version": "pytoce_lca_execution_report_schema_v1",
                "provider_matrix_count": bank.matrix_count,
                "provider_count": bank.provider_count,
                "provider_solve_batch_count": bank.solve_batch_count,
            }
            factors = CharacterizationFactorBank(
                factors=contribution_state.characterization_factors,
            )
            for region in H12_REGIONS:
                print(
                    f"run_lca: {foreground_scenario}, {background_scenario}, {region}: "
                    "calculating the life cycle inventory",
                    file=sys.stderr,
                )
                principal = assemble_principal_inventory(
                    timelines[region],
                    bank,
                )
                characterized = characterize_principal_inventory(
                    principal,
                    methods,
                    factors,
                    foreground_scenario=foreground_scenario,
                    background_scenario=background_scenario,
                    background_dimensions=background_dimensions,
                    region=region,
                )
                scores.append(characterized.scores)
                annual_scores.append(characterized.annual_scores)
                timeline_identity = "pytoce_timeline_v1::" + payload_sha256(
                    {
                        "foreground_identity": foreground.foreground_identity,
                        "prepared_background_identity": background_manifest["identity"],
                        "region": region,
                        "regional_root": roots[region],
                        "timex": timex_arguments,
                    }
                )
                principal_inventory_identity = "pytoce_principal_inventory_v1::" + payload_sha256(
                    {
                        "timeline_identity": timeline_identity,
                        "principal_inventory_method": (PRINCIPAL_INVENTORY_METHOD),
                    }
                )
                inventory_rows.append(
                    {
                        "foreground_scenario": foreground_scenario,
                        "background_scenario": background_scenario,
                        **background_dimensions,
                        "region": region,
                        "foreground_database": foreground.database_name,
                        "regional_root_database": roots[region][0],
                        "regional_root_code": roots[region][1],
                        "timeline_identity": timeline_identity,
                        "principal_inventory_identity": principal_inventory_identity,
                        "timeline_rows": len(principal.timeline),
                        "dynamic_inventory_rows": len(principal.inventory),
                        "provider_matrix_count": (principal.provider_matrix_count),
                        "provider_count": principal.provider_count,
                        "provider_solve_batch_count": (principal.provider_solve_batch_count),
                        "base_static_score": principal.base_static_score,
                    }
                )
                del principal, characterized
            print(
                f"run_lca: {foreground_scenario}, {background_scenario}: "
                "writing contribution tables",
                file=sys.stderr,
            )
            del bank
            del factors
            del timelines
            gc.collect()
            contribution_paths = {
                "contributions": "contributions.parquet",
                "stage_contributions": "stage_contributions.parquet",
                "annual_stage_scores": "annual_life_cycle_stage_scores.parquet",
                "cumulative_stage_scores": "life_cycle_stage_scores.parquet",
                "structural_paths": "structural_paths.parquet",
                "structural_edges": "structural_edges.parquet",
                "structural_attribution": "structural_attribution.parquet",
            }
            scenario_tables = {
                "contributions",
                "stage_contributions",
                "annual_stage_scores",
                "cumulative_stage_scores",
            }
            contribution_artifacts: dict[str, dict[str, object]] = {}
            finalized_tables = finalize_contributions(contribution_state)
            _, annual_norms = next(finalized_tables)
            for table_name, table in finalized_tables:
                if table_name in scenario_tables:
                    table.insert(0, "foreground_scenario", foreground_scenario)
                    table.insert(1, "background_scenario", background_scenario)
                artifact_name = contribution_paths[table_name]
                contribution_artifacts[artifact_name] = _table(
                    runtime_directory / artifact_name,
                    table,
                )
                del table
            del finalized_tables
            del contribution_state
            gc.collect()
            score_table, annual_table = attach_contribution_norms(
                pd.concat(scores, ignore_index=True),
                pd.concat(annual_scores, ignore_index=True),
                annual_norms,
            )
            del annual_norms
            del scores
            del annual_scores
            inventory_table = pd.DataFrame(inventory_rows)
            del inventory_rows
            mechanical_recycling_loss = foreground.mechanical_recycling_loss.copy()
            mechanical_recycling_loss.insert(
                1,
                "background_scenario",
                background_scenario,
            )
            runtime_artifacts: dict[str, dict[str, object]] = {}
            for name, frames in (
                ("pa_receiving_inventory.parquet", pa_records),
                ("pa_receiving_physical_events.parquet", pa_physical_records),
            ):
                table = pd.concat(
                    [frame for frame in frames if not frame.empty] or frames[:1],
                    ignore_index=True,
                )
                table.insert(0, "foreground_scenario", foreground_scenario)
                table.insert(1, "background_scenario", background_scenario)
                runtime_artifacts[name] = _table(runtime_directory / name, table)
                frames.clear()
                del table
            del pa_records, pa_physical_records
            runtime_artifacts["scores.parquet"] = _table(
                runtime_directory / "scores.parquet",
                score_table,
            )
            del score_table
            runtime_artifacts["annual_scores.parquet"] = _table(
                runtime_directory / "annual_scores.parquet",
                annual_table,
            )
            del annual_table
            runtime_artifacts["annual_life_cycle_stage_scores.parquet"] = (
                contribution_artifacts.pop("annual_life_cycle_stage_scores.parquet")
            )
            runtime_artifacts["life_cycle_stage_scores.parquet"] = contribution_artifacts.pop(
                "life_cycle_stage_scores.parquet"
            )
            runtime_artifacts["inventory_authority.parquet"] = _table(
                runtime_directory / "inventory_authority.parquet",
                inventory_table,
            )
            del inventory_table
            runtime_artifacts["mechanical_recycling_loss.parquet"] = _table(
                runtime_directory / "mechanical_recycling_loss.parquet",
                mechanical_recycling_loss,
            )
            del mechanical_recycling_loss
            runtime_artifacts["mfa_to_lca_flow_coverage.parquet"] = _table(
                runtime_directory / "mfa_to_lca_flow_coverage.parquet",
                flow_coverage,
            )
            del flow_coverage
            runtime_artifacts["contributions.parquet"] = contribution_artifacts.pop(
                "contributions.parquet"
            )
            runtime_artifacts["stage_contributions.parquet"] = contribution_artifacts.pop(
                "stage_contributions.parquet"
            )
            runtime_artifacts["structural_paths.parquet"] = contribution_artifacts.pop(
                "structural_paths.parquet"
            )
            runtime_artifacts["structural_edges.parquet"] = contribution_artifacts.pop(
                "structural_edges.parquet"
            )
            runtime_artifacts["structural_attribution.parquet"] = contribution_artifacts.pop(
                "structural_attribution.parquet"
            )
            gc.collect()
            print(
                f"run_lca: {foreground_scenario}, {background_scenario}: writing provider tables",
                file=sys.stderr,
            )
            activation_lineage = restore_activation_lineage(activation_lineage_spool)
            activation_lineage_artifact = _table(
                runtime_directory / "activation_lineage.parquet",
                activation_lineage,
            )
            del activation_lineage
            gc.collect()
            carbon_activations = _restore_resolved_activations(
                activation_spool,
                columns=_CARBON_REPORTING_ACTIVATION_COLUMNS,
                life_cycle_stages=tuple(sorted(TREATMENT_STAGES)),
            )
            carbon_consistency = carbon_consistency_table(bd, carbon_activations)
            carbon_consistency.insert(0, "foreground_scenario", foreground_scenario)
            carbon_consistency.insert(1, "background_scenario", background_scenario)
            runtime_artifacts["carbon_consistency.csv"] = _csv_table(
                runtime_directory / "carbon_consistency.csv",
                carbon_consistency,
            )
            del carbon_consistency
            direct_treatment_carbon = direct_treatment_carbon_flows(
                bd,
                carbon_activations,
            )
            actual_carbon = [frame for frame in pa_carbon_records if not frame.empty]
            if actual_carbon:
                if not direct_treatment_carbon.empty:
                    actual_carbon.insert(0, direct_treatment_carbon)
                direct_treatment_carbon = pd.concat(
                    actual_carbon,
                    ignore_index=True,
                )
            del actual_carbon
            del pa_carbon_records
            direct_treatment_carbon.insert(0, "foreground_scenario", foreground_scenario)
            direct_treatment_carbon.insert(1, "background_scenario", background_scenario)
            runtime_artifacts["direct_treatment_carbon_flows.parquet"] = _table(
                runtime_directory / "direct_treatment_carbon_flows.parquet",
                direct_treatment_carbon,
            )
            del direct_treatment_carbon
            del carbon_activations
            gc.collect()
            runtime_artifacts["activation_lineage.parquet"] = activation_lineage_artifact
            mismanaged_activations = _restore_resolved_activations(
                activation_spool,
                columns=MISMANAGED_ACTIVATION_COLUMNS,
                life_cycle_stages=MISMANAGED_STAGES,
            )
            mismanaged_waste_fate, mismanaged_waste_fate_closure = mismanaged_result_tables(
                bd,
                mismanaged_activations,
                foreground_scenario=foreground_scenario,
                background_scenario=background_scenario,
            )
            runtime_artifacts["mismanaged_waste_fate.parquet"] = _table(
                runtime_directory / "mismanaged_waste_fate.parquet",
                mismanaged_waste_fate,
            )
            del mismanaged_waste_fate
            runtime_artifacts["mismanaged_waste_fate_closure.parquet"] = _table(
                runtime_directory / "mismanaged_waste_fate_closure.parquet",
                mismanaged_waste_fate_closure,
            )
            del mismanaged_waste_fate_closure
            del mismanaged_activations
            gc.collect()
        identity_payload: dict[str, object] = {
            "schema_version": _RESULT_SCHEMA,
            "foreground_scenario": foreground_scenario,
            "background_scenario": background_scenario,
            **background_dimensions,
            "mfa_result_identity": mfa_manifest["identity"],
            "mfa_manifest_sha256": mfa_manifest_sha256,
            "prepared_background_identity": background_manifest["identity"],
            "prepared_background_manifest_sha256": background_manifest_sha256,
            "foreground_identity": foreground.foreground_identity,
            "scientific_software": _scientific_software(),
            "lcia_methods": [
                {
                    "method_id": item.method_id,
                    "method_version": item.method_version,
                    "impact_category_id": item.impact_category_id,
                    "method_tuple": list(item.method_tuple),
                    "factor_identity": item.factor_identity,
                }
                for item in methods
            ],
            "upstream_analysis_year": list(years),
            "timex": timex_arguments,
            "contribution_method_identity": CONTRIBUTION_METHOD_IDENTITY,
            "carbon_consistency_method_identity": CARBON_CONSISTENCY_METHOD_IDENTITY,
            "direct_treatment_carbon_flows_identity": (DIRECT_TREATMENT_CARBON_FLOWS_IDENTITY),
            "flow_ownership_method_identity": FLOW_OWNERSHIP_METHOD_IDENTITY,
            "mismanaged_waste_fate_method": (
                "cottom_2024_open_burning_nonburned_split_and_native_direct_treatment"
            ),
        }
        identity = f"{_RESULT_SCHEMA}::{payload_sha256(identity_payload)}"
        artifacts = dict(runtime_artifacts)
        atomic_write(
            runtime_directory / "execution_report.json",
            canonical_json_bytes(execution_report),
        )
        artifacts["execution_report.json"] = {
            "path": "execution_report.json",
            "artifact_sha256": file_sha256(runtime_directory / "execution_report.json"),
        }
        manifest = {
            "schema_version": _RESULT_SCHEMA,
            "identity": identity,
            "foreground_scenario": foreground_scenario,
            "background_scenario": background_scenario,
            "ssp_scenario": "SSP2",
            "identity_payload": identity_payload,
            "artifacts": artifacts,
            "complete": True,
        }
        write_workspace_guide(
            runtime_directory,
            "pytoce.run_lca",
            "lca_result",
        )
        atomic_write(
            runtime_directory / "manifest.json",
            canonical_json_bytes(manifest),
        )
        replace_directory(runtime_directory, result_directory, replace_existing=replacing)
        return "updated" if replacing else "computed"
