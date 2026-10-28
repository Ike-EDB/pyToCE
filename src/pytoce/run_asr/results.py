"""Read matched monitoring inputs and write annual and cumulative ASR tables."""

import json
import shutil
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from pytoce.run_lca.calculation.lcia import LCIAMethodSelection, readable_lcia_method_selection
from pytoce.run_lca.calculation.result_inputs import LCAResultReference, select_lca_result
from pytoce.run_optimization.input.allocated_cc import (
    select_allocated_cc_cases,
    select_allocated_cc_result,
)
from pytoce.run_optimization.input.allocated_cc_contract import AllocatedCCSelection, FloatArray
from pytoce.run_optimization.input.allocated_cc_product import compile_allocated_cc_result
from pytoce.shared.assessment.asr import calculate_asr
from pytoce.shared.runtime.file_lock import writer_lock
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

_SCHEMA = "pytoce_asr_result_schema_v1"
_FILES = {"annual_asr.parquet", "cumulative_gwp100_asr.parquet", "manifest.json", "README.md"}
_METHOD_COLUMNS = (
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "unit",
)


def _score_array(
    reference: LCAResultReference,
    methods: pd.DataFrame,
    capacities: AllocatedCCSelection,
) -> FloatArray:
    """Admit a complete annual EF table with the same methods, factors and units."""
    path = reference.directory / "annual_scores.parquet"
    descriptor = reference.manifest["artifacts"][path.name]
    if file_sha256(path) != descriptor["artifact_sha256"]:
        raise ValueError(f"LCA score file differs from its completion manifest: {path}")
    recorded = {
        (item["method_id"], item["impact_category_id"]): item
        for item in reference.manifest["identity_payload"]["lcia_methods"]
    }
    for row in methods.itertuples(index=False):
        method = recorded.get((row.method_id, row.impact_category_id))
        if method is None or (
            method["method_version"],
            method["method_tuple"],
            method["factor_identity"],
        ) != (
            row.method_version,
            json.loads(str(row.impact_category_tuple_json)),
            row.factor_identity,
        ):
            raise ValueError(
                f"LCA and aCC methods or characterization factors differ for "
                f"{row.impact_category_id}. Recalculate LCA with the intended prepared methods."
            )
    frame = pd.read_parquet(path)
    columns = ["region", "year", *_METHOD_COLUMNS]
    # Recalculate global ratios from the regional numerators and denominators.
    ef = frame.loc[frame["method_id"].eq("ef31") & frame["region"].ne("global_derived")].set_index(
        columns
    )
    expected = pd.MultiIndex.from_frame(
        pd.DataFrame.from_records(
            [
                (region, year, *signature)
                for region, year, signature in product(
                    capacities.regions,
                    capacities.years,
                    methods.loc[:, list(_METHOD_COLUMNS)].itertuples(index=False, name=None),
                )
            ],
            columns=columns,
        )
    )
    if (
        not ef.index.is_unique
        or len(ef) != len(expected)
        or not ef.index.difference(expected).empty
    ):
        raise ValueError(
            f"Annual LCA scores must cover exactly the twelve regions, 2019--2100, and "
            f"sixteen matched EF v3.1 categories with matching units: {path}"
        )
    values = ef.reindex(expected)["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Annual LCA scores must be finite: {path}")
    return values.reshape(len(capacities.regions), len(capacities.years), len(methods))


def _assessment_tables(
    scores: FloatArray,
    capacities: AllocatedCCSelection,
    methods: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Format regional and global assessments from the canonical shared ASR calculation."""
    climate_index = capacities.annual_categories.index(("ef31", "climate_change"))
    annual_rows: list[dict[str, object]] = []
    cumulative_rows: list[dict[str, object]] = []
    signatures = [
        {column: str(getattr(row, column)) for column in _METHOD_COLUMNS}
        for row in methods.itertuples(index=False)
    ]
    for case_index, case in enumerate(capacities.allocated_cc_cases):
        annual_acc = capacities.annual_allocated_cc[case_index]
        cumulative_acc = capacities.cumulative_allocated_cc[case_index]
        values = calculate_asr(
            annual_scores=scores,
            annual_allocated_cc=annual_acc,
            annual_budget_compatible_gwp100_scores=scores[:, :, climate_index],
            cumulative_gwp100_allocated_cc=cumulative_acc,
            control_start=capacities.years.index(2020),
        )
        world = values.global_metrics
        region_scores = np.concatenate((scores, world.annual_scores[None, ...]))
        region_acc = np.concatenate((annual_acc, world.annual_allocated_cc[None, ...]))
        region_asr = np.concatenate((values.annual_asr, world.annual_asr[None, ...]))
        cumulative_scores = (*values.cumulative_gwp100_scores, world.cumulative_gwp100_score)
        cumulative_capacities = (*cumulative_acc, world.cumulative_gwp100_allocated_cc)
        cumulative_asrs = (*values.cumulative_gwp100_asr, world.cumulative_gwp100_asr)
        for region_index, region in enumerate((*capacities.regions, "global")):
            for year_index, year in enumerate(capacities.years):
                for category_index, signature in enumerate(signatures):
                    asr = float(region_asr[region_index, year_index, category_index])
                    annual_rows.append(
                        {
                            "allocated_cc_case": case,
                            "region": region,
                            "year": year,
                            "lcia_method": "ef31",
                            **signature,
                            "score": float(region_scores[region_index, year_index, category_index]),
                            "allocated_cc": float(
                                region_acc[region_index, year_index, category_index]
                            ),
                            "asr": asr,
                            "compliant": asr <= 1.0,
                            "assessed_for_compliance": year >= 2020,
                        }
                    )
            asr = float(cumulative_asrs[region_index])
            cumulative_rows.append(
                {
                    "allocated_cc_case": case,
                    "region": region,
                    "period_start": 2019,
                    "period_end": 2100,
                    "lcia_method": "ef31",
                    **signatures[climate_index],
                    "score": float(cumulative_scores[region_index]),
                    "allocated_cc": float(cumulative_capacities[region_index]),
                    "asr": asr,
                    "compliant": asr <= 1.0,
                }
            )
    return {
        "annual_asr.parquet": pd.DataFrame.from_records(annual_rows),
        "cumulative_gwp100_asr.parquet": pd.DataFrame.from_records(cumulative_rows),
    }


def write_asr_result(
    workspace: Path,
    *,
    foreground: str,
    background: str,
    allocated_cc_result: str,
    cases: tuple[str, ...] | None,
    selection: LCIAMethodSelection,
    refresh: bool,
) -> tuple[ItemStatus, Path]:
    """Reuse or atomically write one completed monitoring result under a writer lock."""
    case_token = "all_cases" if cases is None else "cases_" + payload_sha256(list(cases))[:16]
    name = (
        f"{background}__{readable_lcia_method_selection(selection)}"
        f"__{allocated_cc_result}__{case_token}"
    )
    if len(name) > 212:
        raise ValueError("ASR result name exceeds 212 characters; use shorter input names")
    directory = workspace / "results" / foreground / "asr_results" / name
    directory.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(directory.parent / ".writer.lock"):
        manifest_path = directory / "manifest.json"
        complete = manifest_path.is_file()
        if complete and not refresh:
            return "reused", directory
        if directory.exists() or directory.is_symlink():
            if (
                not complete
                or directory.is_symlink()
                or any(
                    path.name not in _FILES or path.is_symlink() or not path.is_file()
                    for path in directory.iterdir()
                )
            ):
                raise ValueError(
                    f"Cannot replace {directory}: use a completed ASR result containing only "
                    "ordinary result files. Existing files have not been changed."
                )
        reference = select_lca_result(
            workspace,
            foreground_scenario=foreground,
            background_scenario=background,
            selection=selection,
        )
        result = compile_allocated_cc_result(workspace, allocated_cc_result)
        selected_cases = select_allocated_cc_cases(result, cases)
        capacities = select_allocated_cc_result(result, selected_cases)
        methods = pd.read_parquet(result.result_directory / "lcia_method_registry.parquet")
        scores = _score_array(reference, methods, capacities)
        tables = _assessment_tables(scores, capacities, methods)
        identity_payload = {
            "schema_version": _SCHEMA,
            "foreground_scenario": foreground,
            "background_scenario": background,
            "lca_result_identity": reference.manifest["identity"],
            "lca_manifest_sha256": reference.manifest_sha256,
            "allocated_cc_selection_identity": capacities.allocated_cc_selection_identity,
        }
        staging = Path(tempfile.mkdtemp(prefix=".asr.", dir=directory.parent))
        try:
            artifacts = {}
            for filename, frame in tables.items():
                frame.insert(0, "background_scenario", background)
                frame.insert(0, "foreground_scenario", foreground)
                path = staging / filename
                frame.to_parquet(path, index=False)
                artifacts[filename] = {
                    "path": filename,
                    "artifact_sha256": file_sha256(path),
                    "canonical_sha256": canonical_table_sha256(frame),
                    "rows": len(frame),
                }
            write_workspace_guide(staging, "pytoce.run_asr", "asr_result")
            manifest = {
                "schema_version": _SCHEMA,
                "complete": True,
                "identity": f"{_SCHEMA}::{payload_sha256(identity_payload)}",
                "identity_payload": identity_payload,
                "allocated_cc_result": allocated_cc_result,
                "allocated_cc_cases": list(selected_cases),
                "lca_result_directory": str(reference.directory.relative_to(workspace)),
                "artifacts": artifacts,
            }
            atomic_write(staging / "manifest.json", canonical_json_bytes(manifest))
            replace_directory(staging, directory, replace_existing=complete)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    return ("updated" if complete else "computed"), directory
