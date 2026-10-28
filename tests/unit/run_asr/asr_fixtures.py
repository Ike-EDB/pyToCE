"""Small completed LCA results and external aCC inputs for monitoring tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from pytoce.prepare_backgrounds.preparation.project_biosphere import load_lcia_catalogue
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_lca.run_lca import lca_result_directory
from pytoce.run_optimization.input import allocated_cc_product
from pytoce.run_optimization.input.allocated_cc_contract import _h12_regions, _MethodAuthority
from pytoce.set_workspace.runtime.workspace import set_active_workspace
from pytoce.shared.runtime.filesystem import canonical_json_bytes, file_sha256, payload_sha256

FOREGROUND = "monitoring"
BACKGROUND = "remind_ssp2_npi"


def methods(groups: tuple[str, ...] = ("ef31",)) -> pd.DataFrame:
    """Use real category definitions with invented characterization identities."""
    catalogue, _ = load_lcia_catalogue()
    return pd.DataFrame.from_records(
        [
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
                "factor_identity": f"invented::{spec.impact_category_id}",
            }
            for spec in catalogue
            if spec.method_id in groups
        ]
    )


def write_lca(
    workspace: Path,
    *,
    background: str = BACKGROUND,
    years=(2019, 2050, 2100),
    groups: tuple[str, ...] = ("ef31",),
) -> Path:
    """Persist annual scores and a minimal completion manifest using the real layout."""
    directory = lca_result_directory(
        workspace,
        FOREGROUND,
        background,
        normalized_lcia_method(groups),
        years,
    )
    directory.mkdir(parents=True)
    method_rows = methods(groups)
    frame = pd.DataFrame.from_records(
        [
            {
                "foreground_scenario": FOREGROUND,
                "background_scenario": background,
                "region": region,
                "year": year,
                **{
                    column: getattr(method, column)
                    for column in (
                        "method_id",
                        "method_version",
                        "impact_category_id",
                        "impact_category_tuple_json",
                        "unit",
                    )
                },
                # A 2019 transgression is reported but excluded from annual compliance.
                "score": 4.0 if year == 2019 else 1.0,
            }
            for region in _h12_regions()
            for year in range(2019, 2101)
            for method in method_rows.itertuples(index=False)
        ]
    )
    global_rows = frame.loc[frame.region.eq(_h12_regions()[0])].copy()
    global_rows["region"] = "global_derived"
    global_rows["score"] = 999.0  # The monitoring calculation must derive its own global ratio.
    pd.concat((frame, global_rows), ignore_index=True).to_parquet(
        directory / "annual_scores.parquet", index=False
    )
    identity = {
        "foreground_scenario": FOREGROUND,
        "background_scenario": background,
        "upstream_analysis_year": list(years),
        "lcia_methods": [
            {
                "method_id": row.method_id,
                "method_version": row.method_version,
                "impact_category_id": row.impact_category_id,
                "method_tuple": json.loads(row.impact_category_tuple_json),
                "factor_identity": row.factor_identity,
            }
            for row in method_rows.itertuples(index=False)
        ],
    }
    manifest = {
        "complete": True,
        "schema_version": "pytoce_lca_result_schema_v1",
        "foreground_scenario": FOREGROUND,
        "background_scenario": background,
        "identity_payload": identity,
        "identity": f"pytoce_lca_result_schema_v1::{payload_sha256(identity)}",
        "artifacts": {
            "annual_scores.parquet": {
                "artifact_sha256": file_sha256(directory / "annual_scores.parquet")
            }
        },
    }
    (directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return directory


@pytest.fixture
def asr_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create synthetic assessment inputs."""
    set_active_workspace(tmp_path)
    frame = methods()
    authority = _MethodAuthority(
        frame=frame,
        cumulative_climate=frame.loc[frame.impact_category_id.eq("climate_change")],
        identity="invented-authority",
        selection_identity="invented-selection",
        prepared_background_shared_identity="invented-shared",
        prepared_background_shared_manifest_sha256="invented-sha",
        prepared_lcia_authority_identity="invented-lcia",
    )
    monkeypatch.setattr(allocated_cc_product, "_method_authority", lambda workspace: authority)
    raw = tmp_path / "data_raw/allocated_cc/results/monitoring_cc"
    raw.mkdir(parents=True)
    annual = pd.DataFrame.from_records(
        [
            {
                "allocated_cc_case": case,
                "lcia_method": "ef31",
                "region": region,
                "impact_category_id": method.impact_category_id,
                "year": year,
                "allocated_cc": scale,
                "unit": method.unit,
                "cc_authority": "invented",
            }
            for case, scale in (("p5", 2.0), ("p95", 4.0))
            for region in _h12_regions()
            for year in range(2019, 2101)
            for method in frame.itertuples(index=False)
        ]
    )
    climate_unit = frame.loc[frame.impact_category_id.eq("climate_change"), "unit"].iloc[0]
    cumulative = pd.DataFrame.from_records(
        [
            {
                "allocated_cc_case": case,
                "region": region,
                "impact_category_id": "climate_change",
                "period_start": 2019,
                "period_end": 2100,
                "cumulative_allocated_cc": scale,
                "unit": climate_unit,
                "budget_metric": "ef31_climate_change_gwp100_all_ghg",
                "cc_authority": "invented",
            }
            for case, scale in (("p5", 85.0), ("p95", 170.0))
            for region in _h12_regions()
        ]
    )
    annual.to_csv(raw / "annual.csv", index=False)
    cumulative.to_csv(raw / "cumulative.csv", index=False)
    (raw / "allocated_cc_result.yaml").write_text(
        "schema_version: pytoce_allocated_cc_result_schema_v1\n"
        "allocated_cc_result: monitoring_cc\ndescription: Invented monitoring inputs\n"
        "ssp_scenario: SSP2\nlcia_method: [ef31]\nannual_allocated_cc: annual.csv\n"
        "cumulative_gwp100_allocated_cc: cumulative.csv\n",
        encoding="utf-8",
    )
    write_lca(tmp_path)
    return tmp_path
