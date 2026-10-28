"""Manifest-bound baseline LCA products shared by optimization owner tests."""

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd

from pytoce.shared.runtime.filesystem import (
    canonical_json_bytes,
    canonical_table_sha256,
    payload_sha256,
)

_SCHEMA = "pytoce_lca_result_schema_v1"
_FOREGROUND = "frozen_2019_default_baseline"
_BACKGROUND = "remind_ssp2_npi"
_METHOD_TUPLE = (
    "EF v3.1",
    "climate change",
    "global warming potential (GWP100)",
)


def _method_columns() -> dict[str, object]:
    """Create the common LCIA method columns used by baseline result tables."""

    return {
        "method_id": "ef31",
        "method_version": "3.1",
        "impact_category_id": "climate_change",
        "impact_category_tuple_json": json.dumps(_METHOD_TUPLE, separators=(",", ":")),
        "unit": "kg CO2-Eq",
    }


def _tables() -> dict[str, pd.DataFrame]:
    """Create the two baseline stage tables consumed by selected reporting."""

    method = _method_columns()
    return {
        "annual_life_cycle_stage_scores.parquet": pd.DataFrame.from_records(
            [
                {
                    "region": "CAZ",
                    "year": 2019,
                    "life_cycle_stage": "virgin_resin_production",
                    **method,
                    "score": 1.0,
                }
            ]
        ),
        "life_cycle_stage_scores.parquet": pd.DataFrame.from_records(
            [
                {
                    "region": "CAZ",
                    "life_cycle_stage": "virgin_resin_production",
                    **method,
                    "score": 1.0,
                }
            ]
        ),
    }


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize one fixture table to real Parquet bytes."""

    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def _write_baseline_result(
    directory: Path,
    *,
    foreground_scenario: str = _FOREGROUND,
    background_scenario: str = _BACKGROUND,
    years: tuple[int, ...] = (2019, 2050, 2100),
    mfa_identity: str = "mfa::parent",
) -> dict[str, object]:
    """Write one canonical baseline LCA result with real Parquet artifacts."""

    directory.mkdir(parents=True)
    artifacts: dict[str, object] = {}
    for name, frame in _tables().items():
        content = _parquet_bytes(frame)
        (directory / name).write_bytes(content)
        artifacts[name] = {
            "path": name,
            "artifact_sha256": sha256(content).hexdigest(),
            "canonical_sha256": canonical_table_sha256(frame),
            "rows": len(frame),
            "columns": list(frame.columns),
            "dtypes": [str(dtype) for dtype in frame.dtypes],
        }
    identity_payload = {
        "foreground_scenario": foreground_scenario,
        "background_scenario": background_scenario,
        "mfa_result_identity": mfa_identity,
        "upstream_analysis_year": list(years),
        "lcia_methods": [
            {
                "method_id": "ef31",
                "method_version": "3.1",
                "impact_category_id": "climate_change",
                "method_tuple": list(_METHOD_TUPLE),
                "factor_identity": "a" * 64,
            }
        ],
    }
    manifest: dict[str, object] = {
        "schema_version": _SCHEMA,
        "complete": True,
        "ssp_scenario": "SSP2",
        "foreground_scenario": foreground_scenario,
        "background_scenario": background_scenario,
        "identity_payload": identity_payload,
        "identity": f"{_SCHEMA}::{payload_sha256(identity_payload)}",
        "artifacts": artifacts,
    }
    (directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest
