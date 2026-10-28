"""Prepared table persistence and descendant manifest provenance."""

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    read_prepared_manifest,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    prepared_provider_authority,
)
from tests.integration.brightway_lca_fixtures import write_prepared_inputs


def _prepared_inputs(workspace: Path):
    """Write complete shared metadata and packaged routes using synthetic inputs."""
    provider = prepared_provider_authority()
    shared, _ = write_prepared_inputs(
        workspace,
        BrightwayNames(
            workspace / "data_processed/brightway", "input_fixture", "biosphere", "native"
        ),
        methods=pd.DataFrame(),
        provider=provider,
        carbon_rows=(),
        backgrounds=(),
    )
    records = provider.routes.astype(object).where(provider.routes.notna(), None).to_dict("records")
    return shared, records


def test_prepared_inputs_preserve_exact_records_and_manifest_bytes(tmp_path: Path) -> None:
    """Admit writer-produced descriptors with their original order, nulls, and values."""
    written, records = _prepared_inputs(tmp_path)
    path = written.path
    payload = written.payload
    before = path.read_bytes()
    parent = read_prepared_manifest(path)
    observed = pq.read_table(parent.path.parent / "provider_service_routes.parquet")
    provider = json.loads((parent.path.parent / "provider_service_manifest.json").read_bytes())
    assert observed.equals(pa.Table.from_pylist(records))
    assert provider["identity"] == payload["provider_authority_identity"]
    assert parent.payload == payload
    assert parent.sha256 == sha256(before).hexdigest()
    assert path.read_bytes() == before
