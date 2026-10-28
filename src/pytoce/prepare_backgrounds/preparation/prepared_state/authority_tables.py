"""Canonical prepared-background authority-table persistence."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pytoce.shared.runtime.filesystem import file_sha256, payload_sha256

SHARED_SCHEMA = "pytoce_background_shared_authority_schema_v1"
MEMBER_SCHEMA = "pytoce_prepared_background_schema_v1"


class PreparedDataError(ValueError):
    """Required prepared data or Brightway storage is unavailable."""


@dataclass(frozen=True, slots=True)
class PreparedManifest:
    """One completed manifest and the digest of the exact bytes parsed."""

    path: Path
    payload: dict[str, object]
    sha256: str


def read_prepared_manifest(path: Path) -> PreparedManifest:
    """Read a completed manifest and retain its digest for new descendant identities."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise PreparedDataError(
            f"Prepared manifest {path} is unavailable. Run prepare_backgrounds first."
        ) from exc
    return PreparedManifest(path, json.loads(raw), sha256(raw).hexdigest())


def canonical_records(
    records: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return records with one deterministic first-seen column order."""

    columns = tuple(dict.fromkeys(field for record in records for field in record))
    return [{field: record.get(field) for field in columns} for record in records]


def write_table(
    path: Path,
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Write one Parquet authority and return its manifest descriptor."""

    canonical = canonical_records(records)
    table = pa.Table.from_pylist(canonical)
    pq.write_table(table, path)
    return {
        "path": path.name,
        "rows": table.num_rows,
        "columns": list(table.column_names),
        "scientific_identity": payload_sha256(canonical),
        "artifact_sha256": file_sha256(path),
    }


def database_row(name: str, *, role: str) -> dict[str, object]:
    """Describe one database after its authoritative writer completes."""

    import bw2data as bd  # type: ignore[import-untyped]

    metadata = dict(bd.databases[name])
    return {
        "database_name": name,
        "database_role": role,
        "activity_count": metadata.get("number"),
        "depends_json": json.dumps(
            sorted(str(value) for value in metadata.get("depends", [])),
            separators=(",", ":"),
        ),
        "backend": str(metadata.get("backend", "")),
    }
