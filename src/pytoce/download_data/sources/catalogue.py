"""Source catalog, readable storage paths, and source manifests."""

import json
from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import cast

from pytoce.shared.runtime.filesystem import (
    canonical_json_bytes,
    payload_sha256,
)
from pytoce.shared.runtime.manifests import ItemStatus

SOURCE_SCHEMA = "pytoce_source_owner_schema_v1"
CATALOGUE_RESOURCE = "source_catalogue.json"
DOWNLOAD_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Describe one file required by ``download_data``.

    Attributes:
        source_id: Stable identifier used in source manifests.
        kind: Declared source category, such as ``source_code`` or ``licensed_lcia_factors``.
        version: Version published by the source project.
        revision: Commit, release, or other immutable upstream revision.
        filename: Required filename in the workspace source directory.
        size_bytes: Expected file size in bytes.
        sha256: Expected lowercase SHA-256 digest.
        license: License identifier or licensed-data notice.
        citation: Citation text recorded for the source.
        metadata: Remaining catalog fields included in source manifests.
        url: Download URL for an open source, or ``None`` for a licensed input.
    """

    source_id: str
    kind: str
    version: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str
    license: str
    citation: str
    metadata: dict[str, object]
    url: str | None = None


@dataclass(frozen=True, slots=True)
class AcquiredSource:
    """One completed source file's catalog declarations and local paths.

    Version, revision and digest come from the packaged catalog. Completed reuse
    uses these declarations after finding the completion manifest.

    Attributes:
        source_id: Stable source identifier from the packaged catalog.
        status: Whether the file was ``computed``, ``updated``, or ``reused``.
        version: Declared upstream source version.
        revision: Declared immutable source revision.
        license: Source license identifier or licensed-data notice.
        artifact_path: Recorded local artifact path.
        artifact_sha256: Expected source digest declared by the packaged catalog;
            checked during download or registration.
        manifest_path: Adjacent completion and provenance manifest.
    """

    source_id: str
    status: ItemStatus
    version: str
    revision: str
    license: str
    artifact_path: Path
    artifact_sha256: str
    manifest_path: Path

    def as_dict(self) -> dict[str, object]:
        """Serialize the source file record with paths represented as text."""

        return {
            "source_id": self.source_id,
            "status": self.status,
            "version": self.version,
            "revision": self.revision,
            "license": self.license,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "manifest_path": str(self.manifest_path),
        }


def _parse_source(item: dict[str, object], *, licensed: bool) -> SourceSpec:
    """Convert one catalog record to a typed source specification."""
    metadata = {
        str(key): value
        for key, value in item.items()
        if key not in {"url", "filename", "bytes", "sha256"}
    }
    return SourceSpec(
        source_id=str(item["source_id"]),
        kind=str(item["kind"]),
        version=str(item["version"]),
        revision=str(item["revision"]),
        filename=str(item["filename"]),
        size_bytes=int(cast(int, item["bytes"])),
        sha256=str(item["sha256"]).lower(),
        license=str(item["license"]),
        citation=str(item["citation"]),
        metadata=metadata,
        url=None if licensed else str(item["url"]),
    )


def load_catalogue() -> tuple[tuple[SourceSpec, ...], tuple[SourceSpec, ...]]:
    """Load the package-owned source catalog."""

    node = resources.files("pytoce.download_data.data")
    for part in PurePosixPath(CATALOGUE_RESOURCE).parts:
        node = node.joinpath(part)
    raw: dict[str, list[dict[str, object]]] = json.loads(node.read_text(encoding="utf-8"))
    return (
        tuple(_parse_source(item, licensed=False) for item in raw["open_sources"]),
        tuple(_parse_source(item, licensed=True) for item in raw["licensed_inputs"]),
    )


def source_directory(workspace: Path, spec: SourceSpec) -> Path:
    """Return the readable owner directory for one source."""

    if spec.source_id.startswith("remind_mfa_"):
        owner = "remind_mfa"
    elif spec.source_id == "aesa_pb_lcia":
        owner = "pb_lcia"
    else:
        owner = "ecoinvent"
    return workspace / "data_raw" / owner


def read_verified_source(path: Path, *, expected_bytes: int, expected_sha256: str) -> bytes:
    """Read bounded source bytes and authenticate that exact content for its consumer."""
    with path.open("rb") as stream:
        content = stream.read(expected_bytes + 1)
    if len(content) != expected_bytes:
        raise ValueError(
            f"The source file {path} has an unexpected byte length; "
            f"expected exactly {expected_bytes} bytes. "
            "Replace it with the documented file and try again."
        )
    if sha256(content).hexdigest() != expected_sha256:
        raise ValueError(
            f"The source file {path} does not match its expected SHA-256 digest. "
            "Replace it with the documented file and try again."
        )
    return content


def owner_manifest(specs: tuple[SourceSpec, ...]) -> dict[str, object]:
    """Build one manifest for all files in the same source directory."""

    records = [
        {
            "source_id": spec.source_id,
            "kind": spec.kind,
            "version": spec.version,
            "revision": spec.revision,
            "filename": spec.filename,
            "bytes": spec.size_bytes,
            "sha256": spec.sha256,
            "license": spec.license,
            "citation": spec.citation,
            "source_url": spec.url,
            "source_metadata": spec.metadata,
        }
        for spec in specs
    ]
    identity_payload = {
        "schema_version": SOURCE_SCHEMA,
        "sources": records,
    }
    return {
        "schema_version": SOURCE_SCHEMA,
        "identity": f"{SOURCE_SCHEMA}::{payload_sha256(identity_payload)}",
        "identity_payload": identity_payload,
        "complete": True,
    }


def owner_complete(root: Path) -> bool:
    """Return whether a source directory has its completion manifest."""

    return (root / "manifest.json").is_file()


def validate_source_owner_for_refresh(root: Path, specs: tuple[SourceSpec, ...]) -> None:
    """Restrict refresh deletion to the completed owner's regular files."""

    expected_names = {spec.filename for spec in specs} | {"README.md", "manifest.json"}
    if root.is_symlink():
        raise ValueError(
            f"Cannot refresh {root} because it is a symbolic link. A package source "
            "directory is required. Keep it unchanged and create a new workspace."
        )
    for artifact in root.iterdir():
        if artifact.name not in expected_names:
            raise ValueError(
                f"Cannot refresh {root} because it contains {artifact}. Only catalogued "
                "source files, README.md and manifest.json may be replaced. Keep it "
                "unchanged and create a new workspace."
            )
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(
                f"Cannot refresh {root} because {artifact} is not a regular file. Keep it "
                "unchanged and create a new workspace."
            )


def source_reports(
    root: Path,
    specs: tuple[SourceSpec, ...],
    status: ItemStatus,
) -> tuple[AcquiredSource, ...]:
    """Create report items for one completed source directory."""

    manifest_path = root / "manifest.json"
    return tuple(
        AcquiredSource(
            source_id=spec.source_id,
            status=status,
            version=spec.version,
            revision=spec.revision,
            license=spec.license,
            artifact_path=root / spec.filename,
            artifact_sha256=spec.sha256,
            manifest_path=manifest_path,
        )
        for spec in specs
    )


def manifest_bytes(specs: tuple[SourceSpec, ...]) -> bytes:
    """Serialize one source directory manifest."""

    return canonical_json_bytes(owner_manifest(specs))
