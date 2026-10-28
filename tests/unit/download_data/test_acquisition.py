import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import py7zr
import pytest

from pytoce import download_data
from pytoce.download_data.download_data import _download_from_catalogue
from pytoce.download_data.sources import catalogue
from pytoce.download_data.sources.catalogue import SourceSpec
from pytoce.download_data.sources.ecoinvent import (
    _extract_lcia_workbook,
)
from pytoce.download_data.sources.network import DownloadTransport
from pytoce.set_workspace.set_workspace import set_workspace
from pytoce.shared.runtime.filesystem import payload_sha256


def _place_licensed_archives(
    workspace: Path,
    specs: tuple[SourceSpec, ...],
    sources: tuple[Path, ...],
) -> None:
    root = workspace / "data_raw" / "ecoinvent"
    root.mkdir(parents=True, exist_ok=True)
    for spec, source in zip(specs, sources, strict=True):
        shutil.copyfile(source, root / spec.filename)


def test_packaged_catalogue_binds_locked_source_authorities() -> None:
    open_sources, licensed = catalogue.load_catalogue()

    assert [(source.source_id, source.filename) for source in open_sources[:2]] == [
        ("remind_mfa_code", "remind_mfa_v0_2_0.tar.gz"),
        ("remind_mfa_input_data", "input_data_1_0_0.zip"),
    ]
    assert [
        (source.source_id, source.version, source.revision, source.license)
        for source in open_sources
    ] == [
        (
            "remind_mfa_code",
            "0.2.0",
            "bf95c898e0f6f363196bccb62d34e02d82e4726b",
            "GPL-3.0-only",
        ),
        (
            "remind_mfa_input_data",
            "1.0.0",
            "rev1.0.0_h12_1584d780_mfa",
            "CC-BY-SA-4.0",
        ),
        (
            "aesa_pb_lcia",
            "0.9.2",
            "00003e27a04bfc5662bf561cf7a092ba5324d67c",
            "BSD-3-Clause",
        ),
    ]
    assert open_sources[1].sha256 == (
        "73bcf80f53513dcdf6f946afabb333a13792f087029550a95ea6c52f8d068419"
    )
    assert [source.source_id for source in licensed] == [
        "ecoinvent_3_12_cutoff_ecospold02",
        "ecoinvent_3_12_lcia_implementation",
    ]
    assert licensed[0].sha256 == (
        "c00cf9a5704f2b8139995e8af9a62c5a432bd33b60f3c978788073e82bdc416f"
    )
    assert licensed[0].metadata["system_model"] == "Allocation, cut-off by classification"
    assert licensed[0].metadata["ecospold_member_count"] == 26533
    assert licensed[1].sha256 == (
        "ac6d6de67b7c46be0b935554d1e851811e3073dba8e8364e52fa27c409af0c53"
    )
    assert licensed[1].metadata["workbook_sha256"] == (
        "6bc4e6475877e8e90d8a6184b681366154cc5f6ec42c7cec54eb871224b33e02"
    )


@pytest.mark.parametrize("initial_refresh", (False, True))
def test_download_data_uses_flat_readable_source_owners(
    tmp_path: Path,
    acquisition_catalogue: tuple[
        tuple[SourceSpec, ...],
        tuple[SourceSpec, ...],
        tuple[Path, ...],
    ],
    acquisition_transport: DownloadTransport,
    initial_refresh: bool,
) -> None:
    open_sources, licensed, archives = acquisition_catalogue
    workspace = tmp_path / "workspace"
    set_workspace(workspace)
    _place_licensed_archives(workspace, licensed, archives)

    report = _download_from_catalogue(
        workspace,
        open_sources=open_sources,
        ecoinvent_specs=licensed,
        refresh=initial_refresh,
        started=perf_counter(),
        transport=acquisition_transport,
    )

    assert report.status == "computed"
    assert [source.source_id for source in report.sources] == [
        "remind_mfa_code",
        "remind_mfa_input_data",
        "aesa_pb_lcia",
        "ecoinvent_3_12_cutoff_ecospold02",
        "ecoinvent_3_12_lcia_implementation",
    ]
    assert {source.status for source in report.sources} == {"computed"}
    assert all(source.artifact_path.is_file() for source in report.sources)
    assert all(source.manifest_path.is_file() for source in report.sources)
    assert {path.name for path in (workspace / "data_raw").iterdir() if path.is_dir()} == {
        "allocated_cc",
        "ecoinvent",
        "foreground_scenarios",
        "methodological_notes",
        "pb_lcia",
        "remind_mfa",
    }
    assert not (workspace / "data_raw" / "licensed_input").exists()
    assert not (workspace / "data_raw" / "source_data").exists()
    assert not list(workspace.rglob("summary.log"))
    assert len({source.manifest_path for source in report.sources}) == 3
    licensed_manifest = report.sources[-1].manifest_path.read_text(encoding="utf-8")
    assert all(str(archive.resolve()) not in licensed_manifest for archive in archives)
    assert "bw2data" not in licensed_manifest
    assert (workspace / "data_raw" / "ecoinvent" / "lcia_implementation.xlsx").read_bytes() == (
        tmp_path / "lcia_source" / str(licensed[1].metadata["workbook_member"])
    ).read_bytes()
    recipe_path = workspace / "data_processed" / "ecoinvent" / "material_recipe.json"
    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    identity = recipe.pop("identity")
    assert identity == f"{recipe['schema_version']}::{payload_sha256(recipe)}"
    assert recipe["source_identity"] == f"{licensed[0].source_id}::{licensed[0].sha256}"
    assert (recipe_path.parent / "README.md").is_file()
    assert not list(recipe_path.parent.parent.glob(".ecoinvent_materials.*"))

    report.sources[0].artifact_path.unlink()
    refreshed = _download_from_catalogue(
        workspace,
        open_sources=open_sources,
        ecoinvent_specs=licensed,
        refresh=True,
        started=perf_counter(),
        transport=acquisition_transport,
    )
    assert refreshed.status == "updated"
    assert [source.status for source in refreshed.sources] == [
        "updated",
        "updated",
        "updated",
        "reused",
        "reused",
    ]
    assert recipe_path.read_bytes() == recipe_bytes
    assert (
        report.sources[0].artifact_path.read_bytes()
        == (tmp_path / "official_sources" / open_sources[0].filename).read_bytes()
    )

    reused = download_data()
    assert reused.status == "reused"
    assert reused.as_dict() == {
        "status": "reused",
        "workspace": str(workspace),
        "elapsed_seconds": reused.elapsed_seconds,
        "sources": [source.as_dict() for source in reused.sources],
    }


def test_corrupt_open_source_fails_before_owner_manifest(
    tmp_path: Path,
    acquisition_catalogue: tuple[
        tuple[SourceSpec, ...],
        tuple[SourceSpec, ...],
        tuple[Path, ...],
    ],
    acquisition_transport: DownloadTransport,
) -> None:
    open_sources, licensed, archives = acquisition_catalogue
    bad = replace(open_sources[0], sha256="0" * 64)
    workspace = tmp_path / "workspace"
    set_workspace(workspace)
    _place_licensed_archives(workspace, licensed, archives)

    with pytest.raises(ValueError, match="SHA-256"):
        _download_from_catalogue(
            workspace,
            open_sources=(bad, *open_sources[1:]),
            ecoinvent_specs=licensed,
            refresh=False,
            started=perf_counter(),
            transport=acquisition_transport,
        )

    assert not (workspace / "data_raw" / "remind_mfa" / "manifest.json").exists()


def test_download_data_requires_initialized_workspace(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "from pytoce import download_data; download_data()"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "set_workspace" in completed.stderr


@pytest.mark.parametrize("source_index", (0, 1), ids=("ecospold", "lcia"))
@pytest.mark.parametrize("changed_size", (False, True), ids=("digest", "size"))
def test_changed_licensed_archive_cannot_publish_workbook_or_manifest(
    tmp_path: Path,
    acquisition_catalogue: tuple[tuple[SourceSpec, ...], tuple[SourceSpec, ...], tuple[Path, ...]],
    acquisition_transport: DownloadTransport,
    source_index: int,
    changed_size: bool,
) -> None:
    """Reject changed external sources and preserve the source files."""
    open_sources, licensed, archives = acquisition_catalogue
    changed = bytearray(archives[source_index].read_bytes())
    if changed_size:
        changed.append(0)
    else:
        changed[-1] ^= 1
    archives[source_index].write_bytes(changed)
    workspace = tmp_path / "workspace"
    set_workspace(workspace)
    _place_licensed_archives(workspace, licensed, archives)
    root = workspace / "data_raw" / "ecoinvent"
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    with pytest.raises(ValueError, match="byte length" if changed_size else "SHA-256"):
        _download_from_catalogue(
            workspace,
            open_sources=open_sources,
            ecoinvent_specs=licensed,
            refresh=False,
            started=perf_counter(),
            transport=acquisition_transport,
        )

    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert not (root / "manifest.json").exists()
    assert not (root / "lcia_implementation.xlsx").exists()


def test_lcia_extraction_uses_the_admitted_archive_bytes(
    tmp_path: Path,
    acquisition_catalogue: tuple[tuple[SourceSpec, ...], tuple[SourceSpec, ...], tuple[Path, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep extraction bound to authenticated content when the raw path changes."""
    _open, licensed, archives = acquisition_catalogue
    spec = licensed[1]
    archive_path = archives[1]
    expected = (tmp_path / "lcia_source" / str(spec.metadata["workbook_member"])).read_bytes()
    replacement = tmp_path / "replacement.7z"
    with py7zr.SevenZipFile(replacement, mode="w") as archive:
        archive.writestr(b"different workbook bytes", str(spec.metadata["workbook_member"]))
    replacement_bytes = replacement.read_bytes()
    original_reader = py7zr.SevenZipFile

    def open_after_source_change(*args: object, **kwargs: object) -> py7zr.SevenZipFile:
        """Change only the external file, then perform real archive parsing."""
        archive_path.write_bytes(replacement_bytes)
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(py7zr, "SevenZipFile", open_after_source_change)
    destination = tmp_path / "implementation.xlsx"
    _extract_lcia_workbook(archive_path, spec, destination=destination)
    assert destination.read_bytes() == expected
    assert archive_path.read_bytes() == replacement_bytes
    assert not list(tmp_path.glob(".ecoinvent_lcia.*"))
