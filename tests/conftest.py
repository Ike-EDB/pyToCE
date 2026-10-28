# Scientific fixtures import Brightway during collection, after storage is isolated below.
# ruff: noqa: E402

import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

# xdist sets this label before loading conftest in each worker. Fresh directories
# also isolate the controller and serial runs from any existing Brightway project.
_test_storage = Path(
    tempfile.mkdtemp(prefix=f"pytoce_pytest_{os.environ.get('PYTEST_XDIST_WORKER', 'master')}_")
)
_brightway_storage = _test_storage / "brightway"
_temporary_storage = _test_storage / "tmp"
_brightway_storage.mkdir()
_temporary_storage.mkdir()
os.environ["BRIGHTWAY2_DIR"] = str(_brightway_storage)
for _temporary_variable in ("TEMP", "TMP", "TMPDIR"):
    os.environ[_temporary_variable] = str(_temporary_storage)
# Pytest may already have cached the inherited temporary directory before this import.
tempfile.tempdir = str(_temporary_storage)

import py7zr
import pytest
from material_recipe_fixtures import material_source_members
from mfa_workspace_fixtures import StoredMFAFixture, prepare_stored_mfa
from premise_fixtures import (
    inline_premise_process,
    isolated_premise,
    premise_worker_streams,
)
from remind_mfa_source_fixtures import external_remind_mfa_import_state

from pytoce.download_data.sources.catalogue import SourceSpec, load_catalogue
from pytoce.download_data.sources.ecoinvent_materials import derive_material_recipe
from pytoce.download_data.sources.network import DownloadTransport

__all__ = (
    "external_remind_mfa_import_state",
    "inline_premise_process",
    "isolated_premise",
    "premise_worker_streams",
)


@pytest.fixture
def stored_material_recipe(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Persist invented quantities at the REMIND-MFA 0.2.0 input boundary."""
    _, sources = load_catalogue()
    source = next(item for item in sources if item.source_id == "ecoinvent_3_12_cutoff_ecospold02")
    recipe = derive_material_recipe(
        material_source_members(), source_identity=f"{source.source_id}::{source.sha256}"
    )
    payload = json.loads(json.dumps(recipe.as_dict()))
    path = tmp_path / "data_processed" / "ecoinvent" / "material_recipe.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


@pytest.fixture
def saved_mfa(tmp_path: Path, remind_mfa_source_root: Path) -> StoredMFAFixture:
    """Provide a real bounded stored parent for reuse and worker admission tests."""
    return prepare_stored_mfa(tmp_path, remind_mfa_source_root)


@pytest.fixture
def acquisition_catalogue(
    tmp_path: Path,
) -> tuple[tuple[SourceSpec, ...], tuple[SourceSpec, ...], tuple[Path, ...]]:
    """Return bounded immutable open-source and ecoinvent fixtures."""

    source_root = tmp_path / "official_sources"
    source_root.mkdir()
    open_specs: list[SourceSpec] = []
    for index, source_id in enumerate(
        ("remind_mfa_code", "remind_mfa_input_data", "aesa_pb_lcia"),
        start=1,
    ):
        content = f"immutable fixture {source_id} {index}\n".encode()
        source_path = source_root / f"{source_id}.tar"
        source_path.write_bytes(content)
        open_specs.append(
            SourceSpec(
                source_id=source_id,
                kind="test_source",
                version=f"{index}.0.0",
                revision=f"revision-{index}",
                filename=source_path.name,
                size_bytes=len(content),
                sha256=sha256(content).hexdigest(),
                license="BSD-3-Clause",
                citation=f"Test citation {index}",
                metadata={"repository": f"https://example.test/{source_id}"},
                url=source_path.as_uri(),
            )
        )

    archive_source = tmp_path / "licensed_source"
    datasets = archive_source / "datasets"
    master_data = archive_source / "MasterData"
    datasets.mkdir(parents=True)
    master_data.mkdir()
    members = material_source_members()
    for member, content in members.items():
        (archive_source / member).write_bytes(content)
    (master_data / "ActivityIndex.xml").write_text("<index/>", encoding="utf-8")
    lookup = archive_source / "FilenameToActivityLookup.csv"
    lookup.write_text("filename,activity\n", encoding="utf-8")
    archive_path = tmp_path / "ecoinvent_fixture.7z"
    with py7zr.SevenZipFile(archive_path, mode="w") as archive:
        archive.write(datasets, "datasets")
        for member in members:
            archive.write(archive_source / member, member)
        archive.write(master_data, "MasterData")
        archive.write(master_data / "ActivityIndex.xml", "MasterData/ActivityIndex.xml")
        archive.write(lookup, "FilenameToActivityLookup.csv")
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        names = tuple(name.replace("\\", "/") for name in archive.getnames())
    names_digest = sha256(("\n".join(sorted(names)) + "\n").encode("utf-8")).hexdigest()
    ecospold_spec = SourceSpec(
        source_id="ecoinvent_3_12_cutoff_ecospold02",
        kind="licensed_source_data",
        version="3.12",
        revision="allocation_cutoff_by_classification_ecospold02",
        filename="ecoinvent_fixture.7z",
        size_bytes=archive_path.stat().st_size,
        sha256=sha256(archive_path.read_bytes()).hexdigest(),
        license="LicenseRef-ecoinvent-3.12",
        citation="Test ecoinvent authority",
        metadata={
            "member_count": len(names),
            "ecospold_member_count": sum(name.lower().endswith(".spold") for name in names),
            "member_names_sha256": names_digest,
            "system_model": "Allocation, cut-off by classification",
        },
    )
    workbook_bytes = b"bounded ecoinvent 3.12 LCIA workbook fixture\n"
    lcia_source = tmp_path / "lcia_source"
    lcia_source.mkdir()
    (lcia_source / "LCIA Implementation 3.12.xlsx").write_bytes(workbook_bytes)
    (lcia_source / "LCIA Implementation 3.12.pdf").write_bytes(b"bounded report fixture\n")
    lcia_archive = tmp_path / "ecoinvent_lcia_fixture.7z"
    with py7zr.SevenZipFile(lcia_archive, mode="w") as archive:
        archive.write(
            lcia_source / "LCIA Implementation 3.12.xlsx",
            "LCIA Implementation 3.12.xlsx",
        )
        archive.write(
            lcia_source / "LCIA Implementation 3.12.pdf",
            "LCIA Implementation 3.12.pdf",
        )
    lcia_spec = SourceSpec(
        source_id="ecoinvent_3_12_lcia_implementation",
        kind="licensed_lcia_factors",
        version="3.12",
        revision="lcia_implementation",
        filename=lcia_archive.name,
        size_bytes=lcia_archive.stat().st_size,
        sha256=sha256(lcia_archive.read_bytes()).hexdigest(),
        license="LicenseRef-ecoinvent-3.12",
        citation="Test ecoinvent LCIA authority",
        metadata={
            "member_count": 2,
            "workbook_member": "LCIA Implementation 3.12.xlsx",
            "workbook_filename": "lcia_implementation.xlsx",
            "workbook_bytes": len(workbook_bytes),
            "workbook_sha256": sha256(workbook_bytes).hexdigest(),
        },
    )
    return (
        tuple(open_specs),
        (ecospold_spec, lcia_spec),
        (archive_path, lcia_archive),
    )


@pytest.fixture
def acquisition_transport() -> DownloadTransport:
    """Copy one deterministic local source through the production boundary."""

    def copy_source(source: SourceSpec, destination: Path) -> None:
        assert source.url is not None
        parsed = urlparse(source.url)
        assert parsed.scheme == "file"
        source_path = Path(url2pathname(unquote(parsed.path)))
        shutil.copyfile(source_path, destination)

    return copy_source


@pytest.fixture(scope="session")
def remind_mfa_source_root() -> Path:
    """Use REMIND-MFA 0.2.0 source provisioned during test dependency setup.

    Pytest uses the provisioned dependency and disposable test directories.
    The source tree is read-only; synthetic model inputs belong to tmp_path.
    """
    root = Path(
        os.environ.get("REMIND_MFA_TEST_SOURCE", "build/test_dependencies/remind_mfa")
    ).resolve()
    if not root.is_dir():
        pytest.fail(
            "The REMIND-MFA 0.2.0 test source is missing. Complete the dependency "
            "setup in CONTRIBUTING.md before running the MFA tests."
        )
    return root
