"""EcoSpold archive custody and real import of invented process inventories."""

import math
import shutil
from hashlib import sha256
from pathlib import Path

import py7zr
import pytest

from pytoce.download_data.sources.catalogue import SourceSpec, manifest_bytes
from pytoce.prepare_backgrounds.preparation import ecoinvent as owner


def test_licensed_inputs_use_the_completed_download_owner(tmp_path: Path) -> None:
    archive_content = b"synthetic EcoSpold fixture"
    workbook_content = b"synthetic LCIA fixture"
    specs = (
        SourceSpec(
            source_id="ecoinvent_3_12_cutoff_ecospold02",
            kind="licensed_archive",
            version="3.12",
            revision="fixture",
            filename="ecoinvent.7z",
            size_bytes=len(archive_content),
            sha256=sha256(archive_content).hexdigest(),
            license="licensed fixture",
            citation="fixture",
            metadata={},
        ),
        SourceSpec(
            source_id="ecoinvent_3_12_lcia_implementation",
            kind="licensed_archive",
            version="3.12",
            revision="fixture",
            filename="ecoinvent_lcia.7z",
            size_bytes=0,
            sha256=sha256(b"").hexdigest(),
            license="licensed fixture",
            citation="fixture",
            metadata={
                "workbook_filename": "implementation.xlsx",
                "workbook_sha256": sha256(workbook_content).hexdigest(),
                "workbook_bytes": len(workbook_content),
            },
        ),
    )
    with pytest.raises(FileNotFoundError, match="run download_data"):
        owner.resolve_licensed_inputs(tmp_path, specs)

    by_id = {spec.source_id: spec for spec in specs}
    ecospold = by_id["ecoinvent_3_12_cutoff_ecospold02"]
    lcia = by_id["ecoinvent_3_12_lcia_implementation"]
    root = owner.source_directory(tmp_path, ecospold)
    root.mkdir(parents=True)
    (root / ecospold.filename).write_bytes(archive_content)
    (root / str(lcia.metadata["workbook_filename"])).write_bytes(workbook_content)
    (root / "manifest.json").write_bytes(manifest_bytes(specs))

    inputs = owner.resolve_licensed_inputs(tmp_path, specs)

    assert inputs.archive == root / ecospold.filename
    assert inputs.archive_sha256 == ecospold.sha256
    assert inputs.archive_manifest == root / "manifest.json"
    assert inputs.ef_workbook == root / str(lcia.metadata["workbook_filename"])
    assert inputs.ef_workbook_sha256 == lcia.metadata["workbook_sha256"]


def test_identity_bound_brightway_names_are_snake_case(tmp_path: Path) -> None:
    names = owner.brightway_names(
        tmp_path,
        source_identity="pytoce_background_sources_schema_v1::" + "a" * 64,
    )
    assert names.project == "pytoce_aaaaaaaaaaaa"
    assert names.biosphere_database == "pytoce__biosphere_3_12__aaaaaaaaaaaa"
    assert names.source_database == ("pytoce__ecoinvent_3_12_cutoff__aaaaaaaaaaaa")
    assert names.store.is_relative_to(tmp_path)


def test_admitted_ecoinvent_archive_extracts_source_members(tmp_path: Path) -> None:
    """Extract authenticated source members with their original contents."""
    source = tmp_path / "source"
    (source / "datasets").mkdir(parents=True)
    (source / "datasets" / "activity.spold").write_text("fixture", encoding="utf-8")
    (source / "MasterData").mkdir()
    (source / "MasterData" / "ElementaryExchanges.xml").write_text(
        "<elementaryExchanges/>",
        encoding="utf-8",
    )
    archive = tmp_path / "ecoinvent.7z"
    with py7zr.SevenZipFile(archive, mode="w") as handle:
        handle.write(source / "datasets" / "activity.spold", "datasets/activity.spold")
        handle.write(
            source / "MasterData" / "ElementaryExchanges.xml",
            "MasterData/ElementaryExchanges.xml",
        )

    destination = tmp_path / "extracted"
    owner.extract_ecoinvent(archive, destination)
    assert {
        str(path.relative_to(destination)).replace("\\", "/"): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == {
        "datasets/activity.spold": b"fixture",
        "MasterData/ElementaryExchanges.xml": b"<elementaryExchanges/>",
    }


def test_ecoinvent_import_preserves_linked_inventories(tmp_path: Path) -> None:
    """Import invented XML through real strategies, links and database persistence."""
    import bw2calc as bc
    import bw2data as bd

    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "ecoinvent_import"
    extracted = tmp_path / "extracted"
    shutil.copytree(fixture, extracted)
    inputs = {p.relative_to(extracted): p.read_bytes() for p in extracted.rglob("*") if p.is_file()}
    names = owner.brightway_names(
        tmp_path,
        source_identity="pytoce_background_sources_schema_v1::" + "a" * 64,
    )
    previous = (bd.projects.current, bd.projects.dir, bd.projects.logs_dir)
    with owner.selected_brightway_project(names):
        report = owner.import_ecoinvent(extracted, names)

        activities = {node["name"]: node for node in bd.Database(names.source_database)}
        assert set(activities) == {
            "fixture product production",
            "fixture waste treatment",
        }
        product = activities["fixture product production"]
        treatment = activities["fixture waste treatment"]
        assert (product["reference product"], treatment["reference product"]) == (
            "fixture product",
            "fixture waste",
        )
        assert product["product_information"] == "Invented product for importer validation."
        assert treatment["product_information"] == "Invented waste for signed treatment validation."
        assert {(node["location"], node["unit"]) for node in (product, treatment)} == {
            ("GLO", "kilogram")
        }
        flow = bd.get_node(
            database=names.biosphere_database,
            code="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        )
        assert (flow["name"], flow["unit"], flow["categories"]) == (
            "fixture emission",
            "kilogram",
            ("air",),
        )
        expected = {
            (product.key, product.key, "production"): (2.0, 0, 2.0, None),
            (product.key, treatment.key, "technosphere"): (-0.5, 0, -0.5, None),
            (product.key, flow.key, "biosphere"): (3.0, 2, math.log(3.0), 0.2),
            (treatment.key, treatment.key, "production"): (-4.0, 0, -4.0, None),
            (treatment.key, flow.key, "biosphere"): (-0.25, 0, -0.25, None),
        }
        actual = {}
        for node in (product, treatment):
            for exchange in node.exchanges():
                key = (node.key, exchange.input.key, exchange["type"])
                assert key not in actual
                actual[key] = (
                    exchange["amount"],
                    exchange["uncertainty type"],
                    exchange["loc"],
                    exchange.get("scale"),
                )
                assert exchange["unit"] == "kilogram"
        assert actual == expected

        assert report == {
            "use_mp": True,
            "method": "pytoce_ecoinvent_native_import_v1",
        }
        # Per 2 kg product: 3 kg direct emission plus (-0.5/-4) treatment
        # activities, each emitting -0.25 kg. Both reference bases apply once.
        calculation = bc.LCA({product: 1.0})
        calculation.lci()
        assert float(calculation.inventory.sum()) == pytest.approx(1.484375)

    assert (bd.projects.current, bd.projects.dir, bd.projects.logs_dir) == previous
    assert {
        p.relative_to(extracted): p.read_bytes() for p in extracted.rglob("*") if p.is_file()
    } == (inputs)
