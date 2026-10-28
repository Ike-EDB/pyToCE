import io
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from pytoce.run_mfa.remind_mfa.execution import (
    RemindMFAExecutionError,
    RemindMFASource,
    _configuration,
    _dimension_schema_sha256,
    _extract_code,
    _extract_data,
    _safe_member,
    prepare_execution,
)


def test_source_archive_paths_stay_inside_the_extraction_directory() -> None:
    assert _safe_member("source/remind_mfa/model.py").as_posix() == ("source/remind_mfa/model.py")

    for name in ("", "/absolute", "../outside", "source/../../outside", "C:/outside"):
        with pytest.raises(RemindMFAExecutionError, match="unsafe member path"):
            _safe_member(name)


def test_code_and_data_archives_extract_regular_files(tmp_path: Path) -> None:
    code_archive = tmp_path / "code.tar.gz"
    payload = b"print('source')\n"
    with tarfile.open(code_archive, mode="w:gz") as archive:
        member = tarfile.TarInfo("remind-mfa/remind_mfa/model.py")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    data_archive = tmp_path / "data.zip"
    with zipfile.ZipFile(data_archive, mode="w") as archive:
        archive.writestr("plastics/input_data/version.txt", "revision\n")

    code_root = _extract_code(code_archive, tmp_path / "code")
    data_root = tmp_path / "data"
    _extract_data(data_archive, data_root)

    assert (code_root / "remind_mfa" / "model.py").read_bytes() == payload
    assert (data_root / "plastics" / "input_data" / "version.txt").read_text() == ("revision\n")


def test_code_archive_rejects_links_and_multiple_roots(tmp_path: Path) -> None:
    linked = tmp_path / "linked.tar.gz"
    with tarfile.open(linked, mode="w:gz") as archive:
        member = tarfile.TarInfo("source/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "target"
        archive.addfile(member)
    with pytest.raises(RemindMFAExecutionError, match="unsupported links or devices"):
        _extract_code(linked, tmp_path / "linked")

    multiple = tmp_path / "multiple.tar.gz"
    with tarfile.open(multiple, mode="w:gz") as archive:
        for name in ("first/file.txt", "second/file.txt"):
            member = tarfile.TarInfo(name)
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(RemindMFAExecutionError, match="no unique root"):
        _extract_code(multiple, tmp_path / "multiple")

    invalid = tmp_path / "invalid.tar.gz"
    invalid.write_bytes(b"not a tar file")
    with pytest.raises(RemindMFAExecutionError, match="cannot be extracted"):
        _extract_code(invalid, tmp_path / "invalid")


def test_data_archive_rejects_symlinks_and_invalid_zip(tmp_path: Path) -> None:
    linked = tmp_path / "linked.zip"
    with zipfile.ZipFile(linked, mode="w") as archive:
        member = zipfile.ZipInfo("source/link")
        member.create_system = 3
        member.external_attr = 0o120777 << 16
        archive.writestr(member, "target")
    with pytest.raises(RemindMFAExecutionError, match="contains a symlink"):
        _extract_data(linked, tmp_path / "linked")

    invalid = tmp_path / "invalid.zip"
    invalid.write_bytes(b"not a zip file")
    with pytest.raises(RemindMFAExecutionError, match="cannot be extracted"):
        _extract_data(invalid, tmp_path / "invalid")


def test_configuration_uses_the_selected_source_and_disables_exports(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    data_root = tmp_path / "data"
    configuration_path = code_root / "config" / "plastics.yml"
    configuration_path.parent.mkdir(parents=True)
    data_root.mkdir()
    configuration_path.write_text(
        yaml.safe_dump(
            {
                "input": {"input_data_path": "old", "scenarios_path": "old"},
                "model_switches": {"scenario": "old"},
                "export": {"csv": {"do_export": True}},
                "visualization": {
                    "do_visualize": True,
                    "do_show_figs": True,
                    "sankey": {"do_visualize": True},
                },
            }
        ),
        encoding="utf-8",
    )
    source = RemindMFASource(
        code_root=code_root,
        data_root=data_root,
        code_archive_sha256="a" * 64,
        data_archive_sha256="b" * 64,
        source_manifest=tmp_path / "source_manifest.json",
    )

    configuration = _configuration(source)

    assert configuration["input"] == {
        "input_data_path": str(data_root),
        "scenarios_path": str(code_root / "config" / "scenarios"),
    }
    assert configuration["model_switches"]["scenario"] == "SSP2"
    assert configuration["export"]["do_export"] is False
    assert all(
        not configuration["export"][name]["do_export"]
        for name in ("pickle", "csv", "iamc", "assumptions", "docs")
    )
    assert configuration["visualization"]["do_visualize"] is False
    assert configuration["visualization"]["do_show_figs"] is False
    assert configuration["visualization"]["sankey"]["do_visualize"] is False

    configuration_path.write_text("[invalid", encoding="utf-8")
    with pytest.raises(RemindMFAExecutionError, match="configuration is unreadable"):
        _configuration(source)


def test_dimension_identity_covers_dimension_and_region_mapping_files(tmp_path: Path) -> None:
    plastics = tmp_path / "plastics"
    dimensions = plastics / "dimensions"
    input_data = plastics / "input_data"
    dimensions.mkdir(parents=True)
    input_data.mkdir()
    (dimensions / "regions.csv").write_text("region\nEUR\n", encoding="utf-8")
    mapping = input_data / "regionmappingH12.csv"
    mapping.write_text("source,target\nEUR,EUR\n", encoding="utf-8")

    before = _dimension_schema_sha256(tmp_path)
    mapping.write_text("source,target\nEUR,EUR\nUSA,NAM\n", encoding="utf-8")
    after = _dimension_schema_sha256(tmp_path)

    assert before != after


def test_execution_request_records_configuration_and_dimension_identities(
    tmp_path: Path,
) -> None:
    code_root = tmp_path / "code"
    data_root = tmp_path / "data"
    configuration = code_root / "config" / "plastics.yml"
    configuration.parent.mkdir(parents=True)
    configuration.write_text(
        yaml.safe_dump(
            {
                "input": {},
                "model_switches": {},
                "export": {},
                "visualization": {},
            }
        ),
        encoding="utf-8",
    )
    dimensions = data_root / "plastics" / "dimensions"
    dimensions.mkdir(parents=True)
    (dimensions / "regions.csv").write_text("region\nEUR\n", encoding="utf-8")
    mapping = data_root / "plastics" / "input_data" / "regionmappingH12.csv"
    mapping.parent.mkdir()
    mapping.write_text("source,target\nEUR,EUR\n", encoding="utf-8")
    source = RemindMFASource(
        code_root=code_root,
        data_root=data_root,
        code_archive_sha256="a" * 64,
        data_archive_sha256="b" * 64,
        source_manifest=tmp_path / "source_manifest.json",
    )

    request = prepare_execution(source)

    assert request.source is source
    assert request.identity.commit == "bf95c898e0f6f363196bccb62d34e02d82e4726b"
    assert request.identity.scenario == "SSP2"
    assert request.identity.dimension_schema_sha256 == _dimension_schema_sha256(data_root)
    assert yaml.safe_load(request.configuration_yaml)["input"]["input_data_path"] == str(data_root)
