"""External allocated carrying capacity inputs and parsed-source provenance."""

import gzip
import json
import tarfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import yaml
from allocated_cc_fixtures import _source_frames, _write_raw_result, _write_shared_authority

from pytoce.run_optimization.input.allocated_cc import select_allocated_cc_result
from pytoce.run_optimization.input.allocated_cc_contract import (
    AllocatedCCError,
)
from pytoce.run_optimization.input.allocated_cc_paths import (
    _allocated_cc_metadata,
    _secure_source_directory,
    _secure_source_file,
)
from pytoce.run_optimization.input.allocated_cc_product import (
    compile_allocated_cc_result,
)


def _metadata_text(name: str = "study_result") -> str:
    """Return one complete v1 external aCC metadata document."""

    return "\n".join(
        (
            "schema_version: pytoce_allocated_cc_result_schema_v1",
            f"allocated_cc_result: {name}",
            "description: External carrying capacity fixture",
            "ssp_scenario: SSP2",
            "lcia_method:",
            "  - ef31",
            "annual_allocated_cc: annual.csv",
            "cumulative_gwp100_allocated_cc: cumulative.csv",
            "",
        )
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- not-a-mapping\n", "must contain one mapping"),
        (
            _metadata_text().replace(
                "schema_version: pytoce_allocated_cc_result_schema_v1\n",
                "",
            ),
            "missing required field",
        ),
        (_metadata_text() + "unknown_field: value\n", "Unknown allocated_cc_result.yaml field"),
        (
            _metadata_text().replace("pytoce_allocated_cc_result_schema_v1", "obsolete"),
            "schema_version=.*not supported",
        ),
        (
            _metadata_text().replace("ssp_scenario: SSP2", "ssp_scenario: SSP1"),
            "Use exactly 'SSP2'",
        ),
        (_metadata_text().replace("  - ef31", "  - pb_lcia"), r"Use exactly \['ef31'\]"),
        (
            _metadata_text().replace(
                "description: External carrying capacity fixture",
                "description: ''",
            ),
            "description must not be empty",
        ),
        (
            _metadata_text().replace(
                "description: External carrying capacity fixture",
                "description:\n  nested: value",
            ),
            "description must be text",
        ),
        ("1: value\n", "Cannot read allocated carrying capacity metadata"),
    ],
)
def test_allocated_cc_metadata_rejects_ambiguous_or_changed_documents(
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    """Reject malformed external metadata before any scientific table is compiled."""

    path = tmp_path / "allocated_cc_result.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(AllocatedCCError, match=message):
        _allocated_cc_metadata(path, "study_result")


def test_allocated_cc_source_paths_report_missing_workspace_root_and_result(tmp_path: Path) -> None:
    """Distinguish missing workspace, missing owner root, and unknown result names."""

    with pytest.raises(AllocatedCCError, match="selected workspace is unavailable"):
        _secure_source_directory(tmp_path / "missing-workspace", "study_result")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(AllocatedCCError, match="results directory is missing"):
        _secure_source_directory(workspace, "study_result")

    results = workspace / "data_raw" / "allocated_cc" / "results"
    (results / "available_result").mkdir(parents=True)
    with pytest.raises(
        AllocatedCCError, match="Unknown allocated carrying capacity.*available_result"
    ):
        _secure_source_directory(workspace, "study_result")


def test_allocated_cc_source_file_requires_existing_declared_content(tmp_path: Path) -> None:
    """Require each source file declared by external metadata to exist."""

    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(AllocatedCCError, match="Cannot find annual_allocated_cc"):
        _secure_source_file(source, "missing.csv", field="annual_allocated_cc")


@pytest.mark.parametrize(
    "filename", ["annual_allocated_cc.csv", "cumulative_gwp100_allocated_cc.csv"]
)
def test_allocated_cc_compilation_rejects_header_only_tables(tmp_path: Path, filename: str) -> None:
    """Reject either empty scientific table before publishing an aCC result."""

    _write_shared_authority(tmp_path)
    source = _write_raw_result(tmp_path, "study_result")
    path = source / filename
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    with pytest.raises(AllocatedCCError, match="at least one complete case"):
        compile_allocated_cc_result(tmp_path, "study_result")

    assert not (tmp_path / "data_processed/allocated_cc/study_result").exists()


@pytest.mark.parametrize(
    "filename",
    ["allocated_cc_result.yaml", "annual_allocated_cc.csv", "cumulative_gwp100_allocated_cc.csv"],
)
def test_allocated_cc_manifest_identifies_parsed_bytes_when_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    """Bind stored values to their input bytes even if a user replaces a parsed source."""

    _write_shared_authority(tmp_path)
    source = _write_raw_result(tmp_path, "study_result")
    source_bytes = {path.name: path.read_bytes() for path in source.iterdir()}
    path = source / filename
    if filename.endswith(".yaml"):
        replacement = source_bytes[filename].replace(
            b"Direct external aCC fixture", b"Replacement aCC description"
        )
        load_yaml = yaml.load

        def load_then_replace(stream: str, Loader: type[yaml.SafeLoader]) -> object:
            """Complete real YAML parsing before simulating an external file replacement."""
            payload = load_yaml(stream, Loader=Loader)
            path.write_bytes(replacement)
            return payload

        monkeypatch.setattr(yaml, "load", load_then_replace)
    else:
        replacement = source_bytes[filename].replace(b"external_authority", b"revised_authority")
        read_csv = pd.read_csv
        value_column = (
            "allocated_cc" if filename == "annual_allocated_cc.csv" else "cumulative_allocated_cc"
        )

        def read_then_replace(source: object, **kwargs: object) -> pd.DataFrame:
            """Complete real CSV parsing before replacing the relevant external table."""
            frame = read_csv(source, **kwargs)
            if value_column in frame:
                path.write_bytes(replacement)
            return frame

        monkeypatch.setattr(pd, "read_csv", read_then_replace)

    result = compile_allocated_cc_result(tmp_path, "study_result")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert path.read_bytes() == replacement != source_bytes[filename]
    assert manifest["source_files"] == {
        name: sha256(content).hexdigest() for name, content in source_bytes.items()
    }
    assert manifest["identity_payload"]["source_files"] == manifest["source_files"]
    assert manifest["description"] == "Direct external aCC fixture"
    for table in ("annual_allocated_cc.parquet", "cumulative_gwp100_allocated_cc.parquet"):
        assert (
            pd.read_parquet(result.result_directory / table)["cc_authority"]
            .eq("external_authority")
            .all()
        )
    assert (
        select_allocated_cc_result(result, result.allocated_cc_cases).allocated_cc_result_identity
        == result.allocated_cc_result_identity
    )


@pytest.mark.parametrize("suffix", [".csv.gz", ".csv.tar.gz", ".CSV.GZ", ".csv.Tar.Gz"])
def test_allocated_cc_compilation_preserves_compressed_csv_inputs(
    tmp_path: Path, suffix: str
) -> None:
    """Parse a compressed CSV while identifying its original compressed source bytes."""

    _write_shared_authority(tmp_path)
    source = _write_raw_result(tmp_path, "study_result")
    annual_path = source / "annual_allocated_cc.csv"
    content = annual_path.read_bytes()
    filename = "annual" + suffix
    if suffix.lower() == ".csv.gz":
        compressed = gzip.compress(content, mtime=0)
    else:
        buffer = BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            member = tarfile.TarInfo("annual.csv")
            member.size = len(content)
            archive.addfile(member, BytesIO(content))
        compressed = buffer.getvalue()
    (source / filename).write_bytes(compressed)
    annual_path.unlink()
    metadata_path = source / "allocated_cc_result.yaml"
    metadata_path.write_bytes(
        metadata_path.read_bytes().replace(b"annual_allocated_cc.csv", filename.encode("utf-8"))
    )

    result = compile_allocated_cc_result(tmp_path, "study_result")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_files"][filename] == sha256(compressed).hexdigest()
    annual, _cumulative = _source_frames()
    stored = pd.read_parquet(result.result_directory / "annual_allocated_cc.parquet")
    pd.testing.assert_frame_equal(stored.loc[:, annual.columns], annual)
