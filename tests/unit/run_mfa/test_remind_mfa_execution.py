"""REMIND-MFA source custody and execution-boundary contracts."""

import io
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import flodym as fd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest

from pytoce.run_mfa.remind_mfa.execution import (
    RemindMFAExecutionError,
    RemindMFASource,
    _activate_source,
    _import_source_module,
    _source_artifact,
    prepare_workspace_source,
)
from pytoce.run_mfa.remind_mfa.process_inputs import (
    _resolved_waste_trade,
    default_waste_trade_propensities,
)


def _source(code_root: Path, tmp_path: Path) -> RemindMFASource:
    """Build the source descriptor needed by import-custody checks."""

    return RemindMFASource(
        code_root=code_root,
        data_root=tmp_path / "data",
        code_archive_sha256="a" * 64,
        data_archive_sha256="b" * 64,
        source_manifest=tmp_path / "manifest.json",
    )


def _remove_remind_modules() -> dict[str, Any]:
    """Remove and return REMIND-MFA modules so process state can be restored."""

    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "remind_mfa" or name.startswith("remind_mfa.")
    }
    for name in tuple(sys.modules):
        if name == "remind_mfa" or name.startswith("remind_mfa."):
            del sys.modules[name]
    return saved


def _restore_remind_modules(saved: dict[str, Any]) -> None:
    """Restore the exact REMIND-MFA module state present before one test."""

    _remove_remind_modules()
    sys.modules.update(saved)


def test_incomplete_visible_source_is_preserved_and_rejected(tmp_path: Path) -> None:
    """Report an ambiguous source directory when its completion manifest is absent."""

    target = tmp_path / "data_processed" / "remind_mfa"
    target.mkdir(parents=True)
    evidence = target / "partial.txt"
    evidence.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(RemindMFAExecutionError, match="has no completion manifest"):
        prepare_workspace_source(tmp_path)

    assert evidence.read_text(encoding="utf-8") == "preserve me\n"


def test_source_artifact_requires_both_download_manifest_and_named_archive(tmp_path: Path) -> None:
    """Fail before extraction unless the exact downloaded archive is present."""

    source_root = tmp_path / "data_raw" / "remind_mfa"
    source_root.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="download_data"):
        _source_artifact(tmp_path, "remind_mfa_code", "code.tar.gz")

    (source_root / "manifest.json").write_text("complete\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="remind_mfa_code"):
        _source_artifact(tmp_path, "remind_mfa_code", "code.tar.gz")

    archive = source_root / "code.tar.gz"
    archive.write_bytes(b"archive")
    assert _source_artifact(tmp_path, "remind_mfa_code", archive.name) == archive


def test_failed_source_extraction_removes_staging_but_not_raw_archives(tmp_path: Path) -> None:
    """Contain incomplete extracted sources while preserving their downloaded parents."""

    raw = tmp_path / "data_raw" / "remind_mfa"
    raw.mkdir(parents=True)
    (raw / "manifest.json").write_text("complete\n", encoding="utf-8")
    code_archive = raw / "remind_mfa_v0_2_0.tar.gz"
    code = b"class PlasticsModel:\n    pass\n"
    with tarfile.open(code_archive, mode="w:gz") as archive:
        member = tarfile.TarInfo("remind-mfa/remind_mfa/plastics/plastics_model.py")
        member.size = len(code)
        archive.addfile(member, io.BytesIO(code))
    data_archive = raw / "input_data_1_0_0.zip"
    with zipfile.ZipFile(data_archive, mode="w") as archive:
        archive.writestr("../outside.txt", "unsafe\n")

    with pytest.raises(RemindMFAExecutionError, match="unsafe member path"):
        prepare_workspace_source(tmp_path)

    processed = tmp_path / "data_processed"
    assert not (processed / "remind_mfa").exists()
    assert not list(processed.glob(".remind_mfa_source.*"))
    assert code_archive.is_file()
    assert data_archive.is_file()


def test_completed_source_is_reused_without_downloaded_archives(tmp_path: Path) -> None:
    """Reuse an actual completed extraction after its downloaded archives are removed."""

    fixture = tmp_path / "fixture"
    code_root = fixture / "remind-mfa"
    model = code_root / "remind_mfa" / "plastics" / "plastics_model.py"
    model.parent.mkdir(parents=True)
    model.write_text("class PlasticsModel:\n    pass\n", encoding="utf-8")
    configuration = code_root / "config" / "plastics.yml"
    configuration.parent.mkdir()
    configuration.write_text("model_switches: {}\n", encoding="utf-8")
    data_root = fixture / "data"
    version = data_root / "plastics" / "input_data" / "version.txt"
    version.parent.mkdir(parents=True)
    version.write_text("fixture_revision\n", encoding="utf-8")
    (data_root / "plastics" / "dimensions").mkdir()

    raw = tmp_path / "workspace" / "data_raw" / "remind_mfa"
    raw.mkdir(parents=True)
    (raw / "manifest.json").write_text('{"complete": true}\n', encoding="utf-8")
    code_archive = raw / "remind_mfa_v0_2_0.tar.gz"
    with tarfile.open(code_archive, mode="w:gz") as archive:
        archive.add(code_root, arcname="remind-mfa")
    data_archive = raw / "input_data_1_0_0.zip"
    with zipfile.ZipFile(data_archive, mode="w") as archive:
        for path in sorted(data_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(data_root).as_posix())
    source = prepare_workspace_source(tmp_path / "workspace")
    for archive in (code_archive, data_archive):
        archive.unlink()
    reused = prepare_workspace_source(tmp_path / "workspace")

    assert source == reused
    assert source.source_manifest.is_file()
    assert (source.code_root / "remind_mfa/plastics/plastics_model.py").read_bytes() == (
        model.read_bytes()
    )
    assert (source.data_root / "plastics/input_data/version.txt").read_bytes() == (
        version.read_bytes()
    )


def test_source_import_uses_one_real_filesystem_authority_per_process(tmp_path: Path) -> None:
    """Import from the selected tree and reject a later, different source authority."""

    first_root = tmp_path / "first"
    package = first_root / "remind_mfa"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "probe.py").write_text("AUTHORITY = 'first'\n", encoding="utf-8")
    second_root = tmp_path / "second"
    second_package = second_root / "remind_mfa"
    second_package.mkdir(parents=True)
    (second_package / "__init__.py").write_text("\n", encoding="utf-8")

    original_path = list(sys.path)
    saved_modules = _remove_remind_modules()
    try:
        module = _import_source_module(_source(first_root, tmp_path), "remind_mfa.probe")
        assert isinstance(module, ModuleType)
        assert module.AUTHORITY == "first"
        assert first_root.resolve() in Path(module.__file__).resolve().parents

        _activate_source(_source(first_root, tmp_path))
        with pytest.raises(RemindMFAExecutionError, match="different REMIND-MFA source"):
            _activate_source(_source(second_root, tmp_path))
    finally:
        _restore_remind_modules(saved_modules)
        sys.path[:] = original_path


def test_source_import_rejects_namespace_module_without_source_file(tmp_path: Path) -> None:
    """Reject a namespace-only module whose source file is absent."""

    code_root = tmp_path / "code"
    package = code_root / "remind_mfa"
    namespace = package / "namespace_probe"
    namespace.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")

    original_path = list(sys.path)
    saved_modules = _remove_remind_modules()
    try:
        with pytest.raises(RemindMFAExecutionError, match="has no source path"):
            _import_source_module(_source(code_root, tmp_path), "remind_mfa.namespace_probe")
        _activate_source(_source(code_root, tmp_path))
        assert Path(sys.path[0]).resolve() == code_root.resolve()
    finally:
        _restore_remind_modules(saved_modules)
        sys.path[:] = original_path


def test_source_import_rejects_module_resolved_from_another_source_root(
    tmp_path: Path,
) -> None:
    """Reject a real module file found outside the selected REMIND-MFA tree."""

    selected_root = tmp_path / "selected"
    (selected_root / "remind_mfa").mkdir(parents=True)
    other_root = tmp_path / "other"
    other_package = other_root / "remind_mfa"
    other_package.mkdir(parents=True)
    (other_package / "__init__.py").write_text("\n", encoding="utf-8")
    (other_package / "external_probe.py").write_text("AUTHORITY = 'other'\n", encoding="utf-8")

    original_path = list(sys.path)
    saved_modules = _remove_remind_modules()
    sys.path.insert(0, str(other_root))
    try:
        with pytest.raises(RemindMFAExecutionError, match="module path differs"):
            _import_source_module(
                _source(selected_root, tmp_path),
                "remind_mfa.external_probe",
            )
    finally:
        _restore_remind_modules(saved_modules)
        sys.path[:] = original_path


def _trade_inputs(
    *,
    local_outflow: tuple[float, float] = (10.0, 20.0),
    collection_rate: tuple[float, float] = (1.0, 1.0),
    waste_imports: tuple[float, float] = (0.0, 0.0),
    waste_exports: tuple[float, float] = (0.0, 0.0),
) -> tuple[fd.DimensionSet, fd.FlodymArray, dict[str, fd.Parameter]]:
    """Return source-shaped external arrays for waste-trade admission."""

    dimensions = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name="Time", letter="t", items=[2020]),
            fd.Dimension(name="Element", letter="e", items=["C"]),
            fd.Dimension(name="Region", letter="r", items=["A", "B"]),
            fd.Dimension(name="Material", letter="m", items=["polymer"]),
            fd.Dimension(name="Good", letter="g", items=["product"]),
        ]
    )
    outflow = fd.FlodymArray(
        dims=dimensions[("t", "e", "r", "m", "g")],
        values=np.asarray(local_outflow, dtype=float).reshape(1, 1, 2, 1, 1),
        name="in_use_outflow",
    )

    def parameter(name: str, values: tuple[float, float]) -> fd.Parameter:
        """Create one real time-region FLODYM parameter."""

        return fd.Parameter(
            dims=dimensions[("t", "r")],
            values=np.asarray(values, dtype=float).reshape(1, 2),
            name=name,
        )

    parameters = {
        "collection_rate": parameter("collection_rate", collection_rate),
        "waste_imports": parameter("waste_imports", waste_imports),
        "waste_exports": parameter("waste_exports", waste_exports),
    }
    return dimensions, outflow, parameters


def test_default_trade_propensity_rejects_mass_over_a_zero_collection_denominator() -> None:
    """Require a local collection basis for the default trade propensity."""

    dimensions, outflow, parameters = _trade_inputs(
        local_outflow=(0.0, 20.0),
        waste_imports=(1.0, 0.0),
    )

    with pytest.raises(ValueError, match="nonzero amount and zero denominator"):
        default_waste_trade_propensities(outflow, parameters)


@pytest.mark.parametrize(
    ("imports", "exports"),
    (
        ((-1.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (11.0, 0.0)),
    ),
)
def test_default_trade_propensity_rejects_negative_or_overcomplete_shares(
    imports: tuple[float, float],
    exports: tuple[float, float],
) -> None:
    """Reject default propensities outside import/export bounds."""

    dimensions, outflow, parameters = _trade_inputs(waste_imports=imports, waste_exports=exports)

    with pytest.raises(ValueError, match="propensities are invalid"):
        default_waste_trade_propensities(outflow, parameters)


def test_deferred_export_cannot_exceed_local_pretrade_collection(
    remind_trade_modules: tuple[ModuleType, ModuleType],
) -> None:
    """Reject an authored export propensity that removes more waste than exists locally."""

    dimensions, outflow, parameters = _trade_inputs()
    deferred = pd.DataFrame.from_records(
        [
            {
                "parameter_id": "waste_exports",
                "year": 2020,
                "region": "A",
                "deferred_propensity": 1.1,
                "value_mode": "retain_2019_propensity_pretrade_collected",
            }
        ]
    )
    default = pd.DataFrame.from_records(
        [
            {
                "parameter_id": "waste_exports",
                "year": 2020,
                "region": "A",
                "default_propensity": 0.0,
                "default_propensity_authority_identity": "default::fixture",
            }
        ]
    )

    with pytest.raises(ValueError, match="waste_exports at year=2020, region=A"):
        _resolved_waste_trade(
            outflow, parameters, dimensions, remind_trade_modules[0], deferred, default
        )
