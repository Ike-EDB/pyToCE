"""Premise 2.4.6 integration and background-selection tests."""

from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from pytoce import prepare_backgrounds, set_workspace
from pytoce.prepare_backgrounds.preparation import premise as owner
from pytoce.shared.runtime.filesystem import (
    atomic_write,
    canonical_json_bytes,
    payload_sha256,
)


def test_installed_preparation_dependencies_bind_versions_and_records() -> None:
    authorities = owner.installed_preparation_dependency_authorities()

    assert tuple((row["distribution"], row["version"]) for row in authorities) == (
        ("bw2data", "4.7"),
        ("bw2io", "0.9.17"),
        ("premise", "2.4.6"),
    )
    for row in authorities:
        assert len(row["record_sha256"]) == 64
        assert str(row["identity"]).startswith("pytoce_installed_distribution_schema_v1::")


def test_dependency_errors_identify_the_required_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report missing, changed, and incomplete installed distributions directly."""

    def missing(_name: str) -> Any:
        raise metadata.PackageNotFoundError("bounded")

    monkeypatch.setattr(owner.metadata, "distribution", missing)
    with pytest.raises(RuntimeError, match="not installed.*fresh supported Python"):
        owner._installed_distribution_authority("bounded", expected_version="1.0")

    class Distribution:
        version = "2.0"

        def read_text(self, filename: str) -> str | None:
            return None if filename == "RECORD" else "{}"

    distribution = Distribution()
    monkeypatch.setattr(owner.metadata, "distribution", lambda _name: distribution)
    with pytest.raises(RuntimeError, match="version 1.0 is required"):
        owner._installed_distribution_authority("bounded", expected_version="1.0")

    distribution.version = "1.0"
    with pytest.raises(RuntimeError, match="has no wheel RECORD"):
        owner._installed_distribution_authority("bounded", expected_version="1.0")


def test_background_selection_is_canonical_and_ssp2_only() -> None:
    assert len(owner.select_background_scenarios()) == 4
    assert len(owner.select_background_scenarios("all")) == 4
    assert all(item.premise_model == "remind" for item in owner.select_background_scenarios("all"))
    assert owner.select_background_scenarios("remind_eu_ssp2_npi")[0].premise_model == "remind-eu"
    assert tuple(
        item.background_scenario
        for item in owner.select_background_scenarios(
            ("remind_ssp2_npi", "remind_ssp2_ndc", "remind_ssp2_npi")
        )
    ) == ("remind_ssp2_ndc", "remind_ssp2_npi")
    with pytest.raises(ValueError, match="Unsupported"):
        owner.select_background_scenarios("ssp1")
    with pytest.raises(ValueError, match="cannot be combined"):
        owner.select_background_scenarios(("all", "remind_ssp2_npi"))
    with pytest.raises(ValueError, match="one available name or a sequence"):
        owner.select_background_scenarios(1)  # type: ignore[arg-type]


def test_remind_eu_prepared_backgrounds_are_reuse_only(tmp_path: Path) -> None:
    workspace = set_workspace(tmp_path).workspace
    result = workspace / "data_processed" / "prepared_backgrounds" / "remind_eu_ssp2_npi"
    result.mkdir(parents=True)
    manifest = result / "manifest.json"
    manifest.write_text("ordinary reuse does not reopen this file\n", encoding="utf-8")

    reused = prepare_backgrounds(background_scenario="remind_eu_ssp2_npi")
    assert reused.status == "reused"
    assert reused.results[0].manifest_path == manifest

    manifest.unlink()
    with pytest.raises(FileNotFoundError, match="Choose a completed historical scenario"):
        prepare_backgrounds(background_scenario="remind_eu_ssp2_npi")


def test_premise_uses_single_exact_transformation_path() -> None:
    assert owner.premise_configuration() == {
        "source_version": "3.12",
        "source_type": "brightway",
        "system_model": "cutoff",
        "use_cached_inventories": True,
        "use_cached_database": True,
        "external_scenarios": None,
        "quiet": False,
        "keep_imports_uncertainty": False,
        "keep_source_db_uncertainty": False,
        "gains_scenario": "CLE",
        "use_absolute_efficiency": False,
        "generate_reports": True,
        "update_sectors": list(owner.PREMISE_UPDATE_SECTORS),
    }
    assert owner.PREMISE_UPDATE_SECTORS == (
        "biomass",
        "electricity",
        "cement",
        "steel",
        "fuels",
        "renewable",
        "metals",
        "mining",
        "heat",
        "cdr",
        "battery",
        "cars",
        "two_wheelers",
        "trucks",
        "ships",
        "buses",
        "trains",
        "final energy",
        "emissions",
    )


def test_premise_key_validation_does_not_echo_secret(tmp_path: Path) -> None:
    set_workspace(tmp_path)
    with pytest.raises(ValueError, match="44-byte") as caught:
        prepare_backgrounds(premise_key="secret")
    assert "secret" not in str(caught.value)


def test_workspace_iam_custody_is_exact_and_identity_bound(tmp_path: Path) -> None:
    content = b"gAAAA-synthetic-encrypted-iam"
    scenario = owner.BackgroundScenario(
        background_scenario="synthetic_ssp2",
        premise_model="remind",
        iam_model="REMIND",
        iam_scenario="SSP2-NPi",
        gmst="test",
        iam_filename="remind_SSP2-NPi.csv",
        iam_file_size=len(content),
        iam_file_sha256=sha256(content).hexdigest(),
    )
    premise_authority = next(
        row
        for row in owner.installed_preparation_dependency_authorities()
        if row["distribution"] == "premise"
    )
    identity_payload = owner._iam_identity_payload(
        scenario,
        premise_authority=premise_authority,
    )
    identity = f"pytoce_premise_iam_input_schema_v1::{payload_sha256(identity_payload)}"
    root = tmp_path / "data_raw" / "premise" / scenario.background_scenario
    root.mkdir(parents=True)
    (root / scenario.iam_filename).write_bytes(content)
    atomic_write(
        root / "manifest.json",
        canonical_json_bytes(
            {
                "schema_version": "pytoce_premise_iam_input_schema_v1",
                "identity": identity,
                "identity_payload": identity_payload,
                "artifact": {
                    "path": scenario.iam_filename,
                    "bytes": len(content),
                    "sha256": sha256(content).hexdigest(),
                },
                "complete": True,
            }
        ),
    )
    accepted = owner.acquire_iam_input(
        tmp_path,
        scenario,
        premise_authority=premise_authority,
    )
    assert accepted.identity == identity
    assert accepted.payload == root / scenario.iam_filename
    assert not (root / "logs").exists()

    (root / scenario.iam_filename).write_bytes(content + b"-changed")
    assert (
        owner.acquire_iam_input(
            tmp_path,
            scenario,
            premise_authority=premise_authority,
        ).identity
        == identity
    )


@pytest.mark.usefixtures("isolated_premise")
def test_iam_input_errors_preserve_existing_data_and_reject_changed_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an incomplete saved directory and a changed network response."""

    scenario = owner.BackgroundScenario(
        background_scenario="synthetic_ssp2",
        premise_model="remind",
        iam_model="REMIND",
        iam_scenario="SSP2-NPi",
        gmst="test",
        iam_filename="remind_SSP2-NPi.csv",
        iam_file_size=8,
        iam_file_sha256=sha256(b"expected").hexdigest(),
    )
    premise_authority = {"identity": "premise::bounded"}
    existing_workspace = tmp_path / "existing"
    incomplete = existing_workspace / "data_raw" / "premise" / scenario.background_scenario
    incomplete.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="has no completion manifest.*new workspace"):
        owner.acquire_iam_input(
            existing_workspace,
            scenario,
            premise_authority=premise_authority,
        )

    import premise.scenario_downloader as downloader

    def changed_download(filename: str, _url: str, directory: Path) -> Path:
        destination = Path(directory) / filename
        destination.write_bytes(b"changed!")
        return destination

    monkeypatch.setattr(downloader, "download_csv", changed_download)
    with pytest.raises(RuntimeError, match="does not match the official file"):
        owner.acquire_iam_input(
            tmp_path / "download",
            scenario,
            premise_authority=premise_authority,
        )
