"""Readable scenario discovery and manifest-last compiler boundaries."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from pytoce import set_workspace
from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue
from pytoce.run_mfa.scenario.compiler import (
    compile_foreground_scenario,
)
from pytoce.run_mfa.scenario.scenario_contract import ScenarioError


def test_catalogue_lookup_reports_the_supported_parameter_names() -> None:
    """Give an actionable error when a scenario names no catalog parameter."""

    with pytest.raises(ScenarioError, match="Unknown foreground parameter") as error:
        load_parameter_catalogue("not_a_plastics_parameter")

    assert "stock_projection" in str(error.value)
    assert "collection_rate" in str(error.value)


def test_compiler_reports_missing_readable_scenario_in_an_initialized_workspace(
    tmp_path: Path,
) -> None:
    """Distinguish an unavailable readable name from an uninitialized workspace."""

    workspace = set_workspace(tmp_path).workspace

    with pytest.raises(ScenarioError, match="is not available") as error:
        compile_foreground_scenario(workspace, "unavailable_scenario")

    assert "foreground_scenarios" in str(error.value)


@pytest.mark.parametrize("visible_kind", ("file", "directory"))
def test_compiler_preserves_ambiguous_visible_products(
    tmp_path: Path,
    visible_kind: str,
) -> None:
    """Preserve a compiled path lacking its completion manifest."""

    workspace = set_workspace(tmp_path).workspace
    result = workspace / "data_processed" / "foreground_scenarios" / "ambiguous"
    result.parent.mkdir(parents=True, exist_ok=True)
    if visible_kind == "file":
        result.write_text("not a directory\n", encoding="utf-8")
        expected = "not a package result directory"
    else:
        result.mkdir()
        (result / "evidence.txt").write_text("preserve\n", encoding="utf-8")
        expected = "has no valid completion manifest"

    with pytest.raises(ScenarioError, match=expected):
        compile_foreground_scenario(workspace, "ambiguous")

    assert result.exists()


def test_refresh_requires_and_replaces_the_exact_completed_compiler_product(
    tmp_path: Path,
) -> None:
    """Refresh only a complete owned product and preserve a mutated product."""

    workspace = set_workspace(tmp_path).workspace
    first = compile_foreground_scenario(workspace, "remind_mfa_default")
    refreshed = compile_foreground_scenario(workspace, "remind_mfa_default", refresh=True)

    assert first.status == "computed"
    assert refreshed.status == "computed"
    assert refreshed.foreground_scenario_identity == first.foreground_scenario_identity

    extra = refreshed.result_directory / "unowned.txt"
    extra.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match="Cannot refresh"):
        compile_foreground_scenario(workspace, "remind_mfa_default", refresh=True)
    assert extra.read_text(encoding="utf-8") == "preserve\n"


def test_compiler_preserves_manifest_and_descendant_identity_on_reuse(
    tmp_path: Path,
) -> None:
    """Retain the completed compiler product and its exact descendant provenance."""
    workspace = set_workspace(tmp_path).workspace
    first = compile_foreground_scenario(workspace, "remind_mfa_default")
    reused = compile_foreground_scenario(workspace, "remind_mfa_default")
    assert reused.status == "reused"
    assert reused.foreground_scenario_identity == first.foreground_scenario_identity
    assert reused.manifest == first.manifest
    assert reused.manifest_sha256 == first.manifest_sha256
    assert reused.manifest_sha256 == sha256(first.manifest_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("refresh", (False, True))
def test_compiler_guide_failure_preserves_unpublished_or_previous_product(
    tmp_path: Path, refresh: bool
) -> None:
    """A failed parent guide cannot publish a new result or replace its predecessor."""
    workspace = set_workspace(tmp_path).workspace
    name = "guide_fixture"
    result = workspace / "data_processed/foreground_scenarios" / name
    scenario_path = workspace / "data_raw/foreground_scenarios/scenarios" / (name + ".yaml")
    scenario = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": name,
        "ssp_scenario": "SSP2",
        "changes": [],
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    before = {}
    if refresh:
        first = compile_foreground_scenario(workspace, name)
        before = {path.name: path.read_bytes() for path in first.result_directory.iterdir()}
        scenario["changes"] = [
            {
                "parameter": "collection_rate",
                "region": "EUR",
                "changes": [{"year": 2050, "mode": "absolute", "value": 0.7}],
            }
        ]
        scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    result.parent.mkdir(parents=True, exist_ok=True)
    guide = result.parent / "README.md"
    if guide.is_file():
        guide.unlink()
    guide.mkdir()
    with pytest.raises(OSError):
        compile_foreground_scenario(workspace, name, refresh=refresh)
    assert guide.is_dir()
    if refresh:
        assert {path.name: path.read_bytes() for path in result.iterdir()} == before
    else:
        assert not result.exists()
    assert list(result.parent.glob(".guide_fixture.*")) == []
