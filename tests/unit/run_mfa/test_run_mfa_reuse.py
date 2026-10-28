"""Public MFA selection and manifest-last reuse with actual stored products."""

import re
from pathlib import Path

import pytest
from mfa_workspace_fixtures import StoredMFAFixture, prepare_stored_mfa

from pytoce.run_mfa.run_mfa import (
    _aggregate_status,
    _mfa_result_report,
    _run_one,
    run_mfa,
)


def test_public_reuse_deduplicates_and_sorts_readable_scenario_names(
    saved_mfa: StoredMFAFixture, tmp_path: Path, remind_mfa_source_root: Path
) -> None:
    """Return each completed scientific parent once in readable-name order."""
    second = prepare_stored_mfa(
        tmp_path, remind_mfa_source_root, foreground_scenario="another_fixture"
    )
    names = (saved_mfa.compiled.foreground_scenario, second.compiled.foreground_scenario)
    manifests = {
        item.target.result_directory / "manifest.json": (
            item.target.result_directory / "manifest.json"
        ).read_bytes()
        for item in (saved_mfa, second)
    }
    report = run_mfa(foreground_scenario=[names[0], names[1], names[0]])
    assert report.status == "reused"
    assert report.workspace == tmp_path
    assert tuple(item.foreground_scenario for item in report.results) == tuple(sorted(names))
    assert all(item.status == "reused" for item in report.results)
    assert [item["foreground_scenario"] for item in report.as_dict()["results"]] == sorted(names)
    assert {path: path.read_bytes() for path in manifests} == manifests


def test_reuse_preserves_changed_yaml_until_explicit_refresh(
    saved_mfa: StoredMFAFixture, tmp_path: Path
) -> None:
    """Reuse the admitted parent, but reopen user YAML when refresh is requested."""
    name = saved_mfa.compiled.foreground_scenario
    scenario = tmp_path / "data_raw/foreground_scenarios/scenarios" / (name + ".yaml")
    scenario.write_text("not valid YAML: [\n", encoding="utf-8")
    manifest = saved_mfa.target.result_directory / "manifest.json"
    original = manifest.read_bytes()
    report = run_mfa(foreground_scenario=name)
    assert report.status == "reused"
    assert report.results[0].manifest_path == manifest
    with pytest.raises(ValueError, match="cannot be read as UTF-8 YAML"):
        run_mfa(foreground_scenario=name, refresh=True)
    assert manifest.read_bytes() == original


def test_refresh_preserves_unknown_contents(saved_mfa: StoredMFAFixture, tmp_path: Path) -> None:
    """Refuse to replace a result containing user files before recalculation."""
    directory = saved_mfa.target.result_directory
    note = directory / "notes.txt"
    note.write_text("user notes", encoding="utf-8")
    manifest = directory / "manifest.json"
    original = manifest.read_bytes()
    progress: list[tuple[str, int | None]] = []

    with pytest.raises(ValueError, match="not the completed package MFA result"):
        _run_one(
            tmp_path,
            saved_mfa.compiled,
            request=saved_mfa.request,
            session=saved_mfa.session,
            recipe=saved_mfa.recipe,
            composition=saved_mfa.composition,
            refresh=True,
            progress=lambda phase, year: progress.append((phase, year)),
        )

    assert note.read_text(encoding="utf-8") == "user notes"
    assert manifest.read_bytes() == original
    assert progress == []


def test_aggregate_status_distinguishes_uniform_and_mixed_result_sets(tmp_path: Path) -> None:
    """Report updated whenever a request combines different item outcomes."""
    computed = _mfa_result_report("a", tmp_path / "a", status="computed")
    reused = _mfa_result_report("b", tmp_path / "b", status="reused")
    updated = _mfa_result_report("c", tmp_path / "c", status="updated")
    assert _aggregate_status((computed,)) == "computed"
    assert _aggregate_status((reused,)) == "reused"
    assert _aggregate_status((computed, reused)) == "updated"
    assert _aggregate_status((updated,)) == "updated"


@pytest.mark.parametrize("incomplete", (False, True))
def test_locked_admission_preserves_a_result_that_appeared_before_the_writer_lock(
    saved_mfa: StoredMFAFixture, tmp_path: Path, incomplete: bool
) -> None:
    """Recheck external completion state under the lock before calculation or replacement."""
    manifest = saved_mfa.target.result_directory / "manifest.json"
    before = {path: path.read_bytes() for path in manifest.parent.rglob("*") if path.is_file()}
    if incomplete:
        manifest.unlink()
        del before[manifest]
    progress: list[tuple[str, int | None]] = []

    def calculate():
        return _run_one(
            tmp_path,
            saved_mfa.compiled,
            request=saved_mfa.request,
            session=saved_mfa.session,
            recipe=saved_mfa.recipe,
            composition=saved_mfa.composition,
            refresh=False,
            progress=lambda phase, year: progress.append((phase, year)),
        )

    if incomplete:
        with pytest.raises(ValueError, match=re.escape(str(manifest.parent))):
            calculate()
        assert progress == []
    else:
        report = calculate()
        assert report.status == "reused"
        assert report.manifest_path == manifest
        assert progress == [("reuse", None)]
    assert {path: path.read_bytes() for path in before} == before
