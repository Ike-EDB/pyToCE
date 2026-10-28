import json
import subprocess
import sys
from pathlib import Path

import pytest

from pytoce import run_lca, set_workspace
from pytoce.prepare_backgrounds.preparation.project_biosphere import load_lcia_catalogue
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_lca.calculation.results import (
    _RESULT_ARTIFACTS,
    validate_lca_result_for_refresh,
)
from pytoce.run_lca.run_lca import (
    _aggregate_status,
    _calculate_and_write_product,
    _isolated_product,
    _isolated_product_entry,
    _names,
    _Request,
    _result,
    _years,
    lca_result_directory,
)
from pytoce.shared.runtime.file_lock import writer_lock
from pytoce.shared.runtime.filesystem import replace_directory


def _write_completed_layout(result: Path) -> Path:
    """Create the files required for current manifest-last reuse tests."""

    result.mkdir(parents=True)
    for name in _RESULT_ARTIFACTS:
        (result / name).write_bytes(b"")
    manifest = result / "manifest.json"
    manifest.write_text(
        "ordinary reuse does not reopen this file\n",
        encoding="utf-8",
    )
    return manifest


def _completed_product(
    workspace: Path,
    foreground_scenario: str,
    background_scenario: str,
) -> Path:
    result = (
        workspace
        / "results"
        / foreground_scenario
        / "lca_results"
        / (f"{background_scenario}__ef31_pb_lcia__upstream_analysis_2019_2050_2100")
    )
    return _write_completed_layout(result)


def _pending_parents(
    workspace: Path,
    foreground_scenario: str,
    background_scenarios: tuple[str, ...],
) -> None:
    mfa_manifest = workspace / "results" / foreground_scenario / "mfa_results" / "manifest.json"
    mfa_manifest.parent.mkdir(parents=True)
    mfa_manifest.write_text('{"ssp_scenario":"SSP2"}\n', encoding="utf-8")
    for background in background_scenarios:
        background_manifest = (
            workspace / "data_processed" / "prepared_backgrounds" / background / "manifest.json"
        )
        background_manifest.parent.mkdir(parents=True)
        background_manifest.write_text('{"ssp_scenario":"SSP2"}\n', encoding="utf-8")


def test_readable_cartesian_products_reuse_manifest_last_results(
    tmp_path: Path,
) -> None:
    workspace = set_workspace(tmp_path).workspace
    foregrounds = ("research_b", "research_a")
    backgrounds = ("remind_ssp2_npi", "remind_ssp2_ndc")
    manifests = {
        (foreground, background): _completed_product(
            workspace,
            foreground,
            background,
        )
        for foreground in foregrounds
        for background in backgrounds
    }

    report = run_lca(
        foreground_scenario=foregrounds,
        background_scenario=backgrounds,
    )

    assert report.status == "reused"
    assert tuple(
        (item.foreground_scenario, item.background_scenario) for item in report.results
    ) == (
        ("research_a", "remind_ssp2_ndc"),
        ("research_a", "remind_ssp2_npi"),
        ("research_b", "remind_ssp2_ndc"),
        ("research_b", "remind_ssp2_npi"),
    )
    assert all(item.status == "reused" for item in report.results)
    first_item = report.results[0].as_dict()
    assert str(first_item["activation_lineage"]).endswith("activation_lineage.parquet")
    assert str(first_item["direct_treatment_carbon_flows"]).endswith(
        "direct_treatment_carbon_flows.parquet"
    )
    assert str(first_item["mismanaged_waste_fate_closure"]).endswith(
        "mismanaged_waste_fate_closure.parquet"
    )
    assert str(first_item["mfa_to_lca_flow_coverage"]).endswith("mfa_to_lca_flow_coverage.parquet")

    conventional = lca_result_directory(
        workspace,
        "research_a",
        "remind_ssp2_npi",
        normalized_lcia_method(("IPCC 2021", "climate change", "GWP 100a")),
        (2025,),
    )
    _write_completed_layout(conventional)
    conventional_report = run_lca(
        foreground_scenario="research_a",
        background_scenario="remind_ssp2_npi",
        lcia_method=("IPCC 2021", "climate change", "GWP 100a"),
        upstream_analysis_year=2025,
    )
    assert conventional_report.results[0].result_directory == conventional
    assert conventional_report.results[0].status == "reused"

    completed_bytes = {path: path.read_bytes() for path in manifests.values()}
    with pytest.raises(RuntimeError, match="FileNotFoundError.*Run run_mfa") as failure:
        run_lca(
            foreground_scenario=foregrounds,
            background_scenario=backgrounds,
            refresh=True,
        )
    assert "No completed LCA result was recorded" not in str(failure.value)
    assert {path: path.read_bytes() for path in manifests.values()} == completed_bytes


@pytest.mark.parametrize("extra_name", ["notes", "README.md"])
def test_refresh_preserves_an_unexpected_directory_in_a_completed_lca_result(
    tmp_path: Path,
    extra_name: str,
) -> None:
    result = tmp_path / "lca_result"
    result.mkdir()
    expected_files = (*_RESULT_ARTIFACTS, "README.md", "execution_report.json")
    for filename in expected_files:
        (result / filename).write_bytes(b"")
    manifest = {
        "schema_version": "pytoce_lca_result_schema_v1",
        "complete": True,
        "foreground_scenario": "remind_mfa_default",
        "background_scenario": "remind_ssp2_npi",
        "artifacts": {filename: {} for filename in (*_RESULT_ARTIFACTS, "execution_report.json")},
    }
    (result / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    note = result / extra_name
    if note.exists():
        note.unlink()
    note.mkdir()

    with pytest.raises(ValueError, match="only ordinary LCA result files"):
        validate_lca_result_for_refresh(result)

    assert note.is_dir()


def test_remind_eu_lca_products_are_reuse_only(tmp_path: Path) -> None:
    workspace = set_workspace(tmp_path).workspace
    manifest = _completed_product(
        workspace,
        "research",
        "remind_eu_ssp2_npi",
    )
    reused = run_lca(
        foreground_scenario="research",
        background_scenario="remind_eu_ssp2_npi",
    )
    assert reused.status == "reused"
    assert reused.results[0].manifest_path == manifest

    with pytest.raises(ValueError, match="retained for reuse only"):
        run_lca(
            foreground_scenario="research",
            background_scenario="remind_eu_ssp2_npi",
            refresh=True,
        )

    manifest.unlink()
    with pytest.raises(ValueError, match="retained for reuse only"):
        run_lca(
            foreground_scenario="research",
            background_scenario="remind_eu_ssp2_npi",
        )


def test_exact_method_names_are_collision_free() -> None:
    first = lca_result_directory(
        Path("workspace"),
        "research",
        "remind_ssp2_npi",
        normalized_lcia_method(("A-B", "x")),
        (2025,),
    )
    second = lca_result_directory(
        Path("workspace"),
        "research",
        "remind_ssp2_npi",
        normalized_lcia_method(("A B", "x")),
        (2025,),
    )

    assert first != second
    assert first.name == ("remind_ssp2_npi__method__ua_h_ub__x__upstream_analysis_2025")
    assert second.name == ("remind_ssp2_npi__method__ua_s_ub__x__upstream_analysis_2025")


def test_exact_methods_remain_distinct_on_case_insensitive_filesystems(tmp_path: Path) -> None:
    """Keep case, punctuation, escapes, tuple boundaries and Unicode distinct."""
    methods = (
        ("A-B", "x"),
        ("A B", "x"),
        ("a b", "x"),
        ("A_B", "x"),
        ("_ua_s_ub", "x"),
        ("A", "B x"),
        ("A B__x",),
        ("\u00e9",),
        ("e\u0301",),
        ("\U0001f600",),
        ("_z01f600",),
    )
    paths = tuple(
        lca_result_directory(
            tmp_path, "research", "remind_ssp2_npi", normalized_lcia_method(method), (2025,)
        )
        for method in methods
    )
    assert len({path.name.casefold() for path in paths}) == len(methods)
    for path in paths:
        path.mkdir(parents=True)


@pytest.mark.parametrize(
    ("method", "years"),
    [(("x" * 213,), 2019), ("ef31", tuple(range(2019, 2101, 2)))],
)
def test_overlong_lca_selection_fails_before_creating_results(
    tmp_path: Path, method: str | tuple[str, ...], years: int | tuple[int, ...]
) -> None:
    """Reject an unwritable external selection before starting a calculation."""
    workspace = set_workspace(tmp_path).workspace
    with pytest.raises(ValueError, match="at most 212.*Split analysis years"):
        run_lca(
            foreground_scenario="research",
            background_scenario="remind_ssp2_npi",
            lcia_method=method,
            upstream_analysis_year=years,
        )
    assert not (workspace / "results/research").exists()


def test_catalogued_exact_methods_support_atomic_result_replacement(tmp_path: Path) -> None:
    """Write, lock and refresh every supported exact category under all backgrounds."""
    specs, _ = load_lcia_catalogue()
    for background in (
        "remind_ssp2_ndc",
        "remind_ssp2_npi",
        "remind_ssp2_pkbudg1000",
        "remind_ssp2_pkbudg650",
    ):
        for spec in specs:
            directory = lca_result_directory(
                tmp_path,
                "research",
                background,
                normalized_lcia_method(spec.impact_category_tuple),
                (2019, 2050, 2100),
            )
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text("old", encoding="utf-8")
            staging = tmp_path / "staging"
            staging.mkdir()
            (staging / "manifest.json").write_text("new", encoding="utf-8")
            with writer_lock(directory.parent / f".{directory.name}.writer.lock"):
                replace_directory(staging, directory, replace_existing=True)
            assert (directory / "manifest.json").read_text(encoding="utf-8") == "new"


def test_public_lca_input_normalization_reports_valid_choices() -> None:
    assert _names(("scenario_b", "scenario_a", "scenario_b"), "foreground_scenario") == (
        "scenario_a",
        "scenario_b",
    )
    assert _years((2021, 2019, 2020, 2021)) == (2019, 2020, 2021)

    invalid_names: tuple[object, ...] = (None, (), ("Not Readable",), (1,))
    for value in invalid_names:
        with pytest.raises(ValueError, match="lowercase snake_case"):
            _names(value, "foreground_scenario")  # type: ignore[arg-type]

    invalid_years: tuple[object, ...] = (None, (), (True,), (2018,), (2101,))
    for value in invalid_years:
        with pytest.raises(ValueError, match="2019"):
            _years(value)  # type: ignore[arg-type]


def test_lca_report_serialization_and_mixed_status() -> None:
    request = _Request("scenario", "remind_ssp2_npi", Path("result"))
    computed = _result(request, "computed")
    reused = _result(request, "reused")

    assert _aggregate_status((computed,)) == "computed"
    assert _aggregate_status((reused,)) == "reused"
    assert _aggregate_status((computed, reused)) == "updated"

    from pytoce.run_lca.run_lca import RunLCAReport

    serialized = RunLCAReport("updated", Path("workspace"), 1.5, (computed, reused)).as_dict()
    assert serialized["workspace"] == "workspace"
    assert len(serialized["results"]) == 2  # type: ignore[arg-type]


def test_lca_product_requires_completed_mfa_and_preserves_incomplete_output(tmp_path: Path) -> None:
    """Require the MFA parent and preserve an incomplete result directory."""
    selection = normalized_lcia_method("ef31")
    missing_mfa = _Request("scenario", "remind_ssp2_npi", tmp_path / "missing_mfa")
    with pytest.raises(FileNotFoundError, match="Run run_mfa"):
        _calculate_and_write_product(tmp_path, missing_mfa, selection, (2019,), refresh=False)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="has no valid completion manifest"):
        _calculate_and_write_product(
            tmp_path,
            _Request("scenario", "remind_ssp2_npi", incomplete),
            selection,
            (2019,),
            refresh=False,
        )


def test_isolated_entry_reuses_completed_product(tmp_path: Path) -> None:
    result = tmp_path / "result"
    result.mkdir()
    (result / "manifest.json").write_text("complete\n", encoding="utf-8")
    payload = tmp_path / "payload.json"
    response = tmp_path / "response.json"
    payload.write_text(
        json.dumps(
            {
                "schema_version": "pytoce_lca_product_request_schema_v1",
                "workspace": str(tmp_path),
                "foreground_scenario": "scenario",
                "background_scenario": "remind_ssp2_npi",
                "result_directory": str(result),
                "lcia_method": {"groups": ["ef31"], "exact_method": []},
                "years": [2019],
                "refresh": False,
            }
        ),
        encoding="utf-8",
    )

    assert _isolated_product_entry(str(payload), str(response)) == 0
    assert json.loads(response.read_text(encoding="utf-8")) == {
        "schema_version": "pytoce_lca_product_response_schema_v1",
        "status": "reused",
    }

    blocked_response = tmp_path / "blocked_response"
    blocked_response.mkdir()
    with pytest.raises(OSError):
        _isolated_product_entry(str(payload), str(blocked_response))
    assert (result / "manifest.json").read_text(encoding="utf-8") == "complete\n"


@pytest.mark.parametrize(
    ("method", "expected_groups", "expected_exact"),
    [
        (("ef31", "pb_lcia"), ["ef31", "pb_lcia"], []),
        (
            ("IPCC 2021", "climate change", "GWP 100a"),
            [],
            ["IPCC 2021", "climate change", "GWP 100a"],
        ),
        ('["ef31"]', [], ["ef31"]),
        ('["ef31","pb_lcia"]', [], ["ef31", "pb_lcia"]),
    ],
)
def test_pending_products_preserve_method_selection_across_interpreters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str | tuple[str, ...],
    expected_groups: list[str],
    expected_exact: list[str],
) -> None:
    """Keep explicit exact methods distinct from groups at the process boundary."""
    workspace = set_workspace(tmp_path).workspace
    foreground = "research"
    backgrounds = ("remind_ssp2_npi", "remind_ssp2_ndc")
    _pending_parents(workspace, foreground, backgrounds)
    observed_payloads: list[dict[str, object]] = []

    def isolated_boundary(
        command: list[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert command[:2] == [sys.executable, "-c"]
        payload_path = Path(command[-2])
        response_path = Path(command[-1])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        observed_payloads.append(payload)
        result_directory = Path(str(payload["result_directory"]))
        result_directory.mkdir(parents=True)
        (result_directory / "manifest.json").write_text(
            "isolated completed product\n",
            encoding="utf-8",
        )
        response_path.write_text(
            '{"schema_version":"pytoce_lca_product_response_schema_v1","status":"computed"}\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", isolated_boundary)

    report = run_lca(
        foreground_scenario=foreground,
        background_scenario=backgrounds,
        lcia_method=method,
    )

    assert report.status == "computed"
    assert [payload["background_scenario"] for payload in observed_payloads] == [
        "remind_ssp2_ndc",
        "remind_ssp2_npi",
    ]
    assert all(payload["foreground_scenario"] == foreground for payload in observed_payloads)
    assert all(
        payload["lcia_method"] == {"groups": expected_groups, "exact_method": expected_exact}
        for payload in observed_payloads
    )
    assert all(payload["years"] == [2019, 2050, 2100] for payload in observed_payloads)
    assert all(item.status == "computed" for item in report.results)
    assert all(item.manifest_path.is_file() for item in report.results)


def test_isolated_product_failure_has_public_product_context(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            "foreground_scenario='research'.*"
            "background_scenario='remind_ssp2_npi'.*FileNotFoundError"
        ),
    ):
        _isolated_product(
            tmp_path,
            _Request(
                foreground_scenario="research",
                background_scenario="remind_ssp2_npi",
                result_directory=tmp_path / "result",
            ),
            normalized_lcia_method(("ef31", "pb_lcia")),
            (2019, 2050, 2100),
        )
