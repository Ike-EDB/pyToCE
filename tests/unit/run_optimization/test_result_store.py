import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
from completed_search_fixtures import make_completed_search
from jsonschema import Draft202012Validator, validate  # type: ignore[import-untyped]
from result_store_fixtures import (
    _METHOD,
    _PARENTS,
    _PRODUCT,
    _SELECTED_TABLES,
    _SUMMARY,
    _save_result,
    _tables,
)
from selected_reporting_fixtures import BACKGROUNDS, METHOD

from pytoce.run_optimization.calculation.integrated_search import (
    INTEGRATED_SEARCH_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.selected_stage_reporting import (
    _stage_schema_sample,
)
from pytoce.run_optimization.runtime import result_store
from pytoce.run_optimization.runtime.completed_search import encode_completed_search
from pytoce.shared.runtime.filesystem import canonical_table_sha256


def _stage_frames() -> tuple[tuple[pd.DataFrame, ...], pd.DataFrame]:
    """Return annual and cumulative virgin-resin LCIA rows with native nullable years."""
    annual = pd.DataFrame(
        {
            "candidate_identity": pd.Series(["candidate::one"] * 82, dtype="string"),
            "target_year": pd.Series([2050] * 82, dtype="Int64"),
            "characterization_background_scenario": BACKGROUNDS[0],
            "region": "EUR",
            "year": pd.Series(range(2019, 2101), dtype="int64").astype(object),
            **METHOD,
            "life_cycle_stage": "virgin_resin_production",
            "stage_group": "material_production",
            "cutoff_cycle_role": "primary_material_cycle",
            "material_cycle_source": "primary_resin",
            "physical_event_basis": "year primary material enters the virgin node",
            "physical_event_year": pd.Series(range(2019, 2101), dtype="int64").astype(object),
            "score": pd.Series([1.5, -0.5, *([0.0] * 80)], dtype="float64"),
            "contribution_norm": pd.Series([1.5, 0.5, *([0.0] * 80)], dtype="float64"),
            "period_start_year": range(2019, 2101),
            "period_end_year": range(2019, 2101),
            "period_kind": "annual",
        }
    )
    cumulative = annual.iloc[[0]].copy()
    # Cumulative stage rows have no individual event year. The reporting owner
    # supplies object columns with NaN after combining the 12 world regions and derived-global rows.
    cumulative["year"] = pd.Series(float("nan"), index=cumulative.index, dtype=object)
    cumulative["physical_event_year"] = pd.Series(
        float("nan"), index=cumulative.index, dtype=object
    )
    cumulative["score"] = 1.0
    cumulative["contribution_norm"] = 2.0
    cumulative["period_start_year"] = 2019
    cumulative["period_end_year"] = 2100
    cumulative["period_kind"] = "cumulative_2019_2100"
    frames = (annual, cumulative)
    return frames, pd.concat(frames, ignore_index=True)


def _save_streamed_result(
    directory: Path,
    frames: tuple[pd.DataFrame, ...],
) -> tuple[result_store.SavedOptimizationResult, result_store.StagedTable]:
    with result_store.stage_selected_life_cycle_scores(directory) as writer:
        writer.start(_stage_schema_sample(frames))
        for frame in frames:
            writer.write(frame)
        staged = writer.finish()
        tables: dict[str, pd.DataFrame | result_store.StagedTable] = _tables(_SELECTED_TABLES)
        tables["selected_life_cycle_stage_scores.parquet"] = staged
        receipt = result_store.write_optimization_result(
            directory,
            product_identity=_PRODUCT,
            parent_identities=_PARENTS,
            method_identity=_METHOD,
            tables=tables,
            summary=_SUMMARY,
        )
        return receipt, staged


def test_target_uses_one_readable_request_root(tmp_path: Path) -> None:
    target = result_store.OptimizationResultTarget(
        workspace=tmp_path.resolve(),
        foreground_scenario="remind_mfa_default",
        allocated_cc_result="pyaesa_h12",
        optimization_lcia_selection="ef31",
        readable_request_selection="blocks_demand_lifetime_integrated_cases_p95",
        initial_grid_step_pct=20,
        endpoint_refinement_steps_pct=(10, 5),
    )

    expected = (
        tmp_path.resolve()
        / "results/remind_mfa_default/optimization_results"
        / "pyaesa_h12/ef31/blocks_demand_lifetime_integrated_cases_p95__stock_lp_v6"
    )
    assert target.root == expected
    assert target.selected_results("ef31_pb_lcia") == (
        expected
        / "selected_results"
        / "joint_lines_v1_stock_lp_v6__ef31_pb_lcia__grid_20pct__refinement_10_5pct"
    )
    assert not hasattr(target, "private_article_atlas")
    assert "deterministic_reference" not in str(target.root)
    assert "candidate_archive" not in str(target.root)


@pytest.mark.parametrize("previous_suffix", ("", "__joint_roles_v2_stock_lp_v6"))
def test_changed_search_does_not_adopt_a_completed_previous_method(
    tmp_path: Path, previous_suffix: str
) -> None:
    """A completed method-specific product must not satisfy a different search request."""
    target = result_store.OptimizationResultTarget(
        tmp_path, "remind_mfa_default", "capacity", "ef31", "all_blocks", 20, (10, 5)
    )
    previous_root = target.root.with_name(target.readable_request_selection + previous_suffix)
    previous = previous_root / "selected_results/ef31__grid_20pct__refinement_10_5pct"
    _save_result(previous)
    original = {path.name: path.read_bytes() for path in previous.iterdir()}

    assert not result_store.product_complete(target.selected_results("ef31"))
    _save_result(target.selected_results("ef31"), product_identity="product::new-search")
    assert result_store.read_product_manifest(previous).product_identity == _PRODUCT
    assert {path.name: path.read_bytes() for path in previous.iterdir()} == original
    assert (
        result_store.read_product_manifest(target.selected_results("ef31")).product_identity
        == "product::new-search"
    )


def test_finer_selected_product_preserves_completed_coarser_result(tmp_path: Path) -> None:
    """A later resolution publishes separately while retaining the same candidate owner."""
    coarse = result_store.OptimizationResultTarget(
        workspace=tmp_path,
        foreground_scenario="remind_mfa_default",
        allocated_cc_result="pyaesa_h12",
        optimization_lcia_selection="ef31",
        readable_request_selection="blocks_integrated_cases_p95",
        initial_grid_step_pct=20,
        endpoint_refinement_steps_pct=(10, 5),
    )
    fine = replace(coarse, endpoint_refinement_steps_pct=(10, 5, 1))
    coarse_directory = coarse.selected_results("ef31")
    coarse_receipt = _save_result(coarse_directory)
    original = {path.name: path.read_bytes() for path in coarse_directory.iterdir()}
    fine_receipt = _save_result(
        fine.selected_results("ef31"), product_identity="optimization-product::finer"
    )
    assert fine.root == coarse.root
    assert fine_receipt.product_identity != coarse_receipt.product_identity
    assert result_store.read_product_manifest(coarse_directory).product_identity == _PRODUCT
    assert {path.name: path.read_bytes() for path in coarse_directory.iterdir()} == original


@pytest.mark.parametrize(
    "longer",
    (
        "absent",
        "input_changed",
        "ranking_changed",
        "method_changed",
        "configuration_changed",
        "compatible",
    ),
)
def test_extension_reads_longest_compatible_completed_prefix(tmp_path: Path, longer: str) -> None:
    """Resume complete matching stages; retain the five-point source bytes unchanged."""
    target = result_store.OptimizationResultTarget(
        tmp_path, "remind_mfa_default", "capacity", "ef31", "all_blocks", 100, (10, 5, 2, 1)
    )
    arguments = dict(
        result_lcia_selection="ef31",
        candidate_input_identity="inputs::same",
        selection_method_identity="ranking::same",
        requested_blocks=("integrated_frontier",),
    )
    assert result_store.read_refinement_prefix(target, **arguments) is None
    completed = make_completed_search()
    summary = {
        "candidate_input_identity": "inputs::same",
        "selection_method": "ranking::same",
        "integrated_search_method": INTEGRATED_SEARCH_METHOD_IDENTITY,
        "completed_search": encode_completed_search(completed),
    }
    coarse = replace(target, endpoint_refinement_steps_pct=(10, 5))
    directory = coarse.selected_results("ef31")
    _save_result(directory, summary=summary)
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    if longer != "absent":
        intermediate = replace(target, endpoint_refinement_steps_pct=(10, 5, 2))
        longer_summary = {
            **summary,
            "completed_search": encode_completed_search(make_completed_search(steps=(10, 5, 2))),
        }
        if longer == "configuration_changed":
            longer_summary["completed_search"]["search_configuration_identity"] = (
                "different::search"
            )
        for label, key in (
            ("input_changed", "candidate_input_identity"),
            ("ranking_changed", "selection_method"),
            ("method_changed", "integrated_search_method"),
        ):
            if longer == label:
                longer_summary[key] = "different::authority"
        _save_result(
            intermediate.selected_results("ef31"),
            summary=longer_summary,
            product_identity="product::longer",
        )
    prefix = result_store.read_refinement_prefix(target, **arguments)
    assert prefix is not None
    expected = make_completed_search(steps=(10, 5, 2)) if longer == "compatible" else completed
    assert prefix.completed_search == expected
    assert prefix.product_identity == ("product::longer" if longer == "compatible" else _PRODUCT)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
    assert (
        result_store.read_refinement_prefix(replace(target, initial_grid_step_pct=50), **arguments)
        is None
    )


def test_standalone_final_stage_is_not_an_intermediate_pruning_prefix(tmp_path: Path) -> None:
    """A final 10-point refinement is unpruned; the 10-point stage of a 10/5 schedule can prune."""
    target = result_store.OptimizationResultTarget(
        tmp_path, "remind_mfa_default", "capacity", "ef31", "all_blocks", 100, (10, 5)
    )
    summary = {
        "candidate_input_identity": "inputs::same",
        "selection_method": "ranking::same",
        "integrated_search_method": INTEGRATED_SEARCH_METHOD_IDENTITY,
        "completed_search": encode_completed_search(make_completed_search(steps=(10,))),
    }
    _save_result(
        replace(target, endpoint_refinement_steps_pct=(10,)).selected_results("ef31"),
        summary=summary,
    )
    assert (
        result_store.read_refinement_prefix(
            target,
            result_lcia_selection="ef31",
            candidate_input_identity="inputs::same",
            selection_method_identity="ranking::same",
            requested_blocks=("integrated_frontier",),
        )
        is None
    )


def test_manifest_presence_is_the_only_fast_completion_check(tmp_path: Path) -> None:
    directory = tmp_path / "selected"
    directory.mkdir()
    assert not result_store.product_complete(directory)

    (directory / "manifest.json").write_text("not opened", encoding="utf-8")
    assert result_store.product_complete(directory)


def test_save_result_writes_only_the_exact_selected_product(tmp_path: Path) -> None:
    directory = tmp_path / "ef31_pb_lcia"
    receipt = _save_result(directory)

    assert receipt.status == "computed"
    assert receipt.product_identity == _PRODUCT
    assert {path.name for path in directory.iterdir()} == {
        *_SELECTED_TABLES,
        "method_summary.json",
        "manifest.json",
    }
    for name in _SELECTED_TABLES:
        frame = pd.read_parquet(directory / name)
        assert len(frame) == 1
        assert frame.loc[0, "value"] == -1.5
    assert json.loads((directory / "method_summary.json").read_text(encoding="utf-8")) == (_SUMMARY)

    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert "product_kind" not in manifest
    assert manifest["product_identity"] == _PRODUCT
    assert manifest["parent_identities"] == dict(sorted(_PARENTS.items()))
    assert manifest["method_identity"] == _METHOD
    assert set(manifest["artifacts"]) == {*_SELECTED_TABLES, "method_summary.json"}
    assert result_store.product_complete(directory)


def test_streamed_table_preserves_logical_order_schema_hash_and_parquet(tmp_path: Path) -> None:
    frames, logical = _stage_frames()
    directory = tmp_path / "chunked"
    _receipt, staged = _save_streamed_result(directory, frames)

    assert staged.row_count == len(logical)
    assert staged.canonical_sha256 == canonical_table_sha256(logical)

    reference = tmp_path / "reference.parquet"
    logical.to_parquet(reference, index=False, compression="zstd")
    stored = pd.read_parquet(directory / "selected_life_cycle_stage_scores.parquet")
    pd.testing.assert_frame_equal(
        stored,
        pd.read_parquet(reference),
        check_dtype=True,
        check_exact=True,
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    descriptor = manifest["artifacts"]["selected_life_cycle_stage_scores.parquet"]
    assert descriptor["row_count"] == len(logical)
    assert descriptor["columns"] == list(logical.columns)
    assert stored["unit"].eq("kg CO2-Eq").all()
    annual = stored.loc[stored["period_kind"].eq("annual")]
    cumulative = stored.loc[stored["period_kind"].eq("cumulative_2019_2100")]
    assert annual["year"].tolist() == list(range(2019, 2101))
    assert annual["physical_event_year"].equals(annual["year"])
    assert cumulative[["year", "physical_event_year"]].isna().all().all()
    assert cumulative["score"].item() == annual["score"].sum() == 1.0
    assert cumulative["contribution_norm"].item() == annual["contribution_norm"].sum() == 2.0


def test_packaged_schema_matches_a_saved_result_manifest(tmp_path: Path) -> None:
    directory = tmp_path / "ef31"
    _save_result(directory)
    schema_path = (
        Path(result_store.__file__).parents[1]
        / "data/schemas/optimization_selected_result_schema_v1.json"
    )
    schema_bytes = schema_path.read_bytes()
    schema = json.loads(schema_bytes)
    manifest = json.loads((directory / "manifest.json").read_bytes())

    Draft202012Validator.check_schema(schema)
    validate(instance=manifest, schema=schema)
    assert schema["$id"] == result_store.SELECTED_RESULT_SCHEMA
    assert result_store.selected_result_schema_sha256() == sha256(schema_bytes).hexdigest()


def test_ordinary_reuse_reads_only_the_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "ef31"
    first = _save_result(directory)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ordinary reuse attempted table serialization")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", forbidden)
    second = _save_result(directory)
    assert second.status == "reused"
    assert second.product_identity == first.product_identity


def test_reuse_reads_manifest_only(tmp_path: Path) -> None:
    directory = tmp_path / "ef31"
    receipt = _save_result(directory)
    for name in _SELECTED_TABLES:
        (directory / name).unlink()

    reused = result_store.read_product_manifest(directory)
    assert reused.status == "reused"
    assert reused.product_identity == receipt.product_identity


def test_refresh_replaces_only_the_exact_completed_owner_with_new_identity(tmp_path: Path) -> None:
    directory = tmp_path / "ef31"
    _save_result(directory)
    updated_tables = _tables(_SELECTED_TABLES, value=4.0)

    receipt = _save_result(
        directory,
        tables=updated_tables,
        product_identity="optimization-product::updated",
        summary={"method": "candidate-search-v1"},
        refresh=True,
    )

    assert receipt.status == "updated"
    assert receipt.product_identity == "optimization-product::updated"
    assert pd.read_parquet(directory / "selected_coordinates.parquet").loc[0, "value"] == 4.0
    assert not tuple(tmp_path.glob(".ef31.staging.*"))
    assert not tuple(tmp_path.glob(".ef31.previous.*"))


def test_refresh_refuses_to_remove_an_unowned_entry(tmp_path: Path) -> None:
    directory = tmp_path / "ef31"
    _save_result(directory)
    user_file = directory / "user_note.txt"
    user_file.write_text("preserve", encoding="utf-8")

    with pytest.raises(result_store.OptimizationResultError, match="unexpected entries"):
        _save_result(directory, refresh=True)
    assert user_file.read_text(encoding="utf-8") == "preserve"
    assert not tuple(tmp_path.glob(".ef31.staging.*"))


def test_refresh_preflight_rejects_an_unowned_entry(tmp_path: Path) -> None:
    directory = tmp_path / "ef31"
    _save_result(directory)
    note = directory / "user_note.txt"
    note.write_text("preserve", encoding="utf-8")

    with pytest.raises(result_store.OptimizationResultError, match="unexpected entries"):
        result_store.validate_result_for_refresh(directory)

    assert note.read_text(encoding="utf-8") == "preserve"


def test_ordinary_write_failure_cleans_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "ef31"

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("bounded write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="bounded write failure"):
        _save_result(directory)
    assert not directory.exists()
    assert not tuple(tmp_path.glob(".ef31.staging.*"))


def test_streamed_write_failure_cleans_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "ef31"
    frames, _logical = _stage_frames()

    class FailingParquetWriter:
        def __init__(self, where: Path, *_args: object, **_kwargs: object) -> None:
            Path(where).write_bytes(b"partial")

        def write_table(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("bounded chunk write failure")

        def close(self) -> None:
            pass

    monkeypatch.setattr(result_store.pq, "ParquetWriter", FailingParquetWriter)
    with (
        pytest.raises(OSError, match="bounded chunk write failure"),
        result_store.stage_selected_life_cycle_scores(directory) as writer,
    ):
        writer.start(_stage_schema_sample(frames))
        writer.write(frames[0])
    assert not directory.exists()
    assert not tuple(tmp_path.glob(".ef31.staging.*"))


def test_save_preserves_an_unrelated_hidden_directory(tmp_path: Path) -> None:
    directory = tmp_path / "ef31"
    unrelated = tmp_path / ".ef31.staging.external"
    unrelated.mkdir()
    note = unrelated / "user_note.txt"
    note.write_text("preserve", encoding="utf-8")

    receipt = _save_result(directory)

    assert receipt.status == "computed"
    assert result_store.product_complete(directory)
    assert note.read_text(encoding="utf-8") == "preserve"


def test_guide_is_adjacent_complete_and_never_replaced(tmp_path: Path) -> None:
    target = result_store.OptimizationResultTarget(
        workspace=tmp_path.resolve(),
        foreground_scenario="remind_mfa_default",
        allocated_cc_result="pyaesa_h12",
        optimization_lcia_selection="ef31",
        readable_request_selection="blocks_demand_lifetime",
        initial_grid_step_pct=20,
        endpoint_refinement_steps_pct=(10, 5),
    )
    guide = result_store.ensure_optimization_results_guide(target)
    content = guide.read_text(encoding="utf-8")
    assert guide == (
        tmp_path.resolve() / "results/remind_mfa_default/optimization_results/README.md"
    )
    assert "selected results" in content
    assert "Manifest" in content or "manifest" in content

    guide.write_text("user-owned\n", encoding="utf-8")
    assert result_store.ensure_optimization_results_guide(target) == guide
    assert guide.read_text(encoding="utf-8") == "user-owned\n"
