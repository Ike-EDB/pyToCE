import json
import os
import subprocess
from importlib import resources
from pathlib import Path

import pandas as pd
import pytest
from allocated_cc_fixtures import (
    _authority,
    _methods,
    _source_frames,
    _write_raw_result,
    _write_shared_authority,
)
from jsonschema import Draft202012Validator

from pytoce.run_optimization.input.allocated_cc import (
    select_allocated_cc_cases,
    select_allocated_cc_result,
)
from pytoce.run_optimization.input.allocated_cc_contract import (
    AllocatedCCError,
    validate_allocated_cc_frames,
)
from pytoce.run_optimization.input.allocated_cc_paths import _read_csv
from pytoce.run_optimization.input.allocated_cc_product import (
    compile_allocated_cc_result,
)
from pytoce.run_optimization.run_optimization import normalize_run_optimization_request
from pytoce.shared.runtime.file_lock import writer_lock


def test_direct_allocated_cc_contract_validates_coordinates_and_identities() -> None:
    annual_source, cumulative_source = _source_frames(("strict_case", "lenient_case"))
    annual, cumulative, cases = validate_allocated_cc_frames(
        annual_source.sample(frac=1.0, random_state=7),
        cumulative_source.sample(frac=1.0, random_state=11),
        _authority(),
    )

    assert cases == ("lenient_case", "strict_case")
    assert len(annual) == 2 * 12 * 16 * 82
    assert len(cumulative) == 2 * 12
    assert annual["factor_identity"].str.startswith("factor::").all()
    assert cumulative["factor_identity"].str.startswith("factor::").all()
    assert annual["region"].nunique() == 12
    assert cumulative["region"].nunique() == 12


def test_annual_allocated_cc_uses_the_complete_ef31_domain() -> None:
    annual_source, cumulative_source = _source_frames()
    annual, cumulative, cases = validate_allocated_cc_frames(
        annual_source,
        cumulative_source,
        _authority(),
    )

    assert cases == ("direct_case",)
    assert len(annual) == 12 * 82 * 16
    assert len(cumulative) == 12
    assert set(annual["lcia_method"]) == {"ef31"}
    assert set(cumulative["lcia_method"]) == {"ef31"}


def test_allocated_cc_csv_years_preserve_exact_integer_decimal_and_exponent_forms(
    tmp_path: Path,
) -> None:
    """Admit equivalent lexical years and preserve ordered values and table types."""
    annual, cumulative = _source_frames()
    authority = _authority()
    expected_annual, expected_cumulative, expected_cases = validate_allocated_cc_frames(
        annual, cumulative, authority
    )
    annual["year"] = annual["year"].map(
        lambda year: str(year) if year % 3 == 0 else f"{year}.000" if year % 3 == 1 else f"{year}e0"
    )
    cumulative["period_start"] = "2019.000"
    cumulative["period_end"] = "2.100e3"
    annual_path = tmp_path / "annual.csv"
    cumulative_path = tmp_path / "cumulative.csv"
    annual.to_csv(annual_path, index=False)
    cumulative.to_csv(cumulative_path, index=False)
    annual_source, _ = _read_csv(annual_path, role="annual_allocated_cc")
    cumulative_source, _ = _read_csv(cumulative_path, role="cumulative_gwp100_allocated_cc")

    observed_annual, observed_cumulative, cases = validate_allocated_cc_frames(
        annual_source, cumulative_source, authority
    )

    pd.testing.assert_frame_equal(observed_annual, expected_annual, check_exact=True)
    pd.testing.assert_frame_equal(observed_cumulative, expected_cumulative, check_exact=True)
    assert cases == expected_cases


@pytest.mark.parametrize(
    ("column", "invalid"),
    (
        ("year", "2019.5"),
        ("year", "2019.0000000000000001"),
        ("year", "NaN"),
        ("period_start", "not_a_year"),
        ("period_start", "-9223372036854775809"),
        ("period_end", "9223372036854775808"),
    ),
)
def test_allocated_cc_csv_rejects_inexact_or_unrepresentable_years(
    tmp_path: Path, column: str, invalid: str
) -> None:
    """Reject real CSV year fields before int64 conversion or coordinate admission."""
    annual, cumulative = _source_frames()
    source = annual if column == "year" else cumulative
    source[column] = source[column].astype(str)
    source.loc[0, column] = invalid
    path = tmp_path / "invalid_year.csv"
    source.to_csv(path, index=False)
    admitted_source, _ = _read_csv(path, role="annual" if column == "year" else "cumulative")
    if column == "year":
        annual = admitted_source
    else:
        cumulative = admitted_source

    with pytest.raises(AllocatedCCError, match=f"Column '{column}' must contain whole numbers"):
        validate_allocated_cc_frames(annual, cumulative, _authority())


def test_direct_allocated_cc_validation_fails_invalid_scientific_coordinates() -> None:
    annual_source, cumulative_source = _source_frames()
    methods = _methods()

    invalid_annual = annual_source.iloc[:-1].copy()
    with pytest.raises(AllocatedCCError, match="must contain one row for every case"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = annual_source.copy()
    invalid_annual.loc[0, "allocated_cc"] = 0.0
    with pytest.raises(AllocatedCCError, match="0.0 at row 0.*greater than zero"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = annual_source.copy()
    invalid_annual.loc[0, "allocated_cc"] = float("nan")
    with pytest.raises(AllocatedCCError, match="nan at row 0.*finite value greater than zero"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = annual_source.copy()
    invalid_annual.loc[0, "allocated_cc"] = -1.0
    with pytest.raises(AllocatedCCError, match="-1.0 at row 0.*greater than zero"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = pd.concat([annual_source, annual_source.iloc[[0]]], ignore_index=True)
    with pytest.raises(AllocatedCCError, match="repeats a case"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = annual_source.copy()
    invalid_annual.loc[0, "impact_category_id"] = "not_a_standard_category"
    with pytest.raises(AllocatedCCError, match="not_a_standard_category.*Choose from"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = annual_source.copy()
    invalid_annual.loc[0, "region"] = "not_h12"
    with pytest.raises(AllocatedCCError, match="not_h12.*world region names"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = annual_source.copy()
    invalid_annual.loc[0, "year"] = 2018
    with pytest.raises(AllocatedCCError, match="2019 through 2100"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = annual_source.copy()
    invalid_annual.loc[0, "unit"] = "wrong unit"
    with pytest.raises(AllocatedCCError, match="wrong unit.*Replace it with"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = annual_source.copy()
    invalid_annual.loc[0, "region"] = f"{invalid_annual.loc[0, 'region']} "
    with pytest.raises(AllocatedCCError, match="surrounding whitespace"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = annual_source.copy()
    invalid_annual.loc[0, "unit"] = f" {invalid_annual.loc[0, 'unit']}"
    with pytest.raises(AllocatedCCError, match="surrounding whitespace"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_annual = annual_source.copy()
    invalid_annual.loc[0, "lcia_method"] = "ef31 "
    with pytest.raises(AllocatedCCError, match="surrounding whitespace"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))

    invalid_cumulative = cumulative_source.copy()
    invalid_cumulative.loc[0, "budget_metric"] = "sum_of_annual_ratios"
    with pytest.raises(AllocatedCCError, match="must use impact_category_id"):
        validate_allocated_cc_frames(annual_source, invalid_cumulative, _authority(methods))

    invalid_cumulative = cumulative_source.copy()
    invalid_cumulative.loc[0, "allocated_cc_case"] = "unmatched_case"
    with pytest.raises(AllocatedCCError, match="same allocated_cc_case names"):
        validate_allocated_cc_frames(annual_source, invalid_cumulative, _authority(methods))

    invalid_cumulative = cumulative_source.copy()
    invalid_cumulative.loc[0, "impact_category_id"] = "acidification"
    with pytest.raises(AllocatedCCError, match="climate_change"):
        validate_allocated_cc_frames(annual_source, invalid_cumulative, _authority(methods))

    invalid_cumulative = cumulative_source.copy()
    invalid_cumulative.loc[0, "period_end"] = 2099
    with pytest.raises(AllocatedCCError, match="period_end=2100"):
        validate_allocated_cc_frames(annual_source, invalid_cumulative, _authority(methods))

    invalid_cumulative = cumulative_source.copy()
    invalid_cumulative.loc[0, "unit"] = "wrong unit"
    with pytest.raises(AllocatedCCError, match="wrong unit.*Replace it with"):
        validate_allocated_cc_frames(annual_source, invalid_cumulative, _authority(methods))

    invalid_cumulative = cumulative_source.iloc[:-1].copy()
    with pytest.raises(AllocatedCCError, match="one row for every case and world region"):
        validate_allocated_cc_frames(annual_source, invalid_cumulative, _authority(methods))

    invalid_cumulative = pd.concat(
        [cumulative_source, cumulative_source.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(AllocatedCCError, match="repeats a case and region row"):
        validate_allocated_cc_frames(annual_source, invalid_cumulative, _authority(methods))

    for invalid_value in (0.0, -1.0, float("nan")):
        invalid_cumulative = cumulative_source.copy()
        invalid_cumulative.loc[0, "cumulative_allocated_cc"] = invalid_value
        with pytest.raises(AllocatedCCError, match="finite value greater than zero"):
            validate_allocated_cc_frames(
                annual_source,
                invalid_cumulative,
                _authority(methods),
            )

    invalid_cumulative = cumulative_source.copy()
    invalid_cumulative.loc[0, "region"] = "not_h12"
    with pytest.raises(AllocatedCCError, match="not_h12.*world region names"):
        validate_allocated_cc_frames(annual_source, invalid_cumulative, _authority(methods))

    invalid_annual = annual_source.assign(allocated_share=1.0)
    with pytest.raises(AllocatedCCError, match="columns"):
        validate_allocated_cc_frames(invalid_annual, cumulative_source, _authority(methods))


def test_allocated_cc_accepts_arbitrarily_many_complete_named_cases() -> None:
    requested = ("allocation_a", "allocation_b", "null")
    annual_source, cumulative_source = _source_frames(requested)
    annual, cumulative, cases = validate_allocated_cc_frames(
        annual_source,
        cumulative_source,
        _authority(),
    )
    assert cases == requested
    assert len(annual) == len(requested) * 12 * 16 * 82
    assert len(cumulative) == len(requested) * 12


def test_allocated_cc_compilation_is_manifest_last_and_tracks_source_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)
    first_source = _write_raw_result(workspace, "first_result")
    _write_raw_result(workspace, "second_result", reverse=True)

    first = compile_allocated_cc_result(workspace, "first_result")
    second = compile_allocated_cc_result(workspace, "second_result")
    assert first.status == "computed"
    assert second.status == "computed"
    assert first.allocated_cc_result_identity != second.allocated_cc_result_identity
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["schema_version"] == "pytoce_allocated_cc_product_schema_v1"
    assert manifest["allocated_cc_cases"] == ["direct_case"]
    assert manifest["lcia_method"] == ["ef31"]
    assert manifest["lcia_method_identity"].startswith("pytoce_optimization_lcia_selection_v1::")
    assert manifest["identity_payload"]["allocated_cc_result"] == "first_result"
    assert manifest["identity_payload"]["source_files"] == manifest["source_files"]
    assert set(manifest["artifacts"]) == {
        "annual_allocated_cc.parquet",
        "cumulative_gwp100_allocated_cc.parquet",
        "lcia_method_registry.parquet",
        "cumulative_climate_method_registry.parquet",
    }
    assert manifest["source_files"] == {
        "allocated_cc_result.yaml": manifest["source_files"]["allocated_cc_result.yaml"],
        "annual_allocated_cc.csv": manifest["source_files"]["annual_allocated_cc.csv"],
        "cumulative_gwp100_allocated_cc.csv": manifest["source_files"][
            "cumulative_gwp100_allocated_cc.csv"
        ],
    }
    assert all(len(value) == 64 for value in manifest["source_files"].values())
    for filename, descriptor in manifest["artifacts"].items():
        assert descriptor["path"] == filename
        assert descriptor["rows"] > 0
        assert len(descriptor["artifact_sha256"]) == 64
        assert len(descriptor["scientific_identity"]) == 64
        assert descriptor["columns"]
    assert manifest["method_authority_parent"]["annual_lcia_registry_identity"].startswith(
        "pytoce_optimization_lcia_registry_v1::"
    )
    assert len(pd.read_parquet(first.result_directory / "annual_allocated_cc.parquet")) == (
        12 * 16 * 82
    )

    (first_source / "annual_allocated_cc.csv").write_text("invalid\n", encoding="utf-8")
    reused = compile_allocated_cc_result(workspace, "first_result")
    assert reused.status == "reused"
    assert reused.allocated_cc_result_identity == first.allocated_cc_result_identity

    _write_raw_result(workspace, "null_named_case", cases=("null",))
    null_named = compile_allocated_cc_result(workspace, "null_named_case")
    assert null_named.allocated_cc_cases == ("null",)


def test_completed_allocated_cc_result_is_reused_while_another_result_is_written(
    tmp_path: Path,
) -> None:
    """Reuse the immutable allocated carrying capacity result."""

    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)
    _write_raw_result(workspace, "completed_result")
    completed = compile_allocated_cc_result(workspace, "completed_result")
    lock_path = workspace / "data_processed" / "allocated_cc" / ".writer.lock"

    with writer_lock(lock_path):
        reused = compile_allocated_cc_result(workspace, "completed_result")

    assert reused.status == "reused"
    assert reused.allocated_cc_result_identity == completed.allocated_cc_result_identity


def test_allocated_cc_compilation_reports_conflicting_processed_results(tmp_path: Path) -> None:
    """Preserve an active writer and a visible incomplete result directory."""

    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)
    _write_raw_result(workspace, "waiting_result")
    processed = workspace / "data_processed" / "allocated_cc"
    lock_path = processed / ".writer.lock"
    processed.mkdir(parents=True, exist_ok=True)

    with writer_lock(lock_path):
        with pytest.raises(AllocatedCCError, match="Another package process"):
            compile_allocated_cc_result(workspace, "waiting_result")

    incomplete = processed / "incomplete_result"
    incomplete.mkdir()
    with pytest.raises(AllocatedCCError, match=r"directory is incomplete: .*incomplete_result"):
        compile_allocated_cc_result(workspace, "incomplete_result")


def test_allocated_cc_compilation_requires_shared_method_authority(tmp_path: Path) -> None:
    """Require prepared methods before publishing an otherwise valid external aCC bundle."""
    workspace = tmp_path / "workspace"
    _write_raw_result(workspace, "waiting_result")
    shared_manifest = workspace / "data_processed/prepared_backgrounds/shared/manifest.json"

    with pytest.raises(AllocatedCCError) as error:
        compile_allocated_cc_result(workspace, "waiting_result")

    assert str(error.value) == (
        f"Prepared manifest {shared_manifest} is unavailable. Run prepare_backgrounds first."
    )
    assert not (workspace / "data_processed/allocated_cc/waiting_result").exists()


def test_ef31_bundle_selects_cases_without_changing_parent(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)
    _write_raw_result(workspace, "ef_result")
    result = compile_allocated_cc_result(workspace, "ef_result")
    selected = select_allocated_cc_result(result, select_allocated_cc_cases(result, None))

    assert result.annual_impact_category_count == 16
    assert selected.annual_allocated_cc.shape == (1, 12, 82, 16)
    assert not selected.annual_allocated_cc.flags.writeable
    with pytest.raises(ValueError):
        selected.annual_allocated_cc.setflags(write=True)
    assert selected.optimization_lcia_method == ("ef31",)
    assert selected.allocated_cc_result_identity == result.allocated_cc_result_identity
    assert selected.cumulative_allocated_cc.shape == (1, 12)

    _write_raw_result(
        workspace,
        "multiple_cases",
        cases=("allocation_a", "allocation_b"),
    )
    multiple = compile_allocated_cc_result(workspace, "multiple_cases")
    cases = select_allocated_cc_cases(multiple, None)
    all_cases = select_allocated_cc_result(multiple, cases)
    repeated_all_cases = select_allocated_cc_result(multiple, cases)
    assert all_cases.allocated_cc_cases == ("allocation_a", "allocation_b")
    assert all_cases.allocated_cc_selection_identity == (
        repeated_all_cases.allocated_cc_selection_identity
    )
    request = normalize_run_optimization_request(
        foreground_scenario="study_parent",
        allocated_cc_result="multiple_cases",
        allocated_cc_case="allocation_a",
    )
    allocation_a = select_allocated_cc_result(
        multiple, select_allocated_cc_cases(multiple, request.allocated_cc_cases)
    )
    allocation_b = select_allocated_cc_result(
        multiple,
        select_allocated_cc_cases(multiple, ("allocation_b",)),
    )
    assert allocation_a.allocated_cc_cases == ("allocation_a",)
    assert allocation_a.annual_allocated_cc.shape == (1, 12, 82, 16)
    assert (
        allocation_a.allocated_cc_selection_identity != allocation_b.allocated_cc_selection_identity
    )


def test_acc_selection_binds_case_order_tables_and_climate_parent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)
    _write_raw_result(workspace, "multiple_cases", cases=("allocation_a", "allocation_b"))
    result = compile_allocated_cc_result(workspace, "multiple_cases")
    reversed_request = select_allocated_cc_result(
        result,
        select_allocated_cc_cases(result, ("allocation_b", "allocation_a")),
    )
    assert reversed_request.allocated_cc_cases == ("allocation_a", "allocation_b")
    assert reversed_request.annual_allocated_cc[1, 11, 81, 0] == 1968.0
    assert reversed_request.cumulative_allocated_cc[1, 11] == 24000.0
    assert not reversed_request.annual_allocated_cc.flags.writeable


def test_allocated_cc_templates_are_excluded_and_parser_fails_closed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    template = workspace / "data_raw/allocated_cc/templates"
    template.mkdir(parents=True)
    result_directory = workspace / "data_raw/allocated_cc/results/valid_result"
    result_directory.mkdir(parents=True)

    _write_shared_authority(workspace)
    annual, cumulative = _source_frames()
    annual.to_csv(result_directory / "annual_allocated_cc.csv", index=False)
    cumulative.to_csv(result_directory / "cumulative_gwp100_allocated_cc.csv", index=False)
    (result_directory / "allocated_cc_result.yaml").write_text(
        "\n".join(
            (
                "schema_version: pytoce_allocated_cc_result_schema_v1",
                "allocated_cc_result: valid_result",
                "allocated_cc_result: duplicated_name",
                "description: invalid duplicate key",
                "ssp_scenario: SSP2",
                "annual_allocated_cc: annual_allocated_cc.csv",
                "cumulative_gwp100_allocated_cc: cumulative_gwp100_allocated_cc.csv",
                "",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(AllocatedCCError, match="Cannot read allocated carrying capacity metadata"):
        compile_allocated_cc_result(workspace, "valid_result")

    template_root = resources.files("pytoce.run_optimization.data").joinpath(
        "workspace/data_raw/allocated_cc/templates"
    )
    assert (
        template_root.joinpath("annual_allocated_cc.csv").read_text(encoding="utf-8").count("\n")
        == 1
    )


def _write_allocated_cc_input(source: Path, name: str, **replacements: str) -> None:
    values = {
        "allocated_cc_result": name,
        "annual_allocated_cc": "annual_allocated_cc.csv",
        "cumulative_gwp100_allocated_cc": "cumulative_gwp100_allocated_cc.csv",
    }
    values.update(replacements)
    (source / "allocated_cc_result.yaml").write_text(
        "\n".join(
            (
                "schema_version: pytoce_allocated_cc_result_schema_v1",
                f"allocated_cc_result: {values['allocated_cc_result']}",
                "description: Direct external aCC fixture",
                "ssp_scenario: SSP2",
                "lcia_method:",
                "  - ef31",
                f"annual_allocated_cc: {values['annual_allocated_cc']}",
                f"cumulative_gwp100_allocated_cc: {values['cumulative_gwp100_allocated_cc']}",
                "",
            )
        ),
        encoding="utf-8",
    )


def test_allocated_cc_bundle_paths_names_and_headers_fail_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)
    source = _write_raw_result(workspace, "valid_result")

    _write_allocated_cc_input(source, "different_result")
    with pytest.raises(AllocatedCCError, match="must match its directory name"):
        compile_allocated_cc_result(workspace, "valid_result")

    _write_allocated_cc_input(source, "valid_result", annual_allocated_cc="../outside.csv")
    (source.parent / "outside.csv").write_text("outside\n", encoding="utf-8")
    with pytest.raises(AllocatedCCError, match="regular file inside"):
        compile_allocated_cc_result(workspace, "valid_result")

    _write_allocated_cc_input(
        source,
        "valid_result",
        annual_allocated_cc=(tmp_path / "absolute.csv").as_posix(),
    )
    with pytest.raises(AllocatedCCError, match="must be a relative path"):
        compile_allocated_cc_result(workspace, "valid_result")

    _write_allocated_cc_input(source, "valid_result")
    (source / "annual_allocated_cc.csv").write_text("wrong_header\n", encoding="utf-8")
    with pytest.raises(AllocatedCCError, match="columns do not match the v1 schema"):
        compile_allocated_cc_result(workspace, "valid_result")

    with pytest.raises(AllocatedCCError, match="lowercase snake_case"):
        compile_allocated_cc_result(workspace, "../invalid")


def _directory_link_or_skip(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError as exc:
        symlink_error = exc
    if os.name == "nt":
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
    pytest.skip(f"This platform cannot create a directory link: {symlink_error}")


def _file_symlink_or_skip(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=False)
    except OSError as exc:
        pytest.skip(f"This platform cannot create the required file symlink: {exc}")


def test_allocated_cc_source_directory_links_cannot_escape_results(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)
    external = tmp_path / "external_result"
    external.mkdir()
    annual, cumulative = _source_frames()
    annual.to_csv(external / "annual_allocated_cc.csv", index=False)
    cumulative.to_csv(external / "cumulative_gwp100_allocated_cc.csv", index=False)
    _write_allocated_cc_input(external, "linked_result")
    results = workspace / "data_raw/allocated_cc/results"
    results.mkdir(parents=True)
    _directory_link_or_skip(external, results / "linked_result")
    with pytest.raises(AllocatedCCError, match="regular directory directly inside"):
        compile_allocated_cc_result(workspace, "linked_result")


def test_allocated_cc_compile_rejects_an_aliased_results_root(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)
    external = tmp_path / "external_results"
    external.mkdir()
    raw_owner = workspace / "data_raw/allocated_cc"
    raw_owner.mkdir(parents=True)
    _directory_link_or_skip(external, raw_owner / "results")
    with pytest.raises(AllocatedCCError, match="not a link or junction"):
        compile_allocated_cc_result(workspace, "valid_result")


def test_allocated_cc_bundle_symlink_cannot_escape_the_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)

    source = _write_raw_result(workspace, "bundle_link_result")
    metadata = source / "allocated_cc_result.yaml"
    external_metadata = tmp_path / "external_metadata.yaml"
    external_metadata.write_bytes(metadata.read_bytes())
    metadata.unlink()
    _file_symlink_or_skip(external_metadata, metadata)
    with pytest.raises(AllocatedCCError, match="regular file inside"):
        compile_allocated_cc_result(workspace, "bundle_link_result")


def test_allocated_cc_nested_source_files_and_lcia_schema_runtime_grammar(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)
    source = _write_raw_result(workspace, "nested_result")
    tables = source / "tables"
    tables.mkdir()
    (source / "annual_allocated_cc.csv").replace(tables / "annual.csv")
    (source / "cumulative_gwp100_allocated_cc.csv").replace(tables / "cumulative.csv")
    _write_allocated_cc_input(
        source,
        "nested_result",
        annual_allocated_cc="tables/annual.csv",
        cumulative_gwp100_allocated_cc="tables/cumulative.csv",
    )
    assert compile_allocated_cc_result(workspace, "nested_result").status == "computed"

    schema = json.loads(
        resources.files("pytoce.run_optimization.data")
        .joinpath("schemas/allocated_cc_result_schema_v1.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    base = {
        "schema_version": "pytoce_allocated_cc_result_schema_v1",
        "allocated_cc_result": "schema_result",
        "description": "Schema and runtime parity fixture",
        "ssp_scenario": "SSP2",
        "annual_allocated_cc": "annual.csv",
        "cumulative_gwp100_allocated_cc": "cumulative.csv",
    }
    validator.validate({**base, "lcia_method": ["ef31"]})
    for selection in ("ef31", "pb_lcia", ["pb_lcia"], ["ef31", "pb_lcia"], []):
        assert tuple(validator.iter_errors({**base, "lcia_method": selection}))

    product_schema = json.loads(
        resources.files("pytoce.run_optimization.data")
        .joinpath("schemas/allocated_cc_product_schema_v1.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(product_schema)
    product = compile_allocated_cc_result(workspace, "nested_result")
    manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))
    product_validator = Draft202012Validator(product_schema)
    product_validator.validate(manifest)
    obsolete_manifest = {**manifest, "declared_lcia_coverage": ["ef31"]}
    assert tuple(product_validator.iter_errors(obsolete_manifest))
    for field in ("identity_payload", "method_authority_parent", "source_files", "artifacts"):
        invalid = {**manifest, field: {}}
        assert tuple(product_validator.iter_errors(invalid))


def test_allocated_cc_bundle_rejects_unknown_keys_and_templates_are_header_only(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_shared_authority(workspace)
    source = _write_raw_result(workspace, "valid_result")
    with (source / "allocated_cc_result.yaml").open("a", encoding="utf-8") as stream:
        stream.write("allocated_share: 0.1\n")
    with pytest.raises(AllocatedCCError, match="Unknown allocated_cc_result.yaml field"):
        compile_allocated_cc_result(workspace, "valid_result")

    template_root = resources.files("pytoce.run_optimization.data").joinpath(
        "workspace/data_raw/allocated_cc/templates"
    )
    annual_header = template_root.joinpath("annual_allocated_cc.csv").read_text(encoding="utf-8")
    cumulative_header = template_root.joinpath("cumulative_gwp100_allocated_cc.csv").read_text(
        encoding="utf-8"
    )
    assert annual_header.splitlines() == [
        ",".join(
            (
                "allocated_cc_case",
                "lcia_method",
                "region",
                "impact_category_id",
                "year",
                "allocated_cc",
                "unit",
                "cc_authority",
            )
        )
    ]
    assert cumulative_header.splitlines() == [
        ",".join(
            (
                "allocated_cc_case",
                "region",
                "impact_category_id",
                "period_start",
                "period_end",
                "cumulative_allocated_cc",
                "unit",
                "budget_metric",
                "cc_authority",
            )
        )
    ]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_text", "is missing at row"),
        ("empty_text", "is empty at row"),
        ("nonnumeric", "must contain numbers"),
        ("method", "invalid lcia_method"),
        ("cumulative_columns", "columns do not match the v1 schema"),
    ],
)
def test_external_tables_report_the_first_invalid_cell_or_schema(
    case: str,
    message: str,
) -> None:
    """Give corrective context for malformed text, numbers, method labels and CSV columns."""

    annual, cumulative = _source_frames()
    if case == "missing_text":
        annual.loc[0, "cc_authority"] = None
    elif case == "empty_text":
        annual.loc[0, "cc_authority"] = ""
    elif case == "nonnumeric":
        annual["allocated_cc"] = annual["allocated_cc"].astype(object)
        annual.loc[0, "allocated_cc"] = "not-a-number"
    elif case == "method":
        annual.loc[0, "lcia_method"] = "not-a-method"
    else:
        cumulative = cumulative.drop(columns="budget_metric")

    with pytest.raises(AllocatedCCError, match=message):
        validate_allocated_cc_frames(annual, cumulative, _authority())
