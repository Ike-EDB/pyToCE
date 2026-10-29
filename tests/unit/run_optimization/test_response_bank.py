import json
import os
import subprocess
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from response_bank_fixtures import (
    _ANCHORS,
    _BACKGROUNDS,
    _direct_factors,
    _methods,
    _native_responses,
    _services,
    _write_bank,
)

from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    PreparedDataError,
)
from pytoce.prepare_backgrounds.preparation.project_biosphere import load_lcia_catalogue
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_optimization.calculation.response_bank import (
    project_optimization_response_bank,
    read_optimization_response_bank,
    select_response_bank_method_registry,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    _NATIVE_COLUMNS,
    ResponseBankError,
)
from pytoce.run_optimization.calculation.response_bank_store import (
    _owner_lock,
    load_optimization_response_bank,
)
from pytoce.run_optimization.calculation.response_builder import (
    ensure_optimization_response_bank,
    response_method_selection,
)
from pytoce.run_optimization.calculation.response_kernel import (
    ResponseMethodSelection,
)


def test_response_bank_writer_lock_file_is_reusable_and_exclusive(
    tmp_path: Path,
) -> None:
    owner = tmp_path / "response_bank"
    owner.mkdir()
    lock_path = owner / ".response_bank.writer.lock"
    lock_path.write_bytes(b"\0")

    with _owner_lock(owner):
        with pytest.raises(ResponseBankError, match="Another package process"):
            with _owner_lock(owner):
                pass

    with _owner_lock(owner):
        assert lock_path.is_file()


def test_response_bank_preserves_a_missing_workspace(tmp_path: Path) -> None:
    """Require the selected workspace to exist during response publication."""
    workspace = tmp_path / "missing"
    with pytest.raises(FileNotFoundError) as caught:
        ensure_optimization_response_bank(
            workspace,
            selection=response_method_selection(normalized_lcia_method("ef31")),
            member_progress=[].append,
        )
    assert Path(caught.value.filename) == workspace
    assert not workspace.exists()


def test_response_bank_writes_manifest_last_and_reuses(tmp_path: Path) -> None:
    assert load_optimization_response_bank(tmp_path, "ef31_pb_lcia") is None
    assert not (tmp_path / "data_processed").exists()
    result = _write_bank(tmp_path)

    assert result.status == "computed"
    assert result.identity.startswith("pytoce_optimization_response_bank_schema_v1::")
    assert sorted(path.name for path in result.result_directory.iterdir()) == [
        "direct_characterization_factors.parquet",
        "manifest.json",
        "native_unit_responses.parquet",
        "provider_unit_responses.parquet",
        "timex_year_weights.parquet",
    ]
    manifest = json.loads((result.result_directory / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads(
        files("pytoce.run_optimization.data.schemas")
        .joinpath("optimization_response_bank_schema_v1.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(manifest)
    assert manifest["identity_payload"]["background_scenarios"] == list(_BACKGROUNDS)
    assert manifest["identity_payload"]["occurrence_years"] == [2019, 2100]

    reused = load_optimization_response_bank(tmp_path, "ef31_pb_lcia")
    assert reused is not None
    assert reused.status == "reused"
    assert reused.identity == result.identity
    assert not list(result.result_directory.parent.glob(".ef31_pb_lcia.*.response_bank.*"))


@pytest.mark.parametrize(
    "selection",
    [
        response_method_selection(normalized_lcia_method(("ef31", "pb_lcia"))),
        response_method_selection(normalized_lcia_method(("ef31",))),
    ],
)
def test_completed_response_bank_is_reused_without_taking_the_writer_lock(
    tmp_path: Path,
    selection: ResponseMethodSelection,
) -> None:
    written = _write_bank(tmp_path, response_method_union="ef31_pb_lcia_response_v3")
    owner = written.result_directory.parent

    with _owner_lock(owner):
        reused = ensure_optimization_response_bank(
            tmp_path,
            selection=selection,
            member_progress=[].append,
        )

    assert reused.status == "reused"
    assert reused.identity == written.identity


@pytest.mark.parametrize(
    "selection",
    [
        response_method_selection(normalized_lcia_method(("ef31", "pb_lcia"))),
        response_method_selection(normalized_lcia_method(("ef31",))),
    ],
)
def test_response_bank_requires_prepared_backgrounds(
    tmp_path: Path,
    selection: ResponseMethodSelection,
) -> None:
    with pytest.raises(PreparedDataError, match="prepare_backgrounds"):
        ensure_optimization_response_bank(
            tmp_path,
            selection=selection,
            member_progress=[].append,
        )


def test_exact_pb_subset_uses_a_bounded_reusable_name(tmp_path: Path) -> None:
    exact = (
        "AESA (PBs-LCIA)",
        "0.9.2",
        "climate change",
        "atmospheric CO2 concentration",
    )
    selection = response_method_selection(normalized_lcia_method(exact))

    assert len(selection.readable_name) < 255
    written = _write_bank(tmp_path, response_method_union=selection.readable_name)
    reused = load_optimization_response_bank(tmp_path, selection.readable_name)

    assert written.status == "computed"
    assert reused is not None
    assert reused.status == "reused"
    assert reused.identity == written.identity


def test_response_bank_projections_are_independently_identity_bound(tmp_path: Path) -> None:
    bank = _write_bank(tmp_path)
    loaded = read_optimization_response_bank(bank)
    methods = _methods()

    optimization = project_optimization_response_bank(
        loaded,
        select_response_bank_method_registry(methods, normalized_lcia_method("ef31")),
        role="optimization",
    )
    result = project_optimization_response_bank(
        loaded,
        select_response_bank_method_registry(methods, normalized_lcia_method("pb_lcia")),
        role="result",
    )
    climate = project_optimization_response_bank(
        loaded,
        methods.iloc[[0]].reset_index(drop=True),
        role="mandatory_climate",
    )

    expected_rows = len(_BACKGROUNDS) * len(_ANCHORS) * len(_services())
    assert len(optimization.provider_unit_responses) == expected_rows
    assert len(result.provider_unit_responses) == expected_rows
    assert len(climate.provider_unit_responses) == expected_rows
    assert set(optimization.provider_unit_responses["method_id"]) == {"ef31"}
    assert set(result.provider_unit_responses["method_id"]) == {"pb_lcia"}
    assert optimization.projection_identity != result.projection_identity
    assert optimization.projection_identity != climate.projection_identity
    assert optimization.response_bank_identity == result.response_bank_identity == bank.identity
    assert optimization.provider_route_identity == "provider_routes::fixture"
    assert optimization.provider_service_registry.equals(_services().astype("string"))
    assert optimization.provider_service_registry_identity
    assert optimization.provider_unit_responses_identity
    assert optimization.timex_year_weights_identity == result.timex_year_weights_identity
    assert len(optimization.timex_year_weights) == len(_BACKGROUNDS) * 82
    pd.testing.assert_frame_equal(
        loaded.native_unit_responses, _native_responses(loaded.method_registry)
    )
    pd.testing.assert_frame_equal(
        loaded.direct_characterization_factors, _direct_factors(loaded.method_registry)
    )
    assert set(optimization.native_unit_responses["method_id"]) == {"ef31"}
    assert set(result.direct_characterization_factors["method_id"]) == {"pb_lcia"}
    direct = result.direct_characterization_factors.set_index("flow_code")
    assert bool(direct.loc["zero", "present"])
    assert not bool(direct.loc["absent", "present"])
    assert direct.loc[["zero", "absent"], "factor"].eq(0.0).all()
    assert optimization.native_unit_responses_identity != result.native_unit_responses_identity
    assert (
        optimization.direct_characterization_factors_identity
        != result.direct_characterization_factors_identity
    )
    changed_native = project_optimization_response_bank(
        replace(loaded, native_unit_responses=loaded.native_unit_responses.assign(unit_score=7.0)),
        optimization.method_registry,
        role="optimization",
    )
    changed_direct = project_optimization_response_bank(
        replace(
            loaded,
            direct_characterization_factors=loaded.direct_characterization_factors.assign(
                factor=4.0
            ),
        ),
        optimization.method_registry,
        role="optimization",
    )
    assert changed_native.projection_identity != optimization.projection_identity
    assert changed_direct.projection_identity != optimization.projection_identity
    assert (
        changed_native.provider_unit_responses_identity
        == optimization.provider_unit_responses_identity
    )
    assert changed_direct.timex_year_weights_identity == optimization.timex_year_weights_identity


def test_exact_method_union_retains_independent_within_selection_order(tmp_path: Path) -> None:
    climate = _methods().iloc[[0]].reset_index(drop=True)
    exact = climate.copy()
    specifications, _ = load_lcia_catalogue()
    spec = next(method for method in specifications if method.method_id == "pb_lcia")
    exact_method = json.dumps(spec.impact_category_tuple, separators=(",", ":"))
    exact["lcia_method"] = exact_method
    exact["method_id"] = spec.method_id
    exact["method_version"] = spec.method_version
    exact["impact_category_id"] = spec.impact_category_id
    exact["impact_category_tuple_json"] = exact_method
    exact["unit"] = spec.unit
    exact["factor_identity"] = "9" * 64
    methods = pd.concat([climate, exact], ignore_index=True)

    bank = _write_bank(
        tmp_path,
        methods=methods,
        response_method_union="ef31_exact_pb",
    )
    selected = select_response_bank_method_registry(
        methods, normalized_lcia_method(spec.impact_category_tuple)
    )
    projection = project_optimization_response_bank(
        read_optimization_response_bank(bank), selected, role="result"
    )

    assert set(projection.provider_unit_responses["method_id"]) == {spec.method_id}
    assert set(projection.method_registry["method_version"]) == {spec.method_version}


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


@pytest.mark.parametrize("linked_component", ["data_processed", "owner", "result"])
def test_writing_rejects_directory_links(
    tmp_path: Path,
    linked_component: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / f"external_{linked_component}"
    external.mkdir()
    if linked_component == "data_processed":
        _directory_link_or_skip(external, workspace / "data_processed")
    else:
        data_processed = workspace / "data_processed"
        data_processed.mkdir()
        if linked_component == "owner":
            _directory_link_or_skip(external, data_processed / "optimization_response_bank")
        else:
            owner = data_processed / "optimization_response_bank" / "four_remind"
            owner.mkdir(parents=True)
            _directory_link_or_skip(external, owner / "ef31_pb_lcia")

    with pytest.raises(ResponseBankError, match="symlink or junction|path already exists"):
        _write_bank(workspace)


def test_writing_never_overwrites_an_existing_result_directory(tmp_path: Path) -> None:
    claimed = tmp_path / "claimed"
    destination = (
        claimed / "data_processed" / "optimization_response_bank" / "four_remind" / "ef31_pb_lcia"
    )
    destination.mkdir(parents=True)
    marker = destination / "external_owner.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ResponseBankError, match="response bank path already exists"):
        _write_bank(claimed)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_response_bank_preserves_an_incomplete_visible_product(tmp_path: Path) -> None:
    """Report an interrupted response publication and preserve its files."""
    directory = tmp_path / "data_processed/optimization_response_bank/four_remind/ef31_pb_lcia"
    directory.mkdir(parents=True)
    marker = directory / "provider_unit_responses.parquet"
    marker.write_bytes(b"incomplete fixture")
    with pytest.raises(ResponseBankError, match="no completion manifest"):
        load_optimization_response_bank(tmp_path, "ef31_pb_lcia")
    assert marker.read_bytes() == b"incomplete fixture"


def test_selected_backgrounds_read_only_their_rows_from_the_complete_bank(tmp_path: Path) -> None:
    """Reuse one immutable four-member bank while identity-binding the selected axes."""
    bank = _write_bank(tmp_path)
    complete = read_optimization_response_bank(bank)
    backgrounds = ("remind_ssp2_npi", "remind_ssp2_pkbudg650")
    selected = read_optimization_response_bank(bank, background_scenarios=backgrounds)
    assert selected.identity == complete.identity == bank.identity
    for name in ("provider_unit_responses", "native_unit_responses", "timex_year_weights"):
        expected = getattr(complete, name)
        expected = expected.loc[expected.background_scenario.isin(backgrounds)].reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(getattr(selected, name), expected, check_exact=True)
    pd.testing.assert_frame_equal(
        selected.direct_characterization_factors, complete.direct_characterization_factors
    )
    full_projection = project_optimization_response_bank(complete, _methods(), role="result")
    subset_projection = project_optimization_response_bank(selected, _methods(), role="result")
    assert full_projection.projection_identity != subset_projection.projection_identity


def test_empty_native_response_table_can_be_read_with_background_filter(tmp_path: Path) -> None:
    """Read an empty native-response table when selecting background scenarios."""
    empty_native_responses = pd.DataFrame(columns=_NATIVE_COLUMNS)
    bank = _write_bank(tmp_path, native_unit_responses=empty_native_responses)

    selected = read_optimization_response_bank(
        bank,
        background_scenarios=("remind_ssp2_npi", "remind_ssp2_pkbudg650"),
    )

    assert selected.native_unit_responses.empty
    assert tuple(selected.native_unit_responses.columns) == _NATIVE_COLUMNS
