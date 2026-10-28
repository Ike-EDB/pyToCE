"""Public report and member-product failure contracts."""

from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
from material_recipe_fixtures import physical_material_recipe
from pa_receiving_fixtures import pa_receiving_reference

from pytoce import prepare_backgrounds, set_workspace
from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.premise import (
    BACKGROUND_BY_ID,
    PremiseIAMInput,
)
from pytoce.prepare_backgrounds.preparation.prepared_state import (
    member_preparation as owner,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.shared_preparation import (
    SharedAuthority,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.source_authority import (
    Sources,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.prepare_backgrounds import (
    PrepareBackgroundsReport,
    _aggregate_status,
)
from pytoce.shared.runtime.file_lock import FileLockError, writer_lock


def _result(
    tmp_path: Path,
    *,
    status: owner.PreparedBackgroundStatus,
) -> owner.PreparedBackgroundResult:
    """Build one typed public result with readable paths."""
    root = tmp_path / f"background-{status}"
    return owner.PreparedBackgroundResult(
        status=status,
        background_scenario=f"scenario-{status}",
        ssp_scenario="SSP2",
        result_directory=root,
        manifest_path=root / "manifest.json",
    )


def _sources(tmp_path: Path) -> Sources:
    """Build source records with complete identities."""
    premise_authority = {
        "distribution": "premise",
        "version": "2.4.6",
        "identity": "premise::bounded",
    }
    return Sources(
        ecoinvent_archive=tmp_path / "ecoinvent.7z",
        ef_workbook=tmp_path / "ef.xlsx",
        pb_archive=tmp_path / "pb.tar.gz",
        source_identity="source::" + "d" * 64,
        source_identity_payload={"schema_version": "source_schema_v1"},
        dependency_authorities=(premise_authority,),
        premise_authority=premise_authority,
        premise_authority_identity="premise::bounded",
        material_recipe=physical_material_recipe(),
        provider=prepared_provider_authority(),
        preparation_method={
            "schema_version": "preparation_method_schema_v1",
            "identity": "preparation::bounded",
        },
    )


def _shared(tmp_path: Path) -> SharedAuthority:
    """Build the typed shared identity needed before member preparation."""
    return SharedAuthority(
        pa_receiving_reference=pa_receiving_reference("ecoinvent"),
        names=BrightwayNames(
            store=tmp_path / "brightway",
            project="pytoce_member_failure",
            biosphere_database="biosphere",
            source_database="ecoinvent",
        ),
        identity="shared::" + "1" * 64,
        source_identity="source::" + "d" * 64,
        preparation_method_identity="preparation::bounded",
        lcia_identity="lcia::" + "2" * 64,
        provider_identity="provider::" + "3" * 64,
        provider_interface_database="pytoce__provider_interface",
        provider_timeline_identity="timeline::" + "4" * 64,
        provider_timeline_database_by_year={},
        provider_routes=(),
        impact_categories=(),
        database_rows=(),
        matrix_rows=(),
        biosphere_rows=(),
    )


def _iam_input(tmp_path: Path) -> PremiseIAMInput:
    """Build the recorded IAM identity consumed by the member owner."""
    scenario = BACKGROUND_BY_ID["remind_ssp2_npi"]
    payload = tmp_path / scenario.iam_filename
    return PremiseIAMInput(
        identity="iam::bounded",
        payload=payload,
        manifest_sha256=sha256(b"manifest").hexdigest(),
        payload_sha256=sha256(b"payload").hexdigest(),
        premise_authority_identity="premise::bounded",
    )


def test_prepare_report_serializes_nested_results_and_all_status_combinations(
    tmp_path: Path,
) -> None:
    """Keep computed and reused item statuses and updated for a mixed public request."""
    computed = _result(tmp_path, status="computed")
    reused = _result(tmp_path, status="reused")

    assert _aggregate_status((computed,)) == "computed"
    assert _aggregate_status((reused,)) == "reused"
    assert _aggregate_status((computed, reused)) == "updated"

    report = PrepareBackgroundsReport(
        status="updated",
        workspace=tmp_path,
        elapsed_seconds=1.25,
        results=(computed, reused),
    )
    assert report.as_dict() == {
        "status": "updated",
        "workspace": str(tmp_path),
        "elapsed_seconds": 1.25,
        "results": [
            {
                "status": "computed",
                "background_scenario": "scenario-computed",
                "ssp_scenario": "SSP2",
                "result_directory": str(tmp_path / "background-computed"),
                "manifest_path": str(tmp_path / "background-computed" / "manifest.json"),
            },
            {
                "status": "reused",
                "background_scenario": "scenario-reused",
                "ssp_scenario": "SSP2",
                "result_directory": str(tmp_path / "background-reused"),
                "manifest_path": str(tmp_path / "background-reused" / "manifest.json"),
            },
        ],
    }


def test_public_preparation_reports_progress_before_missing_licensed_sources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Give actionable progress and source custody errors before any preparation."""
    set_workspace(tmp_path)
    capsys.readouterr()

    with pytest.raises(FileNotFoundError, match="required ecoinvent 3.12 files"):
        prepare_backgrounds(background_scenario="remind_ssp2_npi")

    assert "preparing ecoinvent and LCIA data" in capsys.readouterr().err


def test_premise_report_flattening_rejects_ambiguous_dependency_outputs(
    tmp_path: Path,
) -> None:
    """Require one report of each kind and preserve ambiguous dependency output."""
    report_root = tmp_path / "premise-reports"
    scenario_root = report_root / "export" / "scenario_report"
    change_root = report_root / "export" / "change reports"
    scenario_root.mkdir(parents=True)
    change_root.mkdir(parents=True)
    (scenario_root / "first.xlsx").write_bytes(b"first")
    (scenario_root / "second.xlsx").write_bytes(b"second")
    (change_root / "only.xlsx").write_bytes(b"change")

    with pytest.raises(RuntimeError, match="must write one workbook"):
        owner.flatten_premise_reports(tmp_path / "staging", report_root, (2030, 2040))

    assert {path.name for path in scenario_root.iterdir()} == {"first.xlsx", "second.xlsx"}


def test_member_preserves_visible_directory_without_completion_manifest(
    tmp_path: Path,
) -> None:
    """Fail before Brightway or premise work when a readable member is ambiguous."""
    sources = _sources(tmp_path)
    shared = _shared(tmp_path)
    iam_input = _iam_input(tmp_path)
    scenario = BACKGROUND_BY_ID["remind_ssp2_npi"]
    prepared_target = owner.target(
        tmp_path,
        sources,
        shared,
        iam_input,
        scenario,
        years=(2030,),
    )
    prepared_target.result_directory.mkdir(parents=True)
    note = prepared_target.result_directory / "user-note.txt"
    note.write_text("preserve", encoding="utf-8")
    provider = PreparedProviderAuthority(
        identity=shared.provider_identity,
        interface_database=shared.provider_interface_database,
        catalogue={},
        routes=pd.DataFrame(),
    )

    with pytest.raises(RuntimeError, match="has no completion manifest"):
        owner.prepare_member(
            tmp_path,
            sources,
            shared,
            provider,
            iam_input,
            scenario,
            years=(2030,),
            premise_key=None,
            progress=lambda _scenario, _phase: None,
            provider_resolvers=_NativeResolverCache(
                shared.names.source_database,
                shared.names.biosphere_database,
            ),
        )

    assert note.read_text(encoding="utf-8") == "preserve"
    assert not (prepared_target.result_directory / "manifest.json").exists()


def test_member_writer_lock_is_shared_across_workspaces(tmp_path: Path) -> None:
    """Reject a second workspace writer before it changes the same Brightway project."""

    shared = _shared(tmp_path)
    shared.names.store.mkdir(parents=True)
    with writer_lock(shared.names.writer_lock_path):
        with pytest.raises(FileLockError, match="Another package process"):
            owner.prepare_member(
                tmp_path / "another_workspace",
                _sources(tmp_path),
                shared,
                prepared_provider_authority(),
                _iam_input(tmp_path),
                BACKGROUND_BY_ID["remind_ssp2_npi"],
                years=(2030,),
                premise_key=None,
                progress=lambda _scenario, _phase: None,
                provider_resolvers=_NativeResolverCache(
                    shared.names.source_database,
                    shared.names.biosphere_database,
                ),
            )
    assert not (tmp_path / "another_workspace").exists()
