"""Shared prepared-background persistence and reuse contracts."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from material_carbon_fixtures import prepared_material_carbon
from material_recipe_fixtures import physical_material_recipe
from pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    brightway_names,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.premise import (
    BACKGROUND_BY_ID,
    BackgroundScenario,
    PremiseIAMInput,
)
from pytoce.prepare_backgrounds.preparation.prepared_state import (
    batch_preparation,
    member_preparation,
)
from pytoce.prepare_backgrounds.preparation.prepared_state import (
    shared_preparation as owner,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    read_prepared_manifest,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.source_authority import Sources
from pytoce.prepare_backgrounds.preparation.project_biosphere import load_lcia_catalogue
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    prepared_provider_authority,
)
from pytoce.run_lca.calculation.providers import load_prepared_provider_product
from pytoce.shared.runtime.file_lock import FileLockError, writer_lock


def _sources(tmp_path: Path, *, identity: str = "sources::bounded") -> Sources:
    """Return path-neutral bounded source authority data for persistence tests."""
    return Sources(
        ecoinvent_archive=tmp_path / "ecoinvent.7z",
        ef_workbook=tmp_path / "ef.xlsx",
        pb_archive=tmp_path / "pb.tar.gz",
        source_identity=identity,
        source_identity_payload={"schema_version": "bounded_sources_schema_v1"},
        dependency_authorities=({"distribution": "premise", "version": "2.4.6"},),
        premise_authority={"distribution": "premise", "identity": "premise::2.4.6"},
        premise_authority_identity="premise::2.4.6",
        material_recipe=physical_material_recipe(),
        provider=prepared_provider_authority(),
        preparation_method={"identity": "preparation::bounded"},
    )


def _names(workspace: Path) -> BrightwayNames:
    """Return collision-free bounded Brightway names beneath a temporary workspace."""
    return brightway_names(
        workspace,
        source_identity="sources::bounded",
    )


def _provider() -> PreparedProviderAuthority:
    """Return one bounded provider authority with an explicit missing value."""
    return PreparedProviderAuthority(
        identity="provider::bounded",
        interface_database="provider_interface_bounded",
        catalogue={"schema_version": "bounded_provider_catalogue_schema_v1"},
        routes=pd.DataFrame(
            [
                {
                    "route_id": "route_bounded",
                    "share": 1.0,
                    "optional_note": float("nan"),
                }
            ]
        ),
        source_authorities=({"source_authority_id": "source_bounded"},),
        method_authorities=({"method_authority_id": "method_bounded"},),
    )


def _write_shared(
    workspace: Path,
    *,
    provider: PreparedProviderAuthority | None = None,
    names: BrightwayNames | None = None,
) -> owner.SharedAuthority:
    """Write one bounded shared product through its real Parquet/manifest boundary."""
    (workspace / "data_processed" / "prepared_backgrounds").mkdir(parents=True, exist_ok=True)
    names = _names(workspace) if names is None else names
    return owner.write_shared_data(
        workspace,
        _sources(workspace),
        names,
        pa_receiving_reference=pa_receiving_reference(names.source_database),
        provider=_provider() if provider is None else provider,
        material_carbon_rows=prepared_material_carbon(workspace.parent / "carbon_fixture"),
        impact_rows=(
            {
                "selection_order": 1,
                "method_id": "ef31",
                "factor_identity": "factor::bounded",
            },
        ),
        database_rows=(
            {
                "database_name": "biosphere_bounded",
                "database_role": "ecoinvent_biosphere",
            },
        ),
        matrix_rows=({"database_name": "biosphere_bounded"},),
        biosphere_rows=(
            {
                "authority_role": "ecoinvent_biosphere",
                "ecoinvent_biosphere_database": "biosphere_bounded",
            },
        ),
        import_report={"activity_count": 1},
    )


@pytest.mark.parametrize("external_store", [False, True])
def test_shared_product_round_trips_manifest_last_with_provider_data(
    tmp_path: Path,
    external_store: bool,
) -> None:
    """Persist exact authorities, normalize only missing cells, and load them unchanged."""
    workspace = tmp_path / "workspace"
    names = _names(workspace)
    if external_store:
        names = replace(names, store=tmp_path / "trusted_brightway")
    written = _write_shared(workspace, names=names)
    root = owner.shared_root(workspace)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["identity"] == written.identity
    assert manifest["brightway_store_relative_path"] == (
        "../trusted_brightway" if external_store else "data_processed/brightway"
    )
    assert (workspace / manifest["brightway_store_relative_path"]).resolve() == names.store
    assert manifest["artifacts"]["provider_service_routes.parquet"]["rows"] == 1
    assert not any(path.name.startswith(".shared_background.") for path in root.parent.iterdir())

    loaded = owner.load_shared(workspace, names=names)
    assert loaded == written
    assert loaded is not None
    assert loaded.provider_routes == (
        {"route_id": "route_bounded", "share": 1.0, "optional_note": None},
    )
    assert loaded.impact_categories[0]["factor_identity"] == "factor::bounded"
    assert loaded.provider_timeline_database_by_year[2020] == _provider().interface_database
    assert all(
        database.startswith(_provider().interface_database)
        for database in loaded.provider_timeline_database_by_year.values()
    )


def test_shared_preparation_respects_the_physical_project_writer_lock(tmp_path: Path) -> None:
    """Reject a second package writer before entering the selected Brightway project."""
    workspace = tmp_path / "workspace"
    names = _names(workspace)
    names.store.mkdir(parents=True)
    with writer_lock(names.writer_lock_path), pytest.raises(FileLockError):
        owner.ensure_shared_authority(workspace, _sources(workspace))
    assert not owner.shared_root(workspace).exists()


def test_shared_product_is_not_published_when_parent_guide_cannot_be_written(
    tmp_path: Path,
) -> None:
    """Reject an obstructed guide path and preserve the existing object."""
    workspace = tmp_path / "workspace"
    root = owner.shared_root(workspace)
    obstacle = root.parent / "README.md"
    obstacle.mkdir(parents=True)

    with pytest.raises(OSError):
        _write_shared(workspace)

    assert obstacle.is_dir()
    assert not root.exists()
    assert not any(path.name.startswith(".shared_background.") for path in root.parent.iterdir())


def test_shared_and_member_products_preserve_detached_pa_receiving_sources(tmp_path: Path) -> None:
    """Persist original calibration once and restore each dated operation."""
    workspace = tmp_path / "workspace"
    shared = _write_shared(workspace)
    scenario = BACKGROUND_BY_ID["remind_ssp2_npi"]
    iam = PremiseIAMInput(
        identity="iam::bounded",
        payload=workspace / "invented_iam.csv",
        manifest_sha256="d" * 64,
        payload_sha256="e" * 64,
        premise_authority_identity="premise::2.4.6",
    )
    target = member_preparation.target(
        workspace, _sources(workspace), shared, iam, scenario, years=(2050,)
    )
    # Deliberately provide anchors out of chronological order. Persisted source
    # coordinates and the reader's integer-keyed map must still agree exactly.
    sources_by_year = {
        year: pa_receiving_sources(
            year,
            background_scenario=scenario.background_scenario,
            source_database=shared.names.source_database,
        )
        for year in (2050, 2025, 2020)
    }
    tables = member_preparation.item_tables(
        shared,
        target,
        (
            {
                "database_name": "native_remind_ssp2_npi_2050",
                "database_role": "premise_anchor",
                "anchor_year": 2050,
            },
        ),
        {
            "provider_technology_crosswalk.parquet": tuple(
                {"anchor_year": year, "database_name": f"provider_{year}"}
                for year in sorted(sources_by_year)
            ),
            "provider_semantic_ledger.parquet": (),
            "provider_database_rows": (),
            "provider_matrix_rows": (),
        },
    )
    target.result_directory.mkdir()
    member_preparation.write_prepared_background(
        target,
        shared,
        target.result_directory,
        tables,
        pa_receiving_sources_by_year=sources_by_year,
        premise_report={"database_rows": []},
        report_artifacts=(),
    )
    shared_manifest = read_prepared_manifest(owner.shared_root(workspace) / "manifest.json")
    member_manifest = read_prepared_manifest(target.result_directory / "manifest.json")
    loaded = load_prepared_provider_product(shared_manifest, member_manifest)
    assert loaded.pa_receiving_reference == shared.pa_receiving_reference
    assert loaded.pa_receiving_sources_by_year == sources_by_year
    assert tuple(loaded.pa_receiving_sources_by_year) == (2020, 2025, 2050)
    assert loaded.database_by_year == {year: f"provider_{year}" for year in (2020, 2025, 2050)}
    reference_bytes = (shared_manifest.path.parent / "pa_receiving_reference.json").read_bytes()
    reference_identity = sha256(reference_bytes).hexdigest()
    shared_identity = cast(dict[str, object], shared_manifest.payload["identity_payload"])
    table_identities = cast(dict[str, str], shared_identity["table_identities"])
    assert table_identities["pa_receiving_reference.json"] == reference_identity
    dated_bytes = (member_manifest.path.parent / "pa_receiving_sources.json").read_bytes()
    member_artifacts = cast(dict[str, dict[str, object]], member_manifest.payload["artifacts"])
    assert (
        member_artifacts["pa_receiving_sources.json"]["scientific_identity"]
        == sha256(dated_bytes).hexdigest()
    )
    assert [record["provider_anchor_year"] for record in json.loads(dated_bytes)] == [
        2020,
        2025,
        2050,
    ]
    for year, source in loaded.pa_receiving_sources_by_year.items():
        thermal = source.thermal_source(loaded.pa_receiving_reference)
        assert thermal.donor_elements_kg == shared.pa_receiving_reference.donor_elements_kg
        assert thermal.waste_supplies[
            "hydrogen peroxide, without water, in 50% solution state"
        ] == (0.004 if year <= 2025 else 0.006)
        assert (
            source.ash_source(loaded.pa_receiving_reference).element_contents
            == shared.pa_receiving_reference.ash_element_contents
        )


def test_shared_reuse_rejects_a_completed_product_from_different_authorities(
    tmp_path: Path,
) -> None:
    """Reject a completed shared product whose provider authority changed."""
    workspace = tmp_path / "workspace"
    _write_shared(workspace)

    with pytest.raises(RuntimeError, match="different source files, package code, or provider"):
        owner.ensure_shared_authority(workspace, _sources(workspace))


def test_shared_reuse_returns_completed_matching_authority_without_brightway_work(
    tmp_path: Path,
) -> None:
    """Reuse a matching completed product."""
    workspace = tmp_path / "workspace"
    provider = prepared_provider_authority()
    written = _write_shared(workspace, provider=provider)

    reused = owner.ensure_shared_authority(workspace, _sources(workspace))

    assert reused.identity == written.identity
    assert reused.provider_identity == provider.identity
    assert reused.names == _names(workspace)


def test_incomplete_shared_directory_is_preserved_without_entering_brightway(
    tmp_path: Path,
) -> None:
    """Fail on ambiguous visible state before any licensed import or database write."""
    workspace = tmp_path / "workspace"
    owner.shared_root(workspace).mkdir(parents=True)

    with pytest.raises(RuntimeError, match="has no completion manifest.*left unchanged"):
        owner.ensure_shared_authority(workspace, _sources(workspace))


@pytest.mark.parametrize("collision", ["database", "method"])
def test_shared_preparation_rejects_brightway_name_collisions_before_source_import(
    tmp_path: Path,
    collision: str,
) -> None:
    """Preserve a project containing a package database or LCIA method name."""
    workspace = tmp_path / "workspace"
    sources = _sources(workspace)
    names = _names(workspace)
    with selected_brightway_project(names) as bd:
        if collision == "database":
            bd.Database(names.source_database).write(
                {
                    (names.source_database, "bounded"): {
                        "name": "bounded collision",
                        "reference product": "bounded collision",
                        "unit": "kilogram",
                        "location": "GLO",
                        "type": "process",
                    }
                },
                searchable=False,
            )
            expected = f"database '{names.source_database}'"
        else:
            spec = load_lcia_catalogue()[0][0]
            bd.Method(spec.impact_category_tuple).register(unit=spec.unit)
            expected = f"LCIA method '{spec.method_id}'"

    with pytest.raises(RuntimeError, match=expected):
        owner.ensure_shared_authority(workspace, sources)


def test_selected_batch_reuses_completed_iam_shared_and_member_products(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reuse all parents whose completion manifests were written last."""
    workspace = tmp_path / "workspace"
    provider = prepared_provider_authority()
    _write_shared(workspace, provider=provider)
    scenario = BackgroundScenario(
        background_scenario="synthetic_ssp2",
        premise_model="remind",
        iam_model="REMIND",
        iam_scenario="SSP2-NPi",
        gmst="bounded",
        iam_filename="remind_SSP2-NPi.csv",
        iam_file_size=7,
        iam_file_sha256="d" * 64,
    )
    iam_root = workspace / "data_raw" / "premise" / scenario.background_scenario
    iam_root.mkdir(parents=True)
    (iam_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    result_root = (
        workspace / "data_processed" / "prepared_backgrounds" / scenario.background_scenario
    )
    result_root.mkdir()
    (result_root / "manifest.json").write_text("{}\n", encoding="utf-8")

    results = batch_preparation.prepare_selected(
        workspace,
        _sources(workspace),
        (scenario,),
        years=(2030,),
        premise_key=None,
    )

    assert len(results) == 1
    assert results[0].status == "reused"
    assert results[0].manifest_path == result_root / "manifest.json"
    assert "synthetic_ssp2: reusing the completed background scenario" in capsys.readouterr().err
