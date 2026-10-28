"""Isolate four default REMIND and four historical REMIND-EU background identities."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import cast

from material_recipe_fixtures import physical_material_recipe
from pa_receiving_fixtures import pa_receiving_reference

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.premise import (
    BACKGROUND_SCENARIOS,
    GENERAL_BACKGROUND_ANCHOR_YEARS,
    PremiseIAMInput,
    database_name,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.member_preparation import (
    target,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.shared_preparation import (
    SharedAuthority,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.source_authority import (
    Sources,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    prepared_provider_authority,
)


def _background_identity(background_scenario: str) -> str:
    return "background::" + sha256(background_scenario.encode("utf-8")).hexdigest()


def test_all_eight_members_have_88_distinct_anchor_database_names() -> None:
    names = {
        database_name(
            scenario,
            year,
            background_identity=_background_identity(scenario.background_scenario),
        )
        for scenario in BACKGROUND_SCENARIOS
        for year in GENERAL_BACKGROUND_ANCHOR_YEARS
    }
    assert len(BACKGROUND_SCENARIOS) == 8
    assert len(names) == 88
    for scenario in BACKGROUND_SCENARIOS:
        own = {
            database_name(
                scenario,
                year,
                background_identity=_background_identity(scenario.background_scenario),
            )
            for year in GENERAL_BACKGROUND_ANCHOR_YEARS
        }
        others = names.difference(own)
        assert len(own) == 11
        assert own.isdisjoint(others)


def test_member_identity_is_not_a_multi_background_identity(tmp_path: Path) -> None:
    source = Sources(
        ecoinvent_archive=tmp_path / "ecoinvent",
        ef_workbook=tmp_path / "ef",
        pb_archive=tmp_path / "pb",
        source_identity="source::" + "d" * 64,
        source_identity_payload={"schema_version": "source::synthetic"},
        dependency_authorities=(
            {
                "distribution": "premise",
                "version": "2.4.6",
                "identity": "premise::synthetic",
            },
        ),
        premise_authority={
            "distribution": "premise",
            "version": "2.4.6",
            "identity": "premise::synthetic",
        },
        premise_authority_identity="premise::synthetic",
        material_recipe=physical_material_recipe(),
        provider=prepared_provider_authority(),
        preparation_method={
            "schema_version": "pytoce_background_preparation_method_schema_v1",
            "identity": "software::synthetic",
        },
    )
    shared = SharedAuthority(
        pa_receiving_reference=pa_receiving_reference("source"),
        names=BrightwayNames(tmp_path / "store", "project", "biosphere", "source"),
        identity="shared::" + "1" * 64,
        source_identity="source::" + "d" * 64,
        preparation_method_identity="software::synthetic",
        lcia_identity="lcia::" + "2" * 64,
        provider_identity="provider::" + "3" * 64,
        provider_interface_database="pytoce__provider_interface",
        provider_timeline_identity="provider_timeline::" + "4" * 64,
        provider_timeline_database_by_year={
            2020: "pytoce__provider_interface",
            2025: "pytoce__provider_timeline__2025",
        },
        provider_routes=(),
        impact_categories=(),
        database_rows=(),
        matrix_rows=(),
        biosphere_rows=(),
    )
    iam_inputs = {
        scenario.background_scenario: PremiseIAMInput(
            identity=f"iam::{scenario.background_scenario}",
            payload=tmp_path / scenario.iam_filename,
            manifest_sha256="e" * 64,
            payload_sha256=scenario.iam_file_sha256,
            premise_authority_identity="premise::synthetic",
        )
        for scenario in BACKGROUND_SCENARIOS
    }
    targets = [
        target(
            tmp_path,
            source,
            shared,
            iam_inputs[scenario.background_scenario],
            scenario,
            years=GENERAL_BACKGROUND_ANCHOR_YEARS,
        )
        for scenario in BACKGROUND_SCENARIOS
    ]
    assert len({target.identity for target in targets}) == 8
    assert all(
        cast(dict[str, object], target.identity_payload["background_scenario_payload"])[
            "background_scenario"
        ]
        == target.scenario.background_scenario
        for target in targets
    )
    changed_source = replace(
        source,
        dependency_authorities=(
            {
                "distribution": "premise",
                "version": "2.4.6",
                "identity": "premise::changed",
            },
        ),
    )
    changed = target(
        tmp_path,
        changed_source,
        shared,
        iam_inputs[BACKGROUND_SCENARIOS[0].background_scenario],
        BACKGROUND_SCENARIOS[0],
        years=GENERAL_BACKGROUND_ANCHOR_YEARS,
    )
    assert changed.background_identity != targets[0].background_identity
    assert changed.identity != targets[0].identity

    changed_provider = target(
        tmp_path,
        source,
        replace(
            shared,
            provider_identity="provider::" + "4" * 64,
        ),
        iam_inputs[BACKGROUND_SCENARIOS[0].background_scenario],
        BACKGROUND_SCENARIOS[0],
        years=GENERAL_BACKGROUND_ANCHOR_YEARS,
    )
    assert changed_provider.background_identity == targets[0].background_identity
    assert changed_provider.identity != targets[0].identity

    changed_timeline_identity = target(
        tmp_path,
        source,
        replace(
            shared,
            provider_timeline_identity="provider_timeline::" + "5" * 64,
        ),
        iam_inputs[BACKGROUND_SCENARIOS[0].background_scenario],
        BACKGROUND_SCENARIOS[0],
        years=GENERAL_BACKGROUND_ANCHOR_YEARS,
    )
    assert changed_timeline_identity.background_identity == targets[0].background_identity
    assert changed_timeline_identity.identity != targets[0].identity

    changed_timeline_map = target(
        tmp_path,
        source,
        replace(
            shared,
            provider_timeline_database_by_year={
                **shared.provider_timeline_database_by_year,
                2030: "pytoce__provider_timeline__2030",
            },
        ),
        iam_inputs[BACKGROUND_SCENARIOS[0].background_scenario],
        BACKGROUND_SCENARIOS[0],
        years=GENERAL_BACKGROUND_ANCHOR_YEARS,
    )
    assert changed_timeline_map.background_identity == targets[0].background_identity
    assert changed_timeline_map.identity != targets[0].identity
    assert targets[0].identity_payload["provider_timeline_identity"] == (
        shared.provider_timeline_identity
    )
    assert targets[0].identity_payload["provider_timeline_database_by_year"] == (
        shared.provider_timeline_database_by_year
    )
