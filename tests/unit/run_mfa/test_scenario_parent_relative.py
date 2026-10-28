"""Focused scenario compilation contracts."""

from copy import deepcopy
from importlib.resources import files

import pandas as pd
import yaml
from scenario_fixtures import compile_scenario_payload

from pytoce.run_mfa.scenario.parent_relative import (
    OPTIMIZATION_PARENT_OVERLAY_METHOD_IDENTITY,
    compile_parent_relative_scenario_assignments,
    compiled_scenario_table_identity,
)
from pytoce.run_mfa.scenario.scenario_contract import ScenarioEvaluation
from pytoce.shared.runtime.filesystem import payload_sha256


def _assert_overlay_identity_from_complete_tables(
    parent: ScenarioEvaluation, candidate: ScenarioEvaluation
) -> None:
    """Reconstruct and verify the retained identity from its inputs."""
    schema = "pytoce_optimization_foreground_scenario_schema_v1"
    parent_digest = compiled_scenario_table_identity(parent.compiled_parameters)
    candidate_digest = compiled_scenario_table_identity(candidate.compiled_parameters)
    assert parent.compiled_table_sha256 == parent_digest
    assert candidate.compiled_table_sha256 == candidate_digest
    payload = {
        "schema_version": schema,
        "method_identity": OPTIMIZATION_PARENT_OVERLAY_METHOD_IDENTITY,
        "parent_foreground_scenario_identity": parent.foreground_scenario_identity,
        "parent_compiled_table_sha256": parent_digest,
        "realized_compiled_table_sha256": candidate_digest,
        "normalized_change_points_sha256": payload_sha256(candidate.normalized_change_points),
    }
    assert candidate.foreground_scenario_identity == f"{schema}::{payload_sha256(payload)}"


def test_parent_overlay_preserves_unselected_parent_and_exact_noop_identity() -> None:
    parent = compile_scenario_payload(
        {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": "custom_parent",
            "ssp_scenario": "SSP2",
            "changes": [
                {
                    "parameter": "collection_rate",
                    "region": "EUR",
                    "changes": [
                        {"year": 2030, "mode": "absolute", "value": 0.7},
                    ],
                },
                {
                    "parameter": "collection_rate",
                    "region": "USA",
                    "changes": [
                        {
                            "year": 2030,
                            "mode": "multiplier_from_default",
                            "value": 0.9,
                        },
                    ],
                },
            ],
        }
    )
    parent_table = parent.compiled_parameters.copy()
    coordinate_columns = [
        "parameter_id",
        "year",
        "region",
        "material",
        "good",
        "process",
        "route",
        "resolved_value",
    ]
    assignments = parent.compiled_parameters.loc[
        parent.compiled_parameters["parameter_id"].eq("collection_rate")
        & parent.compiled_parameters["region"].eq("USA"),
        coordinate_columns,
    ].reset_index(drop=True)

    no_op = compile_parent_relative_scenario_assignments(
        parent,
        "optimization_noop",
        assignments,
    )
    assert no_op is parent
    assert no_op.foreground_scenario_identity == parent.foreground_scenario_identity
    assert no_op.compiled_table_sha256 == parent.compiled_table_sha256

    changed_assignments = assignments.copy()
    changed_assignments.loc[changed_assignments["year"].ge(2050), "resolved_value"] = 0.8
    candidate = compile_parent_relative_scenario_assignments(
        parent,
        "optimization_candidate",
        changed_assignments.iloc[::-1],
    )

    assert candidate.foreground_scenario_identity != parent.foreground_scenario_identity
    _assert_overlay_identity_from_complete_tables(parent, candidate)
    assert candidate.compiled_parameters["year"].min() == 2020
    assert 2019 not in set(candidate.compiled_parameters["year"])
    pd.testing.assert_frame_equal(parent.compiled_parameters, parent_table, check_exact=True)
    candidate_table = candidate.compiled_parameters.copy()
    selected = parent_table["parameter_id"].eq("collection_rate") & parent_table["region"].eq("USA")
    pd.testing.assert_frame_equal(
        candidate_table.loc[~selected].reset_index(drop=True),
        parent_table.loc[~selected].reset_index(drop=True),
        check_exact=True,
    )
    usa = candidate_table.loc[selected].set_index("year")
    parent_usa = parent_table.loc[selected].set_index("year")
    pd.testing.assert_series_equal(
        usa.loc[2020:2049, "resolved_value"],
        parent_usa.loc[2020:2049, "resolved_value"],
        check_exact=True,
    )
    assert usa.loc[2050:2100, "resolved_value"].eq(0.8).all()
    assert usa.loc[2050:2100, "value_origin"].eq("optimization_parent_overlay").all()
    assert (
        candidate.normalized_change_points[: len(parent.normalized_change_points)]
        == parent.normalized_change_points
    )
    overlay_points = candidate.normalized_change_points[len(parent.normalized_change_points) :]
    assert len(overlay_points) == 51
    assert [int(point["year"]) for point in overlay_points] == list(range(2050, 2101))
    assert all(point["parameter"] == "collection_rate" for point in overlay_points)
    assert all(point["region"] == "USA" for point in overlay_points)
    assert all(point["mode"] == "absolute_parent_overlay" for point in overlay_points)
    eur = candidate_table.loc[
        candidate_table["parameter_id"].eq("collection_rate") & candidate_table["region"].eq("EUR")
    ]
    parent_eur = parent_table.loc[
        parent_table["parameter_id"].eq("collection_rate") & parent_table["region"].eq("EUR")
    ]
    pd.testing.assert_frame_equal(
        eur.reset_index(drop=True),
        parent_eur.reset_index(drop=True),
        check_exact=True,
    )

    reordered = compile_parent_relative_scenario_assignments(
        parent,
        "optimization_candidate",
        changed_assignments,
    )
    assert reordered.foreground_scenario_identity == candidate.foreground_scenario_identity
    assert reordered.normalized_change_points == candidate.normalized_change_points
    pd.testing.assert_frame_equal(
        reordered.compiled_parameters,
        candidate.compiled_parameters,
        check_exact=True,
    )


def test_zero_ab_overlay_reuses_the_exact_frozen_parent() -> None:
    source = files("pytoce.run_mfa").joinpath(
        "data/workspace/data_raw/foreground_scenarios/scenarios/frozen_2019_default_baseline.yaml"
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    parent = compile_scenario_payload(payload)
    assignments = parent.compiled_parameters.loc[
        parent.compiled_parameters["parameter_id"].isin({"stock_projection", "lifetime_mean"}),
        [
            "parameter_id",
            "year",
            "region",
            "material",
            "good",
            "process",
            "route",
            "resolved_value",
        ],
    ].copy()

    candidate = compile_parent_relative_scenario_assignments(
        parent,
        "optimization_frozen_zero_ab",
        assignments,
    )

    assert candidate is parent
    assert candidate.foreground_scenario_identity == parent.foreground_scenario_identity
    assert candidate.normalized_change_points == parent.normalized_change_points
    pd.testing.assert_frame_equal(
        candidate.compiled_parameters,
        parent.compiled_parameters,
        check_exact=True,
    )


def test_successive_overlays_bind_each_distinct_parent_without_mutation() -> None:
    """Keep complete identities correct across child parents and reused readable names."""
    coordinate_columns = [
        "parameter_id",
        "year",
        "region",
        "material",
        "good",
        "process",
        "route",
        "resolved_value",
    ]
    parent_digests = []
    candidate_identities = []
    for collection in (0.6, 0.7):
        parent = compile_scenario_payload(
            {
                "schema_version": "pytoce_foreground_scenario_schema_v1",
                "foreground_scenario": "same_readable_parent",
                "ssp_scenario": "SSP2",
                "changes": [
                    {
                        "parameter": "collection_rate",
                        "region": "EUR",
                        "changes": [{"year": 2030, "mode": "absolute", "value": collection}],
                    },
                ],
            }
        )
        parent_table = parent.compiled_parameters.copy(deep=True)
        parent_points = deepcopy(parent.normalized_change_points)
        first_assignments = parent_table.loc[
            parent_table["parameter_id"].eq("collection_rate")
            & parent_table["region"].eq("USA")
            & parent_table["year"].ge(2050),
            coordinate_columns,
        ].copy()
        first_assignments["resolved_value"] = 0.8
        first = compile_parent_relative_scenario_assignments(
            parent, "same_readable_candidate", first_assignments
        )
        _assert_overlay_identity_from_complete_tables(parent, first)
        first_table = first.compiled_parameters.copy(deep=True)
        first_points = deepcopy(first.normalized_change_points)
        second_assignments = first_table.loc[
            first_table["parameter_id"].eq("collection_rate")
            & first_table["region"].eq("EUR")
            & first_table["year"].ge(2070),
            coordinate_columns,
        ].copy()
        second_assignments["resolved_value"] = 0.9
        second = compile_parent_relative_scenario_assignments(
            first, "successive_candidate", second_assignments
        )
        _assert_overlay_identity_from_complete_tables(first, second)
        assert second.compiled_table_sha256 != first.compiled_table_sha256
        pd.testing.assert_frame_equal(parent.compiled_parameters, parent_table, check_exact=True)
        pd.testing.assert_frame_equal(first.compiled_parameters, first_table, check_exact=True)
        assert parent.normalized_change_points == parent_points
        assert first.normalized_change_points == first_points
        repeated = compile_parent_relative_scenario_assignments(
            parent, "same_readable_candidate", first_assignments.iloc[::-1]
        )
        assert repeated.foreground_scenario_identity == first.foreground_scenario_identity
        assert repeated.compiled_table_sha256 == first.compiled_table_sha256
        pd.testing.assert_frame_equal(
            repeated.compiled_parameters, first.compiled_parameters, check_exact=True
        )
        parent_digests.append(parent.compiled_table_sha256)
        candidate_identities.append(first.foreground_scenario_identity)
    assert len(set(parent_digests)) == 2
    assert len(set(candidate_identities)) == 2
