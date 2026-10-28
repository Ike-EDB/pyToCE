"""Direct session results, intervention timing and scientific identities."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
from material_recipe_fixtures import material_source_members

from pytoce.download_data.sources.ecoinvent_materials import derive_material_recipe
from pytoce.run_mfa.model.materials import load_material_composition
from pytoce.run_mfa.model.mfa import (
    MFAEvaluationSession,
    _changed_parameter_ids,
    _first_changed_year,
    mfa_evaluation_session_identity,
    mfa_evaluation_session_identity_from_source,
)
from pytoce.run_mfa.remind_mfa.execution import (
    REMIND_MFA_ADAPTER_METHOD,
    RemindMFAExecutionRequest,
    RemindMFASource,
)
from pytoce.run_mfa.remind_mfa.translation import RemindMFAIdentity
from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue


def test_changed_parameter_detection_uses_only_prospective_authoring_origins() -> None:
    """Find the earliest scenario intervention relative to the admitted defaults."""

    compiled = pd.DataFrame.from_records(
        [
            {
                "parameter_id": "collection_rate",
                "year": 2020,
                "value_origin": "remind_mfa_default",
            },
            {
                "parameter_id": "collection_rate",
                "year": 2035,
                "value_origin": "user_foreground_scenario",
            },
            {
                "parameter_id": "lifetime_mean",
                "year": 2040,
                "value_origin": "optimization_parent_overlay",
            },
            {
                "parameter_id": "lifetime_std",
                "year": 2040,
                "value_origin": "optimization_parent_overlay",
            },
        ]
    )

    assert _changed_parameter_ids(compiled) == frozenset(
        {"collection_rate", "lifetime_mean", "lifetime_std"}
    )
    assert _first_changed_year(compiled, {"collection_rate", "lifetime_mean"}) == 2035


def _source_identity() -> dict[str, str]:
    """Return one immutable REMIND-MFA source identity mapping."""

    return {
        "repository": "https://example.invalid/remind-mfa",
        "tag": "v0.2.0",
        "commit": "commit",
        "tree": "tree",
        "input_version": "1.0.0",
        "input_revision": "revision",
        "input_archive_sha256": "a" * 64,
        "mrmfa_version": "1.1.0",
        "mrmfa_commit": "mrmfa",
        "madrat_version": "3.28.4",
        "configuration_sha256": "b" * 64,
        "dimension_schema_sha256": "c" * 64,
        "scenario": "SSP2",
        "adapter_identity": "pytoce_remind_mfa_stock_and_trade_v2",
    }


def test_evaluation_session_identity_binds_source_and_parameter_catalogue(tmp_path: Path) -> None:
    """Make candidate-session reuse depend on both source and parameter semantics."""

    catalogue = load_parameter_catalogue()
    assert isinstance(catalogue, tuple)
    source_identity = _source_identity()
    material_inputs = {
        "material_recipe_identity": "recipe::fixture",
        "material_composition_identity": "composition::fixture",
    }
    identity = mfa_evaluation_session_identity_from_source(
        source_identity, catalogue, **material_inputs
    )
    assert identity.startswith("pytoce_mfa_session_schema_v1::")
    assert identity == mfa_evaluation_session_identity_from_source(
        dict(reversed(tuple(source_identity.items()))),
        catalogue,
        **material_inputs,
    )
    assert identity != mfa_evaluation_session_identity_from_source(
        {**source_identity, "commit": "different"},
        catalogue,
        **material_inputs,
    )
    # Parent provenance is reused with the current execution session.
    current_source_identity = {**source_identity, "adapter_identity": REMIND_MFA_ADAPTER_METHOD}
    assert identity == mfa_evaluation_session_identity_from_source(
        current_source_identity,
        catalogue,
        **material_inputs,
    )

    source = RemindMFASource(
        code_root=tmp_path / "code",
        data_root=tmp_path / "data",
        code_archive_sha256="a" * 64,
        data_archive_sha256="b" * 64,
        source_manifest=tmp_path / "manifest.json",
    )
    request = RemindMFAExecutionRequest(
        source=source,
        configuration_yaml="{}\n",
        identity=RemindMFAIdentity(**current_source_identity),
    )
    assert mfa_evaluation_session_identity(request, catalogue, **material_inputs) == identity
    assert source_identity["adapter_identity"] == "pytoce_remind_mfa_stock_and_trade_v2"
    for name in material_inputs:
        changed = {**material_inputs, name: "different"}
        assert mfa_evaluation_session_identity(request, catalogue, **changed) != identity


def test_real_source_session_shares_one_direct_owner_for_saved_and_candidate_results(
    remind_source_request: RemindMFAExecutionRequest,
) -> None:
    """Exercise source preparation, complete MFA calculation and both result views."""
    # This boundary fixture declares collection as its sole editable control;
    # all other operational and cohort inputs come from the real CSV source.
    catalogue = tuple(
        item for item in load_parameter_catalogue() if item.parameter_id == "collection_rate"
    )
    compiled = pd.MultiIndex.from_product(
        [range(2019, 2101), ["EUR", "USA"]], names=["year", "region"]
    ).to_frame(index=False)
    compiled = compiled.assign(
        parameter_id="collection_rate",
        resolved_value=0.65,
        default_value=0.65,
        value_origin="remind_mfa_default",
        value_mode="retain_default",
        parameter_role="editable_input",
        controlling_change_point_year=2019,
        material=None,
        good=None,
        process=None,
        route=None,
        unit="fraction",
    )
    session = MFAEvaluationSession(
        remind_source_request,
        catalogue,
        recipe=derive_material_recipe(material_source_members(), source_identity="invented_xml"),
        composition=load_material_composition(),
    )
    progress: list[tuple[str, int | None]] = []
    saved = session.compute(
        foreground_scenario="direct_fixture",
        compiled=compiled,
        scenario_change_points=(),
        progress=lambda phase, year: progress.append((phase, year)),
    )
    assert progress == [("physical_material_realization", 2020), ("result_tables", 2100)]
    candidate = session.evaluate(foreground_scenario="direct_fixture", compiled=compiled)
    assert saved.physical_state_identity == candidate.physical_state_identity
    expected = saved.model_flows.loc[
        saved.model_flows["year"].ge(2019), candidate.model_flows.columns
    ]
    pd.testing.assert_frame_equal(
        candidate.model_flows, expected.reset_index(drop=True), check_exact=True
    )
    for name in ("material_realization", "nominal_feedstock_equivalents"):
        full = getattr(saved, name)
        pd.testing.assert_frame_equal(
            getattr(candidate, name),
            full.loc[full["year"].ge(2019)].reset_index(drop=True),
            check_exact=True,
        )
    changed = compiled.copy()
    select = changed["year"].ge(2020) & changed["region"].eq("EUR")
    changed.loc[select, "resolved_value"] = 0.45
    changed.loc[select, "value_origin"] = "user_foreground_scenario"
    changed.loc[select, "value_mode"] = "absolute"
    changed.loc[select, "controlling_change_point_year"] = 2020
    intervention = session.evaluate(foreground_scenario="changed", compiled=changed)
    changed_saved = session.compute(
        foreground_scenario="changed",
        compiled=changed,
        scenario_change_points=(
            {"parameter": "collection_rate", "year": 2020, "region": "EUR", "value": 0.45},
        ),
    )
    assert changed_saved.physical_state_identity == intervention.physical_state_identity
    pd.testing.assert_frame_equal(
        intervention.model_flows,
        changed_saved.model_flows.loc[
            changed_saved.model_flows["year"].ge(2019), intervention.model_flows.columns
        ].reset_index(drop=True),
        check_exact=True,
    )
    discard = changed_saved.cohort_state.loc[
        changed_saved.cohort_state["event_year"].ge(2020)
        & changed_saved.cohort_state["stock_quantity"].eq("outflow"),
        intervention.cohort_state.columns,
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(intervention.cohort_state, discard, check_exact=True)
    repeated = session.evaluate(foreground_scenario="direct_fixture", compiled=compiled)
    assert repeated.physical_state_identity == candidate.physical_state_identity
    assert intervention.physical_state_identity != candidate.physical_state_identity
    for result in (candidate, intervention):
        np.testing.assert_allclose(
            result.model_flows.loc[result.model_flows["year"].eq(2019), "amount"],
            candidate.model_flows.loc[candidate.model_flows["year"].eq(2019), "amount"],
            rtol=0.0,
            atol=0.0,
        )

    # Changed-scenario and candidate-first execution orders preserve the admitted history and
    # reports.
    changed_first = MFAEvaluationSession(
        remind_source_request,
        catalogue,
        recipe=derive_material_recipe(material_source_members(), source_identity="invented_xml"),
        composition=load_material_composition(),
    )
    first_candidate = changed_first.evaluate(foreground_scenario="changed", compiled=changed)
    first_report = changed_first.compute(
        foreground_scenario="changed",
        compiled=changed,
        scenario_change_points=(
            {"parameter": "collection_rate", "year": 2020, "region": "EUR", "value": 0.45},
        ),
    )
    later_default = changed_first.compute(
        foreground_scenario="direct_fixture", compiled=compiled, scenario_change_points=()
    )
    for actual, expected_result in (
        (first_candidate, intervention),
        (first_report, changed_saved),
        (later_default, saved),
    ):
        for field in fields(expected_result):
            expected_value = getattr(expected_result, field.name)
            actual_value = getattr(actual, field.name)
            if isinstance(expected_value, pd.DataFrame):
                pd.testing.assert_frame_equal(actual_value, expected_value, check_exact=True)
            else:
                assert actual_value == expected_value
