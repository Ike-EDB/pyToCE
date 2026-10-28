"""Selected-state assembly coverage for detailed public optimization reporting."""

import json
from dataclasses import replace
from math import fsum
from pathlib import Path

import pandas as pd
import pytest
from candidate_characterization_fixtures import _empty_pa_scoring, _physical_parent
from candidate_worker_fixtures import (
    _infeasible_result,
    _intervention,
    _result_and_acc_metrics,
    _summary,
)
from result_table_fixtures import (
    _integrated_candidate,
    _search_products,
    _selected,
    _selected_acc_reconstruction_fixture,
)
from selected_reporting_fixtures import BACKGROUNDS, METHOD, STAGE, _parents, _projection

from pytoce.run_lca.calculation.foreground import (
    OPTIMIZATION_ACTIVATION_COLUMNS,
    optimization_activation_delta,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    build_response_parent_score_projection,
)
from pytoce.run_optimization.calculation.request_search import (
    SelectedCandidateEvaluation,
    _candidate_evaluation,
)
from pytoce.run_optimization.calculation.results import (
    SELECTED_RESULTS_METHOD_IDENTITY,
    _selected_reporting_states,
    build_selected_result_tables,
    expand_detailed_acc_metrics,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    _selected_candidate_result,
    decode_candidate_worker_result,
    encode_candidate_worker_result,
)
from pytoce.run_optimization.runtime.result_store import (
    StagedTable,
    stage_selected_life_cycle_scores,
    write_optimization_result,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256


def test_selected_reporting_state_binds_2050_selection_to_fixed_timing_results() -> None:
    """Preserve selection evidence while attaching the separate 2100 evaluation."""
    selected_2050, _context, detailed = _selected_acc_reconstruction_fixture()
    summary = _summary(selected_2050.selected_result.result)
    metric = _candidate_evaluation(selected_2050.candidate, summary).state_metrics[0]
    timing = tuple(
        replace(
            _selected(_integrated_candidate(target_year=year)),
            selection_identities=(metric.state_identity,),
        )
        for year in (2100,)
    )
    products = _search_products(selected_2050, timing)
    physical = selected_2050.selected_result.result.physical.physical_candidate_identity

    states = _selected_reporting_states(products, {physical: detailed})

    assert len(states) == 1
    state = states[0]
    assert state.state_identity == metric.state_identity
    assert state.selection_candidate_identity == selected_2050.candidate.identity
    assert state.timing_2100_candidate_identity == timing[0].candidate.identity
    assert state.declared_neighborhood_verified
    assert state.selection_stable
    assert state.selection_target_year == 2050
    assert state.collection_pct == selected_2050.candidate.circularity.collection_pct
    assert state.transgressing_coordinates == ()


def test_detailed_metric_expansion_deduplicates_one_physical_candidate() -> None:
    """Reconstruct a repeated selected MFA result only once across reporting roles."""

    selected, request_context, expected = _selected_acc_reconstruction_fixture()
    result = selected.selected_result.result

    observed = expand_detailed_acc_metrics((result, result), request_context)

    physical = result.physical.physical_candidate_identity
    assert tuple(observed) == (physical,)
    assert tuple(item.acc_evaluation_identity for item in observed[physical]) == tuple(
        item.acc_evaluation_identity for item in expected
    )


@pytest.mark.parametrize("continuation_identity", (None, "completed-five-point-result"))
@pytest.mark.parametrize("delayed_feasible", (False, True))
def test_complete_selected_result_table_set_is_assembled_from_owned_inputs(
    tmp_path: Path,
    continuation_identity: str | None,
    delayed_feasible: bool,
) -> None:
    """Keep failed timing paths and roles while writing only feasible numerical results."""

    unit_scores = (0.25, -1.0, 0.0, 1.0)
    parents = _parents(unit_scores)
    projection = _projection(unit_scores)
    services = pd.concat(
        (
            projection.provider_service_registry,
            pd.DataFrame.from_records(
                [
                    {
                        "provider_service_id": f"service::{name}",
                        "provider_service_code": f"service_{name}",
                        "provider_unit": "kilogram",
                    }
                    for name in ("positive", "negative")
                ]
            ),
        ),
        ignore_index=True,
    )
    responses = pd.concat(
        (
            projection.provider_unit_responses,
            pd.DataFrame.from_records(
                [
                    {
                        "background_scenario": background,
                        "provider_anchor_year": 2030,
                        "provider_service_id": f"service::{name}",
                        "provider_service_code": f"service_{name}",
                        "provider_unit": "kilogram",
                        **METHOD,
                        "unit_score": score,
                        "unit_contribution_norm": 1.0,
                    }
                    for background in BACKGROUNDS
                    for name, score in (("positive", 1.0), ("negative", -1.0))
                ]
            ),
        ),
        ignore_index=True,
    )
    projection = replace(
        projection,
        provider_service_registry=services,
        provider_service_registry_identity=canonical_table_sha256(services),
        provider_unit_responses=responses,
        provider_unit_responses_identity=canonical_table_sha256(responses),
    )
    activations = pd.DataFrame.from_records(
        [
            {
                "year": year,
                "region": region,
                "demand_type": STAGE["life_cycle_stage"],
                "provider_service_id": f"service::{name}",
                "provider_service_code": f"service_{name}",
                **STAGE,
                "physical_event_year": year,
                "amount": amount,
            }
            for year, region, name, amount in (
                (2020, "EUR", "one", 1.0),
                (2021, "CAZ", "positive", 1.0e16),
                (2021, "CHA", "positive", 1.0),
                (2021, "EUR", "negative", 1.0e16),
            )
        ],
        columns=OPTIMIZATION_ACTIVATION_COLUMNS,
    )
    candidate_activations = activations.copy()
    candidate_activations.loc[candidate_activations["year"].eq(2020), "amount"] = 0.0
    delta = optimization_activation_delta(activations, candidate_activations)
    result, metrics = _result_and_acc_metrics(activation_delta=delta)
    result = decode_candidate_worker_result(encode_candidate_worker_result(result))
    summary = _summary(result)
    selected_result = _selected_candidate_result(
        result,
        summary,
        selection_identities=(metrics.metrics.optimization_item_identity,),
    )
    evaluation = SelectedCandidateEvaluation(
        candidate=_integrated_candidate(),
        selection_identities=selected_result.selection_identities,
        selected_result=selected_result,
    )
    delayed_result, _ = _result_and_acc_metrics(
        activation_delta=delta, intervention=_intervention(target_year=2100)
    )
    failed_result = _infeasible_result(intervention=_intervention(target_year=2100))
    timing = tuple(
        SelectedCandidateEvaluation(
            candidate=_integrated_candidate(target_year=year),
            selection_identities=selected_result.selection_identities,
            selected_result=_selected_candidate_result(
                saved_result,
                _summary(saved_result),
                selection_identities=selected_result.selection_identities,
            ),
        )
        for year, saved_result in ((2100, delayed_result if delayed_feasible else failed_result),)
    )
    physical_parent = replace(
        _physical_parent(),
        identity=result.physical.physical_parent_identity,
        provider_route_identity=projection.provider_route_identity,
        activations=activations,
    )
    pa_projection, pa_parent = _empty_pa_scoring(projection.method_registry)
    result_parent = build_response_parent_score_projection(
        physical_parent,
        projection,
        pa_parent=pa_parent,
        pa_method_registry=pa_projection.method_registry,
    )
    expected_parent = pd.concat(
        [
            parent.annual_life_cycle_stage_scores.groupby(
                ["region", "year", *tuple(METHOD)[:-1]], observed=True, sort=True
            )[["score", "contribution_norm"]]
            .agg(fsum)
            .reset_index()
            .assign(
                background_scenario=parent.record.background_scenario,
                factor_identity=METHOD["factor_identity"],
            )
            for parent in parents
        ],
        ignore_index=True,
    )
    parent_keys = ["background_scenario", "region", "year", *METHOD]
    pd.testing.assert_frame_equal(
        result_parent.annual_scores.set_index(parent_keys).sort_index().reset_index(),
        expected_parent.set_index(parent_keys).sort_index().reset_index(),
        check_dtype=False,
        check_exact=True,
    )
    _unused_selected, request_context, _unused_metrics = _selected_acc_reconstruction_fixture()
    request_context = replace(
        request_context,
        search_scoring=replace(
            request_context.search_scoring,
            physical_parent=physical_parent,
            pa_projection=pa_projection,
            pa_parent=pa_parent,
        ),
        result_parent=result_parent,
        result_projection=projection,
        result_pa_projection=pa_projection,
        result_pa_parent=pa_parent,
        result_lcia_selection_identity="result-selection::ef31",
    )
    execution = _search_products(evaluation, timing)
    execution = replace(
        execution,
        search=replace(execution.search, continuation_parent_identity=continuation_identity),
    )
    progress: list[tuple[int, int]] = []

    directory = tmp_path / "selected"
    with stage_selected_life_cycle_scores(directory) as writer:
        built = build_selected_result_tables(
            execution,
            request_context=request_context,
            baseline_lca_results=parents,
            stage_writer=writer,
            stage_progress=lambda completed, total: progress.append((completed, total)),
        )
        assert set(built.tables) == {
            "selected_coordinates.parquet",
            "selected_annual_interventions.parquet",
            "selection_roles.parquet",
            "selected_annual_acc_asr.parquet",
            "selected_environmental_scores.parquet",
            "selected_life_cycle_stage_scores.parquet",
            "selected_feedstock_requirements.parquet",
        }
        stage = built.tables["selected_life_cycle_stage_scores.parquet"]
        assert isinstance(stage, StagedTable)
        assert built.parent_identities.get("optimization_continuation") == continuation_identity
        assert ("optimization_continuation" in built.parent_identities) == (
            continuation_identity is not None
        )
        write_optimization_result(
            directory,
            product_identity=built.identity,
            parent_identities=built.parent_identities,
            method_identity=built.method_identity,
            tables=built.tables,
            summary=built.summary,
        )
    stage_scores = pd.read_parquet(directory / "selected_life_cycle_stage_scores.parquet")
    environmental = pd.read_parquet(directory / "selected_environmental_scores.parquet")
    coordinates = pd.read_parquet(directory / "selected_coordinates.parquet").set_index(
        "target_year"
    )
    assert coordinates.physical_status.to_dict() == {
        2050: "physical_feasible",
        2100: "physical_feasible" if delayed_feasible else "physical_infeasible",
    }
    assert pd.isna(coordinates.loc[2050, "diagnostic_code"])
    if delayed_feasible:
        assert pd.isna(coordinates.loc[2100, "diagnostic_code"])
    else:
        assert (
            coordinates.loc[2100, "diagnostic_code"] == "optimization_stock_below_surviving_cohort"
        )
    interventions = pd.read_parquet(directory / "selected_annual_interventions.parquet")
    assert interventions.groupby("candidate_identity").size().to_dict() == {
        item.candidate.identity: 2 * 82 + 8 * 12 * 81 for item in (evaluation, *timing)
    }
    assert interventions.columns.tolist() == [
        "candidate_identity",
        "year",
        "region",
        "parameter_id",
        "value",
    ]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["selected_annual_interventions.parquet"]["columns"] == (
        interventions.columns.tolist()
    )
    requested = interventions.merge(
        coordinates.reset_index()[["candidate_identity", "physical_status"]],
        on="candidate_identity",
    )
    failed_path = requested.loc[requested.physical_status.eq("physical_infeasible")]
    if delayed_feasible:
        assert failed_path.empty
    else:
        assert set(failed_path.candidate_identity) == {timing[0].candidate.identity}
        assert set(failed_path.loc[failed_path.region.notna(), "year"]) == set(range(2020, 2101))
        assert failed_path.loc[
            failed_path.year.eq(2100)
            & failed_path.region.eq("EUR")
            & failed_path.parameter_id.eq("landfill_share"),
            "value",
        ].item() == pytest.approx(0.1)
    roles = pd.read_parquet(directory / "selection_roles.parquet")
    assert roles.selection_candidate_identity.eq(evaluation.candidate.identity).all()
    assert roles.timing_2100_candidate_identity.eq(timing[0].candidate.identity).all()
    assert roles.selection_status.eq("intervention_compliant").all()
    assert roles.worst_asr.eq(execution.search.block_results[0].selected_states[0].worst_asr).all()
    failed_physical = failed_result.physical.physical_candidate_identity
    for name, identity_column in (
        ("selected_environmental_scores.parquet", "physical_candidate_identity"),
        ("selected_annual_acc_asr.parquet", "physical_candidate_identity"),
        ("selected_life_cycle_stage_scores.parquet", "candidate_identity"),
        ("selected_feedstock_requirements.parquet", "physical_candidate_identity"),
    ):
        table = pd.read_parquet(directory / name)
        assert set(table.target_year) == ({2050, 2100} if delayed_feasible else {2050})
        if not delayed_feasible:
            assert not table[identity_column].eq(failed_physical).any()
    keys = [
        "candidate_identity",
        "target_year",
        "characterization_background_scenario",
        "region",
        "year",
        *METHOD,
    ]
    values = ["score", "contribution_norm"]
    annual_stage = stage_scores.loc[stage_scores["period_kind"].eq("annual")]
    grouped_stage = annual_stage.groupby(keys, observed=True, sort=True)[values].agg(fsum)
    annual_environmental = (
        environmental.rename(
            columns={
                "physical_candidate_identity": "candidate_identity",
                "background_scenario": "characterization_background_scenario",
            }
        )
        # Both labels describe the sum of the 12 world regions.
        .replace({"region": {"global": "global_derived"}})
        .set_index(keys)[values]
        .sort_index()
    )
    pd.testing.assert_frame_equal(
        annual_environmental.reset_index(),
        grouped_stage.reset_index(),
        check_dtype=False,
        check_exact=True,
    )
    assert environmental.loc[environmental["year"].eq(2020), values].eq(0.0).all().all()
    for region, score, absolute_total in (
        ("CAZ", 1.0e16, 1.0e16),
        ("CHA", 1.0, 1.0),
        ("EUR", -1.0e16, 1.0e16),
        ("global", 1.0, 2.0e16),
    ):
        selected = environmental.loc[
            environmental["year"].eq(2021) & environmental["region"].eq(region), values
        ]
        assert selected.shape == ((1 + int(delayed_feasible)) * len(BACKGROUNDS), 2)
        assert selected.eq([score, absolute_total]).all().all()
    cumulative = stage_scores.loc[stage_scores["period_kind"].eq("cumulative_2019_2100")]
    cumulative_keys = [key for key in keys if key != "year"]
    grouped_cumulative = cumulative.groupby(cumulative_keys, observed=True, sort=True)[values].agg(
        fsum
    )
    cumulative_environmental = annual_environmental.groupby(
        level=cumulative_keys, observed=True, sort=True
    ).agg(fsum)
    pd.testing.assert_frame_equal(
        grouped_cumulative.reset_index(),
        cumulative_environmental.reset_index(),
        check_dtype=False,
        check_exact=True,
    )
    assert cumulative[["year", "physical_event_year"]].isna().all().all()
    global_cumulative = cumulative.loc[cumulative["region"].eq("global_derived")]
    assert len(global_cumulative) == (1 + int(delayed_feasible)) * len(BACKGROUNDS)
    assert global_cumulative["score"].eq(1.0).all()
    assert built.identity.startswith(SELECTED_RESULTS_METHOD_IDENTITY + "::")
    assert built.summary["selected_coordinate_count"] == 2
    assert built.summary["selected_unique_physical_candidate_count"] == 2
    assert built.summary["table_identities"]
    assert progress[-1][0] == progress[-1][1]
