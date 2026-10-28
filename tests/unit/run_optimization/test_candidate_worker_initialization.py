"""Preserve source-matched parent controls, worker results and candidate reuse."""

import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pytoce import set_workspace
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    read_prepared_manifest,
)
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_lca.run_lca import DEFAULT_UPSTREAM_ANALYSIS_YEARS, lca_result_directory
from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue
from pytoce.run_mfa.scenario.scenario_contract import ScenarioFeasibilityError
from pytoce.run_optimization.calculation.candidate_search import (
    demand_lifetime_initial_design,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestBridge,
    initialize_candidate_worker,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.request_search import (
    build_candidate_worker_metric_scopes,
)
from pytoce.run_optimization.calculation.response_bank_store import (
    load_optimization_response_bank,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    CandidateWorkerResult,
    CandidateWorkerTask,
    decode_candidate_worker_result,
    encode_candidate_worker_task,
)
from pytoce.run_optimization.runtime.candidate_results import CandidateResultStore
from tests.integration.candidate_mfa_fixtures import (
    _WORKER_FOREGROUND,
    _prepare_worker_workspace,
)


def _assert_same_candidate_result(
    actual: object, expected: object, field_path: str = "worker_result"
) -> None:
    """Compare every decoded field after interleaved MFA calculations."""
    assert type(actual) is type(expected), field_path
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True, obj=field_path)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=field_path, strict=True)
    elif field_path.endswith(
        ("physical_repair_elapsed_seconds", "environmental_evaluation_elapsed_seconds")
    ):
        assert actual >= 0.0 and expected >= 0.0
    elif is_dataclass(expected):
        for field in fields(expected):
            _assert_same_candidate_result(
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{field_path}.{field.name}",
            )
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), field_path
        for index, (observed, reference) in enumerate(zip(actual, expected, strict=True)):
            _assert_same_candidate_result(observed, reference, f"{field_path}[{index}]")
    else:
        assert actual == expected, field_path


@pytest.mark.scientific_integration
def test_parent_reuse_and_interleaved_cohort_histories_preserve_complete_results(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> None:
    """Preserve complete results across parent reuse and interleaved stock/lifetime paths."""
    workspace = tmp_path / "workspace"
    initializer, context = _prepare_worker_workspace(workspace, remind_mfa_source_root)
    assert initializer == context.worker_initializer_payload
    assert "pa_parent" in json.loads(initializer)

    compiled = pd.read_parquet(
        workspace
        / "data_processed/foreground_scenarios"
        / _WORKER_FOREGROUND
        / "compiled_annual_input_parameters.parquet"
    )
    cohort_controls = compiled.loc[
        compiled.parameter_id.isin({"stock_projection", "plastic_family_share", "lifetime_mean"})
    ]
    assert set(cohort_controls.year) == set(range(2020, 2101))
    assert cohort_controls.value_origin.eq("user_foreground_scenario").all()
    stocks = compiled.loc[compiled.parameter_id.eq("stock_projection")].set_index(
        ["year", "region", "good"]
    )
    parent_stocks = context.physical_parent.demand_lifetime_parent.stock_targets.set_index(
        ["year", "region", "good"]
    )
    # Polymer-stock aggregation may differ by floating-point rounding; Mt units and coordinates are
    # preserved.
    np.testing.assert_allclose(
        stocks.resolved_value.reindex(parent_stocks.index),
        parent_stocks.stock_projection,
        rtol=1e-13,
        atol=1e-12,
    )
    source_inputs = workspace / "data_processed/remind_mfa/data/plastics/input_data"
    source_shares = pd.read_csv(
        source_inputs / "pl_material_shares_in_goods.cs4r",
        comment="*",
        header=None,
        names=["region", "material", "good", "source_share"],
    )
    shares = compiled.loc[compiled.parameter_id.eq("plastic_family_share")].merge(
        source_shares, on=["region", "material", "good"], validate="many_to_one"
    )
    assert len(shares) == 81 * len(source_shares)
    np.testing.assert_array_equal(shares.resolved_value, shares.source_share)
    assert set(shares.loc[shares.resolved_value.ne(0), "material"]) == {"HDPE", "PP"}
    for parameter in ("lifetime_mean", "lifetime_std"):
        source_lifetimes = pd.read_csv(
            source_inputs / f"pl_{parameter}.cs4r",
            comment="*",
            header=None,
            names=["good", "source_value"],
        )
        lifetimes = compiled.loc[compiled.parameter_id.eq(parameter)].merge(
            source_lifetimes, on="good", validate="many_to_one"
        )
        np.testing.assert_array_equal(lifetimes.resolved_value, lifetimes.source_value)

    # The response bank and public baseline must identify the actual completed
    # members, rather than unrelated test doubles with matching numerical rows.
    bank = load_optimization_response_bank(workspace, "ef31_response_v3")
    assert bank is not None
    inputs = json.loads((bank.result_directory / "manifest.json").read_bytes())["identity_payload"]
    prepared = workspace / "data_processed/prepared_backgrounds"
    shared = read_prepared_manifest(prepared / "shared/manifest.json")
    assert inputs["prepared_background_shared_identity"] == shared.payload["identity"]
    assert inputs["prepared_background_shared_manifest_sha256"] == shared.sha256
    assert len(inputs["member_parents"]) == 4
    for parent in inputs["member_parents"]:
        background = parent["background_scenario"]
        member = read_prepared_manifest(prepared / background / "manifest.json")
        assert parent["prepared_background_identity"] == member.payload["identity"]
        assert parent["prepared_background_manifest_sha256"] == member.sha256
        baseline = lca_result_directory(
            workspace,
            _WORKER_FOREGROUND,
            background,
            normalized_lcia_method(("ef31",)),
            DEFAULT_UPSTREAM_ANALYSIS_YEARS,
        )
        provenance = json.loads((baseline / "manifest.json").read_bytes())["identity_payload"]
        assert provenance["prepared_background_identity"] == member.payload["identity"]
        assert provenance["prepared_background_manifest_sha256"] == member.sha256

    worker = initialize_candidate_worker(initializer)
    scopes = build_candidate_worker_metric_scopes(("demand_lifetime_frontier",))
    intervention = context.planner.plan_intervention(
        demand_lifetime=DemandLifetimeDecision(0, 0), circularity=CircularityDecision.parent()
    )
    assert intervention.uses_exact_parent
    with CandidateResultStore.open(
        workspace / "results/initialization_cache",
        candidate_input_identity="request::exact-parent-initialization-fixture",
        worker_authority_identity=context.planner.worker_authority_identity,
    ) as store:
        bridge = CandidateWorkerRequestBridge(
            context.planner,
            scopes,
            result_store=store,
            allocated_cc_context=context.allocated_cc_context,
            search_scoring=context.search_scoring,
            worker_scientific_authority_identity=context.worker_scientific_authority_identity,
        )
        task = bridge.build_task(intervention, scopes)
        payload = worker(task.payload)
        result = decode_candidate_worker_result(payload)
        saved = store.save(task, payload)
        assert store.read_record(task.record) == payload
        summary = bridge.read_saved_result(saved)
        reused = bridge.load_candidate(task.candidate_identity)

    assert summary.worker_result_identity == result.identity == reused.identity
    assert result.worker_scientific_authority_identity == (
        context.worker_scientific_authority_identity
    )
    assert result.physical.status == "physical_feasible"
    assert result.physical.pa_spinning_events == ()
    assert result.physical.pa_recycling_events == ()
    assert result.physical.pa_disposal_events == ()
    assert result.physical.activation_delta is not None and result.physical.activation_delta.empty
    assert result.scores is not None
    np.testing.assert_array_equal(
        result.scores.annual_scores, context.search_scoring.optimization_baseline_scores
    )
    np.testing.assert_array_equal(
        result.scores.annual_climate_scores, context.search_scoring.climate_baseline_scores[..., 0]
    )
    assert result.pa_scores is not None and reused.pa_scores is not None
    assert np.count_nonzero(result.pa_scores.annual_scores) == 0
    assert np.count_nonzero(result.pa_scores.annual_contribution_norms) == 0
    assert result.pa_scores.stage_scores.empty
    np.testing.assert_array_equal(
        reused.pa_scores.annual_scores, context.search_scoring.pa_parent.annual_scores
    )
    np.testing.assert_array_equal(
        reused.pa_scores.annual_contribution_norms,
        context.search_scoring.pa_parent.annual_contribution_norms,
    )
    pd.testing.assert_frame_equal(
        reused.pa_scores.stage_scores, context.search_scoring.pa_parent.stage_scores
    )

    # Interleave three distinct stock/lifetime histories. A is reused while
    # recent; B returns after C displaced its cohort calculations. The complete
    # result must be independent of earlier requests and retained cache state.
    # Call the real worker directly: the persistent result store would correctly
    # bypass MFA calculation for repeated interventions.
    decisions = {
        "A": DemandLifetimeDecision(25, 25),
        "B": DemandLifetimeDecision(0, 25),
        "C": DemandLifetimeDecision(0, 50),
    }
    first_results: dict[str, CandidateWorkerResult] = {}
    for label in ("A", "B", "A", "C", "B"):
        changed = context.planner.plan_intervention(
            demand_lifetime=decisions[label], circularity=CircularityDecision.parent()
        )
        assert not changed.uses_exact_parent
        changed_task = CandidateWorkerTask(changed, scopes)
        changed_result = decode_candidate_worker_result(
            worker(encode_candidate_worker_task(changed_task))
        )
        assert changed_result.physical.status == "physical_feasible"
        # Cohort changes retain the source-matched HDPE/PP composition.
        assert changed_result.physical.pa_spinning_events == ()
        assert changed_result.physical.pa_recycling_events == ()
        assert changed_result.physical.pa_disposal_events == ()
        assert changed_result.scores is not None
        if label in first_results:
            _assert_same_candidate_result(changed_result, first_results[label])
        else:
            first_results[label] = changed_result
    assert len({item.physical.physical_state_identity for item in first_results.values()}) == 3


@pytest.mark.scientific_integration
def test_worker_propagates_uncoded_lifetime_numerical_error(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> None:
    """Keep an actual numerical failure distinct from a scored infeasible candidate."""
    workspace = tmp_path / "workspace"
    foreground = "numerical_lifetime_parent"
    set_workspace(workspace)
    controls = {
        "collection_rate": 0.65,
        "mechanical_recycling_rate": 0.2,
        "chemical_recycling_rate": 0.1,
        "incineration_rate": 0.3,
        "bio_production_rate": 0.05,
        "daccu_production_rate": 0.0,
    }
    changes = [
        {
            "parameter": parameter,
            "selector": "uniform_all_h12",
            "changes": [{"year": 2020, "mode": "absolute", "value": value}],
        }
        for parameter, value in controls.items()
    ]
    # Retain the fixture's treatment/feedstock controls and set final-year absolute waste imports
    # and exports to zero.
    changes[0]["changes"].append({"year": 2100, "mode": "absolute", "value": 0.0})
    for parameter in ("waste_imports", "waste_exports"):
        changes.append(
            {
                "parameter": parameter,
                "regions": list(load_parameter_catalogue(parameter).coordinates["region"]),
                "changes": [{"year": 2100, "mode": "absolute", "value": 0.0}],
            }
        )
    # FLODYM squares the mean: (1e154 years)**2 is finite; squaring the doubled mean overflows. This
    # tests finite-input overflow with the source standard-deviation/mean ratio (3/13).
    changes.append(
        {
            "parameter": "lifetime_mean",
            "good": "Transportation",
            "changes": [{"year": 2100, "mode": "absolute", "value": 1e154}],
        }
    )
    scenario = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": foreground,
        "ssp_scenario": "SSP2",
        "changes": changes,
    }
    scenario_path = workspace / "data_raw/foreground_scenarios/scenarios" / f"{foreground}.yaml"
    scenario_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    # Author all changes before the first parent calculation. The real fixture
    # preserves them while adding its original stock, material and 2020 means.
    initializer, context = _prepare_worker_workspace(
        workspace, remind_mfa_source_root, foreground_scenario=foreground
    )
    candidate = next(
        item
        for item in demand_lifetime_initial_design()
        if item.demand_lifetime.demand_reduction_pct == 0
        and item.demand_lifetime.lifetime_extension_pct == 100
    )
    assert candidate.demand_lifetime.target_year == 2050
    intervention = context.planner.plan_intervention(
        demand_lifetime=candidate.demand_lifetime, circularity=candidate.circularity
    )
    worker = initialize_candidate_worker(initializer)
    task = CandidateWorkerTask(
        intervention, build_candidate_worker_metric_scopes(("demand_lifetime_frontier",))
    )
    with pytest.raises(ScenarioFeasibilityError, match="finite, nonnegative inflow") as error:
        worker(encode_candidate_worker_task(task))
    assert error.value.code is None
