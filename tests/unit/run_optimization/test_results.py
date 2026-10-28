from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from candidate_worker_fixtures import _summary
from intervention_fixtures import _shaped_decision
from result_table_fixtures import (
    _integrated_candidate,
    _search_products,
    _selected,
    _selected_acc_reconstruction_fixture,
)

from pytoce.run_optimization.calculation.candidate_acc import (
    CandidateACCMetrics,
)
from pytoce.run_optimization.calculation.candidate_evaluation import (
    SelectedCandidateCharacterization,
)
from pytoce.run_optimization.calculation.candidate_metrics import (
    candidate_metric_summary,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    ResultScores,
)
from pytoce.run_optimization.calculation.candidate_search import (
    InterventionCandidate,
    RefinementPass,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.request_search import (
    RequestSearchResult,
    SearchExecutionProducts,
    SelectedCandidateEvaluation,
)
from pytoce.run_optimization.calculation.results import (
    _annual_interventions_table,
    _coordinates_table,
    _environmental_table,
    _result_summary,
    _selected_result_identity,
    _states_table,
    build_acc_table,
    build_feedstock_requirements_table,
    expand_detailed_acc_metrics,
)
from pytoce.run_optimization.calculation.selected_stage_reporting import (
    stream_selected_stage_reporting,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    _selected_candidate_result,
)
from pytoce.run_optimization.runtime.result_store import (
    stage_selected_life_cycle_scores,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256


def _reporting_evaluation(
    candidate: InterventionCandidate,
    *,
    physical_candidate_identity: str,
    selection_identity: str,
) -> SelectedCandidateEvaluation:
    from selected_reporting_fixtures import _candidate_evaluation

    result = _candidate_evaluation(candidate_amount=2.0).result
    changed_physical = replace(
        result.physical,
        physical_candidate_identity=physical_candidate_identity,
    )
    changed_result = replace(
        result,
        identity=f"worker-result::{physical_candidate_identity}",
        physical=changed_physical,
        intervention=replace(
            result.intervention,
            demand_lifetime_evaluation_projection=replace(
                result.intervention.demand_lifetime_evaluation_projection,
                decision=candidate.demand_lifetime,
            ),
        ),
    )
    summary = _summary(changed_result)
    selected_result = _selected_candidate_result(
        changed_result,
        summary,
        selection_identities=(selection_identity,),
    )
    return SelectedCandidateEvaluation(
        candidate, selected_result.selection_identities, selected_result
    )


def _acc_index(
    selected: SelectedCandidateEvaluation,
    *,
    copies: int = 1,
) -> dict[str, tuple[CandidateACCMetrics, ...]]:
    from candidate_worker_fixtures import _result_and_acc_metrics

    _, metrics = _result_and_acc_metrics()
    physical = selected.selected_result.result.physical.physical_candidate_identity
    return {physical: (metrics,) * copies}


def test_selected_acc_expansion_reproduces_original_compact_metrics() -> None:
    selected, context, expected = _selected_acc_reconstruction_fixture()

    observed = expand_detailed_acc_metrics(
        (selected.selected_result.result,),
        context,
    )

    physical = selected.selected_result.result.physical.physical_candidate_identity
    assert tuple(item.acc_evaluation_identity for item in observed[physical]) == tuple(
        item.acc_evaluation_identity for item in expected
    )
    assert tuple(candidate_metric_summary(item.metrics) for item in observed[physical]) == tuple(
        candidate_metric_summary(item.metrics) for item in expected
    )


def test_coordinate_table_reports_direct_and_derived_share_values() -> None:
    selected = _selected(_integrated_candidate())

    table = _coordinates_table((selected,))

    assert table.loc[0, "demand_reduction_pct"] == 25
    assert table.loc[0, "lifetime_extension_pct"] == 50
    assert table.loc[0, "total_recycling_pct"] == 60
    assert table.loc[0, "mechanical_fraction_of_recycling"] == pytest.approx(2.0 / 3.0)
    assert table.loc[0, "target_year"] == 2050


def test_selected_progress_and_annual_fractions_distinguish_equal_endpoints() -> None:
    """Publish global controls once per year alongside complete regional circularity paths."""
    source = _integrated_candidate()
    linear = replace(source, demand_lifetime=DemandLifetimeDecision(25, 50))
    shaped = replace(
        linear,
        demand_lifetime=_shaped_decision(
            25,
            50,
        ),
    )
    timing = replace(
        shaped,
        demand_lifetime=DemandLifetimeDecision(25, 50, 2075),
        circularity=replace(shaped.circularity, target_year=2075),
    )
    selected = tuple(
        _reporting_evaluation(
            candidate,
            physical_candidate_identity=f"physical::{index}",
            selection_identity="selection",
        )
        for index, candidate in enumerate((linear, shaped, timing))
    )
    coordinates = _coordinates_table(selected).set_index("candidate_identity")
    assert len(coordinates) == 3
    assert coordinates["demand_reduction_pct"].eq(25).all()
    assert coordinates["lifetime_extension_pct"].eq(50).all()
    assert not any("progress_tick" in column for column in coordinates.columns)
    context = _selected_acc_reconstruction_fixture()[1]
    parent = context.planner.intervention_parent.circularity_parent
    annual = _annual_interventions_table(selected, parent)
    assert len(annual) == 3 * (2 * 82 + 8 * 12 * 81)
    assert annual.columns.tolist() == [
        "candidate_identity",
        "year",
        "region",
        "parameter_id",
        "value",
    ]
    assert not annual.duplicated(["candidate_identity", "year", "region", "parameter_id"]).any()
    regional = annual.loc[annual.region.notna()]
    assert set(regional.region) == set(parent.table.region)
    assert set(regional.year) == set(range(2020, 2101))
    assert set(regional.parameter_id) == set(parent.table.columns).difference({"year", "region"})
    for candidate in (linear, shaped, timing):
        observed = annual.loc[
            annual.candidate_identity.eq(candidate.identity) & annual.region.isna()
        ].pivot(index="year", columns="parameter_id", values="value")
        years = observed.index.to_numpy()
        assert years.tolist() == list(range(2019, 2101))
        demand = candidate.demand_lifetime
        np.testing.assert_array_equal(
            observed.demand_reduction_fraction, [float(value) for value in demand.demand_fraction]
        )
        np.testing.assert_array_equal(
            observed.lifetime_extension_fraction,
            [float(value) for value in demand.lifetime_fraction],
        )


def test_parent_circularity_is_reported_as_not_applicable_not_as_zero() -> None:
    candidate = InterventionCandidate(
        "demand_lifetime_frontier",
        DemandLifetimeDecision(25, 50),
        CircularityDecision.parent(),
    )

    table = _coordinates_table((_selected(candidate),))

    assert pd.isna(table.loc[0, "collection_pct"])
    assert pd.isna(table.loc[0, "total_recycling_pct"])
    context = _selected_acc_reconstruction_fixture()[1]
    parent = context.planner.intervention_parent.circularity_parent
    annual = _annual_interventions_table((_selected(candidate),), parent)
    regional = annual.loc[annual.region.notna()].pivot(
        index=["year", "region"], columns="parameter_id", values="value"
    )
    pd.testing.assert_frame_equal(
        regional.sort_index(axis=1),
        parent.table.set_index(["year", "region"]).sort_index().sort_index(axis=1),
        check_names=False,
        check_exact=True,
    )


def test_state_table_preserves_joint_selection_and_timing_provenance() -> None:
    from selected_reporting_fixtures import BACKGROUNDS, _states

    state = replace(
        _states(BACKGROUNDS[0])[0],
        physical_candidate_identity="physical",
        optimization_block="integrated_frontier",
        timing_2100_candidate_identity="candidate::2100",
        timing_2100_physical_candidate_identity="physical::2100",
        lifetime_extension_pct=50,
    )

    table = _states_table((state,))

    assert len(table) == 1
    assert table.loc[0, "optimization_block"] == "integrated_frontier"
    assert table.loc[0, "selection_background_scenario"] == BACKGROUNDS[0]
    assert table.loc[0, "selection_target_year"] == 2050
    assert table.loc[0, "timing_2100_candidate_identity"] == "candidate::2100"
    assert table.loc[0, "timing_2100_physical_candidate_identity"] == "physical::2100"
    assert table.loc[0, "worst_asr"] == pytest.approx(0.9)
    assert table.loc[0, "demand_reduction_pct"] == 25
    assert table.loc[0, "lifetime_extension_pct"] == 50
    assert table.loc[0, "collection_pct"] == 10
    assert table.loc[0, "selection_status"] == "intervention_compliant"
    assert table.loc[0, "declared_neighborhood_verified"]


def test_stage_reporting_forwards_distinct_timing_evaluations_and_parent_alias(
    tmp_path: Path,
) -> None:
    from candidate_characterization_fixtures import _empty_pa_scoring
    from selected_reporting_fixtures import (
        BACKGROUNDS,
        _parents,
        _projection,
        _states,
    )

    state = replace(
        _states(BACKGROUNDS[0])[0],
        physical_candidate_identity="physical::2050",
        timing_2100_physical_candidate_identity="physical::2100",
    )
    evaluations = tuple(
        _reporting_evaluation(
            _integrated_candidate(target_year=target_year),
            physical_candidate_identity=physical,
            selection_identity=state.state_identity,
        )
        for target_year, physical in (
            (2050, "physical::2050"),
            (2100, "physical::2100"),
        )
    )
    physical_index = {
        item.selected_result.result.physical.physical_candidate_identity: item
        for item in evaluations
    }
    unit_scores = (0.25, -1.0, 0.0, 1.0)
    projection = _projection(unit_scores)
    # This fixed-service fixture has no PA events in either parent or candidate.
    _, pa_parent = _empty_pa_scoring(projection.method_registry)

    with stage_selected_life_cycle_scores(tmp_path / "timings") as writer:
        stream_selected_stage_reporting(
            states=(state,),
            selected_evaluations=tuple(item.selected_result for item in evaluations),
            baseline_lca_results=_parents(unit_scores),
            response_projection=projection,
            pa_parent=pa_parent,
            pa_candidates={identity: pa_parent for identity in physical_index},
            sink=writer,
            progress=lambda _completed, _total: None,
        )
        staged = writer.finish()
        stage = pd.read_parquet(
            staged.staging_directory / "selected_life_cycle_stage_scores.parquet"
        )
    assert set(stage["candidate_identity"]) == set(physical_index)
    assert stage["period_kind"].tolist() == sorted(
        stage["period_kind"],
        key=lambda value: value != "annual",
    )
    parent_state = _states(BACKGROUNDS[0])[0]
    parent = _reporting_evaluation(
        _integrated_candidate(),
        physical_candidate_identity=parent_state.physical_candidate_identity,
        selection_identity=parent_state.state_identity,
    )
    with stage_selected_life_cycle_scores(tmp_path / "parent") as writer:
        stream_selected_stage_reporting(
            states=(parent_state,),
            selected_evaluations=(parent.selected_result,),
            baseline_lca_results=_parents(unit_scores),
            response_projection=projection,
            pa_parent=pa_parent,
            pa_candidates={parent_state.physical_candidate_identity: pa_parent},
            sink=writer,
            progress=lambda _completed, _total: None,
        )
        staged = writer.finish()
        parent_stage = pd.read_parquet(
            staged.staging_directory / "selected_life_cycle_stage_scores.parquet"
        )
    assert set(parent_stage["candidate_identity"]) == {parent_state.physical_candidate_identity}


def test_feedstock_table_keeps_exact_product_and_route_coordinates() -> None:
    selected = _selected(_integrated_candidate(target_year=2100))
    requirements = selected.selected_result.result.feedstock_requirements
    assert requirements is not None
    requirements = requirements.copy()
    keys = ["year", "region", "material", "constituent", "product", "virgin_feedstock_route"]
    coordinate = (2030, "EUR", "HDPE", "hdpe", "ethylene", "fossil")
    requirements.loc[pd.MultiIndex.from_frame(requirements[keys]).isin([coordinate]), "amount"] = (
        2.5
    )
    changed_result = replace(selected.selected_result.result, feedstock_requirements=requirements)
    changed_selected_result = replace(selected.selected_result, result=changed_result)
    selected = replace(selected, selected_result=changed_selected_result)

    table = build_feedstock_requirements_table(
        (
            (
                selected.candidate.target_year,
                selected.selected_result.result,
            ),
        )
    )

    row = table.loc[pd.MultiIndex.from_frame(table[keys]).isin([coordinate])].iloc[0]
    assert row["target_year"] == 2100
    assert row["physical_candidate_identity"] == (
        changed_result.physical.physical_candidate_identity
    )
    assert row["quantity"] == "virgin_monomer_consumption"
    assert row["product"] == "ethylene"
    assert row["virgin_feedstock_route"] == "fossil"
    assert row["amount"] == 2.5
    assert "good" not in table.columns


def test_timing_acc_rows_never_claim_selection_control() -> None:
    selected = _selected(_integrated_candidate(target_year=2075))
    table = build_acc_table(
        {selected.selected_result.result.physical.physical_candidate_identity: 2075},
        _acc_index(selected),
    )

    assert not table["controlled_for_selection"].any()
    assert table["selection_compliant"].isna().all()
    assert table["compliant"].notna().all()


def test_environmental_global_uses_stable_regional_summation() -> None:
    selected = _selected(_integrated_candidate())
    physical = selected.selected_result.result.physical.physical_candidate_identity
    regions = tuple(f"R{index:02d}" for index in range(12))
    regional_scores = np.zeros((1, 12, 1, 1), dtype=np.float64)
    regional_scores[0, :3, 0, 0] = (1.0e16, 1.0, -1.0e16)
    norms = np.abs(regional_scores)
    registry = pd.DataFrame.from_records(
        [
            {
                "method_id": "ef31",
                "method_version": "3.1",
                "impact_category_id": "climate_change",
                "impact_category_tuple_json": '["EF v3.1","climate change"]',
                "unit": "kg CO2-Eq",
                "factor_identity": "factor::climate",
            }
        ]
    )
    scores = ResultScores(
        physical_candidate_identity=physical,
        result_scoring_identity="scoring",
        backgrounds=("remind_ssp2_ndc",),
        regions=regions,
        years=(2050,),
        method_registry=registry,
        annual_scores=regional_scores,
        annual_contribution_norms=norms,
        score_identity="score",
    )
    characterization = SelectedCandidateCharacterization(
        identity="characterization",
        selected_result_identity=selected.selected_result.identity,
        result_lcia_selection_identity="result-selection",
        scores=scores,
    )

    table = _environmental_table((characterization,), {physical: selected})
    global_row = table.loc[table["region"].eq("global")].iloc[0]

    assert global_row["score"] == 1.0
    assert global_row["contribution_norm"] == 2.0e16

    second_physical = f"{physical}::second"
    second = replace(
        characterization,
        identity="characterization::second",
        scores=replace(scores, physical_candidate_identity=second_physical),
    )
    combined = _environmental_table(
        (characterization, second),
        {physical: selected, second_physical: selected},
    )
    separate = pd.concat(
        (
            table,
            _environmental_table((second,), {second_physical: selected}),
        ),
        ignore_index=True,
    ).sort_values(
        [
            "physical_candidate_identity",
            "target_year",
            "background_scenario",
            "region",
            "year",
            "method_id",
            "impact_category_id",
        ],
        kind="stable",
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(combined, separate, check_exact=True)
    assert canonical_table_sha256(combined) == canonical_table_sha256(separate)


def test_acc_table_contains_real_member_rows_and_marks_selection_coordinates() -> None:
    selected = _selected(_integrated_candidate())

    table = build_acc_table(
        {selected.selected_result.result.physical.physical_candidate_identity: 2050},
        _acc_index(selected),
    )
    assert set(table["background_scenario"]) == {"remind_ssp2_ndc"}
    assert set(table["projection_kind"]) == {"member"}
    regional = table.loc[table["geography"].ne("global")]
    global_rows = table.loc[table["geography"].eq("global")]
    history = regional.loc[regional["year"].eq(2019)]
    controlled = regional.loc[regional["year"].ge(2020)]
    assert len(history) == 12
    assert not history["controlled_for_selection"].any()
    assert history["selection_compliant"].isna().all()
    assert controlled["controlled_for_selection"].all()
    assert controlled["selection_compliant"].all()
    assert not global_rows["controlled_for_selection"].any()
    assert global_rows["selection_compliant"].isna().all()


def test_acc_table_deduplicates_identical_block_metric_scopes() -> None:
    selected = _selected(_integrated_candidate())
    target_years = {selected.selected_result.result.physical.physical_candidate_identity: 2050}
    original = build_acc_table(target_years, _acc_index(selected))
    observed = build_acc_table(target_years, _acc_index(selected, copies=3))

    assert len(observed) == len(original)
    assert observed.equals(original)
    assert tuple(str(dtype) for dtype in observed.dtypes) == tuple(
        str(dtype) for dtype in original.dtypes
    )
    assert canonical_table_sha256(observed) == canonical_table_sha256(original)


@pytest.mark.parametrize("endpoint_steps", ((10, 5), (10, 5, 2, 1)))
def test_method_summary_and_selected_result_identity_are_exact(
    endpoint_steps: tuple[int, ...],
) -> None:
    products = SearchExecutionProducts(
        search=RequestSearchResult(
            identity="search",
            block_results=(),
            endpoint_refinement_steps_pct=endpoint_steps,
            candidate_input_identity="candidate-input",
            continuation_parent_identity=(
                "completed-five-point-result" if endpoint_steps[-1] == 1 else None
            ),
        ),
        selected_2050=(),
        selected_timing=(),
        physical_repair_elapsed_seconds=0.0,
        environmental_evaluation_elapsed_seconds=0.0,
        mfa_evaluation_count=0,
        stock_lp_count=0,
        environmental_evaluation_count=0,
        repair_status_counts=(),
        worker_result_set_identity="worker-result-set",
        request_coordinate_count=120,
        request_unique_physical_candidate_count=100,
        request_physical_candidate_deduplication_count=20,
    )

    summary = _result_summary(
        products,
        selected_coordinate_count=6,
        selected_unique_physical_candidate_count=4,
        states=(),
        target_years=(2050, 2075, 2100),
        table_identities={"selected_coordinates.parquet": "a" * 64},
    )

    assert summary == {
        "selected_coordinate_count": 6,
        "selected_unique_physical_candidate_count": 4,
        "state_count": 0,
        "declared_neighborhood_verified_state_count": 0,
        "parent_terminal_state_count": 0,
        "not_declared_neighborhood_verified_state_count": 0,
        "request_coordinate_count": 120,
        "request_physical_repair_elapsed_seconds": 0.0,
        "request_environmental_evaluation_elapsed_seconds": 0.0,
        "request_mfa_evaluation_count": 0,
        "request_stock_lp_count": 0,
        "request_environmental_evaluation_count": 0,
        "work_accounting": {
            "request_totals": "embodied work in all unique completed candidate results",
            "elapsed_values": "summed worker effort, not wall-clock time",
            "newly_computed_worker_result_count": 0,
            "reused_worker_result_count": 0,
            "incremental_physical_repair_elapsed_seconds": 0.0,
            "incremental_environmental_evaluation_elapsed_seconds": 0.0,
            "incremental_mfa_evaluation_count": 0,
            "incremental_stock_lp_count": 0,
        },
        "request_repair_status_counts": {},
        "request_unique_physical_candidate_count": 100,
        "request_physical_candidate_deduplication_count": 20,
        "refinement": [],
        "target_years": [2050, 2075, 2100],
        "search_method": "deterministic_candidate_search",
        "refinement_method": "pytoce_candidate_refinement_v12",
        "integrated_search_method": "pytoce_integrated_domain_search_v3",
        "completed_search": None,
        "selection_method": "pytoce_candidate_selection_v3",
        "candidate_input_identity": "candidate-input",
        "continuation_parent_identity": "completed-five-point-result"
        if endpoint_steps[-1] == 1
        else None,
        "initial_grid_step_pct": 20,
        "endpoint_refinement_steps_pct": list(endpoint_steps),
        "endpoint_resolution_pct": endpoint_steps[-1],
        "annual_control_resolution_pct": 1,
        "annual_control_reference": "linear_first_demand_only_repair",
        "annual_control_years": list(range(2019, 2101)),
        "conditional_stock_proposal_method": "pytoce_annual_stock_proposal_v6",
        "global_optimum_claim": False,
        "table_identities": {"selected_coordinates.parquet": "a" * 64},
    }
    identity = _selected_result_identity(
        "pytoce_selected_results_v13",
        {
            "worker_result_set": "worker-result-set",
            "candidate_search": "search",
        },
        summary,
    )
    assert identity.startswith("pytoce_selected_results_v13::")
    assert identity != _selected_result_identity(
        "pytoce_selected_results_v13",
        {"worker_result_set": "worker-result-set", "candidate_search": "search"},
        {**summary, "request_physical_repair_elapsed_seconds": 1.0},
    )


def test_refinement_summary_labels_only_the_moves_counted_in_each_pass() -> None:
    """Readers can distinguish full lines from additional coordinated joint moves."""
    products = _search_products(_selected(_integrated_candidate()), ())
    original = products.search.block_results[0]
    blocks = []
    for block, coordinated_modes in (
        ("demand_lifetime_frontier", (False, True)),
        ("circularity_feedstock_anchor", (False,)),
        ("integrated_frontier", (False, True)),
    ):
        passes = tuple(
            RefinementPass(5, 1, (), (), (), (), (), coordinated=mode) for mode in coordinated_modes
        )
        blocks.append(
            replace(
                original,
                optimization_block=block,
                refinement=replace(original.refinement, optimization_block=block, passes=passes),
            )
        )
    products = replace(
        products,
        search=replace(
            products.search, block_results=tuple(blocks), endpoint_refinement_steps_pct=(5,)
        ),
    )

    summary = _result_summary(
        products,
        selected_coordinate_count=1,
        selected_unique_physical_candidate_count=1,
        states=(),
        target_years=(2050,),
        table_identities={},
    )

    observed = {
        block["optimization_block"]: tuple(
            (item["proposal_family"], item["pass_count"])
            for item in block["refinement_pass_counts"]
        )
        for block in summary["refinement"]
    }
    assert observed == {
        "demand_lifetime_frontier": (("scalar", 1), ("scalar_and_coordinated", 1)),
        "circularity_feedstock_anchor": (("complete_lines", 1),),
        "integrated_frontier": (("complete_lines", 1), ("coordinated_and_guided", 1)),
    }
