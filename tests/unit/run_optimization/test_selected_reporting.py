"""Selected-result reporting through complete semantic fixtures."""

from dataclasses import replace
from math import fsum

import pytest
from candidate_characterization_fixtures import _empty_pa_scoring, _pa_scores_at_2050
from candidate_worker_fixtures import _result_and_acc_metrics, _summary
from selected_reporting_fixtures import (
    BACKGROUNDS,
    METHOD,
    STAGE,
    Path,
    _baseline_lca_result,
    _candidate_evaluation,
    _parents,
    _projection,
    _states,
    _stream_stage,
    _with_pa_parent,
    canonical_table_sha256,
    pd,
)

from pytoce.run_optimization.calculation.selected_reporting import (
    _METHOD_COLUMNS,
    _STAGE_COLUMNS,
    _stage_response_adjustments,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    _selected_candidate_result,
    decode_candidate_worker_result,
    encode_candidate_worker_result,
)


@pytest.mark.parametrize("backgrounds", (BACKGROUNDS, ("remind_ssp2_npi", "remind_ssp2_pkbudg650")))
def test_selected_stage_reporting_covers_each_background_member(
    tmp_path: Path, backgrounds: tuple[str, ...]
) -> None:
    background = backgrounds[0]
    evaluation = _candidate_evaluation(candidate_amount=2.0)
    states = _states(
        background,
        physical_candidate_identity=evaluation.result.physical.physical_candidate_identity,
    )

    stage, _stage_hash = _stream_stage(
        tmp_path / "object_parents",
        selected_evaluations=(evaluation,),
        states=states,
        baseline_lca_results=tuple(
            parent
            for parent in _parents((0.25, -1.0, 0.0, 1.0))
            if parent.record.background_scenario in backgrounds
        ),
        response_projection=_projection((0.25, -1.0, 0.0, 1.0)),
        pa_parent=_empty_pa_scoring(_projection((0.25, -1.0, 0.0, 1.0)).method_registry)[1],
    )

    assert not stage.empty
    assert set(stage["characterization_background_scenario"]) == set(backgrounds)
    assert "lifetime_extension_pct" not in stage


def test_response_adjustments_bound_each_exact_method_and_preserve_predecessor() -> None:
    """Aggregate two methods independently with positive material requirements."""
    second_method = {
        **METHOD,
        "impact_category_id": "acidification",
        "impact_category_tuple_json": '["EF v3.1","acidification"]',
        "unit": "mol H+-Eq",
        "factor_identity": "b" * 64,
    }
    # Three services increase from 1 to 2 kg. Each retains its characterized inventory and impact
    # sign.
    selected = pd.concat(
        [
            _candidate_evaluation(candidate_amount=2.0).result.physical.activation_delta.assign(
                provider_service_id=f"service::{index}",
                provider_service_code=f"service_{index}",
            )
            for index in range(3)
        ],
        ignore_index=True,
    )
    method_scores = (
        (second_method, (-1.0e16, 2.0, 1.0e16)),
        (METHOD, (1.0e16, 1.0, -1.0e16)),
    )
    effective = pd.DataFrame.from_records(
        [
            {
                "background_scenario": BACKGROUNDS[0],
                "occurrence_year": 2020,
                "provider_service_id": f"service::{index}",
                "provider_service_code": f"service_{index}",
                **method,
                "effective_unit_score": score,
                "effective_unit_norm": abs(score),
            }
            for method, scores in method_scores
            for index, score in enumerate(scores)
        ]
    )
    selected_before = selected.copy(deep=True)
    effective_before = effective.copy(deep=True)
    # An independent scalar oracle uses the known 1 kg parent, 2 kg candidate
    # and 1 kg change, retaining the small residual between opposing scores.
    stage_keys = (
        "characterization_background_scenario",
        "region",
        "year",
        *_METHOD_COLUMNS,
        *_STAGE_COLUMNS,
        "physical_event_year",
    )
    values = (
        "score_adjustment",
        "norm_adjustment",
        "changed_parent_score",
        "changed_candidate_score",
        "changed_parent_norm",
        "changed_candidate_norm",
    )
    expected_stage = pd.DataFrame.from_records(
        [
            {
                "characterization_background_scenario": BACKGROUNDS[0],
                "region": "EUR",
                "year": 2020,
                **method,
                **STAGE,
                "physical_event_year": 2020,
                "score_adjustment": fsum(scores),
                "norm_adjustment": fsum(map(abs, scores)),
                "changed_parent_score": fsum(scores),
                "changed_candidate_score": fsum(2.0 * value for value in scores),
                "changed_parent_norm": fsum(map(abs, scores)),
                "changed_candidate_norm": fsum(2.0 * abs(value) for value in scores),
            }
            for method, scores in method_scores
        ],
        columns=[*stage_keys, *values],
    ).sort_values(list(stage_keys), ignore_index=True)
    observed_stage = _stage_response_adjustments(selected, effective, background=BACKGROUNDS[0])
    pd.testing.assert_frame_equal(observed_stage, expected_stage, check_exact=True)
    pd.testing.assert_frame_equal(selected, selected_before, check_exact=True)
    pd.testing.assert_frame_equal(effective, effective_before, check_exact=True)
    assert canonical_table_sha256(observed_stage) == canonical_table_sha256(expected_stage)


def test_selected_reporting_retains_new_stage_coordinates(tmp_path: Path) -> None:
    """A newly activated recycling stage survives annual, cumulative and global writing."""
    original = _candidate_evaluation(candidate_amount=0.0).result.physical.activation_delta
    assert original is not None
    newly_recycled = original.copy()
    newly_recycled["year"] = 2022
    newly_recycled["physical_event_year"] = 2022
    newly_recycled["life_cycle_stage"] = "mechanical_recycling_operation"
    newly_recycled["stage_group"] = "recycling"
    newly_recycled["cutoff_cycle_role"] = "secondary_material_cycle_entry"
    newly_recycled["material_cycle_source"] = "mechanical_recycling_input"
    newly_recycled["physical_event_basis"] = "mechanical_recycling_input"
    newly_recycled["parent_amount_kg"] = 0.0
    newly_recycled["candidate_amount_kg"] = 2.0
    newly_recycled["delta_amount_kg"] = 2.0
    delta = pd.concat((original, newly_recycled), ignore_index=True)
    result = decode_candidate_worker_result(
        encode_candidate_worker_result(_result_and_acc_metrics(activation_delta=delta)[0])
    )
    selected = _selected_candidate_result(
        result,
        _summary(result),
        selection_identities=("selection::new_recycling",),
    )
    unit_scores = (0.25, -1.0, 0.0, 1.0)
    stage, _ = _stream_stage(
        tmp_path,
        selected_evaluations=(selected,),
        states=_states(
            BACKGROUNDS[0],
            physical_candidate_identity=result.physical.physical_candidate_identity,
        ),
        baseline_lca_results=_parents(unit_scores),
        response_projection=_projection(unit_scores),
        pa_parent=_empty_pa_scoring(_projection(unit_scores).method_registry)[1],
    )
    recycling = stage.loc[stage["life_cycle_stage"].eq("mechanical_recycling_operation")]
    # The input is two kg in EUR only. Each selected background must retain
    # its own response, including an explicit zero, and derive the world once.
    assert len(recycling) == 4 * 2 * 2
    assert set(recycling["region"]) == {"EUR", "global_derived"}
    for background, unit_score in zip(BACKGROUNDS, unit_scores, strict=True):
        member = recycling.loc[recycling["characterization_background_scenario"].eq(background)]
        assert member["score"].eq(2.0 * unit_score).all()
        assert member["contribution_norm"].eq(6.0).all()
        annual = member.loc[member["period_kind"].eq("annual")]
        assert annual["year"].eq(2022).all()
        assert annual["physical_event_year"].eq(2022).all()
        cumulative = member.loc[member["period_kind"].eq("cumulative_2019_2100")]
        assert cumulative["year"].isna().all()
        assert cumulative["physical_event_year"].isna().all()
    removed = stage.loc[
        stage["life_cycle_stage"].eq("production")
        & stage["year"].eq(2020)
        & stage["region"].eq("EUR")
    ]
    assert len(removed) == 4
    assert removed[["score", "contribution_norm"]].eq(0.0).all().all()
    untouched = stage.loc[
        stage["life_cycle_stage"].eq("production")
        & stage["year"].eq(2021)
        & stage["region"].eq("CHA")
    ]
    assert len(untouched) == 4
    assert untouched[["score", "contribution_norm"]].eq(1.0).all().all()


def test_selected_pa_stage_addition_and_removal_survive_serialized_reporting(
    tmp_path: Path,
) -> None:
    """A PA-only change retains removed and new stages in each selected background scenario."""
    projection = _projection((0.25, -1.0, 0.0, 1.0))
    pa_parent = _pa_scores_at_2050(projection.method_registry, scores=(3.0,), norms=(5.0,))
    pa_candidate = _pa_scores_at_2050(projection.method_registry, scores=(-2.0,), norms=(7.0,))
    candidate_stages = pa_candidate.stage_scores.copy()
    candidate_stages["life_cycle_stage"] = "landfill_treatment"
    candidate_stages["physical_event_basis"] = "year collected end-of-life waste is landfilled"
    pa_candidate = replace(pa_candidate, stage_scores=candidate_stages)
    unchanged_delta = _candidate_evaluation(candidate_amount=1.0).result.physical.activation_delta
    assert unchanged_delta is not None
    # No fixed-service demand changes: both stage adjustments come only from
    # saved actual PA responses, including an absent candidate incineration row.
    result, _ = _result_and_acc_metrics(
        activation_delta=unchanged_delta.iloc[0:0], pa_scores=pa_candidate
    )
    result = decode_candidate_worker_result(encode_candidate_worker_result(result))
    selected = _selected_candidate_result(
        result,
        _summary(result),
        selection_identities=("selection::pa_terminal",),
    )
    stage, _ = _stream_stage(
        tmp_path,
        selected_evaluations=(selected,),
        states=_states(
            BACKGROUNDS[0],
            physical_candidate_identity=result.physical.physical_candidate_identity,
        ),
        baseline_lca_results=tuple(
            _with_pa_parent(parent, pa_parent) for parent in _parents((0.25, -1.0, 0.0, 1.0))
        ),
        response_projection=projection,
        pa_parent=pa_parent,
    )
    for name, score, norm in (
        ("incineration_treatment", 0.0, 0.0),
        ("landfill_treatment", -2.0, 7.0),
    ):
        terminal = stage.loc[stage.life_cycle_stage.eq(name)]
        assert len(terminal) == 4 * 2 * 2
        assert set(terminal.characterization_background_scenario) == set(BACKGROUNDS)
        assert set(terminal.region) == {"EUR", "global_derived"}
        assert terminal.score.eq(score).all()
        assert terminal.contribution_norm.eq(norm).all()
        annual = terminal.loc[terminal.period_kind.eq("annual")]
        assert annual.year.eq(2050).all()
        assert annual.physical_event_year.eq(2050).all()
        cumulative = terminal.loc[terminal.period_kind.eq("cumulative_2019_2100")]
        assert cumulative.year.isna().all()
        assert cumulative.physical_event_year.isna().all()


def test_selected_guard_uses_combined_fixed_and_pa_endpoints(tmp_path: Path) -> None:
    """A complete combined parent recovers the small selected score after cancellation."""
    projection = _projection((1.0, 1.0, 1.0, 1.0), unit_contribution_norm=1.0)
    pa_parent = _pa_scores_at_2050(projection.method_registry, scores=(1.0e16,), norms=(1.0e16,))
    pa_candidate = _pa_scores_at_2050(projection.method_registry, scores=(2.0,), norms=(2.0,))
    pa_parent = replace(pa_parent, stage_scores=pa_parent.stage_scores.assign(**STAGE))
    pa_candidate = replace(pa_candidate, stage_scores=pa_candidate.stage_scores.assign(**STAGE))
    delta = _candidate_evaluation(candidate_amount=1.0).result.physical.activation_delta
    assert delta is not None
    delta = delta.assign(
        year=2050,
        physical_event_year=2050,
        parent_amount_kg=1.0e16,
        candidate_amount_kg=1.0,
        delta_amount_kg=1.0 - 1.0e16,
    )
    result, _ = _result_and_acc_metrics(activation_delta=delta, pa_scores=pa_candidate)
    result = decode_candidate_worker_result(encode_candidate_worker_result(result))
    selected = _selected_candidate_result(
        result,
        _summary(result),
        selection_identities=("selection::combined_endpoint",),
    )
    parents = []
    for background in BACKGROUNDS:
        fixed = _baseline_lca_result(background, parent_score=0.0, parent_norm=0.0)
        annual = fixed.annual_life_cycle_stage_scores.copy()
        annual.loc[
            annual.region.eq("EUR") & annual.year.eq(2050), ["score", "contribution_norm"]
        ] = 1.0e16
        parents.append(
            _with_pa_parent(replace(fixed, annual_life_cycle_stage_scores=annual), pa_parent)
        )
    stage, _ = _stream_stage(
        tmp_path,
        selected_evaluations=(selected,),
        states=_states(
            BACKGROUNDS[0],
            physical_candidate_identity=result.physical.physical_candidate_identity,
        ),
        baseline_lca_results=parents,
        response_projection=projection,
        pa_parent=pa_parent,
    )
    selected_year = stage.loc[
        stage.life_cycle_stage.eq("production")
        & stage.period_kind.eq("annual")
        & stage.year.eq(2050)
        & stage.region.isin(("EUR", "global_derived"))
    ]
    # Combined parent score/norm is 2e16, while the selected endpoints are 1+2.
    # Applying the guard to either component separately cannot recover this 3.
    assert len(selected_year) == 8
    assert selected_year.score.eq(3.0).all()
    assert selected_year.contribution_norm.eq(3.0).all()
