"""Search and selected totals preserve the same complete receiving endpoints."""

from dataclasses import replace

import pandas as pd
import pytest
from candidate_characterization_fixtures import (
    _empty_pa_scoring,
    _pa_scores_at_2050,
    _physical,
    _physical_parent,
)
from selected_reporting_fixtures import BACKGROUNDS, STAGE, _projection

from pytoce.run_lca.calculation.foreground import H12_REGIONS, optimization_activation_delta
from pytoce.run_optimization.calculation.candidate_evaluation import (
    characterize_selected_physical_candidates,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    _optimization_scores,
    build_response_parent_score_projection,
    prepare_search_scoring,
)
from pytoce.run_optimization.calculation.selected_reporting import _effective_responses
from pytoce.run_optimization.calculation.selected_stage_reporting import (
    _physical_reporting_component,
)


@pytest.mark.parametrize("backgrounds", (BACKGROUNDS, ("remind_ssp2_npi", "remind_ssp2_pkbudg650")))
@pytest.mark.parametrize(
    ("fixed_unit_score", "unchanged"),
    ((1.0, False), (-1.0, False), (1.0, True)),
    ids=("positive-unit-score", "negative-unit-score", "unchanged-fixed-only"),
)
def test_complete_endpoint_scores_agree_in_search_annual_and_stage(
    fixed_unit_score: float, unchanged: bool, backgrounds: tuple[str, ...]
) -> None:
    """Recover small complete endpoints after subtracting large parent scores.

    These characterized quantities test numerical contraction. A fixed supplier has unit score +1 or
    -1
    and unit norm 1; its positive demand changes from 1e16 to 1. PA changes
    from score/norm 1e16 to 2. The independent endpoint sums are therefore
    score 3 or 1 and norm 3. Service demands remain positive; native production signs stay inside
    the supplier.
    The unchanged case preserves a four-unit fixed parent when both the actual
    PA stream and fixed-service delta are empty, as in baseline selection.
    """
    projection = _projection(
        (fixed_unit_score,) * len(backgrounds), unit_contribution_norm=1.0, backgrounds=backgrounds
    )
    pa_projection, empty_pa = _empty_pa_scoring(projection.method_registry, backgrounds=backgrounds)
    if unchanged:
        pa_parent = pa_candidate = empty_pa
    else:
        pa_parent = _pa_scores_at_2050(
            projection.method_registry, scores=(1e16,), norms=(1e16,), backgrounds=backgrounds
        )
        pa_candidate = _pa_scores_at_2050(
            projection.method_registry, scores=(2.0,), norms=(2.0,), backgrounds=backgrounds
        )
        pa_parent = replace(pa_parent, stage_scores=pa_parent.stage_scores.assign(**STAGE))
        pa_candidate = replace(pa_candidate, stage_scores=pa_candidate.stage_scores.assign(**STAGE))
    parent_fixed = 4.0 if unchanged else 1e16
    candidate_fixed = parent_fixed if unchanged else 1.0
    coordinate = {
        "region": "EUR",
        "year": 2050,
        "physical_event_year": 2050,
        "provider_service_id": "service::one",
        "provider_service_code": "service_one",
        **STAGE,
    }
    physical_parent = replace(
        _physical_parent(),
        activations=pd.DataFrame([{**coordinate, "amount": parent_fixed}]).rename(
            columns={"life_cycle_stage": "demand_type"}
        ),
    )
    delta = optimization_activation_delta(
        physical_parent.activations, physical_parent.activations.assign(amount=candidate_fixed)
    )
    parent = build_response_parent_score_projection(
        physical_parent,
        projection,
        pa_parent=pa_parent,
        pa_method_registry=projection.method_registry,
    )
    optimization = replace(projection, role="optimization")
    climate = replace(projection, role="mandatory_climate")
    scoring = prepare_search_scoring(
        physical_parent=physical_parent,
        optimization_parent=build_response_parent_score_projection(
            physical_parent,
            optimization,
            pa_parent=pa_parent,
            pa_method_registry=projection.method_registry,
        ),
        climate_parent=build_response_parent_score_projection(
            physical_parent,
            climate,
            pa_parent=pa_parent,
            pa_method_registry=projection.method_registry,
        ),
        optimization_projection=optimization,
        climate_projection=climate,
        pa_projection=pa_projection,
        pa_parent=pa_parent,
    )
    physical = _physical("complete-endpoint", delta=delta)
    search = _optimization_scores(physical, scoring, delta, pa_candidate)
    selected = characterize_selected_physical_candidates(
        (("selection::complete-endpoint", physical, pa_candidate),),
        result_lcia_selection_identity="selection::ef31",
        physical_parent=physical_parent,
        result_parent=parent,
        result_projection=projection,
        pa_parent=pa_parent,
        pa_method_registry=projection.method_registry,
    )[0].scores
    assert search.backgrounds == selected.backgrounds == backgrounds
    methods, effective = _effective_responses(projection)
    stage_scores = []
    stage_norms = []
    for background in search.backgrounds:
        # All other parent coordinates are zero. Preserve this exact sparse
        # annual coordinate and its corresponding cumulative parent.
        annual = parent.annual_scores.loc[
            parent.annual_scores.background_scenario.eq(background)
            & parent.annual_scores.region.eq("EUR")
            & parent.annual_scores.year.eq(2050)
        ].rename(columns={"background_scenario": "characterization_background_scenario"})
        annual = annual.assign(**STAGE, physical_event_year=2050)
        cumulative = annual.copy()
        cumulative["year"] = pd.NA
        cumulative["physical_event_year"] = pd.NA
        stage = _physical_reporting_component(
            candidate_identity=physical.physical_candidate_identity,
            target_year=2050,
            background=background,
            annual_stage_parent=annual,
            cumulative_stage_parent=cumulative,
            selected_delta=delta,
            effective=effective,
            pa_parent=pa_parent.stage_scores,
            pa_candidate=pa_candidate.stage_scores,
            methods=methods,
        )
        regional = stage.annual_life_cycle_stage_scores.loc[
            stage.annual_life_cycle_stage_scores.region.eq("EUR")
        ]
        stage_scores.append(float(regional.score.iloc[0]))
        stage_norms.append(float(regional.contribution_norm.iloc[0]))
    eur, year = H12_REGIONS.index("EUR"), search.years.index(2050)
    expected_score = 4.0 if unchanged else fixed_unit_score + 2.0
    expected_norm = 4.0 if unchanged else 3.0
    # The small complete endpoint must survive subtraction in every consumer,
    # including annual totals used for selection and the reported stage totals.
    assert search.annual_scores[:, eur, year, 0].tolist() == [expected_score] * len(backgrounds)
    assert search.annual_climate_scores[:, eur, year].tolist() == [expected_score] * len(
        backgrounds
    )
    assert (
        selected.annual_scores[:, eur, year, 0].tolist()
        == stage_scores
        == [expected_score] * len(backgrounds)
    )
    assert (
        selected.annual_contribution_norms[:, eur, year, 0].tolist()
        == stage_norms
        == [expected_norm] * len(backgrounds)
    )
