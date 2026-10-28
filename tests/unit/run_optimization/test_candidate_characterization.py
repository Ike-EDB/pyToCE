"""Characterize selected candidates against their exact parent scores."""

from candidate_characterization_fixtures import (
    _BACKGROUNDS,
    _empty_pa_scoring,
    _method_registry,
    _pa_scores_at_2050,
    _physical,
    _physical_parent,
    _result_projection,
    _score_parent,
    pd,
)

from pytoce.run_optimization.calculation.candidate_evaluation import (
    characterize_selected_physical_candidates,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    build_response_parent_score_projection,
)
from tests.unit.run_optimization.response_projection_fixtures import _response_parent


def test_selected_characterization_applies_changes_and_sorts_candidates() -> None:
    """Characterize each selected MFA state once against a shared exact parent."""

    methods = _method_registry()
    pa_projection, pa_parent = _empty_pa_scoring(methods)
    parent = _physical_parent()
    score_parent = _score_parent(methods, parent)
    projection = _result_projection(methods)
    changed_delta = pd.DataFrame.from_records(
        [
            {
                "year": 2050,
                "region": "EUR",
                "provider_service_id": "service",
                "provider_service_code": "service-code",
                "parent_amount_kg": 3.0,
                "candidate_amount_kg": 0.0,
                "delta_amount_kg": -3.0,
            }
        ]
    )
    changed = _physical("z-changed", delta=changed_delta)
    baseline = _physical("a-baseline", delta=pd.DataFrame())

    observed = characterize_selected_physical_candidates(
        (("selected::z", changed, pa_parent), ("selected::a", baseline, pa_parent)),
        result_lcia_selection_identity="selection::ef31",
        physical_parent=parent,
        result_parent=score_parent,
        result_projection=projection,
        pa_parent=pa_parent,
        pa_method_registry=pa_projection.method_registry,
    )

    assert tuple(item.scores.physical_candidate_identity for item in observed) == (
        "candidate::a-baseline",
        "candidate::z-changed",
    )
    changed_scores = observed[1].scores
    background_i = changed_scores.backgrounds.index(_BACKGROUNDS[0])
    region_i = changed_scores.regions.index("EUR")
    year_i = changed_scores.years.index(2050)
    assert changed_scores.annual_scores[background_i, region_i, year_i, 0] == 0.0
    assert changed_scores.annual_contribution_norms[background_i, region_i, year_i, 0] == 0.0
    assert (observed[0].scores.annual_scores == 6.0).all()
    assert (observed[0].scores.annual_contribution_norms == 9.0).all()
    assert not changed_scores.annual_scores.flags.writeable
    assert observed[0].identity != observed[1].identity


def test_selected_pa_difference_uses_saved_scores_without_recomputing_receiving() -> None:
    """Report unchanged and changed cached PA endpoints with one exact shared parent."""
    methods = _method_registry()
    physical_parent = _response_parent()
    pa_projection, _ = _empty_pa_scoring(methods)
    pa_parent = _pa_scores_at_2050(methods, scores=(-2.0,), norms=(7.0,))
    candidate = _pa_scores_at_2050(methods, scores=(1.0,), norms=(3.0,))
    # Use the matching provider fixture to calculate the fixed + PA parent;
    # selected characterization receives only already-calculated PA records.
    from tests.unit.run_optimization.response_projection_fixtures import _response_projection

    projection = _response_projection("result")
    result_parent = build_response_parent_score_projection(
        physical_parent, projection, pa_parent=pa_parent, pa_method_registry=methods
    )
    observed = characterize_selected_physical_candidates(
        (
            ("selected::changed", _physical("changed", delta=pd.DataFrame()), candidate),
            ("selected::parent", _physical("parent", delta=pd.DataFrame()), pa_parent),
        ),
        result_lcia_selection_identity="selected::fixture",
        physical_parent=physical_parent,
        result_parent=result_parent,
        result_projection=projection,
        pa_parent=pa_parent,
        pa_method_registry=pa_projection.method_registry,
    )
    changed, parent = (item.scores for item in observed)
    eur, year = changed.regions.index("EUR"), changed.years.index(2050)
    assert (changed.annual_scores[:, eur, year, 0] == 1.0).all()
    assert (changed.annual_contribution_norms[:, eur, year, 0] == 3.0).all()
    assert (parent.annual_scores[:, eur, year, 0] == -2.0).all()
    assert (parent.annual_contribution_norms[:, eur, year, 0] == 7.0).all()
    assert (changed.annual_scores[:, eur, 0, 0] == 2.0).all()
