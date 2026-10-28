"""Baseline response projection and region ordering for allocated carrying capacities."""

import numpy as np
import pandas as pd
import pytest
from candidate_characterization_fixtures import (
    _empty_pa_scoring,
    _pa_scores_at_2050,
    _physical,
)

from pytoce.run_lca.calculation.foreground import H12_REGIONS, optimization_activation_delta
from pytoce.run_optimization.calculation.candidate_acc import prepare_candidate_acc_context
from pytoce.run_optimization.calculation.candidate_projection import (
    _optimization_scores,
    build_response_parent_score_projection,
    prepare_search_scoring,
)
from pytoce.run_optimization.calculation.candidate_projection_arrays import (
    _group_delta,
    _projection_baseline,
    _score_arrays,
)
from tests.unit.run_optimization.response_projection_fixtures import (
    _BACKGROUNDS,
    _REGIONS,
    _RESPONSE_YEARS,
    _allocated_cc_selection,
    _response_parent,
    _response_projection,
)


def test_parent_response_projection_uses_baseline_activation() -> None:
    """Prepare score arrays from the baseline activation data."""

    physical_parent = _response_parent()
    optimization_projection = _response_projection("optimization")
    climate_projection = _response_projection("mandatory_climate")
    pa_projection, pa_parent = _empty_pa_scoring(optimization_projection.method_registry)
    optimization_parent = build_response_parent_score_projection(
        physical_parent,
        optimization_projection,
        pa_parent=pa_parent,
        pa_method_registry=pa_projection.method_registry,
    )
    climate_parent = build_response_parent_score_projection(
        physical_parent,
        climate_projection,
        pa_parent=pa_parent,
        pa_method_registry=pa_projection.method_registry,
    )

    context = prepare_search_scoring(
        physical_parent=physical_parent,
        optimization_parent=optimization_parent,
        climate_parent=climate_parent,
        optimization_projection=optimization_projection,
        climate_projection=climate_projection,
        pa_projection=pa_projection,
        pa_parent=pa_parent,
    )

    expected_shape = (len(_BACKGROUNDS), len(_REGIONS), len(_RESPONSE_YEARS), 1)
    assert context.optimization_baseline_scores.shape == expected_shape
    assert context.climate_baseline_scores.shape == expected_shape
    assert np.count_nonzero(context.optimization_baseline_scores) == len(_BACKGROUNDS)
    region_index = H12_REGIONS.index("EUR")
    np.testing.assert_array_equal(
        context.optimization_baseline_scores[:, region_index, 0, 0],
        np.full(len(_BACKGROUNDS), 2.0),
    )
    baseline_scores, baseline_norms = _projection_baseline(
        optimization_parent,
        context.optimization_projection,
    )
    candidate = physical_parent.activations.copy(deep=True)
    candidate["amount"] = 2.0
    delta = optimization_activation_delta(physical_parent.activations, candidate)
    changed_scores, changed_norms = _score_arrays(
        baseline_scores,
        baseline_norms,
        context.optimization_projection,
        delta,
        pa_parent_scores=pa_parent.annual_scores,
        pa_parent_norms=pa_parent.annual_contribution_norms,
        pa_candidate_scores=pa_parent.annual_scores,
        pa_candidate_norms=pa_parent.annual_contribution_norms,
    )
    empty_scores, empty_norms = _score_arrays(
        baseline_scores,
        baseline_norms,
        context.optimization_projection,
        pd.DataFrame(),
        pa_parent_scores=pa_parent.annual_scores,
        pa_parent_norms=pa_parent.annual_contribution_norms,
        pa_candidate_scores=pa_parent.annual_scores,
        pa_candidate_norms=pa_parent.annual_contribution_norms,
    )
    np.testing.assert_array_equal(empty_scores, baseline_scores)
    np.testing.assert_array_equal(empty_norms, baseline_norms)
    duplicate_delta = pd.concat((delta, delta), ignore_index=True)
    grouped_delta = _group_delta(duplicate_delta, context.optimization_projection.services)[0]
    np.testing.assert_array_equal(
        grouped_delta[region_index, 0],
        np.array([2.0]),
    )
    np.testing.assert_array_equal(
        changed_scores[:, region_index, 0, 0],
        np.full(len(_BACKGROUNDS), 4.0),
    )
    np.testing.assert_array_equal(
        changed_norms[:, region_index, 0, 0],
        np.full(len(_BACKGROUNDS), 4.0),
    )


def test_allocated_cc_context_reorders_regions_without_interpreting_case_names() -> None:
    """Reorder aCC arrays while retaining the supplied case names and order."""

    selection = _allocated_cc_selection()
    context = prepare_candidate_acc_context(selection)

    assert context.regions == H12_REGIONS
    assert context.allocated_cc_cases == ("lower_values", "higher_values")
    source_region_index = _REGIONS.index("CAZ")
    np.testing.assert_array_equal(
        context.annual_allocated_cc[:, 0],
        selection.annual_allocated_cc[:, source_region_index],
    )
    np.testing.assert_array_equal(
        context.cumulative_allocated_cc[:, 0],
        selection.cumulative_allocated_cc[:, source_region_index],
    )


@pytest.mark.parametrize("removed", (False, True))
def test_pa_only_change_uses_full_method_order_and_actual_parent_norm(removed: bool) -> None:
    """Apply cached PA changes despite an empty fixed ledger, including complete removal."""
    physical_parent = _response_parent()
    optimization = _response_projection("optimization")
    climate = _response_projection("mandatory_climate")
    # A separate reporting method precedes the selected climate method in the
    # full bank. Its deliberately different score/unit exposes index-based
    # selection instead of exact method-signature selection.
    other = optimization.method_registry.assign(
        lcia_method="reporting_fixture",
        method_id="reporting_fixture",
        method_version="1",
        impact_category_id="other",
        impact_category_tuple_json='["fixture","other reporting impact"]',
        unit="other impact unit",
        factor_identity="b" * 64,
    )
    methods = pd.concat((other, optimization.method_registry), ignore_index=True)
    pa_projection, empty = _empty_pa_scoring(methods)
    pa_parent = _pa_scores_at_2050(methods, scores=(1000.0, -3.0), norms=(1001.0, 7.0))
    candidate = (
        empty if removed else _pa_scores_at_2050(methods, scores=(300.0, 2.0), norms=(301.0, 5.0))
    )
    parent = build_response_parent_score_projection(
        physical_parent, optimization, pa_parent=pa_parent, pa_method_registry=methods
    )
    scoring = prepare_search_scoring(
        physical_parent=physical_parent,
        optimization_parent=parent,
        climate_parent=build_response_parent_score_projection(
            physical_parent, climate, pa_parent=pa_parent, pa_method_registry=methods
        ),
        optimization_projection=optimization,
        climate_projection=climate,
        pa_projection=pa_projection,
        pa_parent=pa_parent,
    )
    delta = pd.DataFrame()
    selected = _optimization_scores(_physical("pa-change", delta=delta), scoring, delta, candidate)
    baseline, baseline_norm = _projection_baseline(parent, scoring.optimization_projection)
    scores, norms = _score_arrays(
        baseline,
        baseline_norm,
        scoring.optimization_projection,
        delta,
        pa_parent_scores=pa_parent.annual_scores[..., [1]],
        pa_parent_norms=pa_parent.annual_contribution_norms[..., [1]],
        pa_candidate_scores=candidate.annual_scores[..., [1]],
        pa_candidate_norms=candidate.annual_contribution_norms[..., [1]],
    )
    eur, year = H12_REGIONS.index("EUR"), _RESPONSE_YEARS.index(2050)
    expected_score, expected_norm = (0.0, 0.0) if removed else (2.0, 5.0)
    np.testing.assert_array_equal(scoring.optimization_pa_indices, np.array([1]))
    np.testing.assert_array_equal(scoring.climate_pa_indices, np.array([1]))
    assert (selected.annual_scores[:, eur, year, 0] == expected_score).all()
    assert (selected.annual_climate_scores[:, eur, year] == expected_score).all()
    assert (scores[:, eur, year, 0] == expected_score).all()
    assert (norms[:, eur, year, 0] == expected_norm).all()
    # Production at 2019 stays fixed. Norms subtract the parent's contributor total on the specified
    # scale.
    assert (norms[:, eur, 0, 0] == 2.0).all()
    assert (scores[:, eur, 0, 0] == 2.0).all()
