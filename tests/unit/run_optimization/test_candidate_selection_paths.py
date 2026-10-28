"""Scientific optimization paths not exercised by the main workflow fixtures."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from candidate_characterization_fixtures import _empty_pa_scoring
from request_search_fixtures import _competitive_evaluator, _compliant_evaluation

from pytoce.run_lca.calculation.foreground import optimization_activation_delta
from pytoce.run_optimization.calculation.candidate_acc import (
    _assert_acc_scoring_compatibility,
    prepare_candidate_acc_context,
)
from pytoce.run_optimization.calculation.candidate_authority import (
    CandidateEvaluationError,
    PhysicalEvaluation,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    _optimization_scores,
    build_response_parent_score_projection,
    prepare_search_scoring,
)
from pytoce.run_optimization.calculation.candidate_search import (
    CandidateEvaluation,
    InterventionCandidate,
    neighboring_candidates,
    refine_candidates,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.selected_reporting import (
    _stable_group,
)
from pytoce.run_optimization.input.allocated_cc import select_allocated_cc_cases
from pytoce.run_optimization.input.allocated_cc_paths import AllocatedCCError
from tests.unit.run_optimization.response_projection_fixtures import (
    _allocated_cc_selection,
    _response_parent,
    _response_projection,
)


def test_allocated_cc_selection_reports_unknown_cases(tmp_path: Path) -> None:
    """Tell the user which cases are available in the selected result."""

    selection = _allocated_cc_selection()
    # ``select_allocated_cc_cases`` reads compact result metadata.
    from pytoce.run_optimization.input.allocated_cc_contract import AllocatedCCResult

    result = AllocatedCCResult(
        status="computed",
        allocated_cc_result="fixture_result",
        allocated_cc_result_identity=selection.allocated_cc_result_identity,
        allocated_cc_cases=selection.allocated_cc_cases,
        annual_impact_category_count=selection.annual_impact_category_count,
        result_directory=tmp_path,
        manifest_path=tmp_path / "manifest.json",
    )
    with pytest.raises(AllocatedCCError, match="unknown_case.*Choose from"):
        select_allocated_cc_cases(result, ("lower_values", "unknown_case"))


def test_search_scoring_rejects_allocated_cc_for_different_methods() -> None:
    """Reject an aCC result whose LCIA identities differ from the response projection."""

    physical_parent = _response_parent()
    optimization_projection = _response_projection("optimization")
    climate_projection = _response_projection("mandatory_climate")
    pa_projection, pa_parent = _empty_pa_scoring(optimization_projection.method_registry)
    scoring = prepare_search_scoring(
        physical_parent=physical_parent,
        optimization_parent=build_response_parent_score_projection(
            physical_parent,
            optimization_projection,
            pa_parent=pa_parent,
            pa_method_registry=pa_projection.method_registry,
        ),
        climate_parent=build_response_parent_score_projection(
            physical_parent,
            climate_projection,
            pa_parent=pa_parent,
            pa_method_registry=pa_projection.method_registry,
        ),
        optimization_projection=optimization_projection,
        climate_projection=climate_projection,
        pa_projection=pa_projection,
        pa_parent=pa_parent,
    )
    allocated_cc = prepare_candidate_acc_context(_allocated_cc_selection())

    with pytest.raises(CandidateEvaluationError, match="selected LCA methods do not match"):
        _assert_acc_scoring_compatibility(allocated_cc, scoring)


def test_candidate_response_scoring_applies_the_activation_change() -> None:
    """Change a future provider demand across backgrounds while retaining 2019 history."""

    physical_parent = _response_parent()
    physical_parent = replace(
        physical_parent,
        activations=pd.concat(
            (
                physical_parent.activations,
                physical_parent.activations.assign(year=2050, physical_event_year=2050),
            ),
            ignore_index=True,
        ),
    )
    optimization_projection = _response_projection("optimization")
    climate_projection = _response_projection("mandatory_climate")
    pa_projection, pa_parent = _empty_pa_scoring(optimization_projection.method_registry)
    scoring = prepare_search_scoring(
        physical_parent=physical_parent,
        optimization_parent=build_response_parent_score_projection(
            physical_parent,
            optimization_projection,
            pa_parent=pa_parent,
            pa_method_registry=pa_projection.method_registry,
        ),
        climate_parent=build_response_parent_score_projection(
            physical_parent,
            climate_projection,
            pa_parent=pa_parent,
            pa_method_registry=pa_projection.method_registry,
        ),
        optimization_projection=optimization_projection,
        climate_projection=climate_projection,
        pa_projection=pa_projection,
        pa_parent=pa_parent,
    )
    candidate_activations = physical_parent.activations.copy(deep=True)
    candidate_activations.loc[candidate_activations["year"].eq(2050), "amount"] = 0.0
    delta = optimization_activation_delta(
        physical_parent.activations,
        candidate_activations,
    )
    candidate = PhysicalEvaluation(
        physical_request_identity="physical-request::fixture",
        physical_candidate_identity="physical-candidate::fixture",
        physical_parent_identity=physical_parent.identity,
        intervention_authority_identity="intervention::fixture",
        status="physical_feasible",
        diagnostic=None,
        physical_state_identity="physical-state::fixture",
        activation_identity="activation::fixture",
        activation_delta=delta,
        pa_spinning_events=(),
        pa_recycling_events=(),
        pa_disposal_events=(),
    )

    scores = _optimization_scores(candidate, scoring, delta, pa_parent)

    eur = scores.regions.index("EUR")
    future = scores.years.index(2050)
    history = scores.years.index(2019)
    assert np.all(scores.annual_scores[:, eur, future, 0] == 0.0)
    assert np.all(scores.annual_climate_scores[:, eur, future] == 0.0)
    np.testing.assert_array_equal(
        scores.annual_scores[:, :, history, :],
        scoring.optimization_baseline_scores[:, :, history, :],
    )
    np.testing.assert_array_equal(
        scores.annual_climate_scores[:, :, history],
        scoring.climate_baseline_scores[:, :, history, 0],
    )
    assert not scores.annual_scores.flags.writeable


def test_candidate_neighbors_cover_circularity_bounds_and_circular_target_year() -> None:
    """Keep circularity-only search within percentage bounds and its chosen year."""

    candidate = InterventionCandidate(
        "circularity_feedstock_anchor",
        DemandLifetimeDecision(0, 0, 2050),
        CircularityDecision(0, 100, 0, 0, 0, 100, 0, 0, 2075),
    )

    neighbors = neighboring_candidates(candidate, step_pct=1)

    assert candidate.target_year == 2075
    assert neighbors
    for item in neighbors:
        values = item.circularity.percentages()
        assert values is not None
        assert all(0 <= value <= 100 for value in values)
        assert sum(values[1:5]) == 100
        assert sum(values[5:]) == 100


def test_refinement_reuses_neighbors_already_present_in_the_initial_design() -> None:
    """Reuse completed candidate calculations during percentage refinement."""

    selected = InterventionCandidate(
        "demand_lifetime_frontier",
        DemandLifetimeDecision(10, 0),
        CircularityDecision.parent(),
    )
    initial = (
        selected,
        InterventionCandidate(
            "demand_lifetime_frontier",
            DemandLifetimeDecision(0, 0),
            CircularityDecision.parent(),
        ),
        InterventionCandidate(
            "demand_lifetime_frontier",
            DemandLifetimeDecision(20, 0),
            CircularityDecision.parent(),
        ),
        InterventionCandidate(
            "demand_lifetime_frontier",
            DemandLifetimeDecision(10, 10),
            CircularityDecision.parent(),
        ),
    )

    calls: Counter[str] = Counter()

    def evaluate(
        candidates: tuple[InterventionCandidate, ...],
    ) -> tuple[CandidateEvaluation, ...]:
        """Make only the selected fixture state compliant."""
        calls.update(item.identity for item in candidates)
        return tuple(
            _compliant_evaluation(candidate)
            if candidate.identity == selected.identity
            else CandidateEvaluation(
                candidate=candidate,
                physical_candidate_identity=f"physical::{candidate.identity}",
                state_metrics=(),
                intervention_distance=1.0,
            )
            for candidate in candidates
        )

    result = refine_candidates(
        optimization_block="demand_lifetime_frontier",
        initial_candidates=initial,
        evaluate=evaluate,
        evaluate_competitive=_competitive_evaluator(
            evaluate,
            lambda candidate: (
                _compliant_evaluation(candidate).intervention_distance
                if candidate.identity == selected.identity
                else 1.0
            ),
        ),
    )

    assert set(result.passes[0].newly_evaluated_candidate_identities).isdisjoint(
        item.identity for item in initial
    )
    assert all(count == 1 for count in calls.values())
    assert result.reused_candidate_count >= 3


def test_empty_selected_adjustments_keep_the_requested_table_schema() -> None:
    """Retain requested columns for an empty selected-adjustment aggregate."""

    observed = _stable_group(
        pd.DataFrame(),
        ("region", "year"),
        ("score", "contribution_norm"),
    )

    assert tuple(observed.columns) == ("region", "year", "score", "contribution_norm")
