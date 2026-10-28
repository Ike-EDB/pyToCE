"""Write selected optimization results one candidate and background at a time."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import pandas as pd

from pytoce.run_optimization.calculation.pa_candidate_projection import PaCandidateScores
from pytoce.run_optimization.calculation.response_bank_contract import (
    ResponseBankProjection,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    CandidateWorkerResult,
    SelectedCandidateResult,
)

from .selected_reporting import (
    _CANDIDATE_COLUMNS,
    _METHOD_COLUMNS,
    _STAGE_COLUMNS,
    BaselineLCAResult,
    SelectedReportingState,
    _apply_adjustment,
    _effective_responses,
    _global,
    _pa_stage_adjustments,
    _parent_stage_table,
    _stable_group,
    _stage_response_adjustments,
    _with_candidate,
)


class SelectedStageTableSink(Protocol):
    """One-use sink for the large selected public stage table."""

    def start(self, schema_sample: pd.DataFrame) -> None:
        """Fix the logical Pandas and Arrow schema."""

    def write(self, frame: pd.DataFrame) -> None:
        """Consume one frame."""


@dataclass(frozen=True, slots=True)
class _PhysicalReportingComponent:
    """Annual and cumulative stage tables for one MFA calculation."""

    annual_life_cycle_stage_scores: pd.DataFrame
    life_cycle_stage_scores: pd.DataFrame


def _physical_reporting_components(
    evaluation_index: Mapping[str, CandidateWorkerResult],
    target_years: Mapping[str, int],
    parent_tables: Mapping[
        str,
        tuple[pd.DataFrame, pd.DataFrame],
    ],
    effective: pd.DataFrame,
    pa_parent: PaCandidateScores,
    pa_candidates: Mapping[str, PaCandidateScores],
    methods: pd.DataFrame,
) -> Iterator[_PhysicalReportingComponent]:
    """Yield each selected candidate and background in fixed order."""

    for candidate in sorted(evaluation_index):
        result = evaluation_index[candidate]
        physical = result.physical
        selected_delta = cast(pd.DataFrame, physical.activation_delta)
        for background in parent_tables:
            annual_stage_parent, cumulative_stage_parent = parent_tables[background]
            yield _physical_reporting_component(
                candidate_identity=candidate,
                target_year=target_years[candidate],
                background=background,
                annual_stage_parent=annual_stage_parent,
                cumulative_stage_parent=cumulative_stage_parent,
                selected_delta=selected_delta,
                effective=effective,
                pa_parent=pa_parent.stage_scores,
                pa_candidate=pa_candidates[candidate].stage_scores,
                methods=methods,
            )
        del physical, selected_delta


def _physical_reporting_component(
    *,
    candidate_identity: str,
    target_year: int,
    background: str,
    annual_stage_parent: pd.DataFrame,
    cumulative_stage_parent: pd.DataFrame,
    selected_delta: pd.DataFrame,
    effective: pd.DataFrame,
    pa_parent: pd.DataFrame,
    pa_candidate: pd.DataFrame,
    methods: pd.DataFrame,
) -> _PhysicalReportingComponent:
    """Build one component."""

    stage_adjustment = _stage_response_adjustments(
        selected_delta,
        effective,
        background=background,
    )
    total_keys = [
        "characterization_background_scenario",
        "region",
        "year",
        *_METHOD_COLUMNS,
    ]
    stage_keys = [*total_keys, *_STAGE_COLUMNS, "physical_event_year"]
    pa_adjustment = _pa_stage_adjustments(pa_parent, pa_candidate, methods, background=background)
    adjustment_values = [column for column in stage_adjustment.columns if column not in stage_keys]
    frames = [frame for frame in (stage_adjustment, pa_adjustment) if not frame.empty]
    stage_adjustment = _stable_group(
        pd.concat(frames, ignore_index=True) if frames else stage_adjustment,
        stage_keys,
        adjustment_values,
    )
    annual_stage = _apply_adjustment(
        annual_stage_parent,
        stage_adjustment,
        stage_keys,
    )
    cumulative_adjustment = _stable_group(
        stage_adjustment,
        [
            "characterization_background_scenario",
            "region",
            *_METHOD_COLUMNS,
            *_STAGE_COLUMNS,
        ],
        (
            "score_adjustment",
            "norm_adjustment",
            "changed_parent_score",
            "changed_candidate_score",
            "changed_parent_norm",
            "changed_candidate_norm",
        ),
    )
    cumulative_parent = cumulative_stage_parent.drop(columns=["year", "physical_event_year"])
    cumulative_keys = [
        "characterization_background_scenario",
        "region",
        *_METHOD_COLUMNS,
        *_STAGE_COLUMNS,
    ]
    cumulative_stage = _apply_adjustment(
        cumulative_parent,
        cumulative_adjustment,
        cumulative_keys,
    )
    annual_stage = _with_candidate(
        annual_stage,
        candidate_identity=candidate_identity,
        target_year=target_year,
    )
    cumulative_stage = _with_candidate(
        cumulative_stage,
        candidate_identity=candidate_identity,
        target_year=target_year,
    )
    cumulative_stage["year"] = pd.NA
    cumulative_stage["physical_event_year"] = pd.NA
    cumulative_stage["period_start_year"] = 2019
    cumulative_stage["period_end_year"] = 2100
    annual_stage["period_start_year"] = annual_stage["year"]
    annual_stage["period_end_year"] = annual_stage["year"]
    global_total_keys = [
        *_CANDIDATE_COLUMNS,
        "characterization_background_scenario",
        "year",
        *_METHOD_COLUMNS,
    ]
    global_stage_keys = [
        *global_total_keys,
        *_STAGE_COLUMNS,
        "physical_event_year",
        "period_start_year",
        "period_end_year",
    ]
    global_cumulative_keys = [
        *_CANDIDATE_COLUMNS,
        "characterization_background_scenario",
        *_METHOD_COLUMNS,
        *_STAGE_COLUMNS,
        "period_start_year",
        "period_end_year",
    ]
    global_annual_stage = _global(annual_stage, global_stage_keys)
    global_cumulative_stage = _global(cumulative_stage, global_cumulative_keys)
    global_cumulative_stage["year"] = pd.NA
    global_cumulative_stage["physical_event_year"] = pd.NA
    annual_stage_all = pd.concat(
        (annual_stage, global_annual_stage),
        ignore_index=True,
    )
    cumulative_stage_all = pd.concat(
        (cumulative_stage, global_cumulative_stage),
        ignore_index=True,
    )
    return _PhysicalReportingComponent(
        annual_life_cycle_stage_scores=annual_stage_all,
        life_cycle_stage_scores=cumulative_stage_all,
    )


def stream_selected_stage_reporting(
    *,
    selected_evaluations: Sequence[SelectedCandidateResult],
    states: Sequence[SelectedReportingState],
    baseline_lca_results: Sequence[BaselineLCAResult],
    response_projection: ResponseBankProjection,
    pa_parent: PaCandidateScores,
    pa_candidates: Mapping[str, PaCandidateScores],
    sink: SelectedStageTableSink,
    progress: Callable[[int, int], None],
) -> None:
    """Write selected stage tables one MFA candidate at a time.

    Args:
        selected_evaluations: Saved feasible 2050 and 2100 candidate calculations.
        states: Selection metrics and timing identities for every requested state.
        baseline_lca_results: Completed LCA results for the selected backgrounds.
        response_projection: Provider unit responses for the requested result methods.
        pa_parent: Parent PA scores characterized with the reporting methods.
        pa_candidates: Once-characterized reporting PA scores by MFA identity.
        sink: Table writer that receives annual rows first and cumulative rows second.
        progress: Callback receiving completed and total candidate-background components.

    Notes:
        Annual rows contain region, year, LCIA method, life cycle stage, and
        event dimensions. Cumulative rows retain the same dimensions
        with null years and cover 2019--2100. LCIA results retain native category
        units. ``contribution_norm`` is the sum of the absolute values of the
        characterized contributions, in the LCIA category's unit. Each component
        is consumed once so its large intermediate tables can be released.
    """

    target_years: dict[str, int] = {}
    for state in states:
        for target_year, candidate in (
            (2050, state.physical_candidate_identity),
            (2100, state.timing_2100_physical_candidate_identity),
        ):
            target_years[candidate] = min(target_years.get(candidate, target_year), target_year)
    evaluation_index = {
        item.result.physical.physical_candidate_identity: item.result
        for item in selected_evaluations
    }
    methods, effective = _effective_responses(response_projection)
    parent_index = {parent.record.background_scenario: parent for parent in baseline_lca_results}
    parent_tables = {
        background: (
            _parent_stage_table(
                parent_index[background],
                parent_index[background].annual_life_cycle_stage_scores,
                methods,
                cumulative=False,
            ),
            _parent_stage_table(
                parent_index[background],
                parent_index[background].life_cycle_stage_scores,
                methods,
                cumulative=True,
            ),
        )
        for background in sorted(parent_index)
    }
    cumulative_chunks: list[pd.DataFrame] = []
    started = False
    component_count = len(evaluation_index) * len(parent_tables)
    for completed_components, component in enumerate(
        _physical_reporting_components(
            evaluation_index,
            target_years,
            parent_tables,
            effective,
            pa_parent,
            pa_candidates,
            methods,
        ),
        start=1,
    ):
        annual = component.annual_life_cycle_stage_scores
        annual["year"] = annual["year"].astype(object)
        annual["physical_event_year"] = annual["physical_event_year"].astype(object)
        annual["period_kind"] = "annual"
        cumulative = component.life_cycle_stage_scores
        cumulative["period_kind"] = "cumulative_2019_2100"
        cumulative = cumulative.loc[:, list(annual.columns)]
        del component
        if not started:
            sink.start(_stage_schema_sample((annual, cumulative)))
            started = True
        sink.write(annual)
        cumulative_chunks.append(cumulative)
        del annual
        progress(completed_components, component_count)
    for cumulative in cumulative_chunks:
        sink.write(cumulative)
        del cumulative
    cumulative_chunks.clear()


def _stage_schema_sample(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Keep enough rows to determine every selected stage column's Arrow type."""

    samples: list[pd.DataFrame] = []
    for frame in frames:
        labels = {frame.index[0]}
        for column in frame.columns:
            first_valid = frame[column].first_valid_index()
            if first_valid is not None:
                labels.add(first_valid)
        samples.append(frame.loc[sorted(labels)])
    return pd.concat(samples, ignore_index=True)
