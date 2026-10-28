"""Calculate environmental tables for the selected candidates.

The calculation uses supplied candidate records, baseline LCA tables, and a
response-bank projection to characterize the selected pathways.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import fsum
from typing import Literal, cast

import numpy as np
import pandas as pd

from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.candidate_projection_arrays import (
    _apply_score_adjustments,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    ResponseBankProjection,
)
from pytoce.shared.runtime.filesystem import (
    payload_sha256,
)

SELECTED_REPORTING_METHOD_IDENTITY = "pytoce_optimization_selected_reporting_method_v8"
_PHYSICAL_IDENTITY_PREFIX = "pytoce_optimization_physical_reporting_v2::"
_BASELINE_LCA_RECORD_PREFIX = (
    "pytoce_optimization_selected_reporting_baseline_lca_result_record_v1::"
)
_STATE_PREFIX = "pytoce_optimization_selected_reporting_state_v2::"
_METHOD_COLUMNS = (
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "unit",
    "factor_identity",
)
_PARENT_METHOD_COLUMNS = _METHOD_COLUMNS[:-1]
_STAGE_COLUMNS = (
    "life_cycle_stage",
    "stage_group",
    "cutoff_cycle_role",
    "material_cycle_source",
    "physical_event_basis",
)
_CANDIDATE_COLUMNS = (
    "candidate_identity",
    "target_year",
)


def _stable_sum(values: pd.Series) -> float:
    """Return a sum computed with ``math.fsum``."""
    return fsum(float(value) for value in values)


def _stable_group(
    frame: pd.DataFrame,
    keys: Sequence[str],
    values: Sequence[str],
) -> pd.DataFrame:
    """Group value columns with stable floating-point summation."""
    if frame.empty:
        return pd.DataFrame(columns=[*keys, *values])
    return (
        frame.sort_values([*keys, *values], kind="stable", na_position="last")
        .groupby(list(keys), observed=True, sort=True, dropna=False)[list(values)]
        .agg(_stable_sum)
        .reset_index()
    )


@dataclass(frozen=True, slots=True)
class SelectedReportingState:
    """Store one selected 2050 candidate and the same percentages in 2100.

    Attributes:
        physical_candidate_identity: Content identifier of the selected 2050 MFA outcome.
        state_identity: Content identifier calculated from the optimization block,
            background scenario, and allocated carrying capacity label.
        selection_candidate_identity: Content identifier of the selected 2050 values.
        timing_2100_candidate_identity: Content identifier of those values completed in 2100.
        timing_2100_physical_candidate_identity: Content identifier of the 2100 MFA outcome,
            including a recognized infeasible outcome.
        optimization_block: Decision group in which the candidate was selected.
        selection_background_scenario: REMIND background used for selection.
        allocated_cc_case: Allocated carrying capacity label used for selection.
        selection_status: Whether the baseline or an intervention is compliant, or
            the best available noncompliant candidate was selected.
        all_region_joint_compliant: Whether every required coordinate in the
            12 world regions has ASR at most 1.
        intervention_distance: Distance from baseline intervention values.
        binding_slack: ``1 - worst_asr`` when all required coordinates comply;
            otherwise ``None``.
        compliant_region_count: Number of fully compliant world regions.
        fully_compliant_impact_target_count: Number of targets compliant in every region.
        compliant_region_impact_group_count: Compliant region and impact-target groups.
        compliant_coordinate_count: Compliant annual and cumulative ASR coordinates.
        region_count: Total number of world regions assessed.
        impact_target_count: Total annual categories plus the cumulative climate target.
        region_impact_group_count: Total region and impact-target groups.
        coordinate_count: Total annual and cumulative ASR coordinates.
        worst_asr: Largest ASR among the coordinates required for selection.
        maximum_positive_exceedance: Largest positive ``ASR - 1``.
        balanced_mean_positive_exceedance: Mean positive exceedance with equal weight
            across impact targets and 12 world regions.
        fully_compliant_regions: World regions compliant for every required target.
        fully_compliant_impact_targets: Targets compliant in every world region.
        binding_coordinates: Coordinates with the smallest slack or largest exceedance.
        transgressing_coordinates: Coordinates whose ASR exceeds 1.
        selected_is_exact_parent: Whether the selected candidate retains baseline values.
        selected_has_non_grid_baseline_component: Whether one unchanged component comes
            directly from baseline values rather than an integer percentage.
        declared_neighborhood_verified: Whether the final declared neighborhood is complete
            with all demand-repair outcomes resolved, including guided moves for explicit III
            circularity.
        selection_stable: Whether none of those alternatives ranks ahead of the selection.
        demand_reduction_pct: Demand reduction completed in ``selection_target_year``.
        lifetime_extension_pct: Lifetime extension completed in that year.
        collection_pct: Collection target, or ``None`` when unchanged.
        mechanical_recycling_pct: Mechanical recycling share, or ``None`` when unchanged.
        chemical_recycling_pct: Chemical recycling share, or ``None`` when unchanged.
        incineration_pct: Incineration share, or ``None`` when unchanged.
        landfill_pct: Landfill share, or ``None`` when unchanged.
        fossil_feedstock_pct: Fossil feedstock share, or ``None`` when unchanged.
        biogenic_feedstock_pct: Biogenic feedstock share, or ``None`` when unchanged.
        daccu_feedstock_pct: DACCU feedstock share, or ``None`` when unchanged.
        selection_target_year: Completion year used during selection, currently 2050.
        identity: Content identifier calculated from the selection, metrics, intervention
            percentages, and 2050 and 2100 results.
    """

    physical_candidate_identity: str
    state_identity: str
    selection_candidate_identity: str
    timing_2100_candidate_identity: str
    timing_2100_physical_candidate_identity: str
    optimization_block: OptimizationBlock
    selection_background_scenario: str
    allocated_cc_case: str
    selection_status: Literal[
        "intervention_compliant",
        "baseline_compliant",
        "selected_noncompliant",
    ]
    all_region_joint_compliant: bool
    intervention_distance: float
    binding_slack: float | None
    compliant_region_count: int
    fully_compliant_impact_target_count: int
    compliant_region_impact_group_count: int
    compliant_coordinate_count: int
    region_count: int
    impact_target_count: int
    region_impact_group_count: int
    coordinate_count: int
    worst_asr: float
    maximum_positive_exceedance: float
    balanced_mean_positive_exceedance: float
    fully_compliant_regions: tuple[str, ...]
    fully_compliant_impact_targets: tuple[str, ...]
    binding_coordinates: tuple[tuple[str, str, str, str, int | None, float], ...]
    transgressing_coordinates: tuple[tuple[str, str, str, str, int | None, float], ...]
    selected_is_exact_parent: bool
    selected_has_non_grid_baseline_component: bool
    declared_neighborhood_verified: bool
    selection_stable: bool
    demand_reduction_pct: int
    lifetime_extension_pct: int
    collection_pct: int | None
    mechanical_recycling_pct: int | None
    chemical_recycling_pct: int | None
    incineration_pct: int | None
    landfill_pct: int | None
    fossil_feedstock_pct: int | None
    biogenic_feedstock_pct: int | None
    daccu_feedstock_pct: int | None
    selection_target_year: int
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        """Identify the selected state and its delayed timing result."""
        circularity = (
            self.collection_pct,
            self.mechanical_recycling_pct,
            self.chemical_recycling_pct,
            self.incineration_pct,
            self.landfill_pct,
            self.fossil_feedstock_pct,
            self.biogenic_feedstock_pct,
            self.daccu_feedstock_pct,
        )
        payload = {
            "method_identity": SELECTED_REPORTING_METHOD_IDENTITY,
            "physical_candidate_identity": self.physical_candidate_identity,
            "state_identity": self.state_identity,
            "selection_candidate_identity": self.selection_candidate_identity,
            "timing_2100_candidate_identity": self.timing_2100_candidate_identity,
            "timing_2100_physical_candidate_identity": (
                self.timing_2100_physical_candidate_identity
            ),
            "optimization_block": self.optimization_block,
            "selection_background_scenario": self.selection_background_scenario,
            "allocated_cc_case": self.allocated_cc_case,
            "selection_status": self.selection_status,
            "all_region_joint_compliant": self.all_region_joint_compliant,
            "intervention_distance": self.intervention_distance,
            "binding_slack": self.binding_slack,
            "coverage_counts": [
                self.compliant_region_count,
                self.fully_compliant_impact_target_count,
                self.compliant_region_impact_group_count,
                self.compliant_coordinate_count,
                self.region_count,
                self.impact_target_count,
                self.region_impact_group_count,
                self.coordinate_count,
            ],
            "worst_asr": self.worst_asr,
            "maximum_positive_exceedance": self.maximum_positive_exceedance,
            "balanced_mean_positive_exceedance": (self.balanced_mean_positive_exceedance),
            "fully_compliant_regions": list(self.fully_compliant_regions),
            "fully_compliant_impact_targets": list(self.fully_compliant_impact_targets),
            "binding_coordinates": [list(item) for item in self.binding_coordinates],
            "transgressing_coordinates": [list(item) for item in self.transgressing_coordinates],
            "selected_is_exact_parent": self.selected_is_exact_parent,
            "selected_has_non_grid_baseline_component": (
                self.selected_has_non_grid_baseline_component
            ),
            "declared_neighborhood_verified": (self.declared_neighborhood_verified),
            "selection_stable": self.selection_stable,
            "demand_reduction_pct": self.demand_reduction_pct,
            "lifetime_extension_pct": self.lifetime_extension_pct,
            "circularity_percentages": list(circularity),
            "selection_target_year": self.selection_target_year,
        }
        object.__setattr__(self, "identity", _STATE_PREFIX + payload_sha256(payload))


@dataclass(frozen=True, slots=True)
class BaselineLCAResultRecord:
    """Identities for one foreground and background LCA result."""

    foreground_scenario: str
    background_scenario: str
    result_identity: str
    result_manifest_sha256: str
    annual_life_cycle_stage_scores_identity: str
    life_cycle_stage_scores_identity: str
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        """Identify the completed baseline and its two consumed stage tables."""
        payload = {
            "method_identity": SELECTED_REPORTING_METHOD_IDENTITY,
            "baseline_lca_schema": "pytoce_lca_result_schema_v1",
            "foreground_scenario": self.foreground_scenario,
            "background_scenario": self.background_scenario,
            "result_identity": self.result_identity,
            "result_manifest_sha256": self.result_manifest_sha256,
            "annual_life_cycle_stage_scores_identity": (
                self.annual_life_cycle_stage_scores_identity
            ),
            "life_cycle_stage_scores_identity": self.life_cycle_stage_scores_identity,
        }
        object.__setattr__(self, "identity", _BASELINE_LCA_RECORD_PREFIX + payload_sha256(payload))


@dataclass(frozen=True, slots=True)
class BaselineLCAResult:
    """Baseline LCA tables for one foreground and background scenario."""

    record: BaselineLCAResultRecord
    annual_life_cycle_stage_scores: pd.DataFrame
    life_cycle_stage_scores: pd.DataFrame


def _effective_responses(projection: ResponseBankProjection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply annual provider weights to the unit responses."""
    methods = projection.method_registry.loc[:, list(_METHOD_COLUMNS)].drop_duplicates(
        ignore_index=True
    )
    weights = projection.timex_year_weights
    responses = projection.provider_unit_responses
    joined = weights.merge(
        responses,
        on=["background_scenario", "provider_anchor_year"],
        how="inner",
    )
    joined["effective_unit_score"] = joined["temporal_provider_share"].astype(float) * joined[
        "unit_score"
    ].astype(float)
    joined["effective_unit_norm"] = joined["temporal_provider_share"].astype(float) * joined[
        "unit_contribution_norm"
    ].astype(float)
    keys = (
        "background_scenario",
        "occurrence_year",
        "provider_service_id",
        "provider_service_code",
        *_METHOD_COLUMNS,
    )
    effective = _stable_group(
        joined,
        keys,
        ("effective_unit_score", "effective_unit_norm"),
    )
    return methods.sort_values(list(_METHOD_COLUMNS), kind="stable", ignore_index=True), effective


def _parent_stage_table(
    parent: BaselineLCAResult,
    frame: pd.DataFrame,
    methods: pd.DataFrame,
    *,
    cumulative: bool,
) -> pd.DataFrame:
    """Select regional stage results and join their characterization-factor identities.

    Annual rows retain their activity year; cumulative rows have null
    years and cover 2019--2100. Scores and absolute characterized contributions
    retain each LCIA method's native unit. The completed parent supplies the
    stage coordinates and quantities.
    """
    columns = [
        "region",
        "year",
        *_PARENT_METHOD_COLUMNS,
        *_STAGE_COLUMNS,
        "physical_event_year",
        "score",
        "contribution_norm",
    ]
    selected = frame.loc[frame["region"].isin(H12_REGIONS), columns].copy()
    if not cumulative:
        selected[["year", "physical_event_year"]] = selected[
            ["year", "physical_event_year"]
        ].astype(np.int64)
    selected[["score", "contribution_norm"]] = selected[["score", "contribution_norm"]].astype(
        np.float64
    )
    selected = selected.merge(methods, on=list(_PARENT_METHOD_COLUMNS), how="left")
    keys = ["region", "year", *_METHOD_COLUMNS, *_STAGE_COLUMNS, "physical_event_year"]
    selected.insert(0, "characterization_background_scenario", parent.record.background_scenario)
    return selected.loc[
        :, ["characterization_background_scenario", *keys, "score", "contribution_norm"]
    ]


def _stage_response_adjustments(
    selected: pd.DataFrame,
    effective: pd.DataFrame,
    *,
    background: str,
) -> pd.DataFrame:
    """Project stage-attributed activation changes for one background scenario."""

    values = (
        "score_adjustment",
        "norm_adjustment",
        "changed_parent_score",
        "changed_candidate_score",
        "changed_parent_norm",
        "changed_candidate_norm",
    )
    total_keys = ("characterization_background_scenario", "region", "year", *_METHOD_COLUMNS)
    stage_keys = (*total_keys, *_STAGE_COLUMNS, "physical_event_year")
    if selected.empty:
        return pd.DataFrame(columns=[*stage_keys, *values])
    member_effective = effective.loc[effective["background_scenario"].eq(background)]
    stage_frames: list[pd.DataFrame] = []
    grouped = member_effective.groupby(
        list(_METHOD_COLUMNS),
        observed=True,
        sort=True,
        dropna=False,
    )
    method_indices = {
        tuple(str(value) for value in cast(tuple[object, ...], raw_record)): indices
        for raw_record, indices in grouped.indices.items()
    }
    for record in sorted(method_indices):
        method_effective = member_effective.iloc[method_indices[record]]
        stage = _method_stage_response_adjustment(
            selected,
            method_effective,
            stage_keys=stage_keys,
            values=values,
        )
        stage_frames.append(stage)
        del method_effective, stage
    stage = _stable_group(
        pd.concat(stage_frames, ignore_index=True),
        stage_keys,
        values,
    )
    stage_frames.clear()
    return stage


def _pa_stage_adjustments(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    methods: pd.DataFrame,
    *,
    background: str,
) -> pd.DataFrame:
    """Subtract complete actual PA stage scores and absolute totals separately."""
    keys = ["region", "year", *_METHOD_COLUMNS, *_STAGE_COLUMNS, "physical_event_year"]
    values = ["score", "contribution_norm"]
    frames = []
    for frame, role in ((parent, "parent"), (candidate, "candidate")):
        selected = frame.loc[frame.characterization_background_scenario.eq(background)]
        selected = selected.merge(methods.loc[:, list(_METHOD_COLUMNS)], how="inner")
        selected = selected.loc[:, [*keys, *values]].set_index(keys)
        frames.append(selected.rename(columns={name: f"{role}_{name}" for name in values}))
    # A new or removed stage has zero impact on the absent side of this pair.
    joined = frames[0].join(frames[1], how="outer").fillna(0.0).reset_index()
    joined["score_adjustment"] = joined.candidate_score - joined.parent_score
    joined["norm_adjustment"] = joined.candidate_contribution_norm - joined.parent_contribution_norm
    joined = joined.rename(
        columns={
            "parent_score": "changed_parent_score",
            "candidate_score": "changed_candidate_score",
            "parent_contribution_norm": "changed_parent_norm",
            "candidate_contribution_norm": "changed_candidate_norm",
        }
    )
    joined.insert(0, "characterization_background_scenario", background)
    return joined


def _method_stage_response_adjustment(
    selected: pd.DataFrame,
    method_effective: pd.DataFrame,
    *,
    stage_keys: Sequence[str],
    values: Sequence[str],
) -> pd.DataFrame:
    """Calculate stage LCIA-result changes for one method and background."""
    joined = selected.merge(
        method_effective,
        left_on=["year", "provider_service_id", "provider_service_code"],
        right_on=["occurrence_year", "provider_service_id", "provider_service_code"],
        how="left",
    )
    joined["score_adjustment"] = joined["delta_amount_kg"] * joined["effective_unit_score"]
    joined["changed_parent_score"] = joined["parent_amount_kg"] * joined["effective_unit_score"]
    joined["changed_candidate_score"] = (
        joined["candidate_amount_kg"] * joined["effective_unit_score"]
    )
    joined["norm_adjustment"] = (
        np.abs(joined["candidate_amount_kg"]) - np.abs(joined["parent_amount_kg"])
    ) * joined["effective_unit_norm"]
    joined["changed_parent_norm"] = (
        np.abs(joined["parent_amount_kg"]) * joined["effective_unit_norm"]
    )
    joined["changed_candidate_norm"] = (
        np.abs(joined["candidate_amount_kg"]) * joined["effective_unit_norm"]
    )
    joined = joined.rename(columns={"background_scenario": "characterization_background_scenario"})
    return _stable_group(joined, stage_keys, values)


def _apply_adjustment(
    parent: pd.DataFrame,
    adjustment: pd.DataFrame,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Apply changes over the union of parent and selected stage coordinates.

    A candidate can introduce a stage, cycle or event year absent from its
    parent's sparse result. That coordinate has zero parent impact; retaining
    only parent rows would discard the newly selected activity's burdens.
    """
    value_columns = (
        "score_adjustment",
        "norm_adjustment",
        "changed_parent_score",
        "changed_candidate_score",
        "changed_parent_norm",
        "changed_candidate_norm",
    )
    result = parent.merge(adjustment, on=list(keys), how="outer")
    # Missing parent or candidate coordinates contribute zero activity in the
    # matched LCIA unit. The outer join retains unchanged parent rows and new
    # candidate rows.
    value_columns_with_parent = ["score", "contribution_norm", *value_columns]
    result[value_columns_with_parent] = (
        result[value_columns_with_parent].astype(np.float64).fillna(0.0)
    )
    score, norm = _apply_score_adjustments(
        result["score"].to_numpy(dtype=np.float64),
        result["contribution_norm"].to_numpy(dtype=np.float64),
        score_adjustment=result["score_adjustment"].to_numpy(dtype=np.float64),
        norm_adjustment=result["norm_adjustment"].to_numpy(dtype=np.float64),
        changed_parent_scores=result["changed_parent_score"].to_numpy(dtype=np.float64),
        changed_parent_norms=result["changed_parent_norm"].to_numpy(dtype=np.float64),
        changed_candidate_scores=result["changed_candidate_score"].to_numpy(dtype=np.float64),
        changed_candidate_norms=result["changed_candidate_norm"].to_numpy(dtype=np.float64),
    )
    result["score"] = score
    result["contribution_norm"] = norm
    return result.drop(columns=list(value_columns))


def _with_candidate(
    frame: pd.DataFrame,
    *,
    candidate_identity: str,
    target_year: int,
) -> pd.DataFrame:
    """Add the candidate identity and completion year to result rows."""
    result = frame.copy()
    result.insert(0, "candidate_identity", candidate_identity)
    result.insert(1, "target_year", target_year)
    result["candidate_identity"] = result["candidate_identity"].astype("string")
    result["target_year"] = result["target_year"].astype("Int64")
    return result


def _global(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Sum the regional values and label the result as global."""
    grouped = _stable_group(frame, keys, ("score", "contribution_norm"))
    grouped.insert(len(_CANDIDATE_COLUMNS) + 1, "region", "global_derived")
    return grouped
