"""Calculate annual and cumulative stage scores, global totals and shares."""

from collections.abc import Sequence
from math import fsum
from typing import Any, cast

import numpy as np
import pandas as pd

from .burdens import BURDEN_SPECS
from .contribution_accumulation import GLOBAL_REGION, ContributionState
from .contribution_records import _BatchedRecords, _frame_from_records, _method_dimensions


def _stage_share_status(score: float, norm: float) -> str:
    """Return whether a life cycle stage share can be calculated.

    ``available`` means the denominator can be used. ``zero_contribution_norm`` means the sum of
    absolute characterized contributions is zero. ``zero_score`` means the
    LCIA total is zero. ``near_zero_score`` means the absolute LCIA total is at
    most 1e-10 of the sum of absolute characterized contributions: cancellation
    makes the ratio unreliable. This criterion is independent of category units.
    """
    if norm == 0.0:
        return "zero_contribution_norm"
    if score == 0.0:
        return "zero_score"
    if abs(score) / norm <= 1e-10:
        return "near_zero_score"
    return "available"


def _with_stage_score_shares(
    frame: pd.DataFrame,
    denominator_keys: Sequence[str],
) -> pd.DataFrame:
    """Calculate score and ``contribution_norm`` shares by life cycle stage."""
    denominators = _stable_grouped_sum(frame, denominator_keys).rename(
        columns={"score": "total_score", "contribution_norm": "total_norm"}
    )
    output = frame.merge(
        denominators,
        on=list(denominator_keys),
        how="left",
    )
    statuses = [
        _stage_share_status(float(item.total_score), float(item.total_norm))
        for row in output.itertuples(index=False)
        for item in (cast(Any, row),)
    ]
    output["share_status"] = statuses
    output["share"] = [
        float(score) / float(total) if status == "available" else np.nan
        for score, total, status in zip(
            output["score"], output["total_score"], statuses, strict=True
        )
    ]
    output["norm_share"] = [
        float(norm) / float(total) if float(total) != 0.0 else np.nan
        for norm, total in zip(output["contribution_norm"], output["total_norm"], strict=True)
    ]
    return output.drop(columns=["total_score", "total_norm"])


def _stage_score_tables(
    state: ContributionState,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create complete annual and cumulative life cycle stage score tables.

    Stored regional contributions are stably summed by region, year, LCIA
    category, stage, material-cycle source, and activity. Missing valid
    coordinates are filled with zero before a global row is derived from all
    12 world regions. Shares use annual or 2019--2100 totals with the same
    coordinates as their denominators.
    """
    rows = _BatchedRecords()
    for key, (score, norm) in state.annual_stage_values.items():
        rows.append(
            {
                "region": key[0],
                "year": key[1],
                **_method_dimensions(state.methods[key[2]]),
                "life_cycle_stage": key[3],
                "stage_group": key[4],
                "cutoff_cycle_role": key[5],
                "material_cycle_source": key[6],
                "physical_event_basis": key[7],
                "physical_event_year": key[8],
                "score": score,
                "contribution_norm": norm,
            }
        )
    observed = rows.frame()
    stage_keys = [
        column for column in observed.columns if column not in {"score", "contribution_norm"}
    ]
    observed = _stable_grouped_sum(observed, stage_keys)
    complete = _frame_from_records(
        {
            "region": region,
            "year": year,
            **_method_dimensions(method),
            "life_cycle_stage": spec.demand_type,
            "stage_group": spec.stage_group,
            "cutoff_cycle_role": spec.cutoff_cycle_role,
            "material_cycle_source": spec.material_cycle_source,
            "physical_event_basis": spec.physical_event_basis,
            "physical_event_year": year,
        }
        for region in state.regions
        for year in range(2019, 2101)
        for method in state.methods
        for spec in BURDEN_SPECS
    )
    regional = complete.merge(
        observed,
        on=stage_keys,
        how="left",
    )
    regional[["score", "contribution_norm"]] = regional[["score", "contribution_norm"]].fillna(0.0)
    global_keys = [column for column in stage_keys if column != "region"]
    global_rows = _stable_grouped_sum(regional, global_keys)
    global_rows.insert(0, "region", GLOBAL_REGION)
    annual = pd.concat((regional, global_rows), ignore_index=True)
    annual["period_start_year"] = annual["year"]
    annual["period_end_year"] = annual["year"]
    annual_denominators = [
        "region",
        "year",
        "method_id",
        "method_version",
        "impact_category_id",
        "impact_category_tuple_json",
        "unit",
        "period_start_year",
        "period_end_year",
    ]
    annual = _with_stage_score_shares(annual, annual_denominators)

    cumulative_keys = [
        column
        for column in annual.columns
        if column
        not in {
            "year",
            "physical_event_year",
            "period_start_year",
            "period_end_year",
            "score",
            "contribution_norm",
            "share",
            "norm_share",
            "share_status",
        }
    ]
    cumulative = _stable_grouped_sum(annual, cumulative_keys)
    cumulative["year"] = pd.NA
    cumulative["physical_event_year"] = pd.NA
    cumulative["period_start_year"] = 2019
    cumulative["period_end_year"] = 2100
    cumulative_denominators = [
        "region",
        "method_id",
        "method_version",
        "impact_category_id",
        "impact_category_tuple_json",
        "unit",
        "period_start_year",
        "period_end_year",
    ]
    cumulative = _with_stage_score_shares(cumulative, cumulative_denominators)
    annual["year"] = annual["year"].astype("int64")
    annual["physical_event_year"] = annual["physical_event_year"].astype("int64")
    cumulative["year"] = cumulative["year"].astype("Int64")
    cumulative["physical_event_year"] = cumulative["physical_event_year"].astype("Int64")
    return annual, cumulative


def _stable_grouped_sum(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Group contribution rows using reproducible floating-point summation."""
    return (
        frame.groupby(
            list(group_columns),
            sort=True,
            observed=True,
            dropna=False,
        )[["score", "contribution_norm"]]
        .agg(_stable_sum)
        .reset_index()
    )


def _stable_sum(values: pd.Series) -> float:
    """Return a reproducible floating-point sum."""
    return fsum(float(value) for value in values)


def derive_global_contributions(regional: pd.DataFrame) -> pd.DataFrame:
    """Calculate global contribution values from the 12 world regions."""

    group_columns = [
        column
        for column in regional.columns
        if column
        not in {
            "region",
            "score",
            "contribution_norm",
            "share",
            "norm_share",
            "share_status",
        }
    ]
    global_rows = _stable_grouped_sum(regional, group_columns)
    global_rows.insert(0, "region", GLOBAL_REGION)
    return pd.concat((regional, global_rows), ignore_index=True)


def _with_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate score and ``contribution_norm`` shares within each group."""
    stage = frame.loc[frame["contribution_view"].eq("life_cycle_stage")]
    denominator_keys = (
        "region",
        "year",
        "method_id",
        "method_version",
        "impact_category_id",
        "impact_category_tuple_json",
        "unit",
    )
    denominators = _stable_grouped_sum(stage, denominator_keys).rename(
        columns={
            "score": "total_score",
            "contribution_norm": "total_norm",
        }
    )
    output = frame.merge(
        denominators,
        on=list(denominator_keys),
        how="left",
    )
    statuses = []
    shares = []
    norm_shares = []
    for raw_row in output.itertuples(index=False):
        row = cast(Any, raw_row)
        total_score = float(row.total_score)
        total_norm = float(row.total_norm)
        status = _stage_share_status(total_score, total_norm)
        statuses.append(status)
        shares.append(float(row.score) / total_score if status == "available" else np.nan)
        norm_shares.append(
            float(row.contribution_norm) / total_norm if total_norm != 0.0 else np.nan
        )
    output["share"] = shares
    output["norm_share"] = norm_shares
    output["share_status"] = statuses
    return output.drop(columns=["total_score", "total_norm"])


def _with_stage_contribution_shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate contribution shares within each life-cycle stage."""
    denominator_keys = (
        "region",
        "year",
        "method_id",
        "method_version",
        "impact_category_id",
        "impact_category_tuple_json",
        "unit",
        "life_cycle_stage",
        "stage_group",
        "cutoff_cycle_role",
        "material_cycle_source",
        "physical_event_basis",
        "physical_event_year",
        "source_flow_names",
        "lca_fate",
        "fate_origin",
    )
    stage = frame.loc[frame["contribution_view"].eq("life_cycle_stage")]
    denominators = _stable_grouped_sum(stage, denominator_keys).rename(
        columns={"score": "stage_score", "contribution_norm": "stage_norm"}
    )
    output = frame.merge(
        denominators,
        on=list(denominator_keys),
        how="left",
    )
    statuses = [
        _stage_share_status(float(item.stage_score), float(item.stage_norm))
        for row in output.itertuples(index=False)
        for item in (cast(Any, row),)
    ]
    output["share_status"] = statuses
    output["share"] = [
        float(score) / float(total) if status == "available" else np.nan
        for score, total, status in zip(
            output["score"], output["stage_score"], statuses, strict=True
        )
    ]
    output["norm_share"] = [
        float(norm) / float(total) if float(total) != 0.0 else np.nan
        for norm, total in zip(output["contribution_norm"], output["stage_norm"], strict=True)
    ]
    return output.drop(columns=["stage_score", "stage_norm"])


def annual_contribution_norms(state: ContributionState) -> pd.DataFrame:
    """Return annual LCIA scores and ``contribution_norm`` values by region."""

    rows = []
    for region in state.regions:
        for year in range(2019, 2101):
            for method_index, method in enumerate(state.methods):
                score, norm = state.annual_totals.get(
                    (region, year, method_index),
                    (0.0, 0.0),
                )
                rows.append(
                    {
                        "region": region,
                        "year": year,
                        **_method_dimensions(method),
                        "score": score,
                        "contribution_norm": norm,
                    }
                )
    regional = pd.DataFrame(rows)
    return derive_global_contributions(regional)


def attach_contribution_norms(
    regional_scores: pd.DataFrame,
    regional_annual_scores: pd.DataFrame,
    annual_norms: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach ``contribution_norm`` and calculate global results.

    Args:
        regional_scores: Cumulative LCIA scores by world region and LCIA category.
        regional_annual_scores: Annual LCIA scores by world region, year, and LCIA
            category from 2019 through 2100.
        annual_norms: Annual scores and ``contribution_norm`` values with the same
            regional, temporal, and LCIA coordinates. ``contribution_norm`` is the
            sum of the absolute values of the characterized contributions in the
            LCIA category's unit.

    Returns:
        Cumulative and annual score tables containing ``contribution_norm``,
        with global rows derived from the twelve regional values.

    Notes:
        Global values are calculated once from the twelve regional rows.
    """

    method_keys = (
        "region",
        "method_id",
        "method_version",
        "impact_category_id",
        "impact_category_tuple_json",
        "unit",
    )
    regional_norms = annual_norms.loc[annual_norms["region"].ne(GLOBAL_REGION)]
    annual = regional_annual_scores.merge(
        regional_norms.loc[
            :,
            [*method_keys, "year", "contribution_norm"],
        ],
        on=[*method_keys, "year"],
        how="left",
    )
    totals = (
        regional_norms.groupby(
            list(method_keys),
            sort=True,
            observed=True,
        )[["contribution_norm"]]
        .agg(_stable_sum)
        .reset_index()
    )
    scores = regional_scores.merge(
        totals,
        on=list(method_keys),
        how="left",
    )
    score_group = [
        column
        for column in scores.columns
        if column not in {"region", "score", "contribution_norm"}
    ]
    global_scores = _stable_grouped_sum(scores, score_group)
    global_scores.insert(
        cast(int, scores.columns.get_loc("region")),
        "region",
        GLOBAL_REGION,
    )
    annual_group = [
        column
        for column in annual.columns
        if column not in {"region", "score", "contribution_norm"}
    ]
    global_annual = _stable_grouped_sum(annual, annual_group)
    global_annual.insert(
        cast(int, annual.columns.get_loc("region")),
        "region",
        GLOBAL_REGION,
    )
    return (
        pd.concat((scores, global_scores), ignore_index=True),
        pd.concat((annual, global_annual), ignore_index=True),
    )
