"""Regional, stage, and global additive contribution finalization."""

import json
from collections.abc import Iterator
from math import fsum
from typing import Any, cast

import numpy as np
import pandas as pd

from .contribution_accumulation import (
    _RECEIVING_DESCRIPTOR_COLUMNS,
    GLOBAL_REGION,
    ContributionState,
    _activity_metadata,
    _add,
    _flow_metadata,
    _FlowValues,
    _merge_flow_values,
    _nonnegative_residual,
    _StageCoordinate,
    _StageMethodKey,
    _top_mapping,
)
from .contribution_records import (
    _BatchedRecords,
    _direct_rows,
    _row,
    _stage_descriptors,
)
from .contribution_structure import (
    _structural_attribution,
    _structural_edges,
    _structural_paths,
)
from .stage_scores import (
    _stage_score_tables,
    _with_shares,
    _with_stage_contribution_shares,
    annual_contribution_norms,
    derive_global_contributions,
)

STAGE_DETAIL_LIMIT = 25


_VIEWS = (
    "life_cycle_stage",
    "provider_frontier",
    "native_activity",
    "elementary_flow",
    "classification",
)


def _append_detail_rows(
    rows: _BatchedRecords,
    state: ContributionState,
    key: tuple[str, int, int],
    total: tuple[float, float],
    direct: tuple[float, float],
    flow_values: _FlowValues | None,
) -> None:
    """Write independently selected native/flow details and their exact residuals."""
    region, year, method_index = key
    native_details = [
        (identity[0], identity[1], values[0], values[1])
        for identity, values in state.native_values.get(key, {}).items()
    ]
    native_details.sort(key=lambda item: (-item[3], item[0], item[1]))
    for provider_database, activity_id, score, norm in native_details:
        database, code, name, location = _activity_metadata(
            state,
            activity_id,
        )
        rows.append(
            _row(
                state,
                region,
                year,
                method_index,
                "native_activity",
                "detail",
                score,
                norm,
                provider_database=provider_database,
                activity_database=database,
                activity_code=code,
                activity_name=name,
                activity_location=location,
            )
        )
    rows.append(
        _row(
            state,
            region,
            year,
            method_index,
            "native_activity",
            "exact_residual",
            fsum((total[0], -direct[0], -fsum(item[2] for item in native_details))),
            _nonnegative_residual(
                total[1],
                fsum((direct[1], fsum(item[3] for item in native_details))),
            ),
        )
    )
    flow_details, residual = _top_mapping(
        flow_values,
        total,
    )
    for flow_id, score, norm in flow_details:
        metadata = _flow_metadata(state, flow_id)
        rows.append(
            _row(
                state,
                region,
                year,
                method_index,
                "elementary_flow",
                "detail",
                score,
                norm,
                elementary_flow_database=metadata[0],
                elementary_flow_code=metadata[1],
                elementary_flow_name=metadata[2],
                elementary_flow_categories=metadata[3],
                elementary_flow_unit=metadata[5],
            )
        )
    rows.append(
        _row(
            state,
            region,
            year,
            method_index,
            "elementary_flow",
            "exact_residual",
            residual[0],
            residual[1],
        )
    )


def _regional_contributions(state: ContributionState) -> pd.DataFrame:
    """Build requested-year contribution rows for each world region and LCIA method.

    Results are reported by life cycle stage, provider, native Brightway
    activity, and elementary flow. Global native/flow ranking uses the complete regional
    contributions.
    """
    rows = _BatchedRecords()
    for stage_key, (score, norm) in sorted(state.stage_values.items()):
        (
            region,
            year,
            method_index,
            stage,
            stage_group,
            cutoff_cycle_role,
            material_cycle_source,
            physical_event_basis,
            physical_event_year,
            source_flows,
            lca_fate,
            fate_origin,
        ) = stage_key
        rows.append(
            _row(
                state,
                str(region),
                int(year),
                int(method_index),
                "life_cycle_stage",
                "detail",
                score,
                norm,
                life_cycle_stage=str(stage),
                stage_group=str(stage_group),
                cutoff_cycle_role=str(cutoff_cycle_role),
                material_cycle_source=str(material_cycle_source),
                physical_event_basis=str(physical_event_basis),
                physical_event_year=int(physical_event_year),
                source_flow_names=str(source_flows),
                lca_fate=str(lca_fate),
                fate_origin=str(fate_origin),
            )
        )
    for provider_key, (score, norm) in sorted(state.provider_values.items()):
        region, year, method_index, database, code, receiving = provider_key
        rows.append(
            _row(
                state,
                str(region),
                int(year),
                int(method_index),
                "provider_frontier",
                "detail",
                score,
                norm,
                provider_database=str(database),
                provider_code=str(code),
                **dict(zip(_RECEIVING_DESCRIPTOR_COLUMNS, receiving, strict=True)),
            )
        )
    direct_totals: dict[tuple[str, int, int], tuple[float, float]] = {}
    for (coordinate, method_index), values in sorted(state.direct_values.items()):
        if coordinate[0] == GLOBAL_REGION:
            continue
        for row in _direct_rows(state, coordinate, method_index):
            rows.append(row)
        _add(
            direct_totals,
            (*coordinate[:2], method_index),
            fsum(value[0] for value in values.values()),
            fsum(value[1] for value in values.values()),
        )
    for region, year in state.groups:
        for method_index in range(len(state.methods)):
            key = (region, year, method_index)
            total = state.annual_totals.get(key, (0.0, 0.0))
            direct = direct_totals.get(key, (0.0, 0.0))
            _append_detail_rows(rows, state, key, total, direct, state.flow_values.get(key))
    for classification_key, (score, norm) in sorted(state.classification_values.items()):
        region, year, method_index, classification = classification_key
        rows.append(
            _row(
                state,
                str(region),
                int(year),
                int(method_index),
                "classification",
                "detail",
                score,
                norm,
                classification=str(classification),
            )
        )
    regional = rows.frame()
    existing = {
        (
            str(row.region),
            int(cast(Any, row.year)),
            str(row.impact_category_tuple_json),
            str(row.contribution_view),
        )
        for row in regional.itertuples(index=False)
    }
    missing = _BatchedRecords()
    for region, year in state.groups:
        for method_index, method in enumerate(state.methods):
            method_tuple_json = json.dumps(
                method.method_tuple,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            for view in _VIEWS:
                view_key = (region, year, method_tuple_json, view)
                if view_key not in existing:
                    missing.append(
                        _row(
                            state,
                            region,
                            year,
                            method_index,
                            view,
                            "exact_residual",
                            0.0,
                            0.0,
                        )
                    )
    remainder = missing.frame()
    if remainder.empty:
        return regional
    return pd.concat((regional, remainder), ignore_index=True)


def _global_detail_contributions(state: ContributionState) -> pd.DataFrame:
    """Rank complete global native and flow values independently of regional detail.

    Native values were reduced before matrix activity pruning. Flow arrays still
    contain every regional supplier and direct contribution; merge one global
    year/method at a time and release it after constructing its bounded rows.
    """
    rows = _BatchedRecords()
    direct_totals: dict[tuple[str, int, int], tuple[float, float]] = {}
    for (coordinate, method_index), values in sorted(state.direct_values.items()):
        if coordinate[0] != GLOBAL_REGION:
            continue
        for row in _direct_rows(state, coordinate, method_index):
            if row["contribution_view"] == "native_activity":
                rows.append(row)
        _add(
            direct_totals,
            (*coordinate[:2], method_index),
            fsum(value[0] for value in values.values()),
            fsum(value[1] for value in values.values()),
        )
    for year in state.years:
        for method_index in range(len(state.methods)):
            key = (GLOBAL_REGION, year, method_index)
            regional_keys = [(region, year, method_index) for region in state.regions]
            total = (
                fsum(
                    state.annual_totals.get(regional, (0.0, 0.0))[0] for regional in regional_keys
                ),
                fsum(
                    state.annual_totals.get(regional, (0.0, 0.0))[1] for regional in regional_keys
                ),
            )
            flows: dict[tuple[str, int, int], _FlowValues] = {}
            for regional in regional_keys:
                regional_flows = state.flow_values.get(regional)
                if regional_flows is not None:
                    _merge_flow_values(
                        state,
                        key,
                        regional_flows.identities,
                        regional_flows.scores,
                        regional_flows.norms,
                        target=flows,
                    )
            _append_detail_rows(
                rows, state, key, total, direct_totals.get(key, (0.0, 0.0)), flows.get(key)
            )
            flows.clear()
    return rows.frame()


def _selected_stage_totals(
    state: ContributionState,
) -> dict[_StageMethodKey, tuple[float, float]]:
    """Collect requested-year regional stage totals and derive their global sums."""
    totals: dict[_StageMethodKey, tuple[float, float]] = {}
    for key, (score, norm) in state.stage_values.items():
        coordinate: _StageCoordinate = (
            str(key[0]),
            int(key[1]),
            str(key[3]),
            str(key[4]),
            str(key[5]),
            str(key[6]),
            str(key[7]),
            int(key[8]),
            str(key[9]),
            str(key[10]),
            str(key[11]),
        )
        method_index = int(key[2])
        _add(totals, (coordinate, method_index), score, norm)
        _add(
            totals,
            ((GLOBAL_REGION, *coordinate[1:]), method_index),
            score,
            norm,
        )
    return totals


def _provider_effective_values(
    amount: float,
    absolute_amount: float,
    score: float,
    norm: float,
) -> tuple[float, float, str]:
    """Return provider values per unit of total activity.

    ``zero_total_activity`` means all retained amounts are zero; both per-unit values are undefined.
    ``near_zero_total_activity`` means the absolute
    signed total is at most 1e-10 of the sum of absolute activity amounts;
    cancellation prevents a reliable score per unit. Otherwise the status is
    ``available``. These criteria are independent of the activity's native unit.
    """
    if absolute_amount == 0.0:
        return np.nan, np.nan, "zero_total_activity"
    if abs(amount) / absolute_amount <= 1e-10:
        return np.nan, norm / absolute_amount, "near_zero_total_activity"
    return score / amount, norm / absolute_amount, "available"


def _regional_stage_contributions(state: ContributionState) -> pd.DataFrame:
    """Build stage contribution details for 12 world regions and their derived global result.

    Each row retains the activity year and stage descriptors attached to
    the foreground activation. All provider and modeled direct contributions
    are retained; native activity and elementary-flow details are ranked by
    ``contribution_norm``.
    """
    totals = _selected_stage_totals(state)
    rows = _BatchedRecords()
    for (coordinate, method_index), (score, norm) in sorted(totals.items()):
        region, year = coordinate[:2]
        descriptors = _stage_descriptors(coordinate)
        rows.append(
            _row(
                state,
                region,
                year,
                method_index,
                "life_cycle_stage",
                "detail",
                score,
                norm,
                **descriptors,
            )
        )

        provider_keys = sorted(
            (
                key
                for key in state.stage_provider_values
                if key[0] == coordinate and key[1] == method_index
            ),
            key=lambda key: (key[2], key[3], key[4]),
        )
        for key in provider_keys:
            provider_score, provider_norm = state.stage_provider_values[key]
            amount, absolute_amount = state.stage_provider_amounts[
                (coordinate, key[2], key[3], key[4])
            ]
            unit_score, unit_norm, unit_status = _provider_effective_values(
                amount,
                absolute_amount,
                provider_score,
                provider_norm,
            )
            rows.append(
                _row(
                    state,
                    region,
                    year,
                    method_index,
                    "provider_frontier",
                    "detail",
                    provider_score,
                    provider_norm,
                    **descriptors,
                    provider_database=key[2],
                    provider_code=key[3],
                    **dict(zip(_RECEIVING_DESCRIPTOR_COLUMNS, key[4], strict=True)),
                    activation_amount=amount,
                    absolute_activation_amount=absolute_amount,
                    effective_unit_score=unit_score,
                    effective_unit_norm=unit_norm,
                    effective_unit_status=unit_status,
                )
            )

        for row in _direct_rows(state, coordinate, method_index):
            rows.append(row)
        direct_values = state.direct_values.get((coordinate, method_index), {})
        direct_score = fsum(value[0] for value in direct_values.values())
        direct_norm = fsum(value[1] for value in direct_values.values())
        native_values = state.stage_native_values.get((coordinate, method_index), {})
        native_details = sorted(
            (
                (identity[0], identity[1], values[0], values[1])
                for identity, values in native_values.items()
            ),
            key=lambda item: (-item[3], item[0], item[1]),
        )[:STAGE_DETAIL_LIMIT]
        for provider_database, activity_id, native_score, native_norm in native_details:
            database, code, name, location = _activity_metadata(state, activity_id)
            rows.append(
                _row(
                    state,
                    region,
                    year,
                    method_index,
                    "native_activity",
                    "detail",
                    native_score,
                    native_norm,
                    **descriptors,
                    provider_database=provider_database,
                    activity_database=database,
                    activity_code=code,
                    activity_name=name,
                    activity_location=location,
                )
            )
        rows.append(
            _row(
                state,
                region,
                year,
                method_index,
                "native_activity",
                "exact_residual",
                fsum((score, -direct_score, -fsum(item[2] for item in native_details))),
                _nonnegative_residual(
                    norm,
                    fsum((direct_norm, fsum(item[3] for item in native_details))),
                ),
                **descriptors,
            )
        )

        flow_details, flow_residual = _top_mapping(
            state.stage_flow_values.get((coordinate, method_index)),
            (score, norm),
            STAGE_DETAIL_LIMIT,
        )
        for flow_id, flow_score, flow_norm in flow_details:
            metadata = _flow_metadata(state, flow_id)
            rows.append(
                _row(
                    state,
                    region,
                    year,
                    method_index,
                    "elementary_flow",
                    "detail",
                    flow_score,
                    flow_norm,
                    **descriptors,
                    elementary_flow_database=metadata[0],
                    elementary_flow_code=metadata[1],
                    elementary_flow_name=metadata[2],
                    elementary_flow_categories=metadata[3],
                    elementary_flow_unit=metadata[5],
                )
            )
        rows.append(
            _row(
                state,
                region,
                year,
                method_index,
                "elementary_flow",
                "exact_residual",
                flow_residual[0],
                flow_residual[1],
                **descriptors,
            )
        )

        classification_rows = [
            (key[2], values)
            for key, values in state.stage_classification_values.items()
            if key[0] == coordinate and key[1] == method_index
        ]
        if classification_rows:
            for classification, (class_score, class_norm) in sorted(classification_rows):
                rows.append(
                    _row(
                        state,
                        region,
                        year,
                        method_index,
                        "classification",
                        "detail",
                        class_score,
                        class_norm,
                        **descriptors,
                        classification=classification,
                    )
                )
        else:
            rows.append(
                _row(
                    state,
                    region,
                    year,
                    method_index,
                    "classification",
                    "exact_residual",
                    0.0,
                    0.0,
                    **descriptors,
                )
            )
    return rows.frame()


def finalize_contributions(
    state: ContributionState,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield contribution tables in their publication order.

    The result writer saves completed tables as they are yielded.
    """

    annual_norms = annual_contribution_norms(state)
    yield "annual_norms", annual_norms

    state.characterization_factors.clear()
    state.factor_cache.clear()

    regional = _regional_contributions(state)
    # Only complete views can be summed from their published regional rows.
    # Native and flow details have separate global inputs and selection.
    untruncated = regional.loc[
        ~regional["contribution_view"].isin(("native_activity", "elementary_flow"))
    ]
    derived = derive_global_contributions(untruncated)
    global_rows = derived.loc[derived["region"].eq(GLOBAL_REGION)]
    del derived, untruncated
    global_details = _global_detail_contributions(state)
    state.provider_values.clear()
    state.annual_totals.clear()
    state.native_values.clear()
    state.flow_values.clear()
    state.classification_values.clear()
    contributions = _with_shares(
        pd.concat((regional, global_rows, global_details), ignore_index=True)
    )
    del regional, global_rows, global_details
    contributions["physical_event_year"] = contributions["physical_event_year"].astype("Int64")
    structural_attribution = _structural_attribution(contributions)
    yield "structural_attribution", structural_attribution
    del structural_attribution
    yield "contributions", contributions
    del contributions

    regional_stage = _regional_stage_contributions(state)
    state.activity_metadata.clear()
    state.flow_metadata.clear()
    state.stage_values.clear()
    state.stage_provider_values.clear()
    state.stage_provider_amounts.clear()
    state.stage_native_values.clear()
    state.stage_flow_values.clear()
    state.stage_classification_values.clear()
    state.direct_values.clear()
    stage_contributions = _with_stage_contribution_shares(regional_stage)
    del regional_stage
    stage_contributions["physical_event_year"] = stage_contributions["physical_event_year"].astype(
        "Int64"
    )
    yield "stage_contributions", stage_contributions
    del stage_contributions

    annual_stage_scores, cumulative_stage_scores = _stage_score_tables(state)
    state.annual_stage_values.clear()
    yield "cumulative_stage_scores", cumulative_stage_scores
    del cumulative_stage_scores
    yield "annual_stage_scores", annual_stage_scores
    del annual_stage_scores

    structural_paths = _structural_paths(state)
    yield "structural_paths", structural_paths
    del structural_paths
    structural_edges = _structural_edges(state)
    state.structural_values.clear()
    yield "structural_edges", structural_edges
    del structural_edges
