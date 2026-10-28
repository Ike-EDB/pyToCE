"""Construct life cycle contribution tables in batches."""

import json
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd

from .contribution_accumulation import (
    _RECEIVING_DESCRIPTOR_COLUMNS,
    ContributionMethod,
    ContributionState,
    _flow_metadata,
    _StageCoordinate,
)

_RECORD_BATCH_SIZE = 25_000


_DESCRIPTOR_COLUMNS = (
    *_RECEIVING_DESCRIPTOR_COLUMNS,
    "life_cycle_stage",
    "stage_group",
    "cutoff_cycle_role",
    "material_cycle_source",
    "physical_event_basis",
    "physical_event_year",
    "source_flow_names",
    "lca_fate",
    "fate_origin",
    "provider_database",
    "provider_code",
    "activity_database",
    "activity_code",
    "activity_name",
    "activity_location",
    "elementary_flow_database",
    "elementary_flow_code",
    "elementary_flow_name",
    "elementary_flow_categories",
    "elementary_flow_unit",
    "classification",
)


def _stage_descriptors(coordinate: _StageCoordinate) -> dict[str, object]:
    """Map a stage coordinate to the descriptive columns in the result table."""
    return {
        "life_cycle_stage": coordinate[2],
        "stage_group": coordinate[3],
        "cutoff_cycle_role": coordinate[4],
        "material_cycle_source": coordinate[5],
        "physical_event_basis": coordinate[6],
        "physical_event_year": coordinate[7],
        "source_flow_names": coordinate[8],
        "lca_fate": coordinate[9],
        "fate_origin": coordinate[10],
    }


class _BatchedRecords:
    """Bound temporary Python records while preserving exact append order."""

    __slots__ = ("_batches", "_pending")

    def __init__(self) -> None:
        """Start an ordered record buffer with no materialized batches."""
        self._batches: list[pd.DataFrame] = []
        self._pending: list[dict[str, object]] = []

    def append(self, record: dict[str, object]) -> None:
        """Append one complete record and flush full batches to DataFrames."""

        self._pending.append(record)
        if len(self._pending) == _RECORD_BATCH_SIZE:
            self._flush()

    def frame(self) -> pd.DataFrame:
        """Return one frame with the original record and column order."""

        self._flush()
        if not self._batches:
            return pd.DataFrame()
        if len(self._batches) == 1:
            return self._batches.pop()
        frame = pd.concat(self._batches, ignore_index=True)
        self._batches.clear()
        return frame

    def _flush(self) -> None:
        """Convert pending contribution records into one DataFrame batch."""
        if not self._pending:
            return
        self._batches.append(pd.DataFrame.from_records(self._pending))
        self._pending.clear()


def _frame_from_records(records: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Build an ordered frame from bounded record batches."""

    batches = _BatchedRecords()
    for record in records:
        batches.append(record)
    return batches.frame()


def _method_dimensions(
    method: ContributionMethod,
) -> dict[str, object]:
    """Format the LCIA method columns shared by contribution tables."""
    return {
        "method_id": method.method_id,
        "method_version": method.method_version,
        "impact_category_id": method.impact_category_id,
        "impact_category_tuple_json": json.dumps(
            method.method_tuple,
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "unit": method.unit,
    }


def _row(
    state: ContributionState,
    region: str,
    year: int,
    method_index: int,
    view: str,
    role: str,
    score: float,
    norm: float,
    **descriptors: object,
) -> dict[str, object]:
    """Build one contribution row with complete method and descriptor columns."""
    empty_descriptors: dict[str, object] = {column: "" for column in _DESCRIPTOR_COLUMNS}
    empty_descriptors["physical_event_year"] = pd.NA
    row: dict[str, object] = {
        "region": region,
        "year": year,
        **_method_dimensions(state.methods[method_index]),
        "contribution_view": view,
        "record_role": role,
        **empty_descriptors,
        "activation_amount": np.nan,
        "absolute_activation_amount": np.nan,
        "effective_unit_score": np.nan,
        "effective_unit_norm": np.nan,
        "effective_unit_status": "not_applicable",
        "score": score,
        "contribution_norm": norm,
    }
    row.update(descriptors)
    return row


def _direct_rows(
    state: ContributionState,
    coordinate: _StageCoordinate,
    method_index: int,
) -> Iterator[dict[str, object]]:
    """Label modeled emissions in both complete provider and activity views.

    The source authority list is provenance; the provider and native-producer
    identifiers deliberately remain empty. Flow and compartment views aggregate
    these same quantities by flow and compartment, providing alternative views of the same
    contributions. Retaining the stage here keeps different processes legible
    even in the region/year table.
    """

    for (receiving, flow_id), (score, norm) in sorted(
        state.direct_values.get((coordinate, method_index), {}).items()
    ):
        metadata = _flow_metadata(state, flow_id)
        for view in ("provider_frontier", "native_activity"):
            yield _row(
                state,
                coordinate[0],
                coordinate[1],
                method_index,
                view,
                "modelled_direct",
                score,
                norm,
                **_stage_descriptors(coordinate),
                **dict(zip(_RECEIVING_DESCRIPTOR_COLUMNS, receiving, strict=True)),
                elementary_flow_database=metadata[0],
                elementary_flow_code=metadata[1],
                elementary_flow_name=metadata[2],
                elementary_flow_categories=metadata[3],
                elementary_flow_unit=metadata[5],
            )
