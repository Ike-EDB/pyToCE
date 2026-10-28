"""Characterize direct receiving emissions alongside native supplier inventories."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import matrix_utils as mu  # type: ignore[import-untyped]
import numpy as np
from bw2calc.utils import consistent_global_index  # type: ignore[import-untyped]
from numpy.typing import NDArray

from .contribution_accumulation import (
    GLOBAL_REGION,
    ContributionState,
    _add,
    _flow_metadata,
    _merge_flow_values,
    _ReceivingDescriptors,
    _StageCoordinate,
)


@dataclass(frozen=True, slots=True)
class DirectCharacterizationFactors:
    """Registered factors for selected direct flows, in method and flow order.

    ``factors`` and ``present`` have axes ``[method, flow]``. Factors use each
    method's native impact unit per native flow unit. ``present`` records an
    applicable registered row, including an explicit zero; absent factors have
    value zero with ``present=False``. These arrays need no supplier inventory.
    """

    flow_ids: NDArray[np.int64]
    factors: NDArray[np.float64]
    present: NDArray[np.bool_]


def admit_direct_characterization_factors(
    state: ContributionState,
    flow_ids: Sequence[int],
) -> DirectCharacterizationFactors:
    """Read native processed factors for actual direct receiving flows once.

    Args:
        state: Selected project and ordered methods at the existing LCIA owner.
        flow_ids: Admitted native elementary-flow IDs in the required output order.

    Returns:
        Ordered factors and registration presence, independent of supplier
        biosphere coverage. Present factors also enter ``state`` for ordinary
        annual-inventory characterization; supplier solves retain these entries.

    Raises:
        ValueError: A mapped characterization factor for a required direct flow
            is nonfinite.

    Notes:
        Brightway 4.7 processes native method amounts at float32 precision.
        The public matrix_utils 0.9 constructor preserves that precision,
        duplicate summation, and any native flip/rescale semantics. Its finite
        diagonal contains only requested flows. The global-location filter is the same as bw2calc
        2.5.0's conventional method switching. Characterization uses central factor values.
    """
    requested = np.asarray(flow_ids, dtype=np.int64)
    factors = np.zeros((len(state.methods), len(requested)), dtype=np.float64)
    present = np.zeros(factors.shape, dtype=np.bool_)
    for method_index, method in enumerate(state.methods):
        package = state.bd.Method(method.method_tuple).datapackage()
        global_index = consistent_global_index([package])

        def selected(indices: np.ndarray) -> np.ndarray:
            """Keep requested flows at the method's global characterization location."""
            return (indices["col"] == global_index) & np.isin(indices["row"], requested)

        mapped = mu.MappedMatrix(
            packages=[package],
            matrix="characterization_matrix",
            use_arrays=False,
            use_distributions=False,
            diagonal=True,
            custom_filter=selected,
            empty_ok=True,
        )
        positions = mapped.row_mapper.map_array(requested)
        present[method_index] = positions >= 0
        factors[method_index, present[method_index]] = mapped.matrix.diagonal()[
            positions[present[method_index]]
        ]
        if not np.isfinite(factors[method_index]).all():
            raise ValueError(
                f"Brightway method {method.method_tuple!r} has nonfinite processed "
                "characterization factors for required direct flows. Select or register "
                "a method with finite processed factors and run the calculation again."
            )
        state.characterization_factors.setdefault(method.method_tuple, {}).update(
            {
                int(flow_id): float(value)
                for flow_id, value in zip(
                    requested[present[method_index]],
                    factors[method_index, present[method_index]],
                    strict=True,
                )
            }
        )
    return DirectCharacterizationFactors(requested, factors, present)


@dataclass(frozen=True, slots=True)
class DirectContribution:
    """One already grouped actual flow at a process and dated anchor.

    ``amount`` is in ``endpoint_unit``, the registered elementary-flow unit.
    The caller groups the same flow within the actual process before
    applying anchor shares. Origin attribution and ``source_records_json`` are
    retained separately from that grouping.
    The JSON list identifies source authorities by database, code, name, location
    and unit. Original calibration, dated operations or fixed spinning sources
    document the equations and coefficients used by the process models.
    """

    stage_coordinate: _StageCoordinate
    receiving_function: str
    source_records_json: str
    endpoint_unit: str
    flow_id: int
    amount: float


def add_direct_contributions(
    state: ContributionState,
    rows: Iterable[DirectContribution],
) -> None:
    """Accumulate actual direct scores and absolute totals alongside suppliers.

    The caller has admitted native factors once with
    ``admit_direct_characterization_factors``. Missing native factors characterize
    to zero; registered zeros remain present in that factor owner.
    This function uses the supplied flows and admitted characterization factors.
    Each quantity is already summed within its process;
    ``abs(amount * factor)`` is consequently taken before annual/stage reporting,
    but after scientific mixing. Absolute contributions from distinct functions
    and dated anchors are added even when their positive and negative impacts
    sum to zero.
    """

    for row in rows:
        coordinate = row.stage_coordinate
        region, year = coordinate[:2]
        descriptors: _ReceivingDescriptors = (
            row.receiving_function,
            row.source_records_json,
            row.endpoint_unit,
        )
        for method_index, method in enumerate(state.methods):
            factor = state.characterization_factors[method.method_tuple].get(row.flow_id, 0.0)
            score = row.amount * factor
            norm = abs(score)
            key = (region, year, method_index)
            stage_key = (region, year, method_index, *coordinate[2:])
            _add(state.annual_totals, key, score, norm)
            _add(state.annual_stage_values, stage_key, score, norm)
            if year not in state.years:
                continue
            _add(state.stage_values, stage_key, score, norm)
            classification = _flow_metadata(state, row.flow_id)[4]
            _add(state.classification_values, (*key, classification), score, norm)
            identities = np.array([row.flow_id], dtype=np.int64)
            scores = np.array([score], dtype=np.float64)
            norms = np.array([norm], dtype=np.float64)
            _merge_flow_values(state, key, identities, scores, norms)
            # Global stage detail is built from these regional values once, just as native matrix
            # contributions. Regional tables derive their own global rows later.
            for stage in (coordinate, (GLOBAL_REGION, *coordinate[1:])):
                stage_method = (stage, method_index)
                values = state.direct_values.setdefault(stage_method, {})
                _add(values, (descriptors, row.flow_id), score, norm)
                _merge_flow_values(
                    state,
                    stage_method,
                    identities,
                    scores,
                    norms,
                    target=state.stage_flow_values,
                )
                _add(
                    state.stage_classification_values,
                    (stage, method_index, classification),
                    score,
                    norm,
                )
