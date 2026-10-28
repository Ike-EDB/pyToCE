"""Array calculations that apply dated provider unit responses."""

from dataclasses import dataclass
from hashlib import sha256
from math import fsum
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pytoce.run_lca.calculation.foreground import H12_REGIONS, optimization_activation_delta
from pytoce.run_optimization.calculation.candidate_authority import (
    _METHOD_SIGNATURE_COLUMNS,
    CandidateParentAuthority,
    CandidateParentMember,
    _signatures,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    ResponseBankProjection,
)
from pytoce.run_optimization.input.allocated_cc_product import (
    optimization_lcia_method_compatibility_identity,
)
from pytoce.shared.runtime.filesystem import canonical_table_sha256, payload_sha256

_YEARS = tuple(range(2019, 2101))
_BASELINE_COLUMNS = (
    "background_scenario",
    "region",
    "year",
    *_METHOD_SIGNATURE_COLUMNS,
    "score",
    "contribution_norm",
)
_DATED_PROJECTION_IDENTITY_PREFIX = "pytoce_optimization_dated_projection_v2::"

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CandidateParentScoreProjection:
    """Store annual baseline LCIA results for the selected REMIND backgrounds."""

    role: Literal["optimization", "mandatory_climate", "result"]
    identity: str
    physical_parent_identity: str
    annual_scores_identity: str
    method_registry_identity: str
    member_parents: tuple[CandidateParentMember, ...]
    annual_scores: pd.DataFrame
    method_registry: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _DatedProjection:
    """Store provider responses by background, year, service, and LCIA registry entry.

    Attributes:
        identity: Content identifier calculated from all method, service, and response inputs.
        projection_identity: Content identifier of the selected response-table rows.
        backgrounds: Selected REMIND members in canonical array order.
        response_bank_identity: Content identifier of the complete saved response bank.
        method_registry_identity: Content identifier of the ordered LCIA method table.
        method_compatibility_identity: Content identifier used to match aCC method rows.
        method_registry: Ordered entries identifying the method, impact category,
            unit, and factor identity.
        services: Ordered provider-service definitions.
        scores: Provider-service LCIA results per unit with axes
            ``[background, year, service, registry entry]`` in native LCIA units.
        norms: Values of ``contribution_norm`` with the same axes and units.
            ``contribution_norm`` is the sum of the absolute values of the
            characterized contributions, in the impact category's unit.
        scores_identity: Content identifier calculated from the shape and values of ``scores``.
        norms_identity: Content identifier calculated from the shape and values of ``norms``.
        services_identity: Content identifier of the ordered provider-service table.
    """

    identity: str
    projection_identity: str
    backgrounds: tuple[str, ...]
    response_bank_identity: str
    method_registry_identity: str
    method_compatibility_identity: str
    method_registry: pd.DataFrame
    services: pd.DataFrame
    scores: FloatArray
    norms: FloatArray
    scores_identity: str
    norms_identity: str
    services_identity: str


def _readonly(values: FloatArray) -> FloatArray:
    """Copy values to a contiguous read-only float array."""
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)


def _array_identity(values: FloatArray) -> str:
    """Identify an array from its shape, dtype, and byte-level SHA-256 digest."""
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    payload = {
        "dtype": "float64",
        "shape": list(contiguous.shape),
        "bytes_sha256": sha256(contiguous.tobytes()).hexdigest(),
    }
    return "pytoce_float64_array_v1::" + payload_sha256(payload)


def _dated_projection(projection: ResponseBankProjection) -> _DatedProjection:
    """Convert response tables to arrays for every occurrence year.

    Provider LCIA results and ``contribution_norm`` values first use axes
    ``[background, provider year, service, impact category]``. Timex shares combine
    the active provider years into arrays with axes
    ``[background, occurrence year, service, impact category]`` for 2019--2100.
    ``contribution_norm`` is the sum of the absolute values of the characterized
    contributions, in the impact category's unit. Returned arrays are read-only;
    their identities bind the array bytes, ordered methods, services, and source
    response selection.
    """
    methods = projection.method_registry.reset_index(drop=True)
    method_registry_identity = canonical_table_sha256(methods)
    method_compatibility_identity = optimization_lcia_method_compatibility_identity(methods)
    services = projection.provider_service_registry
    responses = projection.provider_unit_responses
    weights = projection.timex_year_weights
    backgrounds = tuple(sorted(str(value) for value in weights["background_scenario"].unique()))
    signatures = _signatures(methods)
    service_ids = tuple(services["provider_service_id"].astype(str))
    anchors = tuple(sorted(int(value) for value in responses["provider_anchor_year"].unique()))
    background_index = {value: index for index, value in enumerate(backgrounds)}
    year_index = {value: index for index, value in enumerate(_YEARS)}
    anchor_index = {value: index for index, value in enumerate(anchors)}
    service_index = {value: index for index, value in enumerate(service_ids)}
    method_index = {value: index for index, value in enumerate(signatures)}
    unit_shape = (len(backgrounds), len(anchors), len(service_ids), len(signatures))
    unit_scores = np.empty(unit_shape, dtype=np.float64)
    unit_norms = np.empty(unit_shape, dtype=np.float64)
    for row in responses.itertuples(index=False):
        signature = tuple(str(getattr(row, column)) for column in _METHOD_SIGNATURE_COLUMNS)
        response_coordinate = (
            background_index[str(row.background_scenario)],
            anchor_index[int(cast(Any, row.provider_anchor_year))],
            service_index[str(row.provider_service_id)],
            method_index[signature],
        )
        unit_scores[response_coordinate] = float(cast(Any, row.unit_score))
        unit_norms[response_coordinate] = float(cast(Any, row.unit_contribution_norm))
    weight_tensor = np.zeros((len(backgrounds), len(_YEARS), len(anchors)), dtype=np.float64)
    for row in weights.itertuples(index=False):
        weight_coordinate = (
            background_index[str(row.background_scenario)],
            year_index[int(cast(Any, row.occurrence_year))],
            anchor_index[int(cast(Any, row.provider_anchor_year))],
        )
        weight_tensor[weight_coordinate] = float(cast(Any, row.temporal_provider_share))
    dated_shape = (len(backgrounds), len(_YEARS), len(service_ids), len(signatures))
    dated_scores = np.empty(dated_shape, dtype=np.float64)
    dated_norms = np.empty(dated_shape, dtype=np.float64)
    for background_i in range(len(backgrounds)):
        for year_i in range(len(_YEARS)):
            active = np.flatnonzero(weight_tensor[background_i, year_i])
            score_parts = tuple(
                weight_tensor[background_i, year_i, anchor_i] * unit_scores[background_i, anchor_i]
                for anchor_i in active
            )
            norm_parts = tuple(
                weight_tensor[background_i, year_i, anchor_i] * unit_norms[background_i, anchor_i]
                for anchor_i in active
            )
            dated_scores[background_i, year_i] = np.fromiter(
                (fsum(values) for values in zip(*(part.ravel() for part in score_parts))),
                dtype=np.float64,
                count=len(service_ids) * len(signatures),
            ).reshape(len(service_ids), len(signatures))
            dated_norms[background_i, year_i] = np.fromiter(
                (fsum(values) for values in zip(*(part.ravel() for part in norm_parts))),
                dtype=np.float64,
                count=len(service_ids) * len(signatures),
            ).reshape(len(service_ids), len(signatures))
    dated_scores = _readonly(dated_scores)
    dated_norms = _readonly(dated_norms)
    scores_identity = _array_identity(dated_scores)
    norms_identity = _array_identity(dated_norms)
    services_identity = canonical_table_sha256(services)
    dated_payload = {
        "response_projection_identity": projection.projection_identity,
        "source_method_registry_identity": projection.method_registry_identity,
        "method_registry_identity": method_registry_identity,
        "method_compatibility_identity": method_compatibility_identity,
        "services_identity": services_identity,
        "scores_identity": scores_identity,
        "norms_identity": norms_identity,
    }
    return _DatedProjection(
        identity=_DATED_PROJECTION_IDENTITY_PREFIX + payload_sha256(dated_payload),
        projection_identity=projection.projection_identity,
        backgrounds=backgrounds,
        response_bank_identity=projection.response_bank_identity,
        method_registry_identity=method_registry_identity,
        method_compatibility_identity=method_compatibility_identity,
        method_registry=methods,
        services=services,
        scores=dated_scores,
        norms=dated_norms,
        scores_identity=scores_identity,
        norms_identity=norms_identity,
        services_identity=services_identity,
    )


def _projection_baseline(
    parent: CandidateParentScoreProjection,
    projection: _DatedProjection,
) -> tuple[FloatArray, FloatArray]:
    """Align baseline LCIA results and contribution data to projection coordinates."""
    signatures = _signatures(projection.method_registry)
    backgrounds = projection.backgrounds
    frame = parent.annual_scores
    signature_index = pd.MultiIndex.from_frame(frame.loc[:, list(_METHOD_SIGNATURE_COLUMNS)])
    selected_index = pd.MultiIndex.from_tuples(signatures, names=_METHOD_SIGNATURE_COLUMNS)
    selected = frame.loc[signature_index.isin(selected_index)]
    shape = (len(backgrounds), len(H12_REGIONS), len(_YEARS), len(signatures))
    scores = np.empty(shape, dtype=np.float64)
    norms = np.empty(shape, dtype=np.float64)
    background_index = {value: index for index, value in enumerate(backgrounds)}
    region_index = {value: index for index, value in enumerate(H12_REGIONS)}
    year_index = {value: index for index, value in enumerate(_YEARS)}
    method_index = {value: index for index, value in enumerate(signatures)}
    for row in selected.itertuples(index=False):
        signature = tuple(str(getattr(row, column)) for column in _METHOD_SIGNATURE_COLUMNS)
        coordinate = (
            background_index[str(row.background_scenario)],
            region_index[str(row.region)],
            year_index[int(cast(Any, row.year))],
            method_index[signature],
        )
        scores[coordinate] = float(cast(Any, row.score))
        norms[coordinate] = float(cast(Any, row.contribution_norm))
    return scores, norms


def _group_delta(
    delta: pd.DataFrame,
    services: pd.DataFrame,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Group provider activation changes by region, year, and service.

    For changed coordinates, return candidate-minus-baseline kg, baseline kg and
    candidate kg. Service demands are nonnegative quantities; native production signs
    remain inside prepared activities. The same quantities therefore scale both
    signed unit scores and absolute contribution totals. Keep both endpoints to preserve a small
    candidate score when subtracting a large baseline.
    Duplicate coordinates use ``math.fsum``. Empty
    changes return zero arrays with axes ``[world region, year, service]``.
    """
    shape = (len(H12_REGIONS), len(_YEARS), len(services))
    arrays = (
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
    )
    if delta.empty:
        return arrays
    selected = delta.loc[delta["delta_amount_kg"].ne(0.0)].copy()
    region_index = {value: index for index, value in enumerate(H12_REGIONS)}
    year_index = {value: index for index, value in enumerate(_YEARS)}
    service_index = {
        str(value): index for index, value in enumerate(services["provider_service_id"])
    }
    selected["_region_i"] = selected["region"].astype(str).map(region_index)
    selected["_year_i"] = selected["year"].map(year_index)
    selected["_service_i"] = selected["provider_service_id"].astype(str).map(service_index)
    region_i = selected["_region_i"].to_numpy(dtype=np.int64)
    year_i = selected["_year_i"].to_numpy(dtype=np.int64)
    service_i = selected["_service_i"].to_numpy(dtype=np.int64)
    flat_coordinates = (region_i * len(_YEARS) + year_i) * len(services) + service_i
    values = selected[
        [
            "delta_amount_kg",
            "parent_amount_kg",
            "candidate_amount_kg",
        ]
    ].to_numpy(dtype=np.float64)
    order = np.argsort(flat_coordinates, kind="stable")
    ordered_coordinates = flat_coordinates[order]
    ordered_values = values[order]
    unique_coordinates, starts, counts = np.unique(
        ordered_coordinates,
        return_index=True,
        return_counts=True,
    )
    flattened = tuple(array.reshape(-1) for array in arrays)
    single = counts == 1
    for value_i, target in enumerate(flattened):
        target[unique_coordinates[single]] = ordered_values[starts[single], value_i]
    for flat_coordinate, start, count in zip(
        unique_coordinates[~single],
        starts[~single],
        counts[~single],
        strict=True,
    ):
        stop = int(start + count)
        for value_i, target in enumerate(flattened):
            target[int(flat_coordinate)] = fsum(
                float(value) for value in ordered_values[int(start) : stop, value_i]
            )
    return arrays


def _score_arrays(
    baseline_scores: FloatArray,
    baseline_norms: FloatArray,
    projection: _DatedProjection,
    delta: pd.DataFrame,
    *,
    pa_parent_scores: FloatArray,
    pa_parent_norms: FloatArray,
    pa_candidate_scores: FloatArray,
    pa_candidate_norms: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply service and PA score differences to the baseline.

    Use candidate scores and absolute contribution totals directly where the
    changed activities account for the complete parent."""
    delta_q, parent_q, candidate_q = _group_delta(delta, projection.services)
    response = _stable_service_response(delta_q, projection.scores)
    adjustment = _stable_service_response(delta_q, projection.norms)
    changed_parent_norm = _stable_service_response(parent_q, projection.norms)
    changed_candidate_norm = _stable_service_response(candidate_q, projection.norms)
    changed_parent_score = _stable_service_response(parent_q, projection.scores)
    changed_candidate_score = _stable_service_response(candidate_q, projection.scores)
    response += pa_candidate_scores - pa_parent_scores
    adjustment += pa_candidate_norms - pa_parent_norms
    return _apply_score_adjustments(
        baseline_scores,
        baseline_norms,
        score_adjustment=response,
        norm_adjustment=adjustment,
        changed_parent_scores=changed_parent_score + pa_parent_scores,
        changed_parent_norms=changed_parent_norm + pa_parent_norms,
        changed_candidate_scores=changed_candidate_score + pa_candidate_scores,
        changed_candidate_norms=changed_candidate_norm + pa_candidate_norms,
    )


def _apply_score_adjustments(
    baseline_scores: FloatArray,
    baseline_norms: FloatArray,
    *,
    score_adjustment: FloatArray,
    norm_adjustment: FloatArray,
    changed_parent_scores: FloatArray,
    changed_parent_norms: FloatArray,
    changed_candidate_scores: FloatArray,
    changed_candidate_norms: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply LCIA changes, using the candidate directly for complete replacements.

    All arrays share coordinates and native LCIA units. Contribution totals are
    sums of absolute characterized contributions, computed before netting positive and negative
    contributions. They
    establish whether the changed activities account for the complete parent.
    Reusing the candidate sum in that case prevents a large subtracted parent
    from erasing a small candidate impact. This is a floating-point calculation
    rule shared by search, annual characterization and stage reporting; MFA feasibility and ASR
    constraints use their declared limits.
    """
    scores = baseline_scores.copy()
    norms = baseline_norms.copy()
    # A zero adjustment preserves the parent's exact stored value, including
    # signed zero, because score identities include array bits.
    np.add(scores, score_adjustment, out=scores, where=score_adjustment != 0.0)
    np.add(norms, norm_adjustment, out=norms, where=norm_adjustment != 0.0)
    # Sixteen float64 rounding units retain the existing contribution-total
    # matching rule. The score test retains the accepted reporting precision:
    # 1e-12 LCIA units absolute or 1e-10 relative to the participating totals.
    roundoff = (
        16.0
        * np.finfo(np.float64).eps
        * np.maximum(np.abs(baseline_norms), np.abs(changed_parent_norms))
    )
    score_roundoff = np.maximum.reduce(
        (
            np.full_like(baseline_scores, 1e-12),
            np.abs(baseline_scores) * 1e-10,
            np.abs(changed_parent_scores) * 1e-10,
            baseline_norms * 1e-10,
            changed_parent_norms * 1e-10,
        )
    )
    endpoint_covered = (
        np.isfinite(changed_parent_scores)
        & np.isfinite(changed_candidate_scores)
        & np.isfinite(changed_parent_norms)
        & np.isfinite(changed_candidate_norms)
        & (changed_parent_norms > 0.0)
        & (changed_candidate_norms >= 0.0)
        & (np.abs(baseline_scores - changed_parent_scores) <= score_roundoff)
        & (np.abs(baseline_norms - changed_parent_norms) <= roundoff)
    )
    scores[endpoint_covered] = changed_candidate_scores[endpoint_covered]
    norms[endpoint_covered] = changed_candidate_norms[endpoint_covered]
    return scores, norms


def _stable_service_response(amounts: FloatArray, unit_values: FloatArray) -> FloatArray:
    """Sum provider-service responses deterministically."""

    region_count, year_count, _service_count = amounts.shape
    background_count, _unit_year_count, _unit_service_count, method_count = unit_values.shape
    result = np.zeros(
        (background_count, region_count, year_count, method_count),
        dtype=np.float64,
    )
    correction = np.zeros_like(result)
    active_services = np.flatnonzero(np.any(amounts != 0.0, axis=(0, 1)))
    for service_i in active_services:
        term = (
            amounts[np.newaxis, :, :, service_i, np.newaxis]
            * unit_values[:, np.newaxis, :, service_i, :]
        )
        updated = result + term
        correction += np.where(
            np.abs(result) >= np.abs(term),
            (result - updated) + term,
            (term - updated) + result,
        )
        result = updated
    result += correction
    return result


def _baseline_service_quantities(
    physical_parent: CandidateParentAuthority,
    services: pd.DataFrame,
) -> FloatArray:
    """Sum nonnegative baseline service demands in kg by region and year."""

    activations = physical_parent.activations
    zero = activations.copy(deep=True)
    zero["amount"] = 0.0
    baseline_delta = optimization_activation_delta(zero, activations)
    delta_q, _parent_q, _candidate_q = _group_delta(baseline_delta, services)
    return _readonly(delta_q)
