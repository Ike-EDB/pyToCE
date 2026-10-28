"""Characterize complete actual PA events with the prepared native response bank.

The receiving owners calculate each mixture at each active source anchor before
temporal weighting. This owner applies the selected methods together. Absolute
contribution totals include each supplier's full supply chain; taking the
absolute value of its net impact would understate opposing contributions.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum
from typing import cast

import numpy as np
import pandas as pd

from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
)
from pytoce.run_lca.calculation.burdens import pa_event_burden
from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_lca.calculation.pa_event_receiving import receive_pa_request
from pytoce.run_lca.calculation.pa_projection import (
    PaProjectedRequest,
    group_pa_receiving_quantities,
    iter_pa_projection_requests,
)
from pytoce.run_lca.calculation.pa_receiving import (
    PaDisposalEvent,
    PaReceivingAmount,
    PaRecyclingEvent,
    PaSpinningEvent,
)
from pytoce.run_lca.calculation.providers import PreparedProviderProduct

from .candidate_projection_arrays import _YEARS, FloatArray
from .response_bank_contract import _METHOD_SIGNATURE_COLUMNS, ResponseBankProjection

_NATIVE_KEYS = (
    "background_scenario",
    "source_anchor_year",
    "source_database",
    "matrix_database",
    "endpoint_database",
    "endpoint_code",
    "endpoint_unit",
)
_FLOW_KEYS = ("flow_database", "flow_code", "endpoint_unit")
# This is the existing selected-result stage grain. Finer source, fate and
# constituent provenance remains with the actual receiving inventory owner.
_STAGE_COLUMNS = (
    "life_cycle_stage",
    "stage_group",
    "cutoff_cycle_role",
    "material_cycle_source",
    "physical_event_basis",
    "physical_event_year",
)
_NativeKey = tuple[str, int, str, str, str, str, str]
_StageKey = tuple[str, str, int, str, str, str, str, str, int]


@dataclass(frozen=True, slots=True)
class PaCandidateProjection:
    """One method-selected response lookup and its completed source parents.

    Background order is the supplied prepared-member order: the public caller
    supplies each selected REMIND background scenario. Method and impact category
    entries retain the selected registry order. The response projection and
    prepared-parent identities own this context; the projection uses those completed products.
    """

    response_projection_identity: str
    prepared_receiving_identity: str
    backgrounds: tuple[str, ...]
    method_registry: pd.DataFrame
    providers_by_background: Mapping[str, PreparedProviderProduct]
    dump_shares: Mapping[str, float]
    native_responses: dict[_NativeKey, tuple[FloatArray, FloatArray]]
    direct_factors: dict[tuple[str, str, str], FloatArray]


@dataclass(frozen=True, slots=True)
class PaCandidateScores:
    """Actual scores and absolute totals before any candidate-parent subtraction.

    Arrays are indexed by background, world region, year (2019--2100), and
    method-registry entry. Each entry identifies a method, impact category and unit.
    ``stage_scores`` retains actual sparse stage coordinates in the existing
    selected-reporting grain. ``score`` is the net impact and
    ``contribution_norm`` is the total absolute contribution, both in method
    units. A parent is evaluated once by the caller; its impact and absolute
    contribution total are subtracted separately. Impact changes and changes in absolute
    contribution totals are calculated separately.
    """

    annual_scores: FloatArray
    annual_contribution_norms: FloatArray
    stage_scores: pd.DataFrame


def build_pa_candidate_projection(
    projection: ResponseBankProjection,
    providers_by_background: Mapping[str, PreparedProviderProduct],
    dump_shares: Mapping[str, float],
    *,
    prepared_receiving_identity: str,
) -> PaCandidateProjection:
    """Index native responses and factors once in the selected method order.

    Source admission and bank construction already guarantee complete keys and
    units. Activity reference-product normalization is in the native response;
    the projection uses those normalized responses directly. Explicit zero and absent CFs
    both use their admitted zero value, while their distinction stays in the
    bank's factor table and identity.
    """
    methods = projection.method_registry
    signatures = list(_METHOD_SIGNATURE_COLUMNS)
    method_index = pd.MultiIndex.from_frame(methods.loc[:, signatures])
    native = {}
    for key, frame in projection.native_unit_responses.groupby(list(_NATIVE_KEYS), sort=False):
        ordered = frame.set_index(signatures).loc[method_index]
        native[cast(_NativeKey, key)] = (
            ordered["unit_score"].to_numpy(dtype=np.float64),
            ordered["unit_contribution_norm"].to_numpy(dtype=np.float64),
        )
    direct = {}
    for key, frame in projection.direct_characterization_factors.groupby(
        list(_FLOW_KEYS), sort=False
    ):
        ordered = frame.set_index(signatures).loc[method_index]
        direct[cast(tuple[str, str, str], key)] = ordered["factor"].to_numpy(dtype=np.float64)
    return PaCandidateProjection(
        projection.projection_identity,
        prepared_receiving_identity,
        tuple(providers_by_background),
        methods,
        providers_by_background,
        dump_shares,
        native,
        direct,
    )


def _score_pa_receiving(
    context: PaCandidateProjection,
    background: str,
    projected: PaProjectedRequest,
    amounts: tuple[PaReceivingAmount, ...],
) -> tuple[FloatArray, FloatArray]:
    """Characterize complete quantities in canonical function/endpoint order."""
    score_terms = []
    norm_terms = []
    for row in group_pa_receiving_quantities(amounts):
        endpoint = row.endpoint
        if isinstance(endpoint, PaReceivingActivity):
            response, norm = context.native_responses[
                (
                    background,
                    projected.source_anchor_year,
                    projected.source_database,
                    projected.matrix_database,
                    *endpoint.key,
                    endpoint.unit,
                )
            ]
            with np.errstate(over="ignore", invalid="ignore"):
                amount_scores = row.amount * response
                amount_norms = abs(row.amount) * norm
            finite = np.isfinite(amount_scores) & np.isfinite(amount_norms)
            if not finite.all():
                method = context.method_registry.iloc[int(np.flatnonzero(~finite)[0])]
                event = projected.receiving_request.event
                raise ValueError(
                    "PA receiving produced a nonfinite impact or absolute "
                    "contribution total for "
                    f"background={background!r}, region={event.region!r}, "
                    f"year={event.year}, source_anchor={projected.source_anchor_year}, "
                    f"matrix={projected.matrix_database!r}, endpoint={endpoint.key!r}, "
                    f"amount={row.amount!r} {endpoint.unit}, "
                    f"method={method['method_id']}/{method['impact_category_id']} "
                    f"({method['unit']})."
                )
            score_terms.append(amount_scores.tolist())
            norm_terms.append(amount_norms.tolist())
        else:
            direct = row.amount * context.direct_factors[(*endpoint.key, endpoint.unit)]
            score_terms.append(direct.tolist())
            norm_terms.append(np.abs(direct).tolist())
    # Absolute totals belong to each process before aggregation.
    # Preserve the original float64 terms and their order in both sums.
    return (
        np.array([fsum(column) for column in zip(*score_terms, strict=True)]),
        np.array([fsum(column) for column in zip(*norm_terms, strict=True)]),
    )


def calculate_pa_candidate_projection(
    context: PaCandidateProjection,
    *,
    spinning_events: tuple[PaSpinningEvent, ...],
    recycling_events: tuple[PaRecyclingEvent, ...],
    disposal_events: tuple[PaDisposalEvent, ...],
) -> PaCandidateScores:
    """Calculate actual PA supplier and direct impacts for the selected methods.

    For each process, sum identical endpoint amounts before applying
    their unit impacts and absolute contribution totals. Distinct functions,
    source anchors and dump classes retain separate absolute contributions.
    Each source's temporal share multiplies both resulting totals once;
    the interpolation applies to the calculated impacts.
    The projection uses the prepared endpoint responses.

    Raises:
        ValueError: If a finite native response overflows when multiplied by
            its actual receiving quantity. Finite signed impacts are retained.
    """
    methods = context.method_registry.loc[:, list(_METHOD_SIGNATURE_COLUMNS)]
    shape = (len(context.backgrounds), len(H12_REGIONS), len(_YEARS), len(methods))
    scores = np.zeros(shape, dtype=np.float64)
    norms = np.zeros(shape, dtype=np.float64)
    stages: dict[_StageKey, list[tuple[FloatArray, FloatArray]]] = {}
    # Open burning and dumping use the shared prepared biosphere reference and this call's direct
    # factors. Retain compact unweighted vectors. Original tuple positions identify distinct
    # occurrences and preserve exact inputs, including signed zeros.
    original_scores: dict[tuple[int | None, str], tuple[FloatArray, FloatArray]] = {}
    for background in context.backgrounds:
        providers = context.providers_by_background[background]
        for projected in iter_pa_projection_requests(
            spinning_events=spinning_events,
            recycling_events=recycling_events,
            disposal_events=disposal_events,
            reference=providers.pa_receiving_reference,
            sources_by_year=providers.pa_receiving_sources_by_year,
            matrix_by_year=providers.database_by_year,
            dump_shares=context.dump_shares,
        ):
            request = projected.receiving_request
            event = request.event
            spec = pa_event_burden(event)
            key: _StageKey = (
                background,
                event.region,
                event.year,
                spec.demand_type,
                spec.stage_group,
                spec.cutoff_cycle_role,
                spec.material_cycle_source,
                spec.physical_event_basis,
                event.year,
            )
            original_key = (
                (request.disposal_index, request.infiltration_class)
                if isinstance(event, PaDisposalEvent)
                and event.demand_type
                in {"mismanaged_open_dump_treatment", "mismanaged_open_burning_treatment"}
                else None
            )
            cached = original_scores.get(original_key) if original_key is not None else None
            if cached is None:
                source = providers.pa_receiving_sources_by_year[projected.source_anchor_year]
                receiving = receive_pa_request(
                    request,
                    reference=providers.pa_receiving_reference,
                    sources=source,
                    spinning_providers=source.spinning_providers,
                )
                event_scores, event_norms = _score_pa_receiving(
                    context, background, projected, receiving.amounts
                )
                del receiving
                if original_key is not None:
                    event_scores.setflags(write=False)
                    event_norms.setflags(write=False)
                    original_scores[original_key] = event_scores, event_norms
            else:
                event_scores, event_norms = cached
            stages.setdefault(key, []).append(
                (projected.temporal_share * event_scores, projected.temporal_share * event_norms)
            )
    records = []
    annual: dict[tuple[int, int, int, int], list[tuple[float, float]]] = {}
    method_records = methods.to_dict("records")
    for key, values in sorted(stages.items()):
        background, region, year, *stage = key
        for method_index, method in enumerate(method_records):
            score = fsum(float(value[0][method_index]) for value in values)
            norm = fsum(float(value[1][method_index]) for value in values)
            records.append(
                {
                    "characterization_background_scenario": background,
                    "region": region,
                    "year": year,
                    **method,
                    **dict(zip(_STAGE_COLUMNS, stage, strict=True)),
                    "score": score,
                    "contribution_norm": norm,
                }
            )
            coordinate = (
                context.backgrounds.index(background),
                H12_REGIONS.index(region),
                _YEARS.index(year),
                method_index,
            )
            annual.setdefault(coordinate, []).append((score, norm))
    for coordinate, values in annual.items():
        scores[coordinate] = fsum(value[0] for value in values)
        norms[coordinate] = fsum(value[1] for value in values)
    scores.setflags(write=False)
    norms.setflags(write=False)
    return PaCandidateScores(
        scores,
        norms,
        pd.DataFrame.from_records(
            records,
            columns=(
                "characterization_background_scenario",
                "region",
                "year",
                *_METHOD_SIGNATURE_COLUMNS,
                *_STAGE_COLUMNS,
                "score",
                "contribution_norm",
            ),
        ),
    )
