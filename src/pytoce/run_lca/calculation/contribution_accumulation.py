"""Additive selected-year contributions from the canonical provider solve."""

import json
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from math import fsum
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
from scipy import sparse  # type: ignore[import-untyped]

DETAIL_LIMIT = 100
CONTRIBUTION_METHOD_IDENTITY = "timex_frontier_native_flow_stage_contributions_v10"
GLOBAL_REGION = "global_derived"

_CONTRIBUTION_ACTIVATION_COLUMNS = (
    "region",
    "year",
    "matrix_database",
    "provider_database",
    "provider_code",
    "provider_product_id",
    "amount",
    "regional_root_database",
    "regional_root_code",
    "stage_database",
    "stage_code",
    "life_cycle_stage",
    "stage_group",
    "cutoff_cycle_role",
    "material_cycle_source",
    "physical_event_basis",
    "physical_event_year",
    "source_flow_names",
    "lca_fate",
    "fate_origin",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "ultimate_carbon_origin",
    "good",
)
_Key = TypeVar("_Key", bound=Hashable)
_StageCoordinate = tuple[str, int, str, str, str, str, str, int, str, str, str]
_StageMethodKey = tuple[_StageCoordinate, int]
_ReceivingDescriptors = tuple[str, str, str]
_RECEIVING_DESCRIPTOR_COLUMNS = ("receiving_function", "source_records_json", "endpoint_unit")


@dataclass(frozen=True, slots=True)
class ContributionMethod:
    """Identity and unit of one registered Brightway LCIA category."""

    method_id: str
    method_version: str
    impact_category_id: str
    method_tuple: tuple[str, ...]
    unit: str


@dataclass(slots=True)
class _FlowValues:
    """Flow identities with LCIA scores and ``contribution_norm`` values."""

    identities: np.ndarray
    scores: np.ndarray
    norms: np.ndarray


@dataclass(slots=True)
class ContributionState:
    """Contribution data accumulated across dated provider calculations.

    ``contribution_norm`` is the sum of the absolute values of the characterized
    contributions, expressed in the LCIA category's unit.

    Attributes:
        bd: Open Brightway project used to resolve activity and flow metadata.
        methods: Selected registered LCIA categories and their units.
        years: Years selected for detailed contribution analysis.
        regions: 12 world regions in the foreground result.
        groups: Region and year columns accumulated for detailed contributions.
        activations: Provider demands combined across the regional roots.
        retain_contributions: Whether to retain detail in addition to unit responses.
        stage_values: LCIA scores and ``contribution_norm`` values by stage coordinate.
        annual_stage_values: Values used in annual life cycle stage result tables.
        provider_values: LCIA values by region, year, method, and provider.
        stage_provider_values: Provider values within each stage coordinate.
        stage_provider_amounts: Provider amounts and their absolute values by stage.
        direct_values: Actual direct-flow contributions by stage, method and receiving
            descriptors; source records describe provenance.
        annual_totals: LCIA values by region, year, and LCIA category.
        native_values: Independently ranked regional and global native activity
            contributions by year, accumulated before regional detail selection.
        flow_values: Elementary flow contributions by region and year.
        classification_values: Contributions grouped by biosphere compartment.
        stage_native_values: Largest native activity values within each stage.
        stage_flow_values: Elementary flow values within each stage.
        stage_classification_values: Biosphere-compartment values within each stage.
        structural_values: Provider activation amounts used for path and edge tables.
        activity_metadata: Cached Brightway activity labels by integer identifier.
        flow_metadata: Cached Brightway flow labels and compartments by identifier.
        characterization_factors: Supplier factors and admitted direct-flow factors,
            including their registered zeros, for each Brightway method.
        factor_cache: Factor arrays indexed by the ordered biosphere flow identifiers.
    """

    bd: Any
    methods: tuple[ContributionMethod, ...]
    years: tuple[int, ...]
    regions: tuple[str, ...]
    groups: tuple[tuple[str, int], ...]
    activations: pd.DataFrame
    retain_contributions: bool = True
    stage_values: dict[
        tuple[str, int, int, str, str, str, str, str, int, str, str, str],
        tuple[float, float],
    ] = field(default_factory=dict)
    annual_stage_values: dict[
        tuple[str, int, int, str, str, str, str, str, int, str, str, str],
        tuple[float, float],
    ] = field(default_factory=dict)
    provider_values: dict[
        tuple[str, int, int, str, str, _ReceivingDescriptors], tuple[float, float]
    ] = field(default_factory=dict)
    stage_provider_values: dict[
        tuple[_StageCoordinate, int, str, str, _ReceivingDescriptors], tuple[float, float]
    ] = field(default_factory=dict)
    stage_provider_amounts: dict[
        tuple[_StageCoordinate, str, str, _ReceivingDescriptors], tuple[float, float]
    ] = field(default_factory=dict)
    direct_values: dict[
        _StageMethodKey, dict[tuple[_ReceivingDescriptors, int], tuple[float, float]]
    ] = field(default_factory=dict)
    annual_totals: dict[tuple[str, int, int], tuple[float, float]] = field(default_factory=dict)
    native_values: dict[
        tuple[str, int, int],
        dict[tuple[str, int], tuple[float, float]],
    ] = field(default_factory=dict)
    flow_values: dict[tuple[str, int, int], _FlowValues] = field(default_factory=dict)
    classification_values: dict[tuple[str, int, int, str], tuple[float, float]] = field(
        default_factory=dict
    )
    stage_native_values: dict[
        _StageMethodKey,
        dict[tuple[str, int], tuple[float, float]],
    ] = field(default_factory=dict)
    stage_flow_values: dict[_StageMethodKey, _FlowValues] = field(default_factory=dict)
    stage_classification_values: dict[tuple[_StageCoordinate, int, str], tuple[float, float]] = (
        field(default_factory=dict)
    )
    structural_values: dict[tuple[object, ...], float] = field(default_factory=dict)
    activity_metadata: dict[int, tuple[str, str, str, str]] = field(default_factory=dict)
    flow_metadata: dict[int, tuple[str, str, str, str, str, str]] = field(default_factory=dict)
    characterization_factors: dict[tuple[str, ...], dict[int, float]] = field(default_factory=dict)
    factor_cache: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class MatrixContributionState:
    """Contribution arrays for one dated provider database.

    Attributes:
        parent: Contribution data shared across all dated provider databases.
        matrix_database: Selected matrix context, distinct from endpoint databases.
        biosphere: Biosphere matrix with ``[flow, activity]`` axes.
        absolute_biosphere: Absolute biosphere matrix with the same axes.
        factors: Characterization factors with ``[method, flow]`` axes.
        activity_scores: Unit LCIA scores with ``[method, activity]`` axes.
        activity_norms: Sums of absolute characterized unit contributions with
            ``[method, activity]`` axes.
        activity_ids: Brightway activity identifier for each activity column.
        flow_ids: Brightway flow identifier for each biosphere row.
        supply: Supply by activity, region, and year.
        absolute_supply: Absolute supply by activity, region, and year.
        group_index: Array-column index for each region and year.
        stage_group_index: Array-column index for each detailed stage coordinate.
        stage_supply: Supply by activity and stage coordinate.
        stage_absolute_supply: Absolute supply by activity and stage coordinate.
    """

    parent: ContributionState
    matrix_database: str
    biosphere: sparse.csc_matrix
    absolute_biosphere: sparse.csc_matrix
    factors: np.ndarray
    activity_scores: np.ndarray
    activity_norms: np.ndarray
    activity_ids: np.ndarray
    flow_ids: np.ndarray
    supply: np.ndarray
    absolute_supply: np.ndarray
    group_index: dict[tuple[str, int], int]
    stage_group_index: dict[_StageCoordinate, int]
    stage_supply: np.ndarray
    stage_absolute_supply: np.ndarray


@dataclass(frozen=True, slots=True)
class ProviderBatchCharacterization:
    """LCIA values calculated per provider product for one solve batch."""

    scores: np.ndarray
    norms: np.ndarray


def _stage_coordinate(item: Any) -> _StageCoordinate:
    """Build the contribution coordinate for one foreground activation."""
    return (
        str(item.region),
        int(item.year),
        str(item.life_cycle_stage),
        str(item.stage_group),
        str(item.cutoff_cycle_role),
        str(item.material_cycle_source),
        str(item.physical_event_basis),
        int(item.physical_event_year),
        str(item.source_flow_names),
        str(item.lca_fate),
        str(item.fate_origin),
    )


def _add(
    values: dict[_Key, tuple[float, float]],
    key: _Key,
    score: float,
    norm: float,
) -> None:
    """Add one score and ``contribution_norm`` value to an existing total."""
    previous = values.get(key, (0.0, 0.0))
    values[key] = (
        fsum((previous[0], float(score))),
        fsum((previous[1], float(norm))),
    )


def create_contribution_state(
    bd: Any,
    regional_activations: Mapping[str, pd.DataFrame],
    regional_roots: Mapping[str, tuple[str, str]],
    methods: Sequence[ContributionMethod],
    years: Sequence[int],
) -> ContributionState:
    """Create the one reduction state consumed by all provider matrix blocks."""

    selected_years = tuple(sorted({int(year) for year in years}))
    frames = []
    for region, activations in regional_activations.items():
        frame = activations.copy()
        frame["region"] = region
        frame["regional_root_database"] = regional_roots[region][0]
        frame["regional_root_code"] = regional_roots[region][1]
        # Only actual receiving activations carry function/source descriptors.
        # Existing service activations retain their established empty labels.
        required = frame.loc[:, _CONTRIBUTION_ACTIVATION_COLUMNS]
        descriptors = frame.reindex(columns=_RECEIVING_DESCRIPTOR_COLUMNS, fill_value="").fillna("")
        frames.append(pd.concat((required, descriptors), axis=1))
    # Active quantity frames determine dtypes; inactive regions retain their reporting groups.
    combined = pd.concat(
        [frame for frame in frames if not frame.empty] or frames[:1], ignore_index=True
    )
    groups = tuple((region, year) for region in regional_activations for year in selected_years)
    return ContributionState(
        bd=bd,
        methods=tuple(methods),
        years=selected_years,
        regions=tuple(regional_activations),
        groups=groups,
        activations=combined,
    )


def create_provider_response_state(
    bd: Any,
    methods: Sequence[ContributionMethod],
) -> ContributionState:
    """Create the response-only factor owner for a complete provider solve."""

    return ContributionState(
        bd=bd,
        methods=tuple(methods),
        years=(),
        regions=(),
        groups=(),
        activations=pd.DataFrame(columns=_CONTRIBUTION_ACTIVATION_COLUMNS),
        retain_contributions=False,
    )


def begin_matrix_contributions(
    state: ContributionState,
    lca: Any,
    matrix_database: str,
) -> MatrixContributionState:
    """Prepare activity scores for one dated Brightway technosphere matrix.

    The biosphere matrix has axes ``[elementary flow, activity]`` and the
    characterization factor matrix has axes ``[LCIA method, elementary flow]``.
    Their product gives the LCIA score of one unit of each activity. Repeating
    the calculation with absolute biosphere exchanges and absolute factors gives
    ``contribution_norm`` for each unit.

    Args:
        state: Accumulated LCIA methods, activations, and result tables.
        lca: Calculated Brightway LCA object for the dated database.
        matrix_database: Selected frontier context represented by ``lca``; the
            demanded native activities can reside in other databases.

    Returns:
        Matrix data and empty regional accumulators for this provider database.

    Raises:
        ValueError: If a selected method has nonfinite processed characterization
            factors, including overflow during Brightway processing.
    """

    biosphere = cast(sparse.csc_matrix, lca.biosphere_matrix.tocsc())
    absolute_biosphere = abs(biosphere)
    flow_ids = np.fromiter(
        (int(lca.dicts.biosphere.reversed[row]) for row in range(biosphere.shape[0])),
        dtype=np.int64,
        count=biosphere.shape[0],
    )
    activity_ids = np.fromiter(
        (int(lca.dicts.activity.reversed[column]) for column in range(biosphere.shape[1])),
        dtype=np.int64,
        count=biosphere.shape[1],
    )
    flow_key = tuple(int(flow_id) for flow_id in flow_ids)
    factors = state.factor_cache.get(flow_key)
    if factors is None:
        rows = []
        for method in state.methods:
            lca.switch_method(method.method_tuple)
            vector = np.asarray(lca.characterization_matrix.sum(axis=0)).ravel()
            if not np.isfinite(vector).all():
                raise ValueError(
                    f"Brightway method {method.method_tuple!r} has nonfinite characterization "
                    f"factors for {matrix_database!r}. Select or register a method with "
                    "finite processed factors and run the calculation again."
                )
            rows.append(vector)
            accepted = state.characterization_factors.setdefault(method.method_tuple, {})
            accepted.update(
                {
                    flow_id: float(value)
                    for flow_id, value in zip(flow_key, vector, strict=True)
                    if value != 0.0
                }
            )
        factors = np.asarray(rows, dtype=np.float64)
        state.factor_cache[flow_key] = factors
    # Matrix multiplication preserves the Brightway activity and biosphere
    # ordering recorded above; the resulting axes are [method, activity].
    activity_scores = np.asarray(
        biosphere.T @ factors.T,
        dtype=np.float64,
    ).T
    activity_norms = np.asarray(
        absolute_biosphere.T @ np.abs(factors).T,
        dtype=np.float64,
    ).T
    selected = state.activations.loc[
        state.activations["matrix_database"].eq(matrix_database)
        & state.activations["year"].isin(state.years)
    ]
    groups = tuple(
        sorted(
            {
                (str(region), int(year))
                for region, year in selected[["region", "year"]].itertuples(
                    index=False,
                    name=None,
                )
            }
        )
    )
    regional_stage_groups = {
        _stage_coordinate(cast(Any, item)) for item in selected.itertuples(index=False)
    }
    stage_groups = tuple(
        sorted(
            regional_stage_groups
            | {(GLOBAL_REGION, *coordinate[1:]) for coordinate in regional_stage_groups}
        )
    )
    group_index = {group: index for index, group in enumerate(groups)}
    stage_group_index = {group: index for index, group in enumerate(stage_groups)}
    shape = (biosphere.shape[1], len(group_index))
    stage_shape = (biosphere.shape[1], len(stage_group_index))
    return MatrixContributionState(
        parent=state,
        matrix_database=matrix_database,
        biosphere=biosphere,
        absolute_biosphere=absolute_biosphere,
        factors=factors,
        activity_scores=activity_scores,
        activity_norms=activity_norms,
        activity_ids=activity_ids,
        flow_ids=flow_ids,
        supply=np.zeros(shape, dtype=np.float64),
        absolute_supply=np.zeros(shape, dtype=np.float64),
        group_index=group_index,
        stage_group_index=stage_group_index,
        stage_supply=np.zeros(stage_shape, dtype=np.float64),
        stage_absolute_supply=np.zeros(stage_shape, dtype=np.float64),
    )


def add_provider_batch(
    matrix: MatrixContributionState,
    provider_rows: pd.DataFrame,
    product_ids: Sequence[int],
    supply: np.ndarray,
    unit_inventory: np.ndarray,
) -> ProviderBatchCharacterization:
    """Characterize a batch of provider products calculated by Brightway.

    ``supply`` has axes ``[activity, requested product]`` and ``unit_inventory``
    has axes ``[elementary flow, requested product]``. ``contribution_norm`` is
    calculated from the absolute supply, biosphere flows, and characterization
    factors.
    Regional, stage, provider, and activity contributions are accumulated once
    from the matching foreground activations.

    Args:
        matrix: Prepared scores and contribution arrays for one dated database.
        provider_rows: Product identifiers and Brightway provider identities.
        product_ids: Product columns represented by the calculated arrays.
        supply: Technosphere supply array for the requested products.
        unit_inventory: Biosphere inventory array per requested product.

    Returns:
        LCIA scores and ``contribution_norm`` values for every requested product.

    Raises:
        ValueError: If characterization or scaling by foreground demand produces
            a nonfinite score or absolute contribution total.
    """

    state = matrix.parent
    columns = {int(product_id): index for index, product_id in enumerate(product_ids)}
    # Annual result tables cover 2019--2100. ``state.years`` limits only the
    # larger provider, activity, flow, and stage detail tables.
    selected = state.activations.loc[
        state.activations["matrix_database"].eq(matrix.matrix_database)
        & state.activations["provider_product_id"].isin(product_ids)
    ]
    demand = np.zeros(
        (len(product_ids), len(matrix.group_index)),
        dtype=np.float64,
    )
    absolute_demand = np.zeros_like(demand)
    stage_demand = np.zeros(
        (len(product_ids), len(matrix.stage_group_index)),
        dtype=np.float64,
    )
    stage_absolute_demand = np.zeros_like(stage_demand)
    with np.errstate(over="ignore", invalid="ignore"):
        unit_scores = matrix.factors @ unit_inventory
        unit_norms = matrix.activity_norms @ np.abs(supply)
    nonfinite = np.argwhere(~np.isfinite(unit_scores) | ~np.isfinite(unit_norms))
    if nonfinite.size:
        method_index, column = (int(value) for value in nonfinite[0])
        product_id = int(product_ids[column])
        provider = provider_rows.loc[provider_rows["provider_product_id"].eq(product_id)].iloc[0]
        raise ValueError(
            f"Provider matrix {matrix.matrix_database!r} produced nonfinite characterization "
            f"for provider {(provider.provider_database, provider.provider_code)!r}, "
            f"product {product_id}, method {state.methods[method_index].method_tuple!r}: "
            f"score={float(unit_scores[method_index, column])!r}, "
            f"contribution_norm={float(unit_norms[method_index, column])!r}. "
            "Check the source exchanges and characterization factors before a new calculation."
        )
    characterized = ProviderBatchCharacterization(
        scores=np.asarray(unit_scores, dtype=np.float64),
        norms=np.asarray(unit_norms, dtype=np.float64),
    )
    if not state.retain_contributions:
        return characterized
    provider_by_product = {
        int(item.provider_product_id): (
            str(item.provider_database),
            str(item.provider_code),
        )
        for row in provider_rows.itertuples(index=False)
        for item in (cast(Any, row),)
        if int(item.provider_product_id) in columns
    }

    for activation in selected.itertuples(index=False):
        item = cast(Any, activation)
        product_id = int(item.provider_product_id)
        column = columns[product_id]
        amount = float(item.amount)
        region = str(item.region)
        year = int(item.year)
        stage_coordinate = _stage_coordinate(item)
        provider_key = provider_by_product[product_id]
        receiving: _ReceivingDescriptors = (
            str(item.receiving_function),
            str(item.source_records_json),
            str(item.endpoint_unit),
        )
        for method_index in range(len(state.methods)):
            score = amount * float(unit_scores[method_index, column])
            norm = abs(amount) * float(unit_norms[method_index, column])
            if not np.isfinite((score, norm)).all():
                raise ValueError(
                    f"Provider matrix {matrix.matrix_database!r} produced nonfinite impacts "
                    f"for provider {provider_key!r}, product {product_id}, "
                    f"method {state.methods[method_index].method_tuple!r}, "
                    f"region {region!r}, year {year}, foreground amount {amount!r}: "
                    f"score={score!r}, contribution_norm={norm!r}. "
                    "Check the source exchanges, characterization factors and foreground "
                    "demand before a new calculation."
                )
            _add(
                state.annual_totals,
                (region, year, method_index),
                score,
                norm,
            )
            annual_stage_key = (
                region,
                year,
                method_index,
                *stage_coordinate[2:],
            )
            _add(
                state.annual_stage_values,
                annual_stage_key,
                score,
                norm,
            )
            if year in state.years:
                _add(
                    state.stage_values,
                    (
                        region,
                        year,
                        method_index,
                        str(item.life_cycle_stage),
                        str(item.stage_group),
                        str(item.cutoff_cycle_role),
                        str(item.material_cycle_source),
                        str(item.physical_event_basis),
                        int(item.physical_event_year),
                        str(item.source_flow_names),
                        str(item.lca_fate),
                        str(item.fate_origin),
                    ),
                    score,
                    norm,
                )
                _add(
                    state.provider_values,
                    (
                        region,
                        year,
                        method_index,
                        provider_key[0],
                        provider_key[1],
                        receiving,
                    ),
                    score,
                    norm,
                )
                _add(
                    state.stage_provider_values,
                    (stage_coordinate, method_index, provider_key[0], provider_key[1], receiving),
                    score,
                    norm,
                )
                global_stage_coordinate = (GLOBAL_REGION, *stage_coordinate[1:])
                _add(
                    state.stage_provider_values,
                    (
                        global_stage_coordinate,
                        method_index,
                        provider_key[0],
                        provider_key[1],
                        receiving,
                    ),
                    score,
                    norm,
                )
        if year not in state.years:
            continue
        for coordinate in (stage_coordinate, (GLOBAL_REGION, *stage_coordinate[1:])):
            _add(
                state.stage_provider_amounts,
                (coordinate, provider_key[0], provider_key[1], receiving),
                amount,
                abs(amount),
            )
            stage_group = matrix.stage_group_index[coordinate]
            stage_demand[column, stage_group] += amount
            stage_absolute_demand[column, stage_group] += abs(amount)
        group = matrix.group_index[(region, year)]
        demand[column, group] += amount
        absolute_demand[column, group] += abs(amount)
        structural_key = (
            region,
            year,
            str(item.regional_root_database),
            str(item.regional_root_code),
            str(item.stage_database),
            str(item.stage_code),
            str(item.life_cycle_stage),
            str(item.stage_group),
            str(item.cutoff_cycle_role),
            str(item.material_cycle_source),
            str(item.physical_event_basis),
            int(item.physical_event_year),
            str(item.source_flow_names),
            str(item.lca_fate),
            str(item.fate_origin),
            str(item.material),
            str(item.constituent),
            str(item.product),
            str(item.virgin_feedstock_route),
            str(item.ultimate_carbon_origin),
            str(item.good),
            str(item.provider_database),
            str(item.provider_code),
        )
        state.structural_values[structural_key] = fsum(
            (state.structural_values.get(structural_key, 0.0), amount)
        )

    matrix.supply += supply @ demand
    matrix.absolute_supply += np.abs(supply) @ absolute_demand
    matrix.stage_supply += supply @ stage_demand
    matrix.stage_absolute_supply += np.abs(supply) @ stage_absolute_demand
    return characterized


def _top_context(
    score: np.ndarray,
    norm: np.ndarray,
    identities: np.ndarray,
    provider_database: str,
) -> list[tuple[str, int, float, float]]:
    """Select the largest native activity contributions in one dated matrix context."""
    active = np.flatnonzero((score != 0.0) | (norm != 0.0))
    selected = _top_indices(active, norm, identities)
    return [
        (
            provider_database,
            int(identities[index]),
            float(score[index]),
            float(norm[index]),
        )
        for index in selected
    ]


def _top_indices(
    active: np.ndarray,
    norm: np.ndarray,
    identities: np.ndarray,
    limit: int = DETAIL_LIMIT,
) -> list[int]:
    """Select the largest ``contribution_norm`` values with stable tie breaking."""
    if active.size > limit:
        active_norms = norm[active]
        threshold = float(
            np.partition(
                active_norms,
                active_norms.size - limit,
            )[active_norms.size - limit]
        )
        above = active[active_norms > threshold]
        tied = active[active_norms == threshold]
        remaining = limit - above.size
        selected_ties = tied[np.argsort(identities[tied], kind="stable")[:remaining]]
        active = np.concatenate((above, selected_ties))
    return sorted(
        active.tolist(),
        key=lambda index: (-float(norm[index]), int(identities[index])),
    )


def _merge_flow_values(
    state: ContributionState,
    key: Hashable,
    identities: np.ndarray,
    scores: np.ndarray,
    norms: np.ndarray,
    *,
    target: dict[Any, _FlowValues] | None = None,
) -> None:
    """Merge characterized elementary-flow values into one contribution group."""
    values = cast(dict[Any, _FlowValues], state.flow_values) if target is None else target
    existing = values.get(key)
    if existing is None:
        values[key] = _FlowValues(
            identities=identities.copy(),
            scores=scores.copy(),
            norms=norms.copy(),
        )
        return
    if np.array_equal(existing.identities, identities):
        existing.scores += scores
        existing.norms += norms
        return
    combined = np.union1d(existing.identities, identities)
    combined_scores = np.zeros(combined.size, dtype=np.float64)
    combined_norms = np.zeros(combined.size, dtype=np.float64)
    old = np.searchsorted(combined, existing.identities)
    new = np.searchsorted(combined, identities)
    combined_scores[old] = existing.scores
    combined_norms[old] = existing.norms
    combined_scores[new] += scores
    combined_norms[new] += norms
    values[key] = _FlowValues(
        identities=combined,
        scores=combined_scores,
        norms=combined_norms,
    )


def _flow_metadata(
    state: ContributionState,
    flow_id: int,
) -> tuple[str, str, str, str, str, str]:
    """Read and cache one Brightway biosphere flow's labels and classification."""
    cached = state.flow_metadata.get(flow_id)
    if cached is not None:
        return cached
    flow = state.bd.get_node(id=flow_id)
    categories = tuple(str(value) for value in flow.get("categories", ()))
    result = (
        str(flow.key[0]),
        str(flow.key[1]),
        str(flow.get("name", "")),
        json.dumps(categories, ensure_ascii=True, separators=(",", ":")),
        categories[0] if categories else "unclassified",
        str(flow["unit"]),
    )
    state.flow_metadata[flow_id] = result
    return result


def _retain_native_values(
    retained: dict[tuple[str, int], tuple[float, float]],
    score: np.ndarray,
    norm: np.ndarray,
    matrix: MatrixContributionState,
) -> None:
    """Merge one matrix's largest activities into a bounded contribution group.

    Matrix database and activity ID jointly identify a row. Different matrices
    therefore have disjoint keys: retaining each matrix's top entries before
    merging cannot omit a member of the group's global top ``DETAIL_LIMIT``.
    """
    for database, activity_id, value, magnitude in _top_context(
        score, norm, matrix.activity_ids, matrix.matrix_database
    ):
        _add(retained, (database, activity_id), value, magnitude)
    if len(retained) > DETAIL_LIMIT:
        largest = sorted(
            retained.items(),
            key=lambda item: (-item[1][1], item[0][0], item[0][1]),
        )[:DETAIL_LIMIT]
        retained.clear()
        retained.update(largest)


def finish_matrix_contributions(matrix: MatrixContributionState) -> None:
    """Add activity, elementary flow, and compartment detail from one matrix.

    Args:
        matrix: Dated provider matrices and accumulated supply arrays. The function adds
            its results to ``matrix.parent``.

    Notes:
        Detail is calculated by region and year and by the full life cycle stage
        coordinate. Global activity values sum all regional values before their
        independent selection by ``contribution_norm``.
    """

    state = matrix.parent
    if not state.retain_contributions:
        return
    year_columns: dict[int, list[int]] = {}
    for (_, year), column in matrix.group_index.items():
        year_columns.setdefault(year, []).append(column)
    for year, columns in year_columns.items():
        for method_index in range(len(state.methods)):
            # Keep only two activity vectors for this year/method. Summing the
            # complete regional values preserves contributors omitted locally;
            # norms remain sums of absolute contributions across regions.
            score = np.zeros(matrix.activity_ids.size, dtype=np.float64)
            norm = np.zeros_like(score)
            for column in columns:
                score += matrix.activity_scores[method_index] * matrix.supply[:, column]
                norm += matrix.activity_norms[method_index] * matrix.absolute_supply[:, column]
            _retain_native_values(
                state.native_values.setdefault((GLOBAL_REGION, year, method_index), {}),
                score,
                norm,
                matrix,
            )
            del score, norm
    # Apply each biosphere matrix to all accumulated regional columns at once. The
    # matrix row order remains aligned with ``flow_ids`` and the factor arrays.
    flows = np.asarray(
        matrix.biosphere @ matrix.supply,
        dtype=np.float64,
    )
    absolute_flows = np.asarray(
        matrix.absolute_biosphere @ matrix.absolute_supply,
        dtype=np.float64,
    )
    stage_flows = np.asarray(
        matrix.biosphere @ matrix.stage_supply,
        dtype=np.float64,
    )
    stage_absolute_flows = np.asarray(
        matrix.absolute_biosphere @ matrix.stage_absolute_supply,
        dtype=np.float64,
    )
    for (region, year), group_index in matrix.group_index.items():
        for method_index in range(len(state.methods)):
            native_score = matrix.activity_scores[method_index] * matrix.supply[:, group_index]
            native_norm = (
                matrix.activity_norms[method_index] * matrix.absolute_supply[:, group_index]
            )
            native_key = (region, year, method_index)
            _retain_native_values(
                state.native_values.setdefault(native_key, {}),
                native_score,
                native_norm,
                matrix,
            )
            flow_score = matrix.factors[method_index] * flows[:, group_index]
            flow_norm = np.abs(matrix.factors[method_index]) * absolute_flows[:, group_index]
            active = np.flatnonzero((flow_score != 0.0) | (flow_norm != 0.0))
            _merge_flow_values(
                state,
                (region, year, method_index),
                matrix.flow_ids[active],
                flow_score[active],
                flow_norm[active],
            )
            for row in active:
                flow_id = int(matrix.flow_ids[row])
                classification = _flow_metadata(state, flow_id)[4]
                _add(
                    state.classification_values,
                    (region, year, method_index, classification),
                    float(flow_score[row]),
                    float(flow_norm[row]),
                )

    for stage_coordinate, group_index in matrix.stage_group_index.items():
        for method_index in range(len(state.methods)):
            native_score = (
                matrix.activity_scores[method_index] * matrix.stage_supply[:, group_index]
            )
            native_norm = (
                matrix.activity_norms[method_index] * matrix.stage_absolute_supply[:, group_index]
            )
            key = (stage_coordinate, method_index)
            _retain_native_values(
                state.stage_native_values.setdefault(key, {}),
                native_score,
                native_norm,
                matrix,
            )
            flow_score = matrix.factors[method_index] * stage_flows[:, group_index]
            flow_norm = np.abs(matrix.factors[method_index]) * stage_absolute_flows[:, group_index]
            active = np.flatnonzero((flow_score != 0.0) | (flow_norm != 0.0))
            _merge_flow_values(
                state,
                key,
                matrix.flow_ids[active],
                flow_score[active],
                flow_norm[active],
                target=state.stage_flow_values,
            )
            for row in active:
                flow_id = int(matrix.flow_ids[row])
                classification = _flow_metadata(state, flow_id)[4]
                _add(
                    state.stage_classification_values,
                    (stage_coordinate, method_index, classification),
                    float(flow_score[row]),
                    float(flow_norm[row]),
                )


def _activity_metadata(
    state: ContributionState,
    activity_id: int,
) -> tuple[str, str, str, str]:
    """Read and cache one Brightway activity's database, code, name, and location."""
    cached = state.activity_metadata.get(activity_id)
    if cached is not None:
        return cached
    activity = state.bd.get_node(id=activity_id)
    result = (
        str(activity.key[0]),
        str(activity.key[1]),
        str(activity.get("name", "")),
        str(activity.get("location", "")),
    )
    state.activity_metadata[activity_id] = result
    return result


def _top_mapping(
    values: _FlowValues | None,
    totals: tuple[float, float],
    limit: int = DETAIL_LIMIT,
) -> tuple[
    list[tuple[int, float, float]],
    tuple[float, float],
]:
    """Retain the largest detailed flow rows and combine the remaining total."""
    if values is None:
        return [], totals
    active = np.flatnonzero((values.scores != 0.0) | (values.norms != 0.0))
    retained = [
        (
            int(values.identities[index]),
            float(values.scores[index]),
            float(values.norms[index]),
        )
        for index in _top_indices(
            active,
            values.norms,
            values.identities,
            limit,
        )
    ]
    residual = (
        fsum((totals[0], -fsum(item[1] for item in retained))),
        _nonnegative_residual(
            totals[1],
            fsum(item[2] for item in retained),
        ),
    )
    return retained, residual


def _nonnegative_residual(total: float, retained: float) -> float:
    """Subtract the retained contribution total and clamp rounding below zero."""

    return max(fsum((total, -retained)), 0.0)
