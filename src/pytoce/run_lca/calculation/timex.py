"""Canonical time-explicit inventory calculation."""

import gc
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from math import fsum
from typing import Any, cast

import bw2calc as bc  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from bw_timex import TimexLCA  # type: ignore[import-untyped]

from .contribution_accumulation import (
    ContributionState,
    add_provider_batch,
    begin_matrix_contributions,
    finish_matrix_contributions,
)

STARTING_DATETIME = datetime(2019, 1, 1)
TEMPORAL_GROUPING = "year"
INTERPOLATION_TYPE = "linear"
TRAVERSAL_CUTOFF = 1e-15
MAX_CALCULATIONS = 2000
GRAPH_TRAVERSAL = "bfs"
PROVIDER_SOLVE_BATCH_SIZE = 32
PRINCIPAL_INVENTORY_METHOD = "timex_multi_root_timeline_brightway_multi_rhs_v48"
_PUBLIC_TIMELINE_COLUMNS = [
    "date_producer",
    "producer_name",
    "date_consumer",
    "consumer_name",
    "amount",
    "temporal_market_shares",
]
_ACTIVATION_GROUP_COLUMNS = [
    "year",
    "region",
    "stage_database",
    "stage_code",
    "life_cycle_stage",
    "stage_group",
    "cutoff_cycle_role",
    "physical_event_basis",
    "physical_event_year",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "good",
    "custody_status",
    "source_parent_amount_kg",
    "source_allocation_share",
    "material_cycle_source",
    "ultimate_carbon_origin",
    "origin_fraction",
    "branch_lineage_id",
    "source_flow_names",
    "lca_fate",
    "fate_origin",
    "route_id",
    "route_component_id",
    "provider_service_id",
    "provider_role",
    "provider_material",
    "infiltration_class",
    "material_component_share",
    "infiltration_share",
    "provider_class",
    "analogue_status",
    "throughput_multiplier",
    "resolution_status",
    "material_functional_unit",
    "styrene_component_share",
    "butadiene_analogue_component_share",
    "sbr_share_in_material",
    "sbr_proxy_component_role",
    "whole_tyre_constituents_excluded",
    "no_treatment_sensitivity_executed",
    "declared_provider_source_role",
    "matrix_database",
    "provider_database",
    "provider_anchor_year",
    "provider_source_database",
    "provider_source_role",
    "general_background_database",
    "general_background_anchor_year",
    "general_background_role",
    "selected_background_member",
    "provider_code",
    "provider_activity_id",
    "provider_product_id",
]
_ACTIVATION_VALUE_COLUMNS = [
    "amount",
    "temporal_provider_share",
    "mfa_carbon_amount_kg",
]
_PROVIDER_IDENTITY_COLUMNS = [
    "matrix_database",
    "provider_database",
    "provider_code",
    "provider_activity_id",
    "provider_product_id",
]
_PROVIDER_RESPONSE_IDENTITY_COLUMNS = [
    *_PROVIDER_IDENTITY_COLUMNS,
    "provider_anchor_year",
]
PROVIDER_UNIT_RESPONSE_COLUMNS = [
    *_PROVIDER_RESPONSE_IDENTITY_COLUMNS,
    "provider_product_database",
    "provider_product_code",
    "provider_unit",
    "method_id",
    "method_version",
    "impact_category_id",
    "method_tuple_json",
    "impact_unit",
    "unit_score",
    "unit_contribution_norm",
]
_PRINCIPAL_RUNTIME_ACTIVATION_COLUMNS = [
    *_PROVIDER_IDENTITY_COLUMNS,
    "year",
    "amount",
]


@dataclass(frozen=True, slots=True)
class PrincipalTimeline:
    """One regional timeline, dated supplier demands and modeled direct flows.

    ``direct_inventory`` contains year, native Brightway flow ID and amount in
    that flow's unit. The receiving owner adds these amounts after its
    complete mixture calculation and temporal allocation. They are combined
    with supplier inventories before characterization.
    """

    timeline: pd.DataFrame
    activations: pd.DataFrame
    base_static_score: float
    direct_inventory: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ProviderUnitInventory:
    """One provider's elementary flow inventory."""

    flow_ids: np.ndarray
    amounts: np.ndarray


@dataclass(frozen=True, slots=True)
class _ResolvedProviderTarget:
    """Dated provider activity and its matching general background database."""

    activity_id: int
    product_id: int
    provider_source_database: str
    provider_source_role: str
    general_background_database: str
    general_background_anchor_year: int
    general_background_role: str
    selected_background_member: str


@dataclass(frozen=True, slots=True)
class ProviderInventoryBank:
    """Unit inventories solved once for the providers used by regional timelines."""

    inventories: dict[tuple[str, str, str], ProviderUnitInventory]
    matrix_count: int
    provider_count: int
    solve_batch_count: int
    unit_responses: pd.DataFrame


@dataclass(frozen=True, slots=True)
class CharacterizationFactorBank:
    """Registered characterization factors keyed by biosphere flow ID."""

    factors: dict[tuple[str, ...], dict[int, float]]


@dataclass(frozen=True, slots=True)
class PrincipalInventory:
    """One regional timeline and its time-explicit principal inventory."""

    timeline: pd.DataFrame
    inventory: pd.DataFrame
    base_static_score: float
    provider_matrix_count: int
    provider_count: int
    provider_solve_batch_count: int


def _provider_activations(
    timeline: pd.DataFrame,
    bd: Any,
    foreground_activations: pd.DataFrame,
    provider_databases: dict[int, str],
    timeline_databases: dict[int, str],
    *,
    activation_keys: set[tuple[str, int, str]],
    producer_codes: dict[int, str],
    targets: dict[tuple[str, str], _ResolvedProviderTarget],
) -> pd.DataFrame:
    """Resolve annual foreground activations to dated provider products.

    The Timex timeline supplies provider shares by temporal-market database,
    activity code, and year. Each foreground activation is split among its
    dated provider shares by world region, activity year, and provider
    service code. Each output amount retains the activation unit and equals the
    foreground amount multiplied by its dated provider share.

    Args:
        timeline: Timex temporal-market rows for the selected background.
        bd: Imported ``bw2data`` module used to resolve product identifiers.
        foreground_activations: Annual regional foreground service amounts.
        provider_databases: Prepared provider database name by anchor year.
        timeline_databases: Timex temporal-market database name by anchor year.
        activation_keys: Precomputed activation join keys.
        producer_codes: Temporal-market producer codes by Brightway activity ID.
        targets: Cache of resolved Brightway provider products.

    Returns:
        Dated provider activations with one row per matched foreground amount and
        provider share.
    """

    provider_by_timeline_database = {
        database: provider_databases[year] for year, database in timeline_databases.items()
    }
    anchor_year_by_provider_database = {
        database: year for year, database in provider_databases.items()
    }
    regional_roots: dict[int, tuple[str, str, str]] = {}
    mapping_rows: list[dict[str, object]] = []
    frontier = timeline.loc[timeline["temporal_market_shares"].notna()]
    for raw_item in frontier.itertuples(index=False):
        item = cast(Any, raw_item)
        producer_id = int(item.producer)
        code = producer_codes.get(producer_id)
        if code is None:
            code = str(bd.get_node(id=producer_id)["code"])
            producer_codes[producer_id] = code
        root_id = int(item.consumer)
        root = regional_roots.get(root_id)
        if root is None:
            activity = bd.get_node(id=root_id)
            root = (
                str(activity.key[0]),
                str(activity.key[1]),
                str(activity["region"]),
            )
            regional_roots[root_id] = root
        year = pd.Timestamp(item.date_producer).year
        if (root[2], year, code) not in activation_keys:
            raise KeyError(
                "Timex mapped a provider service without a matching foreground "
                f"activation: {(root[2], year, code)!r}"
            )
        for database, share in cast(dict[str, float], item.temporal_market_shares).items():
            key = (provider_by_timeline_database[str(database)], code)
            target = targets.get(key)
            if target is None:
                provider = bd.get_node(database=key[0], code=key[1])
                target = _ResolvedProviderTarget(
                    activity_id=int(provider.id),
                    product_id=int(provider.rp_exchange().input.id),
                    provider_source_database=str(provider["provider_source_database"]),
                    provider_source_role=str(provider["provider_source_role"]),
                    general_background_database=str(provider["general_background_database"]),
                    general_background_anchor_year=int(
                        cast(int | str, provider["general_background_anchor_year"])
                    ),
                    general_background_role=str(provider["general_background_role"]),
                    selected_background_member=str(provider["selected_background_member"]),
                )
                targets[key] = target
            provider_anchor_year = anchor_year_by_provider_database[key[0]]
            mapping_rows.append(
                {
                    "region": root[2],
                    "year": year,
                    "provider_service_code": code,
                    "stage_database": root[0],
                    "stage_code": root[1],
                    "provider_database": key[0],
                    "matrix_database": key[0],
                    "provider_anchor_year": provider_anchor_year,
                    "provider_code": key[1],
                    "provider_activity_id": target.activity_id,
                    "provider_product_id": target.product_id,
                    "provider_source_database": target.provider_source_database,
                    "provider_source_role": target.provider_source_role,
                    "general_background_database": target.general_background_database,
                    "general_background_anchor_year": (target.general_background_anchor_year),
                    "general_background_role": target.general_background_role,
                    "selected_background_member": target.selected_background_member,
                    "temporal_provider_share": float(share),
                }
            )
    mapping = pd.DataFrame(
        mapping_rows,
        columns=[
            "region",
            "year",
            "provider_service_code",
            "stage_database",
            "stage_code",
            "matrix_database",
            "provider_database",
            "provider_anchor_year",
            "provider_code",
            "provider_activity_id",
            "provider_product_id",
            "provider_source_database",
            "provider_source_role",
            "general_background_database",
            "general_background_anchor_year",
            "general_background_role",
            "selected_background_member",
            "temporal_provider_share",
        ],
    )
    expected_mapping_keys = {
        (str(region), int(cast(Any, year)), str(service_code))
        for region, year, service_code in foreground_activations[
            ["region", "year", "provider_service_code"]
        ]
        .drop_duplicates(ignore_index=True)
        .itertuples(index=False, name=None)
    }
    observed_mapping_keys = {
        (str(region), int(cast(Any, year)), str(service_code))
        for region, year, service_code in mapping[["region", "year", "provider_service_code"]]
        .drop_duplicates(ignore_index=True)
        .itertuples(index=False, name=None)
    }
    missing_mapping_keys = sorted(expected_mapping_keys.difference(observed_mapping_keys))
    if missing_mapping_keys:
        raise KeyError(
            f"Foreground activations have no Timex provider mapping: {missing_mapping_keys[:10]!r}"
        )
    frame = foreground_activations.merge(
        mapping,
        on=["region", "year", "provider_service_code"],
        how="inner",
    )
    frame["life_cycle_stage"] = frame["demand_type"]
    frame["source_flow_names"] = frame["basis_flow_names"]
    frame["amount"] = frame["amount"] * frame["temporal_provider_share"]
    frame["mfa_carbon_amount_kg"] = frame["mfa_carbon_amount_kg"] * frame["temporal_provider_share"]
    lineage_columns = [
        "provider_demand_type",
        "mapping_basis",
        "lca_boundary",
        "burden_attribution_basis",
    ]
    result = frame[_ACTIVATION_GROUP_COLUMNS + _ACTIVATION_VALUE_COLUMNS + lineage_columns]
    return result.sort_values(
        _ACTIVATION_GROUP_COLUMNS,
        kind="stable",
        ignore_index=True,
    )


def build_principal_timelines(
    regional_roots: dict[str, object],
    method: tuple[str, ...],
    database_dates: dict[str, str | datetime],
    bd: Any,
    foreground_activations: pd.DataFrame,
    provider_databases: dict[int, str],
    timeline_databases: dict[int, str],
    *,
    use_global_lci_cache: bool = True,
) -> dict[str, PrincipalTimeline]:
    """Build all regional mapping timelines in one supported Timex traversal.

    The twelve roots remain independent consumers. Timex computes their direct
    provider-service edges together so identical interpolation dates are mapped
    once; each returned timeline and activation table is then restricted to its
    owning regional root.
    """

    root_count = len(regional_roots)
    calculation = TimexLCA(
        {regional_root: 1.0 for regional_root in regional_roots.values()},
        method,
        database_dates,
        use_global_lci_cache=use_global_lci_cache,
    )
    calculation.build_timeline(
        starting_datetime=STARTING_DATETIME,
        temporal_grouping=TEMPORAL_GROUPING,
        interpolation_type=INTERPOLATION_TYPE,
        cutoff=TRAVERSAL_CUTOFF / root_count,
        max_calc=MAX_CALCULATIONS * root_count,
        graph_traversal=GRAPH_TRAVERSAL,
        traverse_background=False,
    )
    # bw_timex returns display columns but retains the activity identifiers used
    # to separate the regional roots on the documented ``timeline`` attribute.
    timeline = cast(pd.DataFrame, calculation.timeline)
    base_static_score = float(calculation.base_lca.score)
    if base_static_score != 0.0:
        raise ValueError(
            "Timex returned a non-zero score before provider exchanges were added: "
            f"{base_static_score}."
        )
    activation_keys = {
        (str(region), int(cast(Any, year)), str(service_code))
        for region, year, service_code in foreground_activations[
            ["region", "year", "provider_service_code"]
        ]
        .drop_duplicates(ignore_index=True)
        .itertuples(index=False, name=None)
    }
    producer_codes: dict[int, str] = {}
    targets: dict[tuple[str, str], _ResolvedProviderTarget] = {}
    result: dict[str, PrincipalTimeline] = {}
    for region, regional_root in regional_roots.items():
        root_id = int(cast(Any, regional_root).id)
        regional_timeline = timeline.loc[
            timeline["consumer"].eq(root_id)
            | (timeline["consumer"].eq(-1) & timeline["producer"].eq(root_id))
        ].copy()
        regional_activations = foreground_activations.loc[
            foreground_activations["region"].eq(region)
        ]
        result[region] = PrincipalTimeline(
            timeline=regional_timeline[_PUBLIC_TIMELINE_COLUMNS].copy(),
            activations=_provider_activations(
                regional_timeline,
                bd,
                regional_activations,
                provider_databases,
                timeline_databases,
                activation_keys=activation_keys,
                producer_codes=producer_codes,
                targets=targets,
            ),
            base_static_score=0.0,
            direct_inventory=pd.DataFrame(columns=("year", "flow_id", "amount")),
        )
    del calculation
    gc.collect()
    return result


def compact_principal_timelines(
    timelines: dict[str, PrincipalTimeline],
) -> dict[str, PrincipalTimeline]:
    """Retain only fields consumed by bank and principal-inventory phases."""

    return {
        region: PrincipalTimeline(
            timeline=timeline.timeline,
            activations=timeline.activations[_PRINCIPAL_RUNTIME_ACTIVATION_COLUMNS].copy(),
            base_static_score=timeline.base_static_score,
            direct_inventory=timeline.direct_inventory[["year", "flow_id", "amount"]],
        )
        for region, timeline in timelines.items()
    }


def _unit_inventories(
    bd: Any,
    providers: pd.DataFrame,
    contributions: ContributionState,
    *,
    matrix_database: str,
) -> tuple[dict[tuple[str, str, str], ProviderUnitInventory], int, pd.DataFrame]:
    """Calculate native product inventories in one selected dated matrix context.

    One Brightway technosphere matrix includes the selected frontier and the
    public dependencies of its requested native endpoints. Products are solved in batches by
    placing one unit of each product in separate right-hand-side columns. The
    biosphere matrix multiplied by the supply array gives the unit inventories.
    When full contribution reporting is requested, nonzero elementary flows are
    retained per provider; otherwise only the LCIA values required for the
    optimization response are returned.

    Args:
        bd: Imported ``bw2data`` module with the selected project active.
        providers: Native endpoints requested in one selected matrix context.
        contributions: LCIA methods and contribution accumulators for this LCA run.
        matrix_database: Selected frontier; endpoint activities may reside elsewhere.

    Returns:
        Unit inventories keyed by matrix, endpoint database and activity code,
        number of matrix batches, and provider unit LCIA responses. Only the representation needed
        by the caller is populated.

    Raises:
        ValueError: A requested provider produces a nonfinite supply or inventory
            value in the processed Brightway matrices.
    """

    provider_rows = providers.sort_values(
        ["provider_code", "provider_product_id"],
        kind="stable",
    )
    provider_by_product = {}
    response_metadata_by_product: dict[int, dict[str, object]] = {}
    for raw_row in provider_rows.itertuples(index=False):
        row = cast(Any, raw_row)
        product_id = int(row.provider_product_id)
        provider_by_product[product_id] = (
            matrix_database,
            str(row.provider_database),
            str(row.provider_code),
        )
        if not contributions.retain_contributions:
            response_metadata_by_product[product_id] = {
                column: getattr(row, column) for column in _PROVIDER_RESPONSE_IDENTITY_COLUMNS
            }
            product = bd.get_node(id=product_id)
            response_metadata_by_product[product_id].update(
                provider_unit=str(product["unit"]),
                provider_product_database=str(product["database"]),
                provider_product_code=str(product["code"]),
            )
    product_ids = provider_rows["provider_product_id"].astype("int64").tolist()
    # Include each actual endpoint's dependency closure in the matrix context defined by the
    # selected frontier.
    database_names = sorted(
        {
            dependency
            for database in {matrix_database, *provider_rows["provider_database"]}
            for dependency in bd.Database(database).find_graph_dependents()
        }
    )
    data_objs = [bd.Database(database).datapackage() for database in database_names]
    lca = bc.LCA({product_ids[0]: 1.0}, data_objs=data_objs)
    lca.load_lci_data()
    matrix_contributions = begin_matrix_contributions(
        contributions,
        lca,
        matrix_database,
    )
    flow_ids = np.fromiter(
        (int(lca.dicts.biosphere.reversed[row]) for row in range(lca.biosphere_matrix.shape[0])),
        dtype=np.int64,
        count=lca.biosphere_matrix.shape[0],
    )

    inventories: dict[tuple[str, str, str], ProviderUnitInventory] = {}
    response_rows: list[dict[str, object]] = []
    batch_count = 0
    for offset in range(0, len(product_ids), PROVIDER_SOLVE_BATCH_SIZE):
        batch = product_ids[offset : offset + PROVIDER_SOLVE_BATCH_SIZE]
        # Each column asks for one unit of a provider product. Reusing the loaded
        # technosphere matrix avoids one Brightway calculation per service.
        rhs = np.zeros(
            (lca.technosphere_matrix.shape[0], len(batch)),
            dtype=np.float64,
            order="F",
        )
        rhs[
            np.fromiter(
                (int(lca.dicts.product[item]) for item in batch),
                dtype=np.int64,
                count=len(batch),
            ),
            np.arange(len(batch), dtype=np.int64),
        ] = 1.0
        supply = np.asarray(
            lca.solve_linear_system(rhs),
            dtype=np.float64,
        ).reshape(lca.technosphere_matrix.shape[0], -1)
        if not np.isfinite(supply).all():
            raise ValueError(
                f"Provider matrix {matrix_database!r} produced nonfinite supply values "
                f"for products {[provider_by_product[item] for item in batch]!r}. "
                "LCI calculation requires finite values; check the source exchanges "
                "and prepare a new workspace with corrected inputs."
            )
        unit_inventory = np.asarray(
            lca.biosphere_matrix @ supply,
            dtype=np.float64,
        )
        if not np.isfinite(unit_inventory).all():
            raise ValueError(
                f"Provider matrix {matrix_database!r} produced nonfinite inventory "
                f"values for products {[provider_by_product[item] for item in batch]!r}. "
                "LCI calculation requires finite values; check the source exchanges "
                "and prepare a new workspace with corrected inputs."
            )
        characterized = add_provider_batch(
            matrix_contributions,
            provider_rows,
            batch,
            supply,
            unit_inventory,
        )
        for column, product_id in enumerate(batch):
            if contributions.retain_contributions:
                active = np.flatnonzero(unit_inventory[:, column])
                inventories[provider_by_product[product_id]] = ProviderUnitInventory(
                    flow_ids=flow_ids[active].copy(),
                    amounts=unit_inventory[active, column].copy(),
                )
            if not contributions.retain_contributions:
                metadata = response_metadata_by_product[product_id]
                for method_index, method in enumerate(contributions.methods):
                    score = float(characterized.scores[method_index, column])
                    norm = float(characterized.norms[method_index, column])
                    response_rows.append(
                        {
                            **metadata,
                            "method_id": method.method_id,
                            "method_version": method.method_version,
                            "impact_category_id": method.impact_category_id,
                            "method_tuple_json": json.dumps(
                                list(method.method_tuple),
                                ensure_ascii=True,
                                separators=(",", ":"),
                            ),
                            "impact_unit": method.unit,
                            "unit_score": score,
                            "unit_contribution_norm": norm,
                        }
                    )
        batch_count += 1
    finish_matrix_contributions(matrix_contributions)
    responses = pd.DataFrame(response_rows, columns=PROVIDER_UNIT_RESPONSE_COLUMNS)
    return inventories, batch_count, responses


def build_provider_inventory_bank(
    bd: Any,
    provider_requests: pd.DataFrame,
    contributions: ContributionState,
) -> ProviderInventoryBank:
    """Solve actual endpoint requests once per selected dated matrix context.

    Requests identify the matrix database, native activity database/code/ID and
    reference-product ID. Response-only requests also include the provider anchor
    year. Package service labels are joined by their registry owner after solving.
    """

    provider_columns = (
        _PROVIDER_IDENTITY_COLUMNS
        if contributions.retain_contributions
        else _PROVIDER_RESPONSE_IDENTITY_COLUMNS
    )
    providers = provider_requests[provider_columns].drop_duplicates(_PROVIDER_IDENTITY_COLUMNS)
    inventories: dict[tuple[str, str, str], ProviderUnitInventory] = {}
    response_frames: list[pd.DataFrame] = []
    batch_count = 0
    for _database, group in providers.groupby(
        "matrix_database",
        sort=True,
        observed=True,
    ):
        try:
            solved, batches, responses = _unit_inventories(
                bd,
                group,
                contributions,
                matrix_database=str(_database),
            )
        finally:
            if bc.PYPARDISO:
                # bw2calc 2.5 delegates to pypardiso's public global solver.
                # Each provider frontier is an independent matrix, so its
                # retained factorization cannot be reused by the next one.
                from pypardiso import ps  # type: ignore[import-untyped]

                ps.free_memory(everything=True)
        inventories.update(solved)
        response_frames.append(responses)
        batch_count += batches
        del solved
        gc.collect()
    contributions.activations = contributions.activations.iloc[0:0].copy()
    gc.collect()
    return ProviderInventoryBank(
        inventories=inventories,
        matrix_count=providers["matrix_database"].nunique(),
        provider_count=len(providers),
        solve_batch_count=batch_count,
        unit_responses=pd.concat(response_frames, ignore_index=True).sort_values(
            [
                "provider_anchor_year",
                "matrix_database",
                "provider_database",
                "provider_code",
                "method_id",
                "method_version",
                "impact_category_id",
            ],
            kind="stable",
            ignore_index=True,
        ),
    )


def assemble_principal_inventory(
    timeline: PrincipalTimeline,
    bank: ProviderInventoryBank,
) -> PrincipalInventory:
    """Combine annual supplier inventories and actual modeled elementary flows.

    Args:
        timeline: Regional Timex timeline, dated supplier demands and direct flows.
        bank: Unit biosphere inventories calculated for the required dated providers.

    Returns:
        Annual biosphere inventory rows keyed by year and Brightway flow identifier, with
        amounts in each flow's native unit. The result also reports the number of provider
        matrices, providers, and solve batches used.

    Notes:
        For provider ``p`` and year ``y``, each flow amount is
        ``provider_demand[p, y] * unit_inventory[p, flow]``. Actual direct flows
        already include the receiving load and temporal share. ``fsum`` combines
        both sources on the shared year/flow axis.

    Raises:
        ValueError: If a finite provider inventory and foreground demand produce
            a nonfinite annual elementary-flow quantity.
    """

    provider_keys = tuple(bank.inventories)
    provider_rows = {key: row for row, key in enumerate(provider_keys)}
    # Align provider inventories on one sorted flow axis and preserve native values.
    direct: dict[tuple[int, int], list[float]] = {}
    for raw in timeline.direct_inventory.itertuples(index=False, name=None):
        year, flow_id, amount = cast(tuple[int, int, float], raw)
        direct.setdefault((int(year), int(flow_id)), []).append(float(amount))
    direct_flow_ids = timeline.direct_inventory["flow_id"].to_numpy(dtype=np.int64)
    direct_years = timeline.direct_inventory["year"].to_numpy(dtype=np.int16)
    flow_ids = np.unique(
        np.concatenate(
            (direct_flow_ids, *(inventory.flow_ids for inventory in bank.inventories.values()))
        )
    )
    unit_amounts = np.zeros((len(provider_keys), len(flow_ids)), dtype=np.float64)
    for provider_row, key in enumerate(provider_keys):
        unit = bank.inventories[key]
        columns = np.searchsorted(flow_ids, unit.flow_ids)
        unit_amounts[provider_row, columns] = unit.amounts

    activations = timeline.activations.reset_index(drop=True)
    activation_provider_rows = np.fromiter(
        (
            provider_rows[
                (str(item.matrix_database), str(item.provider_database), str(item.provider_code))
            ]
            for item in activations.itertuples(index=False)
        ),
        dtype=np.int64,
        count=len(activations),
    )
    activation_years = activations["year"].to_numpy(dtype=np.int16, copy=False)
    activation_scales = activations["amount"].to_numpy(dtype=np.float64, copy=False)
    rows: list[dict[str, object]] = []
    for year in np.union1d(activation_years, direct_years):
        selected = np.flatnonzero(activation_years == year)
        provider_block = unit_amounts[activation_provider_rows[selected]]
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = provider_block * activation_scales[selected, np.newaxis]
        nonfinite = np.argwhere(~np.isfinite(scaled))
        if nonfinite.size:
            activation_column, flow_column = (int(value) for value in nonfinite[0])
            activation_row = int(selected[activation_column])
            provider_key = provider_keys[activation_provider_rows[activation_row]]
            raise ValueError(
                f"Year {int(year)} provider {provider_key!r} produced a nonfinite inventory "
                f"amount for flow {int(flow_ids[flow_column])}, "
                f"foreground amount {float(activation_scales[activation_row])!r}. "
                "Check the source exchanges and foreground demand before a new calculation."
            )
        for flow_column, flow_id in enumerate(flow_ids):
            present = provider_block[:, flow_column] != 0.0
            # A receiving flow or year may be absent from every supplier's B
            # matrix. Preserve its actual native identity on the same LCI axis.
            amount = fsum(
                chain(
                    (float(value) for value in scaled[present, flow_column]),
                    direct.get((int(year), int(flow_id)), ()),
                )
            )
            if amount != 0.0:
                rows.append(
                    {
                        "year": int(year),
                        "flow_id": int(flow_id),
                        "amount": amount,
                    }
                )
    inventory = pd.DataFrame(rows, columns=("year", "flow_id", "amount"))
    return PrincipalInventory(
        timeline=timeline.timeline,
        inventory=inventory,
        base_static_score=timeline.base_static_score,
        provider_matrix_count=bank.matrix_count,
        provider_count=bank.provider_count,
        provider_solve_batch_count=bank.solve_batch_count,
    )


def database_dates(
    foreground_database: str,
    timeline_databases: dict[int, str],
) -> dict[str, str | datetime]:
    """Return exact dates for the foreground and compact Timex family."""

    dates: dict[str, str | datetime] = {
        foreground_database: "dynamic",
    }
    for year, database in sorted(timeline_databases.items()):
        dates[database] = datetime(year, 1, 1)
    return dates
