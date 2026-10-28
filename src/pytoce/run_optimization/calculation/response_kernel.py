"""Calculate provider unit responses for one prepared background scenario."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingFlow,
)
from pytoce.run_lca.calculation.contribution_accumulation import (
    ContributionMethod,
    create_provider_response_state,
)
from pytoce.run_lca.calculation.direct_contributions import (
    admit_direct_characterization_factors,
)
from pytoce.run_lca.calculation.lcia import (
    ImpactCategory,
    LCIAMethodSelection,
    resolve_lcia_methods_from_authority,
)
from pytoce.run_lca.calculation.pa_receiving_endpoints import pa_receiving_endpoints
from pytoce.run_lca.calculation.providers import PreparedProviderProduct
from pytoce.run_lca.calculation.timex import (
    build_provider_inventory_bank,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    _DIRECT_COLUMNS,
    _NATIVE_COLUMNS,
    _PROVIDER_COLUMNS,
    _WEIGHT_COLUMNS,
    method_registry_from_records,
)

_BUILDER_METHOD_IDENTITY = "pytoce_optimization_response_builder_v3"
_RESPONSE_BANK_READABLE_VERSION = "response_v3"


class ResponseBuilderError(RuntimeError):
    """Prepared backgrounds cannot supply an optimization response bank."""


@dataclass(frozen=True, slots=True)
class ResponseMethodSelection:
    """One response-bank union spanning package groups and exact methods."""

    readable_name: str
    selections: tuple[LCIAMethodSelection, ...]
    groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemberResponse:
    """Provider responses and direct elementary flows for one background scenario.

    Scores and norms are per actual reference-product unit. Native records keep
    the general source and selected matrix separate. Direct factors are admitted
    once for the union of these flow descriptors after all member calculations.
    """

    provider_unit_responses: pd.DataFrame
    timex_year_weights: pd.DataFrame
    native_unit_responses: pd.DataFrame
    direct_flows: tuple[PaReceivingFlow, ...]


def _pa_provider_rows(
    bd: Any, providers: PreparedProviderProduct
) -> tuple[pd.DataFrame, tuple[PaReceivingFlow, ...]]:
    """Resolve the listed native provider products in each ordinary source matrix.

    Reference-product IDs belong only to the live solve. The resulting response
    table stores portable product keys. Early premise technology is excluded:
    the 2025 receiving source supplies every modeled year through 2025.
    """
    rows: list[dict[str, object]] = []
    flows: dict[tuple[str, str], PaReceivingFlow] = {}
    for anchor, source in sorted(providers.pa_receiving_sources_by_year.items()):
        if anchor < 2025:
            continue
        endpoints, direct = pa_receiving_endpoints(providers.pa_receiving_reference, source)
        flows.update((flow.key, flow) for flow in direct)
        for endpoint in endpoints:
            activity = bd.get_node(database=endpoint.key[0], code=endpoint.key[1])
            product = activity.rp_exchange().input
            rows.append(
                {
                    "matrix_database": providers.database_by_year[anchor],
                    "provider_database": endpoint.key[0],
                    "provider_code": endpoint.key[1],
                    "provider_activity_id": int(activity.id),
                    "provider_product_id": int(product.id),
                    "provider_anchor_year": anchor,
                    "source_database": source.general_background_database,
                    "year": anchor,
                    "amount": 0.0,
                }
            )
    return pd.DataFrame.from_records(rows), tuple(flow for _, flow in sorted(flows.items()))


def _direct_factor_table(
    names: BrightwayNames,
    methods: tuple[ImpactCategory, ...],
    registry: pd.DataFrame,
    flows: tuple[PaReceivingFlow, ...],
) -> pd.DataFrame:
    """Admit portable direct-flow factors once for the complete member union.

    Use the same processed native factor owner as ordinary LCA. An explicit
    registered zero and an absent factor remain distinct in ``present``; both retain their admitted
    identities.
    """
    with selected_brightway_project(names) as bd:
        state = create_provider_response_state(
            bd,
            tuple(
                ContributionMethod(
                    method.method_id,
                    method.method_version,
                    method.impact_category_id,
                    method.method_tuple,
                    method.unit,
                )
                for method in methods
            ),
        )
        ids = [int(bd.get_node(database=flow.key[0], code=flow.key[1]).id) for flow in flows]
        admitted = admit_direct_characterization_factors(state, ids)
    records = []
    method_positions = {
        json.dumps(method.method_tuple, ensure_ascii=True, separators=(",", ":")): index
        for index, method in enumerate(methods)
    }
    for method in registry.to_dict("records"):
        method_index = method_positions[str(method["impact_category_tuple_json"])]
        for flow_index, flow in enumerate(flows):
            records.append(
                {
                    "flow_database": flow.key[0],
                    "flow_code": flow.key[1],
                    "flow_name": flow.name,
                    "flow_categories_json": json.dumps(flow.categories, separators=(",", ":")),
                    "endpoint_unit": flow.unit,
                    **method,
                    "factor": float(admitted.factors[method_index, flow_index]),
                    "present": bool(admitted.present[method_index, flow_index]),
                }
            )
    return pd.DataFrame.from_records(records, columns=_DIRECT_COLUMNS).astype(
        {
            column: "float64" if column == "factor" else "bool" if column == "present" else "string"
            for column in _DIRECT_COLUMNS
        }
    )


def _service_registry(routes: pd.DataFrame) -> pd.DataFrame:
    """Extract unique provider services in deterministic identifier order."""
    columns = ["provider_service_id", "provider_service_code"]
    return (
        routes.loc[:, columns]
        .drop_duplicates(ignore_index=True)
        .astype("string")
        .sort_values(columns, kind="stable", ignore_index=True)
    )


def _method_registry(methods: Sequence[ImpactCategory], groups: Sequence[str]) -> pd.DataFrame:
    """Construct the LCIA method records used in response calculations."""
    group_ids = set(groups)
    records = [
        {
            "lcia_method": method.method_id
            if method.method_id in group_ids
            else json.dumps(list(method.method_tuple), ensure_ascii=True, separators=(",", ":")),
            "selection_order": method.selection_order,
            "method_id": method.method_id,
            "method_version": method.method_version,
            "impact_category_id": method.impact_category_id,
            "impact_category_tuple_json": json.dumps(
                list(method.method_tuple), ensure_ascii=True, separators=(",", ":")
            ),
            "unit": method.unit,
            "factor_identity": method.factor_identity,
        }
        for method in methods
    ]
    return method_registry_from_records(records)


def _provider_rows(
    bd: Any, providers: PreparedProviderProduct, services: pd.DataFrame
) -> pd.DataFrame:
    """Resolve provider and reference-product IDs at each prepared anchor."""
    from bw2data.errors import UnknownObject  # type: ignore[import-untyped]

    rows: list[dict[str, object]] = []
    for anchor_year, database in providers.database_by_year.items():
        for service_id, service_code in services.itertuples(index=False, name=None):
            try:
                node = bd.get_node(database=str(database), code=str(service_code))
                product = node.rp_exchange().input
            except (UnknownObject, ValueError) as exc:
                raise ResponseBuilderError(
                    "Cannot resolve a prepared provider and its single reference product: "
                    f"database={database!r}, year={int(anchor_year)}, "
                    f"service={service_code!r}. Create a new workspace "
                    "and run prepare_backgrounds again."
                ) from exc
            rows.append(
                {
                    "provider_service_id": str(service_id),
                    "matrix_database": str(database),
                    "provider_database": str(database),
                    "provider_code": str(service_code),
                    "provider_activity_id": int(node.id),
                    "provider_product_id": int(product.id),
                    "provider_anchor_year": int(anchor_year),
                    "year": int(anchor_year),
                    "amount": 0.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def _timex_weights(providers: PreparedProviderProduct, background_scenario: str) -> pd.DataFrame:
    """Interpolate the provider-technology anchors used by ordinary Timex.

    The prepared general-background crosswalk separately keeps ecoinvent through
    2025. Provider services have their own 2020 and 2025 technology anchors;
    conflating those authorities changes the provider response projection.
    """

    anchors = tuple(sorted(providers.database_by_year))
    rows: list[dict[str, object]] = []
    for stage_year in range(2019, 2101):
        lower = max((anchor for anchor in anchors if anchor <= stage_year), default=anchors[0])
        upper = min((anchor for anchor in anchors if anchor >= stage_year), default=anchors[-1])
        if lower == upper:
            by_anchor = {lower: 1.0}
        else:
            upper_share = (stage_year - lower) / (upper - lower)
            by_anchor = {lower: 1.0 - upper_share, upper: upper_share}
        rows.extend(
            {
                "background_scenario": background_scenario,
                "occurrence_year": int(stage_year),
                "provider_anchor_year": anchor,
                "temporal_provider_share": share,
            }
            for anchor, share in sorted(by_anchor.items())
        )
    result = pd.DataFrame.from_records(rows, columns=_WEIGHT_COLUMNS)
    result["background_scenario"] = result["background_scenario"].astype("string")
    return result


def _resolved_response_methods(
    names: BrightwayNames,
    selection: ResponseMethodSelection,
    impact_categories: pd.DataFrame,
) -> tuple[tuple[ImpactCategory, ...], pd.DataFrame]:
    """Admit exact prepared methods and their factors before provider calculations.

    The admitted selections have disjoint method sets. Preserve their order. The
    returned registry uses the same textual representation as its stored form.
    """
    with selected_brightway_project(names) as bd:
        resolved = tuple(
            method
            for request in selection.selections
            for method in resolve_lcia_methods_from_authority(impact_categories, request, bd)
        )
    methods = tuple(replace(method, selection_order=order) for order, method in enumerate(resolved))
    registry = _method_registry(methods, selection.groups)
    return methods, registry


def _member_response(
    names: BrightwayNames,
    background_scenario: str,
    methods: tuple[ImpactCategory, ...],
    registry: pd.DataFrame,
    services: pd.DataFrame,
    providers: PreparedProviderProduct,
) -> MemberResponse:
    """Calculate provider LCIA responses for one prepared background scenario.

    The caller has resolved the selected LCIA methods and prepared providers.
    Every provider service is resolved to its Brightway product and characterized per unit through
    the same batched matrix calculation used by ordinary LCA. The returned year
    weights map provider anchor years to activity years from 2019 to 2100.

    Args:
        names: Brightway project, store, and prepared database names.
        background_scenario: REMIND background scenario being calculated.
        methods: Admitted exact methods, in requested order.
        registry: Matching method coordinates, native units and factor identities.
        services: Prepared provider service identifiers and provider codes.
        providers: Admitted provider and timeline databases by anchor year.

    Returns:
        Provider service responses, year weights, native product responses and direct
        flow descriptors. The latter are characterized once across members.
    """
    with selected_brightway_project(names) as bd:
        activations = _provider_rows(bd, providers, services)
        native_requests, direct_flows = _pa_provider_rows(bd, providers)
        state = create_provider_response_state(
            bd,
            tuple(
                ContributionMethod(
                    method_id=method.method_id,
                    method_version=method.method_version,
                    impact_category_id=method.impact_category_id,
                    method_tuple=method.method_tuple,
                    unit=method.unit,
                )
                for method in methods
            ),
        )

        bank = build_provider_inventory_bank(
            bd, pd.concat([activations, native_requests], ignore_index=True), state
        )
    keys = ["matrix_database", "provider_database", "provider_code", "provider_anchor_year"]
    factors = registry.loc[
        :, ["method_id", "method_version", "impact_category_id", "factor_identity"]
    ]
    native = bank.unit_responses.merge(
        native_requests[[*keys, "source_database"]], on=keys, how="inner"
    ).rename(
        columns={
            "provider_anchor_year": "source_anchor_year",
            "provider_database": "endpoint_database",
            "provider_code": "endpoint_code",
            "provider_product_database": "product_database",
            "provider_product_code": "product_code",
            "provider_unit": "endpoint_unit",
            "method_tuple_json": "impact_category_tuple_json",
            "impact_unit": "unit",
        }
    )
    native.insert(0, "background_scenario", background_scenario)
    native = (
        native.merge(factors, on=["method_id", "method_version", "impact_category_id"], how="left")
        .loc[:, list(_NATIVE_COLUMNS)]
        .astype(
            {
                column: "int64"
                if column == "source_anchor_year"
                else "float64"
                if column in ("unit_score", "unit_contribution_norm")
                else "string"
                for column in _NATIVE_COLUMNS
            }
        )
        .sort_values(
            [
                "source_anchor_year",
                "matrix_database",
                "endpoint_database",
                "endpoint_code",
                "method_id",
                "method_version",
                "impact_category_id",
            ],
            kind="stable",
            ignore_index=True,
        )
    )
    responses = _service_responses(bank.unit_responses, activations, registry, background_scenario)
    return MemberResponse(
        responses, _timex_weights(providers, background_scenario), native, direct_flows
    )


def _service_responses(
    unit_responses: pd.DataFrame,
    activations: pd.DataFrame,
    registry: pd.DataFrame,
    background_scenario: str,
) -> pd.DataFrame:
    """Join fixed-service labels after the native product calculation.

    Only admitted service requests enter this view. Other actual endpoint
    responses retain their portable identities in the separate native table.
    The selected responses retain their scores, norms, factors, and product units.
    """
    # Service identifiers belong to the package registry calculation. Receiving endpoints retain
    # their database, activity, and product identities.
    keys = ["matrix_database", "provider_database", "provider_code", "provider_anchor_year"]
    responses = unit_responses.merge(
        activations[[*keys, "provider_service_id"]], on=keys, how="inner"
    ).rename(
        columns={
            "provider_code": "provider_service_code",
            "method_tuple_json": "impact_category_tuple_json",
            "impact_unit": "unit",
        }
    )
    responses.insert(0, "background_scenario", background_scenario)
    factors = registry.loc[
        :, ["method_id", "method_version", "impact_category_id", "factor_identity"]
    ]
    responses = responses.merge(
        factors,
        on=["method_id", "method_version", "impact_category_id"],
        how="left",
    )
    responses = responses.loc[:, list(_PROVIDER_COLUMNS)].sort_values(
        [
            "background_scenario",
            "provider_anchor_year",
            "provider_service_id",
            "method_id",
            "method_version",
            "impact_category_id",
        ],
        kind="stable",
        ignore_index=True,
    )
    for column in (
        "background_scenario",
        "provider_service_id",
        "provider_service_code",
        "provider_unit",
        "method_id",
        "method_version",
        "impact_category_id",
        "impact_category_tuple_json",
        "unit",
        "factor_identity",
    ):
        responses[column] = responses[column].astype("string")
    return responses
