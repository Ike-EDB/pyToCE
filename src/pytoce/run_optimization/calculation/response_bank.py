"""Read optimization response banks and select method projections."""

import json
from collections.abc import Sequence

import pandas as pd

from pytoce.run_lca.calculation.lcia import (
    LCIAMethodSelection,
    lcia_method_discriminators,
)
from pytoce.shared.runtime.filesystem import (
    canonical_table_sha256,
    payload_sha256,
)

from .response_bank_contract import (
    _METHOD_SIGNATURE_COLUMNS,
    _PROJECTION_PREFIX,
    _SERVICE_COLUMNS,
    LoadedOptimizationResponseBank,
    OptimizationResponseBank,
    ProjectionRole,
    ResponseBankProjection,
    _method_signature_records,
    method_registry_from_records,
)


def read_optimization_response_bank(
    response_bank: OptimizationResponseBank,
    *,
    background_scenarios: Sequence[str] | None = None,
) -> LoadedOptimizationResponseBank:
    """Read admitted member rows and unchanged factors from the shared complete bank."""
    filters = (
        None
        if background_scenarios is None
        else [("background_scenario", "in", list(background_scenarios))]
    )
    directory = response_bank.result_directory
    payload = json.loads((directory / "manifest.json").read_bytes())["identity_payload"]
    return LoadedOptimizationResponseBank(
        identity=response_bank.identity,
        response_method_union=response_bank.response_method_union,
        provider_route_identity=str(payload["provider_route_identity"]),
        provider_service_registry_identity=str(payload["provider_service_registry_identity"]),
        method_registry=method_registry_from_records(payload["method_union_registry"]),
        provider_service_registry=pd.DataFrame.from_records(
            payload["provider_service_registry"], columns=_SERVICE_COLUMNS
        ).astype("string"),
        provider_unit_responses=pd.read_parquet(
            directory / "provider_unit_responses.parquet", filters=filters
        ),
        timex_year_weights=pd.read_parquet(
            directory / "timex_year_weights.parquet", filters=filters
        ),
        native_unit_responses=pd.read_parquet(
            directory / "native_unit_responses.parquet", filters=filters
        ),
        direct_characterization_factors=pd.read_parquet(
            directory / "direct_characterization_factors.parquet"
        ),
    )


def select_response_bank_method_registry(
    method_registry: pd.DataFrame,
    selection: LCIAMethodSelection,
) -> pd.DataFrame:
    """Select admitted LCIA methods from their completed response bank.

    Package method groups select their complete registered category set. An
    exact Brightway method selects its single registered method-tuple row. The
    returned registry is suitable for :func:`project_optimization_response_bank`.
    """

    discriminators = lcia_method_discriminators(selection)
    if selection.groups:
        selected = method_registry.loc[method_registry["lcia_method"].isin(discriminators)]
    else:
        selected = method_registry.loc[
            method_registry["impact_category_tuple_json"].eq(discriminators[0])
        ]
    return selected.reset_index(drop=True)


def response_projection_selection_identity(
    lcia_method: Sequence[str],
    projection: ResponseBankProjection,
) -> str:
    """Identify an ordered LCIA selection and its method registry."""

    methods = tuple(lcia_method)
    return "pytoce_optimization_result_lcia_selection_v1::" + payload_sha256(
        {
            "lcia_method": list(methods),
            "method_registry_identity": projection.method_registry_identity,
        }
    )


def project_optimization_response_bank(
    response_bank: LoadedOptimizationResponseBank,
    method_registry: pd.DataFrame,
    *,
    role: ProjectionRole,
) -> ResponseBankProjection:
    """Select the response tables for one LCIA use."""

    requested = method_registry
    services = response_bank.provider_service_registry
    requested_signatures = set(_method_signature_records(requested))
    responses = response_bank.provider_unit_responses
    weights = response_bank.timex_year_weights
    selected_index = pd.MultiIndex.from_tuples(
        sorted(requested_signatures),
        names=_METHOD_SIGNATURE_COLUMNS,
    )

    def select(frame: pd.DataFrame) -> pd.DataFrame:
        """Keep complete method signatures from this one response product."""
        signatures = pd.MultiIndex.from_frame(
            frame.loc[:, list(_METHOD_SIGNATURE_COLUMNS)].astype("string")
        )
        return frame.loc[signatures.isin(selected_index)].reset_index(drop=True)

    selected = select(responses)
    native = select(response_bank.native_unit_responses)
    direct = select(response_bank.direct_characterization_factors)
    registry_identity = canonical_table_sha256(requested)
    weights_identity = canonical_table_sha256(weights)
    response_identity = canonical_table_sha256(selected)
    native_identity = canonical_table_sha256(native)
    direct_identity = canonical_table_sha256(direct)
    projection_payload = {
        "role": role,
        "provider_route_identity": response_bank.provider_route_identity,
        "provider_service_registry_identity": response_bank.provider_service_registry_identity,
        "method_registry_identity": registry_identity,
        "provider_unit_responses_identity": response_identity,
        "timex_year_weights_identity": weights_identity,
        "native_unit_responses_identity": native_identity,
        "direct_characterization_factors_identity": direct_identity,
    }
    return ResponseBankProjection(
        role=role,
        response_bank_identity=response_bank.identity,
        projection_identity=_PROJECTION_PREFIX + payload_sha256(projection_payload),
        provider_route_identity=response_bank.provider_route_identity,
        provider_service_registry_identity=response_bank.provider_service_registry_identity,
        provider_service_registry=services,
        method_registry_identity=registry_identity,
        method_registry=requested,
        provider_unit_responses_identity=response_identity,
        provider_unit_responses=selected,
        timex_year_weights_identity=weights_identity,
        timex_year_weights=weights,
        native_unit_responses_identity=native_identity,
        native_unit_responses=native,
        direct_characterization_factors_identity=direct_identity,
        direct_characterization_factors=direct,
    )
