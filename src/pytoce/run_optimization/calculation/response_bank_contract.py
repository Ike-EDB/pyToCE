"""Describe provider responses and construct their ordered method registry."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from pytoce.prepare_backgrounds.preparation.premise import (
    ACTIVE_BACKGROUND_SCENARIOS,
    GENERAL_BACKGROUND_ANCHOR_YEARS,
    PROVIDER_TECHNOLOGY_ANCHOR_YEARS,
)
from pytoce.shared.runtime.manifests import OperationStatus

RESPONSE_BANK_SCHEMA = "pytoce_optimization_response_bank_schema_v1"
_IDENTITY_PREFIX = f"{RESPONSE_BANK_SCHEMA}::"
_PROJECTION_PREFIX = "pytoce_optimization_response_projection_v3::"
_OWNER_RELATIVE = Path("data_processed/optimization_response_bank/four_remind")
_BACKGROUNDS = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)
_ANCHOR_YEARS = (*PROVIDER_TECHNOLOGY_ANCHOR_YEARS, *GENERAL_BACKGROUND_ANCHOR_YEARS)
_METHOD_COLUMNS = (
    "lcia_method",
    "selection_order",
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "unit",
    "factor_identity",
)
_METHOD_SIGNATURE_COLUMNS = (
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "unit",
    "factor_identity",
)
_PROVIDER_COLUMNS = (
    "background_scenario",
    "provider_anchor_year",
    "provider_service_id",
    "provider_service_code",
    "provider_unit",
    *_METHOD_SIGNATURE_COLUMNS,
    "unit_score",
    "unit_contribution_norm",
)
_SERVICE_COLUMNS = (
    "provider_service_id",
    "provider_service_code",
    "provider_unit",
)
_NATIVE_COLUMNS = (
    "background_scenario",
    "source_anchor_year",
    "source_database",
    "matrix_database",
    "endpoint_database",
    "endpoint_code",
    "product_database",
    "product_code",
    "endpoint_unit",
    *_METHOD_SIGNATURE_COLUMNS,
    "unit_score",
    "unit_contribution_norm",
)
_DIRECT_COLUMNS = (
    "flow_database",
    "flow_code",
    "flow_name",
    "flow_categories_json",
    "endpoint_unit",
    *_METHOD_SIGNATURE_COLUMNS,
    "factor",
    "present",
)
_WEIGHT_COLUMNS = (
    "background_scenario",
    "occurrence_year",
    "provider_anchor_year",
    "temporal_provider_share",
)
ProjectionRole = Literal["optimization", "result", "mandatory_climate"]


class ResponseBankError(ValueError):
    """A response bank cannot be selected or published."""


@dataclass(frozen=True, slots=True)
class ResponseBankMemberParent:
    """One prepared background represented separately in the response bank."""

    background_scenario: str
    prepared_background_identity: str
    prepared_background_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ResponseBankInputs:
    """Inputs and identities used to calculate one saved provider response set."""

    provider_route_identity: str
    prepared_background_shared_identity: str
    prepared_background_shared_manifest_sha256: str
    builder_identity: str
    member_parents: tuple[ResponseBankMemberParent, ...]


@dataclass(frozen=True, slots=True)
class OptimizationResponseBank:
    """Describe saved provider responses covering the requested LCIA methods."""

    status: OperationStatus
    identity: str
    response_method_union: str
    result_directory: Path


@dataclass(frozen=True, slots=True)
class LoadedOptimizationResponseBank:
    """One completed response bank loaded for calculation."""

    identity: str
    response_method_union: str
    provider_route_identity: str
    provider_service_registry_identity: str
    method_registry: pd.DataFrame
    provider_service_registry: pd.DataFrame
    provider_unit_responses: pd.DataFrame
    timex_year_weights: pd.DataFrame
    native_unit_responses: pd.DataFrame
    direct_characterization_factors: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ResponseBankProjection:
    """Response tables selected for one calculation purpose.

    ``role`` distinguishes optimization constraints, cumulative climate, and
    selected-result reporting. The response-bank and projection identities bind
    the subset to its completed source. Provider route and service identities
    bind ``provider_service_registry``. The method-registry identity binds its
    ordered LCIA rows. The remaining identities bind provider unit responses,
    Timex year weights, actual native endpoint responses, and direct-flow factors.
    Native endpoint amounts are evaluated on the actual PA mixture before the
    shared general-source year mapping applies. PA receiving uses the source parameters at each
    native anchor.
    The corresponding table contracts and producers define their dimensions
    and units; the response-bank JSON schema describes the manifest.
    """

    role: ProjectionRole
    response_bank_identity: str
    projection_identity: str
    provider_route_identity: str
    provider_service_registry_identity: str
    provider_service_registry: pd.DataFrame
    method_registry_identity: str
    method_registry: pd.DataFrame
    provider_unit_responses_identity: str
    provider_unit_responses: pd.DataFrame
    timex_year_weights_identity: str
    timex_year_weights: pd.DataFrame
    native_unit_responses_identity: str
    native_unit_responses: pd.DataFrame
    direct_characterization_factors_identity: str
    direct_characterization_factors: pd.DataFrame


def _method_signature_records(frame: pd.DataFrame) -> tuple[tuple[str, ...], ...]:
    """Convert LCIA signature columns to immutable string records."""
    return tuple(
        tuple(str(value) for value in row)
        for row in frame.loc[:, list(_METHOD_SIGNATURE_COLUMNS)].itertuples(
            index=False,
            name=None,
        )
    )


def method_registry_from_records(records: list[dict[str, object]]) -> pd.DataFrame:
    """Construct ordered LCIA coordinates with compact JSON stored as text."""
    return (
        pd.DataFrame.from_records(records, columns=_METHOD_COLUMNS)
        .astype(
            {
                column: "int64" if column == "selection_order" else "string"
                for column in _METHOD_COLUMNS
            }
        )
        .sort_values(
            ["lcia_method", "selection_order", *_METHOD_SIGNATURE_COLUMNS],
            kind="stable",
            ignore_index=True,
        )
    )
