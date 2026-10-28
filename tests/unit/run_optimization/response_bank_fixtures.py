"""Complete response-bank tables shared by response owner tests."""

from itertools import product
from pathlib import Path

import pandas as pd

from pytoce.prepare_backgrounds.preparation.premise import (
    ACTIVE_BACKGROUND_SCENARIOS,
    GENERAL_BACKGROUND_ANCHOR_YEARS,
    PROVIDER_TECHNOLOGY_ANCHOR_YEARS,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    _DIRECT_COLUMNS,
    _NATIVE_COLUMNS,
    OptimizationResponseBank,
    ResponseBankInputs,
    ResponseBankMemberParent,
    method_registry_from_records,
)
from pytoce.run_optimization.calculation.response_bank_store import (
    _owner_lock,
    _resolved_owner,
    write_optimization_response_bank,
)

_BACKGROUNDS = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)
_ANCHORS = (*PROVIDER_TECHNOLOGY_ANCHOR_YEARS, *GENERAL_BACKGROUND_ANCHOR_YEARS)
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
_PROVIDER_COLUMNS = (
    "background_scenario",
    "provider_anchor_year",
    "provider_service_id",
    "provider_service_code",
    "provider_unit",
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "unit",
    "factor_identity",
    "unit_score",
    "unit_contribution_norm",
)
_METHOD_SIGNATURE_COLUMNS = (
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "unit",
    "factor_identity",
)


def _methods() -> pd.DataFrame:
    """Return EF and PB catalog labels with synthetic fixture factor identities."""

    return pd.DataFrame.from_records(
        [
            {
                "lcia_method": "ef31",
                "selection_order": 0,
                "method_id": "ef31",
                "method_version": "3.1",
                "impact_category_id": "climate_change",
                "impact_category_tuple_json": (
                    '["EF v3.1","climate change","global warming potential (GWP100)"]'
                ),
                "unit": "kg CO2-Eq",
                "factor_identity": "1" * 64,
            },
            {
                "lcia_method": "pb_lcia",
                "selection_order": 1,
                "method_id": "pb_lcia",
                "method_version": "0.9.2",
                "impact_category_id": "climate_change_energy_imbalance",
                "impact_category_tuple_json": (
                    '["AESA (PBs-LCIA)","0.9.2","climate change",'
                    '"energy imbalance at top-of-atmosphere"]'
                ),
                "unit": "Wm-2",
                "factor_identity": "2" * 64,
            },
        ],
        columns=_METHOD_COLUMNS,
    )


def _services() -> pd.DataFrame:
    """Return the provider services represented in the response fixture."""

    return pd.DataFrame.from_records(
        [
            {
                "provider_service_id": "collection",
                "provider_service_code": "provider_service__collection",
                "provider_unit": "kilogram",
            },
            {
                "provider_service_id": "virgin_resin",
                "provider_service_code": "provider_service__virgin_resin",
                "provider_unit": "kilogram",
            },
        ]
    ).astype("string")


def _provider_responses(methods: pd.DataFrame) -> pd.DataFrame:
    """Return complete provider responses for every member and anchor."""

    services = _services()
    records: list[dict[str, object]] = []
    for background_i, background in enumerate(_BACKGROUNDS):
        for anchor, service, method in product(
            _ANCHORS,
            services.to_dict(orient="records"),
            methods.to_dict(orient="records"),
        ):
            score = (
                float(background_i + 1)
                * float(anchor - 2018)
                * (1.0 if method["method_id"] == "ef31" else -2.0)
            )
            records.append(
                {
                    "background_scenario": background,
                    "provider_anchor_year": anchor,
                    **service,
                    "method_id": method["method_id"],
                    "method_version": method["method_version"],
                    "impact_category_id": method["impact_category_id"],
                    "impact_category_tuple_json": method["impact_category_tuple_json"],
                    "unit": method["unit"],
                    "factor_identity": method["factor_identity"],
                    "unit_score": score,
                    "unit_contribution_norm": abs(score) + 1.0,
                }
            )
    return (
        pd.DataFrame.from_records(records, columns=_PROVIDER_COLUMNS)
        .astype(
            {
                column: "int64"
                if column == "provider_anchor_year"
                else "float64"
                if column in ("unit_score", "unit_contribution_norm")
                else "string"
                for column in _PROVIDER_COLUMNS
            }
        )
        .sort_values(
            [
                "background_scenario",
                "provider_anchor_year",
                "provider_service_id",
                *_METHOD_SIGNATURE_COLUMNS,
            ],
            kind="stable",
            ignore_index=True,
        )
    )


def _weights() -> pd.DataFrame:
    """Assign each fixture member-year to one provider anchor for storage and projection tests."""

    records: list[dict[str, object]] = []
    for background, year in product(_BACKGROUNDS, range(2019, 2101)):
        anchor = max((value for value in _ANCHORS if value <= year), default=2020)
        records.append(
            {
                "background_scenario": background,
                "occurrence_year": year,
                "provider_anchor_year": anchor,
                "temporal_provider_share": 1.0,
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .astype({"background_scenario": "string"})
        .sort_values(
            ["background_scenario", "occurrence_year", "provider_anchor_year"],
            kind="stable",
            ignore_index=True,
        )
    )


def _native_responses(methods: pd.DataFrame) -> pd.DataFrame:
    """Represent one native electricity endpoint independently of service IDs."""
    return pd.DataFrame.from_records(
        [
            {
                "background_scenario": background,
                "source_anchor_year": 2025,
                "source_database": "ecoinvent_original",
                "matrix_database": f"{background}_frontier_2025",
                "endpoint_database": "ecoinvent_original",
                "endpoint_code": "electricity_supply",
                "product_database": "ecoinvent_original",
                "product_code": "electricity_product",
                "endpoint_unit": "kilowatt hour",
                **method,
                "unit_score": float(index + 1),
                "unit_contribution_norm": float(index + 3),
            }
            for background in _BACKGROUNDS
            for index, method in enumerate(methods.to_dict("records"))
        ],
        columns=_NATIVE_COLUMNS,
    ).astype(
        {
            column: "int64"
            if column == "source_anchor_year"
            else "float64"
            if column in ("unit_score", "unit_contribution_norm")
            else "string"
            for column in _NATIVE_COLUMNS
        }
    )


def _direct_factors(methods: pd.DataFrame) -> pd.DataFrame:
    """Preserve positive, explicit-zero and absent native factors on portable flows."""
    return pd.DataFrame.from_records(
        [
            {
                "flow_database": "biosphere",
                "flow_code": code,
                "flow_name": code,
                "flow_categories_json": '["air"]',
                "endpoint_unit": "kilogram",
                **method,
                "factor": factor,
                "present": present,
            }
            for method in methods.to_dict("records")
            for code, factor, present in (
                ("positive", 2.0, True),
                ("zero", 0.0, True),
                ("absent", 0.0, False),
            )
        ],
        columns=_DIRECT_COLUMNS,
    ).astype(
        {
            column: "float64" if column == "factor" else "bool" if column == "present" else "string"
            for column in _DIRECT_COLUMNS
        }
    )


def _inputs() -> ResponseBankInputs:
    """Return immutable parent identities for a complete response bank."""

    return ResponseBankInputs(
        provider_route_identity="provider_routes::fixture",
        prepared_background_shared_identity="prepared_shared::fixture",
        prepared_background_shared_manifest_sha256="3" * 64,
        builder_identity="provider_response::fixture",
        member_parents=tuple(
            ResponseBankMemberParent(
                background_scenario=background,
                prepared_background_identity=f"prepared::{background}",
                prepared_background_manifest_sha256=f"{index + 5:x}" * 64,
            )
            for index, background in enumerate(_BACKGROUNDS)
        ),
    )


def _write_bank(
    tmp_path: Path,
    *,
    methods: pd.DataFrame | None = None,
    response_method_union: str = "ef31_pb_lcia",
) -> OptimizationResponseBank:
    """Write and return a complete optimization response-bank fixture."""

    methods = method_registry_from_records(
        (_methods() if methods is None else methods).to_dict("records")
    )
    owner = _resolved_owner(tmp_path)
    with _owner_lock(owner):
        return write_optimization_response_bank(
            owner,
            response_method_union=response_method_union,
            method_union_registry=methods,
            mandatory_climate_method_registry=methods.loc[
                methods["impact_category_id"].eq("climate_change")
            ].reset_index(drop=True),
            provider_service_registry=_services(),
            provider_unit_responses=_provider_responses(methods),
            timex_year_weights=_weights(),
            native_unit_responses=_native_responses(methods),
            direct_characterization_factors=_direct_factors(methods),
            source_inputs=_inputs(),
        )
