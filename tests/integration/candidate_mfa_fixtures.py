"""MFA calculations and synthetic response tables for candidate worker tests."""

import json
from dataclasses import replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import cast

import pandas as pd
from infiltration_fixtures import resolved_infiltration_provider
from material_carbon_fixtures import prepared_material_carbon
from mfa_workspace_fixtures import prepare_mfa_workspace

from pytoce.prepare_backgrounds.preparation.ecoinvent import brightway_names
from pytoce.prepare_backgrounds.preparation.project_biosphere import load_lcia_catalogue
from pytoce.run_lca.calculation.foreground import optimization_activation_delta
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_lca.calculation.providers import load_provider_routes
from pytoce.run_lca.run_lca import DEFAULT_UPSTREAM_ANALYSIS_YEARS, lca_result_directory
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestContext,
    prepare_candidate_worker_request_context,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    _DIRECT_COLUMNS,
    _NATIVE_COLUMNS,
    ResponseBankInputs,
    ResponseBankMemberParent,
    method_registry_from_records,
)
from pytoce.run_optimization.calculation.response_bank_store import (
    _owner_lock,
    _resolved_owner,
    write_optimization_response_bank,
)
from pytoce.run_optimization.calculation.selected_reporting import (
    _effective_responses,
    _stable_group,
    _stage_response_adjustments,
)
from pytoce.run_optimization.calculation.worker_inputs import (
    prepare_candidate_worker_initializer,
)
from pytoce.run_optimization.input.allocated_cc_product import compile_allocated_cc_result
from pytoce.shared.runtime.filesystem import (
    canonical_json_bytes,
    canonical_table_sha256,
    file_sha256,
    payload_sha256,
)
from tests.integration.brightway_lca_fixtures import write_prepared_inputs
from tests.unit.run_optimization.allocated_cc_fixtures import _methods as allocated_cc_methods
from tests.unit.run_optimization.allocated_cc_fixtures import (
    _write_raw_result,
)
from tests.unit.run_optimization.response_bank_fixtures import _inputs, _weights

_BACKGROUND_SCENARIOS = tuple(sorted(_weights()["background_scenario"].unique()))
_WORKER_FOREGROUND = "candidate_worker_parent"


def _provider_tables(
    workspace: Path, methods: pd.DataFrame
) -> tuple[pd.DataFrame, ResponseBankInputs, frozenset[str]]:
    """Write synthetic prepared inputs and link response provenance to their manifests.

    The HDPE/PP fixture and its prospective interventions contain no polyamide.
    Demand, lifetime and circularity changes preserve that material partition.
    Its receiving records retain the synthetic metadata for that material partition.
    """
    authority = resolved_infiltration_provider(workspace / "native_infiltration")
    carbon = prepared_material_carbon(workspace / "fixture_native_carbon")
    shared, members = write_prepared_inputs(
        workspace,
        brightway_names(workspace, source_identity="sources::fixture"),
        methods=methods,
        provider=authority,
        carbon_rows=carbon,
        backgrounds=_BACKGROUND_SCENARIOS,
    )
    routes = load_provider_routes(shared)
    services = routes.routes.loc[
        :, ["provider_service_id", "provider_service_code"]
    ].drop_duplicates(ignore_index=True)
    services["provider_unit"] = "kilogram"
    landfill_service_ids = frozenset(
        routes.routes.loc[
            routes.routes["provider_demand_type"].eq("landfill_treatment"),
            "provider_service_id",
        ].astype(str)
    )
    inputs = _inputs()
    source_inputs = replace(
        inputs,
        provider_route_identity=routes.identity,
        prepared_background_shared_identity=cast(str, shared.payload["identity"]),
        prepared_background_shared_manifest_sha256=shared.sha256,
        member_parents=tuple(
            ResponseBankMemberParent(
                background_scenario=parent.background_scenario,
                prepared_background_identity=cast(
                    str, members[parent.background_scenario].payload["identity"]
                ),
                prepared_background_manifest_sha256=members[parent.background_scenario].sha256,
            )
            for parent in inputs.member_parents
        ),
    )
    return services.astype("string"), source_inputs, landfill_service_ids


def _response_rows(
    services: pd.DataFrame,
    methods: pd.DataFrame,
    *,
    landfill_service_ids: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Return invented unit responses, optionally nonzero only for landfill services."""

    signatures = cast(list[dict[str, object]], methods.to_dict(orient="records"))
    records: list[dict[str, object]] = []
    for background, anchor, service, method in product(
        _BACKGROUND_SCENARIOS,
        sorted(_weights()["provider_anchor_year"].unique()),
        cast(list[dict[str, object]], services.to_dict(orient="records")),
        signatures,
    ):
        unit_response = float(
            landfill_service_ids is None
            or str(service["provider_service_id"]) in landfill_service_ids
        )
        records.append(
            {
                "background_scenario": background,
                "provider_anchor_year": int(anchor),
                **service,
                **{
                    column: method[column]
                    for column in (
                        "method_id",
                        "method_version",
                        "impact_category_id",
                        "impact_category_tuple_json",
                        "unit",
                        "factor_identity",
                    )
                },
                "unit_score": unit_response,
                "unit_contribution_norm": unit_response,
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .astype(
            {
                column: "int64"
                if column == "provider_anchor_year"
                else "float64"
                if column in ("unit_score", "unit_contribution_norm")
                else "string"
                for column in records[0]
            }
        )
        .sort_values(
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
    )


def _baseline_descriptor(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    """Write one baseline LCA table and record its file and table identities."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {
        "path": path.name,
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "dtypes": frame.dtypes.astype(str).tolist(),
        "artifact_sha256": file_sha256(path),
        "canonical_sha256": canonical_table_sha256(frame),
    }


def _write_public_baseline_results(
    workspace: Path,
    context: CandidateWorkerRequestContext,
    foreground: str,
    source_inputs: ResponseBankInputs,
    *,
    result_lcia_method: tuple[str, ...] = ("ef31",),
) -> None:
    """Write selected-reporting parents from the fixture's existing activations and responses."""
    activations = context.physical_parent.activations
    delta = optimization_activation_delta(activations.assign(amount=0.0), activations)
    methods, effective = _effective_responses(context.result_projection)
    parents = {item.background_scenario: item for item in source_inputs.member_parents}
    selection = normalized_lcia_method(result_lcia_method)
    for background in context.search_scoring.optimization_projection.backgrounds:
        stage = _stage_response_adjustments(delta, effective, background=background)
        numerical_columns = {"score_adjustment": "score", "norm_adjustment": "contribution_norm"}
        discarded = [
            "characterization_background_scenario",
            "factor_identity",
            "changed_parent_score",
            "changed_candidate_score",
            "changed_parent_norm",
            "changed_candidate_norm",
        ]
        annual_stage = stage.drop(columns=discarded).rename(columns=numerical_columns)
        cumulative_keys = [
            column
            for column in annual_stage
            if column not in {"year", "physical_event_year", "score", "contribution_norm"}
        ]
        cumulative_stage = _stable_group(
            annual_stage, cast(list[str], cumulative_keys), ("score", "contribution_norm")
        )
        cumulative_stage["year"] = pd.Series(pd.NA, index=cumulative_stage.index, dtype="Int64")
        cumulative_stage["physical_event_year"] = pd.Series(
            pd.NA, index=cumulative_stage.index, dtype="Int64"
        )
        tables = {
            "annual_life_cycle_stage_scores.parquet": annual_stage,
            "life_cycle_stage_scores.parquet": cumulative_stage,
        }
        directory = lca_result_directory(
            workspace,
            foreground,
            background,
            selection,
            DEFAULT_UPSTREAM_ANALYSIS_YEARS,
        )
        artifacts = {
            filename: _baseline_descriptor(directory / filename, frame)
            for filename, frame in tables.items()
        }
        member = parents[background]
        payload = {
            "foreground_scenario": foreground,
            "background_scenario": background,
            "prepared_background_identity": member.prepared_background_identity,
            "prepared_background_manifest_sha256": (member.prepared_background_manifest_sha256),
            "lcia_methods": [
                {
                    "method_id": str(method.method_id),
                    "method_version": str(method.method_version),
                    "impact_category_id": str(method.impact_category_id),
                    "method_tuple": json.loads(str(method.impact_category_tuple_json)),
                    "factor_identity": str(method.factor_identity),
                }
                for method in methods.itertuples(index=False)
            ],
        }
        manifest = {
            "schema_version": "pytoce_lca_result_schema_v1",
            "identity": "pytoce_lca_result_schema_v1::" + payload_sha256(payload),
            "foreground_scenario": foreground,
            "background_scenario": background,
            "ssp_scenario": "SSP2",
            "identity_payload": payload,
            "artifacts": artifacts,
            "complete": True,
        }
        (directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def _prepare_worker_workspace(
    workspace: Path,
    source_root: Path,
    *,
    include_generous_case: bool = False,
    foreground_scenario: str = _WORKER_FOREGROUND,
    include_reporting_methods: bool = False,
    landfill_responses_only: bool = False,
    stock_floor: bool = False,
    include_public_baseline_results: bool = True,
    background_scenarios: tuple[str, ...] = _BACKGROUND_SCENARIOS,
) -> tuple[bytes, CandidateWorkerRequestContext]:
    """Write MFA inputs with synthetic unit responses and allocated carrying capacities.

    ``landfill_responses_only`` gives unit score and contribution norm one to
    services bound to landfill treatment routes and zero to all other services.
    Responses are set before writing their table. The allocated carrying capacities retain
    their fixed direct-case formula and optional generous-case value of 1e30.
    ``stock_floor`` selects the fixed prospective stock boundary. Candidate tests
    can omit the baseline stage results needed to report selected interventions.
    """

    parent = prepare_mfa_workspace(
        workspace,
        source_root,
        foreground_scenario=foreground_scenario,
        stock_floor=stock_floor,
    )

    methods = allocated_cc_methods()
    methods["factor_identity"] = methods["factor_identity"].map(
        lambda value: sha256(str(value).encode()).hexdigest()
    )
    methods = method_registry_from_records(
        cast(list[dict[str, object]], methods.to_dict("records"))
    )
    prepared_methods = (
        pd.concat((methods, _pb_reporting_methods()), ignore_index=True)
        if include_reporting_methods
        else methods
    )
    services, source_inputs, landfill_service_ids = _provider_tables(workspace, prepared_methods)
    cases = ("direct_case", "generous_case") if include_generous_case else ("direct_case",)
    raw_result = _write_raw_result(workspace, "worker_result", cases=cases)
    if include_generous_case:
        annual_path = raw_result / "annual_allocated_cc.csv"
        annual = pd.read_csv(annual_path)
        annual.loc[annual["allocated_cc_case"].eq("generous_case"), "allocated_cc"] = 1e30
        annual.to_csv(annual_path, index=False)
        cumulative_path = raw_result / "cumulative_gwp100_allocated_cc.csv"
        cumulative = pd.read_csv(cumulative_path)
        cumulative.loc[
            cumulative["allocated_cc_case"].eq("generous_case"),
            "cumulative_allocated_cc",
        ] = 1e30
        cumulative.to_csv(cumulative_path, index=False)
    allocated_cc = compile_allocated_cc_result(workspace, "worker_result")
    if landfill_responses_only:
        assert landfill_service_ids
    responses = _response_rows(
        services,
        methods,
        landfill_service_ids=landfill_service_ids if landfill_responses_only else None,
    )
    weights = _weights()
    owner = _resolved_owner(workspace)
    with _owner_lock(owner):
        write_optimization_response_bank(
            owner,
            response_method_union="ef31_response_v3",
            method_union_registry=methods,
            mandatory_climate_method_registry=methods.loc[
                methods["impact_category_id"].eq("climate_change")
            ].reset_index(drop=True),
            provider_service_registry=services,
            provider_unit_responses=responses,
            native_unit_responses=pd.DataFrame(columns=_NATIVE_COLUMNS).astype(
                {
                    column: "int64"
                    if column == "source_anchor_year"
                    else "float64"
                    if column in ("unit_score", "unit_contribution_norm")
                    else "string"
                    for column in _NATIVE_COLUMNS
                }
            ),
            direct_characterization_factors=pd.DataFrame(columns=_DIRECT_COLUMNS),
            timex_year_weights=weights,
            source_inputs=source_inputs,
        )
    initializer = prepare_candidate_worker_initializer(
        workspace,
        foreground_scenario=foreground_scenario,
        mfa_reference=parent.reference,
        allocated_cc=allocated_cc,
        allocated_cc_case=cases,
        response_method_union="ef31_response_v3",
        background_scenarios=background_scenarios,
    )
    context = prepare_candidate_worker_request_context(
        initializer,
        allocated_cc,
        parent_mfa=parent.mfa,
        result_lcia_method=normalized_lcia_method(("ef31",)),
    )
    if include_public_baseline_results:
        _write_public_baseline_results(workspace, context, foreground_scenario, source_inputs)
    return context.worker_initializer_payload, context


def _pb_reporting_methods() -> pd.DataFrame:
    """Return the authorized PB registry for reporting-only fixture responses."""
    specifications, _ = load_lcia_catalogue()
    return method_registry_from_records(
        [
            {
                "lcia_method": spec.method_id,
                "selection_order": spec.order,
                "method_id": spec.method_id,
                "method_version": spec.method_version,
                "impact_category_id": spec.impact_category_id,
                "impact_category_tuple_json": json.dumps(
                    list(spec.impact_category_tuple), separators=(",", ":")
                ),
                "unit": spec.unit,
                "factor_identity": sha256(
                    f"factor::{spec.impact_category_id}".encode()
                ).hexdigest(),
            }
            for spec in specifications
            if spec.method_id == "pb_lcia"
        ]
    )
