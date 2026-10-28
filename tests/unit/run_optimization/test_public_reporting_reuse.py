"""Reporting selections reuse the same public EF search and candidate payloads."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytoce import run_optimization
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_mfa.model.result_inputs import read_mfa_result, read_mfa_result_manifest
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestContext,
    prepare_candidate_worker_request_context,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    _DIRECT_COLUMNS,
    _NATIVE_COLUMNS,
    _SERVICE_COLUMNS,
    ResponseBankInputs,
    ResponseBankMemberParent,
    method_registry_from_records,
)
from pytoce.run_optimization.calculation.response_bank_store import (
    _owner_lock,
    _resolved_owner,
    write_optimization_response_bank,
)
from pytoce.run_optimization.calculation.worker_inputs import (
    prepare_candidate_worker_initializer,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    decode_candidate_worker_result,
)
from pytoce.run_optimization.input.allocated_cc_product import compile_allocated_cc_result
from pytoce.run_optimization.run_optimization import normalize_run_optimization_request
from pytoce.run_optimization.runtime.result_store import OptimizationResultTarget
from tests.integration.candidate_mfa_fixtures import (
    _WORKER_FOREGROUND,
    _pb_reporting_methods,
    _prepare_worker_workspace,
    _response_rows,
    _write_public_baseline_results,
)


def _reporting_contexts(
    workspace: Path,
) -> tuple[
    CandidateWorkerRequestContext, CandidateWorkerRequestContext, CandidateWorkerRequestContext
]:
    """Add a reporting bank after an EF-only search, preserving all prepared parents."""
    owner = _resolved_owner(workspace)
    source = json.loads((owner / "ef31_response_v3/manifest.json").read_bytes())["identity_payload"]
    inputs = ResponseBankInputs(
        provider_route_identity=source["provider_route_identity"],
        prepared_background_shared_identity=source["prepared_background_shared_identity"],
        prepared_background_shared_manifest_sha256=source[
            "prepared_background_shared_manifest_sha256"
        ],
        builder_identity=source["builder_identity"],
        member_parents=tuple(ResponseBankMemberParent(**row) for row in source["member_parents"]),
    )
    ef = method_registry_from_records(source["method_union_registry"])
    methods = method_registry_from_records(
        pd.concat((ef, _pb_reporting_methods()), ignore_index=True).to_dict("records")
    )
    services = pd.DataFrame.from_records(
        source["provider_service_registry"], columns=_SERVICE_COLUMNS
    ).astype("string")
    with _owner_lock(owner):
        write_optimization_response_bank(
            owner,
            response_method_union="ef31_pb_lcia_response_v3",
            method_union_registry=methods,
            mandatory_climate_method_registry=ef.loc[
                ef.impact_category_id.eq("climate_change")
            ].reset_index(drop=True),
            provider_service_registry=services,
            provider_unit_responses=_response_rows(services, methods),
            native_unit_responses=pd.DataFrame(columns=_NATIVE_COLUMNS),
            direct_characterization_factors=pd.DataFrame(columns=_DIRECT_COLUMNS),
            timex_year_weights=pd.read_parquet(
                owner / "ef31_response_v3/timex_year_weights.parquet"
            ),
            source_inputs=inputs,
        )
    reference = read_mfa_result_manifest(
        workspace / "results" / _WORKER_FOREGROUND / "mfa_results/manifest.json"
    )
    parent = read_mfa_result(reference)
    acc = compile_allocated_cc_result(workspace, "worker_result")
    initializer = prepare_candidate_worker_initializer(
        workspace,
        foreground_scenario=_WORKER_FOREGROUND,
        mfa_reference=reference,
        allocated_cc=acc,
        allocated_cc_case=("generous_case",),
        response_method_union="ef31_pb_lcia_response_v3",
    )
    exact_pb = tuple(json.loads(_pb_reporting_methods().impact_category_tuple_json.iloc[0]))
    contexts = []
    for selection in (("ef31", "pb_lcia"), exact_pb, ("ef31",)):
        context = prepare_candidate_worker_request_context(
            initializer,
            acc,
            parent_mfa=parent,
            result_lcia_method=normalized_lcia_method(selection),
        )
        if selection != ("ef31",):
            _write_public_baseline_results(
                workspace, context, _WORKER_FOREGROUND, inputs, result_lcia_method=selection
            )
        contexts.append(context)
    return contexts[0], contexts[1], contexts[2]


def _cache_file_state(cache: Path) -> dict[str, tuple[int, int]]:
    """Record modification times and sizes of completed candidate-cache files."""
    return {
        str(path.relative_to(cache)): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in cache.rglob("*")
        if path.is_file()
    }


@pytest.mark.scientific_integration
def test_public_reporting_methods_reuse_ef_candidate_cache(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> None:
    """Run one bounded search, then publish full/exact reporting from its cache.

    This HDPE/PP parent exercises source parsing, candidates, worker communication,
    persistent cache, and selected-result publication through the package implementation.
    The separate PA projection test covers nonempty native/direct method subsets.
    """
    workspace = tmp_path / "workspace"
    _initializer, initial = _prepare_worker_workspace(
        workspace,
        remind_mfa_source_root,
        include_generous_case=True,
        include_reporting_methods=True,
    )
    # Match the public one-case request rather than the fixture's two-case context.
    reference = read_mfa_result_manifest(
        workspace / "results" / _WORKER_FOREGROUND / "mfa_results/manifest.json"
    )
    acc = compile_allocated_cc_result(workspace, "worker_result")
    initializer = prepare_candidate_worker_initializer(
        workspace,
        foreground_scenario=_WORKER_FOREGROUND,
        mfa_reference=reference,
        allocated_cc=acc,
        allocated_cc_case=("generous_case",),
        response_method_union="ef31_response_v3",
    )
    initial = prepare_candidate_worker_request_context(
        initializer,
        acc,
        parent_mfa=read_mfa_result(reference),
        result_lcia_method=normalized_lcia_method("ef31"),
    )
    arguments = {
        "foreground_scenario": _WORKER_FOREGROUND,
        "allocated_cc_result": "worker_result",
        "optimization_block": "demand_lifetime_frontier",
        "allocated_cc_case": "generous_case",
    }
    first = run_optimization(**arguments, lcia_method="ef31")
    manifests = [json.loads(first.products[0].manifest_path.read_bytes())]
    expected_roles = pd.read_parquet(first.products[0].result_directory / "selection_roles.parquet")
    request = normalize_run_optimization_request(**arguments, lcia_method="ef31")
    target = OptimizationResultTarget(
        workspace,
        _WORKER_FOREGROUND,
        "worker_result",
        "ef31",
        request.readable_request_selection,
        initial_grid_step_pct=20,
        endpoint_refinement_steps_pct=(10, 5),
    )
    cache = target.root / ".candidate_results"
    published = _cache_file_state(cache)
    payloads = sorted(cache.glob("*/payload.bin"))
    assert payloads
    saved = decode_candidate_worker_result(payloads[0].read_bytes())
    assert saved.pa_scores is not None and saved.pa_scores.annual_scores.shape[-1] == 16

    full, exact, ef_from_superset = _reporting_contexts(workspace)
    for context in (full, exact, ef_from_superset):
        assert context.planner.identity == initial.planner.identity
        assert (
            context.planner.worker_authority_identity == initial.planner.worker_authority_identity
        )
        assert (
            context.worker_scientific_authority_identity
            == initial.worker_scientific_authority_identity
        )
        assert context.search_scoring.identity == initial.search_scoring.identity
        assert len(context.search_scoring.pa_projection.method_registry) == 16
        np.testing.assert_array_equal(
            context.search_scoring.optimization_baseline_scores,
            initial.search_scoring.optimization_baseline_scores,
        )
    assert len(full.result_pa_projection.method_registry) == 26
    assert len(exact.result_pa_projection.method_registry) == 1
    assert ef_from_superset.result_pa_projection is ef_from_superset.search_scoring.pa_projection
    assert (
        len(
            {
                initial.result_lcia_selection_identity,
                full.result_lcia_selection_identity,
                exact.result_lcia_selection_identity,
            }
        )
        == 3
    )

    exact_pb = tuple(
        json.loads(exact.result_projection.method_registry.impact_category_tuple_json.iloc[0])
    )
    reports = [first]
    for selection, refresh in ((("ef31", "pb_lcia"), False), (exact_pb, False), (("ef31",), True)):
        report = run_optimization(**arguments, lcia_method=selection, refresh=refresh)
        reports.append(report)
        manifests.append(json.loads(report.products[0].manifest_path.read_bytes()))
        assert _cache_file_state(cache) == published
    assert decode_candidate_worker_result(payloads[0].read_bytes()).identity == saved.identity
    for key in ("candidate_search", "worker_result_set"):
        assert len({manifest["parent_identities"][key] for manifest in manifests}) == 1
    assert (
        len({manifest["parent_identities"]["result_lcia_selection"] for manifest in manifests}) == 3
    )
    assert expected_roles.selected_is_exact_parent.all()
    assert expected_roles.selection_status.eq("baseline_compliant").all()
    for report in reports[1:]:
        pd.testing.assert_frame_equal(
            pd.read_parquet(report.products[0].result_directory / "selection_roles.parquet"),
            expected_roles,
            check_exact=True,
        )
    expected_categories = (16, 26, 1, 16)
    for report, count in zip(reports, expected_categories, strict=True):
        directory = report.products[0].result_directory
        annual = pd.read_parquet(directory / "selected_environmental_scores.parquet")
        stage = pd.read_parquet(directory / "selected_life_cycle_stage_scores.parquet")
        assert annual[["method_id", "impact_category_id"]].drop_duplicates().shape[0] == count
        assert stage[["method_id", "impact_category_id"]].drop_duplicates().shape[0] == count
