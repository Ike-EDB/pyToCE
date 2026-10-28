"""One late PA cohort with complete prepared sources for real worker boundaries."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import cast

import numpy as np
import pandas as pd
import yaml
from infiltration_fixtures import resolved_infiltration_provider
from material_carbon_fixtures import prepared_material_carbon
from mfa_workspace_fixtures import prepare_mfa_inputs
from pa_event_receiving_fixtures import pa_event_receiving_boundary

from pytoce import run_mfa
from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingFlow,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
    admit_pa_receiving_sources,
)
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_lca.calculation.pa_receiving_endpoints import pa_receiving_endpoints
from pytoce.run_lca.calculation.providers import (
    load_prepared_provider_product,
    load_provider_routes,
)
from pytoce.run_mfa.model.result_inputs import (
    MFAResultReference,
    read_mfa_result,
    read_mfa_result_manifest,
)
from pytoce.run_mfa.remind_mfa.execution import (
    RemindMFAExecutionSession,
    prepare_execution,
    prepare_workspace_source,
)
from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestContext,
    prepare_candidate_worker_request_context,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    _ANCHOR_YEARS,
    _BACKGROUNDS,
    _DIRECT_COLUMNS,
    _METHOD_SIGNATURE_COLUMNS,
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
from pytoce.run_optimization.calculation.worker_inputs import (
    prepare_candidate_worker_initializer,
)
from pytoce.run_optimization.input.allocated_cc_product import compile_allocated_cc_result
from tests.integration.brightway_lca_fixtures import write_prepared_inputs
from tests.integration.candidate_mfa_fixtures import _response_rows
from tests.unit.run_optimization.allocated_cc_fixtures import _methods, _write_raw_result
from tests.unit.run_optimization.response_bank_fixtures import _weights

PA_WORKER_FOREGROUND = "pa_last_cohort_worker"


def _declare_last_pa_cohort(workspace: Path) -> None:
    """Declare source-matched 2100 inputs, changing only the incoming EUR textile mix."""
    # An authored cohort rule activates the full cohort continuation. Its
    # stock targets and material partition must therefore match this invented
    # CSV source, rather than the unrelated packaged default scenario values.
    state = RemindMFAExecutionSession(
        prepare_execution(prepare_workspace_source(workspace))
    ).cohort_state
    target = state.annual_stock[state.years.index(2100)]
    partition = state.material_shares.copy()
    region_i, good_i = state.regions.index("EUR"), state.goods.index("Textile sector")
    partition[region_i, :, good_i] = 0.0
    partition[region_i, state.materials.index("Fibres"), good_i] = 1.0
    scenario_path = (
        workspace / "data_raw/foreground_scenarios/scenarios" / f"{PA_WORKER_FOREGROUND}.yaml"
    )
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    # A single step preserves history through 2099. Stock targets use source Mt.
    scenario["changes"].extend(
        {
            "parameter": "stock_projection",
            "region": region,
            "good": good,
            "changes": [
                {
                    "year": 2100,
                    "mode": "absolute",
                    "interpolation": "step",
                    "value": float(target[r, g]),
                }
            ],
        }
        for r, region in enumerate(state.regions)
        for g, good in enumerate(state.goods)
    )
    scenario["changes"].extend(
        {
            "parameter": "plastic_family_share",
            "region": region,
            "good": good,
            "material": material,
            "changes": [
                {
                    "year": 2100,
                    "mode": "absolute",
                    "interpolation": "step",
                    "value": float(partition[r, m, g]),
                }
            ],
        }
        for r, region in enumerate(state.regions)
        for m, material in enumerate(state.materials)
        for g, good in enumerate(state.goods)
    )
    scenario_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    receipt_path = workspace / "fixture_inputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["scenario"] = scenario
    receipt["cohort_source"] = {
        "owner": "RemindMFAExecutionSession.cohort_state",
        "stock_unit": "Mt Plastic",
        "stock_year": 2100,
        "stock_targets": target.tolist(),
        "regions": state.regions,
        "goods": state.goods,
        "materials": state.materials,
        "original_material_partition": state.material_shares.tolist(),
        "lifetime_mean_years": state.lifetime_mean.tolist(),
        "lifetime_std_years": state.lifetime_std.tolist(),
        "changed_partition": {
            "region": "EUR",
            "good": "Textile sector",
            "year": 2100,
            "material": "Fibres",
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    # Exercise the actual input owner before the full MFA. These assertions
    # check coordinate and unit handoff, with no modification of compiled data.
    compiled = compile_foreground_scenario(workspace, PA_WORKER_FOREGROUND)
    table = pd.read_parquet(compiled.compiled_parameters_path)
    cohort = table.loc[table.parameter_id.isin({"stock_projection", "plastic_family_share"})]
    authored = cohort.loc[cohort.value_origin.eq("user_foreground_scenario")]
    assert set(authored.year) == {2100}
    stock = table.loc[table.year.eq(2100) & table.parameter_id.eq("stock_projection")]
    assert set(stock.unit) == {"Mt Plastic"}
    np.testing.assert_array_equal(
        stock.set_index(["region", "good"])
        .resolved_value.reindex(pd.MultiIndex.from_product([state.regions, state.goods]))
        .to_numpy()
        .reshape(target.shape),
        target,
    )
    shares = table.loc[table.year.eq(2100) & table.parameter_id.eq("plastic_family_share")]
    np.testing.assert_array_equal(
        shares.set_index(["region", "material", "good"])
        .resolved_value.reindex(
            pd.MultiIndex.from_product([state.regions, state.materials, state.goods])
        )
        .to_numpy()
        .reshape(partition.shape),
        partition,
    )
    for parameter, expected in (
        ("lifetime_mean", state.lifetime_mean),
        ("lifetime_std", state.lifetime_std),
    ):
        rows = table.loc[table.year.eq(2100) & table.parameter_id.eq(parameter)]
        np.testing.assert_array_equal(
            rows.set_index("good").resolved_value.reindex(state.goods).to_numpy(), expected
        )


def _admitted_profiles(
    root: Path,
) -> tuple[
    BrightwayNames,
    PaReceivingReference,
    PaReceivingSources,
    PaReceivingSources,
    dict[tuple[str, str], tuple[str, str]],
]:
    """Retain real original/future rows and product identities from one complete graph."""
    names = BrightwayNames(root / "brightway", "pa_receiving", "biosphere", "original")
    with pa_event_receiving_boundary(root, original=False) as boundary:
        original = admit_pa_receiving_sources(
            replace(
                boundary.original_context,
                provider_database="frontier",
                technology=boundary.context.technology,
            ),
            boundary.reference,
        )
        # The single completed technological frontier is a declared constant
        # fixture profile. Original general operations remain original rows;
        # all future general operations use the actual future-shaped rows.
        products: dict[tuple[str, str], tuple[str, str]] = {}
        for source in (original, boundary.sources):
            native, _ = pa_receiving_endpoints(boundary.reference, source)
            for endpoint in native:
                node = boundary.bd.get_node(database=endpoint.key[0], code=endpoint.key[1])
                products[endpoint.key] = tuple(node.rp_exchange().input.key)
        return names, boundary.reference, original, boundary.sources, products


def prepare_pa_worker_workspace(
    workspace: Path, source_root: Path
) -> tuple[bytes, CandidateWorkerRequestContext]:
    """Publish one genuine late-cohort parent and complete synthetic response inputs.

    All 52 source coordinates use explicit constant original/future fixture
    profiles. Native response coefficients represent two invented characterized
    contributions, +3 and -1 per actual product unit: score 2 and norm 4. They
    test transport and contraction, not another native LCA solve. The existing
    actual receiving owners calculate every quantity before scoring.
    """
    started = monotonic()
    prepare_mfa_inputs(workspace, source_root, foreground_scenario=PA_WORKER_FOREGROUND)
    _declare_last_pa_cohort(workspace)
    print(
        f"PA worker fixture: source-matched 2100 controls compiled ({monotonic() - started:.2f}s)",
        flush=True,
    )
    report = run_mfa(foreground_scenario=PA_WORKER_FOREGROUND)
    mfa_reference = read_mfa_result_manifest(report.results[0].manifest_path)
    print(
        f"PA worker fixture: public MFA parent complete ({monotonic() - started:.2f}s)", flush=True
    )

    return prepare_pa_worker_context(workspace, mfa_reference)


def prepare_pa_worker_context(
    workspace: Path, mfa_reference: MFAResultReference
) -> tuple[bytes, CandidateWorkerRequestContext]:
    """Attach complete source and response products to one admitted parent material balance."""
    started = monotonic()
    parent = read_mfa_result(mfa_reference)
    # Prepared products record the actual store relative to their workspace.
    # The complete fixture graph therefore belongs under data_processed.
    names, reference, original, future, products = _admitted_profiles(workspace / "data_processed")
    print(
        f"PA worker fixture: native source admission complete ({monotonic() - started:.2f}s)",
        flush=True,
    )

    source_maps = {
        background: {
            year: replace(
                original if year <= 2025 else future,
                background_scenario=background,
                provider_anchor_year=year,
                general_background_anchor_year=2025 if year <= 2025 else year,
            )
            for year in _ANCHOR_YEARS
        }
        for background in _BACKGROUNDS
    }
    provider = resolved_infiltration_provider(workspace / "native_infiltration")
    carbon = prepared_material_carbon(workspace / "native_carbon")
    methods = _methods()
    methods["factor_identity"] = methods["factor_identity"].map(
        lambda value: sha256(str(value).encode()).hexdigest()
    )
    methods = method_registry_from_records(
        cast(list[dict[str, object]], methods.to_dict("records"))
    )
    shared, members = write_prepared_inputs(
        workspace,
        names,
        methods=methods,
        provider=provider,
        carbon_rows=carbon,
        backgrounds=_BACKGROUNDS,
        receiving_reference=reference,
        receiving_sources_by_background=source_maps,
        technology_databases_by_background={
            background: {year: future.general_background_database for year in _ANCHOR_YEARS}
            for background in _BACKGROUNDS
        },
    )
    routes = load_provider_routes(shared)
    services = routes.routes.loc[
        :, ["provider_service_id", "provider_service_code"]
    ].drop_duplicates(ignore_index=True)
    services["provider_unit"] = "kilogram"
    signatures = cast(
        list[dict[str, object]], methods.loc[:, list(_METHOD_SIGNATURE_COLUMNS)].to_dict("records")
    )
    native_rows = []
    flows: dict[tuple[str, str], PaReceivingFlow] = {}
    for background, member in members.items():
        prepared = load_prepared_provider_product(shared, member)
        for year, source in prepared.pa_receiving_sources_by_year.items():
            native, direct = pa_receiving_endpoints(reference, source)
            flows.update((flow.key, flow) for flow in direct)
            for endpoint in native:
                product = products[endpoint.key]
                native_rows.extend(
                    {
                        "background_scenario": background,
                        "source_anchor_year": year,
                        "source_database": source.general_background_database,
                        "matrix_database": prepared.database_by_year[year],
                        "endpoint_database": endpoint.key[0],
                        "endpoint_code": endpoint.key[1],
                        "product_database": product[0],
                        "product_code": product[1],
                        "endpoint_unit": endpoint.unit,
                        **method,
                        "unit_score": 2.0,
                        "unit_contribution_norm": 4.0,
                    }
                    for method in signatures
                )
    direct_rows = [
        {
            "flow_database": flow.key[0],
            "flow_code": flow.key[1],
            "flow_name": flow.name,
            "flow_categories_json": json.dumps(list(flow.categories), separators=(",", ":")),
            "endpoint_unit": flow.unit,
            **method,
            "factor": 0.5,
            "present": True,
        }
        for _, flow in sorted(flows.items())
        for method in signatures
    ]
    inputs = ResponseBankInputs(
        routes.identity,
        cast(str, shared.payload["identity"]),
        shared.sha256,
        "explicit_constant_fixture_native_3_minus_1_and_direct_half",
        tuple(
            ResponseBankMemberParent(
                background,
                cast(str, members[background].payload["identity"]),
                members[background].sha256,
            )
            for background in _BACKGROUNDS
        ),
    )
    owner = _resolved_owner(workspace)
    with _owner_lock(owner):
        write_optimization_response_bank(
            owner,
            response_method_union="ef31_response_v3",
            method_union_registry=methods,
            mandatory_climate_method_registry=methods.loc[
                methods.impact_category_id.eq("climate_change")
            ],
            provider_service_registry=services.astype("string"),
            provider_unit_responses=_response_rows(services, methods),
            native_unit_responses=pd.DataFrame(native_rows, columns=_NATIVE_COLUMNS),
            direct_characterization_factors=pd.DataFrame(direct_rows, columns=_DIRECT_COLUMNS),
            timex_year_weights=_weights(),
            source_inputs=inputs,
        )
    _write_raw_result(workspace, "worker_result", cases=("direct_case",))
    allocated = compile_allocated_cc_result(workspace, "worker_result")
    initializer = prepare_candidate_worker_initializer(
        workspace,
        foreground_scenario=PA_WORKER_FOREGROUND,
        mfa_reference=mfa_reference,
        allocated_cc=allocated,
        allocated_cc_case=("direct_case",),
        response_method_union="ef31_response_v3",
    )
    context = prepare_candidate_worker_request_context(
        initializer,
        allocated,
        parent_mfa=parent,
        result_lcia_method=normalized_lcia_method(("ef31",)),
    )
    physical = context.physical_parent
    print(
        "PA worker fixture: actual parent events "
        f"spinning={len(physical.pa_spinning_events)}, "
        f"recycling={len(physical.pa_recycling_events)}, "
        f"disposal={len(physical.pa_disposal_events)} "
        f"({monotonic() - started:.2f}s)",
        flush=True,
    )
    return context.worker_initializer_payload, context
