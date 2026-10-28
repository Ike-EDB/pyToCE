"""Full-method actual PA responses reproduce ordinary native and direct impacts."""

from dataclasses import replace
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from infiltration_fixtures import resolved_infiltration_provider
from pa_event_receiving_fixtures import mixed_pa_events as mixed_pa_events
from pa_event_receiving_fixtures import pa_event_receiving_boundary

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.premise import ACTIVE_BACKGROUND_SCENARIOS
from pytoce.prepare_backgrounds.preparation.providers.catalogue import provider_service_code
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
)
from pytoce.run_lca.calculation.burdens import BurdenDemands
from pytoce.run_lca.calculation.contribution_accumulation import (
    ContributionMethod,
    create_contribution_state,
)
from pytoce.run_lca.calculation.direct_contributions import (
    add_direct_contributions,
    admit_direct_characterization_factors,
)
from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_lca.calculation.lcia import ImpactCategory
from pytoce.run_lca.calculation.pa_event_receiving import iter_pa_receiving
from pytoce.run_lca.calculation.pa_inventory import build_pa_receiving_inventory
from pytoce.run_lca.calculation.pa_projection import (
    PaProjectedEvent,
    group_pa_receiving_quantities,
    iter_pa_projection,
)
from pytoce.run_lca.calculation.providers import PreparedProviderProduct
from pytoce.run_lca.calculation.timex import (
    PrincipalTimeline,
    assemble_principal_inventory,
    build_provider_inventory_bank,
)
from pytoce.run_optimization.calculation.pa_candidate_projection import (
    PaCandidateProjection,
    build_pa_candidate_projection,
    calculate_pa_candidate_projection,
)
from pytoce.run_optimization.calculation.response_bank import (
    project_optimization_response_bank,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    LoadedOptimizationResponseBank,
)
from pytoce.run_optimization.calculation.response_kernel import (
    _direct_factor_table,
    _member_response,
    _method_registry,
    _timex_weights,
)
from pytoce.run_optimization.calculation.results import build_selected_result_tables
from pytoce.run_optimization.runtime.result_store import (
    StagedTable,
    stage_selected_life_cycle_scores,
)
from tests.unit.run_optimization.result_table_fixtures import selected_pa_reporting_fixture


@pytest.mark.parametrize("unit_score", (-np.finfo(np.float64).max, 0.0), ids=("score", "norm"))
def test_actual_pa_quantities_reject_overflowing_characterization(
    tmp_path: Path, mixed_pa_events: BurdenDemands, unit_score: float
) -> None:
    """Reject overflow after scaling finite supplier responses by real receiving quantities.

    Native response admission is a separate boundary. Finite synthetic responses
    near the float64 limit isolate this later multiplication; the MFA event,
    receiving equations, native endpoint keys and amounts remain real.
    """
    background = "remind_ssp2_npi"
    with pa_event_receiving_boundary(tmp_path / "native", original=True) as boundary:
        reference, source = boundary.reference, boundary.sources
    providers = PreparedProviderProduct({2025: "frontier"}, {}, reference, {2025: source})
    spinning = (mixed_pa_events.pa_spinning_events[0],)
    projected = tuple(
        iter_pa_projection(
            spinning_events=spinning,
            recycling_events=(),
            disposal_events=(),
            reference=reference,
            sources_by_year={2025: source},
            matrix_by_year={2025: "frontier"},
            dump_shares={},
        )
    )
    native = {}
    direct = {}
    amounts = {}
    for event in projected:
        for group in group_pa_receiving_quantities(event.receiving.amounts):
            endpoint = group.endpoint
            if isinstance(endpoint, PaReceivingActivity):
                key = (
                    background,
                    event.source_anchor_year,
                    event.source_database,
                    event.matrix_database,
                    *endpoint.key,
                    endpoint.unit,
                )
                native[key] = (np.array([1.0]), np.array([1.0]))
                amounts.setdefault(key, []).append(group.amount)
            else:
                direct[(*endpoint.key, endpoint.unit)] = np.array([0.0])
    # A positive supplier amount greater than one overflows the finite unit response.
    target = next(key for key, values in amounts.items() if min(values) > 1.0)
    native[target] = (np.array([unit_score]), np.array([np.finfo(np.float64).max]))
    methods = _method_registry(
        (
            ImpactCategory(
                0, "ef31", "fixture", "climate_change", ("PA overflow",), "kg eq", "0" * 64
            ),
        ),
        ("ef31",),
    )
    context = PaCandidateProjection(
        "response::finite-limit-fixture",
        "receiving::fixture",
        (background,),
        methods,
        {background: providers},
        {},
        native,
        direct,
    )
    assert all(np.isfinite(values).all() for pair in native.values() for values in pair)
    with pytest.raises(
        ValueError, match="PA receiving produced a nonfinite impact or absolute contribution total"
    ) as error:
        calculate_pa_candidate_projection(
            context, spinning_events=spinning, recycling_events=(), disposal_events=()
        )
    message = str(error.value)
    assert background in message
    assert spinning[0].region in message and str(spinning[0].year) in message
    assert target[-2] in message and "climate_change" in message


def test_actual_full_method_projection_matches_ordinary_two_anchor_inventory(
    tmp_path: Path, mixed_pa_events: BurdenDemands
) -> None:
    """Compare complete source calculations at both anchors before weighting.

    The existing small MFA supplies both grades, inherited finish, three carbon
    origins and every actual receiving route. Its dates are shifted only to
    exercise interpolation inside the 2025--2050 validation interval; the cohort history remains
    fixed. Two native methods deliberately
    have different signed CF patterns and reversed registry order. The complete source calculations
    supply the responses and provider quantities.
    The method names exercise EF-only search and additional PB reporting;
    their factors are synthetic fixture values.
    """
    background = "remind_ssp2_npi"
    second_background = "remind_ssp2_pkbudg650"
    routes = resolved_infiltration_provider(tmp_path / "infiltration")
    dumping = routes.routes.loc[
        routes.routes.provider_demand_type.eq("mismanaged_open_dump_treatment")
    ]
    shares = dict(zip(dumping.infiltration_class, dumping.infiltration_share, strict=True))
    spinning = tuple(
        replace(event, year=event.year + 16) for event in mixed_pa_events.pa_spinning_events
    )
    recycling = tuple(
        replace(event, year=event.year + 16) for event in mixed_pa_events.pa_recycling_events
    )
    disposal = tuple(
        replace(event, year=event.year + 16) for event in mixed_pa_events.pa_disposal_events
    )
    years = tuple(sorted({event.year for event in (*spinning, *recycling, *disposal)}))
    methods = tuple(
        ImpactCategory(
            index, label, "fixture", category, ("PA candidate", label), unit, str(index) * 64
        )
        for index, (label, category, unit) in enumerate(
            (("pb_lcia", "fixture_category", "test eq"), ("ef31", "climate_change", "kg CO2-Eq"))
        )
    )
    registry = _method_registry(methods, ("pb_lcia", "ef31"))
    contributions = tuple(
        ContributionMethod(
            method.method_id,
            method.method_version,
            method.impact_category_id,
            method.method_tuple,
            method.unit,
        )
        for method in methods
    )
    service = "hydraulic_water_collection_row"
    services = pd.DataFrame(
        [
            {
                "provider_service_id": service,
                "provider_service_code": provider_service_code(service),
            }
        ]
    )
    sources = {}
    references = []
    native_frames = []
    service_frames = []
    second_native_frames = []
    second_service_frames = []
    expected_stages: dict[tuple[object, ...], list[tuple[float, float]]] = {}
    expected_annual: dict[tuple[int, str], list[tuple[float, float]]] = {}
    direct_frames = []
    for original in (True, False):
        root = tmp_path / ("original_graph" if original else "future_graph")
        names = BrightwayNames(root / "brightway", "pa_receiving", "biosphere", "original")
        with pa_event_receiving_boundary(root, original=original) as boundary:
            reference, source, bd = boundary.reference, boundary.sources, boundary.bd
            references.append(reference)
            sources[source.provider_anchor_year] = source
            flows = sorted(bd.Database("biosphere"), key=lambda flow: flow.key)
            for method_index, method in enumerate(methods):
                native_method = bd.Method(method.method_tuple)
                native_method.register(unit=method.unit)
                native_method.write(
                    [
                        (flow.key, ((3.0, -2.0, 0.0), (-1.0, 4.0, -0.0))[method_index][index % 3])
                        for index, flow in enumerate(flows)
                    ]
                )
            providers = PreparedProviderProduct(
                {source.provider_anchor_year: "frontier"},
                {},
                reference,
                {source.provider_anchor_year: source},
            )
            response = _member_response(names, background, methods, registry, services, providers)
            native_frames.append(response.native_unit_responses)
            service_frames.append(response.provider_unit_responses)
            # This small native graph also supplies a second admitted member.
            # Calculate its real responses under its own background coordinates;
            # no fabricated response vectors enter the reuse comparison.
            second_response = _member_response(
                names, second_background, methods, registry, services, providers
            )
            second_native_frames.append(second_response.native_unit_responses)
            second_service_frames.append(second_response.provider_unit_responses)
            direct_frames.append(
                _direct_factor_table(names, methods, registry, response.direct_flows)
            )
            # Ordinary receiving uses one source at a time. Elapsed-year weights supply the
            # comparison with native/direct inventory.
            received = iter_pa_receiving(
                spinning_events=spinning,
                recycling_events=recycling,
                disposal_events=disposal,
                reference=reference,
                sources=source,
                spinning_providers=source.spinning_providers,
                dump_shares=shares,
            )
            projected = tuple(
                PaProjectedEvent(
                    event,
                    infiltration,
                    source.provider_anchor_year,
                    source.general_background_database,
                    "frontier",
                    (2050 - event.year) / 25 if original else (event.year - 2025) / 25,
                    result,
                )
                for event, infiltration, result in received
            )
            actual = build_pa_receiving_inventory(bd, projected, {"EUR": ("foreground", "eur")})
            assert actual.activations.amount.lt(0).any()
            assert {"kilogram", "cubic meter", "kilowatt hour"} <= set(actual.records.endpoint_unit)
            assert set(actual.records.infiltration_class) - {""} == set(shares)
            state = create_contribution_state(
                bd,
                {"EUR": actual.activations},
                {"EUR": ("foreground", "eur")},
                contributions,
                years,
            )
            factors = admit_direct_characterization_factors(
                state, sorted(set(int(value) for value in actual.direct_inventory.flow_id))
            )
            add_direct_contributions(state, actual.direct_contributions)
            inventory_bank = build_provider_inventory_bank(bd, actual.activations, state)
            ordinary = assemble_principal_inventory(
                PrincipalTimeline(
                    pd.DataFrame(),
                    actual.activations,
                    0.0,
                    actual.direct_inventory[["year", "flow_id", "amount"]],
                ),
                inventory_bank,
            )
            for index, method in enumerate(methods):
                inventory_score = fsum(
                    float(amount)
                    * state.characterization_factors[method.method_tuple].get(int(flow_id), 0.0)
                    for flow_id, amount in ordinary.inventory[["flow_id", "amount"]].itertuples(
                        index=False, name=None
                    )
                )
                total = fsum(
                    value[0] for key, value in state.annual_totals.items() if key[2] == index
                )
                assert inventory_score == pytest.approx(total, rel=2e-7)
            assert factors.factors.shape[0] == len(methods)
            for (region, year, index), values in state.annual_totals.items():
                assert region == "EUR"
                expected_annual.setdefault((year, methods[index].method_id), []).append(values)
            for key, values in state.annual_stage_values.items():
                region, year, index, *stage = key
                # Selected reporting intentionally aggregates the finer source
                # flow/fate descriptors after the process norms.
                signature = registry.loc[registry.method_id.eq(methods[index].method_id)].iloc[0]
                expected = (
                    background,
                    region,
                    year,
                    *tuple(
                        signature[column]
                        for column in (
                            "method_id",
                            "method_version",
                            "impact_category_id",
                            "impact_category_tuple_json",
                            "unit",
                            "factor_identity",
                        )
                    ),
                    *stage[:6],
                )
                expected_stages.setdefault(expected, []).append(values)
    pd.testing.assert_frame_equal(direct_frames[0], direct_frames[1])
    providers = PreparedProviderProduct(
        {2025: "frontier", 2050: "frontier"}, {}, references[0], sources
    )
    service_responses = pd.concat(service_frames, ignore_index=True)
    bank = LoadedOptimizationResponseBank(
        "bank::actual_native_fixture",
        "two_methods",
        "routes::fixture",
        "services::fixture",
        registry,
        service_responses[
            ["provider_service_id", "provider_service_code", "provider_unit"]
        ].drop_duplicates(),
        service_responses,
        _timex_weights(providers, background),
        pd.concat(native_frames, ignore_index=True),
        direct_frames[0],
    )
    # Both disposable projects are closed. Only completed portable records and
    # native response tables participate in the actual candidate calculation.
    projection = project_optimization_response_bank(bank, registry, role="result")
    context = build_pa_candidate_projection(
        projection,
        {background: providers},
        shares,
        prepared_receiving_identity="prepared-receiving::fixture",
    )
    result = calculate_pa_candidate_projection(
        context,
        spinning_events=spinning,
        recycling_events=recycling,
        disposal_events=disposal,
    )
    assert context.backgrounds == (background,)
    assert context.method_registry.method_id.tolist() == ["ef31", "pb_lcia"]
    assert result.annual_scores.shape == (1, 12, 82, 2)
    for (year, label), values in expected_annual.items():
        coordinate = (0, H12_REGIONS.index("EUR"), year - 2019, ["ef31", "pb_lcia"].index(label))
        assert result.annual_scores[coordinate] == pytest.approx(
            fsum(v[0] for v in values), rel=2e-7
        )
        assert result.annual_contribution_norms[coordinate] == pytest.approx(
            fsum(v[1] for v in values), rel=2e-7
        )
    keys = list(result.stage_scores.columns[:-2])
    expected = pd.DataFrame.from_records(
        [
            (*key, fsum(v[0] for v in values), fsum(v[1] for v in values))
            for key, values in expected_stages.items()
        ],
        columns=result.stage_scores.columns,
    )
    pd.testing.assert_frame_equal(
        result.stage_scores.sort_values(keys, ignore_index=True),
        expected.sort_values(keys, ignore_index=True),
        check_dtype=False,
        rtol=2e-7,
    )
    assert np.all(result.annual_contribution_norms >= np.abs(result.annual_scores))
    assert result.stage_scores.score.sum() == pytest.approx(result.annual_scores.sum())
    assert result.stage_scores.contribution_norm.sum() == pytest.approx(
        result.annual_contribution_norms.sum()
    )
    assert not result.annual_scores.flags.writeable
    empty = calculate_pa_candidate_projection(
        context, spinning_events=(), recycling_events=(), disposal_events=()
    )
    assert not empty.annual_scores.any() and not empty.annual_contribution_norms.any()
    assert empty.stage_scores.empty
    assert empty.stage_scores.columns.equals(result.stage_scores.columns)

    # Exercise repeated original-only fates across both member and anchor axes,
    # while a dated landfill still receives its distinct operations. Dates are
    # projection fixtures only, as above. Duplicate event occurrences remain
    # separate contributions; opposite signed zeros remain exact input values.
    burning = next(
        event for event in disposal if event.demand_type == "mismanaged_open_burning_treatment"
    )
    dump = next(
        event for event in disposal if event.demand_type == "mismanaged_open_dump_treatment"
    )
    landfill = next(event for event in disposal if event.demand_type == "landfill_treatment")
    principal = fsum(burning.principal_by_origin_kg)
    interleaved = (
        replace(dump, year=2050),
        replace(burning, year=2025),
        replace(landfill, year=2035),
        replace(dump, year=2035),
        replace(dump, year=2035),
        replace(burning, year=2035, principal_by_origin_kg=(principal, 0.0, -0.0)),
        replace(burning, year=2035, principal_by_origin_kg=(principal, -0.0, 0.0)),
    )
    multi_bank = replace(
        bank,
        provider_unit_responses=pd.concat(
            [bank.provider_unit_responses, *second_service_frames], ignore_index=True
        ),
        native_unit_responses=pd.concat(
            [bank.native_unit_responses, *second_native_frames], ignore_index=True
        ),
        timex_year_weights=pd.concat(
            [bank.timex_year_weights, _timex_weights(providers, second_background)],
            ignore_index=True,
        ),
    )
    multi_projection = project_optimization_response_bank(multi_bank, registry, role="result")
    multi_context = build_pa_candidate_projection(
        multi_projection,
        {background: providers, second_background: providers},
        shares,
        prepared_receiving_identity="prepared-receiving::fixture",
    )
    multi_result = calculate_pa_candidate_projection(
        multi_context, spinning_events=(), recycling_events=(), disposal_events=interleaved
    )
    for member_index, member in enumerate((background, second_background)):
        independent_context = build_pa_candidate_projection(
            multi_projection,
            {member: providers},
            shares,
            prepared_receiving_identity="prepared-receiving::fixture",
        )
        independent = calculate_pa_candidate_projection(
            independent_context,
            spinning_events=(),
            recycling_events=(),
            disposal_events=interleaved,
        )
        np.testing.assert_array_equal(
            multi_result.annual_scores[[member_index]], independent.annual_scores
        )
        np.testing.assert_array_equal(
            multi_result.annual_contribution_norms[[member_index]],
            independent.annual_contribution_norms,
        )
        pd.testing.assert_frame_equal(
            multi_result.stage_scores.loc[
                multi_result.stage_scores.characterization_background_scenario.eq(member)
            ].reset_index(drop=True),
            independent.stage_scores,
            check_exact=True,
        )
    reordered = calculate_pa_candidate_projection(
        multi_context,
        spinning_events=(),
        recycling_events=(),
        disposal_events=tuple(reversed(interleaved)),
    )
    np.testing.assert_array_equal(reordered.annual_scores, multi_result.annual_scores)
    np.testing.assert_array_equal(
        reordered.annual_contribution_norms, multi_result.annual_contribution_norms
    )
    pd.testing.assert_frame_equal(
        reordered.stage_scores, multi_result.stage_scores, check_exact=True
    )
    # Stored activities support later reporting selections. Each subset retains signed scores and
    # supplier contribution norms from the common inventory.
    for index in range(len(registry)):
        selected_registry = registry.iloc[[index]].reset_index(drop=True)
        selected_projection = project_optimization_response_bank(
            bank, selected_registry, role="result"
        )
        selected_context = build_pa_candidate_projection(
            selected_projection,
            {background: providers},
            shares,
            prepared_receiving_identity="prepared-receiving::fixture",
        )
        selected = calculate_pa_candidate_projection(
            selected_context,
            spinning_events=spinning,
            recycling_events=recycling,
            disposal_events=disposal,
        )
        np.testing.assert_array_equal(selected.annual_scores, result.annual_scores[..., [index]])
        np.testing.assert_array_equal(
            selected.annual_contribution_norms, result.annual_contribution_norms[..., [index]]
        )
        expected_stage = result.stage_scores.loc[
            result.stage_scores.method_id.eq(selected_registry.method_id.iloc[0])
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected.stage_scores, expected_stage, check_exact=True)

    # Selected reporting must characterize the saved, changed activities
    # for its extra method. Its cached search scores have only the EF column.
    products, reporting_context, parents = selected_pa_reporting_fixture(
        bank, providers, shares, spinning=spinning, recycling=recycling, disposal=disposal
    )
    worker = products.selected_2050[0].selected_result.result
    assert (
        worker.physical.physical_state_identity
        != reporting_context.physical_parent.physical_state_identity
    )
    assert worker.physical.pa_spinning_events and worker.physical.pa_recycling_events
    assert worker.physical.pa_disposal_events
    assert worker.pa_scores is not None and worker.pa_scores.annual_scores.shape[-1] == 1
    assert set(worker.pa_scores.stage_scores.method_id) == {"ef31"}
    assert reporting_context.result_pa_projection.method_registry.method_id.tolist() == [
        "ef31",
        "pb_lcia",
    ]
    directory = tmp_path / "selected_reporting"
    with stage_selected_life_cycle_scores(directory) as writer:
        tables = build_selected_result_tables(
            products,
            request_context=reporting_context,
            baseline_lca_results=parents,
            stage_writer=writer,
            stage_progress=lambda completed, total: None,
        )
        stage_table = tables.tables["selected_life_cycle_stage_scores.parquet"]
        assert isinstance(stage_table, StagedTable)
        stage_scores = pd.read_parquet(
            stage_table.staging_directory / "selected_life_cycle_stage_scores.parquet"
        )
    environmental = tables.tables["selected_environmental_scores.parquet"]
    assert isinstance(environmental, pd.DataFrame)
    assert set(environmental.target_year) == {2050}
    assert environmental.physical_candidate_identity.eq(
        worker.physical.physical_candidate_identity
    ).all()
    assert environmental.loc[environmental.method_id.eq("pb_lcia"), "score"].ne(0.0).any()

    # Expected values follow the ordinary native inventory above.
    members = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)
    signatures = registry.drop(columns=["lcia_method", "selection_order"]).to_dict("records")
    expected_environmental = pd.DataFrame.from_records(
        {
            "background_scenario": member,
            "region": region,
            "year": year,
            **signature,
            "score": fsum(v[0] for v in expected_annual.get((year, signature["method_id"]), ()))
            if region in ("EUR", "global")
            else 0.0,
            "contribution_norm": fsum(
                v[1] for v in expected_annual.get((year, signature["method_id"]), ())
            )
            if region in ("EUR", "global")
            else 0.0,
        }
        for member in members
        for region in (*H12_REGIONS, "global")
        for year in range(2019, 2101)
        for signature in signatures
    )
    annual_keys = [
        column for column in expected_environmental if column not in ("score", "contribution_norm")
    ]
    pd.testing.assert_frame_equal(
        environmental[expected_environmental.columns].sort_values(annual_keys, ignore_index=True),
        expected_environmental.sort_values(annual_keys, ignore_index=True),
        check_dtype=False,
        rtol=2e-7,
    )
    expected_stage = pd.concat(
        [
            expected.assign(characterization_background_scenario=member, region=region)
            for member in members
            for region in ("EUR", "global_derived")
        ],
        ignore_index=True,
    )
    stage_keys = list(expected_stage.columns[:-2])
    annual_stage = stage_scores.loc[stage_scores.period_kind.eq("annual"), expected_stage.columns]
    pd.testing.assert_frame_equal(
        annual_stage.sort_values(stage_keys, ignore_index=True),
        expected_stage.sort_values(stage_keys, ignore_index=True),
        check_dtype=False,
        rtol=2e-7,
    )
    cumulative_keys = [key for key in stage_keys if key not in ("year", "physical_event_year")]
    values = ["score", "contribution_norm"]
    expected_cumulative = (
        expected_stage.groupby(cumulative_keys, observed=True, sort=True)[values]
        .agg(fsum)
        .reset_index()
    )
    cumulative_stage = stage_scores.loc[stage_scores.period_kind.eq("cumulative_2019_2100")]
    pd.testing.assert_frame_equal(
        cumulative_stage[[*cumulative_keys, *values]].sort_values(
            cumulative_keys, ignore_index=True
        ),
        expected_cumulative.sort_values(cumulative_keys, ignore_index=True),
        check_dtype=False,
        rtol=2e-7,
    )
    assert cumulative_stage[["year", "physical_event_year"]].isna().all().all()
