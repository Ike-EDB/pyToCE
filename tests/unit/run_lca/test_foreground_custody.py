"""Cohort weighting and real Brightway foreground persistence from MFA."""

from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from infiltration_fixtures import resolved_infiltration_provider
from material_carbon_fixtures import prepared_material_carbon
from material_recipe_fixtures import material_source_members

from pytoce.download_data.sources.ecoinvent_materials import derive_material_recipe
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    provider_service_code,
)
from pytoce.run_lca.calculation.custody import (
    CarbonCustodyHistory,
    prepare_carbon_cohort_weights,
)
from pytoce.run_lca.calculation.foreground import (
    H12_REGIONS,
    OPTIMIZATION_ACTIVATION_COLUMNS,
    build_foreground_database,
    compile_foreground_activations,
    optimization_activation_context_identity,
)
from pytoce.run_lca.calculation.providers import load_provider_routes
from tests.integration.brightway_lca_fixtures import write_prepared_inputs


def test_empty_optimization_context_does_not_reuse_the_previous_custody_method() -> None:
    """Bind the custody method even when zero demands leave all numerical inputs unchanged."""
    activations = pd.DataFrame(columns=OPTIMIZATION_ACTIVATION_COLUMNS)
    identity = optimization_activation_context_identity(
        activations, CarbonCustodyHistory(2019, (), (), ()), (), (), ()
    )
    # The empty context gets its own comparison identifier.
    assert identity != (
        "pytoce_optimization_activation_context_v1::"
        "421e4e9f28f9c9993e7cdeabe824cab4dd4dc3a78ed6438346b363e41055e585"
    )


def test_prepared_cohort_weights_retain_exact_cohort_order_and_values() -> None:
    cohort_state = pd.DataFrame(
        [
            {
                "event_year": 2021,
                "cohort_year": cohort_year,
                "region": "EUR",
                "good": "Good",
                "stock_quantity": "outflow",
                "amount": amount,
            }
            for cohort_year, amount in ((2019, 1.0), (2020, 3.0), (2021, 2.0))
        ]
    )
    cohort_material_identity = pd.DataFrame(
        [
            {
                "cohort_year": cohort_year,
                "region": "EUR",
                "material": material,
                "good": "Good",
                "material_share": share,
            }
            for cohort_year, shares in (
                (2019, {"A": 0.25, "B": 0.75}),
                (2020, {"A": 0.50, "B": 0.50}),
                (2021, {"A": 0.75, "B": 0.25}),
            )
            for material, share in shares.items()
        ]
    )

    observed = prepare_carbon_cohort_weights(
        cohort_state,
        cohort_material_identity,
    )

    assert tuple(observed) == (2021,)
    assert tuple(observed[2021]) == (("EUR", "A", "Good"), ("EUR", "B", "Good"))
    a_weights = observed[2021][("EUR", "A", "Good")]
    np.testing.assert_array_equal(a_weights.prior_years, np.asarray([2019, 2020]))
    np.testing.assert_array_equal(
        a_weights.prior_weights,
        np.asarray([0.25 / 3.25, 1.50 / 3.25]),
    )
    assert a_weights.current_weight == 1.50 / 3.25
    b_weights = observed[2021][("EUR", "B", "Good")]
    np.testing.assert_array_equal(b_weights.prior_years, np.asarray([2019, 2020]))
    np.testing.assert_array_equal(
        b_weights.prior_weights,
        np.asarray([0.75 / 2.75, 1.50 / 2.75]),
    )
    assert b_weights.current_weight == 0.50 / 2.75


def test_foreground_product_is_written_once_and_then_reused(tmp_path: Path) -> None:
    """Store twelve regional foreground activities and group their annual service demands.

    Keep constituent quantities when shared service demands are combined."""
    carbon_rows = prepared_material_carbon(tmp_path / "native_carbon")
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    state, inputs = empty_model("Others")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    state.annual_inflow[1, region, good] = 1
    state.stock_by_cohort[1:, 1, region, good] = 1
    state.annual_inflow[2, region, good] = 2
    state.stock_by_cohort[2:, 2, region, good] = 2
    mfa = evaluate_mfa(state, inputs, recipe)

    authority = resolved_infiltration_provider(tmp_path / "native_infiltration")
    names = BrightwayNames(
        store=tmp_path / "data_processed/brightway",
        project="foreground_product",
        biosphere_database="biosphere",
        source_database="source",
    )
    shared, _ = write_prepared_inputs(
        tmp_path,
        names,
        methods=pd.DataFrame(),
        provider=authority,
        carbon_rows=carbon_rows,
        backgrounds=(),
    )
    routes = load_provider_routes(shared)
    expected = compile_foreground_activations(mfa, routes, recipe).activations
    provider_data = {
        (routes.interface_database, code): {
            "name": code,
            "reference product": code,
            "unit": "kilogram",
            "location": "GLO",
            "type": "process",
            "exchanges": [
                {"input": (routes.interface_database, code), "amount": 1.0, "type": "production"}
            ],
        }
        for code in sorted(expected.provider_service_code.unique())
    }
    with selected_brightway_project(names) as bd:
        bd.geomapping.add(H12_REGIONS)
        bd.Database(routes.interface_database).write(provider_data, searchable=False)

    computed = build_foreground_database(
        "constituent_fixture",
        names=names,
        routes=routes,
        mfa_result_identity="invented_mfa",
        mfa=mfa,
        recipe=recipe,
    )
    with selected_brightway_project(names) as bd:
        nodes = [bd.get_node(database=db, code=code) for db, code in computed.root_keys]
        original_ids = tuple(node.id for node in nodes)
        assert {node["region"] for node in nodes} == set(H12_REGIONS)
        assert len(nodes) == len(bd.Database(computed.database_name)) == 12
        for node in nodes:
            expected_region = expected.loc[expected.region.eq(node["region"])]
            by_service = {
                code: annual.groupby("year").amount.agg(fsum)
                for code, annual in expected_region.groupby("provider_service_code")
            }
            exchanges = list(node.technosphere())
            assert len(exchanges) == len(by_service)
            for exchange in exchanges:
                assert exchange["amount"] == 1.0
                annual = by_service[exchange.input["code"]]
                distribution = exchange["temporal_distribution"]
                np.testing.assert_array_equal(
                    distribution.date,
                    np.array([f"{year}-01-01" for year in annual.index], dtype="datetime64[s]"),
                )
                np.testing.assert_array_equal(distribution.amount, annual.to_numpy())
        conversion = expected.loc[expected.demand_type.eq("processing_conversion")]
        assert conversion.groupby("year").route_component_id.nunique().to_dict() == {
            2019: 10,
            2020: 10,
        }
        assert set(conversion.provider_service_code) == {
            provider_service_code(service)
            for service in (
                "pur_conversion_included_in_finished_products",
                "others_epoxy_coating_conversion",
                "others_esbr_compounding_conversion",
                "conversion_injection",
                "others_road_marking_conversion",
                "others_upr_hand_lay_up_conversion",
            )
        }
        assert conversion.groupby("year").amount.sum().to_dict() == pytest.approx(
            {2019: 1e9, 2020: 2e9}
        )

    reused = build_foreground_database(
        "constituent_fixture",
        names=names,
        routes=routes,
        mfa_result_identity="invented_mfa",
        mfa=mfa,
        recipe=recipe,
    )
    assert computed.root_keys == reused.root_keys
    assert computed.foreground_identity == reused.foreground_identity
    pd.testing.assert_frame_equal(computed.activations, expected)
    pd.testing.assert_frame_equal(computed.activations, reused.activations)
    pd.testing.assert_frame_equal(
        computed.mfa_to_lca_flow_coverage, reused.mfa_to_lca_flow_coverage
    )
    with selected_brightway_project(names) as bd:
        assert tuple(bd.get_node(database=db, code=code).id for db, code in reused.root_keys) == (
            original_ids
        )

    second_name = "constituent_fixture_"
    second = build_foreground_database(
        second_name,
        names=names,
        routes=routes,
        mfa_result_identity="second_invented_mfa",
        mfa=evaluate_mfa(state, inputs, recipe, foreground_scenario=second_name),
        recipe=recipe,
    )
    assert second.database_name != computed.database_name
    with selected_brightway_project(names) as bd:
        assert set(bd.databases) == {
            routes.interface_database,
            computed.database_name,
            second.database_name,
        }
        for database, code in second.root_keys:
            node = bd.get_node(database=database, code=code)
            assert node["foreground_scenario"] == second_name
            assert node["mfa_result_identity"] == "second_invented_mfa"
        assert tuple(bd.get_node(database=db, code=code).id for db, code in reused.root_keys) == (
            original_ids
        )

    # A corrected MFA under the same readable scenario must coexist with the
    # original foreground; its new roots cannot overwrite the old scientific rows.
    corrected = build_foreground_database(
        "constituent_fixture",
        names=names,
        routes=routes,
        mfa_result_identity="corrected_invented_mfa",
        mfa=mfa,
        recipe=recipe,
    )
    assert corrected.database_name not in {computed.database_name, second.database_name}
    with selected_brightway_project(names) as bd:
        assert set(bd.databases) == {
            routes.interface_database,
            computed.database_name,
            second.database_name,
            corrected.database_name,
        }
        assert tuple(bd.get_node(database=db, code=code).id for db, code in reused.root_keys) == (
            original_ids
        )
