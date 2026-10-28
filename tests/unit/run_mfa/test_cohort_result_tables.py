"""Factorized reporting of actual coherent cohort states."""

import numpy as np
import pandas as pd

from pytoce.run_mfa.model.cohorts import (
    accepted_default_cohort_result,
)
from pytoce.run_mfa.model.result_tables import (
    _cohort_composition_table,
    _cohort_material_identity_table,
    _cohort_state_table,
    _optimization_cohort_material_identity_table,
    _optimization_cohort_state_table,
)
from pytoce.run_mfa.remind_mfa.translation import NormalizedCohortState


def test_factorized_cohort_tables_preserve_exact_material_amounts() -> None:
    """Reconstruct polymer amounts from stock/outflow and cohort material shares."""
    stock = np.array(
        [
            [[[4.0]], [[0.0]]],
            [[[3.0]], [[2.0]]],
        ]
    )
    outflow = np.array(
        [
            [[[0.0]], [[0.0]]],
            [[[1.0]], [[0.5]]],
        ]
    )
    cohort = accepted_default_cohort_result(
        NormalizedCohortState(
            years=(2019, 2020),
            interval_years=np.ones(2),
            regions=("EUR",),
            materials=("HDPE", "BioPlastics"),
            goods=("Transportation",),
            elements=("C", "Other Elements"),
            stock_by_cohort=stock,
            outflow_by_cohort=outflow,
            annual_stock=stock.sum(axis=1),
            annual_inflow=np.array([4.0, 2.5])[:, None, None],
            annual_outflow=outflow.sum(axis=1),
            material_shares=np.array([[[0.75], [0.25]]]),
            carbon_content=np.array([[0.8, 0.6], [0.2, 0.4]]),
            lifetime_mean=np.array([10.0]),
            lifetime_std=np.array([2.0]),
        )
    )

    state = _cohort_state_table(cohort, foreground_scenario="fixture")
    assert set(state.loc[state["cohort_year"].eq(2019), "cohort_origin_status"]) == {
        "model_boundary_initial_2019_cohort_not_pre2019_production_year"
    }
    assert set(state.loc[state["cohort_year"].eq(2020), "cohort_origin_status"]) == {
        "model_inflow_cohort"
    }
    identity = _cohort_material_identity_table(
        cohort,
        foreground_scenario="fixture",
    )
    reconstructed = state.merge(
        identity,
        on=["foreground_scenario", "cohort_year", "region", "good"],
        validate="many_to_many",
    )
    reconstructed["material_amount"] = reconstructed["amount"] * reconstructed["material_share"]

    coordinates = ["event_year", "cohort_year", "stock_quantity", "material"]
    expected = pd.DataFrame.from_records(
        [
            (2019, 2019, "stock", "HDPE", 3.0),
            (2019, 2019, "stock", "BioPlastics", 1.0),
            (2020, 2019, "stock", "HDPE", 2.25),
            (2020, 2019, "stock", "BioPlastics", 0.75),
            (2020, 2020, "stock", "HDPE", 1.5),
            (2020, 2020, "stock", "BioPlastics", 0.5),
            (2020, 2019, "outflow", "HDPE", 0.75),
            (2020, 2019, "outflow", "BioPlastics", 0.25),
            (2020, 2020, "outflow", "HDPE", 0.375),
            (2020, 2020, "outflow", "BioPlastics", 0.125),
        ],
        columns=[*coordinates, "material_amount"],
    )
    for name in ("event_year", "cohort_year"):
        expected[name] = np.asarray(expected[name], dtype=int)
    pd.testing.assert_frame_equal(
        reconstructed[[*coordinates, "material_amount"]]
        .sort_values(coordinates)
        .reset_index(drop=True),
        expected.sort_values(coordinates).reset_index(drop=True),
        check_exact=True,
    )
    expected_composition = pd.DataFrame(
        {
            "element": ["C", "C", "Other Elements", "Other Elements"],
            "material": ["HDPE", "BioPlastics", "HDPE", "BioPlastics"],
            "content_fraction": [0.8, 0.6, 0.2, 0.4],
            "unit": "fraction",
            "parameter_origin": "remind_mfa_input",
            "computation_origin": "nominal_remind_element_partition",
        }
    )
    pd.testing.assert_frame_equal(
        _cohort_composition_table(cohort), expected_composition, check_exact=True
    )
    assert set(
        identity.loc[
            identity["cohort_year"].eq(2019),
            "material_identity_status",
        ]
    ) == {"frozen_historical_material_identity"}
    assert set(
        identity.loc[
            identity["cohort_year"].eq(2020),
            "material_identity_status",
        ]
    ) == {"accepted_default_material_identity"}

    candidate_state = _optimization_cohort_state_table(cohort)
    candidate_identity = _optimization_cohort_material_identity_table(
        cohort,
        candidate_state,
    )
    expected_state = state.loc[
        state["stock_quantity"].eq("outflow")
        & state["event_year"].ge(2020)
        & state["amount"].gt(0.0)
    ].reset_index(drop=True)
    expected_identity = (
        identity.merge(
            expected_state.loc[:, ["cohort_year", "region", "good"]].drop_duplicates(),
            on=["cohort_year", "region", "good"],
            how="inner",
            validate="many_to_one",
        )
        .loc[lambda frame: frame["material_share"].ne(0.0)]
        .reset_index(drop=True)
    )
    expected_state = expected_state.loc[
        :, ["event_year", "cohort_year", "region", "good", "stock_quantity", "amount"]
    ]
    expected_identity = expected_identity.loc[
        :, ["cohort_year", "region", "material", "good", "material_share"]
    ]
    np.testing.assert_array_equal(candidate_state.to_numpy(), expected_state.to_numpy())
    np.testing.assert_array_equal(candidate_identity.to_numpy(), expected_identity.to_numpy())
