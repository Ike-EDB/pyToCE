"""Contribution tests for zero factors and bounded detail."""

from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from contribution_fixtures import diagonal_lca

from pytoce.run_lca.calculation.contribution_accumulation import (
    DETAIL_LIMIT,
    GLOBAL_REGION,
    ContributionMethod,
    add_provider_batch,
    begin_matrix_contributions,
    create_contribution_state,
    finish_matrix_contributions,
)
from pytoce.run_lca.calculation.contributions import (
    _regional_stage_contributions,
    finalize_contributions,
)
from pytoce.run_lca.calculation.direct_contributions import (
    DirectContribution,
    add_direct_contributions,
)
from pytoce.run_lca.calculation.foreground import H12_REGIONS


def _activation(provider_database: str) -> dict[str, object]:
    """Return one complete provider activation at the contribution table grain."""

    return {
        "matrix_database": provider_database,
        "provider_database": provider_database,
        "provider_code": "provider",
        "provider_product_id": 9001,
        "amount": 1.0,
        "year": 2050,
        "life_cycle_stage": "virgin_resin_production",
        "stage_group": "material_production",
        "cutoff_cycle_role": "primary_material_cycle",
        "material_cycle_source": "primary_resin",
        "physical_event_basis": "year new polymer is produced",
        "physical_event_year": 2050,
        "source_flow_names": "virgin production",
        "lca_fate": "production",
        "fate_origin": "model_stage",
        "material": "HDPE",
        "constituent": "hdpe",
        "product": "polymer",
        "virgin_feedstock_route": "fossil",
        "ultimate_carbon_origin": "fossil",
        "good": "Packaging",
        "stage_database": "foreground",
        "stage_code": "stage",
    }


def _method() -> ContributionMethod:
    """Describe the registered fixture method used by contribution calculations."""

    return ContributionMethod(
        method_id="ef31",
        method_version="3.1",
        impact_category_id="climate change",
        method_tuple=("EF v3.1", "climate change"),
        unit="kg CO2-Eq",
    )


def test_registered_method_factors_are_read_from_the_active_brightway_lca(
    contribution_brightway: ModuleType,
) -> None:
    """Use method switching when factors are owned by the registered Brightway method."""

    state = create_contribution_state(
        contribution_brightway,
        {"EUR": pd.DataFrame([_activation("provider_2050")])},
        {"EUR": ("foreground", "root_eur")},
        (_method(),),
        (2050,),
    )
    calculation = diagonal_lca(contribution_brightway, (3.0,), amounts=(2.0,))

    matrix = begin_matrix_contributions(state, calculation, "provider_2050")

    np.testing.assert_array_equal(matrix.factors, np.array([[3.0]]))
    np.testing.assert_array_equal(matrix.activity_scores, np.array([[6.0]]))
    flow_id = int(calculation.dicts.biosphere.reversed[0])
    assert state.characterization_factors == {("EF v3.1", "climate change"): {flow_id: 3.0}}


def test_native_details_remain_bounded_across_provider_frontiers(
    contribution_brightway: ModuleType,
) -> None:
    """Keep the largest one hundred activity details after several dated matrices."""

    size = 60
    databases = ("provider_2025", "provider_2030")
    state = create_contribution_state(
        contribution_brightway,
        {"EUR": pd.DataFrame([_activation(database) for database in databases])},
        {"EUR": ("foreground", "root_eur")},
        (_method(),),
        (2050,),
    )
    for database in databases:
        calculation = diagonal_lca(contribution_brightway, (1.0,) * size, database=database)
        matrix = begin_matrix_contributions(state, calculation, database)
        supply = np.ones((size, 1), dtype=np.float64)
        add_provider_batch(
            matrix,
            pd.DataFrame(
                [
                    {
                        "matrix_database": database,
                        "provider_database": database,
                        "provider_code": "provider",
                        "provider_product_id": 9001,
                    }
                ]
            ),
            (9001,),
            supply,
            supply.copy(),
        )
        finish_matrix_contributions(matrix)

    assert len(state.native_values[("EUR", 2050, 0)]) == DETAIL_LIMIT
    assert state.native_values[(GLOBAL_REGION, 2050, 0)] == state.native_values[("EUR", 2050, 0)]
    regional_stage_key = next(key for key in state.stage_native_values if key[0][0] == "EUR")
    global_stage_key = next(key for key in state.stage_native_values if key[0][0] == GLOBAL_REGION)
    assert len(state.stage_native_values[regional_stage_key]) == DETAIL_LIMIT
    assert len(state.stage_native_values[global_stage_key]) == DETAIL_LIMIT


def test_zero_characterization_factors_retain_an_explicit_stage_residual(
    contribution_brightway: ModuleType,
) -> None:
    """Represent a activation when the selected method has an explicit zero factor."""

    state = create_contribution_state(
        contribution_brightway,
        {"EUR": pd.DataFrame([_activation("provider_2050")])},
        {"EUR": ("foreground", "root_eur")},
        (_method(),),
        (2050,),
    )
    calculation = diagonal_lca(contribution_brightway, (0.0,))
    matrix = begin_matrix_contributions(state, calculation, "provider_2050")
    add_provider_batch(
        matrix,
        pd.DataFrame(
            [
                {
                    "matrix_database": "provider_2050",
                    "provider_database": "provider_2050",
                    "provider_code": "provider",
                    "provider_product_id": 9001,
                }
            ]
        ),
        (9001,),
        np.ones((1, 1)),
        np.ones((1, 1)),
    )
    finish_matrix_contributions(matrix)

    rows = _regional_stage_contributions(state)

    classification = rows.loc[
        rows["contribution_view"].eq("classification") & rows["region"].eq("EUR")
    ]
    assert len(classification) == 1
    assert classification.iloc[0]["record_role"] == "exact_residual"
    assert classification.iloc[0]["score"] == 0.0
    assert classification.iloc[0]["contribution_norm"] == 0.0


@pytest.mark.parametrize(
    ("eur_named", "usa_named", "eur_other", "usa_other", "direct", "selected", "residual"),
    [
        pytest.param(1, 5, 10, 0, 0, False, 6, id="independent-global-cutoff"),
        pytest.param(1, 15, 10, 0, -2, True, 10, id="complete-named-plus-direct"),
        pytest.param(1, -15, 10, 0, 0, True, 10, id="signed-named"),
        pytest.param(6, 6, 7, 7, 0, True, 707, id="omitted-in-every-active-region"),
    ],
)
def test_global_detail_uses_complete_regional_values_before_ranking(
    contribution_brightway: ModuleType,
    eur_named: int,
    usa_named: int,
    eur_other: int,
    usa_other: int,
    direct: int,
    selected: bool,
    residual: int,
) -> None:
    """Rank global activity/flow values with exact residuals over the 12 world regions."""
    database = "provider_2050"
    eur = pd.DataFrame([_activation(database)])
    usa = eur.assign(provider_product_id=9002, provider_code="usa_provider")
    activations = {
        region: {"EUR": eur, "USA": usa}.get(region, eur.iloc[:0]) for region in H12_REGIONS
    }
    state = create_contribution_state(
        contribution_brightway,
        activations,
        {region: ("foreground", f"root_{region}") for region in H12_REGIONS},
        (_method(),),
        (2050,),
    )
    size = 201 if usa_other else 101
    calculation = diagonal_lca(contribution_brightway, (1.0,) * size, database=database)
    named_activity = contribution_brightway.get_node(database=database, code="0")
    named_column = calculation.dicts.activity[int(named_activity.id)]
    other_columns = [column for column in range(size) if column != named_column]
    supply = np.zeros((size, 2))
    supply[named_column] = (eur_named, usa_named)
    supply[other_columns[:100], 0] = eur_other
    supply[other_columns[100:], 1] = usa_other
    matrix = begin_matrix_contributions(state, calculation, database)
    add_provider_batch(
        matrix,
        state.activations,
        (9001, 9002),
        supply,
        np.asarray(calculation.biosphere_matrix @ supply),
    )
    finish_matrix_contributions(matrix)
    assert all(len(values) <= DETAIL_LIMIT for values in state.native_values.values())
    named_key = (database, int(named_activity.id))
    assert named_key not in state.native_values[("EUR", 2050, 0)]
    if usa_other:
        assert named_key not in state.native_values[("USA", 2050, 0)]
    if direct:
        flow = contribution_brightway.get_node(database=f"{database}_biosphere", code="0")
        add_direct_contributions(
            state,
            [
                DirectContribution(
                    stage_coordinate=(
                        "EUR",
                        2050,
                        "virgin_resin_production",
                        "material_production",
                        "primary_material_cycle",
                        "primary_resin",
                        "year new polymer is produced",
                        2050,
                        "virgin production",
                        "production",
                        "model_stage",
                    ),
                    receiving_function="invented direct receiving",
                    source_records_json="[]",
                    endpoint_unit="kilogram",
                    flow_id=int(flow.id),
                    amount=direct,
                )
            ],
        )
    tables = dict(finalize_contributions(state))
    frame = tables["contributions"]
    assert set(frame["region"]) == {*H12_REGIONS, GLOBAL_REGION}
    global_rows = frame.loc[frame["region"].eq(GLOBAL_REGION)]
    expected_score = float(supply.sum()) + direct
    expected_norm = float(np.abs(supply).sum()) + abs(direct)
    totals = global_rows.groupby("contribution_view")[["score", "contribution_norm"]].sum()
    np.testing.assert_array_equal(totals["score"], expected_score)
    np.testing.assert_array_equal(totals["contribution_norm"], expected_norm)
    for view, label, amount, magnitude in (
        (
            "native_activity",
            "activity_code",
            eur_named + usa_named,
            abs(eur_named) + abs(usa_named),
        ),
        (
            "elementary_flow",
            "elementary_flow_code",
            eur_named + usa_named + direct,
            abs(eur_named) + abs(usa_named) + abs(direct),
        ),
    ):
        values = global_rows.loc[global_rows["contribution_view"].eq(view)]
        detail = values.loc[values["record_role"].eq("detail")]
        assert len(detail) == DETAIL_LIMIT
        assert detail["contribution_norm"].is_monotonic_decreasing
        named = detail.loc[detail[label].eq("0")]
        if selected:
            assert len(named) == 1
            assert named.iloc[0]["score"] == amount
            assert named.iloc[0]["contribution_norm"] == magnitude
        else:
            assert named.empty
        remainder = values.loc[values["record_role"].eq("exact_residual")]
        assert len(remainder) == 1
        assert remainder.iloc[0]["score"] == residual
        assert remainder.iloc[0]["contribution_norm"] == residual
