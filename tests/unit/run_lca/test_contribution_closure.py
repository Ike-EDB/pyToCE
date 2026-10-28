"""Exact additive-view closure at the characterized contribution grain."""

import math
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from contribution_fixtures import diagonal_lca

from pytoce.run_lca.calculation.contribution_accumulation import (
    GLOBAL_REGION,
    ContributionMethod,
    ContributionState,
    add_provider_batch,
    begin_matrix_contributions,
    create_contribution_state,
    create_provider_response_state,
    finish_matrix_contributions,
)
from pytoce.run_lca.calculation.contribution_records import (
    _frame_from_records,
)
from pytoce.run_lca.calculation.contributions import (
    _provider_effective_values,
    _regional_contributions,
    finalize_contributions,
)
from pytoce.run_lca.calculation.stage_scores import (
    _with_shares,
)


def _finalized_tables(state: ContributionState) -> dict[str, pd.DataFrame]:
    return dict(finalize_contributions(state))


def test_bounded_record_materialization_preserves_exact_frame() -> None:
    records = [
        {
            "coordinate": index,
            "amount": -0.0 if index == 0 else float(index) / 10.0,
            "label": "" if index % 2 == 0 else f"row_{index}",
            "optional": pd.NA if index % 3 == 0 else index,
        }
        for index in range(25_001)
    ]
    expected = pd.DataFrame.from_records(records)
    actual = _frame_from_records(iter(records))
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_provider_response_state_retains_unit_scores_without_contribution_reductions(
    contribution_brightway: ModuleType,
) -> None:
    state = create_provider_response_state(
        contribution_brightway,
        (
            ContributionMethod(
                method_id="bounded",
                method_version="1",
                impact_category_id="example_category",
                method_tuple=("bounded", "example_category"),
                unit="impact unit",
            ),
        ),
    )
    lca = diagonal_lca(
        contribution_brightway,
        (3.0, -4.0),
        method=("bounded", "example_category"),
        unit="impact unit",
    )
    matrix = begin_matrix_contributions(state, lca, "provider_frontier")
    supply = np.array([[1.0], [-2.0]])
    characterized = add_provider_batch(
        matrix,
        pd.DataFrame(),
        (7,),
        supply,
        np.asarray(lca.biosphere_matrix @ supply),
    )
    finish_matrix_contributions(matrix)

    np.testing.assert_array_equal(characterized.scores, np.array([[11.0]]))
    np.testing.assert_array_equal(characterized.norms, np.array([[11.0]]))
    assert state.annual_totals == {}
    assert state.stage_values == {}


def test_annual_accumulation_retains_activations_outside_detailed_years(
    contribution_brightway: ModuleType,
) -> None:
    """Keep all annual scores while limiting detailed views to selected years."""

    provider_database = "provider_frontier"

    def activation(*, year: int, amount: float) -> dict[str, object]:
        """Return one provider activation for an annual life cycle stage."""

        return {
            "matrix_database": provider_database,
            "provider_database": provider_database,
            "provider_code": "provider",
            "provider_product_id": 9001,
            "amount": amount,
            "year": year,
            "life_cycle_stage": "virgin_resin_production",
            "stage_group": "material_production",
            "cutoff_cycle_role": "primary_material_cycle",
            "material_cycle_source": "primary_resin",
            "physical_event_basis": "year new polymer is produced",
            "physical_event_year": year,
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

    detailed_year = 2050
    annual_only_year = 2035
    state = create_contribution_state(
        contribution_brightway,
        {
            "EUR": pd.DataFrame(
                [
                    activation(year=annual_only_year, amount=2.0),
                    activation(year=detailed_year, amount=4.0),
                ]
            )
        },
        {"EUR": ("foreground", "root_eur")},
        (
            ContributionMethod(
                method_id="ef31",
                method_version="3.1",
                impact_category_id="climate change",
                method_tuple=("EF v3.1", "climate change"),
                unit="kg CO2-Eq",
            ),
        ),
        (detailed_year,),
    )
    lca = diagonal_lca(contribution_brightway, (-3.0,))
    matrix = begin_matrix_contributions(state, lca, provider_database)

    add_provider_batch(
        matrix,
        pd.DataFrame(
            [
                {
                    "matrix_database": provider_database,
                    "provider_database": provider_database,
                    "provider_code": "provider",
                    "provider_product_id": 9001,
                }
            ]
        ),
        (9001,),
        np.ones((1, 1), dtype=np.float64),
        np.ones((1, 1), dtype=np.float64),
    )
    finish_matrix_contributions(matrix)

    assert state.annual_totals == {
        ("EUR", annual_only_year, 0): (-6.0, 6.0),
        ("EUR", detailed_year, 0): (-12.0, 12.0),
    }
    assert {
        (key[0], key[1], key[2]): value for key, value in state.annual_stage_values.items()
    } == {
        ("EUR", annual_only_year, 0): (-6.0, 6.0),
        ("EUR", detailed_year, 0): (-12.0, 12.0),
    }
    assert {key[1] for key in state.stage_values} == {detailed_year}
    assert {key[1] for key in state.provider_values} == {detailed_year}
    assert {key[1] for key in state.native_values} == {detailed_year}


def test_empty_regional_contributions_retain_all_zero_views(
    contribution_brightway: ModuleType,
) -> None:
    """Represent a selected region and year with no characterized burden."""

    method = ContributionMethod(
        method_id="ef31",
        method_version="3.1",
        impact_category_id="climate change",
        method_tuple=("EF v3.1", "climate change"),
        unit="kg CO2-Eq",
    )
    state = ContributionState(
        bd=contribution_brightway,
        methods=(method,),
        years=(2050,),
        regions=("EUR",),
        groups=(("EUR", 2050),),
        activations=pd.DataFrame(),
    )

    observed = _regional_contributions(state)

    assert set(observed["contribution_view"]) == {
        "life_cycle_stage",
        "provider_frontier",
        "native_activity",
        "elementary_flow",
        "classification",
    }
    assert observed["record_role"].eq("exact_residual").all()
    assert observed[["score", "contribution_norm"]].eq(0.0).all(axis=None)


def test_each_additive_view_reports_total_score_and_contribution_norm() -> None:
    rows = []
    for view in (
        "life_cycle_stage",
        "provider_frontier",
        "native_activity",
        "elementary_flow",
        "classification",
    ):
        for record_role, score, norm in (
            ("positive", 2.0, 2.0),
            ("negative", -1.0, 1.0),
        ):
            rows.append(
                {
                    "region": "EUR",
                    "year": 2050,
                    "method_id": "ef31",
                    "method_version": "3.1",
                    "impact_category_id": "climate change",
                    "impact_category_tuple_json": '["EF v3.1","climate change"]',
                    "unit": "kg CO2-Eq",
                    "contribution_view": view,
                    "record_role": record_role,
                    "score": score,
                    "contribution_norm": norm,
                }
            )

    contributions = _with_shares(pd.DataFrame(rows))
    grouped = contributions.groupby("contribution_view")[
        ["score", "contribution_norm", "share", "norm_share"]
    ].sum()

    assert grouped["score"].tolist() == pytest.approx([1.0] * 5)
    assert grouped["contribution_norm"].tolist() == pytest.approx([3.0] * 5)
    assert grouped["share"].tolist() == pytest.approx([1.0] * 5)
    assert grouped["norm_share"].tolist() == pytest.approx([1.0] * 5)
    assert contributions["share_status"].eq("available").all()


def test_total_and_absolute_supply_match_with_bounded_details(
    contribution_brightway: ModuleType,
) -> None:
    size = 101
    region = "EUR"
    year = 2050
    provider_database = "bridge_2050"
    activations = pd.DataFrame(
        [
            {
                "matrix_database": provider_database,
                "provider_database": provider_database,
                "provider_code": "provider",
                "provider_product_id": 9001,
                "amount": 1.0,
                "year": year,
                "life_cycle_stage": "virgin_resin_production",
                "stage_group": "material_production",
                "cutoff_cycle_role": "primary_material_cycle",
                "material_cycle_source": "primary_resin",
                "physical_event_basis": "year new polymer is produced",
                "physical_event_year": year,
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
                "unused_activation_lineage": "not consumed by contribution reduction",
            }
        ]
    )
    # Two constituents use one provider-unit inventory, but retain separate paths.
    activations.loc[0, "material"] = "BioPlastics"
    activations.loc[0, "constituent"] = "other_biodegradable"
    activations.loc[0, "amount"] = 0.4
    other = activations.iloc[0].copy()
    other["constituent"] = "starch_blends"
    other["ultimate_carbon_origin"] = "biogenic"
    other["amount"] = 0.6
    activations = pd.concat((activations, other.to_frame().T), ignore_index=True)
    state = create_contribution_state(
        contribution_brightway,
        {region: activations},
        {region: ("foreground", "root_eur")},
        (
            ContributionMethod(
                method_id="ef31",
                method_version="3.1",
                impact_category_id="climate change",
                method_tuple=("EF v3.1", "climate change"),
                unit="kg CO2-Eq",
            ),
        ),
        (year,),
    )
    assert list(state.activations.columns) == [
        "region",
        "year",
        "matrix_database",
        "provider_database",
        "provider_code",
        "provider_product_id",
        "amount",
        "regional_root_database",
        "regional_root_code",
        "stage_database",
        "stage_code",
        "life_cycle_stage",
        "stage_group",
        "cutoff_cycle_role",
        "material_cycle_source",
        "physical_event_basis",
        "physical_event_year",
        "source_flow_names",
        "lca_fate",
        "fate_origin",
        "material",
        "constituent",
        "product",
        "virgin_feedstock_route",
        "ultimate_carbon_origin",
        "good",
        "receiving_function",
        "source_records_json",
        "endpoint_unit",
    ]
    lca = diagonal_lca(contribution_brightway, (1.0,) * size)
    matrix = begin_matrix_contributions(state, lca, provider_database)
    supply = np.ones((size, 1), dtype=np.float64)
    supply[-1, 0] = -100.0
    characterized = add_provider_batch(
        matrix,
        pd.DataFrame(
            [
                {
                    "matrix_database": provider_database,
                    "provider_database": provider_database,
                    "provider_code": "provider",
                    "provider_product_id": 9001,
                }
            ]
        ),
        (9001,),
        supply,
        np.asarray(lca.biosphere_matrix @ supply),
    )
    np.testing.assert_array_equal(characterized.scores, np.array([[0.0]]))
    np.testing.assert_array_equal(characterized.norms, np.array([[200.0]]))
    finish_matrix_contributions(matrix)
    tables = _finalized_tables(state)
    for field_name in (
        "annual_totals",
        "annual_stage_values",
        "stage_values",
        "provider_values",
        "stage_provider_values",
        "stage_provider_amounts",
        "native_values",
        "stage_native_values",
        "flow_values",
        "stage_flow_values",
        "classification_values",
        "stage_classification_values",
        "structural_values",
        "activity_metadata",
        "flow_metadata",
        "direct_values",
        "characterization_factors",
        "factor_cache",
    ):
        assert not getattr(state, field_name)

    regional = tables["contributions"].loc[tables["contributions"]["region"].eq(region)]
    closure = regional.groupby("contribution_view")[["score", "contribution_norm"]].sum()
    assert closure["score"].tolist() == pytest.approx([0.0] * 5)
    assert closure["contribution_norm"].tolist() == pytest.approx([200.0] * 5)
    stage = regional.loc[regional["contribution_view"].eq("life_cycle_stage")].iloc[0]
    assert stage["stage_group"] == "material_production"
    assert stage["cutoff_cycle_role"] == "primary_material_cycle"
    assert stage["physical_event_basis"] == "year new polymer is produced"
    assert stage["physical_event_year"] == year
    paths = tables["structural_paths"]
    assert paths.set_index("constituent").activation_amount.to_dict() == {
        "other_biodegradable": 0.4,
        "starch_blends": 0.6,
    }
    assert paths.set_index("constituent").ultimate_carbon_origin.to_dict() == {
        "other_biodegradable": "fossil",
        "starch_blends": "biogenic",
    }
    assert paths.path_id.nunique() == 2
    assert tables["structural_edges"].groupby("path_id").size().to_list() == [2, 2]
    path = tables["structural_paths"].iloc[0]
    assert path["stage_group"] == "material_production"
    assert path["cutoff_cycle_role"] == "primary_material_cycle"
    assert path["physical_event_year"] == year
    for view in ("native_activity", "elementary_flow"):
        selected = regional.loc[regional["contribution_view"].eq(view)]
        assert selected["record_role"].eq("detail").sum() == 100
        residual = selected.loc[selected["record_role"].eq("exact_residual")].iloc[0]
        assert residual["score"] == pytest.approx(1.0)
        assert residual["contribution_norm"] == pytest.approx(1.0)

    regional_stage = tables["stage_contributions"].loc[
        tables["stage_contributions"]["region"].eq(region)
    ]
    stage_closure = regional_stage.groupby("contribution_view")[
        [
            "score",
            "contribution_norm",
        ]
    ].sum()
    assert stage_closure["score"].tolist() == pytest.approx([0.0] * 5)
    assert stage_closure["contribution_norm"].tolist() == pytest.approx([200.0] * 5)
    for view in ("native_activity", "elementary_flow"):
        selected = regional_stage.loc[regional_stage["contribution_view"].eq(view)]
        assert selected["record_role"].eq("detail").sum() == 25
        residual = selected.loc[selected["record_role"].eq("exact_residual")].iloc[0]
        assert residual["score"] == pytest.approx(76.0)
        assert residual["contribution_norm"] == pytest.approx(76.0)
    provider = regional_stage.loc[regional_stage["contribution_view"].eq("provider_frontier")].iloc[
        0
    ]
    assert provider["activation_amount"] == pytest.approx(1.0)
    assert provider["absolute_activation_amount"] == pytest.approx(1.0)
    assert provider["effective_unit_score"] == pytest.approx(0.0)
    assert provider["effective_unit_norm"] == pytest.approx(200.0)
    assert provider["effective_unit_status"] == "available"

    annual_stage = (
        tables["annual_stage_scores"]
        .loc[
            tables["annual_stage_scores"]["region"].eq(region)
            & tables["annual_stage_scores"]["year"].eq(year)
            & tables["annual_stage_scores"]["life_cycle_stage"].eq("virgin_resin_production")
        ]
        .iloc[0]
    )
    assert annual_stage["score"] == pytest.approx(0.0)
    assert annual_stage["contribution_norm"] == pytest.approx(200.0)
    assert annual_stage["physical_event_year"] == year
    cumulative_stage = (
        tables["cumulative_stage_scores"]
        .loc[
            tables["cumulative_stage_scores"]["region"].eq(region)
            & tables["cumulative_stage_scores"]["life_cycle_stage"].eq("virgin_resin_production")
        ]
        .iloc[0]
    )
    assert cumulative_stage["score"] == pytest.approx(0.0)
    assert cumulative_stage["contribution_norm"] == pytest.approx(200.0)
    assert pd.isna(cumulative_stage["physical_event_year"])


def test_matrix_contribution_coordinates_are_local_to_the_provider_frontier(
    contribution_brightway: ModuleType,
) -> None:
    year = 2050

    def activation(
        provider_database: str,
        stage: str,
        amount: float,
        *,
        activation_year: int = year,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "matrix_database": provider_database,
                    "provider_database": provider_database,
                    "provider_code": "provider",
                    "provider_product_id": 9001,
                    "amount": amount,
                    "year": activation_year,
                    "life_cycle_stage": stage,
                    "stage_group": "material_production",
                    "cutoff_cycle_role": "primary_material_cycle",
                    "material_cycle_source": "primary_resin",
                    "physical_event_basis": "year new polymer is produced",
                    "physical_event_year": activation_year,
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
                    "stage_code": stage,
                }
            ]
        )

    def contribution_state() -> ContributionState:
        return create_contribution_state(
            contribution_brightway,
            {
                "EUR": pd.concat(
                    [
                        activation("frontier_a", "virgin_resin_production", 1.0),
                        activation(
                            "frontier_a",
                            "virgin_resin_production",
                            99.0,
                            activation_year=year - 1,
                        ),
                        activation("frontier_b", "virgin_resin_production", 1.0),
                    ],
                    ignore_index=True,
                ),
                "USA": activation(
                    "frontier_b",
                    "virgin_resin_production",
                    1.0,
                ),
            },
            {
                "EUR": ("foreground", "root_eur"),
                "USA": ("foreground", "root_usa"),
            },
            (
                ContributionMethod(
                    method_id="ef31",
                    method_version="3.1",
                    impact_category_id="climate change",
                    method_tuple=("EF v3.1", "climate change"),
                    unit="kg CO2-Eq",
                ),
            ),
            (year,),
        )

    state = contribution_state()
    lca_a = diagonal_lca(contribution_brightway, (1.0,), database="frontier_a")
    lca_b = diagonal_lca(contribution_brightway, (1.0,), database="frontier_b", amounts=(-1.0,))
    lca_empty = diagonal_lca(
        contribution_brightway, (1.0,), database="frontier_without_detail_year"
    )
    frontier_a = begin_matrix_contributions(state, lca_a, "frontier_a")
    frontier_b = begin_matrix_contributions(state, lca_b, "frontier_b")
    empty = begin_matrix_contributions(state, lca_empty, "frontier_without_detail_year")

    assert tuple(frontier_a.group_index) == (("EUR", year),)
    assert tuple(frontier_b.group_index) == (("EUR", year), ("USA", year))
    assert {coordinate[0] for coordinate in frontier_a.stage_group_index} == {
        "EUR",
        GLOBAL_REGION,
    }
    assert {coordinate[2] for coordinate in frontier_a.stage_group_index} == {
        "virgin_resin_production"
    }
    assert {coordinate[0] for coordinate in frontier_b.stage_group_index} == {
        "EUR",
        "USA",
        GLOBAL_REGION,
    }
    assert {coordinate[2] for coordinate in frontier_b.stage_group_index} == {
        "virgin_resin_production",
    }
    assert frontier_a.stage_supply.shape[1] == 2
    assert frontier_b.stage_supply.shape[1] == 3
    assert empty.supply.shape[1] == 0
    assert empty.stage_supply.shape[1] == 0

    candidate_matrices = (frontier_a, frontier_b)
    provider_rows = pd.DataFrame(
        [
            {
                "matrix_database": provider_database,
                "provider_database": provider_database,
                "provider_code": "provider",
                "provider_product_id": 9001,
            }
            for provider_database in ("frontier_a", "frontier_b")
        ]
    )
    for provider_database, matrix in zip(
        ("frontier_a", "frontier_b"), candidate_matrices, strict=True
    ):
        supply = np.ones((1, 1), dtype=np.float64)
        add_provider_batch(
            matrix,
            provider_rows.loc[provider_rows["provider_database"].eq(provider_database)],
            (9001,),
            supply,
            np.asarray(matrix.biosphere @ supply),
        )
        finish_matrix_contributions(matrix)

    candidate = _finalized_tables(state)
    regional_stages = (
        candidate["contributions"]
        .loc[
            candidate["contributions"]["contribution_view"].eq("life_cycle_stage")
            & candidate["contributions"]["record_role"].eq("detail")
            & candidate["contributions"]["region"].ne(GLOBAL_REGION),
            ["region", "life_cycle_stage", "score", "contribution_norm"],
        ]
        .sort_values("region", ignore_index=True)
    )
    assert regional_stages["score"].tolist() == pytest.approx([0.0, -1.0])
    assert regional_stages["contribution_norm"].tolist() == pytest.approx([2.0, 1.0])

    global_views = (
        candidate["contributions"]
        .loc[candidate["contributions"]["region"].eq(GLOBAL_REGION)]
        .groupby("contribution_view")[["score", "contribution_norm"]]
        .sum()
    )
    assert global_views["score"].tolist() == pytest.approx([-1.0] * 5)
    assert global_views["contribution_norm"].tolist() == pytest.approx([3.0] * 5)


def test_recycling_and_residue_stay_in_their_physical_event_year(
    contribution_brightway: ModuleType,
) -> None:
    provider_database = "bridge"
    rows = (
        (
            2035,
            "polymerization_operation",
            "material_production",
            "shared_polymerization",
            "new_polymer",
            "year new polymer is produced",
            3.0,
        ),
        (
            2035,
            "virgin_monomer_supply",
            "material_production",
            "primary_material_cycle",
            "virgin_monomer",
            "year new polymer is produced",
            1.75,
        ),
        (
            2035,
            "recovered_monomer_supply",
            "material_production",
            "secondary_material_cycle_entry",
            "recovered_monomer",
            "year new polymer is produced",
            1.25,
        ),
        (
            2035,
            "mechanical_recycling_operation",
            "recycling_operations",
            "secondary_material_cycle_entry",
            "mechanically_recycled_material",
            "year waste enters mechanical recycling",
            2.0,
        ),
        (
            2035,
            "mechanical_recycling_reject_incineration_treatment",
            "recycling_residue_treatment",
            "recycling_residue_treatment",
            "not_applicable",
            "same year as the mechanical-recycling operation that generates the reject",
            0.5,
        ),
        (
            2050,
            "incineration_treatment",
            "managed_terminal_end_of_life",
            "waste_producer_terminal_treatment",
            "not_applicable",
            "year collected end-of-life waste is incinerated",
            1.0,
        ),
    )
    activations = pd.DataFrame(
        [
            {
                "matrix_database": provider_database,
                "provider_database": provider_database,
                "provider_code": "provider",
                "provider_product_id": 9001,
                "amount": amount,
                "region": "EUR",
                "year": year,
                "life_cycle_stage": stage,
                "stage_group": stage_group,
                "cutoff_cycle_role": cutoff_role,
                "material_cycle_source": material_cycle_source,
                "physical_event_basis": basis,
                "physical_event_year": year,
                "source_flow_names": stage,
                "lca_fate": "incineration" if "incineration" in stage else "",
                "fate_origin": "remind_mfa_physical_route",
                "material": "HDPE",
                "constituent": "hdpe",
                "product": "ethylene"
                if stage in {"virgin_monomer_supply", "recovered_monomer_supply"}
                else "polymer",
                "virgin_feedstock_route": "fossil" if stage == "virgin_monomer_supply" else "all",
                "ultimate_carbon_origin": "fossil",
                "good": "Packaging",
                "stage_database": "foreground",
                "stage_code": stage,
            }
            for year, stage, stage_group, cutoff_role, material_cycle_source, basis, amount in rows
        ]
    )
    state = create_contribution_state(
        contribution_brightway,
        {"EUR": activations},
        {"EUR": ("foreground", "root_eur")},
        (
            ContributionMethod(
                method_id="ef31",
                method_version="3.1",
                impact_category_id="climate change",
                method_tuple=("EF v3.1", "climate change"),
                unit="kg CO2-Eq",
            ),
        ),
        (2035, 2050),
    )
    lca = diagonal_lca(contribution_brightway, (1.0,))
    matrix = begin_matrix_contributions(state, lca, provider_database)
    add_provider_batch(
        matrix,
        pd.DataFrame(
            [
                {
                    "matrix_database": provider_database,
                    "provider_database": provider_database,
                    "provider_code": "provider",
                    "provider_product_id": 9001,
                }
            ]
        ),
        (9001,),
        np.ones((1, 1), dtype=np.float64),
        np.ones((1, 1), dtype=np.float64),
    )
    finish_matrix_contributions(matrix)
    annual = _finalized_tables(state)["annual_stage_scores"]
    nonzero = annual.loc[
        annual["region"].eq("EUR") & annual["score"].ne(0.0),
        ["year", "life_cycle_stage", "physical_event_year", "score"],
    ].sort_values(["year", "life_cycle_stage"], ignore_index=True)

    assert set(nonzero.loc[nonzero["year"].eq(2035), "life_cycle_stage"]) == {
        "polymerization_operation",
        "virgin_monomer_supply",
        "recovered_monomer_supply",
        "mechanical_recycling_operation",
        "mechanical_recycling_reject_incineration_treatment",
    }
    assert set(nonzero.loc[nonzero["year"].eq(2050), "life_cycle_stage"]) == {
        "incineration_treatment"
    }
    assert nonzero["physical_event_year"].equals(nonzero["year"])
    assert nonzero.set_index("life_cycle_stage").score.to_dict() == {
        "polymerization_operation": 3.0,
        "virgin_monomer_supply": 1.75,
        "recovered_monomer_supply": 1.25,
        "mechanical_recycling_operation": 2.0,
        "mechanical_recycling_reject_incineration_treatment": 0.5,
        "incineration_treatment": 1.0,
    }


@pytest.mark.parametrize(
    ("score", "norm", "status", "share"),
    (
        (2.0, 2.0, "available", 1.0),
        (-2.0, 2.0, "available", 1.0),
        (1e-10, 1e-10, "available", 1.0),
        (-1e-10, 1e-10, "available", 1.0),
        (0.0, 2.0, "zero_score", math.nan),
        (1e-11, 1.0, "near_zero_score", math.nan),
        (1e-10, 1.0, "near_zero_score", math.nan),
        (2e-10, 1.0, "available", 1.0),
        (0.0, 0.0, "zero_contribution_norm", math.nan),
    ),
)
@pytest.mark.parametrize(("unit", "unit_scale"), (("kg CO2-Eq", 1.0), ("t CO2-Eq", 1e-3)))
def test_share_status_contract(
    score: float,
    norm: float,
    status: str,
    share: float,
    unit: str,
    unit_scale: float,
) -> None:
    """Equivalent category units retain shares, zero cases and cancellation flags."""

    result = _with_shares(
        pd.DataFrame(
            [
                {
                    "region": "EUR",
                    "year": 2050,
                    "method_id": "ef31",
                    "method_version": "3.1",
                    "impact_category_id": "climate change",
                    "impact_category_tuple_json": '["EF v3.1","climate change"]',
                    "unit": unit,
                    "contribution_view": "life_cycle_stage",
                    "record_role": "detail",
                    "score": score * unit_scale,
                    "contribution_norm": norm * unit_scale,
                }
            ]
        )
    ).iloc[0]

    assert result["score"] == score * unit_scale
    assert result["contribution_norm"] == norm * unit_scale
    assert result["share_status"] == status
    if math.isnan(share):
        assert math.isnan(float(result["share"]))
    else:
        assert result["share"] == pytest.approx(share)


@pytest.mark.parametrize(
    ("amount", "absolute_amount", "status"),
    (
        (1e-10, 1e-10, "available"),
        (-1e-10, 1e-10, "available"),
        (0.0, 0.0, "zero_total_activity"),
        (0.0, 2.0, "near_zero_total_activity"),
        (1e-11, 1.0, "near_zero_total_activity"),
        (1e-10, 1.0, "near_zero_total_activity"),
        (2e-10, 1.0, "available"),
    ),
)
@pytest.mark.parametrize("unit_scale", (1.0, 1e-3), ids=("kilograms", "tonnes"))
def test_provider_effective_values_preserve_native_unit_scaling(
    amount: float,
    absolute_amount: float,
    status: str,
    unit_scale: float,
) -> None:
    """Preserve zero PA supplies, signed totals and unit-specific diagnostics."""

    # The service has two impact units per kilogram. Conversion to metric tons scales amounts and
    # per-unit values and preserves total impacts.
    unit_score, unit_norm, observed_status = _provider_effective_values(
        amount * unit_scale,
        absolute_amount * unit_scale,
        2.0 * amount,
        2.0 * absolute_amount,
    )

    assert observed_status == status
    if status == "available":
        assert unit_score == pytest.approx(2.0 / unit_scale)
    else:
        assert math.isnan(unit_score)
    if status == "zero_total_activity":
        assert math.isnan(unit_norm)
    else:
        assert unit_norm == pytest.approx(2.0 / unit_scale)
