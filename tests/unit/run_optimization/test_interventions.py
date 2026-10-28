"""Direct intervention decisions and parent-relative trajectories."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from intervention_fixtures import (
    _circularity_parent,
    _demand,
    _parent_lifetimes,
    _parent_stock,
    _shaped_decision,
)

from pytoce.run_optimization.calculation.candidate_worker import (
    _demand_stock_multipliers,
)
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    demand_path_departure,
    demand_repair_neighbors,
    endpoint_neighbors,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
    build_circularity_trajectories,
    build_intervention_distance_inputs,
    transition_weight,
)


def test_demand_and_lifetime_remain_relative_to_contemporaneous_parent() -> None:
    result = _demand(DemandLifetimeDecision(90, 100, 2050))
    stock = result.assignments.loc[result.assignments["parameter_id"].eq("stock_projection")]
    lifetime = result.assignments.loc[result.assignments["parameter_id"].eq("lifetime_mean")]
    parent_stock = _parent_stock().set_index(["year", "region", "good"])["stock_projection"]
    parent_lifetime = _parent_lifetimes().set_index(["year", "good"])["lifetime_mean"]
    for year in (2050, 2100):
        stock_row = stock.loc[
            stock["year"].eq(year) & stock["region"].eq("EUR") & stock["good"].eq("Packaging")
        ].iloc[0]
        lifetime_row = lifetime.loc[
            lifetime["year"].eq(year) & lifetime["good"].eq("Packaging")
        ].iloc[0]
        assert stock_row["resolved_value"] == pytest.approx(
            0.1 * parent_stock.loc[(year, "EUR", "Packaging")]
        )
        assert lifetime_row["resolved_value"] == pytest.approx(
            2.0 * parent_lifetime.loc[(year, "Packaging")]
        )


def test_exact_parent_omits_assignments() -> None:
    result = _demand(DemandLifetimeDecision(0, 0, 2100))
    assert result.uses_exact_parent
    assert result.assignments.empty


def test_exact_circularity_parent_retains_its_complete_annual_values() -> None:
    """An unchanged path retains every admitted regional value and the parent identity."""
    parent = _circularity_parent()
    trajectories = [
        build_circularity_trajectories(
            parent=parent,
            parent_identity="circularity::accepted",
            decision=CircularityDecision.parent(target_year=year),
        )
        for year in (2050, 2075, 2100)
    ]
    assert len({item.identity for item in trajectories}) == 1
    for item in trajectories:
        assert item.annual_values is parent
        assert item.assignments.empty
        assert item.uses_exact_parent


@pytest.mark.parametrize("target_year", [2050, 2075, 2100])
def test_linear_reference_uses_the_complete_annual_horizon(
    target_year: int,
) -> None:
    """History stays unchanged and the exact linear endpoint remains held through 2100."""
    years = np.arange(1950, 2101)
    assert [transition_weight(int(year), target_year) for year in years] == [
        max(0, min(int(year) - 2019, target_year - 2019)) / (target_year - 2019) for year in years
    ]


def test_nonlinear_assignments_bind_paths_without_changing_endpoint_distance() -> None:
    """Use repaired demand and linear lifetime while retaining parent values and endpoint effort."""
    linear = _demand(DemandLifetimeDecision(50, 25))
    shaped = _demand(_shaped_decision(50, 25))
    assert shaped.identity != linear.identity
    stock = _parent_stock().set_index(["year", "region", "good"])["stock_projection"]
    lifetimes = _parent_lifetimes().set_index(["year", "good"])["lifetime_mean"]
    for row in shaped.assignments.itertuples(index=False):
        t = min((row.year - 2019) / 31, 1)
        q = 2 * t - t**2 if row.parameter_id == "stock_projection" else t
        expected = (
            stock.loc[(row.year, row.region, row.good)] * (1 - 0.5 * q)
            if row.parameter_id == "stock_projection"
            else lifetimes.loc[(row.year, row.good)] * (1 + 0.25 * q)
        )
        assert row.resolved_value == pytest.approx(expected)
    circularity = build_circularity_trajectories(
        parent=_circularity_parent(),
        parent_identity="circularity",
        decision=CircularityDecision.parent(),
    )
    distances = [
        build_intervention_distance_inputs(
            physical_parent_identity="physical",
            demand_lifetime_parent_identity="demand",
            demand_lifetime=item,
            circularity=circularity,
            parent_circularity=_circularity_parent(),
        )
        for item in (linear, shaped)
    ]
    assert [
        (item.demand_reduction_fraction, item.lifetime_extension_fraction) for item in distances
    ] == [(0.5, 0.25), (0.5, 0.25)]


def test_complete_shares_interpolate_from_each_regional_start() -> None:
    decision = CircularityDecision(80, 40, 10, 30, 20, 70, 20, 10, 2050)
    result = build_circularity_trajectories(
        parent=_circularity_parent(),
        parent_identity="circularity::accepted",
        decision=decision,
    )
    selected = result.assignments.loc[
        result.assignments["year"].eq(2050) & result.assignments["region"].eq("EUR")
    ].set_index("parameter_id")["resolved_value"]
    assert selected["collection_rate"] == pytest.approx(0.8)
    assert selected["mechanical_recycling_rate"] == pytest.approx(0.4)
    assert selected["chemical_recycling_rate"] == pytest.approx(0.1)
    assert selected["incineration_rate"] == pytest.approx(0.3)
    assert selected["bio_production_rate"] == pytest.approx(0.2)
    assert selected["daccu_production_rate"] == pytest.approx(0.1)
    assert set(result.assignments["parameter_id"]) == {
        "collection_rate",
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
        "bio_production_rate",
        "daccu_production_rate",
    }


def test_intervention_distance_has_five_direct_equal_families() -> None:
    demand = _demand(DemandLifetimeDecision(50, 25, 2050))
    circularity = build_circularity_trajectories(
        parent=_circularity_parent(),
        parent_identity="circularity::accepted",
        decision=CircularityDecision(80, 40, 10, 30, 20, 70, 20, 10, 2050),
    )
    distance = build_intervention_distance_inputs(
        physical_parent_identity="physical::accepted",
        demand_lifetime_parent_identity="demand::accepted",
        demand_lifetime=demand,
        circularity=circularity,
        parent_circularity=_circularity_parent(),
    )
    assert distance.demand_reduction_fraction == 0.5
    assert distance.lifetime_extension_fraction == 0.25
    assert distance.collection_distance == pytest.approx(
        np.mean([0.8 - (0.4 + index / 100.0) for index in range(12)])
    )
    assert distance.treatment_share_distance == pytest.approx(0.2)
    assert distance.feedstock_share_distance == 0.0


@pytest.mark.parametrize("target_year", (2050, 2075, 2100))
def test_circularity_paths_and_distance_use_the_contemporaneous_parent(
    target_year: int,
) -> None:
    """Follow the varying parent annually and measure its target-year endpoint difference."""
    parent = _circularity_parent()
    # Invented annual increases preserve both simplices and distinguish the
    # parent's endpoint from its first prospective year.
    shift = (parent["year"] - 2020) / 1000.0
    parent["collection_rate"] += shift
    parent["mechanical_recycling_rate"] += shift
    parent["landfill_share"] -= shift
    parent["fossil_share"] -= 2 * shift
    parent["bio_production_rate"] += shift
    parent["daccu_production_rate"] += shift
    decision = CircularityDecision(80, 40, 10, 30, 20, 70, 20, 10, target_year)
    circularity = build_circularity_trajectories(
        parent=parent,
        parent_identity="circularity::varying_parent",
        decision=decision,
    )
    distance = build_intervention_distance_inputs(
        physical_parent_identity="physical::varying_parent",
        demand_lifetime_parent_identity="demand::varying_parent",
        demand_lifetime=_demand(DemandLifetimeDecision(0, 0, target_year)),
        circularity=circularity,
        parent_circularity=parent,
    )
    endpoint_shift = (target_year - 2020) / 1000.0
    assert distance.collection_distance == pytest.approx(
        np.mean([0.8 - (0.4 + index / 100.0 + endpoint_shift) for index in range(12)])
    )
    assert distance.treatment_share_distance == pytest.approx(0.2 - endpoint_shift)
    assert distance.feedstock_share_distance == pytest.approx(2 * endpoint_shift)

    parent_rows = parent.set_index(["year", "region"])
    targets = {
        "collection_rate": 0.8,
        "mechanical_recycling_rate": 0.4,
        "chemical_recycling_rate": 0.1,
        "incineration_rate": 0.3,
        "landfill_share": 0.2,
        "fossil_share": 0.7,
        "bio_production_rate": 0.2,
        "daccu_production_rate": 0.1,
    }
    annual = circularity.annual_values
    pd.testing.assert_frame_equal(annual[["year", "region"]], parent[["year", "region"]])
    assert annual.columns.tolist() == parent.columns.tolist()
    complete = annual.melt(
        id_vars=["year", "region"], var_name="parameter_id", value_name="resolved_value"
    )
    observed = complete.loc[complete["year"].isin((2020, 2040, target_year, 2100))]
    for row in observed.itertuples(index=False):
        initial = parent_rows.loc[(row.year, row.region), row.parameter_id]
        weight = min((row.year - 2019) / (target_year - 2019), 1.0)
        assert row.resolved_value == pytest.approx(
            initial + weight * (targets[row.parameter_id] - initial)
        )
    for members in (
        [
            "mechanical_recycling_rate",
            "chemical_recycling_rate",
            "incineration_rate",
            "landfill_share",
        ],
        ["fossil_share", "bio_production_rate", "daccu_production_rate"],
    ):
        np.testing.assert_allclose(annual[members].sum(axis=1), 1.0, rtol=0, atol=5e-16)
    keys = ["parameter_id", "year", "region"]
    editable = complete.loc[complete.parameter_id.isin(circularity.assignments.parameter_id)]
    pd.testing.assert_series_equal(
        editable.set_index(keys).resolved_value.sort_index(),
        circularity.assignments.set_index(keys).resolved_value.sort_index(),
        check_exact=True,
    )


def test_zero_endpoint_distance_does_not_claim_an_unchanged_annual_path() -> None:
    """A parent reaching the same target later can have different preceding shares."""
    parent = _circularity_parent()
    columns = [
        "collection_rate",
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
        "landfill_share",
        "fossil_share",
        "bio_production_rate",
        "daccu_production_rate",
    ]
    parent.loc[parent["year"].ge(2050), columns] = (0.8, 0.4, 0.1, 0.3, 0.2, 0.7, 0.2, 0.1)
    circularity = build_circularity_trajectories(
        parent=parent,
        parent_identity="circularity::later_parent_change",
        decision=CircularityDecision(80, 40, 10, 30, 20, 70, 20, 10, 2050),
    )
    distance = build_intervention_distance_inputs(
        physical_parent_identity="physical::later_parent_change",
        demand_lifetime_parent_identity="demand::later_parent_change",
        demand_lifetime=_demand(DemandLifetimeDecision(0, 0)),
        circularity=circularity,
        parent_circularity=parent,
    )
    assert distance.collection_distance == 0.0
    assert distance.treatment_share_distance == 0.0
    assert distance.feedstock_share_distance == 0.0
    assert not circularity.uses_exact_parent
    earlier = circularity.assignments.loc[
        circularity.assignments["year"].eq(2040)
        & circularity.assignments["region"].eq("EUR")
        & circularity.assignments["parameter_id"].eq("collection_rate"),
        "resolved_value",
    ].item()
    assert (
        earlier
        > parent.loc[parent["year"].eq(2040) & parent["region"].eq("EUR"), "collection_rate"].item()
    )


@pytest.mark.parametrize("target_year", (2050, 2075, 2100))
def test_demand_repair_pool_is_finite_ordered_and_keeps_other_controls(target_year: int) -> None:
    """Only demand moves on fixed levels, including exact fractional reference values."""
    seed = DemandLifetimeDecision(50, 50, target_year)
    values = list(seed.demand_fraction)
    values[13] += Fraction(1, 10000)
    anchor = replace(seed, demand_fraction=tuple(values))
    pool = demand_repair_neighbors(anchor, failure_year=2100)
    assert len(pool) <= 6 * (target_year - 2020) - 4
    assert len(pool) == len(set(pool))
    assert pool == demand_repair_neighbors(anchor, failure_year=2100)
    assert list(pool) == sorted(
        pool, key=lambda item: (demand_path_departure(item), item.coordinates())
    )
    allowed = (
        {Fraction.from_float(i / 100) for i in range(51)}
        | set(seed.demand_fraction)
        | set(anchor.demand_fraction)
    )
    for path in pool:
        assert path.lifetime_fraction == seed.lifetime_fraction
        assert path.demand_reduction_pct == 50
        assert path.target_year == target_year
        assert tuple(sorted(path.demand_fraction)) == path.demand_fraction
        assert set(path.demand_fraction) <= allowed
        assert path.demand_fraction[0] == 0
        assert path.demand_fraction[target_year - 2019 :] == (Fraction(1, 2),) * (
            2101 - target_year
        )
    assert demand_path_departure(seed) == 0


def test_endpoint_neighbors_discard_previous_repair_shape() -> None:
    """A different endpoint always starts its own exact linear reference."""
    shaped = _shaped_decision(50, 50)
    scalar = endpoint_neighbors(shaped, step_pct=10)
    coordinated = endpoint_neighbors(shaped, step_pct=10, coordinated=True)
    assert len(scalar) == 4
    assert len(coordinated) == 8
    for path in coordinated:
        assert path == DemandLifetimeDecision(
            path.demand_reduction_pct, path.lifetime_extension_pct
        )


def test_fractional_demand_repair_has_nontrivial_fixed_endpoint_paths() -> None:
    """A one-point endpoint still admits absolute levels and exact interior references."""
    seed = DemandLifetimeDecision(1, 0)
    pool = demand_repair_neighbors(seed, failure_year=2022)
    assert pool
    assert any(0 < item.demand_fraction[1] < Fraction.from_float(0.01) for item in pool)
    assert all(item.lifetime_fraction == seed.lifetime_fraction for item in pool)


@pytest.mark.parametrize("plateau", (Fraction(0), Fraction(1, 2)))
def test_demand_repair_pool_respects_zero_and_endpoint_plateaus(plateau: Fraction) -> None:
    """Admissible monotone boundary paths cannot step outside their fixed demand domain."""
    reference = DemandLifetimeDecision(50, 25)
    anchor = replace(
        reference, demand_fraction=(Fraction(0), *((plateau,) * 30), *((Fraction(1, 2),) * 51))
    )
    pool = demand_repair_neighbors(anchor, failure_year=2050)
    assert pool
    for choice in pool:
        assert choice.lifetime_fraction == reference.lifetime_fraction
        assert choice.demand_fraction[0] == 0
        assert choice.demand_fraction[31:] == reference.demand_fraction[31:]
        assert all(
            left <= right for left, right in zip(choice.demand_fraction, choice.demand_fraction[1:])
        )


def test_rejected_stock_equivalence_uses_exact_assignments_and_every_year() -> None:
    """Preserve distinct demand identities for exact fractions with equal binary64 assignments."""
    reference = DemandLifetimeDecision(50, 25, 2100)
    values = list(reference.demand_fraction)
    index = 2059 - 2019
    values[index] = Fraction(1, 4)
    first = replace(reference, demand_fraction=tuple(values))
    values[index] = Fraction.from_float(float(np.nextafter(0.25, np.inf)))
    alias = replace(reference, demand_fraction=tuple(values))
    values[index] = Fraction.from_float(float(np.nextafter(float(values[index]), np.inf)))
    distinct = replace(reference, demand_fraction=tuple(values))

    first_controls = _demand_stock_multipliers(first)
    assert len(first_controls) == 82
    assert first.demand_fraction != alias.demand_fraction
    assert first_controls == _demand_stock_multipliers(alias)
    first_trajectory, alias_trajectory, distinct_trajectory = map(_demand, (first, alias, distinct))
    assert first_trajectory.identity != alias_trajectory.identity
    pd.testing.assert_frame_equal(
        first_trajectory.assignments, alias_trajectory.assignments, check_exact=True
    )
    distinct_controls = _demand_stock_multipliers(distinct)
    assert first_controls[:index] == distinct_controls[:index]
    assert first_controls[index + 1 :] == distinct_controls[index + 1 :]
    assert distinct_controls[index] == np.nextafter(first_controls[index], -np.inf)
    assert not first_trajectory.assignments.equals(distinct_trajectory.assignments)
