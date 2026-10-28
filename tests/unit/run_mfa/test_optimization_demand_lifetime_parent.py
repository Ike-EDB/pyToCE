"""Admitted stock totals and compiled lifetime trajectories used by optimization."""

from fractions import Fraction

import numpy as np
import pandas as pd
from scenario_fixtures import compile_scenario_payload

from pytoce.run_mfa.model.optimization_parents import (
    optimization_demand_lifetime_parent,
)
from pytoce.run_mfa.scenario.catalogue import _catalogue_payload, load_parameter_catalogue
from pytoce.run_mfa.scenario.parent_relative import (
    compile_parent_relative_scenario_assignments,
)
from pytoce.run_mfa.scenario.scenario_contract import (
    ScenarioEvaluation,
)
from pytoce.run_optimization.calculation.interventions import (
    DemandLifetimeDecision,
    build_demand_lifetime_trajectories,
)


def _parent() -> ScenarioEvaluation:
    """Compile population and mean-lifetime changes through the real scenario owner."""
    return compile_scenario_payload(
        {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": "socioeconomic_parent",
            "ssp_scenario": "SSP2",
            "changes": [
                {
                    "parameter": "population",
                    "region": "EUR",
                    "changes": [
                        {
                            "year": 2030,
                            "mode": "multiplier_from_default",
                            "value": 1.01,
                        }
                    ],
                },
                {
                    "parameter": "lifetime_mean",
                    "good": "Packaging",
                    "changes": [
                        {
                            "year": 2040,
                            "mode": "multiplier_from_default",
                            "value": 1.1,
                        }
                    ],
                },
            ],
        }
    )


def _coordinate_sets() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the supported regions and goods for the demand/lifetime projection."""
    stock = next(
        definition
        for definition in load_parameter_catalogue()
        if definition.parameter_id == "stock_projection"
    )
    return stock.coordinates["region"], stock.coordinates["good"]


def _derived_stock() -> pd.DataFrame:
    """Provide complete stock coordinates with positive HDPE/PP and other explicit zeros."""
    regions, goods = _coordinate_sets()
    materials = _catalogue_payload()["coordinate_sets"]["material"]
    table = pd.MultiIndex.from_product(
        [range(2020, 2101), regions, goods, materials],
        names=["year", "region", "good", "material"],
    ).to_frame(index=False)
    base = (
        table["year"]
        - 2019
        + table["region"].map({name: i / 100 for i, name in enumerate(regions)})
        + table["good"].map({name: i / 1000 for i, name in enumerate(goods)})
    )
    table["amount"] = np.where(
        table["material"].eq("HDPE"), base, np.where(table["material"].eq("PP"), base + 1, 0.0)
    )
    return table.assign(parameter_id="plastics_in_use_stock", unit="Mt")


def test_admitted_stock_totals_feed_exact_parent_relative_trajectories() -> None:
    """Use admitted stock totals and preserve the fixed relative lifetime spread."""
    parent = _parent()
    derived = _derived_stock()
    extracted = optimization_demand_lifetime_parent(
        derived.sample(frac=1.0, random_state=7).reset_index(drop=True),
        physical_state_identity="pytoce_mfa_physical_state_schema_v1::1111111111111111111111111111111111111111111111111111111111111111",
        parent=parent,
    )
    regions, goods = _coordinate_sets()

    assert tuple(extracted.stock_targets.columns) == (
        "year",
        "region",
        "good",
        "stock_projection",
    )
    assert tuple(extracted.lifetimes.columns) == (
        "year",
        "good",
        "lifetime_mean",
        "lifetime_std",
    )
    assert len(extracted.stock_targets) == 81 * 12 * 7
    assert len(extracted.lifetimes) == 81 * 7
    assert set(extracted.stock_targets["region"]) == set(regions)
    assert set(extracted.stock_targets["good"]) == set(goods)

    coordinate = extracted.stock_targets.loc[
        extracted.stock_targets["year"].eq(2020)
        & extracted.stock_targets["region"].eq(regions[0])
        & extracted.stock_targets["good"].eq(goods[0]),
        "stock_projection",
    ]
    assert coordinate.item() == 3.0
    compiled_coordinate = parent.compiled_parameters.loc[
        parent.compiled_parameters["parameter_id"].eq("stock_projection")
        & parent.compiled_parameters["year"].eq(2020)
        & parent.compiled_parameters["region"].eq(regions[0])
        & parent.compiled_parameters["good"].eq(goods[0]),
        "resolved_value",
    ]
    assert coordinate.item() != compiled_coordinate.item()
    assert (
        parent.compiled_parameters.loc[
            parent.compiled_parameters["parameter_id"].eq("population")
            & parent.compiled_parameters["region"].eq("EUR"),
            "value_origin",
        ]
        .eq("user_foreground_scenario")
        .any()
    )

    exact_parent = build_demand_lifetime_trajectories(
        parent_stock_targets=extracted.stock_targets,
        parent_lifetimes=extracted.lifetimes,
        parent_stock_target_identity=extracted.stock_target_identity,
        parent_lifetime_identity=extracted.lifetime_identity,
        decision=DemandLifetimeDecision(0, 0),
    )
    assert exact_parent.uses_exact_parent
    assert exact_parent.assignments.empty

    longer = build_demand_lifetime_trajectories(
        parent_stock_targets=extracted.stock_targets,
        parent_lifetimes=extracted.lifetimes,
        parent_stock_target_identity=extracted.stock_target_identity,
        parent_lifetime_identity=extracted.lifetime_identity,
        # This invented lifetime control remains unchanged through 2030, then
        # rises quadratically to a 50% extension in 2050 and holds that value.
        decision=DemandLifetimeDecision(
            0,
            50,
            lifetime_fraction=tuple(
                Fraction(1, 2) * Fraction(min(max(year - 2030, 0), 20), 20) ** 2
                for year in range(2019, 2101)
            ),
        ),
    )
    candidate = compile_parent_relative_scenario_assignments(
        parent, "longer_socioeconomic_parent", longer.assignments
    )
    frame = candidate.compiled_parameters
    means = frame.loc[frame["parameter_id"].eq("lifetime_mean")].set_index(["year", "good"])
    spread = frame.loc[frame["parameter_id"].eq("lifetime_std")].set_index(["year", "good"])
    expected_spread = spread["default_value"] * (means["resolved_value"] / means["default_value"])
    np.testing.assert_array_equal(spread["resolved_value"], expected_spread)
    expected_parent_spread = parent.compiled_parameters.loc[
        parent.compiled_parameters["parameter_id"].eq("lifetime_std")
    ].set_index(["year", "good"])["resolved_value"]
    np.testing.assert_array_equal(spread["reference_value"], expected_parent_spread)
    changed = means["resolved_value"].ne(
        extracted.lifetimes.set_index(["year", "good"])["lifetime_mean"]
    )
    assert spread.loc[changed, "value_origin"].eq("optimization_parent_overlay").all()
    parent_spread = parent.compiled_parameters.loc[
        parent.compiled_parameters["parameter_id"].eq("lifetime_std")
    ].set_index(["year", "good"])
    pd.testing.assert_series_equal(
        spread.loc[~changed, "value_origin"], parent_spread.loc[~changed, "value_origin"]
    )
    np.testing.assert_array_equal(
        spread.loc[changed, "controlling_change_point_year"].to_numpy(dtype=np.int64),
        spread.loc[changed].index.get_level_values("year"),
    )
    np.testing.assert_allclose(
        spread["resolved_value"] / expected_parent_spread,
        means["resolved_value"] / extracted.lifetimes.set_index(["year", "good"])["lifetime_mean"],
        rtol=2e-16,
        atol=0,
    )
    assert candidate.foreground_scenario_identity != parent.foreground_scenario_identity
    repeat = compile_parent_relative_scenario_assignments(
        candidate, "unchanged_longer_parent", longer.assignments
    )
    assert repeat is candidate
    untouched = ~frame["parameter_id"].isin({"stock_projection", "lifetime_mean", "lifetime_std"})
    pd.testing.assert_frame_equal(
        frame.loc[untouched], parent.compiled_parameters.loc[untouched], check_exact=True
    )

    reordered = optimization_demand_lifetime_parent(
        derived,
        physical_state_identity="pytoce_mfa_physical_state_schema_v1::1111111111111111111111111111111111111111111111111111111111111111",
        parent=parent,
    )
    assert reordered.identity == extracted.identity
    pd.testing.assert_frame_equal(
        reordered.stock_targets, extracted.stock_targets, check_exact=True
    )
    pd.testing.assert_frame_equal(reordered.lifetimes, extracted.lifetimes, check_exact=True)
