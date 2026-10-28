"""Mean-lifetime changes preserve each end-use group's original coefficient of variation."""

import numpy as np
import pandas as pd
import pytest
from scenario_fixtures import compile_scenario_payload

from pytoce.run_mfa.model.result_tables import _resolved_input_view
from pytoce.run_mfa.scenario.scenario_contract import ScenarioError


@pytest.mark.parametrize("trajectory", (False, True))
def test_lifetime_mean_derives_spread_for_every_good_and_year(trajectory: bool) -> None:
    """Step changes, resets and linear paths keep the original coefficient of variation."""
    control = (
        {
            "trajectory": {
                "mode": "linear_from_2019",
                "target_year": 2050,
                "target_mode": "multiplier_from_2019",
                "target_value": 1.5,
            }
        }
        if trajectory
        else {
            "changes": [
                {"year": 2030, "mode": "multiplier_from_default", "value": 1.5},
                {"year": 2070, "mode": "retain_default"},
            ]
        }
    )
    evaluation = compile_scenario_payload(
        {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": "longer_use",
            "ssp_scenario": "SSP2",
            "changes": [{"parameter": "lifetime_mean", "selector": "uniform_all_goods", **control}],
        }
    )
    table = evaluation.compiled_parameters
    means = table.loc[table["parameter_id"].eq("lifetime_mean")].set_index(["year", "good"])
    spreads = table.loc[table["parameter_id"].eq("lifetime_std")].set_index(["year", "good"])
    assert len(means) == len(spreads) == 81 * 7
    pd.testing.assert_index_equal(means.index, spreads.index)
    baseline = {
        "Packaging": (0.5, 0.1),
        "Transportation": (13.0, 3.0),
        "Buildings & Construction": (35.0, 7.0),
        "Electrical/Electronics": (8.0, 2.0),
        "Household & Leisure & Sports": (3.0, 1.0),
        "Others": (5.0, 1.5),
        "Textile sector": (5.0, 1.5),
    }
    actual_goods = set(means.index.get_level_values("good"))
    assert actual_goods == set(baseline)
    for good, (mean, std) in baseline.items():
        np.testing.assert_array_equal(means.xs(good, level="good")["default_value"], mean)
        np.testing.assert_array_equal(spreads.xs(good, level="good")["default_value"], std)
    expected = spreads["default_value"] * (means["resolved_value"] / means["default_value"])
    np.testing.assert_array_equal(spreads["resolved_value"], expected)
    assert spreads["parameter_role"].eq("derived_input").all()
    changed = means["resolved_value"].ne(means["default_value"])
    assert changed.any()
    np.testing.assert_array_equal(
        spreads.loc[~changed, "resolved_value"], spreads.loc[~changed, "default_value"]
    )
    assert spreads.loc[changed, "value_mode"].eq("derived_from_mean").all()
    np.testing.assert_array_equal(
        spreads.loc[changed, "value_origin"], means.loc[changed, "value_origin"]
    )
    np.testing.assert_array_equal(
        spreads.loc[changed, "controlling_change_point_year"],
        means.loc[changed, "controlling_change_point_year"],
    )
    assert all(
        point["parameter"] == "lifetime_mean" for point in evaluation.normalized_change_points
    )
    reported = _resolved_input_view(table, evaluation.normalized_change_points, pd.DataFrame())
    reported_spread = reported.loc[reported["parameter_id"].eq("lifetime_std")]
    assert reported_spread["requested_value"].isna().all()
    np.testing.assert_array_equal(reported_spread["resolved_value"], expected)
    np.testing.assert_array_equal(
        reported_spread["ratio_to_default"],
        expected.to_numpy() / spreads["default_value"].to_numpy(),
    )


@pytest.mark.parametrize("trajectory", (False, True))
def test_lifetime_spread_cannot_be_authored_independently(trajectory: bool) -> None:
    """Reject authored lifetime_std changes and direct users to lifetime_mean."""
    control = (
        {
            "trajectory": {
                "mode": "linear_from_2019",
                "target_year": 2050,
                "target_mode": "multiplier_from_2019",
                "target_value": 1.5,
            }
        }
        if trajectory
        else {"changes": [{"year": 2030, "mode": "absolute", "value": 2.0}]}
    )
    with pytest.raises(ScenarioError, match="lifetime_std.*lifetime_mean.*Remove"):
        compile_scenario_payload(
            {
                "schema_version": "pytoce_foreground_scenario_schema_v1",
                "foreground_scenario": "independent_spread",
                "ssp_scenario": "SSP2",
                "changes": [{"parameter": "lifetime_std", "good": "Packaging", **control}],
            }
        )
