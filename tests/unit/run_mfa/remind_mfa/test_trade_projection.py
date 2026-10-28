"""Exact native projection equivalence with reusable historical fit parameters."""

from types import ModuleType

import flodym as fd
import numpy as np
import pytest

from pytoce.run_mfa.remind_mfa.trade_projection import HistoricalTradeProjection


@pytest.mark.parametrize("trade_scale", (0.0, 1e-18, 0.25, 2.0))
@pytest.mark.parametrize("historical_count", (1, 5, 7))
def test_historical_fit_preserves_native_projection_and_candidate_isolation(
    remind_trade_modules: tuple[ModuleType, ModuleType],
    trade_scale: float,
    historical_count: int,
) -> None:
    """Preserve native weights, dimensions, zero/demand tails and detached outputs."""
    _, native = remind_trade_modules
    historic_years = list(range(2020 - historical_count, 2020))
    dimensions = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name="Time", letter="t", items=historic_years + [2020, 2021]),
            fd.Dimension(name="Historical", letter="h", items=historic_years),
            fd.Dimension(name="Region", letter="r", items=["EUR", "USA"]),
            fd.Dimension(name="Material", letter="m", items=["HDPE", "PP"]),
        ]
    )
    historical_predictor = np.arange(1, historical_count * 4 + 1, dtype=float).reshape(
        historical_count, 2, 2
    )
    historic = fd.FlodymArray(
        dims=dimensions[("h", "r", "m")], values=historical_predictor * trade_scale
    )
    original_history = historic.values.copy()
    future_paths = (
        np.array([[[0.0, 1e-18], [8.0, 16.0]], [[0.5, 0.01], [3.0, 20.0]]]),
        np.array([[[10.0, 20.0], [30.0, 40.0]], [[1.0, 20.0], [100.0, 60.0]]]),
    )
    projection = HistoricalTradeProjection(native.ProportionalExtrapolation)
    expected = []
    for future in future_paths:
        scaler = fd.FlodymArray(
            dims=dimensions[("t", "r", "m")],
            values=np.concatenate((historical_predictor, future)),
        )
        native_result = native.extrapolate_to_future(historic, scaler)
        actual = projection.project(("final", "imports"), historic, scaler)
        assert actual.dims == native_result.dims
        assert actual.values.tobytes() == native_result.values.tobytes()
        expected.append((scaler, native_result.values.copy()))
        actual.values.fill(-1.0)
    for scaler, native_values in reversed(expected):
        actual = projection.project(("final", "imports"), historic, scaler)
        assert actual.values.tobytes() == native_values.tobytes()
    np.testing.assert_array_equal(historic.values, original_history)


def test_each_market_and_side_retains_its_own_native_historical_relation(
    remind_trade_modules: tuple[ModuleType, ModuleType],
) -> None:
    """Fit each distinct trade series separately."""
    _, native = remind_trade_modules
    dimensions = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name="Time", letter="t", items=[2018, 2019, 2020]),
            fd.Dimension(name="Historical", letter="h", items=[2018, 2019]),
            fd.Dimension(name="Region", letter="r", items=["EUR", "USA"]),
        ]
    )
    scaler = fd.FlodymArray(
        dims=dimensions[("t", "r")], values=np.array([[2.0, 3.0], [4.0, 5.0], [1.0, 20.0]])
    )
    projection = HistoricalTradeProjection(native.ProportionalExtrapolation)
    for market, share in (("final", 0.1), ("intermediate", 0.2), ("primary", 0.4)):
        for side, factor in (("imports", 1.0), ("exports", 2.0)):
            historic = fd.FlodymArray(
                dims=dimensions[("h", "r")], values=scaler.values[:2] * (share * factor)
            )
            actual = projection.project((market, side), historic, scaler)
            expected = native.extrapolate_to_future(historic, scaler)
            assert actual.values.tobytes() == expected.values.tobytes()
