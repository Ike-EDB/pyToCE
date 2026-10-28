"""Trade preparation through the public REMIND-MFA 0.2.0 APIs."""

from pathlib import Path
from types import ModuleType

import flodym as fd
import numpy as np
import pandas as pd
import pytest

from pytoce.run_mfa.remind_mfa.process_inputs import calculate_process_inputs
from pytoce.run_mfa.remind_mfa.trade_projection import HistoricalTradeProjection


@pytest.mark.parametrize("market", ("final", "intermediate", "manufactured", "primary"))
@pytest.mark.parametrize("declining_demand", (0.0, 0.01))
def test_import_capacity_limits_live_trade_after_native_balancing(
    remind_trade_modules: tuple[ModuleType, ModuleType], market: str, declining_demand: float
) -> None:
    """Keep imports within actual demand at every native trade stage after a decline."""
    trade_api, extrapolation_api = remind_trade_modules
    dims = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name=name, letter=letter, items=items)
            for name, letter, items in (
                ("Time", "t", [2019, 2020]),
                ("Historic Time", "h", [2019]),
                ("Region", "r", ["EUR", "USA"]),
                ("Material", "m", ["HDPE"]),
                ("Element", "e", ["C"]),
                ("Good", "g", ["Packaging"]),
            )
        ]
    )
    # Invented Mt inputs: both regions demand 1 Mt in history; EUR falls to
    # zero or 0.01 Mt. The selected market trades 0.5 Mt before the decline.
    demand = np.array([[1.0, 1.0], [declining_demand, 1.0]])
    inflow = fd.FlodymArray(
        dims=dims[("t", "e", "r", "m", "g")],
        values=demand[:, None, :, None, None],
    )
    historical = {
        name + "_his": trade_api.Trade(
            imports=fd.FlodymArray(
                dims=dims[("h", "r")], values=np.full((1, 2), 0.5 if name == market else 0.0)
            ),
            exports=fd.FlodymArray(
                dims=dims[("h", "r")], values=np.full((1, 2), 0.5 if name == market else 0.0)
            ),
        )
        for name in ("final", "intermediate", "manufactured", "primary")
    }
    parameters = {
        name: fd.Parameter(dims=dims[("t", "r")], values=np.zeros((2, 2)))
        for name in (
            "collection_rate",
            "mechanical_recycling_rate",
            "chemical_recycling_rate",
            "incineration_rate",
            "bio_production_rate",
            "daccu_production_rate",
            "waste_imports",
            "waste_exports",
        )
    }
    actual = calculate_process_inputs(
        inflow=inflow,
        # Nonzero discarded material defines the native waste-composition
        # shares; waste trade itself is zero in this capacity-only example.
        outflow=fd.FlodymArray(dims=inflow.dims, values=np.full(inflow.shape, 0.2)),
        parameters=parameters,
        historical_trade=historical,
        dimensions=dims,
        trade_api=trade_api,
        trade_projection=HistoricalTradeProjection(extrapolation_api.ProportionalExtrapolation),
        deferred_waste_trade=pd.DataFrame(),
        default_propensities=pd.DataFrame(),
    )
    # Manufactured and intermediate goods share the exported pooled table;
    # only the parameterized native market is nonzero in each case.
    output = "intermediate" if market == "manufactured" else market
    imports = getattr(actual, output + "_imports").reshape(2, 2)
    exports = getattr(actual, output + "_exports").reshape(2, 2)
    assert imports[1, 0] == demand[1, 0]
    assert np.all((imports >= 0.0) & (imports <= demand))
    assert np.all(exports >= 0.0)
    np.testing.assert_allclose(imports.sum(axis=1), exports.sum(axis=1), rtol=2e-15, atol=0.0)
    np.testing.assert_array_equal(imports[0], [0.5, 0.5])
    np.testing.assert_array_equal(exports[0], [0.5, 0.5])


def test_finite_csv_trade_inputs_can_overflow_during_projection(
    tmp_path: Path, remind_trade_modules: tuple[ModuleType, ModuleType]
) -> None:
    """Reject a nonfinite forecast even when every admitted input is finite."""
    trade_api, extrapolation_api = remind_trade_modules
    dims = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name=name, letter=letter, items=items)
            for name, letter, items in (
                ("Time", "t", [2019, 2020]),
                ("Historic Time", "h", [2019]),
                ("Region", "r", ["EUR", "USA"]),
                ("Material", "m", ["HDPE"]),
                ("Element", "e", ["C"]),
                ("Good", "g", ["Packaging"]),
            )
        ]
    )
    # Invented Mt inputs fit primary trade at 100 times historical demand.
    # Its product with finite future demand 1e307 exceeds binary64 range
    # before REMIND-MFA blends the forecast with historical trade.
    tables = {
        "demand": "Time,Region,value\n2019,EUR,1\n2019,USA,1\n2020,EUR,1e307\n2020,USA,1e307\n",
        "primary_trade": "Historic Time,Region,value\n2019,EUR,100\n2019,USA,100\n",
        "zero_trade": "Historic Time,Region,value\n2019,EUR,0\n2019,USA,0\n",
    }
    paths = {name: tmp_path / f"{name}.csv" for name in tables}
    for name, content in tables.items():
        paths[name].write_text(content, encoding="utf-8")
    reader = fd.CSVParameterReader({name: str(path) for name, path in paths.items()})
    inflow = reader.read_parameter_values("demand", dims[("t", "e", "r", "m", "g")])
    historical = {
        market + "_his": trade_api.Trade(
            **{
                direction: reader.read_parameter_values(
                    "primary_trade" if market == "primary" else "zero_trade", dims[("h", "r")]
                )
                for direction in ("imports", "exports")
            }
        )
        for market in ("final", "intermediate", "manufactured", "primary")
    }
    parameters = {
        name: fd.Parameter(dims=dims[("t", "r")], values=np.zeros((2, 2)))
        for name in (
            "collection_rate",
            "mechanical_recycling_rate",
            "chemical_recycling_rate",
            "incineration_rate",
            "bio_production_rate",
            "daccu_production_rate",
            "waste_imports",
            "waste_exports",
        )
    }
    # A nonzero outflow gives defined waste shares; zero rates and waste trade
    # keep those calculations independent of the primary-trade overflow.
    outflow = inflow * 0.2
    for array in (
        inflow,
        outflow,
        *parameters.values(),
        *(
            getattr(trade, direction)
            for trade in historical.values()
            for direction in ("imports", "exports")
        ),
    ):
        assert np.isfinite(array.values).all()
    with pytest.raises(ValueError, match="trade or process input contains a nonfinite value"):
        calculate_process_inputs(
            inflow=inflow,
            outflow=outflow,
            parameters=parameters,
            historical_trade=historical,
            dimensions=dims,
            trade_api=trade_api,
            trade_projection=HistoricalTradeProjection(extrapolation_api.ProportionalExtrapolation),
            deferred_waste_trade=pd.DataFrame(),
            default_propensities=pd.DataFrame(),
        )


def test_trade_sequence_preserves_history_balance_and_candidate_isolation(
    remind_trade_modules: tuple[ModuleType, ModuleType],
) -> None:
    """Keep all source trade stages and their actual candidate-dependent scalers."""
    trade_api, extrapolation_api = remind_trade_modules
    dims = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name="Time", letter="t", items=list(range(2015, 2022))),
            fd.Dimension(name="Historic Time", letter="h", items=list(range(2015, 2020))),
            fd.Dimension(name="Region", letter="r", items=["EUR", "USA"]),
            fd.Dimension(name="Material", letter="m", items=["HDPE", "PP"]),
            fd.Dimension(name="Element", letter="e", items=["C", "Other Elements"]),
            fd.Dimension(name="Good", letter="g", items=["Packaging", "Transportation"]),
        ]
    )
    split = np.einsum(
        "m,em,g->emg",
        np.array([0.4, 0.6]),
        np.array([[0.8, 0.85], [0.2, 0.15]]),
        np.array([0.3, 0.7]),
    )
    annual = np.broadcast_to(np.array([10.0, 20.0]), (7, 2))
    inflow = fd.FlodymArray(
        dims=dims[("t", "e", "r", "m", "g")], values=np.einsum("tr,emg->termg", annual, split)
    )
    outflow = inflow * 0.4
    parameters = {
        name: fd.Parameter(dims=dims[("t", "r")], values=np.full((7, 2), value))
        for name, value in {
            "collection_rate": 0.6,
            "mechanical_recycling_rate": 0.2,
            "chemical_recycling_rate": 0.1,
            "incineration_rate": 0.3,
            "bio_production_rate": 0.05,
            "daccu_production_rate": 0.1,
            "emission_capture_rate": 0.0,
        }.items()
    }
    for direction, amounts in (("imports", [0.24, 0.48]), ("exports", [0.48, 0.24])):
        parameters["waste_" + direction] = fd.Parameter(
            dims=dims[("t", "r")], values=np.broadcast_to(amounts, (7, 2)).copy()
        )
    historical = {}
    for name, scale in (
        ("final", 1.0),
        ("intermediate", 0.3),
        ("manufactured", 0.2),
        ("primary", 0.1),
    ):
        trade = trade_api.Trade(
            imports=fd.FlodymArray(
                dims=dims[("h", "r")],
                values=np.broadcast_to(np.array([1.0, 2.0]) * scale, (5, 2)).copy(),
            ),
            exports=fd.FlodymArray(
                dims=dims[("h", "r")],
                values=np.broadcast_to(np.array([2.0, 1.0]) * scale, (5, 2)).copy(),
            ),
        )
        trade.balance(to="maximum")
        historical[name + "_his"] = trade
    snapshots = {
        (name, direction): getattr(trade, direction).values.copy()
        for name, trade in historical.items()
        for direction in ("imports", "exports")
    }
    arguments = {
        "outflow": outflow,
        "parameters": parameters,
        "historical_trade": historical,
        "dimensions": dims,
        "trade_api": trade_api,
        "trade_projection": HistoricalTradeProjection(extrapolation_api.ProportionalExtrapolation),
        "deferred_waste_trade": pd.DataFrame(),
        "default_propensities": pd.DataFrame(),
    }
    baseline = calculate_process_inputs(inflow=inflow, **arguments)
    for name, imports, exports in (
        ("final", [1.0, 2.0], [2.0, 1.0]),
        ("intermediate", [0.5, 1.0], [1.0, 0.5]),
        ("primary", [0.1, 0.2], [0.2, 0.1]),
        ("waste", [0.24, 0.48], [0.48, 0.24]),
    ):
        imported = getattr(baseline, name + "_imports")
        exported = getattr(baseline, name + "_exports")
        axes = (1, 3, 4) if imported.ndim == 5 else (1, 3)
        np.testing.assert_allclose(
            imported.sum(axis=axes), np.broadcast_to(imports, (7, 2)), rtol=1e-13, atol=1e-14
        )
        np.testing.assert_allclose(
            exported.sum(axis=axes), np.broadcast_to(exports, (7, 2)), rtol=1e-13, atol=1e-14
        )
        np.testing.assert_allclose(
            imported.sum(axis=2), exported.sum(axis=2), rtol=1e-13, atol=1e-14
        )
    candidate_inflow = fd.FlodymArray(dims=inflow.dims, values=inflow.values.copy())
    candidate_inflow.values[5:] *= 1.3
    changed = calculate_process_inputs(inflow=candidate_inflow, **arguments)
    repeated = calculate_process_inputs(inflow=inflow, **arguments)
    np.testing.assert_array_equal(changed.final_imports[:5], baseline.final_imports[:5])
    assert not np.array_equal(changed.final_imports[5:], baseline.final_imports[5:])
    np.testing.assert_array_equal(repeated.final_imports, baseline.final_imports)
    for (name, direction), expected in snapshots.items():
        np.testing.assert_array_equal(getattr(historical[name], direction).values, expected)
