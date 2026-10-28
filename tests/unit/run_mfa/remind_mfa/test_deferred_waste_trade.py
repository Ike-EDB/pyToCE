from types import ModuleType

import flodym as fd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest

from pytoce.run_mfa.remind_mfa.process_inputs import (
    WASTE_TRADE_LEDGER_COLUMNS,
    _resolved_waste_trade,
    default_waste_trade_propensities,
)
from pytoce.run_mfa.remind_mfa.translation import apply_parameter_update


def _trade_inputs() -> tuple[fd.DimensionSet, fd.FlodymArray, dict[str, fd.Parameter]]:
    dims = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name="Time", letter="t", items=[2020, 2021]),
            fd.Dimension(name="Element", letter="e", items=["C"]),
            fd.Dimension(name="Region", letter="r", items=["A", "B"]),
            fd.Dimension(name="Material", letter="m", items=["polymer"]),
            fd.Dimension(name="Good", letter="g", items=["product"]),
        ]
    )
    outflow = fd.FlodymArray(
        dims=dims[("t", "e", "r", "m", "g")],
        values=np.array([[[[[10.0]], [[20.0]]]], [[[[15.0]], [[25.0]]]]]),
        name="in_use_outflow",
    )
    collection_rate = fd.Parameter(
        dims=dims[("t", "r")],
        values=np.array([[0.5, 0.5], [0.8, 0.4]]),
        name="collection_rate",
    )
    waste_imports = fd.Parameter(dims=dims[("t", "r")], name="waste_imports")
    waste_exports = fd.Parameter(dims=dims[("t", "r")], name="waste_exports")
    return (
        dims,
        outflow,
        {
            "collection_rate": collection_rate,
            "waste_imports": waste_imports,
            "waste_exports": waste_exports,
        },
    )


def test_parameter_update_assigns_exact_coordinates_in_one_array_operation() -> None:
    parameter = _trade_inputs()[2]["collection_rate"]
    update = pd.DataFrame(
        [
            {"year": 2021, "region": "B", "amount": 0.11},
            {"year": 2020, "region": "A", "amount": 0.22},
            {"year": 2021, "region": "A", "amount": 0.33},
            {"year": 2020, "region": "B", "amount": 0.44},
        ]
    )

    apply_parameter_update(parameter, update)

    np.testing.assert_array_equal(
        parameter.values,
        np.asarray([[0.22, 0.44], [0.33, 0.11]]),
    )
    invalid = update.iloc[[0]].assign(region="foreign")
    with pytest.raises(ValueError, match="unavailable region='foreign'"):
        apply_parameter_update(parameter, invalid)


def test_deferred_trade_uses_current_collection_then_upstream_balance(
    remind_trade_modules: tuple[ModuleType, ModuleType],
) -> None:
    dims, outflow, parameters = _trade_inputs()
    deferred = pd.DataFrame(
        [
            {
                "parameter_id": parameter,
                "year": year,
                "region": region,
                "deferred_propensity": propensity,
                "value_mode": "retain_2019_propensity_pretrade_collected",
            }
            for parameter, propensities in (
                ("waste_imports", {"A": 0.1, "B": 0.2}),
                ("waste_exports", {"A": 0.2, "B": 0.05}),
            )
            for year in (2020, 2021)
            for region, propensity in propensities.items()
        ]
    )
    default_propensities = deferred.loc[
        :, ["parameter_id", "year", "region", "deferred_propensity"]
    ].rename(columns={"deferred_propensity": "default_propensity"})
    default_propensities["default_propensity_authority_identity"] = "default::fixture"

    trade, ledger = _resolved_waste_trade(
        outflow, parameters, dims, remind_trade_modules[0], deferred, default_propensities
    )

    assert ledger["local_pretrade_collected_mt"].tolist() == [5.0, 10.0, 12.0, 10.0]
    np.testing.assert_allclose(
        ledger["source_input_waste_imports_mt"],
        [0.5, 2.0, 1.2, 2.0],
        rtol=1e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        ledger["source_input_waste_exports_mt"],
        [1.0, 0.5, 2.4, 0.5],
        rtol=1e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(ledger["import_formula_residual_mt"], 0.0, atol=1e-15)
    np.testing.assert_allclose(ledger["export_formula_residual_mt"], 0.0, atol=1e-15)
    np.testing.assert_allclose(
        ledger["global_trade_balance_residual_mt"],
        0.0,
        atol=1e-15,
    )
    assert ledger["source_input_waste_imports_mt"].nunique() > 1
    assert ledger["source_input_waste_exports_mt"].nunique() > 1
    assert (ledger["available_collected_after_trade_mt"] >= 0.0).all()
    assert set(ledger["import_propensity_basis"]) == {"accepted_2019_pretrade_collection"}
    compact_dims, compact_outflow, compact_parameters = _trade_inputs()
    compact, compact_ledger = _resolved_waste_trade(
        compact_outflow,
        compact_parameters,
        compact_dims,
        remind_trade_modules[0],
        deferred,
        default_propensities,
        record_diagnostics=False,
    )
    np.testing.assert_array_equal(compact.imports.values, trade.imports.values)
    np.testing.assert_array_equal(compact.exports.values, trade.exports.values)
    assert compact_ledger.empty


def test_empty_deferred_trade_uses_the_remind_mfa_baseline(
    remind_trade_modules: tuple[ModuleType, ModuleType],
) -> None:
    dims, outflow, parameters = _trade_inputs()

    parameters["waste_imports"].values[...] = [[1.0, 2.0], [2.0, 3.0]]
    parameters["waste_exports"].values[...] = [[2.0, 1.0], [3.0, 2.0]]
    trade, ledger = _resolved_waste_trade(
        outflow, parameters, dims, remind_trade_modules[0], pd.DataFrame(), pd.DataFrame()
    )
    np.testing.assert_array_equal(
        trade.imports.sum_over(("g", "e", "m")).values, [[1.0, 2.0], [2.0, 3.0]]
    )
    np.testing.assert_array_equal(
        trade.exports.sum_over(("g", "e", "m")).values, [[2.0, 1.0], [3.0, 2.0]]
    )

    assert ledger.empty
    assert tuple(ledger.columns) == WASTE_TRADE_LEDGER_COLUMNS


def test_deferred_trade_uses_contemporaneous_default_propensity(
    remind_trade_modules: tuple[ModuleType, ModuleType],
) -> None:
    dims, outflow, parameters = _trade_inputs()
    rows = [
        {
            "parameter_id": parameter,
            "year": year,
            "region": region,
            "deferred_propensity": np.nan,
            "value_mode": "retain_default_propensity_pretrade_collected",
        }
        for parameter in ("waste_imports", "waste_exports")
        for year in (2020, 2021)
        for region in ("A", "B")
    ]
    deferred = pd.DataFrame(rows)
    default_propensities = deferred.loc[:, ["parameter_id", "year", "region"]].copy()
    default_propensities["default_propensity"] = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.2,
        0.05,
        0.1,
        0.15,
    ]
    default_propensities["default_propensity_authority_identity"] = "default::fixture"

    trade, ledger = _resolved_waste_trade(
        outflow, parameters, dims, remind_trade_modules[0], deferred, default_propensities
    )

    np.testing.assert_allclose(
        ledger["source_input_waste_imports_mt"],
        [0.5, 2.0, 3.6, 4.0],
        rtol=1e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        ledger["source_input_waste_exports_mt"],
        [1.0, 0.5, 1.2, 1.5],
        rtol=1e-15,
        atol=0.0,
    )
    assert set(ledger["import_propensity_basis"]) == {"contemporaneous_remind_mfa_default"}
    assert set(ledger["export_propensity_authority_identity"]) == {"default::fixture"}


def test_default_trade_propensity_is_derived_from_exact_annual_model_inputs() -> None:
    dims, outflow, parameters = _trade_inputs()
    local = np.array([[5.0, 10.0], [12.0, 10.0]])
    import_propensity = np.array([[0.1, 0.2], [0.3, 0.4]])
    export_propensity = np.array([[0.2, 0.05], [0.1, 0.15]])
    parameters["waste_imports"].values[...] = local * import_propensity
    parameters["waste_exports"].values[...] = local * export_propensity

    table = default_waste_trade_propensities(outflow, parameters)

    imports = table.loc[table["parameter_id"].eq("waste_imports")]
    exports = table.loc[table["parameter_id"].eq("waste_exports")]
    np.testing.assert_allclose(imports["default_propensity"], import_propensity.ravel())
    np.testing.assert_allclose(exports["default_propensity"], export_propensity.ravel())
    assert table["default_propensity_authority_identity"].nunique() == 1
    assert (
        table["default_propensity_authority_identity"]
        .iloc[0]
        .startswith("pytoce_remind_mfa_default_waste_trade_propensity_v1::")
    )
