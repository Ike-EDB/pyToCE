from pathlib import Path

import flodym as fd
import numpy as np
import pandas as pd
import pytest

from pytoce.run_mfa.remind_mfa.translation import (
    normalize_cohort_state,
    translate_annual_inputs,
)
from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue
from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.set_workspace.set_workspace import set_workspace


def test_translation_maps_only_supported_future_upstream_arrays(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    set_workspace(root)
    compiled = compile_foreground_scenario(root, "remind_mfa_default")
    table = pd.read_parquet(compiled.compiled_parameters_path)
    catalogue = load_parameter_catalogue()
    assert isinstance(catalogue, tuple)

    translated = translate_annual_inputs(table, catalogue)
    assert {
        "collection_rate",
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
        "waste_imports",
        "waste_exports",
        "bio_production_rate",
        "daccu_production_rate",
        "population",
        "gdppc",
    } == set(translated.numeric_updates)
    assert {"lifetime_mean", "lifetime_std", "material_shares_in_goods"}.isdisjoint(
        translated.numeric_updates
    )
    assert translated.numeric_updates["collection_rate"]["upstream_dimensions"].eq("t|r").all()
    assert len(translated.numeric_updates["collection_rate"]) == 81 * 12
    assert translated.deferred_waste_trade.empty


def test_translation_defers_baseline_default_propensity_until_collection_is_known(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    set_workspace(root)
    compiled = compile_foreground_scenario(
        root,
        "frozen_2019_default_baseline",
    )
    table = pd.read_parquet(compiled.compiled_parameters_path)
    catalogue = load_parameter_catalogue()
    assert isinstance(catalogue, tuple)

    translated = translate_annual_inputs(table, catalogue)

    assert {"waste_imports", "waste_exports"}.isdisjoint(translated.numeric_updates)
    assert set(translated.deferred_waste_trade["parameter_id"]) == {
        "waste_imports",
        "waste_exports",
    }
    assert len(translated.deferred_waste_trade) == 81 * 12 * 2
    assert set(translated.deferred_waste_trade["value_mode"]) == {
        "retain_default_propensity_pretrade_collected"
    }
    assert translated.deferred_waste_trade["deferred_propensity"].isna().all()


def _stock_source(start_year: int = 1950) -> tuple[fd.StockDrivenDSM, dict[str, fd.Parameter]]:
    """Create a valid default stock reconstruction with material/lifetime inputs."""
    dims = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name="Time", letter="t", items=list(range(start_year, 2101))),
            fd.Dimension(name="Region", letter="r", items=["EUR", "USA"]),
            fd.Dimension(name="Good", letter="g", items=["Packaging", "Transportation"]),
            fd.Dimension(name="Material", letter="m", items=["HDPE", "PP"]),
            fd.Dimension(name="Element", letter="e", items=["C", "Other Elements"]),
        ]
    )
    parameters = {
        "lifetime_mean": fd.Parameter(dims=dims[("g",)], values=np.array([3.0, 13.0])),
        "lifetime_std": fd.Parameter(dims=dims[("g",)], values=np.array([1.0, 3.0])),
        "material_shares_in_goods": fd.Parameter(
            dims=dims[("r", "m", "g")],
            values=np.broadcast_to(np.array([[0.8, 0.2], [0.2, 0.8]]), (2, 2, 2)).copy(),
        ),
        "carbon_content_materials": fd.Parameter(
            dims=dims[("e", "m")], values=np.array([[0.8, 0.85], [0.2, 0.15]])
        ),
    }
    lifetime = fd.LogNormalLifetime(dims=dims[("t", "r", "g")])
    lifetime.set_prms(mean=parameters["lifetime_mean"], std=parameters["lifetime_std"])
    stock = fd.StockDrivenDSM(dims=dims[("t", "r", "g")], lifetime_model=lifetime)
    stock.stock.values[...] = (1 + np.arange(2101 - start_year)[:, None, None]) * np.array(
        [[1.0, 2.0], [3.0, 4.0]]
    )
    stock.compute()
    return stock, parameters


def test_stock_admission_preserves_cohorts_axes_units_and_detaches_source_values() -> None:
    stock, parameters = _stock_source()
    result = normalize_cohort_state(stock, parameters)
    assert result.years == tuple(range(1950, 2101))
    assert result.regions == ("EUR", "USA")
    assert result.goods == ("Packaging", "Transportation")
    assert result.materials == ("HDPE", "PP")
    assert result.elements == ("C", "Other Elements")
    for actual, expected in (
        (result.annual_stock, stock.stock.values),
        (result.annual_inflow, stock.inflow.values),
        (result.annual_outflow, stock.outflow.values),
        (result.stock_by_cohort, stock.get_stock_by_cohort()),
        (result.outflow_by_cohort, stock.get_outflow_by_cohort()),
        (result.material_shares, parameters["material_shares_in_goods"].values),
        (result.carbon_content, parameters["carbon_content_materials"].values),
        (result.lifetime_mean, parameters["lifetime_mean"].values),
        (result.lifetime_std, parameters["lifetime_std"].values),
    ):
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)
    np.testing.assert_allclose(result.stock_by_cohort.sum(axis=1), result.annual_stock, rtol=1e-14)
    np.testing.assert_allclose(
        result.annual_inflow - result.annual_outflow,
        np.diff(result.annual_stock, axis=0, prepend=0),
        rtol=1e-13,
        atol=1e-12,
    )


def test_stock_admission_rejects_nonfinite_source_parameters() -> None:
    stock, parameters = _stock_source()
    parameters["carbon_content_materials"].values[0, 0] = np.nan
    with pytest.raises(ValueError, match="carbon content contains nonfinite"):
        normalize_cohort_state(stock, parameters)


def test_stock_admission_requires_the_complete_pinned_year_axis() -> None:
    """Reject a real stock reconstruction that omits the initial 1950 boundary."""
    stock, parameters = _stock_source(start_year=1951)
    with pytest.raises(ValueError, match="exactly 1950 through 2100"):
        normalize_cohort_state(stock, parameters)


@pytest.mark.parametrize("name", ["material_shares_in_goods", "carbon_content_materials"])
@pytest.mark.parametrize("failure", ["negative", "above_one", "incomplete"])
def test_stock_admission_rejects_invalid_source_partitions(name: str, failure: str) -> None:
    """Admit source fractions themselves even where the affected stock is zero."""
    stock, parameters = _stock_source()
    values = parameters[name].values
    index = (0,) * values.ndim
    values[index] = {"negative": -0.1, "above_one": 1.1, "incomplete": 0.0}[failure]
    stock.stock.values[...] = 0.0
    stock.compute()
    with pytest.raises(ValueError, match=r"fractions in \[0, 1\] summing to one"):
        normalize_cohort_state(stock, parameters)
