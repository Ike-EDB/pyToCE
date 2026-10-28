"""Direct source preparation from actual REMIND-MFA APIs and synthetic CSVs."""

import csv
from dataclasses import fields, replace
from hashlib import sha256
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scenario_fixtures import compile_scenario_payload

from pytoce.run_mfa.model.cohorts import accepted_default_cohort_result
from pytoce.run_mfa.remind_mfa.execution import (
    RemindMFAExecutionRequest,
    RemindMFAExecutionSession,
)
from pytoce.run_mfa.remind_mfa.translation import translate_annual_inputs
from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue


def test_source_session_reads_complete_csv_inputs_and_preserves_their_bytes(
    remind_source_request: RemindMFAExecutionRequest,
) -> None:
    """Reach real parsing, historical reconstruction and the default stock target."""
    inputs = remind_source_request.source.data_root
    before = {
        path: sha256(path.read_bytes()).hexdigest() for path in inputs.rglob("*") if path.is_file()
    }
    session = RemindMFAExecutionSession(remind_source_request)
    state = session.cohort_state
    assert state.years == tuple(range(1950, 2101))
    assert state.regions == ("EUR", "USA")
    assert state.materials == ("HDPE", "PP")
    assert state.annual_stock.shape == (151, 2, 7)
    assert np.all(state.annual_stock > 0)
    np.testing.assert_allclose(
        state.annual_inflow - state.annual_outflow,
        np.diff(state.annual_stock, axis=0, prepend=0),
        rtol=1e-12,
        atol=1e-11,
    )
    np.testing.assert_allclose(state.stock_by_cohort.sum(axis=1), state.annual_stock, rtol=1e-13)
    after = {
        path: sha256(path.read_bytes()).hexdigest() for path in inputs.rglob("*") if path.is_file()
    }
    assert after == before


def test_source_session_isolates_sequential_candidate_trade_and_parameters(
    remind_source_request: RemindMFAExecutionRequest,
) -> None:
    """Keep trade driven by supplied cohorts and isolate later default candidates."""
    session = RemindMFAExecutionSession(remind_source_request)
    cohort = accepted_default_cohort_result(session.cohort_state)
    arguments = {
        "carbon_inflow": cohort.carbon_inflow,
        "carbon_outflow": cohort.carbon_outflow,
        "deferred_waste_trade": pd.DataFrame(),
    }
    baseline = session.process_inputs({}, **arguments)
    # Population and GDP affect stock resolution before this boundary. Hold its
    # supplied cohort fixed to isolate direct process-parameter responsibilities.
    changed = session.process_inputs(
        {
            "collection_rate": pd.DataFrame([{"year": 2020, "region": "EUR", "amount": 0.3}]),
            "population": pd.DataFrame([{"year": 2020, "region": "EUR", "amount": 1500.0}]),
            "gdppc": pd.DataFrame([{"year": 2020, "region": "EUR", "amount": 5000.0}]),
        },
        **arguments,
    )
    repeated = session.process_inputs({}, **arguments)
    assert changed.collection_rate[70, 0] == 0.3
    assert baseline.collection_rate[70, 0] != 0.3
    for market in ("final", "intermediate", "primary"):
        for side in ("imports", "exports"):
            np.testing.assert_array_equal(
                getattr(changed, f"{market}_{side}"), getattr(baseline, f"{market}_{side}")
            )
    for item in fields(baseline):
        first = getattr(baseline, item.name)
        second = getattr(repeated, item.name)
        if isinstance(first, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second, check_exact=True)
        else:
            np.testing.assert_array_equal(first, second)
            assert not np.shares_memory(first, second)
    assert baseline.final_imports.shape == (151, 2, 2, 2, 7)
    assert baseline.primary_imports.shape == (151, 2, 2, 2)


def test_socioeconomic_target_changes_preserve_the_admitted_default_stock(
    remind_source_request: RemindMFAExecutionRequest,
) -> None:
    """Resolve new population/GDP inputs and retain the shared cohort state."""
    session = RemindMFAExecutionSession(remind_source_request)
    before = session.cohort_state.annual_stock.copy()
    years = np.arange(2020, 2101)
    time = years - 1950
    coordinates = pd.MultiIndex.from_product([years, ["EUR", "USA"]], names=["year", "region"])
    population = coordinates.to_frame(index=False)
    population["amount"] = ((1000 + time[:, None] * 5) * np.array([1.0, 1.3])).ravel()
    gdppc = coordinates.to_frame(index=False)
    gdppc["amount"] = ((1000 * np.exp(time[:, None] * 0.02)) * np.array([1.0, 1.2])).ravel()
    changed = session.resolve_stock_target(
        {"population": population.assign(amount=population["amount"] * 1.1), "gdppc": gdppc}
    )
    restored = session.resolve_stock_target({"population": population, "gdppc": gdppc})
    assert changed.shape == (81, 2, 7)
    assert np.isfinite(changed).all()
    assert not np.array_equal(changed, restored)
    np.testing.assert_array_equal(session.cohort_state.annual_stock, before)
    np.testing.assert_allclose(restored, before[70:], rtol=1e-13, atol=1e-12)


def test_finite_population_can_overflow_the_derived_stock_target(
    remind_source_request: RemindMFAExecutionRequest,
) -> None:
    """Reject overflowing stock arithmetic after finite scenario input admission."""
    path = remind_source_request.source.data_root / "plastics/input_data/pl_consumption.cs4r"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = list(csv.reader(lines[1:]))
    # Increase only the invented historical consumption (Mt) so that stock per
    # capita exceeds one; multiplying it by a finite float maximum can overflow.
    for row in rows:
        row[-1] = str(float(row[-1]) * 100)
    output = StringIO()
    output.write(lines[0] + "\n")
    csv.writer(output).writerows(rows)
    path.write_text(output.getvalue(), encoding="utf-8")
    session = RemindMFAExecutionSession(remind_source_request)
    before = session.cohort_state.annual_stock.copy()
    # The fixture's 2020 EUR population is 1000 + (2020 - 1950) * 5.
    assert before[70, 0].max() / 1350 > 1
    compiled = compile_scenario_payload(
        {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": "finite_population_overflow",
            "ssp_scenario": "SSP2",
            "changes": [
                {
                    "parameter": "population",
                    "region": "EUR",
                    "changes": [
                        {"year": 2020, "mode": "absolute", "value": float(np.finfo(float).max)}
                    ],
                }
            ],
        }
    ).compiled_parameters
    population_rows = compiled.loc[
        compiled["parameter_id"].eq("population") & compiled["region"].eq("EUR")
    ]
    assert np.isfinite(population_rows["resolved_value"]).all()
    updates = translate_annual_inputs(population_rows, load_parameter_catalogue()).numeric_updates
    years = np.arange(2020, 2101)
    gdppc = pd.MultiIndex.from_product([years, ["EUR", "USA"]], names=["year", "region"]).to_frame(
        index=False
    )
    # Retain the source fixture's finite GDP per capita in both regions.
    gdppc["amount"] = (
        (1000 * np.exp((years[:, None] - 1950) * 0.02)) * np.array([1.0, 1.2])
    ).ravel()
    updates["gdppc"] = gdppc
    with pytest.raises(ValueError, match="nonfinite.*stock target in Mt.*Reduce"):
        session.resolve_stock_target(updates)
    np.testing.assert_array_equal(session.cohort_state.annual_stock, before)


@pytest.mark.parametrize("trade", ("primary", "intermediate", "manufactured", "final"))
@pytest.mark.parametrize("negative_direction", ("imports", "exports"))
def test_source_session_rejects_negative_gross_historical_trade(
    remind_source_request: RemindMFAExecutionRequest, trade: str, negative_direction: str
) -> None:
    """Positive global trade totals cannot legitimize a negative regional gross flow."""
    before = {}
    for direction in ("imports", "exports"):
        path = (
            remind_source_request.source.data_root
            / "plastics/input_data"
            / f"pl_{trade}_his_{direction}.cs4r"
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = list(csv.reader(lines[1:]))
        values = (
            {"EUR": -0.01, "USA": 0.02}
            if direction == negative_direction
            else {"EUR": 0.005, "USA": 0.005}
        )
        for row in rows:
            if row[0] == "2019":
                row[-1] = str(values[row[1]])
        output = StringIO()
        output.write(lines[0] + "\n")
        csv.writer(output).writerows(rows)
        path.write_text(output.getvalue(), encoding="utf-8")
        before[path] = path.read_bytes()
    with pytest.raises(ValueError, match=f"{trade}_his_{negative_direction}.*nonnegative") as error:
        RemindMFAExecutionSession(remind_source_request)
    assert "2019" in str(error.value)
    assert "EUR" in str(error.value)
    assert {path: path.read_bytes() for path in before} == before


def test_source_session_rejects_unsupported_capture_in_external_input(
    remind_source_request: RemindMFAExecutionRequest,
) -> None:
    """Reject changed source adoption data before calculating stock or flows."""
    source = remind_source_request.source.data_root
    path = source / "plastics" / "input_data" / "pl_emission_capture_rate.cs4r"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = list(csv.reader(lines[1:]))
    rows[0][-1] = "0.1"
    output = StringIO()
    output.write(lines[0] + "\n")
    csv.writer(output).writerows(rows)
    path.write_text(output.getvalue(), encoding="utf-8")
    with pytest.raises(ValueError, match="emission_capture_rate.*must be zero") as error:
        RemindMFAExecutionSession(remind_source_request)
    assert str(source) in str(error.value)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"collection_rate": -0.1}, "collection_rate.*fraction in"),
        ({"collection_rate": 1.01}, "collection_rate.*fraction in"),
        ({"bio_production_rate": 0.8, "daccu_production_rate": 0.3}, "sum must not exceed 1"),
        ({"incineration_rate": float("nan")}, "incineration_rate.*NaN"),
        ({"incineration_rate": float("inf")}, "incineration_rate.*finite"),
        ({"incineration_rate": -float("inf")}, "incineration_rate.*finite"),
        ({"lifetime_mean": float("inf")}, "lifetime_mean.*finite"),
        ({"lifetime_mean": -float("inf")}, "lifetime_mean.*finite"),
    ),
)
def test_source_csv_requires_finite_values_and_consistent_process_fractions(
    remind_source_request: RemindMFAExecutionRequest,
    changes: dict[str, float],
    message: str,
) -> None:
    """Admit source CSV values before stock reconstruction and processing."""
    for name, value in changes.items():
        path = (
            remind_source_request.source.data_root / "plastics" / "input_data" / f"pl_{name}.cs4r"
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = list(csv.reader(lines[1:]))
        for row in rows:
            row[-1] = str(value)
        output = StringIO()
        output.write(lines[0] + "\n")
        csv.writer(output).writerows(rows)
        path.write_text(output.getvalue(), encoding="utf-8")
    inputs = remind_source_request.source.data_root
    before = {path: path.read_bytes() for path in inputs.rglob("*.cs4r")}
    with pytest.raises(ValueError, match=message):
        session = RemindMFAExecutionSession(remind_source_request)
        cohort = accepted_default_cohort_result(session.cohort_state)
        session.process_inputs(
            {},
            carbon_inflow=cohort.carbon_inflow,
            carbon_outflow=cohort.carbon_outflow,
            deferred_waste_trade=pd.DataFrame(),
        )
    assert {path: path.read_bytes() for path in inputs.rglob("*.cs4r")} == before


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("collection_rate_target_year", float("inf")),
        ("collection_rate_target_year", float("nan")),
        ("collection_rate_target_value", -float("inf")),
    ),
)
def test_source_scenario_requires_finite_extrapolation_targets(
    remind_source_request: RemindMFAExecutionRequest,
    tmp_path: Path,
    parameter: str,
    value: float,
) -> None:
    """Reject nonfinite YAML targets even when extrapolation masks them as finite rates."""
    original = remind_source_request.source.code_root / "config/scenarios/SSP2.yml"
    original_bytes = original.read_bytes()
    scenario = yaml.safe_load(original_bytes)
    next(item for item in scenario["data"] if item["parameter"] == parameter)["value"] = value
    directory = tmp_path / "scenario_inputs"
    directory.mkdir()
    path = directory / "SSP2.yml"
    path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    configuration = yaml.safe_load(remind_source_request.configuration_yaml)
    configuration["input"]["scenarios_path"] = str(directory)
    request = replace(remind_source_request, configuration_yaml=yaml.safe_dump(configuration))
    before = path.read_bytes()
    with pytest.raises(ValueError, match=parameter + ".*finite") as error:
        RemindMFAExecutionSession(request)
    assert str(directory) in str(error.value)
    assert path.read_bytes() == before
    assert original.read_bytes() == original_bytes
