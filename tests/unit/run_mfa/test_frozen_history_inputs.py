from pathlib import Path

import pandas as pd
import pytest

from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.run_mfa.scenario.scenario_contract import (
    ScenarioConflictError,
    ScenarioError,
)
from pytoce.set_workspace.set_workspace import set_workspace


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    set_workspace(root)
    return root


def _define(root: Path, name: str, body: str) -> None:
    (root / "data_raw" / "foreground_scenarios" / "scenarios" / f"{name}.yaml").write_text(
        body.strip() + "\n", encoding="utf-8"
    )


def test_first_possible_input_change_is_2020_and_history_is_reference_only(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        "history_attempt",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: history_attempt
ssp_scenario: SSP2
changes:
  - parameter: collection_rate
    region: EUR
    changes:
      - {year: 2019, mode: absolute, value: 0.5}
""",
    )
    with pytest.raises(ScenarioError, match="2020"):
        compile_foreground_scenario(root, "history_attempt")

    _define(
        root,
        "reference_only",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: reference_only
ssp_scenario: SSP2
changes:
  - parameter: collection_rate
    region: EUR
    changes:
      - {year: 2025, mode: retain_2019}
""",
    )
    result = compile_foreground_scenario(root, "reference_only")
    table = pd.read_parquet(result.compiled_parameters_path)
    eur = table.loc[
        table["parameter_id"].eq("collection_rate") & table["region"].eq("EUR")
    ].set_index("year")
    assert eur.loc[2024, "value_origin"] == "remind_mfa_default"
    assert eur.loc[2025, "value_origin"] == "user_foreground_scenario"
    assert eur.loc[2025, "resolved_value"] == pytest.approx(eur.loc[2025, "reference_value"])
    assert 2019 not in eur.index


def test_direct_stock_and_socioeconomic_driver_changes_conflict(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        "driver_conflict",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: driver_conflict
ssp_scenario: SSP2
changes:
  - parameter: stock_projection
    region: EUR
    good: Packaging
    changes:
      - {year: 2030, mode: multiplier_from_default, value: 0.9}
  - parameter: population
    region: EUR
    changes:
      - {year: 2030, mode: multiplier_from_default, value: 0.9}
""",
    )
    with pytest.raises(ScenarioConflictError, match="cannot be changed together"):
        compile_foreground_scenario(root, "driver_conflict")


def test_static_upstream_inputs_compile_annually_without_retroactive_replacement(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    _define(
        root,
        "future_lifetime",
        """
schema_version: pytoce_foreground_scenario_schema_v1
foreground_scenario: future_lifetime
ssp_scenario: SSP2
changes:
  - parameter: lifetime_mean
    good: Packaging
    changes:
      - {year: 2035, mode: absolute, value: 1.0}
""",
    )
    result = compile_foreground_scenario(root, "future_lifetime")
    table = pd.read_parquet(result.compiled_parameters_path)
    lifetime = table.loc[
        table["parameter_id"].eq("lifetime_mean") & table["good"].eq("Packaging")
    ].set_index("year")
    assert lifetime.loc[2034, "resolved_value"] == lifetime.loc[2034, "default_value"]
    assert lifetime.loc[2035, "resolved_value"] == 1.0
    assert lifetime.loc[2100, "resolved_value"] == 1.0
