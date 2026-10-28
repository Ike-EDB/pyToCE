"""Disposable scenario test workspace fixtures."""

from collections.abc import Mapping
from pathlib import Path

from pytoce.run_mfa.scenario.compiler import _compile_payload
from pytoce.run_mfa.scenario.rules import _validate_scenario_payload
from pytoce.run_mfa.scenario.scenario_contract import ScenarioEvaluation
from pytoce.set_workspace.set_workspace import set_workspace


def compile_scenario_payload(payload: Mapping[str, object]) -> ScenarioEvaluation:
    """Compile a scenario through the same owner used by ``run_mfa``."""

    validated = _validate_scenario_payload(
        dict(payload), expected_name=str(payload["foreground_scenario"])
    )
    evaluation, _ = _compile_payload(
        validated, source_path=Path(str(validated["foreground_scenario"]) + ".yaml")
    )
    return evaluation


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    set_workspace(root)
    return root


def _define(root: Path, name: str, text: str) -> None:
    path = root / "data_raw" / "foreground_scenarios" / "scenarios" / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
