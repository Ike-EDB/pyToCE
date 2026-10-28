"""Read completed baseline LCA results for selected optimization reporting."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pytoce.run_lca.calculation.lcia import LCIAMethodSelection
from pytoce.run_lca.calculation.result_inputs import LCAResultError, select_lca_result
from pytoce.run_optimization.calculation.selected_reporting import (
    BaselineLCAResult,
    BaselineLCAResultRecord,
)

_BASELINE_LCA_ARTIFACTS = (
    "annual_life_cycle_stage_scores.parquet",
    "life_cycle_stage_scores.parquet",
)


class OptimizationResultError(ValueError):
    """A required baseline LCA result is missing, incomplete, or ambiguous."""


@dataclass(frozen=True, slots=True)
class BaselineLCAResultReference:
    """Keep a chosen completed parent."""

    directory: Path
    record: BaselineLCAResultRecord


def select_baseline_lca_result(
    workspace: Path,
    *,
    foreground_scenario: str,
    background_scenario: str,
    selection: LCIAMethodSelection,
) -> BaselineLCAResultReference:
    """Choose a completed score parent independently of optional upstream detail years.

    Only immediate results for the exact foreground, background, and LCIA
    selection are considered. Reporting-year variants must have identical
    scientific identity payloads after removing only ``upstream_analysis_year``.
    Prefer the current default-year product, then the first readable name.
    Completed products supply the reporting parent.
    """
    try:
        parent = select_lca_result(
            workspace,
            foreground_scenario=foreground_scenario,
            background_scenario=background_scenario,
            selection=selection,
        )
    except LCAResultError as exc:
        raise OptimizationResultError(str(exc)) from exc
    directory = parent.directory
    manifest = parent.manifest
    descriptors = tuple(manifest["artifacts"][name] for name in _BASELINE_LCA_ARTIFACTS)
    record = BaselineLCAResultRecord(
        foreground_scenario=str(manifest["foreground_scenario"]),
        background_scenario=str(manifest["background_scenario"]),
        result_identity=str(manifest["identity"]),
        result_manifest_sha256=parent.manifest_sha256,
        annual_life_cycle_stage_scores_identity=str(descriptors[0]["canonical_sha256"]),
        life_cycle_stage_scores_identity=str(descriptors[1]["canonical_sha256"]),
    )
    return BaselineLCAResultReference(directory, record)


def load_baseline_lca_result(reference: BaselineLCAResultReference) -> BaselineLCAResult:
    """Read the two stage tables from the parent selected before candidate calculation."""
    frames = tuple(pd.read_parquet(reference.directory / name) for name in _BASELINE_LCA_ARTIFACTS)
    return BaselineLCAResult(reference.record, *frames)
