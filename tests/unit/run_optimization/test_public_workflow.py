"""Public optimization input translation, calculation, and result reuse."""

from pathlib import Path

import pandas as pd
import pytest

from pytoce import run_optimization
from pytoce.run_optimization.run_optimization import (
    RunOptimizationError,
    normalize_run_optimization_request,
)
from pytoce.run_optimization.runtime.result_store import OptimizationResultTarget
from tests.integration.candidate_mfa_fixtures import (
    _WORKER_FOREGROUND,
    _prepare_worker_workspace,
)


@pytest.mark.scientific_integration
def test_public_optimization_reports_missing_inputs_and_cache_collision(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> None:
    """Report unusable prerequisites and preserve a conflicting user cache entry."""

    workspace = tmp_path / "workspace"
    foreground = _WORKER_FOREGROUND
    _prepare_worker_workspace(
        workspace,
        remind_mfa_source_root,
        foreground_scenario=foreground,
        include_generous_case=True,
    )
    mfa_directory = workspace / "results" / foreground / "mfa_results"
    saved_mfa = mfa_directory.with_name("saved_mfa_result")
    mfa_directory.rename(saved_mfa)
    try:
        with pytest.raises(RunOptimizationError, match="No completed MFA result was found"):
            run_optimization(
                foreground_scenario=foreground,
                allocated_cc_result="worker_result",
                lcia_method=("ef31",),
                optimization_block="demand_lifetime_frontier",
            )
    finally:
        saved_mfa.rename(mfa_directory)

    with pytest.raises(RunOptimizationError, match="Cannot use allocated_cc_result='missing'"):
        run_optimization(
            foreground_scenario=foreground,
            allocated_cc_result="missing",
            lcia_method=("ef31",),
            optimization_block="demand_lifetime_frontier",
        )

    response_directory = (
        workspace / "data_processed/optimization_response_bank/four_remind/ef31_response_v3"
    )
    saved_response = response_directory.with_name("saved_response_bank")
    response_directory.rename(saved_response)
    try:
        with pytest.raises(RunOptimizationError, match="Cannot prepare optimization inputs"):
            run_optimization(
                foreground_scenario=foreground,
                allocated_cc_result="worker_result",
                lcia_method=("ef31",),
                optimization_block="demand_lifetime_frontier",
            )
    finally:
        saved_response.rename(response_directory)

    request = normalize_run_optimization_request(
        foreground_scenario=foreground,
        allocated_cc_result="worker_result",
        lcia_method=("ef31",),
        optimization_block="demand_lifetime_frontier",
        allocated_cc_case="generous_case",
    )
    target = OptimizationResultTarget(
        workspace=workspace,
        foreground_scenario=foreground,
        allocated_cc_result=request.allocated_cc_result,
        optimization_lcia_selection="ef31",
        readable_request_selection=request.readable_request_selection,
        initial_grid_step_pct=20,
        endpoint_refinement_steps_pct=(10, 5),
    )
    target.root.mkdir(parents=True, exist_ok=True)
    invalid_cache = target.root / ".candidate_results"
    invalid_cache.write_text("not a directory", encoding="utf-8")
    with pytest.raises(RunOptimizationError, match="Optimization failed.*not a directory"):
        run_optimization(
            foreground_scenario=foreground,
            allocated_cc_result="worker_result",
            lcia_method=("ef31",),
            optimization_block="demand_lifetime_frontier",
            allocated_cc_case="generous_case",
        )
    assert invalid_cache.read_text(encoding="utf-8") == "not a directory"


@pytest.mark.scientific_integration
def test_public_optimization_admits_inputs_computes_and_reuses(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> None:
    """Publish and reuse a complete search from actual admitted parent inputs."""
    workspace = tmp_path / "workspace"
    foreground = _WORKER_FOREGROUND
    _prepare_worker_workspace(
        workspace,
        remind_mfa_source_root,
        foreground_scenario=foreground,
        include_generous_case=True,
        background_scenarios=("remind_ssp2_npi", "remind_ssp2_pkbudg650"),
    )
    computed = run_optimization(
        foreground_scenario=foreground,
        allocated_cc_result="worker_result",
        background_scenario=("remind_ssp2_npi", "remind_ssp2_pkbudg650"),
        lcia_method=("ef31",),
        optimization_block="demand_lifetime_frontier",
        allocated_cc_case="generous_case",
        candidate_workers=3,
    )
    reused = run_optimization(
        foreground_scenario=foreground,
        allocated_cc_result="worker_result",
        background_scenario=("remind_ssp2_npi", "remind_ssp2_pkbudg650"),
        lcia_method=("ef31",),
        optimization_block="demand_lifetime_frontier",
        allocated_cc_case="generous_case",
    )

    assert computed.status == "computed"
    assert reused.status == "reused"
    assert computed.products[0].foreground_scenario == foreground
    assert computed.products[0].result_directory.is_relative_to(
        workspace / "results" / foreground / "optimization_results"
    )
    assert computed.products[0].product_identity == reused.products[0].product_identity
    roles = pd.read_parquet(computed.products[0].result_directory / "selection_roles.parquet")
    assert set(roles["allocated_cc_case"]) == {"generous_case"}
    assert set(roles.selection_background_scenario) == {"remind_ssp2_npi", "remind_ssp2_pkbudg650"}
    assert not any("2075" in column for column in roles)
    assert roles["selected_is_exact_parent"].all()
    assert roles["selection_status"].eq("baseline_compliant").all()
