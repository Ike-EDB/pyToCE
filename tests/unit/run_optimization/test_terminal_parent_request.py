"""Public terminal-parent requests preserve full scientific reporting and reuse."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytoce import run_optimization
from pytoce.run_lca.calculation.foreground import H12_REGIONS
from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.candidate_search import (
    CANDIDATE_REFINEMENT_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    decode_candidate_worker_result,
)
from pytoce.run_optimization.runtime.completed_search import decode_completed_search
from tests.integration.candidate_mfa_fixtures import (
    _BACKGROUND_SCENARIOS,
    _WORKER_FOREGROUND,
    _prepare_worker_workspace,
)


@pytest.mark.scientific_integration
def test_public_terminal_parent_requests_count_prerequisites_and_reuse_physical_results(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> None:
    """Compute Block II and integrated requests from one admitted generous-case fixture."""
    workspace = tmp_path / "workspace"
    _initializer, context = _prepare_worker_workspace(
        workspace, remind_mfa_source_root, include_generous_case=True
    )
    product_identities = set()
    requests: tuple[tuple[OptimizationBlock, int], ...] = (
        ("circularity_feedstock_anchor", 1),
        ("integrated_frontier", 3),
    )
    for block, coordinate_count in requests:
        computed = run_optimization(
            foreground_scenario=_WORKER_FOREGROUND,
            allocated_cc_result="worker_result",
            lcia_method="ef31",
            optimization_block=block,
            allocated_cc_case="generous_case",
        )
        assert computed.status == "computed"
        assert len(computed.products) == 1
        product = computed.products[0]
        product_identities.add(product.product_identity)
        assert product.foreground_scenario == _WORKER_FOREGROUND
        directory = product.result_directory
        summary = json.loads((directory / "method_summary.json").read_bytes())
        assert summary["refinement_method"] == CANDIDATE_REFINEMENT_METHOD_IDENTITY
        assert summary["request_coordinate_count"] == coordinate_count
        assert summary["request_unique_physical_candidate_count"] == 1
        assert summary["request_physical_candidate_deduplication_count"] == coordinate_count - 1
        assert summary["selected_coordinate_count"] == 1
        assert summary["selected_unique_physical_candidate_count"] == 1
        assert summary["state_count"] == 4
        assert summary["parent_terminal_state_count"] == 4
        assert summary["declared_neighborhood_verified_state_count"] == 0
        assert summary["target_years"] == [2050]
        assert summary["global_optimum_claim"] is False
        assert len(summary["refinement"]) == 1
        refinement = summary["refinement"][0]
        assert refinement["optimization_block"] == block
        assert refinement["evaluation_count"] == 1
        assert refinement["reused_candidate_count"] == 0
        assert refinement["unsuccessful_path_search_count"] == 0
        assert refinement["final_neighbor_count"] == 0
        assert all(item["pass_count"] == 0 for item in refinement["refinement_pass_counts"])

        roles = pd.read_parquet(directory / "selection_roles.parquet")
        assert len(roles) == 4
        assert set(roles.selection_background_scenario) == set(_BACKGROUND_SCENARIOS)
        assert set(roles.optimization_block) == {block}
        assert set(roles.allocated_cc_case) == {"generous_case"}
        assert roles.selection_status.eq("baseline_compliant").all()
        assert roles.selected_is_exact_parent.all()
        assert roles.all_region_joint_compliant.all()
        assert roles.intervention_distance.eq(0).all()
        assert roles.region_count.eq(12).all()
        assert roles.impact_target_count.eq(17).all()
        assert not roles.declared_neighborhood_verified.any()
        assert not roles.selection_stable.any()
        for year in (2100,):
            assert roles[f"timing_{year}_candidate_identity"].equals(
                roles.selection_candidate_identity
            )
            assert roles[f"timing_{year}_physical_candidate_identity"].equals(
                roles.physical_candidate_identity
            )

        coordinates = pd.read_parquet(directory / "selected_coordinates.parquet")
        assert len(coordinates) == 1
        assert coordinates.target_year.eq(2050).all()
        assert coordinates.physical_status.eq("physical_feasible").all()
        assert coordinates.demand_reduction_pct.eq(0).all()
        assert coordinates.lifetime_extension_pct.eq(0).all()

        # Each public request owns its own cache. Its prerequisite blocks share
        # one exact parent MFA result rather than writing duplicate payloads.
        cache = directory.parents[1] / ".candidate_results"
        payloads = tuple(cache.glob("*/payload.bin"))
        assert len(payloads) == 1
        payload = payloads[0]
        payload_bytes = payload.read_bytes()
        payload_mtime = payload.stat().st_mtime_ns
        saved = decode_candidate_worker_result(payload_bytes)
        assert saved.intervention.uses_exact_parent
        assert saved.physical.diagnostic is None
        assert saved.scores is not None
        assert set(saved.scores.backgrounds) == set(_BACKGROUND_SCENARIOS)
        assert set(saved.scores.regions) == set(H12_REGIONS)
        assert saved.scores.years == tuple(range(2019, 2101))
        assert saved.scores.annual_scores.shape == (4, 12, 82, 16)
        np.testing.assert_array_equal(
            saved.scores.annual_scores, context.search_scoring.optimization_baseline_scores
        )
        assert roles.physical_candidate_identity.eq(
            saved.physical.physical_candidate_identity
        ).all()

        annual = pd.read_parquet(directory / "selected_environmental_scores.parquet")
        assert set(annual.background_scenario) == set(_BACKGROUND_SCENARIOS)
        assert set(annual.region) == {*H12_REGIONS, "global"}
        assert set(annual.year.dropna()) == set(range(2019, 2101))
        assert annual[["method_id", "impact_category_id"]].drop_duplicates().shape[0] == 16

        manifest_bytes = product.manifest_path.read_bytes()
        reused = run_optimization(
            foreground_scenario=_WORKER_FOREGROUND,
            allocated_cc_result="worker_result",
            lcia_method="ef31",
            optimization_block=block,
            allocated_cc_case="generous_case",
        )
        assert reused.status == "reused"
        assert reused.products[0].product_identity == product.product_identity
        assert product.manifest_path.read_bytes() == manifest_bytes
        assert payload.read_bytes() == payload_bytes
        assert payload.stat().st_mtime_ns == payload_mtime
        pd.testing.assert_frame_equal(
            pd.read_parquet(directory / "selection_roles.parquet"), roles, check_exact=True
        )
        if block == "integrated_frontier":
            # This synthetic terminal-parent fixture exercises public stage continuation.
            completed_stages = decode_completed_search(summary["completed_search"]).stages
            extension = run_optimization(
                foreground_scenario=_WORKER_FOREGROUND,
                allocated_cc_result="worker_result",
                lcia_method="ef31",
                optimization_block=block,
                allocated_cc_case="generous_case",
                endpoint_refinement_steps_pct=(10, 5, 1),
            )
            assert extension.status == "computed"
            successor = extension.products[0]
            assert successor.result_directory != directory
            appended = json.loads((successor.result_directory / "method_summary.json").read_bytes())
            assert appended["continuation_parent_identity"] == product.product_identity
            assert (
                decode_completed_search(appended["completed_search"]).stages[:3] == completed_stages
            )
            assert appended["work_accounting"]["newly_computed_worker_result_count"] == 0
            assert appended["work_accounting"]["reused_worker_result_count"] == 1
            assert appended["work_accounting"]["incremental_mfa_evaluation_count"] == 0
            assert product.manifest_path.read_bytes() == manifest_bytes
            assert payload.read_bytes() == payload_bytes
            assert payload.stat().st_mtime_ns == payload_mtime
    assert len(product_identities) == 2
