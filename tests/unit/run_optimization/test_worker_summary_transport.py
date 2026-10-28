"""Preserve compact completed-stage metrics and exact dispositions."""

import json
from dataclasses import replace
from fractions import Fraction

import pytest
from candidate_worker_fixtures import _infeasible_result, _result, _summary

from pytoce.run_optimization.calculation.worker_protocol import (
    CANDIDATE_WORKER_SUMMARY_SCHEMA,
    candidate_worker_result_identity,
    decode_candidate_worker_summary,
    encode_candidate_worker_summary,
)


@pytest.mark.parametrize("disposition", ("linear", "stock_projection", "unresolved", "rejected"))
def test_completed_stage_summary_preserves_science_without_arrays(disposition: str) -> None:
    """JSON round trips preserve ranking, rational departure and direct diagnostics."""

    result = _infeasible_result() if disposition in ("unresolved", "rejected") else _result()
    if disposition == "stock_projection":
        result = replace(
            result,
            repair=replace(
                result.repair,
                status="stock_projection_feasible",
                mfa_evaluation_count=2,
                stock_lp_count=1,
                demand_departure_pp=Fraction(7, 13),
                linear_diagnostic=_infeasible_result().physical.diagnostic,
            ),
        )
    elif disposition == "unresolved":
        result = replace(
            result,
            repair=replace(
                result.repair,
                status="repair_unresolved",
                mfa_evaluation_count=4,
                stock_lp_count=1,
                pool_candidate_count=2,
                pool_evaluation_count=2,
                stock_projection_diagnostic=result.physical.diagnostic,
            ),
        )
    result = replace(result, identity=candidate_worker_result_identity(result))
    summary = _summary(result)

    record = encode_candidate_worker_summary(summary)
    encoded = json.dumps(record, allow_nan=False, sort_keys=True)
    restored = decode_candidate_worker_summary(json.loads(encoded))

    assert record["schema_version"] == CANDIDATE_WORKER_SUMMARY_SCHEMA
    assert restored == summary
    assert (
        json.dumps(encode_candidate_worker_summary(restored), allow_nan=False, sort_keys=True)
        == encoded
    )
    assert restored.member_metrics == summary.member_metrics
    assert restored.repair == summary.repair
    assert '"annual_scores":' not in encoded
    assert '"activation_delta":' not in encoded
