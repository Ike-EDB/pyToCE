"""Focused candidate worker contracts."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from allocated_cc_fixtures import _write_raw_result, _write_shared_authority
from candidate_characterization_fixtures import _physical
from candidate_worker_fixtures import (
    _initializer,
    _worker_contexts,
)
from mfa_workspace_fixtures import StoredMFAFixture
from response_bank_fixtures import _inputs, _write_bank
from response_projection_fixtures import _response_projection

from pytoce.run_mfa.model.result_inputs import read_mfa_result_manifest
from pytoce.run_optimization.calculation.candidate_projection import (
    _environmental_identity,
    prepare_search_scoring,
)
from pytoce.run_optimization.calculation.worker_inputs import (
    candidate_worker_authority_identity,
    prepare_candidate_worker_initializer,
)
from pytoce.run_optimization.input.allocated_cc_product import (
    compile_allocated_cc_result,
)
from pytoce.shared.runtime.filesystem import payload_sha256


def test_worker_authority_binds_search_projection_independently_of_reporting() -> None:
    """Bind reuse to the MFA, aCC, and EF response inputs."""
    payload = json.loads(_initializer())
    payload["prepared_receiving_identity"] = "prepared-receiving::fixture"
    projection_identity = "projection::ef31"
    authority = candidate_worker_authority_identity(
        payload, optimization_projection_identity=projection_identity
    )
    reordered = dict(reversed(list(payload.items())))
    assert (
        candidate_worker_authority_identity(
            reordered, optimization_projection_identity=projection_identity
        )
        == authority
    )
    for changed in (
        {"allocated_cc_case": ["stringent"]},
        {"foreground_scenario": "another_parent"},
        {"candidate_mfa_session_identity": "candidate-mfa::changed"},
        {"prepared_receiving_identity": "prepared-receiving::changed"},
    ):
        assert (
            candidate_worker_authority_identity(
                {**payload, **changed}, optimization_projection_identity=projection_identity
            )
            != authority
        )
    assert (
        candidate_worker_authority_identity(
            payload, optimization_projection_identity="projection::changed-ef-responses"
        )
        != authority
    )
    assert (
        candidate_worker_authority_identity(
            {
                **payload,
                "workspace": "D:/another-workspace",
                "response_method_union": "another_container",
                "response_bank_manifest_sha256": "f" * 64,
                "response_bank_identity": "response::reporting-superset",
            },
            optimization_projection_identity=projection_identity,
        )
        == authority
    )


@pytest.mark.parametrize(
    "backgrounds",
    (
        tuple(parent.background_scenario for parent in _inputs().member_parents),
        ("remind_ssp2_npi", "remind_ssp2_pkbudg650"),
    ),
)
def test_initializer_uses_admitted_mfa_and_acc_parents(
    tmp_path: Path, saved_mfa: StoredMFAFixture, backgrounds: tuple[str, ...]
) -> None:
    """Serialize real admitted parents while preserving compiled-input correspondence."""
    name = saved_mfa.compiled.foreground_scenario
    reference = read_mfa_result_manifest(saved_mfa.target.result_directory / "manifest.json")
    _write_shared_authority(tmp_path)
    _write_raw_result(tmp_path, "study_acc", cases=("central",))
    allocated_cc = compile_allocated_cc_result(tmp_path, "study_acc")
    _write_bank(tmp_path, response_method_union="ef31_pb_lcia_response_v3")

    payload = prepare_candidate_worker_initializer(
        tmp_path,
        foreground_scenario=name,
        mfa_reference=reference,
        allocated_cc=allocated_cc,
        allocated_cc_case=("central",),
        background_scenarios=backgrounds,
    )

    decoded = json.loads(payload)
    assert decoded["background_scenarios"] == list(backgrounds)
    assert decoded["mfa_result_identity"] == reference.manifest["identity"]
    assert decoded["mfa_result_manifest_sha256"] == reference.sha256
    assert decoded["allocated_cc_result"] == allocated_cc.allocated_cc_result
    assert decoded["allocated_cc_result_identity"] == allocated_cc.allocated_cc_result_identity
    prepared = _inputs()
    assert decoded["prepared_receiving_identity"] == payload_sha256(
        {
            "shared": prepared.prepared_background_shared_identity,
            "members": {
                parent.background_scenario: parent.prepared_background_identity
                for parent in prepared.member_parents
                if parent.background_scenario in backgrounds
            },
        }
    )


def test_receiving_ancestry_identifies_scoring_even_with_zero_parent_pa() -> None:
    """An unused parent route cannot hide a changed prepared receiving source."""
    _, initial = _worker_contexts()
    changed = prepare_search_scoring(
        physical_parent=initial.physical_parent,
        optimization_parent=initial.optimization_parent,
        climate_parent=initial.climate_parent,
        optimization_projection=_response_projection("optimization"),
        climate_projection=_response_projection("mandatory_climate"),
        pa_projection=replace(
            initial.pa_projection, prepared_receiving_identity="prepared-receiving::changed"
        ),
        pa_parent=initial.pa_parent,
    )
    assert not initial.pa_parent.annual_scores.any()
    np.testing.assert_array_equal(
        changed.optimization_baseline_scores, initial.optimization_baseline_scores
    )
    assert changed.identity != initial.identity
    physical = _physical("same-physical-state", delta=initial.physical_parent.activations.iloc[:0])
    assert (
        _environmental_identity(physical, changed)[0]
        != _environmental_identity(physical, initial)[0]
    )
