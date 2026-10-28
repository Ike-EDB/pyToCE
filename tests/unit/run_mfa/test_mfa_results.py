"""Real MFA persistence, manifest admission and owner-scoped replacement."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
from mfa_workspace_fixtures import StoredMFAFixture

from pytoce.run_mfa.model.result_inputs import (
    read_mfa_result,
    read_mfa_result_manifest,
)
from pytoce.run_mfa.model.results import (
    selected_mfa_result,
    validate_mfa_result_for_refresh,
    write_mfa_result,
)


def test_saved_mfa_preserves_scientific_tables_and_manifest_authority(
    saved_mfa: StoredMFAFixture, tmp_path: Path
) -> None:
    """Keep MFA identity and artifact bytes distinct through publication and reuse."""
    target, computation = saved_mfa.target, saved_mfa.computation
    path = target.result_directory / "manifest.json"
    reference = read_mfa_result_manifest(path)
    loaded = read_mfa_result(reference)
    manifest, digest = reference.manifest, reference.sha256
    assert digest == sha256(path.read_bytes()).hexdigest()
    assert manifest["identity"] == target.identity
    assert manifest["scientific_authorities"]["physical_state_identity"] == (
        computation.physical_state_identity
    )
    for name in (
        "model_flows",
        "derived_outputs",
        "material_realization",
        "nominal_feedstock_equivalents",
        "cohort_state",
        "cohort_material_identity",
    ):
        pd.testing.assert_frame_equal(
            getattr(loaded, name),
            getattr(computation, name),
            check_exact=True,
        )
    assert loaded.physical_state_identity == computation.physical_state_identity
    assert loaded.material_identity_method == computation.material_identity_method
    assert selected_mfa_result(tmp_path, "direct_fixture") == path
    validate_mfa_result_for_refresh(target)
    assert write_mfa_result(target, computation, replacing=True) == "updated"
    assert read_mfa_result_manifest(path) == reference


def test_parent_guide_failure_does_not_publish_a_completed_mfa_result(
    saved_mfa: StoredMFAFixture, tmp_path: Path
) -> None:
    """A real parent README collision fails before a new result becomes visible."""
    target = replace(saved_mfa.target, result_directory=tmp_path / "guide_collision/mfa_results")
    guide = target.result_directory.parent / "README.md"
    guide.mkdir(parents=True)
    with pytest.raises(OSError):
        write_mfa_result(target, saved_mfa.computation, replacing=False)
    assert guide.is_dir()
    assert not target.result_directory.exists()
    assert list(target.result_directory.parent.glob(".mfa_result.*")) == []
