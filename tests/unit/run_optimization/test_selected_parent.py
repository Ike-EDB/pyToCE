from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
from selected_parent_fixtures import (
    _BACKGROUND,
    _FOREGROUND,
    _tables,
    _write_baseline_result,
)

from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_lca.run_lca import lca_result_directory
from pytoce.run_optimization.calculation.selected_parent import (
    OptimizationResultError,
    load_baseline_lca_result,
    select_baseline_lca_result,
)


@pytest.mark.parametrize("years", [(2019, 2050, 2100), (2050, 2100)])
def test_load_baseline_lca_result_reads_exact_manifest_and_tables(
    tmp_path: Path, years: tuple[int, ...]
) -> None:
    """Read the same annual scores for either ordinary upstream reporting selection."""

    selection = normalized_lcia_method("ef31")
    directory = lca_result_directory(tmp_path, _FOREGROUND, _BACKGROUND, selection, years)
    manifest = _write_baseline_result(directory, years=years)
    manifest_bytes = (directory / "manifest.json").read_bytes()

    reference = select_baseline_lca_result(
        tmp_path,
        foreground_scenario=_FOREGROUND,
        background_scenario=_BACKGROUND,
        selection=selection,
    )
    result = load_baseline_lca_result(reference)

    assert reference.directory == directory
    assert result.record.result_identity == manifest["identity"]
    assert result.record.result_manifest_sha256 == sha256(manifest_bytes).hexdigest()
    assert result.record.foreground_scenario == _FOREGROUND
    assert result.record.background_scenario == _BACKGROUND
    for observed, expected in zip(
        (
            result.annual_life_cycle_stage_scores,
            result.life_cycle_stage_scores,
        ),
        (
            _tables()["annual_life_cycle_stage_scores.parquet"],
            _tables()["life_cycle_stage_scores.parquet"],
        ),
        strict=True,
    ):
        pd.testing.assert_frame_equal(observed, expected)
    assert (directory / "manifest.json").read_bytes() == manifest_bytes


@pytest.mark.parametrize("include_default", [False, True])
def test_equivalent_reporting_variants_keep_deterministic_parent_custody(
    tmp_path: Path, include_default: bool
) -> None:
    """Choose one unchanged complete parent while preserving other reports and staging."""
    selection = normalized_lcia_method("ef31")
    years = [(2100,), (2050, 2100)]
    if include_default:
        years.append((2019, 2050, 2100))
    manifests = {}
    for reporting_years in years:
        directory = lca_result_directory(
            tmp_path, _FOREGROUND, _BACKGROUND, selection, reporting_years
        )
        _write_baseline_result(directory, years=reporting_years)
        manifests[directory] = (directory / "manifest.json").read_bytes()
    incomplete = lca_result_directory(tmp_path, _FOREGROUND, _BACKGROUND, selection, (2019,))
    incomplete.mkdir()
    marker = incomplete / "unfinished.txt"
    marker.write_text("retained staging", encoding="utf-8")
    # Parent lookup selects the exact background and result level.
    unrelated = lca_result_directory(tmp_path, _FOREGROUND, "remind_ssp2_ndc", selection, (2050,))
    _write_baseline_result(
        unrelated,
        background_scenario="remind_ssp2_ndc",
        years=(2050,),
        mfa_identity="mfa::other",
    )
    nested = incomplete.parent / "archive" / incomplete.name
    _write_baseline_result(nested, years=(2019,), mfa_identity="mfa::other")

    reference = select_baseline_lca_result(
        tmp_path,
        foreground_scenario=_FOREGROUND,
        background_scenario=_BACKGROUND,
        selection=selection,
    )

    expected_years = (2019, 2050, 2100) if include_default else (2050, 2100)
    expected = lca_result_directory(tmp_path, _FOREGROUND, _BACKGROUND, selection, expected_years)
    assert reference.directory == expected
    assert reference.record.result_manifest_sha256 == sha256(manifests[expected]).hexdigest()
    assert marker.read_text(encoding="utf-8") == "retained staging"
    assert all(
        (path / "manifest.json").read_bytes() == content for path, content in manifests.items()
    )


@pytest.mark.parametrize("incomplete", [False, True])
def test_required_baseline_completion_preserves_unfinished_products(
    tmp_path: Path, incomplete: bool
) -> None:
    """Report the exact missing parent and preserve an interrupted ordinary publication."""
    selection = normalized_lcia_method("ef31")
    directory = lca_result_directory(
        tmp_path, _FOREGROUND, _BACKGROUND, selection, (2019, 2050, 2100)
    )
    marker = directory / "unfinished.txt"
    if incomplete:
        directory.mkdir(parents=True)
        marker.write_text("unfinished ordinary result", encoding="utf-8")
    with pytest.raises(OptimizationResultError, match="no completion manifest.*run_lca") as caught:
        select_baseline_lca_result(
            tmp_path,
            foreground_scenario=_FOREGROUND,
            background_scenario=_BACKGROUND,
            selection=selection,
        )
    assert str(directory) in str(caught.value)
    if incomplete:
        assert marker.read_text(encoding="utf-8") == "unfinished ordinary result"
