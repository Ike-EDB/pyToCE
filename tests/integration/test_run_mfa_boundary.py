import re
from pathlib import Path

import pytest

from pytoce import run_mfa, set_workspace


def test_run_mfa_requires_completed_download_data_parent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    stored_material_recipe: tuple[Path, dict[str, object]],
) -> None:
    set_workspace(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="REMIND-MFA source.*download_data",
    ):
        run_mfa()

    progress = capsys.readouterr().err
    assert "reading the scenario YAML" in progress
    assert "reading REMIND-MFA source data" in progress


@pytest.mark.parametrize("refresh", (False, True))
def test_incomplete_mfa_result_is_preserved_before_source_preparation(
    tmp_path: Path, stored_material_recipe: tuple[Path, dict[str, object]], refresh: bool
) -> None:
    """Report incomplete results before compiling scenarios or preparing historical MFA."""
    set_workspace(tmp_path)
    result = tmp_path / "results/unfinished/mfa_results"
    result.mkdir(parents=True)
    partial = result / "flows.parquet"
    partial.write_bytes(b"interrupted write")
    with pytest.raises(ValueError, match=re.escape(str(result)) + ".*completion manifest"):
        run_mfa(foreground_scenario="unfinished", refresh=refresh)
    assert partial.read_bytes() == b"interrupted write"
    assert not (tmp_path / "data_processed/foreground_scenarios/unfinished").exists()


@pytest.mark.parametrize(
    ("selection", "message"),
    (
        ((), "Select at least one"),
        (("",), "non-empty YAML filename stem"),
        (1, "one readable name or a sequence"),
    ),
)
def test_run_mfa_rejects_invalid_scenario_selections_before_reading_data(
    tmp_path: Path,
    selection: object,
    message: str,
) -> None:
    set_workspace(tmp_path)

    with pytest.raises(ValueError, match=message):
        run_mfa(foreground_scenario=selection)  # type: ignore[arg-type]
