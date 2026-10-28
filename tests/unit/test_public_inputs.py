"""Public input admission before workspace access or replacement."""

from pathlib import Path

import pytest

import pytoce as package


@pytest.mark.parametrize("workflow", ("set_workspace", "download_data", "run_mfa", "run_lca"))
@pytest.mark.parametrize("refresh", ("False", 1, None))
def test_refresh_requires_boolean_without_changing_workspace(
    tmp_path: Path, workflow: str, refresh: object
) -> None:
    """Reject truthy and falsy non-Booleans before accessing scientific inputs."""
    report = package.set_workspace(tmp_path / "existing")
    report.parameter_catalogue.write_bytes(b"user-edited input\n")
    original = {
        path.relative_to(report.workspace): path.read_bytes()
        for path in report.workspace.rglob("*")
        if path.is_file()
    }
    function = getattr(package, workflow)
    args = (report.workspace,) if workflow == "set_workspace" else ()

    with pytest.raises(ValueError, match="refresh=.*Use refresh=True.*refresh=False"):
        function(*args, refresh=refresh)

    assert {
        path.relative_to(report.workspace): path.read_bytes()
        for path in report.workspace.rglob("*")
        if path.is_file()
    } == original


def test_invalid_refresh_does_not_create_workspace(tmp_path: Path) -> None:
    """Reject invalid replacement control before creating any directory."""
    workspace = tmp_path / "new_workspace"
    with pytest.raises(ValueError, match="refresh="):
        package.set_workspace(workspace, refresh="False")
    assert not workspace.exists()
