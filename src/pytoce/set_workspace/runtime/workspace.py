"""Create workspace directories and select the workspace for the current Python session."""

from pathlib import Path

from .installation import (
    METHODOLOGICAL_NOTES_RELATIVE,
    RECOMMENDED_CITATIONS_RELATIVE,
    SCENARIOS_RELATIVE,
    TEMPLATES_RELATIVE,
)

REQUIRED_ROOTS = ("data_raw", "data_processed", "results")
_ACTIVE_WORKSPACE_ROOT: dict[str, Path | None] = {"workspace": None}


def resolve_workspace(workspace: str | Path) -> Path:
    """Resolve a user workspace path."""

    if isinstance(workspace, str) and not workspace.strip():
        raise ValueError("workspace must be a non-empty path")
    path = Path(workspace).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"Workspace path is not a directory: {path}")
    return path


def ensure_roots(workspace: Path) -> None:
    """Create the three public workspace roots."""

    workspace.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ROOTS:
        (workspace / name).mkdir(exist_ok=True)


def set_active_workspace(workspace: Path) -> None:
    """Store the resolved workspace selected by the public coordinator."""

    _ACTIVE_WORKSPACE_ROOT["workspace"] = workspace


def get_active_workspace() -> Path:
    """Return the workspace selected by the latest ``set_workspace`` call."""

    workspace = _ACTIVE_WORKSPACE_ROOT["workspace"]
    if workspace is None:
        raise RuntimeError(
            "No workspace is active. Call pytoce.set_workspace(path) before running this workflow."
        )
    return workspace


def render_summary(
    *,
    status: str,
    workspace: Path,
    installed_count: int,
) -> str:
    """Render concise live workspace guidance."""

    return "\n".join(
        (
            "[pyToCE set_workspace]",
            f"status: {status}",
            f"workspace: {workspace}",
            f"installed_package_files: {installed_count}",
            "top_level_directories: data_raw, data_processed, results",
            f"foreground_scenarios: {workspace / SCENARIOS_RELATIVE}",
            f"non_runnable_templates: {workspace / TEMPLATES_RELATIVE}",
            f"methodological_notes: {workspace / METHODOLOGICAL_NOTES_RELATIVE}",
            f"recommended_citations: {workspace / RECOMMENDED_CITATIONS_RELATIVE}",
        )
    )
