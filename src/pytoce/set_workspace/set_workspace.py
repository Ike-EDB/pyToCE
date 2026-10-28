"""Create and select a pyToCE workspace."""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from pytoce.shared.runtime.manifests import OperationStatus


@dataclass(frozen=True, slots=True)
class WorkspaceReport:
    """Result of :func:`pytoce.set_workspace`.

    Attributes:
        status: ``computed`` when files were installed or updated, otherwise
            ``reused``.
        workspace: Selected workspace directory.
        elapsed_seconds: Time used to select and prepare the workspace.
        data_raw: Directory for source inputs.
        data_processed: Directory for files reused by later calculations.
        results: Directory for MFA, LCA, and optimization results.
        parameter_catalogue: Installed foreground parameter catalog.
        default_input_parameters: File containing the default foreground parameter values.
        default_annual_input_values: CSV file of annual default foreground parameter values.
        foreground_scenario_templates: Directory containing scenario templates.
        foreground_scenarios: Directory containing runnable scenario definitions.
        allocated_cc_templates: Directory containing allocated carrying capacity
            input templates.
        allocated_cc_results: Directory containing user-provided allocated
            carrying capacity results.
        methodological_notes: Directory containing the LCI workbook, article
            and Methodological appendix PDFs, and citation guide.
        recommended_citations: Citation guide installed with the method notes.
    """

    status: OperationStatus
    workspace: Path
    elapsed_seconds: float
    data_raw: Path
    data_processed: Path
    results: Path
    parameter_catalogue: Path
    default_input_parameters: Path
    default_annual_input_values: Path
    foreground_scenario_templates: Path
    foreground_scenarios: Path
    allocated_cc_templates: Path
    allocated_cc_results: Path
    methodological_notes: Path
    recommended_citations: Path

    def as_dict(self) -> dict[str, object]:
        """Serialize the report, converting every :class:`~pathlib.Path` to text."""

        payload = asdict(self)
        for field in (
            "workspace",
            "data_raw",
            "data_processed",
            "results",
            "parameter_catalogue",
            "default_input_parameters",
            "default_annual_input_values",
            "foreground_scenario_templates",
            "foreground_scenarios",
            "allocated_cc_templates",
            "allocated_cc_results",
            "methodological_notes",
            "recommended_citations",
        ):
            payload[field] = str(payload[field])
        return payload


def set_workspace(
    workspace: str | Path,
    *,
    refresh: bool = False,
) -> WorkspaceReport:
    """Set the active workspace and install packaged input files.

    The function resolves ``workspace`` as the workspace root, creates that
    root when missing, copies packaged input files according to ``refresh``,
    and records the resolved path as the active workspace for the current
    Python session. Later public functions read inputs and write outputs
    relative to this active workspace.

    Packaged files include foreground scenarios and templates, parameter
    catalogs, input schemas, allocated carrying capacity templates, workspace
    guides, methodological notes, and citation guidance.

    Args:
        workspace: Workspace directory created or reused by this call.
            Accepts a string path or ``pathlib.Path``.
        refresh: If ``True``, overwrite the packaged catalogs, templates,
            schemas, guides, methodological notes, and example scenarios.
            If ``False``, copy missing packaged files and preserve existing
            files. Defaults to ``False``.

    Returns:
        ``WorkspaceReport`` describing the selected workspace, installed input
        paths, completion status, and elapsed time in seconds.

    Raises:
        ValueError: If ``workspace`` is empty or names a non-directory path,
            or ``refresh`` is not a Boolean.
        OSError: If workspace directory or prerequisite file creation, copying,
            or refresh fails.

    Example:
        Initialize a workspace::

            from pathlib import Path
            from pytoce import set_workspace

            report = set_workspace(Path("pytoce_workspace"))
    """

    if not isinstance(refresh, bool):
        raise ValueError(
            f"refresh={refresh!r} is invalid. Use refresh=True to replace packaged "
            "workspace files or refresh=False to preserve existing files."
        )

    from .runtime.installation import (
        ALLOCATED_CC_RESULTS_RELATIVE,
        ALLOCATED_CC_TEMPLATES_RELATIVE,
        CATALOGUE_RELATIVE,
        DEFAULT_ANNUAL_INPUT_VALUES_RELATIVE,
        DEFAULT_INPUT_RELATIVE,
        METHODOLOGICAL_NOTES_RELATIVE,
        RECOMMENDED_CITATIONS_RELATIVE,
        SCENARIOS_RELATIVE,
        TEMPLATES_RELATIVE,
        install_package_files,
        package_files,
    )
    from .runtime.workspace import (
        ensure_roots,
        render_summary,
        resolve_workspace,
        set_active_workspace,
    )

    started = perf_counter()
    root = resolve_workspace(workspace)
    ensure_roots(root)
    files = package_files()
    installed = install_package_files(root, files, refresh=refresh)
    status: OperationStatus = "computed" if installed else "reused"
    set_active_workspace(root)
    print(
        render_summary(status=status, workspace=root, installed_count=installed),
        file=sys.stderr,
    )
    return WorkspaceReport(
        status=status,
        workspace=root,
        elapsed_seconds=perf_counter() - started,
        data_raw=root / "data_raw",
        data_processed=root / "data_processed",
        results=root / "results",
        parameter_catalogue=root / CATALOGUE_RELATIVE,
        default_input_parameters=root / DEFAULT_INPUT_RELATIVE,
        default_annual_input_values=root / DEFAULT_ANNUAL_INPUT_VALUES_RELATIVE,
        foreground_scenario_templates=root / TEMPLATES_RELATIVE,
        foreground_scenarios=root / SCENARIOS_RELATIVE,
        allocated_cc_templates=root / ALLOCATED_CC_TEMPLATES_RELATIVE,
        allocated_cc_results=root / ALLOCATED_CC_RESULTS_RELATIVE,
        methodological_notes=root / METHODOLOGICAL_NOTES_RELATIVE,
        recommended_citations=root / RECOMMENDED_CITATIONS_RELATIVE,
    )
