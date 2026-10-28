"""Package-owned workspace input discovery and installation."""

import json
from collections.abc import Iterator
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from pytoce.shared.runtime.filesystem import atomic_write
from pytoce.shared.runtime.workspace_guides import (
    load_workspace_guide_catalogue,
    render_workspace_guide,
)

CATALOGUE_RELATIVE = Path("data_raw/foreground_scenarios/default_inputs/parameter_catalogue.json")
DEFAULT_INPUT_RELATIVE = Path(
    "data_raw/foreground_scenarios/default_inputs/default_input_values.parquet"
)
SCHEMAS_RELATIVE = Path("data_raw/foreground_scenarios/schemas")
TEMPLATES_RELATIVE = Path("data_raw/foreground_scenarios/templates")
SCENARIOS_RELATIVE = Path("data_raw/foreground_scenarios/scenarios")
ALLOCATED_CC_TEMPLATES_RELATIVE = Path("data_raw/allocated_cc/templates")
ALLOCATED_CC_SCHEMAS_RELATIVE = Path("data_raw/allocated_cc/schemas")
ALLOCATED_CC_RESULTS_RELATIVE = Path("data_raw/allocated_cc/results")
METHODOLOGICAL_NOTES_RELATIVE = Path("data_raw/methodological_notes")
RECOMMENDED_CITATIONS_RELATIVE = METHODOLOGICAL_NOTES_RELATIVE / "recommended_citations.txt"
DEFAULT_ANNUAL_INPUT_VALUES_RELATIVE = TEMPLATES_RELATIVE / "default_annual_input_values.csv"
_RESOURCE_ROOT = PurePosixPath("workspace")
_OPTIMIZATION_DATA_PACKAGE = "pytoce.run_optimization.data"
_SET_WORKSPACE_DATA_PACKAGE = "pytoce.set_workspace.data"
_PARAMETER_CATALOGUE_RESOURCE = PurePosixPath("catalogues/foreground_parameter_catalogue.json")
_SCHEMA_RESOURCE_ROOT = PurePosixPath("schemas")
_METHODOLOGICAL_NOTES_RESOURCE_ROOT = PurePosixPath("methodological_notes")

PackageFile = tuple[Path, bytes]


def package_files() -> tuple[PackageFile, ...]:
    """Collect the files installed by ``set_workspace`` in deterministic order.

    The returned paths cover the foreground parameter catalog, templates, schemas,
    methodological notes, and workspace guides. Packaged example scenarios are
    included. Refresh applies to the listed package resources.

    Returns:
        Relative workspace paths paired with their installed bytes.
    """

    package_root = resources.files("pytoce.run_mfa.data")
    workspace_root = package_root
    for part in _RESOURCE_ROOT.parts:
        workspace_root = workspace_root.joinpath(part)

    def visit(current: Any, prefix: Path) -> Iterator[PackageFile]:
        """Yield every installed workspace resource below one package directory."""
        for child in sorted(current.iterdir(), key=lambda item: item.name):
            relative = prefix / child.name
            if child.is_dir():
                yield from visit(child, relative)
            else:
                yield relative, child.read_bytes()

    found = list(visit(workspace_root, Path()))
    catalogue = package_root
    for part in _PARAMETER_CATALOGUE_RESOURCE.parts:
        catalogue = catalogue.joinpath(part)
    found.append((CATALOGUE_RELATIVE, catalogue.read_bytes()))
    schema_root = package_root
    for part in _SCHEMA_RESOURCE_ROOT.parts:
        schema_root = schema_root.joinpath(part)
    found.extend(
        (SCHEMAS_RELATIVE / schema.name, schema.read_bytes())
        for schema in sorted(schema_root.iterdir(), key=lambda item: item.name)
        if schema.is_file() and schema.name.endswith(".json")
    )
    optimization_package_root = resources.files(_OPTIMIZATION_DATA_PACKAGE)
    optimization_schema_root = optimization_package_root
    for part in _SCHEMA_RESOURCE_ROOT.parts:
        optimization_schema_root = optimization_schema_root.joinpath(part)
    found.extend(
        (ALLOCATED_CC_SCHEMAS_RELATIVE / schema.name, schema.read_bytes())
        for schema in sorted(optimization_schema_root.iterdir(), key=lambda item: item.name)
        if schema.is_file() and schema.name.endswith(".json")
    )
    optimization_root = optimization_package_root
    for part in _RESOURCE_ROOT.parts:
        optimization_root = optimization_root.joinpath(part)
    found.extend(visit(optimization_root, Path()))
    notes_root = resources.files(_SET_WORKSPACE_DATA_PACKAGE)
    for part in _METHODOLOGICAL_NOTES_RESOURCE_ROOT.parts:
        notes_root = notes_root.joinpath(part)
    found.extend(
        (METHODOLOGICAL_NOTES_RELATIVE / note.name, note.read_bytes())
        for note in sorted(notes_root.iterdir(), key=lambda item: item.name)
        if note.is_file()
    )
    for guide in load_workspace_guide_catalogue("pytoce.run_optimization"):
        workspace_path = str(guide["workspace_path"])
        if "<" in workspace_path:
            continue
        found.append(
            (
                Path(workspace_path) / "README.md",
                render_workspace_guide(
                    "pytoce.run_optimization",
                    str(guide["guide_id"]),
                ),
            )
        )
    for guide in load_workspace_guide_catalogue("pytoce.set_workspace"):
        guide_id = str(guide["guide_id"])
        found.append(
            (
                Path(str(guide["workspace_path"])) / "README.md",
                render_workspace_guide("pytoce.set_workspace", guide_id),
            )
        )
    return tuple(found)


def _default_annual_input_values(workspace: Path) -> bytes:
    """Render installed default values with their manifest's source identity."""
    authority_path = workspace / DEFAULT_INPUT_RELATIVE
    authority = pd.read_parquet(authority_path)
    manifest = json.loads(
        authority_path.with_name("default_input_values_manifest.json").read_bytes()
    )
    readable = pd.DataFrame(
        {
            "parameter_id": authority["parameter_id"],
            "year": authority["year"],
            "region": authority["region"].fillna(""),
            "material": authority["material"].fillna(""),
            "good": authority["good"].fillna(""),
            "process_or_route": authority["process"].fillna(authority["route"]).fillna(""),
            "unit": authority["unit"],
            "default_value": authority["default_value"],
            "default_source_identity": manifest["identity"],
        }
    )
    return readable.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")


def install_package_files(
    workspace: Path,
    files: tuple[PackageFile, ...],
    *,
    refresh: bool,
) -> int:
    """Install the supplied package files at their declared paths."""

    installed = 0
    for relative, content in files:
        target = workspace / relative
        if target.exists() and not refresh:
            continue
        atomic_write(target, content)
        installed += 1
    default_values = workspace / DEFAULT_ANNUAL_INPUT_VALUES_RELATIVE
    if refresh or not default_values.exists():
        atomic_write(default_values, _default_annual_input_values(workspace))
        installed += 1
    return installed
