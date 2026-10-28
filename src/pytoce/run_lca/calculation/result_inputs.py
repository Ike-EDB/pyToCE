"""Select completed LCA results independently of upstream contribution years."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from pytoce.run_lca.run_lca import DEFAULT_UPSTREAM_ANALYSIS_YEARS, lca_result_directory

from .lcia import LCIAMethodSelection


class LCAResultError(ValueError):
    """A required LCA result is missing or its scientific parent is ambiguous."""


@dataclass(frozen=True, slots=True)
class LCAResultReference:
    """A selected completion manifest and the directory it describes."""

    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str


def select_lca_result(
    workspace: Path,
    *,
    foreground_scenario: str,
    background_scenario: str,
    selection: LCIAMethodSelection,
) -> LCAResultReference:
    """Select the exact requested LCIA product.

    Equivalent reporting variants must describe identical scientific parents
    after removing only upstream_analysis_year. Prefer the default-year result,
    then the first readable name. Incomplete products remain untouched.
    """
    default_directory = lca_result_directory(
        workspace,
        foreground_scenario,
        background_scenario,
        selection,
        DEFAULT_UPSTREAM_ANALYSIS_YEARS,
    )
    prefix = default_directory.name.rsplit("__upstream_analysis_", 1)[0]
    directories = sorted(
        default_directory.parent.glob(f"{prefix}__upstream_analysis_*"),
        key=lambda directory: (directory != default_directory, directory.name),
    )
    completed = [
        (directory, (directory / "manifest.json").read_bytes())
        for directory in directories
        if (directory / "manifest.json").is_file()
    ]
    if not completed:
        paths = ", ".join(str(directory) for directory in directories or [default_directory])
        raise LCAResultError(
            f"The required LCA result has no completion manifest: {paths}. "
            f"Run run_lca for foreground_scenario={foreground_scenario!r} and "
            f"background_scenario={background_scenario!r} with the requested lcia_method "
            "before run_asr or run_optimization. "
            "Existing incomplete products have not been changed."
        )
    manifests = [json.loads(content) for _directory, content in completed]
    scientific_parents = [
        {
            key: value
            for key, value in item["identity_payload"].items()
            if key != "upstream_analysis_year"
        }
        for item in manifests
    ]
    if any(parent != scientific_parents[0] for parent in scientific_parents[1:]):
        paths = ", ".join(str(directory) for directory, _content in completed)
        raise LCAResultError(
            f"Ambiguous baseline LCA results have different scientific parents: {paths}. "
            "Use run_lca(refresh=True) with each listed upstream_analysis_year selection "
            "to recalculate them from the intended workspace parents, or use a new workspace. "
            "No existing product has been changed."
        )
    directory, manifest_bytes = completed[0]
    return LCAResultReference(directory, manifests[0], sha256(manifest_bytes).hexdigest())
