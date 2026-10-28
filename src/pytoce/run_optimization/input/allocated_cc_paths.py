"""Check allocated carrying capacity paths, metadata, and selections."""

import re
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import yaml

_ALLOCATED_CC_SCHEMA_VERSION = "pytoce_allocated_cc_result_schema_v1"
_RAW_RELATIVE = Path("data_raw/allocated_cc/results")
_CASE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "allocated_cc_result",
        "description",
        "ssp_scenario",
        "lcia_method",
        "annual_allocated_cc",
        "cumulative_gwp100_allocated_cc",
    }
)
_REQUIRED_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS


class AllocatedCCError(ValueError):
    """An allocated carrying capacity input is invalid or unavailable."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    """Build a mapping and reject duplicate keys."""
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "allocated_cc mapping keys must be text",
                key_node.start_mark,
            )
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


def _readable_name(value: object, *, field: str) -> str:
    """Validate and return one lowercase ``snake_case`` result or case name."""
    if not isinstance(value, str) or not _CASE_IDENTIFIER.fullmatch(value):
        raise AllocatedCCError(f"{field} must be a lowercase snake_case identifier")
    return value


def _secure_results_root(workspace: Path) -> Path:
    """Resolve the aCC results directory without aliases."""

    try:
        workspace_root = workspace.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AllocatedCCError(
            f"The selected workspace is unavailable: {workspace}. Keep it unchanged and create "
            "a new v1 workspace with set_workspace."
        ) from exc
    expected = workspace_root / _RAW_RELATIVE
    if not expected.exists() and not expected.is_symlink():
        raise AllocatedCCError(
            f"The allocated carrying capacity (aCC) results directory is missing: {expected}. "
            "Run set_workspace for a new v1 workspace before adding an aCC result."
        )
    try:
        root = expected.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AllocatedCCError(
            f"Cannot resolve the allocated carrying capacity (aCC) results directory: "
            f"{expected}. Keep this path unchanged and create a new v1 workspace."
        ) from exc
    if root != expected or not root.is_dir():
        raise AllocatedCCError(
            "The allocated carrying capacity (aCC) results directory must be a regular "
            f"directory, not a link or junction: {expected}. Keep this path unchanged and "
            "create a new v1 workspace."
        )
    return root


def _secure_source_directory(workspace: Path, allocated_cc_result: str) -> Path:
    """Resolve one named aCC result as a direct child of the results directory."""

    results_root = _secure_results_root(workspace)
    expected = results_root / allocated_cc_result
    try:
        source = expected.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        available = sorted(path.name for path in results_root.iterdir() if path.is_dir())
        raise AllocatedCCError(
            f"Unknown allocated carrying capacity (aCC) result {allocated_cc_result!r} in "
            f"{results_root}. Choose one of {available!r} or add a v1 aCC result."
        ) from exc
    if source != expected or not source.is_dir():
        raise AllocatedCCError(
            "The allocated carrying capacity (aCC) result must be a regular directory directly "
            f"inside {results_root}: {expected}. Choose another result, or copy the YAML and "
            "two CSV files into a regular directory under this results directory."
        )
    return source


def _text(value: object, *, field: str) -> str:
    """Return one nonempty text field without surrounding whitespace."""

    if not isinstance(value, str):
        raise AllocatedCCError(f"{field} must be text")
    result = value.strip()
    if not result:
        raise AllocatedCCError(f"{field} must not be empty")
    return result


def _allocated_cc_metadata(path: Path, allocated_cc_result: str) -> tuple[dict[str, object], str]:
    """Validate one result's YAML and hash the same bytes used for parsing."""
    try:
        content = path.read_bytes()
        raw = yaml.load(content.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise AllocatedCCError(
            f"Cannot read allocated carrying capacity metadata file: {path}. "
            "Replace it with a readable copy of the installed v1 YAML template."
        ) from exc
    if not isinstance(raw, dict):
        raise AllocatedCCError("allocated_cc_result.yaml must contain one mapping")
    payload = cast(dict[str, object], raw)
    unknown = set(payload) - _TOP_LEVEL_KEYS
    missing = _REQUIRED_TOP_LEVEL_KEYS - set(payload)
    if unknown:
        raise AllocatedCCError(
            f"Unknown allocated_cc_result.yaml field(s): {sorted(unknown)!r}. "
            f"Use only {sorted(_TOP_LEVEL_KEYS)!r}."
        )
    if missing:
        raise AllocatedCCError(
            f"allocated_cc_result.yaml is missing required field(s): {sorted(missing)!r}. "
            "Copy the installed v1 template and fill every required field."
        )
    if payload["schema_version"] != _ALLOCATED_CC_SCHEMA_VERSION:
        raise AllocatedCCError(
            f"schema_version={payload['schema_version']!r} is not supported; expected "
            f"{_ALLOCATED_CC_SCHEMA_VERSION!r}. Copy the installed v1 template and update it."
        )
    if payload["allocated_cc_result"] != allocated_cc_result:
        raise AllocatedCCError(
            f"allocated_cc_result={payload['allocated_cc_result']!r} must match its directory "
            f"name {allocated_cc_result!r}"
        )
    if payload["ssp_scenario"] != "SSP2":
        raise AllocatedCCError(
            f"ssp_scenario={payload['ssp_scenario']!r} is invalid. Use exactly 'SSP2'."
        )
    if payload["lcia_method"] != ["ef31"]:
        raise AllocatedCCError(
            f"lcia_method={payload['lcia_method']!r} is invalid. Use exactly ['ef31']; "
            "monitoring and target setting use EF v3.1 for every aCC assessment."
        )
    payload["description"] = _text(payload["description"], field="description")
    return payload, sha256(content).hexdigest()


def _secure_source_file(source_directory: Path, relative_value: object, *, field: str) -> Path:
    """Resolve one declared CSV inside the aCC result directory."""
    relative_text = _text(relative_value, field=field)
    relative = Path(relative_text)
    if relative.is_absolute():
        raise AllocatedCCError(
            f"{field}={relative_text!r} must be a relative path to a regular CSV inside "
            f"{source_directory}"
        )
    root = source_directory.resolve()
    try:
        resolved = (source_directory / relative).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AllocatedCCError(
            f"Cannot find {field}={relative_text!r} inside {source_directory}. "
            "Add the required CSV or correct the YAML filename."
        ) from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise AllocatedCCError(
            f"{field}={relative_text!r} must identify a regular file inside {source_directory}"
        )
    return resolved


def _read_csv(path: Path, *, role: str) -> tuple[pd.DataFrame, str]:
    """Parse one external CSV and hash its captured bytes before frame admission."""
    # pandas cannot infer compression from captured bytes. Preserve its filename
    # convention, matching compound archive extensions before their final suffix.
    compression_suffixes: tuple[
        tuple[str, Literal["tar", "gzip", "bz2", "zip", "xz", "zstd"]], ...
    ] = (
        (".tar.gz", "tar"),
        (".tar.bz2", "tar"),
        (".tar.xz", "tar"),
        (".gz", "gzip"),
        (".bz2", "bz2"),
        (".zip", "zip"),
        (".xz", "xz"),
        (".zst", "zstd"),
        (".tar", "tar"),
    )
    compression: Literal["tar", "gzip", "bz2", "zip", "xz", "zstd"] | None = next(
        (method for suffix, method in compression_suffixes if path.name.lower().endswith(suffix)),
        None,
    )
    try:
        content = path.read_bytes()
        frame = pd.read_csv(
            BytesIO(content), dtype="object", keep_default_na=False, compression=compression
        )
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise AllocatedCCError(
            f"Cannot read {role}: {path}. Supply a readable UTF-8 CSV with the installed "
            "v1 template's header and complete aCC rows."
        ) from exc
    return frame, sha256(content).hexdigest()


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)
