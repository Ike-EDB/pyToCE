"""Execute REMIND-MFA 0.2.0 through its public model API."""

import importlib
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, cast

import flodym as fd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import yaml

from pytoce.shared.runtime.filesystem import (
    atomic_write,
    canonical_json_bytes,
    payload_sha256,
)
from pytoce.shared.runtime.workspace_guides import write_workspace_guide

from .process_inputs import (
    RemindMFAProcessInputs,
    calculate_process_inputs,
    default_waste_trade_propensities,
)
from .trade_projection import HistoricalTradeProjection
from .translation import (
    RemindMFAIdentity,
    apply_parameter_update,
    normalize_cohort_state,
)

_CODE_COMMIT = "bf95c898e0f6f363196bccb62d34e02d82e4726b"
_CODE_TREE = "c16eacdb66a1082891fbbbca62a636993c245711"
_CODE_ARCHIVE_SHA256 = "eee4f40d0c8512c6edd8d5e7a7993e5708f3bfc13a31e72cbe42ff26a59fea24"
_DATA_ARCHIVE_SHA256 = "73bcf80f53513dcdf6f946afabb333a13792f087029550a95ea6c52f8d068419"
_DATA_REVISION = "rev1.0.0_h12_1584d780_mfa"
_MRMFA_COMMIT = "8135ab7a96214b12e9ecc3b3949dac75cccb6828"
REMIND_MFA_ADAPTER_METHOD = "pytoce_remind_mfa_stock_and_trade_v3"
_SOURCE_SCHEMA = "pytoce_remind_mfa_source_schema_v1"
_CODE_RUNTIME_TREE_SHA256 = "9cc11deb4ea459f86136f0fa3f3819c0778aa82e83a35221ae328d6b10e356f2"
_DATA_TREE_SHA256 = "899d57ae67ac7214dc54f99c52b9e720fb347137b4bc54637a8e45b59d573ea8"


class RemindMFAExecutionError(RuntimeError):
    """REMIND-MFA 0.2.0 source preparation or execution cannot complete."""


@dataclass(frozen=True, slots=True)
class RemindMFASource:
    """Exact extracted source and input data used by one execution."""

    code_root: Path
    data_root: Path
    code_archive_sha256: str
    data_archive_sha256: str
    source_manifest: Path


@dataclass(frozen=True, slots=True)
class RemindMFAExecutionRequest:
    """REMIND-MFA 0.2.0 source and configuration for default stock and trade calculations."""

    source: RemindMFASource
    configuration_yaml: str
    identity: RemindMFAIdentity


def _safe_member(name: str) -> PurePosixPath:
    """Reject an archive member that escapes the extraction directory."""
    normalized = PurePosixPath(name.replace("\\", "/"))
    if (
        not normalized.parts
        or normalized.is_absolute()
        or ".." in normalized.parts
        or ":" in normalized.parts[0]
    ):
        raise RemindMFAExecutionError("REMIND-MFA archive contains an unsafe member path")
    return normalized


def _extract_code(archive: Path, destination: Path) -> Path:
    """Extract one verified REMIND-MFA source archive to a new directory."""
    try:
        with tarfile.open(archive, mode="r:gz") as source:
            members = source.getmembers()
            roots: set[str] = set()
            for member in members:
                path = _safe_member(member.name)
                roots.add(path.parts[0])
                if member.issym() or member.islnk() or member.isdev():
                    raise RemindMFAExecutionError(
                        "REMIND-MFA source archive contains unsupported links or devices"
                    )
            if len(roots) != 1:
                raise RemindMFAExecutionError("REMIND-MFA source archive has no unique root")
            source.extractall(destination)
    except (OSError, tarfile.TarError) as exc:
        raise RemindMFAExecutionError("REMIND-MFA source archive cannot be extracted") from exc
    return destination / next(iter(roots))


def _extract_data(archive: Path, destination: Path) -> None:
    """Extract regular members from the checked REMIND-MFA data archive."""
    try:
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                _safe_member(item.filename)
                unix_mode = item.external_attr >> 16
                if unix_mode & 0o170000 == 0o120000:
                    raise RemindMFAExecutionError("REMIND-MFA data archive contains a symlink")
            source.extractall(destination)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RemindMFAExecutionError("REMIND-MFA data archive cannot be extracted") from exc


def _source_artifact(workspace: Path, source_id: str, filename: str) -> Path:
    """Locate a REMIND-MFA archive verified by ``download_data``."""
    root = workspace / "data_raw" / "remind_mfa"
    manifest = root / "manifest.json"
    artifact = root / filename
    if not manifest.is_file() or not artifact.is_file():
        raise FileNotFoundError(
            f"The REMIND-MFA source {source_id!r} is missing from {root}. Run "
            "download_data() in this workspace, then run run_mfa() again."
        )
    return artifact


def prepare_workspace_source(workspace: str | Path) -> RemindMFASource:
    """Extract downloaded sources once and reuse the completed processed directory."""

    root = Path(workspace).expanduser().resolve()
    target = root / "data_processed" / "remind_mfa"
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        code_root = target / "code"
        data_root = target / "data"
        return RemindMFASource(
            code_root=code_root,
            data_root=data_root,
            code_archive_sha256=_CODE_ARCHIVE_SHA256,
            data_archive_sha256=_DATA_ARCHIVE_SHA256,
            source_manifest=manifest_path,
        )
    if target.exists() or target.is_symlink():
        raise RemindMFAExecutionError(
            f"The extracted REMIND-MFA directory {target} has no completion manifest. "
            "Keep it unchanged and create a new workspace."
        )
    identity_payload = {
        "schema_version": _SOURCE_SCHEMA,
        "code_commit": _CODE_COMMIT,
        "code_tree": _CODE_TREE,
        "code_runtime_tree_sha256": _CODE_RUNTIME_TREE_SHA256,
        "code_archive_sha256": _CODE_ARCHIVE_SHA256,
        "data_revision": _DATA_REVISION,
        "data_tree_sha256": _DATA_TREE_SHA256,
        "data_archive_sha256": _DATA_ARCHIVE_SHA256,
    }
    identity = f"{_SOURCE_SCHEMA}::{payload_sha256(identity_payload)}"
    manifest_payload = {
        **identity_payload,
        "identity": identity,
        "code_root": "code",
        "data_root": "data",
        "complete": True,
    }
    expected_manifest = canonical_json_bytes(manifest_payload)
    code_archive = _source_artifact(
        root,
        "remind_mfa_code",
        "remind_mfa_v0_2_0.tar.gz",
    )
    data_archive = _source_artifact(
        root,
        "remind_mfa_input_data",
        "input_data_1_0_0.zip",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".remind_mfa_source.", dir=target.parent))
    try:
        extracted_code = _extract_code(code_archive, staging / "_code")
        _extract_data(data_archive, staging / "data")
        shutil.move(str(extracted_code), str(staging / "code"))
        shutil.rmtree(staging / "_code")
        write_workspace_guide(
            staging,
            "pytoce.run_mfa",
            "normalized_remind_mfa",
        )
        atomic_write(staging / "manifest.json", expected_manifest)
        os.replace(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return RemindMFASource(
        code_root=target / "code",
        data_root=target / "data",
        code_archive_sha256=_CODE_ARCHIVE_SHA256,
        data_archive_sha256=_DATA_ARCHIVE_SHA256,
        source_manifest=manifest_path,
    )


def _configuration(source: RemindMFASource) -> dict[str, Any]:
    """Load the REMIND-MFA 0.2.0 plastics configuration and disable upstream exports."""
    path = source.code_root / "config" / "plastics.yml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RemindMFAExecutionError(
            "The REMIND-MFA 0.2.0 plastics configuration is unreadable"
        ) from exc
    payload["input"]["input_data_path"] = str(source.data_root)
    payload["input"]["scenarios_path"] = str(source.code_root / "config" / "scenarios")
    payload["model_switches"]["scenario"] = "SSP2"
    payload.setdefault("export", {})["do_export"] = False
    for key in ("pickle", "csv", "iamc", "assumptions", "docs"):
        payload["export"].setdefault(key, {})["do_export"] = False
    payload.setdefault("visualization", {})["do_visualize"] = False
    payload["visualization"]["do_show_figs"] = False
    for value in payload["visualization"].values():
        if isinstance(value, dict) and "do_visualize" in value:
            value["do_visualize"] = False
    return payload


def _dimension_schema_sha256(data_root: Path) -> str:
    """Hash the ordered REMIND-MFA dimension and regional mapping files."""
    plastics = data_root / "plastics"
    paths = [
        *sorted((plastics / "dimensions").glob("*.csv")),
        plastics / "input_data" / "regionmappingH12.csv",
    ]
    digest = sha256()
    for path in paths:
        digest.update(path.relative_to(plastics).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_execution(source: RemindMFASource) -> RemindMFAExecutionRequest:
    """Bind exact source, configuration, and identities before computation."""

    configuration = _configuration(source)
    neutral = yaml.safe_load(yaml.safe_dump(configuration))
    neutral["input"]["input_data_path"] = "<remind_mfa_data_root>"
    neutral["input"]["scenarios_path"] = "<remind_mfa_code_root>/config/scenarios"
    configuration_sha256 = payload_sha256(neutral)
    identity = RemindMFAIdentity(
        repository="https://github.com/pik-piam/remind-mfa",
        tag="v0.2.0",
        commit=_CODE_COMMIT,
        tree=_CODE_TREE,
        input_version="1.0.0",
        input_revision=_DATA_REVISION,
        input_archive_sha256=_DATA_ARCHIVE_SHA256,
        mrmfa_version="1.1.0",
        mrmfa_commit=_MRMFA_COMMIT,
        madrat_version="3.28.4",
        configuration_sha256=configuration_sha256,
        dimension_schema_sha256=_dimension_schema_sha256(source.data_root),
        scenario="SSP2",
        adapter_identity=REMIND_MFA_ADAPTER_METHOD,
    )
    return RemindMFAExecutionRequest(
        source=source,
        configuration_yaml=yaml.safe_dump(configuration, sort_keys=True),
        identity=identity,
    )


def _activate_source(source: RemindMFASource) -> None:
    """Place the REMIND-MFA 0.2.0 source first on the import path."""
    resolved_root = source.code_root.resolve()
    code_root = str(resolved_root)
    for name, module in tuple(sys.modules.items()):
        if name != "remind_mfa" and not name.startswith("remind_mfa."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        module_path = Path(str(module_file)).resolve()
        if resolved_root not in module_path.parents:
            raise RemindMFAExecutionError(
                "A different REMIND-MFA source is already imported in this process"
            )
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != resolved_root]
    sys.path.insert(0, code_root)
    importlib.invalidate_caches()


def _import_source_module(source: RemindMFASource, module_name: str) -> ModuleType:
    """Import one module and verify that it came from the specified source tree."""
    _activate_source(source)
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RemindMFAExecutionError(
            f"Imported REMIND-MFA module has no source path: {module_name}"
        )
    module_path = Path(str(module_file)).resolve()
    if source.code_root.resolve() not in module_path.parents:
        raise RemindMFAExecutionError(f"Imported REMIND-MFA module path differs: {module_name}")
    return module


def _copy_parameter(parameter: fd.Parameter) -> fd.Parameter:
    """Copy a FLODYM parameter into an independent mutable value array."""
    return fd.Parameter(
        dims=parameter.dims,
        name=parameter.name,
        values=np.array(parameter.values, dtype=float, copy=True),
    )


class RemindMFAExecutionSession:
    """Admit historical stock and trade inputs for the direct MFA.

    REMIND-MFA 0.2.0 owns historical in-use stock, balanced historical trade,
    socioeconomic stock targets and parameter extrapolation. A single public
    FLODYM stock-driven reconstruction preserves the source's historical
    rounding and supplies the pyToCE cohort boundary. Candidate flows and
    disposal/carbon accumulation belong to the direct pyToCE MFA calculation.
    """

    def __init__(self, request: RemindMFAExecutionRequest) -> None:
        """Prepare default cohort and trade inputs from the authenticated source."""
        extrapolation = _import_source_module(
            request.source, "remind_mfa.common.parameter_extrapolation"
        )
        plastics_model = _import_source_module(request.source, "remind_mfa.plastics.plastics_model")
        self._trade_api = _import_source_module(request.source, "remind_mfa.common.trade")
        regression = _import_source_module(request.source, "remind_mfa.common.data_extrapolations")
        self._trade_projection = HistoricalTradeProjection(regression.ProportionalExtrapolation)
        try:
            model = plastics_model.PlasticsModel(cfg=yaml.safe_load(request.configuration_yaml))
        except OverflowError as exc:
            # FLODYM 0.8.1 tests numeric columns against integer dimensions;
            # infinity can fail there before loaded parameter arrays exist.
            raise ValueError(
                f"Invalid REMIND-MFA input in {request.source.data_root}: {exc}. "
                "Use finite numeric values in the parameter CSV."
            ) from exc
        for name, parameter in model.parameters.items():
            if not np.isfinite(parameter.values).all():
                raise ValueError(
                    f"REMIND-MFA parameter {name!r} in {request.source.data_root} must "
                    "contain only finite numbers. Correct its source CSV."
                )
            if name.endswith(("_his_imports", "_his_exports")):
                invalid = np.argwhere(parameter.values < 0.0)
                if len(invalid):
                    index = tuple(int(value) for value in invalid[0])
                    coordinates = {
                        dimension.name: dimension.items[position]
                        for dimension, position in zip(parameter.dims.dim_list, index, strict=True)
                    }
                    path = request.source.data_root / "plastics/input_data" / f"pl_{name}.cs4r"
                    raise ValueError(
                        f"REMIND-MFA parameter {name!r} in {path} is {parameter.values[index]} "
                        f"at {coordinates}; gross historical trade must be nonnegative. "
                        "Correct its source CSV."
                    )
        # An infinite target year can leave finite, unchanged rates after the
        # source's linear blend. Admit the used targets before extrapolation.
        for name, method in model.cfg.model_switches.parameter_extrapolation_classes.items():
            if method is extrapolation.LinearToTargetExtrapolation:
                for suffix in ("_target_year", "_target_value"):
                    target = name + suffix
                    if not np.isfinite(model.scenario_parameters[target].values).all():
                        raise ValueError(
                            f"REMIND-MFA scenario parameter {target!r} in "
                            f"{model.cfg.input.scenarios_path} must contain only finite numbers. "
                            "Correct its source scenario YAML."
                        )
        if np.any(model.parameters["emission_capture_rate"].values != 0.0):
            raise ValueError(
                f"REMIND-MFA emission_capture_rate in {request.source.data_root} must be zero. "
                "Use the unmodified downloaded inputs."
            )
        model.historic_mfa = model.make_mfa(historic=True)
        model.historic_mfa.compute()
        self._model = model
        self._source_parameters = {
            name: _copy_parameter(parameter) for name, parameter in model.parameters.items()
        }
        stock_target = model.get_long_term_stock()
        self._future_parameters = extrapolation.ParameterExtrapolationManager(
            model.cfg, model.dims["t"]
        ).apply_prm_extrapolation(model.parameters, model.scenario_parameters)
        stock_dims = model.dims[("t", "r", "g")]
        lifetime = fd.LogNormalLifetime(dims=stock_dims)
        lifetime.set_prms(
            mean=self._future_parameters["lifetime_mean"],
            std=self._future_parameters["lifetime_std"],
        )
        stock = fd.StockDrivenDSM(dims=stock_dims, lifetime_model=lifetime)
        cast(fd.StockArray, stock.stock)[...] = stock_target
        stock.compute()
        self.cohort_state = normalize_cohort_state(stock, self._future_parameters)
        # Preserve the source's multiply-before-reduction order and contiguous
        # t,e,r,m,g layout when calculating identity-defining waste propensities.
        split = (
            self._future_parameters["material_shares_in_goods"]
            * self._future_parameters["carbon_content_materials"]
        )
        outflow = fd.FlodymArray(dims=model.dims[("t", "e", "r", "m", "g")])
        outflow[...] = stock.outflow * split
        self._default_waste_trade_propensities = default_waste_trade_propensities(
            outflow, self._future_parameters
        )

    def resolve_stock_target(self, translated_inputs: dict[str, pd.DataFrame]) -> np.ndarray:
        """Resolve a 2020 through 2100 stock target after socioeconomic input changes."""
        model = self._model
        model.parameters = {
            name: _copy_parameter(parameter) for name, parameter in self._source_parameters.items()
        }
        for name in ("population", "gdppc"):
            apply_parameter_update(model.parameters[name], translated_inputs[name])
        target = model.get_long_term_stock()
        values = np.asarray(target.values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                "Population and GDP per capita scenario values produce a nonfinite "
                "REMIND-MFA stock target in Mt. Reduce these values in the foreground scenario."
            )
        return values[70:].copy()

    def process_inputs(
        self,
        translated_inputs: dict[str, pd.DataFrame],
        *,
        carbon_inflow: np.ndarray,
        carbon_outflow: np.ndarray,
        deferred_waste_trade: pd.DataFrame,
        record_diagnostics: bool = True,
    ) -> RemindMFAProcessInputs:
        """Prepare candidate trade and rates on the admitted source coordinates.

        Element inflow and outflow use year, element, region, material and good
        axes in Mt. Fresh parameter arrays and trade objects prevent one
        candidate's source updates from changing a later evaluation.
        """
        parameters = {
            name: _copy_parameter(parameter) for name, parameter in self._future_parameters.items()
        }
        for name, update in translated_inputs.items():
            if name not in {"population", "gdppc"}:
                apply_parameter_update(parameters[name], update)
        dimensions = self._model.dims
        flow_dims = dimensions[("t", "e", "r", "m", "g")]
        return calculate_process_inputs(
            inflow=fd.FlodymArray(dims=flow_dims, values=np.ascontiguousarray(carbon_inflow)),
            outflow=fd.FlodymArray(dims=flow_dims, values=np.ascontiguousarray(carbon_outflow)),
            parameters=parameters,
            historical_trade=self._model.historic_mfa.trade_set.markets,
            dimensions=dimensions,
            trade_api=self._trade_api,
            trade_projection=self._trade_projection,
            deferred_waste_trade=deferred_waste_trade,
            default_propensities=self._default_waste_trade_propensities,
            record_diagnostics=record_diagnostics,
        )
