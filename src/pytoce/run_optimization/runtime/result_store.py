"""Manifest-last storage for selected optimization results."""

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pytoce.run_optimization.blocks import OptimizationBlock
from pytoce.run_optimization.calculation.integrated_design import (
    INTEGRATED_DESIGN_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.integrated_search import (
    INTEGRATED_SEARCH_METHOD_IDENTITY,
)
from pytoce.run_optimization.calculation.request_search import (
    CompletedCandidateSearch,
    _execution_blocks,
    search_configuration_identity,
)
from pytoce.run_optimization.runtime.completed_search import decode_completed_search
from pytoce.shared.runtime.filesystem import (
    CanonicalTableHashSpool,
    _sync_directory,
    file_sha256,
    replace_directory,
)
from pytoce.shared.runtime.manifests import ItemStatus

SELECTED_RESULT_SCHEMA = "pytoce_optimization_selected_result_schema_v1"
_RESULT_SCHEMA_RESOURCE = "data/schemas/optimization_selected_result_schema_v1.json"
_WINDOWS_REPARSE_POINT = 0x0400
_PHYSICAL_PRODUCT_METHOD_NAME = "stock_lp_v6"
_SEARCH_PRODUCT_METHOD_NAME = "joint_lines_v1_stock_lp_v6"

_TABLE_ARTIFACTS = (
    "selected_coordinates.parquet",
    "selected_annual_interventions.parquet",
    "selection_roles.parquet",
    "selected_annual_acc_asr.parquet",
    "selected_environmental_scores.parquet",
    "selected_life_cycle_stage_scores.parquet",
    "selected_feedstock_requirements.parquet",
)
_SUMMARY_ARTIFACT = "method_summary.json"
_MANIFEST = "manifest.json"
_STAGE_ARTIFACT = "selected_life_cycle_stage_scores.parquet"
_STREAM_HASHES = ".selected_life_cycle_stage_scores.row_hashes"

_GUIDE = b"""# Optimization results

This directory contains the selected results written by `run_optimization`.
Result LCIA selections remain readable directory names. The manifest records
content identifiers for the calculation inputs and methods.

Every table is Parquet. `method_summary.json` records the method-specific
summary and compact completed-stage state for compatible strict-prefix extension.
`selected_coordinates.parquet` contains endpoints, completion years
and MFA outcomes. Declared scalar and coordinated endpoint moves define the
local search. Blocks II and III examine complete finite scalar and share-transfer
lines. Block III refines the best candidate for each state in each of its parent-
circularity and explicit-circularity domains, and retains complete finite compromise
routes between actual single-block choices and selected joint guides. Its additional
coordinated and guided moves use the current refinement step. All scored proposals
remain eligible for selection within their search.
The selected results describe the combinations evaluated through coarse pair
coverage and local searches. Demand-only demand-timing repair selects
one annual path per endpoint.
Join `selected_annual_interventions.parquet` by candidate identity for admitted or
unsuccessful diagnostic annual paths. Its columns are `candidate_identity`, `year`,
`region`, `parameter_id` and `value`. Values are fractions. Stock-
reduction and mean-lifetime extension have a null region and cover 2019 through 2100.
Collection, four treatment shares and three virgin-feedstock shares retain all
12 world regions for 2020 through 2100; 2019 remains the unchanged baseline. Exact-parent
circularity paths retain the parent's actual regional values. These controls describe the CE
performance trajectory.
Selected coordinates include each requested timing's MFA status and diagnostic
code. An infeasible timing retains its intervention path and selection-role links,
as a diagnostic path. Environmental, ASR, stage, and feedstock rows use
trajectories admitted by the MFA.
`selection_roles.parquet` maps each block, background and allocated carrying
capacity label to selected 2050 and delayed request and MFA identities. Join
`selection_candidate_identity` to coordinate `candidate_identity`; join score and
feedstock `physical_candidate_identity` to the coordinate MFA identity. In the
stage table, literal `candidate_identity` contains that MFA identity. Keep all
role links and count each distinct MFA result once in aggregation. Compare
complete MFA trajectories and inventories through their corresponding identities
and records.
Selection stability remains distinct from `declared_neighborhood_verified`.

Endpoint columns are percentage targets; annual controls are fractions. Explicit
treatment and feedstock shares each total 100 percent. Null circularity endpoints
mean unchanged parent controls. `mechanical_fraction_of_recycling`
is mechanical divided by mechanical plus chemical recycling and is undefined when
total recycling is zero. Completion in 2100 uses fresh linear transitions to the selected endpoints
and the same demand-only repair.
Active circularity combines each calendar year's parent value with the endpoint
using the revised completion weight. The selected endpoints remain fixed.

Repair statuses are `linear_feasible`, `stock_projection_feasible`,
`local_pool_feasible`, `stock_infeasible_fixed_controls`,
`unique_fixed_path_infeasible` and `repair_unresolved`. Only the first three have
admitted MFA trajectories. A stock projection comes from the single
minimum-departure stock LP and passes complete MFA with the common local
stock-roundoff convention. Solver-reported stock infeasibility concerns the
submitted fixed-control formulation. A pool success records the minimum departure
within the declared pool. Failed demand repair records exhaustion of the tested
alternatives. Numerical or model failures remain execution errors.

`selected_annual_acc_asr.parquet` has `annual` and `cumulative_2019_2100` periods.
Cumulative `year=2100` denotes period end. `compliant` compares every saved row with
ASR=1. `controlled_for_selection` and nullable `selection_compliant` apply only to
2050 regional selection coordinates: annual 2020-2100 and cumulative climate change
over 2019-2100. Annual 2019, global, and delayed rows provide additional assessment results.
Numerators and allocated carrying capacities have matched
native impact category units. Use the admitted method/category correspondence for the units. P5, P50
and P95 are the 5th, 50th (median), and
95th percentiles of the allocated carrying capacity distributions, each used as a
separate deterministic input.

`selected_environmental_scores.parquet` contains annual candidate totals in native
method units. `selected_life_cycle_stage_scores.parquet` contains annual and
cumulative periods. Cumulative `year` and `physical_event_year` are null, with
period bounds 2019 and 2100. Select one period before aggregation. Environmental
and ASR global rows use `global`; stage rows use `global_derived`. A global ASR
divides separately summed regional numerators and denominators. Aggregate within one region scope,
assessment period, and impact category.

`score` is signed LCIA impact; `contribution_norm` sums absolute characterized
contributions in the same native unit.
Candidate-minus-parent provider and PA adjustments reconstruct candidate totals;
negative adjustments express the candidate-minus-parent difference. A sparse
absent side represents zero activity at its exact coordinate. Stage totals
retain their recorded stage coordinates.

`selected_feedstock_requirements.parquet` reports 2019-2100 quantities in
Mt by material, constituent, product and virgin-feedstock route. Eligible
polyolefins use separate gross virgin ethylene and propylene inputs after recovered-
monomer substitution; other constituents use domestic plus exported new polymer.
Keep product and quantity fields separate. Route labels select fossil, biogenic or
DAC-based technology. Sum routes only for compatible named quantities.

`mfa_evaluation_count` and `stock_lp_count` report actual calls.
`pool_candidate_count` and `pool_evaluation_count` distinguish the deduplicated
pool from tested members. `demand_departure_pp` is mean absolute demand departure
from linear in percentage points over adjustable years, blank without a feasible
path. Selected-coordinate and unique-candidate counts include unsuccessful
timing outcomes.

`physical_repair_elapsed_seconds` covers repair or repair reuse, including
admission, solve and pool work. `environmental_evaluation_elapsed_seconds` includes
completion, scoring and feedstock-identity calculation. Both are per-worker
elapsed durations excluding initialization and final result encoding. Concurrent
worker durations can overlap. Request totals in `method_summary.json` represent embodied work;
incremental
fields identify work performed in the call that produced the saved product and
remain unchanged when that product is reused. Count each distinct evaluation once when aggregating
work.

The candidate worker's repair identity payload excludes exactly
`mfa_evaluation_count`, `stock_lp_count`, `pool_evaluation_count` and the two elapsed
durations; `pool_candidate_count` remains part of that identity. The selected product's content
identity binds its complete summary and result tables, including reported work
metadata.

```python
import pandas as pd

selected = pd.read_parquet(
    workspace
    / "results/<foreground_scenario>/optimization_results"
    / "<allocated_cc_result>/<optimization_lcia_selection>"
    / "<readable_request_selection>__stock_lp_v6/selected_results"
    / "joint_lines_v1_stock_lp_v6__<result_lcia_selection>__grid_20pct__refinement_10_5pct"
    / "selected_coordinates.parquet"
)
```

`manifest.json` is written after all tables and marks a completed reusable
result. The search method and full refinement schedule distinguish immutable selected
products, including schedules that end at the same resolution. Compatible searches
reuse completed candidate calculations from the one request-root cache; they do
not repeat MFA or environmental calculations for the same admitted inputs
and requested endpoint.
`refresh=True` replaces only this result; it does not recalculate its
required MFA results, prepared backgrounds, response bank, or allocated carrying capacity inputs.

These results report the deterministic optimization calculation for the tested
combinations, using the documented selection criteria.
"""


class OptimizationResultError(ValueError):
    """An optimization result is invalid or cannot be written safely."""


@dataclass(frozen=True, slots=True)
class StagedTable:
    """The finalized life cycle stage table in the hidden result directory."""

    staging_directory: Path
    columns: tuple[str, ...]
    row_count: int
    canonical_sha256: str


class TableStreamWriter:
    """Write and hash one logical table."""

    def __init__(
        self,
        *,
        staging_directory: Path,
    ) -> None:
        """Prepare the stage-score Parquet file and its incremental table hash."""
        self._staging_directory = staging_directory
        self._path = staging_directory / _STAGE_ARTIFACT
        self._hash_spool = CanonicalTableHashSpool(staging_directory / _STREAM_HASHES)
        self._writer: pq.ParquetWriter | None = None
        self._columns: tuple[str, ...] | None = None
        self._schema: pa.Schema | None = None

    def start(self, schema_sample: pd.DataFrame) -> None:
        """Fix the logical Pandas and Arrow schema from a representative complete sample."""

        columns = tuple(schema_sample.columns)
        self._columns = columns
        self._schema = pa.Table.from_pandas(schema_sample, preserve_index=False).schema
        self._writer = pq.ParquetWriter(self._path, self._schema, compression="zstd")

    def write(self, frame: pd.DataFrame) -> None:
        """Hash and write one completed stage frame in logical row order."""

        self._hash_spool.update(frame)
        arrow = pa.Table.from_pandas(
            frame,
            schema=cast(pa.Schema, self._schema),
            preserve_index=False,
            safe=True,
        )
        cast(pq.ParquetWriter, self._writer).write_table(arrow)

    def finish(self) -> StagedTable:
        """Close the table and return its columns, rows, and table hash."""

        cast(pq.ParquetWriter, self._writer).close()
        self._writer = None
        canonical_sha256 = self._hash_spool.finalize()
        return StagedTable(
            staging_directory=self._staging_directory,
            columns=cast(tuple[str, ...], self._columns),
            row_count=self._hash_spool.row_count,
            canonical_sha256=canonical_sha256,
        )

    def abort(self) -> None:
        """Close any open handles and remove the private hash spool."""

        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self._hash_spool.abort()


@dataclass(frozen=True, slots=True)
class OptimizationResultTarget:
    """Readable paths for one optimization request."""

    workspace: Path
    foreground_scenario: str
    allocated_cc_result: str
    optimization_lcia_selection: str
    readable_request_selection: str
    initial_grid_step_pct: int
    endpoint_refinement_steps_pct: tuple[int, ...]

    @property
    def root(self) -> Path:
        """Return the request root shared by scientifically compatible candidate results."""

        return (
            self.workspace
            / "results"
            / self.foreground_scenario
            / "optimization_results"
            / self.allocated_cc_result
            / self.optimization_lcia_selection
            / f"{self.readable_request_selection}__{_PHYSICAL_PRODUCT_METHOD_NAME}"
        )

    def selected_results(self, result_lcia_selection: str) -> Path:
        """Separate completed search methods, schedules and reporting LCIA selections."""

        steps = "_".join(str(step) for step in self.endpoint_refinement_steps_pct)
        selection = (
            f"{_SEARCH_PRODUCT_METHOD_NAME}__{result_lcia_selection}"
            f"__grid_{self.initial_grid_step_pct}pct__refinement_{steps}pct"
        )
        return self.root / "selected_results" / selection


@dataclass(frozen=True, slots=True)
class SavedOptimizationResult:
    """One completed selected optimization result, newly written or reused."""

    status: ItemStatus
    product_identity: str
    method_identity: str
    parent_identities: tuple[tuple[str, str], ...]
    summary_sha256: str
    directory: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class RefinementPrefix:
    """Completed compatible stages and custody for a strict refinement extension."""

    product_identity: str
    completed_search: CompletedCandidateSearch


def read_refinement_prefix(
    target: OptimizationResultTarget,
    *,
    result_lcia_selection: str,
    candidate_input_identity: str,
    selection_method_identity: str,
    requested_blocks: Sequence[OptimizationBlock],
) -> RefinementPrefix | None:
    """Load the longest completed prefix with matching science and stage policies.

    The selected product contains compact completed stages and worker summaries.
    Reuse requires matching stage records and pruning policies. A standalone
    final coarse refinement can use a different pruning policy from the same
    spacing in a longer request.
    """
    for count in range(len(target.endpoint_refinement_steps_pct) - 1, 0, -1):
        steps = target.endpoint_refinement_steps_pct[:count]
        prefix = replace(target, endpoint_refinement_steps_pct=steps)
        directory = prefix.selected_results(result_lcia_selection)
        if not product_complete(directory):
            continue
        summary = json.loads((directory / _SUMMARY_ARTIFACT).read_bytes())
        completed = cast(dict[str, object] | None, summary.get("completed_search"))
        if (
            summary.get("candidate_input_identity") != candidate_input_identity
            or summary.get("selection_method") != selection_method_identity
            or summary.get("integrated_search_method") != INTEGRATED_SEARCH_METHOD_IDENTITY
            or completed is None
        ):
            continue
        expected_policies = [
            [step, step <= 5 or step == target.endpoint_refinement_steps_pct[-1]] for step in steps
        ]
        if (
            completed["integrated_search_method"] != INTEGRATED_SEARCH_METHOD_IDENTITY
            or completed["integrated_design_method"] != INTEGRATED_DESIGN_METHOD_IDENTITY
            or completed["search_configuration_identity"]
            != search_configuration_identity(
                steps, initial_grid_step_pct=target.initial_grid_step_pct
            )
            or completed["initial_grid_step_pct"] != target.initial_grid_step_pct
            or completed["endpoint_refinement_steps_pct"] != list(steps)
            or completed["executed_blocks"] != list(_execution_blocks(requested_blocks))
            or completed["single_block_stage_policies"] != expected_policies
        ):
            continue
        return RefinementPrefix(
            product_identity=read_product_manifest(directory).product_identity,
            completed_search=decode_completed_search(completed),
        )
    return None


@contextmanager
def stage_selected_life_cycle_scores(directory: Path) -> Iterator[TableStreamWriter]:
    """Open the life cycle stage table inside the hidden result directory."""

    directory = Path(directory)
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    _plain_directory(parent, field="optimization product parent")
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.staging.", dir=parent))
    writer = TableStreamWriter(
        staging_directory=staging,
    )
    try:
        yield writer
    finally:
        writer.abort()
        shutil.rmtree(staging, ignore_errors=True)


def selected_result_schema_sha256() -> str:
    """Hash the packaged selected-result schema."""

    resource = files("pytoce.run_optimization").joinpath(_RESULT_SCHEMA_RESOURCE)
    return sha256(resource.read_bytes()).hexdigest()


def product_complete(directory: Path) -> bool:
    """Check whether the result's completion manifest exists."""
    return (directory / _MANIFEST).exists()


def read_product_manifest(
    directory: Path,
) -> SavedOptimizationResult:
    """Read report fields directly from a completed result's manifest."""
    manifest_path = directory / _MANIFEST
    manifest = json.loads(manifest_path.read_bytes())
    return SavedOptimizationResult(
        status="reused",
        product_identity=manifest["product_identity"],
        method_identity=manifest["method_identity"],
        parent_identities=tuple(sorted(manifest["parent_identities"].items())),
        summary_sha256=manifest["summary_sha256"],
        directory=directory,
        manifest_path=manifest_path,
    )


def write_optimization_result(
    directory: Path,
    *,
    product_identity: str,
    parent_identities: Mapping[str, str],
    method_identity: str,
    tables: Mapping[str, pd.DataFrame | StagedTable],
    summary: Mapping[str, object],
    refresh: bool = False,
) -> SavedOptimizationResult:
    """Write one selected optimization result and its completion manifest.

    Args:
        directory: Final readable directory for the selected result.
        product_identity: Content identifier binding the method, parent inputs,
            complete output summary and result tables, including saved work metadata.
        parent_identities: Content identifiers of the required MFA, LCA, aCC, and response data.
        method_identity: Version identifier for the optimization and reporting method.
        tables: Complete result tables. The stage table is already finalized
            in the adjacent hidden directory owned by
            :func:`stage_selected_life_cycle_scores`.
        summary: Values written to ``method_summary.json``.
        refresh: Replace a completed result after checking that every existing
            entry belongs to the result.

    Returns:
        The newly calculated, updated, or reused result record.

    Raises:
        OptimizationResultError: If an existing result is incomplete or contains
            unexpected files that make replacement unsafe.

    Tables and ``method_summary.json`` are completed in a hidden adjacent directory.
    ``manifest.json`` is written last before the directory is moved into place.
    A failure leaves an existing visible result unchanged.
    """

    directory = Path(directory)
    if product_complete(directory):
        if not refresh:
            return read_product_manifest(directory)
    elif directory.exists() or directory.is_symlink():
        raise OptimizationResultError(
            f"The optimization result directory at {directory} has no completion manifest. "
            "Keep the directory unchanged and use a new workspace."
        )

    requested = {
        "method_identity": method_identity,
        "parent_identities": dict(sorted(parent_identities.items())),
        "product_identity": product_identity,
    }
    summary_bytes = _canonical_json_bytes(dict(summary))
    staged_table = cast(StagedTable, tables[_STAGE_ARTIFACT])
    staging = staged_table.staging_directory
    status: ItemStatus = "updated" if directory.exists() else "computed"
    artifacts: dict[str, dict[str, object]] = {}
    for name in _TABLE_ARTIFACTS:
        table = tables[name]
        path = staging / name
        if name == _STAGE_ARTIFACT:
            columns = list(staged_table.columns)
            row_count = staged_table.row_count
        else:
            frame = cast(pd.DataFrame, table)
            frame.to_parquet(path, index=False, compression="zstd")
            columns = list(frame.columns)
            row_count = len(frame)
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
        artifacts[name] = {
            "columns": columns,
            "media_type": "application/vnd.apache.parquet",
            "row_count": row_count,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }

    summary_path = staging / _SUMMARY_ARTIFACT
    _write_new(summary_path, summary_bytes)
    artifacts[_SUMMARY_ARTIFACT] = {
        "media_type": "application/json",
        "sha256": sha256(summary_bytes).hexdigest(),
        "size_bytes": len(summary_bytes),
    }
    manifest = {
        "artifacts": artifacts,
        "complete": True,
        "method_identity": requested["method_identity"],
        "parent_identities": requested["parent_identities"],
        "product_identity": requested["product_identity"],
        "schema_sha256": selected_result_schema_sha256(),
        "schema_version": SELECTED_RESULT_SCHEMA,
        "summary_sha256": sha256(summary_bytes).hexdigest(),
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    _write_new(staging / _MANIFEST, manifest_bytes)
    _sync_directory(staging)

    if status == "updated":
        validate_result_for_refresh(directory)
    replace_directory(
        staging,
        directory,
        replace_existing=status == "updated",
    )
    return SavedOptimizationResult(
        status=status,
        product_identity=cast(str, requested["product_identity"]),
        method_identity=cast(str, requested["method_identity"]),
        parent_identities=tuple(
            sorted(cast(dict[str, str], requested["parent_identities"]).items())
        ),
        summary_sha256=cast(str, manifest["summary_sha256"]),
        directory=directory,
        manifest_path=directory / _MANIFEST,
    )


def ensure_optimization_results_guide(target: OptimizationResultTarget) -> Path:
    """Create the adjacent optimization-results guide when it is missing."""

    directory = target.workspace / "results" / target.foreground_scenario / "optimization_results"
    directory.mkdir(parents=True, exist_ok=True)
    _plain_directory(directory, field="optimization results directory")
    guide = directory / "README.md"
    if guide.exists():
        record = os.lstat(guide)
        if not stat.S_ISREG(record.st_mode) or _is_reparse(record):
            raise OptimizationResultError(
                f"The optimization results guide at {guide} is linked or is not a regular "
                "file. Keep it unchanged and use a new workspace."
            )
        return guide
    _write_new(guide, _GUIDE)
    _sync_directory(directory)
    return guide


def validate_result_for_refresh(directory: Path) -> None:
    """Check that refresh will replace only owned files in a non-linked directory."""
    _plain_directory(directory, field="optimization product directory")
    expected = set(_TABLE_ARTIFACTS) | {_SUMMARY_ARTIFACT, _MANIFEST}
    entries = {entry.name: entry for entry in os.scandir(directory)}
    unexpected = tuple(sorted(set(entries).difference(expected)))
    if unexpected:
        raise OptimizationResultError(
            f"Cannot refresh {directory}: unexpected entries {unexpected!r}. "
            "Keep the directory unchanged and use a new workspace."
        )
    for name in entries:
        record = os.lstat(directory / name)
        if not stat.S_ISREG(record.st_mode) or _is_reparse(record):
            raise OptimizationResultError(
                f"Cannot refresh {directory}: {name} is linked or is not a regular file. "
                "Keep the directory unchanged and use a new workspace."
            )


def _canonical_json_bytes(payload: object) -> bytes:
    """Encode finite JSON with sorted keys and a final newline."""
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    """Create and synchronize a new file exclusively."""
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _plain_directory(path: Path, *, field: str) -> None:
    """Require an ordinary directory with its own filesystem identity."""
    record = os.lstat(path)
    if not stat.S_ISDIR(record.st_mode) or _is_reparse(record):
        raise OptimizationResultError(f"{field} must be a non-linked directory")


def _is_reparse(record: os.stat_result) -> bool:
    """Return whether a Windows file record is a reparse point."""
    return bool(getattr(record, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT)
