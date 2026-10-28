"""Validate and compile allocated carrying capacity inputs.

This owner validates complete, positive allocated carrying capacity (aCC)
tensors. Calculate the values through the optional pyaesa tutorial workflow or
supply values from another documented allocation method.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray

from pytoce.run_mfa.scenario.catalogue import (
    ParameterDefinition,
    load_parameter_catalogue,
)
from pytoce.run_optimization.input.allocated_cc_paths import (
    _ALLOCATED_CC_SCHEMA_VERSION,
    AllocatedCCError,
    _readable_name,
)
from pytoce.shared.runtime.filesystem import (
    canonical_table_sha256,
    file_sha256,
    payload_sha256,
)
from pytoce.shared.runtime.manifests import OperationStatus

_IDENTITY_PREFIX = f"{_ALLOCATED_CC_SCHEMA_VERSION}::"
_LCIA_REGISTRY_PREFIX = "pytoce_optimization_lcia_registry_v1::"
_LCIA_SELECTION_PREFIX = "pytoce_optimization_lcia_selection_v1::"
_SELECTION_PREFIX = "pytoce_allocated_cc_selection_v1::"
_METHOD_COMPATIBILITY_PREFIX = "pytoce_lcia_method_compatibility_v1::"
_PROCESSED_RELATIVE = Path("data_processed/allocated_cc")
_SHARED_PREPARED_RELATIVE = Path("data_processed/prepared_backgrounds/shared")
_YEARS = tuple(range(2019, 2101))
_ANNUAL_SOURCE_COLUMNS = (
    "allocated_cc_case",
    "lcia_method",
    "region",
    "impact_category_id",
    "year",
    "allocated_cc",
    "unit",
    "cc_authority",
)
_CUMULATIVE_SOURCE_COLUMNS = (
    "allocated_cc_case",
    "region",
    "impact_category_id",
    "period_start",
    "period_end",
    "cumulative_allocated_cc",
    "unit",
    "budget_metric",
    "cc_authority",
)
_METHOD_COLUMNS = (
    "lcia_method",
    "selection_order",
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "unit",
    "factor_identity",
)
_ANNUAL_COMPILED_COLUMNS = (
    "allocated_cc_case",
    "lcia_method",
    "region",
    "selection_order",
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "factor_identity",
    "year",
    "allocated_cc",
    "unit",
    "cc_authority",
)
_CUMULATIVE_COMPILED_COLUMNS = (
    "allocated_cc_case",
    "lcia_method",
    "region",
    "selection_order",
    "method_id",
    "method_version",
    "impact_category_id",
    "impact_category_tuple_json",
    "factor_identity",
    "period_start",
    "period_end",
    "cumulative_allocated_cc",
    "unit",
    "budget_metric",
    "cc_authority",
)
_BUDGET_METRIC = "ef31_climate_change_gwp100_all_ghg"

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AllocatedCCResult:
    """One validated aCC result available to monitoring and target setting.

    The record identifies its processed directory, cases, and EF v3.1 method
    tables. Numerical arrays are loaded when a workflow selects its cases.
    """

    status: OperationStatus
    allocated_cc_result: str
    allocated_cc_result_identity: str
    allocated_cc_cases: tuple[str, ...]
    annual_impact_category_count: int
    result_directory: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class AllocatedCCSelection:
    """Allocated carrying capacities selected for an ASR assessment.

    The assessment selects complete allocated carrying capacity (aCC) values
    calculated through the pyaesa tutorial workflow or supplied by the user.
    Annual arrays have axes ``[case, region, year, impact category]``.
    Cumulative arrays have axes
    ``[case, region]`` and use EF climate change GWP100 units for 2019--2100.

    Attributes:
        allocated_cc_result_identity: Content identifier calculated from the complete
            validated aCC result, its source-file hashes, and the prepared LCIA methods.
        allocated_cc_selection_identity: Content identifier calculated from the aCC result,
            selected cases, annual and cumulative rows, and LCIA method records.
        optimization_lcia_selection_identity: Content identifier of the LCIA method selected
            for optimization, currently EF v3.1.
        optimization_lcia_method: Ordered optimization method discriminators,
            currently ``("ef31",)``.
        annual_impact_category_count: Number of annual EF v3.1 midpoint impact
            categories in the selection.
        allocated_cc_cases: Ordered, unique, readable case labels supplied with the
            input. Source documentation defines the allocation method and
            statistic for each label.
        regions: 12 world regions in array order.
        years: Calendar years 2019--2100 in array order.
        annual_categories: Ordered ``(lcia_method, impact_category_id)`` pairs
            forming the final annual array axis.
        annual_table_identity: Content identifier of all selected annual aCC rows.
        cumulative_gwp100_table_identity: Content identifier of all selected cumulative
            GWP100 aCC rows.
        method_registry_identity: Content identifier of the ordered annual LCIA method rows.
        cumulative_climate_method_registry_identity: Content identifier of the EF climate
            method used for the cumulative constraint.
        method_compatibility_identity: Content identifier used to match annual aCC rows to
            candidate LCIA results.
        cumulative_climate_compatibility_identity: Content identifier used to match the
            cumulative aCC rows to the candidate climate results.
        annual_allocated_cc_identity: Content identifier calculated from the annual array
            shape and values.
        cumulative_allocated_cc_identity: Content identifier calculated from the cumulative
            array shape and values.
        annual_case_table_identities: Content identifier of the annual table for each case.
        cumulative_case_table_identities: Content identifier of the cumulative table for
            each case.
        annual_allocated_cc: Positive annual allocated carrying capacities with
            axes ``[case, world region, year, impact category]`` in each impact
            category's native LCIA unit.
        cumulative_allocated_cc: Positive cumulative allocated carrying capacities
            with axes ``[case, world region]`` in the native EF climate change unit
            for 2019--2100.

    Notes:
        The allocated carrying capacity arrays are read-only. They retain the
        values converted to float64 during input validation. Their LCIA methods must match the
        candidate LCIA results
        before absolute sustainability ratios are calculated. Selecting another
        case or changing a selected table or method record changes the selection
        identifier; the final-result ``lcia_method`` is separate.
    """

    allocated_cc_result_identity: str
    allocated_cc_selection_identity: str
    optimization_lcia_selection_identity: str
    optimization_lcia_method: tuple[str, ...]
    annual_impact_category_count: int
    allocated_cc_cases: tuple[str, ...]
    regions: tuple[str, ...]
    years: tuple[int, ...]
    annual_categories: tuple[tuple[str, str], ...]
    annual_table_identity: str
    cumulative_gwp100_table_identity: str
    method_registry_identity: str
    cumulative_climate_method_registry_identity: str
    method_compatibility_identity: str
    cumulative_climate_compatibility_identity: str
    annual_allocated_cc_identity: str
    cumulative_allocated_cc_identity: str
    annual_case_table_identities: tuple[str, ...]
    cumulative_case_table_identities: tuple[str, ...]
    annual_allocated_cc: FloatArray
    cumulative_allocated_cc: FloatArray


@dataclass(frozen=True, slots=True)
class _MethodAuthority:
    """LCIA tables and source identities used to check one aCC result."""

    frame: pd.DataFrame
    cumulative_climate: pd.DataFrame
    identity: str
    selection_identity: str
    prepared_background_shared_identity: str
    prepared_background_shared_manifest_sha256: str
    prepared_lcia_authority_identity: str


def _h12_regions() -> tuple[str, ...]:
    """Read the 12 world regions from the MFA parameter catalogue."""
    definition = cast(ParameterDefinition, load_parameter_catalogue("collection_rate"))
    return tuple(definition.coordinates["region"])


def _method_tuple_json(values: Sequence[object]) -> str:
    """Serialize an LCIA method tuple as compact JSON."""
    return json.dumps([str(value) for value in values], ensure_ascii=True, separators=(",", ":"))


def _impact_records(methods: Sequence[Any], discriminator: str) -> list[dict[str, object]]:
    """Convert resolved LCIA methods to allocated carrying capacity registry rows."""
    return [
        {
            "lcia_method": discriminator,
            "selection_order": method.selection_order,
            "method_id": method.method_id,
            "method_version": method.method_version,
            "impact_category_id": method.impact_category_id,
            "impact_category_tuple_json": _method_tuple_json(method.method_tuple),
            "unit": method.unit,
            "factor_identity": method.factor_identity,
        }
        for method in methods
    ]


def _method_authority(workspace: Path) -> _MethodAuthority:
    """Resolve annual EF methods and the cumulative climate method from prepared data.

    Args:
        workspace: Active workspace containing the completed shared background data.

    Returns:
        The 16 EF v3.1 midpoint methods, their single climate change GWP100 row,
        and identities for the method tables and prepared shared background.

    Raises:
        AllocatedCCError: If shared background preparation is missing.
    """
    from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
        PreparedDataError,
        read_prepared_manifest,
    )
    from pytoce.run_lca.calculation.lcia import (
        normalized_lcia_method,
        resolve_lcia_methods_from_authority,
    )

    shared = workspace / _SHARED_PREPARED_RELATIVE
    try:
        reference = read_prepared_manifest(shared / "manifest.json")
        authority = pq.read_table(reference.path.parent / "impact_categories.parquet")
    except PreparedDataError as exc:
        raise AllocatedCCError(str(exc)) from exc
    manifest = reference.payload
    shared_identity = str(manifest["identity"])
    lcia_identity = str(manifest["lcia_authority_identity"])
    ef_methods = resolve_lcia_methods_from_authority(
        authority.to_pandas(), normalized_lcia_method("ef31"), None
    )
    climate_methods = tuple(
        method for method in ef_methods if method.impact_category_id == "climate_change"
    )

    frame = pd.DataFrame.from_records(_impact_records(ef_methods, "ef31"), columns=_METHOD_COLUMNS)
    cumulative_climate = pd.DataFrame.from_records(
        _impact_records(climate_methods, "ef31"), columns=_METHOD_COLUMNS
    )
    selection_identity = _LCIA_SELECTION_PREFIX + payload_sha256({"lcia_method": ["ef31"]})
    registry_payload = {
        "lcia_method": ["ef31"],
        "annual_methods": frame.to_dict(orient="records"),
        "cumulative_climate": cumulative_climate.to_dict(orient="records"),
        "prepared_background_shared_identity": shared_identity,
        "prepared_background_shared_manifest_sha256": reference.sha256,
        "prepared_lcia_authority_identity": lcia_identity,
    }
    return _MethodAuthority(
        frame=frame,
        cumulative_climate=cumulative_climate,
        identity=_LCIA_REGISTRY_PREFIX + payload_sha256(registry_payload),
        selection_identity=selection_identity,
        prepared_background_shared_identity=shared_identity,
        prepared_background_shared_manifest_sha256=reference.sha256,
        prepared_lcia_authority_identity=lcia_identity,
    )


def _case_text(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Read one required text column and report how its owner can correct it."""
    values = cast(pd.Series, frame[column])
    missing = values.isna()
    if missing.any():
        row = values.index[missing][0]
        raise AllocatedCCError(
            f"Column {column!r} is missing at row {row!r}. Correct that cell and run again."
        )
    result = values.astype(str)
    stripped = result.str.strip()
    empty = stripped.eq("")
    if empty.any():
        row = result.index[empty][0]
        raise AllocatedCCError(
            f"Column {column!r} is empty at row {row!r}. Correct that cell and run again."
        )
    padded = result.ne(stripped)
    if padded.any():
        row = result.index[padded][0]
        raise AllocatedCCError(
            f"Column {column!r} has surrounding whitespace in {result.loc[row]!r} at row "
            f"{row!r}. Correct that cell and run again."
        )
    return result


def _finite_positive(frame: pd.DataFrame, column: str) -> pd.Series:
    """Read one numeric column and require finite values greater than zero."""
    try:
        values = pd.to_numeric(cast(pd.Series, frame[column]), errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise AllocatedCCError(
            f"Column {column!r} must contain numbers: {exc}. Replace the invalid value and "
            "run again."
        ) from exc
    invalid = ~np.isfinite(values) | values.le(0.0)
    if invalid.any():
        row = values.index[invalid][0]
        raise AllocatedCCError(
            f"Column {column!r} contains {values.loc[row]!r} at row {row!r}. Replace it with "
            "a finite value greater than zero and run again."
        )
    return values


def _integer(frame: pd.DataFrame, column: str) -> pd.Series:
    """Admit exact integer years from an aCC CSV column."""
    lower = int(np.iinfo(np.int64).min)
    upper = int(np.iinfo(np.int64).max)
    years: list[int] = []
    for raw in frame[column]:
        try:
            # CSV fields retain their text. A float conversion could round a
            # fractional year to an integer before admission.
            value = Decimal(str(raw))
            if not value.is_finite() or value != value.to_integral_value():
                raise ValueError(f"Year {raw!r} is not a finite whole calendar year")
            if not lower <= value <= upper:
                raise ValueError(f"Year {raw!r} exceeds the supported integer range")
            years.append(int(value))
        except (InvalidOperation, ValueError) as exc:
            raise AllocatedCCError(
                f"Column {column!r} must contain whole numbers; found {raw!r}: {exc}. "
                "Replace the invalid value and run again."
            ) from exc
    return pd.Series(years, index=frame.index, dtype="int64")


def _validate_annual(
    source: pd.DataFrame,
    methods: pd.DataFrame,
    regions: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Validate and order the annual allocated carrying capacity table.

    The table must contain one positive value and ``cc_authority`` for every
    case, world region, selected impact category, and year from 2019 through 2100.
    Case names are user-defined lowercase ``snake_case`` identifiers. LCIA units
    must match the prepared method records exactly.

    Returns:
        The source rows joined to their LCIA method fields and sorted by case,
        world region, impact category, and year, followed by the ordered case names.
    """
    frame = source.copy()
    for column in (
        "allocated_cc_case",
        "lcia_method",
        "region",
        "impact_category_id",
        "unit",
        "cc_authority",
    ):
        frame[column] = _case_text(frame, column)
    from pytoce.run_lca.calculation.lcia import (
        normalized_lcia_discriminator,
    )

    try:
        frame["lcia_method"] = frame["lcia_method"].map(normalized_lcia_discriminator)
    except ValueError as exc:
        raise AllocatedCCError(f"annual_allocated_cc has an invalid lcia_method: {exc}") from exc
    frame["allocated_cc_case"] = frame["allocated_cc_case"].map(
        lambda value: _readable_name(value, field="allocated_cc_case")
    )
    frame["year"] = _integer(frame, "year")
    frame["allocated_cc"] = _finite_positive(frame, "allocated_cc")
    row_fields = [
        "allocated_cc_case",
        "lcia_method",
        "region",
        "impact_category_id",
        "year",
    ]
    duplicate = frame.loc[frame.duplicated(row_fields, keep=False), row_fields]
    if not duplicate.empty:
        raise AllocatedCCError(
            "annual_allocated_cc.csv repeats a case, method, region, category, and year row. "
            f"Keep one copy of {duplicate.iloc[0].to_dict()!r}."
        )
    unknown_regions = sorted(set(frame["region"]).difference(regions))
    if unknown_regions:
        raise AllocatedCCError(
            f"annual_allocated_cc.csv contains unknown regions {unknown_regions!r}. Replace "
            f"them with world region names from {regions!r}."
        )
    method_coordinates = tuple(
        (str(row.lcia_method), str(row.impact_category_id))
        for row in methods.itertuples(index=False)
    )
    observed_method_coordinates = set(zip(frame["lcia_method"], frame["impact_category_id"]))
    unknown_methods = sorted(observed_method_coordinates.difference(method_coordinates))
    if unknown_methods:
        raise AllocatedCCError(
            "annual_allocated_cc.csv contains unsupported lcia_method and "
            f"impact_category_id pairs {unknown_methods!r}. Choose from "
            f"{method_coordinates!r}."
        )
    unknown_years = sorted(set(frame["year"]).difference(_YEARS))
    if unknown_years:
        raise AllocatedCCError(
            f"annual_allocated_cc.csv contains years outside 2019 through 2100: "
            f"{unknown_years!r}. Remove those rows and supply every year from 2019 through 2100."
        )
    cases = tuple(sorted(set(frame["allocated_cc_case"])))
    expected = pd.MultiIndex.from_tuples(
        product(cases, regions, method_coordinates, _YEARS),
        names=["allocated_cc_case", "region", "method_coordinate", "year"],
    )
    observed = pd.MultiIndex.from_tuples(
        (
            (
                str(row.allocated_cc_case),
                str(row.region),
                (str(row.lcia_method), str(row.impact_category_id)),
                int(cast(Any, row.year)),
            )
            for row in frame.itertuples(index=False)
        ),
        names=["allocated_cc_case", "region", "method_coordinate", "year"],
    )
    missing = expected.difference(observed)
    if len(missing):
        raise AllocatedCCError(
            "annual_allocated_cc.csv must contain one row for every case, world region, selected "
            "LCIA category, and year from 2019 through 2100. "
            f"Missing rows: {len(missing)} (first: {missing[0]!r}). Supply the missing rows."
        )
    enriched = frame.merge(
        methods,
        on=["lcia_method", "impact_category_id", "unit"],
        how="left",
        validate="many_to_one",
    )
    if enriched["factor_identity"].isna().any():
        invalid = enriched.loc[enriched["factor_identity"].isna()].iloc[0]
        expected_units = tuple(
            methods.loc[
                methods["lcia_method"].eq(invalid["lcia_method"])
                & methods["impact_category_id"].eq(invalid["impact_category_id"]),
                "unit",
            ].astype(str)
        )
        raise AllocatedCCError(
            f"annual_allocated_cc.csv uses unit {invalid['unit']!r} for "
            f"lcia_method={invalid['lcia_method']!r} and "
            f"impact_category_id={invalid['impact_category_id']!r}. Replace it with "
            f"{expected_units!r}."
        )
    region_order = {region: index for index, region in enumerate(regions)}
    enriched["_region_order"] = enriched["region"].map(region_order)
    enriched = enriched.sort_values(
        ["allocated_cc_case", "_region_order", "selection_order", "year"],
        kind="stable",
    ).reset_index(drop=True)
    return cast(pd.DataFrame, enriched.loc[:, list(_ANNUAL_COMPILED_COLUMNS)]), cases


def _validate_cumulative(
    source: pd.DataFrame,
    methods: pd.DataFrame,
    regions: tuple[str, ...],
    cases: tuple[str, ...],
) -> pd.DataFrame:
    """Validate cumulative EF GWP100 values for every case and world region.

    Each annual case must have one positive cumulative allocated carrying
    capacity for every region, covering 2019--2100 with the required all-GHG EF
    GWP100 metric. The LCIA unit must equal the prepared EF climate change unit.

    Returns:
        Rows joined to the exact EF climate method and ordered by case and region.
    """
    frame = source.copy()
    for column in (
        "allocated_cc_case",
        "region",
        "impact_category_id",
        "unit",
        "budget_metric",
        "cc_authority",
    ):
        frame[column] = _case_text(frame, column)
    frame["allocated_cc_case"] = frame["allocated_cc_case"].map(
        lambda value: _readable_name(value, field="allocated_cc_case")
    )
    frame["period_start"] = _integer(frame, "period_start")
    frame["period_end"] = _integer(frame, "period_end")
    frame["cumulative_allocated_cc"] = _finite_positive(frame, "cumulative_allocated_cc")
    row_fields = ["allocated_cc_case", "region"]
    duplicate = frame.loc[frame.duplicated(row_fields, keep=False), row_fields]
    if not duplicate.empty:
        raise AllocatedCCError(
            "cumulative_gwp100_allocated_cc.csv repeats a case and region row. Keep one copy "
            f"of {duplicate.iloc[0].to_dict()!r}."
        )
    if set(frame["allocated_cc_case"]) != set(cases):
        raise AllocatedCCError(
            "The annual and cumulative tables must use the same allocated_cc_case names. "
            f"Annual cases: {cases!r}; cumulative cases: "
            f"{tuple(sorted(set(frame['allocated_cc_case'])))!r}."
        )
    unknown_regions = sorted(set(frame["region"]).difference(regions))
    if unknown_regions:
        raise AllocatedCCError(
            "cumulative_gwp100_allocated_cc.csv contains unknown regions "
            f"{unknown_regions!r}. Replace them with world region names from {regions!r}."
        )
    if (
        set(frame["impact_category_id"]) != {"climate_change"}
        or set(frame["period_start"]) != {2019}
        or set(frame["period_end"]) != {2100}
        or set(frame["budget_metric"]) != {_BUDGET_METRIC}
    ):
        raise AllocatedCCError(
            "Each cumulative_gwp100_allocated_cc.csv row must use "
            "impact_category_id='climate_change', period_start=2019, period_end=2100, and "
            f"budget_metric={_BUDGET_METRIC!r}. Correct the row values and run again."
        )
    expected = pd.MultiIndex.from_product([cases, regions], names=["allocated_cc_case", "region"])
    observed = pd.MultiIndex.from_frame(frame[["allocated_cc_case", "region"]])
    missing = expected.difference(observed)
    if len(missing):
        raise AllocatedCCError(
            "cumulative_gwp100_allocated_cc.csv must contain one row for every case and "
            f"world region. Missing rows: {len(missing)} (first: {missing[0]!r}). "
            "Supply the missing rows."
        )
    climate = methods.loc[methods["impact_category_id"] == "climate_change"]
    enriched = frame.merge(
        climate,
        on=["impact_category_id", "unit"],
        how="left",
        validate="many_to_one",
    )
    if enriched["factor_identity"].isna().any():
        invalid = enriched.loc[enriched["factor_identity"].isna()].iloc[0]
        expected_units = tuple(climate["unit"].astype(str))
        raise AllocatedCCError(
            f"cumulative_gwp100_allocated_cc.csv uses unit {invalid['unit']!r}. Replace it "
            f"with {expected_units!r}."
        )
    region_order = {region: index for index, region in enumerate(regions)}
    enriched["_region_order"] = enriched["region"].map(region_order)
    enriched = enriched.sort_values(
        ["allocated_cc_case", "_region_order"], kind="stable"
    ).reset_index(drop=True)
    return cast(pd.DataFrame, enriched.loc[:, list(_CUMULATIVE_COMPILED_COLUMNS)])


def validate_allocated_cc_frames(
    annual_source: pd.DataFrame,
    cumulative_source: pd.DataFrame,
    method_authority: _MethodAuthority,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Validate user aCC tables against the selected prepared LCIA methods.

    This is the single numerical input boundary for allocated carrying capacity.
    It checks the exact v1 columns, complete region/year coordinates, unique rows,
    positive values, units, and row-level ``cc_authority`` text. It matches rows
    to the prepared annual EF methods and cumulative EF climate change method,
    then orders the accepted rows for compilation.

    Returns:
        Ordered annual and cumulative tables with their exact method fields, plus
        the ordered case names shared by both tables.
    """

    if tuple(annual_source.columns) != _ANNUAL_SOURCE_COLUMNS:
        raise AllocatedCCError(
            "annual_allocated_cc.csv columns do not match the v1 schema. Use exactly "
            f"{_ANNUAL_SOURCE_COLUMNS!r} in that order."
        )
    if tuple(cumulative_source.columns) != _CUMULATIVE_SOURCE_COLUMNS:
        raise AllocatedCCError(
            "cumulative_gwp100_allocated_cc.csv columns do not match the v1 schema. Use "
            f"exactly {_CUMULATIVE_SOURCE_COLUMNS!r} in that order."
        )
    if annual_source.empty:
        raise AllocatedCCError("annual_allocated_cc must contain at least one complete case")
    if cumulative_source.empty:
        raise AllocatedCCError(
            "cumulative_gwp100_allocated_cc must contain at least one complete case"
        )
    methods = method_authority.frame
    climate = method_authority.cumulative_climate
    regions = _h12_regions()
    annual, cases = _validate_annual(annual_source, methods, regions)
    cumulative = _validate_cumulative(cumulative_source, climate, regions, cases)
    return annual, cumulative, cases


def _artifact(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    """Describe a stored table by file hash, table hash, row count, and columns."""
    return {
        "path": path.name,
        "artifact_sha256": file_sha256(path),
        "scientific_identity": canonical_table_sha256(frame),
        "rows": len(frame),
        "columns": [str(column) for column in frame.columns],
    }


def _readonly_array(values: FloatArray) -> FloatArray:
    """Copy float64 values into an immutable bytes-backed array."""
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)


def _array_identity(values: FloatArray) -> str:
    """Hash the shape and contiguous float64 bytes of an array."""
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return payload_sha256(
        {
            "dtype": "float64",
            "shape": list(contiguous.shape),
            "bytes_sha256": sha256(contiguous.tobytes()).hexdigest(),
        }
    )
