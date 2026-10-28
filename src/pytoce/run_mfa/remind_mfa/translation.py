"""Admit REMIND-MFA 0.2.0 stock arrays and translate pyToCE scenario inputs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import flodym as fd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from ..scenario.scenario_contract import MATERIAL_SHARE_SUM_TOLERANCE

_DIMENSION_LETTERS = {
    "year": "t",
    "region": "r",
    "material": "m",
    "good": "g",
    "element": "e",
}
_DIRECT_FUTURE_PARAMETERS = frozenset(
    {
        "collection_rate",
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
        "waste_imports",
        "waste_exports",
        "bio_production_rate",
        "daccu_production_rate",
        "emission_capture_rate",
        "population",
        "gdppc",
    }
)
_DEFERRED_TRADE_MODES = frozenset(
    {
        "retain_2019_propensity_pretrade_collected",
        "retain_default_propensity_pretrade_collected",
    }
)


@dataclass(frozen=True, slots=True)
class RemindMFAIdentity:
    """Identify the exact REMIND-MFA calculation normalized by pyToCE.

    Attributes:
        repository: Upstream REMIND-MFA repository URL.
        tag: Upstream release tag.
        commit: Upstream code commit.
        tree: Verified upstream Git tree object ID.
        input_version: REMIND-MFA input-data release.
        input_revision: Immutable input-data revision.
        input_archive_sha256: Verified input-archive SHA-256.
        mrmfa_version: ``mrmfa`` version used by REMIND-MFA.
        mrmfa_commit: ``mrmfa`` source revision.
        madrat_version: ``madrat`` version recorded by the source data.
        configuration_sha256: SHA-256 of the executed REMIND-MFA configuration.
        dimension_schema_sha256: SHA-256 of the accepted dimension mapping.
        scenario: Upstream REMIND-MFA scenario name.
        adapter_identity: Version identifier for the pyToCE normalization method.
    """

    repository: str
    tag: str
    commit: str
    tree: str
    input_version: str
    input_revision: str
    input_archive_sha256: str
    mrmfa_version: str
    mrmfa_commit: str
    madrat_version: str
    configuration_sha256: str
    dimension_schema_sha256: str
    scenario: str
    adapter_identity: str

    def as_dict(self) -> dict[str, str]:
        """Return the source and configuration identifiers for the result manifest."""

        return {
            "repository": self.repository,
            "tag": self.tag,
            "commit": self.commit,
            "tree": self.tree,
            "input_version": self.input_version,
            "input_revision": self.input_revision,
            "input_archive_sha256": self.input_archive_sha256,
            "mrmfa_version": self.mrmfa_version,
            "mrmfa_commit": self.mrmfa_commit,
            "madrat_version": self.madrat_version,
            "configuration_sha256": self.configuration_sha256,
            "dimension_schema_sha256": self.dimension_schema_sha256,
            "scenario": self.scenario,
            "adapter_identity": self.adapter_identity,
        }


@dataclass(frozen=True, slots=True)
class NormalizedCohortState:
    """Store REMIND-MFA cohort arrays.

    Source element fractions define nominal mass accounts, rather than the
    chemical constituents tracked in the life cycle inventory.

    Attributes:
        years: Calendar years from 1950 through 2100.
        interval_years: Duration of each annual interval in years.
        regions: world region labels.
        goods: Plastic-using product groups.
        materials: Plastic material families.
        elements: Labels C and Other Elements for the source nominal mass accounts.
        stock_by_cohort: In-use stock in Mt with axes
            ``[year, cohort, region, good]``.
        outflow_by_cohort: End-of-life outflow in Mt with the same axes.
        annual_stock: Total in-use stock in Mt with axes ``[year, region, good]``.
        annual_inflow: Gross inflow entering use, including same-year outflow,
            in Mt with axes ``[year, region, good]``.
        annual_outflow: Annual stock outflow in Mt with axes ``[year, region, good]``.
        material_shares: Shares with axes ``[region, material, good]``.
        carbon_content: Source C/Other Elements fractions on ``[element, material]`` axes.
        lifetime_mean: Mean lifetime in years by good.
        lifetime_std: Lifetime standard deviation in years by good.
    """

    years: tuple[int, ...]
    interval_years: np.ndarray
    regions: tuple[str, ...]
    goods: tuple[str, ...]
    materials: tuple[str, ...]
    elements: tuple[str, ...]
    stock_by_cohort: np.ndarray
    outflow_by_cohort: np.ndarray
    annual_stock: np.ndarray
    annual_inflow: np.ndarray
    annual_outflow: np.ndarray
    material_shares: np.ndarray
    carbon_content: np.ndarray
    lifetime_mean: np.ndarray
    lifetime_std: np.ndarray


@dataclass(frozen=True, slots=True)
class AnnualInputTranslation:
    """Numeric upstream updates and the pyToCE deferred trade rules."""

    numeric_updates: dict[str, pd.DataFrame]
    deferred_waste_trade: pd.DataFrame


def _finite_array(value: object, label: str) -> np.ndarray:
    """Convert one REMIND-MFA tensor to a finite numeric array."""
    array = np.array(value, dtype=float, copy=True, order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"REMIND-MFA {label} contains nonfinite values")
    return array


def normalize_cohort_state(
    stock: fd.StockDrivenDSM,
    parameters: Mapping[str, fd.Parameter],
) -> NormalizedCohortState:
    """Detach the default stock reconstruction and its material/lifetime inputs.

    Cohort tensors use year, cohort, region and good axes; annual stock and
    flow arrays use year, region and good. Stocks and annual interval flows
    are in Mt. The caller computes this public FLODYM stock model.
    """
    years = tuple(int(value) for value in stock.dims["t"].items)
    if years != tuple(range(1950, 2101)):
        raise ValueError("Cohort years must be exactly 1950 through 2100")
    shares = parameters["material_shares_in_goods"]
    carbon = parameters["carbon_content_materials"]
    material_shares = _finite_array(shares.values, "material shares")
    carbon_content = _finite_array(carbon.values, "carbon content")
    # The retained source supplies two nominal fractions whose sum is one.
    # The tolerance covers representation and addition roundoff.
    for values, axis, label, tolerance in (
        (material_shares, 1, "material shares", MATERIAL_SHARE_SUM_TOLERANCE),
        (carbon_content, 0, "nominal element fractions", 2 * float(np.finfo(float).eps)),
    ):
        if np.any((values < 0) | (values > 1)) or not np.allclose(
            values.sum(axis=axis), 1.0, rtol=0.0, atol=tolerance
        ):
            raise ValueError(
                f"REMIND-MFA {label} must be fractions in [0, 1] summing to one. "
                "Use the unmodified downloaded source inputs."
            )
    # FLODYM constructs all three stock arrays; its field annotations retain None.
    return NormalizedCohortState(
        years=years,
        interval_years=np.ones(len(years), dtype=float),
        regions=tuple(str(value) for value in stock.dims["r"].items),
        goods=tuple(str(value) for value in stock.dims["g"].items),
        materials=tuple(str(value) for value in shares.dims["m"].items),
        elements=tuple(str(value) for value in carbon.dims["e"].items),
        stock_by_cohort=_finite_array(stock.get_stock_by_cohort(), "stock by cohort"),
        outflow_by_cohort=_finite_array(stock.get_outflow_by_cohort(), "outflow by cohort"),
        annual_stock=_finite_array(cast(fd.StockArray, stock.stock).values, "annual cohort stock"),
        annual_inflow=_finite_array(
            cast(fd.StockArray, stock.inflow).values, "annual cohort inflow"
        ),
        annual_outflow=_finite_array(
            cast(fd.StockArray, stock.outflow).values, "annual cohort outflow"
        ),
        material_shares=material_shares,
        carbon_content=carbon_content,
        lifetime_mean=_finite_array(parameters["lifetime_mean"].values, "lifetime mean"),
        lifetime_std=_finite_array(
            parameters["lifetime_std"].values, "lifetime standard deviation"
        ),
    )


def apply_parameter_update(parameter: fd.Parameter, update: pd.DataFrame) -> None:
    """Write translated scenario rows into a fresh source parameter array."""
    names = {letter: name for name, letter in _DIMENSION_LETTERS.items()}
    letters = tuple(str(value) for value in parameter.dims.letters)
    indexers: list[np.ndarray] = []
    for letter in letters:
        column = names[letter]
        positions = pd.Index(parameter.dims[letter].items).get_indexer(pd.Index(update[column]))
        if np.any(positions < 0):
            invalid = update.loc[positions < 0, column].iloc[0]
            raise ValueError(
                f"Translated parameter update contains unavailable {column}={invalid!r}"
            )
        indexers.append(positions)
    cast(np.ndarray, parameter.values)[tuple(indexers)] = pd.to_numeric(
        update["amount"], errors="raise"
    ).to_numpy(dtype=float)


def translate_annual_inputs(
    compiled: pd.DataFrame,
    parameter_catalogue: Sequence[Any],
) -> AnnualInputTranslation:
    """Map annual foreground parameter values to REMIND-MFA updates and trade rules.

    Static lifetime/material inputs and the project stock target are excluded:
    they require the prospective cohort continuation used by pyToCE. The
    function prepares the parameter arrays for the subsequent flow calculation.
    """

    catalogue = {item.parameter_id: item for item in parameter_catalogue}
    translated: dict[str, pd.DataFrame] = {}
    deferred: list[pd.DataFrame] = []
    for parameter_id, rows in compiled.groupby("parameter_id", sort=False):
        definition = catalogue[str(parameter_id)]
        upstream_name = definition.upstream_name
        if upstream_name not in _DIRECT_FUTURE_PARAMETERS:
            continue
        is_deferred = rows["value_mode"].isin(_DEFERRED_TRADE_MODES)
        if is_deferred.any():
            selected = rows.loc[
                is_deferred,
                [
                    "parameter_id",
                    "year",
                    "region",
                    "value_mode",
                    "controlling_change_point_year",
                    "default_value",
                    "reference_value",
                    "deferred_propensity",
                    "deferred_denominator_id",
                ],
            ].copy()
            deferred.append(selected)
        rows = rows.loc[~is_deferred]
        if rows.empty:
            continue
        coordinate_columns = [
            dimension
            for dimension in definition.dimensions
            if dimension in {"year", "region", "material", "good"}
        ]
        update = rows.loc[:, [*coordinate_columns, "resolved_value"]].copy()
        update = update.rename(columns={"resolved_value": "amount"})
        update["upstream_name"] = upstream_name
        update["upstream_dimensions"] = "|".join(
            _DIMENSION_LETTERS[dimension] for dimension in coordinate_columns
        )
        translated[upstream_name] = update.reset_index(drop=True)
    deferred_table = (
        pd.concat(deferred, ignore_index=True)
        if deferred
        else pd.DataFrame(
            columns=[
                "parameter_id",
                "year",
                "region",
                "value_mode",
                "controlling_change_point_year",
                "default_value",
                "reference_value",
                "deferred_propensity",
                "deferred_denominator_id",
            ]
        )
    )
    return AnnualInputTranslation(
        numeric_updates=translated,
        deferred_waste_trade=deferred_table,
    )
