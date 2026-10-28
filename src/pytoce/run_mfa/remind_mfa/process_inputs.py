"""Prepare source trade and process inputs for the direct pyToCE MFA."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

import flodym as fd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from pytoce.shared.runtime.filesystem import canonical_table_sha256

from ..scenario.scenario_contract import OptimizationPhysicalInfeasibility, ScenarioFeasibilityError
from .trade_projection import HistoricalTradeProjection
from .translation import apply_parameter_update

_WASTE_TRADE_PROPENSITY_SHA256 = "ced41733bcbc7d748978622b718ac8f51cda7d35be303666f0996d602dc70960"
_COORDINATE_LETTERS = {"t": "year", "r": "region", "m": "material", "g": "good", "e": "element"}
_TRADE_TOLERANCE_MT = 1e-12
SOURCE_TRADE_BALANCE_INFEASIBLE = "optimization_source_trade_balance_infeasible"
WASTE_TRADE_LEDGER_COLUMNS = (
    "year",
    "region",
    "local_pretrade_collected_mt",
    "import_propensity",
    "export_propensity",
    "import_propensity_basis",
    "export_propensity_basis",
    "import_propensity_authority_identity",
    "export_propensity_authority_identity",
    "source_input_waste_imports_mt",
    "source_input_waste_exports_mt",
    "balanced_waste_imports_mt",
    "balanced_waste_exports_mt",
    "available_collected_after_trade_mt",
    "import_formula_residual_mt",
    "export_formula_residual_mt",
    "global_trade_balance_residual_mt",
    "deferred_denominator_id",
    "status",
)


@dataclass(frozen=True, slots=True)
class RemindMFAProcessInputs:
    """Carry candidate-specific native trade and annual operational controls.

    All trade quantities are Mt. Final goods and waste use year, element,
    region, material and good axes; intermediate and primary trade use the
    year, element, region, and material axes. Rates have year and region axes. Each evaluation owns
    its parameter arrays.
    """

    final_imports: np.ndarray
    final_exports: np.ndarray
    intermediate_imports: np.ndarray
    intermediate_exports: np.ndarray
    primary_imports: np.ndarray
    primary_exports: np.ndarray
    waste_imports: np.ndarray
    waste_exports: np.ndarray
    collection_rate: np.ndarray
    mechanical_recycling_rate: np.ndarray
    chemical_recycling_rate: np.ndarray
    incineration_rate: np.ndarray
    bio_production_rate: np.ndarray
    daccu_production_rate: np.ndarray
    waste_trade_propensity_ledger: pd.DataFrame


def _waste_trade(
    outflow: fd.FlodymArray,
    parameters: dict[str, fd.Parameter],
    dimensions: fd.DimensionSet,
    trade_api: ModuleType,
) -> Any:
    """Split source waste trade by discarded material and balance to the maximum.

    Use the public REMIND-MFA 0.2.0 Trade class. Waste keeps its source array
    order during balancing; conversion to pyToCE axes occurs afterward.
    Very small positive totals use a common binary unit on both sides so the
    source's epsilon divisor cannot change its maximum-balance equation.
    """
    trade = trade_api.Trade(
        imports=fd.FlodymArray(dims=dimensions.get_subset(("t", "r", "m", "e", "g"))),
        exports=fd.FlodymArray(dims=dimensions.get_subset(("t", "r", "m", "e", "g"))),
    )
    split = outflow.get_shares_over(("g", "e", "m"))
    trade.imports[...] = parameters["waste_imports"] * split
    trade.exports[...] = parameters["waste_exports"] * split
    imports = trade.imports.sum_over(("r",)).values
    exports = trade.exports.sum_over(("r",)).values
    positive_minimum = np.where(
        (imports > 0) & (exports > 0),
        np.minimum(imports, exports),
        np.maximum(imports, exports),
    )
    # REMIND-MFA v0.2.0 Trade.balance divides by max(total, binary64 epsilon).
    # In Mt this distorts positive historical tails below 2**-52. A common
    # power-of-two unit makes the smaller positive total lie in [1, 2),
    # preserving both regional proportions and the original maximum total.
    # Unaffected coordinates keep exponent zero; a missing trade side stays zero.
    small = (positive_minimum > 0) & (positive_minimum < np.finfo(float).eps)
    exponent = np.zeros_like(imports, dtype=np.int32)
    exponent[small] = np.frexp(positive_minimum[small])[1] - 1
    # Native waste axes are t,r,m,e,g; one common unit spans the regional axis.
    regional_exponent = exponent[:, np.newaxis, ...]
    with np.errstate(over="raise", invalid="raise"):
        trade.imports.values = np.ldexp(trade.imports.values, -regional_exponent)
        trade.exports.values = np.ldexp(trade.exports.values, -regional_exponent)
        trade.balance(to="maximum")
        trade.imports.values = np.ldexp(trade.imports.values, regional_exponent)
        trade.exports.values = np.ldexp(trade.exports.values, regional_exponent)
    _admit_trade_balance(trade, "waste")
    return trade


def _admit_trade_balance(trade: Any, kind: str) -> None:
    """Require native imports and exports to agree for each traded coordinate.

    Sum only regions: element, material and good cannot compensate for one
    another. Native balancing divides by at least machine epsilon; maximum
    balancing can therefore leave unmatched trade when one side is zero.
    The tolerance covers floating summation.
    """
    imports = trade.imports.sum_over(("r",))
    exports = trade.exports.sum_over(("r",))
    difference = imports.values - exports.values
    scale = np.abs(imports.values) + np.abs(exports.values)
    # Gamma bounds the regional reduction plus division/multiplication and
    # comparison rounding. An absolute mass floor would admit unmatched tiny
    # trades caused by the source's machine-epsilon denominator.
    rounding = (len(trade.imports.dims["r"].items) + 4) * np.finfo(float).eps / 2
    tolerance = (2 * rounding / (1 - rounding)) * scale
    invalid = np.argwhere(np.abs(difference) > tolerance)
    if len(invalid):
        index = tuple(int(value) for value in invalid[0])
        coordinates = tuple(
            (
                _COORDINATE_LETTERS[letter],
                int(imports.dims[letter].items[position])
                if letter == "t"
                else str(imports.dims[letter].items[position]),
            )
            for letter, position in zip(imports.dims.letters, index, strict=True)
        )
        residual = float(difference[index])
        raise ScenarioFeasibilityError(
            f"Native {kind} trade is unbalanced at {dict(coordinates)}: "
            f"imports={float(imports.values[index])} Mt, "
            f"exports={float(exports.values[index])} Mt. Global imports and exports "
            "must agree for each traded element, material and good. Revise the trade inputs.",
            code=SOURCE_TRADE_BALANCE_INFEASIBLE,
            diagnostic=OptimizationPhysicalInfeasibility(
                code=SOURCE_TRADE_BALANCE_INFEASIBLE,
                equation="global_imports - global_exports = 0",
                observed=residual,
                bound=0.0,
                tolerance=float(tolerance[index]),
                shortfall=abs(residual),
                status="unbalanced_native_trade",
                count=len(invalid),
                coordinates=coordinates,
                details=(("trade_kind", kind),),
            ),
        )


def _time_region_frame(array: Any, value_name: str) -> pd.DataFrame:
    """Return a year and region table from a FLODYM array."""

    frame = array.to_df(index=False, sparse=False)
    rename = {
        str(array.dims[letter].name): _COORDINATE_LETTERS[letter] for letter in array.dims.letters
    }
    frame = frame.rename(columns={**rename, "value": value_name})
    return frame.loc[:, ["year", "region", value_name]]


def default_waste_trade_propensities(
    outflow: fd.FlodymArray, parameters: dict[str, fd.Parameter]
) -> pd.DataFrame:
    """Derive exact annual source-input propensities from the default model."""

    local = _time_region_frame(
        (outflow * parameters["collection_rate"]).sum_over(("g", "e", "m")),
        "default_local_pretrade_collected_mt",
    )
    frames: list[pd.DataFrame] = []
    for parameter_id in ("waste_imports", "waste_exports"):
        source = _time_region_frame(
            parameters[parameter_id],
            "default_source_amount_mt",
        )
        frame = local.merge(source, on=["year", "region"], validate="one_to_one")
        denominator = frame["default_local_pretrade_collected_mt"].to_numpy(dtype=float)
        amount = frame["default_source_amount_mt"].to_numpy(dtype=float)
        zero = denominator == 0.0
        if np.any(zero & (amount != 0.0)):
            raise ValueError("Default waste trade has a nonzero amount and zero denominator")
        propensity = np.divide(
            amount,
            denominator,
            out=np.zeros_like(amount),
            where=~zero,
        )
        if (
            not np.isfinite(propensity).all()
            or np.any(propensity < 0.0)
            or (parameter_id == "waste_exports" and np.any(propensity > 1.0 + 1e-12))
        ):
            raise ValueError("Default waste-trade propensities are invalid")
        frame["parameter_id"] = parameter_id
        frame["default_propensity"] = propensity
        frames.append(frame.loc[:, ["parameter_id", "year", "region", "default_propensity"]])
    result = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["parameter_id", "year", "region"], kind="stable")
        .reset_index(drop=True)
    )
    result["default_propensity_authority_identity"] = (
        "pytoce_remind_mfa_default_waste_trade_propensity_v1::" + canonical_table_sha256(result)
    )
    return result


def _resolved_waste_trade(
    outflow: fd.FlodymArray,
    parameters: dict[str, fd.Parameter],
    dimensions: fd.DimensionSet,
    trade_api: ModuleType,
    deferred: pd.DataFrame,
    default_propensities: pd.DataFrame,
    *,
    record_diagnostics: bool = True,
) -> tuple[Any, pd.DataFrame]:
    """Calculate regional waste imports and exports before REMIND-MFA balances trade.

    Each requested import or export amount equals its propensity
    multiplied by local collected waste before trade, in Mt. The values are
    written to fresh REMIND-MFA parameter arrays, the upstream balancing
    calculation is run once, and the returned ledger records source values,
    balanced flows, and residuals by year and world region.

    Args:
        outflow: Annual discarded element mass in Mt on year, element,
            region, material and good axes.
        parameters: Fresh extrapolated parameters for this candidate.
        dimensions: Admitted source coordinates.
        trade_api: Exact source module containing the public Trade class.
        deferred: Scenario rules whose waste-trade mass depends on calculated
            local collection.
        default_propensities: Annual default import and export propensities.
        record_diagnostics: Whether to construct the ordinary provenance ledger.

    Returns:
        The balanced source Trade and its calculation ledger. The ledger is
        empty when no rule changes waste trade or ``record_diagnostics`` is false.

    Raises:
        ValueError: If a calculated amount is negative, not finite, or exports
            exceed locally collected waste.
    """

    if deferred.empty:
        return _waste_trade(outflow, parameters, dimensions, trade_api), pd.DataFrame(
            columns=WASTE_TRADE_LEDGER_COLUMNS
        )

    local_pretrade = (outflow * parameters["collection_rate"]).sum_over(("g", "e", "m"))
    local = _time_region_frame(
        local_pretrade,
        "local_pretrade_collected_mt",
    )
    rules = deferred.merge(local, on=["year", "region"], how="left", validate="many_to_one")
    rules = rules.merge(
        default_propensities,
        on=["parameter_id", "year", "region"],
        how="left",
        validate="one_to_one",
    )
    uses_default = rules["value_mode"].eq("retain_default_propensity_pretrade_collected")
    rules["resolved_propensity"] = np.where(
        uses_default,
        rules["default_propensity"],
        rules["deferred_propensity"],
    )
    rules["propensity_basis"] = np.where(
        uses_default,
        "contemporaneous_remind_mfa_default",
        "accepted_2019_pretrade_collection",
    )
    rules["propensity_authority_identity"] = np.where(
        uses_default,
        rules["default_propensity_authority_identity"],
        "pytoce_waste_trade_2019_propensity_authority_v1::" + _WASTE_TRADE_PROPENSITY_SHA256,
    )
    # Scenario values are propensities because the corresponding waste mass is
    # available only after collection has been calculated.
    rules["source_amount_mt"] = rules["resolved_propensity"] * rules["local_pretrade_collected_mt"]
    invalid = (
        ~np.isfinite(rules["source_amount_mt"])
        | rules["source_amount_mt"].lt(0.0)
        | (
            rules["parameter_id"].eq("waste_exports")
            & rules["source_amount_mt"].gt(
                rules["local_pretrade_collected_mt"] + _TRADE_TOLERANCE_MT
            )
        )
    )
    if invalid.any():
        row = rules.loc[invalid].iloc[0]
        raise ValueError(
            "Deferred waste trade is infeasible for "
            f"{row['parameter_id']} at year={int(row['year'])}, "
            f"region={row['region']}"
        )

    for parameter_id, rows in rules.groupby("parameter_id", sort=False):
        parameter = parameters[str(parameter_id)]
        update = rows.loc[:, ["year", "region", "source_amount_mt"]].rename(
            columns={"source_amount_mt": "amount"}
        )
        apply_parameter_update(parameter, update)

    trade = _waste_trade(outflow, parameters, dimensions, trade_api)
    if not record_diagnostics:
        return trade, pd.DataFrame(columns=WASTE_TRADE_LEDGER_COLUMNS)
    source_imports = _time_region_frame(
        parameters["waste_imports"],
        "source_input_waste_imports_mt",
    )
    source_exports = _time_region_frame(
        parameters["waste_exports"],
        "source_input_waste_exports_mt",
    )
    balanced_imports = _time_region_frame(
        trade.imports.sum_over(("g", "e", "m")),
        "balanced_waste_imports_mt",
    )
    balanced_exports = _time_region_frame(
        trade.exports.sum_over(("g", "e", "m")),
        "balanced_waste_exports_mt",
    )

    coordinates = local.loc[
        local["year"].isin(deferred["year"].unique()),
        ["year", "region"],
    ].drop_duplicates()
    ledger = coordinates.merge(local, on=["year", "region"], validate="one_to_one")
    for frame in (source_imports, source_exports, balanced_imports, balanced_exports):
        ledger = ledger.merge(frame, on=["year", "region"], validate="one_to_one")
    for parameter_id, prefix in (
        ("waste_imports", "import"),
        ("waste_exports", "export"),
    ):
        values = (
            deferred.loc[
                deferred["parameter_id"].eq(parameter_id),
                ["year", "region"],
            ]
            .merge(
                rules.loc[
                    rules["parameter_id"].eq(parameter_id),
                    [
                        "year",
                        "region",
                        "resolved_propensity",
                        "propensity_basis",
                        "propensity_authority_identity",
                    ],
                ],
                on=["year", "region"],
                validate="one_to_one",
            )
            .rename(
                columns={
                    "resolved_propensity": f"{prefix}_propensity",
                    "propensity_basis": f"{prefix}_propensity_basis",
                    "propensity_authority_identity": (f"{prefix}_propensity_authority_identity"),
                }
            )
        )
        ledger = ledger.merge(values, on=["year", "region"], how="left", validate="one_to_one")
    # These residuals show that the stored mass equals propensity times local
    # collection; the global residual checks REMIND-MFA's trade balance.
    ledger["import_formula_residual_mt"] = np.where(
        ledger["import_propensity"].notna(),
        ledger["source_input_waste_imports_mt"]
        - ledger["import_propensity"] * ledger["local_pretrade_collected_mt"],
        np.nan,
    )
    ledger["export_formula_residual_mt"] = np.where(
        ledger["export_propensity"].notna(),
        ledger["source_input_waste_exports_mt"]
        - ledger["export_propensity"] * ledger["local_pretrade_collected_mt"],
        np.nan,
    )
    ledger["available_collected_after_trade_mt"] = (
        ledger["local_pretrade_collected_mt"]
        + ledger["balanced_waste_imports_mt"]
        - ledger["balanced_waste_exports_mt"]
    )
    balance_table = ledger.groupby("year", sort=False)[
        ["balanced_waste_imports_mt", "balanced_waste_exports_mt"]
    ].sum()
    balance = cast(
        pd.Series,
        balance_table["balanced_waste_imports_mt"] - balance_table["balanced_waste_exports_mt"],
    )
    ledger["global_trade_balance_residual_mt"] = ledger["year"].map(balance.to_dict())
    ledger["deferred_denominator_id"] = "local_pretrade_collected_waste"
    ledger["status"] = "resolved"
    return trade, (
        cast(pd.DataFrame, ledger.loc[:, list(WASTE_TRADE_LEDGER_COLUMNS)])
        .sort_values(["year", "region"], kind="stable")
        .reset_index(drop=True)
    )


def calculate_process_inputs(
    *,
    inflow: fd.FlodymArray,
    outflow: fd.FlodymArray,
    parameters: dict[str, fd.Parameter],
    historical_trade: Mapping[str, Any],
    dimensions: fd.DimensionSet,
    trade_api: ModuleType,
    trade_projection: HistoricalTradeProjection,
    deferred_waste_trade: pd.DataFrame,
    default_propensities: pd.DataFrame,
    record_diagnostics: bool = True,
) -> RemindMFAProcessInputs:
    """Extrapolate trade from candidate stock demand using REMIND-MFA 0.2.0 APIs.

    Historical references have already been balanced by the historical source
    owner. Historical proportional fits belong to the retained session; future
    demand still determines fresh projections, balanced trade and import caps.
    Final, intermediate/manufactured and primary trade are scaled in that
    dependency order using the public REMIND-MFA 0.2.0 regression and trade APIs.
    """
    fine_dims = dimensions.get_subset(("t", "e", "r", "m", "g"))
    pooled_dims = dimensions.get_subset(("t", "e", "r", "m"))

    def extrapolate(name: str, split: fd.FlodymArray, scaler: fd.FlodymArray) -> Any:
        """Apply source trade equations with retained historical proportional fits."""
        source = historical_trade[name + "_his"]
        historical_split = split[{"t": dimensions["h"]}]
        historical_imports = source.imports * historical_split
        historical_exports = source.exports * historical_split
        missing = tuple(
            letter
            for letter in scaler.dims.letters[1:]
            if letter not in historical_imports.dims.letters
        )
        if missing:
            # The source expands pooled intermediate trade over historical goods
            # before its independent fits; only the resulting trade is pooled.
            shares = scaler[{"t": dimensions["h"]}].get_shares_over(missing).apply(np.nan_to_num)
            historical_imports = historical_imports * shares
            historical_exports = historical_exports * shares
        letters = ("t", "r", "m", "e", "g") if name == "final" else ("t", "r", "m", "e")
        future = trade_api.Trade(
            imports=fd.FlodymArray(dims=dimensions.get_subset(letters)),
            exports=fd.FlodymArray(dims=dimensions.get_subset(letters)),
        )
        future.imports[...] = trade_projection.project(
            (name, "imports"), historical_imports, scaler
        )
        exports_predictor = (scaler + future.net_imports).maximum(0.0)
        future.exports[...] = trade_projection.project(
            (name, "exports"), historical_exports, exports_predictor
        )
        future.balance(to="hmean")
        future.balance(to="minimum")
        # Limit imports to local demand after the two native balances.
        # If this lowers imports, reduce paired exports through the minimum balance.
        limited_imports = future.imports.minimum(scaler)
        if np.any(limited_imports.values < future.imports.values):
            future.imports = limited_imports
            future.balance(to="minimum")
        _admit_trade_balance(future, name)
        return future

    final = extrapolate("final", inflow.get_shares_over(("g", "e", "m")), inflow)
    final_imports = fd.FlodymArray(dims=fine_dims)
    final_exports = fd.FlodymArray(dims=fine_dims)
    final_imports[...] = final.imports
    final_exports[...] = final.exports
    fabrication = fd.FlodymArray(dims=fine_dims)
    fabrication[...] = inflow - final_imports
    fabrication_split = fabrication.sum_over(("g",)).get_shares_over(("e", "m"))
    intermediate = extrapolate("intermediate", fabrication_split, fabrication)
    manufactured = extrapolate("manufactured", fabrication_split, fabrication)
    intermediate_imports = fd.FlodymArray(dims=pooled_dims)
    intermediate_exports = fd.FlodymArray(dims=pooled_dims)
    intermediate_imports[...] = intermediate.imports + manufactured.imports
    intermediate_exports[...] = intermediate.exports + manufactured.exports
    processing = fd.FlodymArray(dims=pooled_dims)
    processing[...] = fabrication + final_exports - intermediate_imports
    primary = extrapolate("primary", processing.get_shares_over(("e", "m")), processing)
    waste, ledger = _resolved_waste_trade(
        outflow,
        parameters,
        dimensions,
        trade_api,
        deferred_waste_trade,
        default_propensities,
        record_diagnostics=record_diagnostics,
    )

    def values(array: fd.FlodymArray, dims: fd.DimensionSet) -> np.ndarray:
        """Detach source quantities in their declared pyToCE axis order."""
        result = np.array(array.cast_values_to(dims), dtype=float, copy=True, order="C")
        if not np.isfinite(result).all():
            raise ValueError("REMIND-MFA trade or process input contains a nonfinite value")
        return result

    rate_dims = dimensions.get_subset(("t", "r"))
    result = RemindMFAProcessInputs(
        final_imports=values(final_imports, fine_dims),
        final_exports=values(final_exports, fine_dims),
        intermediate_imports=values(intermediate_imports, pooled_dims),
        intermediate_exports=values(intermediate_exports, pooled_dims),
        primary_imports=values(primary.imports, pooled_dims),
        primary_exports=values(primary.exports, pooled_dims),
        waste_imports=values(waste.imports, fine_dims),
        waste_exports=values(waste.exports, fine_dims),
        collection_rate=values(parameters["collection_rate"], rate_dims),
        mechanical_recycling_rate=values(parameters["mechanical_recycling_rate"], rate_dims),
        chemical_recycling_rate=values(parameters["chemical_recycling_rate"], rate_dims),
        incineration_rate=values(parameters["incineration_rate"], rate_dims),
        bio_production_rate=values(parameters["bio_production_rate"], rate_dims),
        daccu_production_rate=values(parameters["daccu_production_rate"], rate_dims),
        waste_trade_propensity_ledger=ledger,
    )
    for name in (
        "collection_rate",
        "mechanical_recycling_rate",
        "chemical_recycling_rate",
        "incineration_rate",
        "bio_production_rate",
        "daccu_production_rate",
    ):
        amounts = getattr(result, name)
        invalid = np.argwhere((amounts < 0.0) | (amounts > 1.0))
        if len(invalid):
            index = tuple(int(value) for value in invalid[0])
            coordinates = {
                _COORDINATE_LETTERS[letter]: dimensions[letter].items[position]
                for letter, position in zip(("t", "r"), index, strict=True)
            }
            raise ValueError(
                f"REMIND-MFA parameter {name} is {amounts[index]} at {coordinates}; "
                "a fraction in [0, 1] is required. Revise the source or foreground inputs."
            )
    alternative = result.bio_production_rate + result.daccu_production_rate
    invalid = np.argwhere(alternative > 1.0)
    if len(invalid):
        time, region = (int(value) for value in invalid[0])
        raise ValueError(
            f"Biogenic plus DAC feedstock shares are {alternative[time, region]} at "
            f"year={dimensions['t'].items[time]}, region={dimensions['r'].items[region]}; "
            "their sum must not exceed 1. Revise the source or foreground inputs."
        )
    return result
