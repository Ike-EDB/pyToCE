"""Accumulate pyToCE stocks from annual flows.

Landfill and uncontrolled accumulation follow the REMIND zero reference before
1950 and accumulate the modeled disposal inflows. The atmosphere is a signed nominal carbon
transfer account, distinct from constituent carbon custody and LCIA emissions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cohorts import ProspectiveCohortResult
from .flows import NOMINAL_ELEMENT_BASIS, MFAFlowState

ACCUMULATED_STOCKS = ("landfill", "uncontrolled", "atmospheric")
_TOLERANCE_MT = 1e-12


@dataclass(frozen=True, slots=True)
class MFAStockState:
    """Final stock, inflow and outflow arrays plus independent recurrence checks.

    In-use arrays have year, region, material and good axes, in Mt polymer.
    Landfill and uncontrolled arrays omit good and retain Mt polymer. The
    atmospheric account uses the pooled axes and Mt nominal C. Closing arrays
    detach the final period for reuse as the next scenario period's opening.
    """

    quantities: dict[str, dict[str, np.ndarray]]
    closing: dict[str, np.ndarray]
    validation: pd.DataFrame


def accumulate_mfa_stocks(
    cohort: ProspectiveCohortResult,
    flows: MFAFlowState,
    *,
    opening: dict[str, np.ndarray],
) -> MFAStockState:
    """Accumulate the admitted annual flows once from their actual opening state.

    The first historical period receives zero arrays for each accumulated
    stock. Later periods receive the preceding final state.
    """
    first = cohort.years.index(flows.years[0])
    stop = first + len(flows.years)
    period = slice(first, stop)
    source_carbon = cohort.carbon_content[cohort.elements.index("C")]
    physical = flows.flows
    zero = np.zeros((len(flows.years), len(flows.regions), len(flows.materials)))
    inputs = {
        "landfill": physical["collected => landfill"] + physical["reclmech => landfill"],
        "uncontrolled": physical["eol => mismanaged"],
        "atmospheric": (
            physical["collected => incineration"] + physical["reclmech => incineration"]
        )
        * source_carbon,
    }
    outputs = {
        "landfill": zero,
        "uncontrolled": zero,
        "atmospheric": (
            flows.nominal_feedstock_equivalents["biogenic"]
            + flows.nominal_feedstock_equivalents["daccu"]
        )
        * source_carbon,
    }
    quantities = {
        "in_use": {
            "stock": cohort.material_stock[period],
            "inflow": cohort.material_inflow[period],
            "outflow": cohort.material_outflow[period],
        }
    }
    for name in ACCUMULATED_STOCKS:
        quantities[name] = {
            "stock": opening[name] + np.cumsum(inputs[name] - outputs[name], axis=0),
            "inflow": inputs[name],
            "outflow": outputs[name],
        }
    checks: list[tuple[object, ...]] = []
    for name, values in quantities.items():
        initial = (
            (cohort.material_stock[first - 1] if first else np.zeros_like(values["stock"][0]))
            if name == "in_use"
            else opening[name]
        )
        prior = np.concatenate((initial[None, ...], values["stock"][:-1]), axis=0)
        residual = values["stock"] - prior - values["inflow"] + values["outflow"]
        error = float(np.max(np.abs(residual)))
        # Increment formation, accumulation, addition of the opening and the
        # recurrence subtraction introduce rounding at the stock's scale.
        # Eight elementary operations bound this local arithmetic. Scale each
        # magnitude before addition to keep the bound finite for finite stocks;
        # evaluate each cell independently so one stock cannot mask another.
        rounding_bound = sum(
            8 * np.finfo(np.float64).eps * np.abs(value)
            for value in (values["stock"], prior, values["inflow"], values["outflow"], initial)
        )
        tolerance = np.maximum(_TOLERANCE_MT, rounding_bound)
        valid = np.isfinite(residual) & (np.abs(residual) <= tolerance)
        checks.append(
            (
                name + "_annual_stock_balance",
                "ok" if np.all(valid) else "invalid_stock_balance",
                error,
                float(np.max(tolerance)),
            )
        )
    return MFAStockState(
        quantities,
        {name: quantities[name]["stock"][-1].copy() for name in ACCUMULATED_STOCKS},
        pd.DataFrame.from_records(
            checks, columns=("validation_check", "status", "max_abs_error_mt", "tolerance_mt")
        ),
    )


def mfa_stock_table(
    stocks: MFAStockState,
    flows: MFAFlowState,
    cohort: ProspectiveCohortResult,
    *,
    foreground_scenario: str,
) -> pd.DataFrame:
    """Report stock partitions and the distinct C-only nominal atmosphere."""
    frames: list[pd.DataFrame] = []
    for name, quantities in stocks.quantities.items():
        for quantity, values in quantities.items():
            axes = np.indices(values.shape)
            base = {
                "foreground_scenario": foreground_scenario,
                "year": np.asarray(flows.years)[axes[0].ravel()],
                "region": np.asarray(flows.regions, dtype=object)[axes[1].ravel()],
                "material": np.asarray(flows.materials, dtype=object)[axes[2].ravel()],
                "good": np.asarray(flows.goods, dtype=object)[axes[3].ravel()]
                if values.ndim == 4
                else "all",
                "source_stock_name": name,
                "stock_quantity": quantity,
                "custody": "h12_physical_custody",
            }
            if name == "atmospheric":
                frames.append(
                    pd.DataFrame(
                        {
                            **base,
                            "element": "C",
                            "amount": values.ravel(),
                            "unit": "Mt nominal C",
                            "basis": "signed_nominal_remind_carbon_transfer",
                        }
                    )
                )
            else:
                for element_index, element in enumerate(cohort.elements):
                    fractions = cohort.carbon_content[element_index, axes[2].ravel()]
                    frames.append(
                        pd.DataFrame(
                            {
                                **base,
                                "element": element,
                                "amount": values.ravel() * fractions,
                                "unit": "Mt",
                                "basis": NOMINAL_ELEMENT_BASIS,
                            }
                        )
                    )
    return pd.concat(frames, ignore_index=True)
