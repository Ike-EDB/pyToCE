from fractions import Fraction

import pandas as pd

from pytoce.run_optimization.calculation.interventions import (
    DemandLifetimeDecision,
    build_demand_lifetime_trajectories,
)

_H12 = ("LAM", "OAS", "SSA", "EUR", "NEU", "MEA", "REF", "CAZ", "CHA", "IND", "JPN", "USA")
_GOODS = (
    "Buildings & Construction",
    "Household & Leisure & Sports",
    "Electrical/Electronics",
    "Packaging",
    "Textile sector",
    "Transportation",
    "Others",
)


def _shaped_decision(
    demand_reduction_pct: int, lifetime_extension_pct: int, target_year: int = 2050
) -> DemandLifetimeDecision:
    """Supply a shaped demand path with prescribed linear lifetime for reporting.

    The invented demand path uses 2t-t squared for normalized annual time t.
    Lifetime remains the exact linear reference. This checks transport of
    unequal demand timing at fixed endpoints, not feasibility or LP optimality.
    """
    time = tuple(
        Fraction(min(year - 2019, target_year - 2019), target_year - 2019)
        for year in range(2019, 2101)
    )
    demand = Fraction.from_float(demand_reduction_pct / 100.0)
    return DemandLifetimeDecision(
        demand_reduction_pct,
        lifetime_extension_pct,
        target_year,
        tuple(demand * (2 * value - value * value) for value in time),
        DemandLifetimeDecision(
            demand_reduction_pct, lifetime_extension_pct, target_year
        ).lifetime_fraction,
    )


def _parent_stock() -> pd.DataFrame:
    return pd.DataFrame(
        (
            {
                "year": year,
                "region": region,
                "good": good,
                "stock_projection": 60.0 + region_index + good_index / 10.0 + (year - 2020) / 10.0,
            }
            for region_index, region in enumerate(_H12)
            for good_index, good in enumerate(_GOODS)
            for year in range(2020, 2101)
        )
    )


def _parent_lifetimes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": year,
            "good": good,
            "lifetime_mean": 20.0 + (year - 2020) / 100.0,
            "lifetime_std": 2.0,
        }
        for good in _GOODS
        for year in range(2020, 2101)
    )


def _circularity_parent() -> pd.DataFrame:
    return pd.DataFrame(
        (
            {
                "year": year,
                "region": region,
                "collection_rate": 0.4 + region_index / 100.0,
                "mechanical_recycling_rate": 0.2,
                "chemical_recycling_rate": 0.1,
                "incineration_rate": 0.3,
                "landfill_share": 0.4,
                "fossil_share": 0.7,
                "bio_production_rate": 0.2,
                "daccu_production_rate": 0.1,
            }
            for year in range(2020, 2101)
            for region_index, region in enumerate(_H12)
        )
    )


def _demand(decision: DemandLifetimeDecision):
    return build_demand_lifetime_trajectories(
        parent_stock_targets=_parent_stock(),
        parent_lifetimes=_parent_lifetimes(),
        parent_stock_target_identity="stock::accepted",
        parent_lifetime_identity="lifetime::accepted",
        decision=decision,
    )
