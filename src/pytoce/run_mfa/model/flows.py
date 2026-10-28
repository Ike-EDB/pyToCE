"""Calculate pyToCE polymer flows directly from cohorts and admitted source inputs.

Polymer production and chemical processing have distinct mass boundaries.
Chemical treatment leaves the polymer account as processed feed and preparation
reject; its gross products and monomer use belong to material realization.
The REMIND element split and virgin-feed equivalents provide nominal accounts.
Constituent carbon and LCI demands are calculated separately. The equivalent ratios
retain the admitted virgin-technology controls for LCI provider selection.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe

from ..remind_mfa.process_inputs import RemindMFAProcessInputs
from ..scenario.scenario_contract import OptimizationPhysicalInfeasibility, ScenarioFeasibilityError
from .cohorts import ProspectiveCohortResult
from .materials import (
    MaterialComposition,
    MaterialRealization,
    balanced_primary_trade,
    realize_materials,
)

DIRECT_MFA_METHOD = "pytoce_gross_manufacturing_fixed_recovery_mfa_v4"
NOMINAL_FEEDSTOCK_METHOD = "pytoce_nominal_principal_monomer_feedstock_v1"
MFA_FLOW_FEASIBILITY_CODES = frozenset(
    {"optimization_negative_burden_bearing_physical_flow_infeasible"}
)
NOMINAL_ELEMENT_BASIS = "nominal_remind_element_partition_of_polymer_mass"
_ROUTES = ("mechanical", "chemical", "incineration", "landfill")
_MATERIAL_FLOWS = {
    "mechanical_treatment": "collected => reclmech",
    "chemical_treatment": "collected => reclchem",
    "incineration_treatment": "collected => incineration",
    "landfill_treatment": "collected => landfill",
    "mechanical_resin": "reclmech => processing",
    "mechanical_waterborne_loss": "reclmech => waterborne_loss",
    "mechanical_airborne_loss": "reclmech => airborne_loss",
    "mechanical_landfilled_reject": "reclmech => landfill",
    "mechanical_incinerated_reject": "reclmech => incineration",
    "chemical_processed_feed": "reclchem => chemical_feedstock",
    "chemical_preparation_reject": "reclchem => chemical_preparation_reject",
    "new_polymer_domestic": "new_polymer => processing",
    "new_polymer_export": "new_polymer => primary_market",
    "manufacturing_principal_loss": "processing => manufacturing_loss",
}


@dataclass(frozen=True, slots=True)
class MFAFlowState:
    """MFA arrays and their constituent calculations for one scenario period.

    Flow arrays are Mt on year, region and material axes, with good additionally
    present for final products and discard. Nominal feedstock equivalents have
    the three pooled axes and units Mt polymer equivalent. Material states keep
    the chemical account; the ledger records only applied route/trade changes.
    """

    years: tuple[int, ...]
    regions: tuple[str, ...]
    materials: tuple[str, ...]
    goods: tuple[str, ...]
    flows: dict[str, np.ndarray]
    material_states: tuple[MaterialRealization, ...]
    nominal_feedstock_equivalents: dict[str, np.ndarray]
    reconciliation_ledger: pd.DataFrame


def _regional_treatment(
    waste: np.ndarray, rates: tuple[float, float, float]
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Construct route masses and apply the pyToCE least-change projection.

    The three specified fractions are mechanical, chemical and incineration;
    landfill is their exact complement. Where it is negative, the regional
    four-route projection preserves the available waste total. Exact arithmetic
    prevents rounded route sums from creating a downstream capacity shortage.
    """
    total = sum((Fraction(float(value)) for value in waste), Fraction())
    specified = tuple(Fraction(value) for value in rates)
    requested = tuple(total * rate for rate in (*specified, 1 - sum(specified)))
    if all(value >= 0 for value in requested):
        return requested, requested
    ordered = sorted(requested, reverse=True)
    cumulative = Fraction()
    threshold = Fraction()
    for count, value in enumerate(ordered, 1):
        cumulative += value
        candidate = (cumulative - total) / count
        if value > candidate:
            threshold = candidate
    admitted = tuple(max(value - threshold, Fraction()) for value in requested)
    return requested, admitted


def _admit_physical_amounts(
    name: str, values: np.ndarray, cohort: ProspectiveCohortResult, years: tuple[int, ...]
) -> None:
    """Admit finite, nonnegative derived demand and trade before allocation."""
    if not np.isfinite(values).all():
        raise ValueError(f"Derived {name} contains a nonfinite mass in Mt")
    negative = np.argwhere(values < 0.0)
    if len(negative):
        position = tuple(int(value) for value in negative[0])
        coordinates: tuple[tuple[str, str | int], ...] = (
            ("year", years[position[0]]),
            ("region", cohort.regions[position[1]]),
            ("material", cohort.materials[position[2]]),
        )
        if values.ndim == 4:
            coordinates += (("good", cohort.goods[position[3]]),)
        amount = float(values[position])
        code = "optimization_negative_burden_bearing_physical_flow_infeasible"
        raise ScenarioFeasibilityError(
            f"{name} is {amount:.17g} Mt at {dict(coordinates)}; demand and trade "
            "must be nonnegative. Revise the demand, collection or trade inputs.",
            code=code,
            diagnostic=OptimizationPhysicalInfeasibility(
                code=code,
                equation=f"{name} >= 0",
                observed=amount,
                bound=0.0,
                tolerance=0.0,
                shortfall=-amount,
                status="negative_physical_flow",
                count=len(negative),
                coordinates=coordinates,
            ),
        )


def _nominal_virgin_equivalent(
    polymer: np.ndarray,
    virgin_monomers: dict[str, np.ndarray],
    recovered_monomers: dict[str, np.ndarray],
    recipe: EcoinventMaterialRecipe,
) -> np.ndarray:
    """Project principal-monomer origins to nominal polymer equivalents.

    The adopted equal effective carbon-retention convention weights gross
    virgin and recovered monomers by their admitted carbon contents. This
    reporting projection expresses the resulting polymer equivalents.
    All arrays retain the same constituent axis and are measured in Mt.
    Return one equivalent per constituent; those outside the monomer system
    have no recovered-monomer share.
    """
    carbon = {item.product: item.carbon_kg_per_kg for item in recipe.cracker_products}
    virgin = sum(carbon[name] * virgin_monomers[name] for name in ("ethylene", "propylene"))
    recovered = sum(carbon[name] * recovered_monomers[name] for name in ("ethylene", "propylene"))
    total = virgin + recovered
    fraction = np.divide(virgin, total, out=np.ones_like(virgin), where=total > 0.0)
    return polymer * fraction


def _manufacturing_demand(
    incoming: np.ndarray,
    inputs: RemindMFAProcessInputs,
    first: int,
) -> tuple[np.ndarray, tuple[tuple[tuple[Fraction, ...], ...], ...]]:
    """Keep finished manufacturing demand on the same exact basis as resin trade.

    Sum local good inflows, subtract imported finished goods, add exported
    finished goods and subtract imported intermediates. Locally manufactured
    intermediate exports then add to the finished processing requirement.
    Converting only the rounded sum would falsely reject a full-import endpoint
    whose primary trade retains the original element quantities exactly.
    Binary64 zeros, including signed zero, contribute the exact rational zero. Nonzero terms retain
    their order and exact values.
    """
    fabrication = np.empty(incoming.shape[:3])
    requirements = []
    for time in range(incoming.shape[0]):
        regions = []
        source_time = first + time
        for region in range(incoming.shape[1]):
            families = []
            for material in range(incoming.shape[2]):
                amount = sum(
                    (
                        Fraction(float(value))
                        for value in incoming[time, region, material]
                        if value != 0.0
                    ),
                    Fraction(),
                )
                for sign, values in (
                    (-1, inputs.final_imports[source_time, :, region, material]),
                    (1, inputs.final_exports[source_time, :, region, material]),
                    (-1, inputs.intermediate_imports[source_time, :, region, material]),
                ):
                    amount += sign * sum(
                        (Fraction(float(value)) for value in values.flat if value != 0.0),
                        Fraction(),
                    )
                fabrication[time, region, material] = float(amount)
                amount += sum(
                    (
                        Fraction(float(value))
                        for value in inputs.intermediate_exports[source_time, :, region, material]
                        if value != 0.0
                    ),
                    Fraction(),
                )
                families.append(amount)
            regions.append(tuple(families))
        requirements.append(tuple(regions))
    return fabrication, tuple(requirements)


def calculate_mfa_flows(
    cohort: ProspectiveCohortResult,
    inputs: RemindMFAProcessInputs,
    *,
    recipe: EcoinventMaterialRecipe,
    composition: MaterialComposition,
    first_year: int,
    last_year: int,
    record_diagnostics: bool = True,
) -> MFAFlowState:
    """Calculate final polymer movements once for the requested contiguous years.

    The source owner admits finite arrays and rates. This owner admits derived
    demand and waste, applies the regional treatment projection, then
    calculates constituent flows. Source trade is pooled once.
    ``record_diagnostics`` requests ordinary route/trade provenance; disabling
    it changes no amounts, boundary checks or material products.
    """
    start = cohort.years.index(first_year)
    stop = cohort.years.index(last_year) + 1
    period = slice(start, stop)
    years = cohort.years[period]
    incoming = cohort.material_inflow[period]
    discard = cohort.material_outflow[period]
    final_imports = inputs.final_imports[period].sum(axis=1)
    final_exports = inputs.final_exports[period].sum(axis=1)
    intermediate_imports = inputs.intermediate_imports[period].sum(axis=1)
    intermediate_exports = inputs.intermediate_exports[period].sum(axis=1)
    domestic_goods = incoming - final_imports
    fabrication, manufacturing = _manufacturing_demand(incoming, inputs, start)
    collection = inputs.collection_rate[period, :, None]
    discarded = discard.sum(axis=-1)
    waste_imports = inputs.waste_imports[period].sum(axis=(1, 4))
    waste_exports = inputs.waste_exports[period].sum(axis=(1, 4))
    collected = collection * discarded
    available = collected + waste_imports - waste_exports
    flows = {
        "good_market => use": final_imports,
        "sysenv => good_market": final_imports,
        "fabrication => good_market": final_exports,
        "good_market => sysenv": final_exports,
        "fabrication => use": domestic_goods,
        "intermediate_market => fabrication": intermediate_imports,
        "sysenv => intermediate_market": intermediate_imports,
        "processing => intermediate_market": intermediate_exports,
        "intermediate_market => sysenv": intermediate_exports,
        "processing => fabrication": fabrication,
        "use => eol": discard,
        "eol => collected": collected,
        "eol => mismanaged": (1 - collection) * discarded,
        "waste_market => collected": waste_imports,
        "sysenv => waste_market": waste_imports,
        "collected => waste_market": waste_exports,
        "waste_market => sysenv": waste_exports,
    }
    for name, amounts in (
        ("good_market => use", final_imports),
        ("fabrication => good_market", final_exports),
        ("intermediate_market => fabrication", intermediate_imports),
        ("processing => intermediate_market", intermediate_exports),
        ("fabrication => use", domestic_goods),
        ("processing => fabrication", fabrication),
        ("eol => collected", collected),
        ("eol => mismanaged", flows["eol => mismanaged"]),
        ("waste_market => collected", waste_imports),
        ("collected => waste_market", waste_exports),
        ("available_collected_waste", available),
    ):
        _admit_physical_amounts(name, amounts, cohort, years)
    shape = available.shape
    for name in (
        *_MATERIAL_FLOWS.values(),
        "primary_market => processing",
        "sysenv => primary_market",
        "primary_market => sysenv",
    ):
        flows[name] = np.empty(shape)
    feedstocks = {origin: np.empty(shape) for origin in ("fossil", "biogenic", "daccu")}
    primary_imports = inputs.primary_imports[period].sum(axis=1)
    primary_exports = inputs.primary_exports[period].sum(axis=1)
    states: list[MaterialRealization] = []
    ledger: list[tuple[object, ...]] = []
    for time, year in enumerate(years):
        source_time = start + time
        closed_imports, closed_exports = balanced_primary_trade(
            inputs.primary_imports[source_time], inputs.primary_exports[source_time]
        )
        for region_index, region in enumerate(cohort.regions):
            index = (time, region_index)
            source_index = (source_time, region_index)
            rates = (
                float(inputs.mechanical_recycling_rate[source_index]),
                float(inputs.chemical_recycling_rate[source_index]),
                float(inputs.incineration_rate[source_index]),
            )
            requested, admitted = _regional_treatment(available[index], rates)
            state = realize_materials(
                year=year,
                region=region,
                materials=cohort.materials,
                composition=composition,
                recipe=recipe,
                collected=available[index],
                route_totals=admitted,
                processing_requirement=manufacturing[time][region_index],
                primary_imports=closed_imports[region_index],
                primary_exports=closed_exports[region_index],
            )
            states.append(state)
            for quantity, name in _MATERIAL_FLOWS.items():
                flows[name][index] = np.bincount(
                    state.family_index, weights=state.amounts[quantity], minlength=shape[2]
                )
            flows["primary_market => processing"][index] = state.primary_imports
            flows["sysenv => primary_market"][index] = state.primary_imports
            flows["primary_market => sysenv"][index] = state.primary_exports
            equivalent = np.bincount(
                state.family_index,
                weights=_nominal_virgin_equivalent(
                    state.amounts["new_polymer_domestic"] + state.amounts["new_polymer_export"],
                    state.virgin_monomers,
                    state.recovered_monomers,
                    recipe,
                ),
                minlength=shape[2],
            )
            bio = inputs.bio_production_rate[source_index]
            dac = inputs.daccu_production_rate[source_index]
            feedstocks["fossil"][index] = equivalent * (1 - (bio + dac))
            feedstocks["biogenic"][index] = equivalent * bio
            feedstocks["daccu"][index] = equivalent * dac
            if record_diagnostics:
                for route, before, after in zip(_ROUTES, requested, admitted, strict=True):
                    if before != after:
                        ledger.append(
                            (
                                year,
                                region,
                                "all",
                                route + "_treatment",
                                float(before),
                                float(after),
                                float(after - before),
                                "Mt",
                                "constrained_least_change_projection",
                            )
                        )
                changed_trade = (state.primary_imports != primary_imports[index]) | (
                    state.primary_exports != primary_exports[index]
                )
                for family_index in np.flatnonzero(changed_trade):
                    for quantity, before, after in (
                        ("primary_imports", primary_imports[index], state.primary_imports),
                        ("primary_exports", primary_exports[index], state.primary_exports),
                    ):
                        ledger.append(
                            (
                                year,
                                region,
                                cohort.materials[family_index],
                                quantity,
                                before[family_index],
                                after[family_index],
                                after[family_index] - before[family_index],
                                "Mt",
                                "closed_primary_market_and_constituent_paired_reduction",
                            )
                        )
    return MFAFlowState(
        years,
        cohort.regions,
        cohort.materials,
        cohort.goods,
        flows,
        tuple(states),
        feedstocks,
        pd.DataFrame.from_records(
            ledger,
            columns=(
                "year",
                "region",
                "material",
                "quantity",
                "requested_amount",
                "admitted_amount",
                "delta",
                "unit",
                "method",
            ),
        ),
    )
