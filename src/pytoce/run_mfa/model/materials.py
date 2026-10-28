"""Realize constituent treatment, new polymer and regional monomer requirements.

Tracked polymer mass, gross chemical products and allocated LCI purchases are
separate accounts. This owner calculates the first two in Mt. The LCA owns
inventory coefficients and accounts for the associated production and treatment
burdens. Carbon origins remain owned by foreground custody.
"""

import json
from dataclasses import dataclass
from fractions import Fraction
from importlib.resources import files
from typing import NoReturn

import numpy as np
import pandas as pd

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.shared.runtime.filesystem import payload_sha256

from ..scenario.scenario_contract import OptimizationPhysicalInfeasibility, ScenarioFeasibilityError
from .recycling_allocation import RecyclingCapacityError, mechanical_allocation

MATERIAL_REALIZATION_METHOD = "pytoce_fixed_recovery_gross_manufacturing_realization_v3"
MATERIAL_REALIZATION_COLUMNS = (
    "year",
    "region",
    "flow_name",
    "source_material",
    "source_constituent",
    "destination_material",
    "destination_constituent",
    "product",
    "amount",
    "unit",
)
MATERIAL_FEASIBILITY_CODES = frozenset(
    {
        "optimization_chemical_feed_eligibility_infeasible",
        "optimization_insufficient_paired_primary_trade_infeasible",
        "optimization_recovered_monomer_surplus_infeasible",
        "optimization_material_recovery_capacity_infeasible",
    }
)


@dataclass(frozen=True, slots=True)
class ConversionShare:
    """One operation's fraction of a constituent's finished manufacturing output."""

    operation: str
    share: Fraction


@dataclass(frozen=True, slots=True)
class ConversionPortfolio:
    """A named finished-output mix and the authority for that study selection."""

    name: str
    components: tuple[ConversionShare, ...]
    source_authority_ids: tuple[str, ...]
    interpretation: str


@dataclass(frozen=True, slots=True)
class MaterialConstituent:
    """A family weight, two recycling bindings and finished-output portfolio."""

    material: str
    constituent: str
    share: Fraction
    chemical_polymer: str | None
    mechanical_recovery: str | None
    conversion: ConversionPortfolio


@dataclass(frozen=True, slots=True)
class MaterialComposition:
    """The complete packaged composition and its scientific content identity."""

    identity: str
    constituents: tuple[MaterialConstituent, ...]
    source_authorities: tuple[dict[str, object], ...]


def load_material_composition() -> MaterialComposition:
    """Normalize rounded constituent and finished-output portfolio weights once.

    The resource retains the original rounded portfolio weights. Both manufacturing and LCI routing
    use the same finished-output fractions.
    """
    payload = json.loads(
        files("pytoce.run_mfa.data.catalogues").joinpath("material_constituents.json").read_bytes()
    )
    portfolios = {}
    for name, portfolio in payload["conversion_portfolios"].items():
        rows = portfolio["components"]
        total = sum((Fraction(row["share"]) for row in rows), Fraction())
        portfolios[name] = ConversionPortfolio(
            name,
            tuple(
                ConversionShare(row["operation"], Fraction(row["share"]) / total) for row in rows
            ),
            tuple(portfolio["source_authority_ids"]),
            portfolio["interpretation"],
        )
    family_totals: dict[str, Fraction] = {}
    for row in payload["constituents"]:
        material = row["material"]
        family_totals[material] = family_totals.get(material, Fraction()) + Fraction(row["share"])
    return MaterialComposition(
        f"{payload['schema_version']}::{payload_sha256(payload)}",
        tuple(
            MaterialConstituent(
                row["material"],
                row["constituent"],
                Fraction(row["share"]) / family_totals[row["material"]],
                row["chemical_polymer"],
                row["mechanical_recovery"],
                portfolios[row["conversion_portfolio"]],
            )
            for row in payload["constituents"]
        ),
        tuple(payload["source_authorities"]),
    )


def balanced_primary_trade(
    imports: np.ndarray, exports: np.ndarray
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[tuple[Fraction, ...], ...]]:
    """Realize one closed annual primary market from admitted source trade weights.

    Inputs are Mt on element, region and material axes. For each family, both
    sides use the smaller global volume and retain their regional proportions,
    matching the source's minimum-volume balancing method. Exact binary ratios
    remain local to this market and its paired reduction; casting before that
    subtraction could turn source rounding into a one-sided residual market.
    The source owner already admits finite, nonnegative and balanced inputs.
    """
    regions, materials = imports.shape[1:]
    balanced_imports = [[Fraction(0) for _ in range(materials)] for _ in range(regions)]
    balanced_exports = [[Fraction(0) for _ in range(materials)] for _ in range(regions)]
    for material in range(materials):
        incoming = tuple(
            sum((Fraction(float(value)) for value in imports[:, region, material]), Fraction(0))
            for region in range(regions)
        )
        outgoing = tuple(
            sum((Fraction(float(value)) for value in exports[:, region, material]), Fraction(0))
            for region in range(regions)
        )
        total_in = sum(incoming, Fraction(0))
        total_out = sum(outgoing, Fraction(0))
        volume = min(total_in, total_out)
        if volume:
            for region in range(regions):
                balanced_imports[region][material] = volume * incoming[region] / total_in
                balanced_exports[region][material] = volume * outgoing[region] / total_out
    return (
        tuple(tuple(row) for row in balanced_imports),
        tuple(tuple(row) for row in balanced_exports),
    )


@dataclass(frozen=True, slots=True)
class MaterialRealization:
    """One year's regional MFA state, all quantities in Mt.

    ``amounts`` has constituent-length vectors for collected waste, treatment,
    mechanical resin/rejects, chemical preparation and new polymer outputs.
    Named chemical products retain source constituent; monomer consumption
    retains destination constituent. Primary trade and paired reductions have
    the caller's family axis. The returned quantities retain their constituent axes.
    """

    year: int
    region: str
    constituents: tuple[MaterialConstituent, ...]
    family_index: np.ndarray
    amounts: dict[str, np.ndarray]
    hydrothermal_products: dict[str, np.ndarray]
    cracker_products: dict[str, np.ndarray]
    recovered_monomers: dict[str, np.ndarray]
    virgin_monomers: dict[str, np.ndarray]
    primary_imports: np.ndarray
    primary_exports: np.ndarray
    paired_trade_reduction: np.ndarray


def _capacity_failure(
    *,
    year: int,
    region: str,
    code: str,
    observed: Fraction,
    capacity: Fraction,
    equation: str,
    coordinate: tuple[str, str],
) -> NoReturn:
    """Report the incompatible material-constraint bounds in Mt.

    Active-set bounds can be signed residual quantities. Subtract their exact fractions before
    converting to float: the
    shortfall can be positive even when both displayed bounds round alike.
    """
    raise ScenarioFeasibilityError(
        f"{year}, {region}, {coordinate[0]}={coordinate[1]}: the constraint requires "
        f"{float(observed):.17g} Mt, but the other requested material limits allow at most "
        f"{float(capacity):.17g} Mt. These are constraint bounds, which may be signed. "
        "Revise the circularity or demand inputs.",
        code=code,
        diagnostic=OptimizationPhysicalInfeasibility(
            code=code,
            equation=equation,
            observed=float(observed),
            bound=float(capacity),
            tolerance=0.0,
            shortfall=float(observed - capacity),
            status=code,
            count=1,
            coordinates=(("year", year), ("region", region), coordinate),
        ),
    )


def realize_materials(
    *,
    year: int,
    region: str,
    materials: tuple[str, ...],
    composition: MaterialComposition,
    recipe: EcoinventMaterialRecipe,
    collected: np.ndarray,
    route_totals: tuple[Fraction, ...],
    processing_requirement: tuple[Fraction, ...],
    primary_imports: tuple[Fraction, ...],
    primary_exports: tuple[Fraction, ...],
) -> MaterialRealization:
    """Satisfy gross resin demand with jointly feasible treatment and primary trade.

    Family vectors use ``materials`` order. Regional route totals are exact Mt
    for mechanical, chemical, incineration and landfill treatment. Finished
    ``processing_requirement`` includes local fabrication and intermediate
    exports, before gross manufacturing losses and primary resin trade.

    Fixed finished-output portfolios determine gross input. Mechanical input
    is the least-change feasible allocation respecting constituent receiving
    capacity, chemical feed and both recovered-monomer absorption limits.
    Remaining eligible waste supplies chemistry proportionally; remaining
    terminal inputs preserve the requested incineration/landfill totals.
    Trade, monomer consumption and mass complements close before float output.
    """
    positions = {material: i for i, material in enumerate(materials)}
    constituents = tuple(item for item in composition.constituents if item.material in positions)
    family_index = np.array([positions[item.material] for item in constituents], dtype=np.int64)
    theta = tuple(item.share for item in constituents)
    conversion = {item.operation: item for item in recipe.conversion}
    recovery = {item.name: item for item in recipe.mechanical_recovery}
    functions = tuple(
        recovery[item.mechanical_recovery] if item.mechanical_recovery is not None else None
        for item in constituents
    )
    eta = tuple(
        Fraction(item.recovered_kg_per_input_kg) if item is not None else Fraction()
        for item in functions
    )
    gross_factors = tuple(
        sum(
            (
                operation.share
                * Fraction(conversion[operation.operation].principal_feed_kg_per_finished_kg)
                for operation in item.conversion.components
            ),
            Fraction(),
        )
        for item in constituents
    )
    finished = tuple(
        share * processing_requirement[family]
        for share, family in zip(theta, family_index, strict=True)
    )
    gross = tuple(value * factor for value, factor in zip(finished, gross_factors, strict=True))
    waste = tuple(
        share * Fraction(float(collected[family]))
        for share, family in zip(theta, family_index, strict=True)
    )
    eligible = tuple(item.chemical_polymer is not None for item in constituents)
    mechanical_total, chemical_total, incineration_total, landfill_total = route_totals
    unrecovered_polymer = tuple(
        amount + share * (primary_exports[family] - primary_imports[family])
        for amount, share, family in zip(gross, theta, family_index, strict=True)
    )
    receiving = tuple(
        amount - share * max(primary_imports[family] - primary_exports[family], Fraction())
        for amount, share, family in zip(gross, theta, family_index, strict=True)
    )
    requirements = {item.polymer: item for item in recipe.polymers}
    monomers = ("ethylene", "propylene")
    monomer_use = tuple(
        tuple(
            Fraction(getattr(requirements[item.chemical_polymer], f"{name}_kg_per_kg"))
            if item.chemical_polymer is not None
            else Fraction()
            for item in constituents
        )
        for name in monomers
    )
    processed_fraction = Fraction(recipe.preparation_processed_kg_per_bale_kg) / (
        Fraction(recipe.preparation_processed_kg_per_bale_kg)
        + Fraction(recipe.preparation_reject_kg_per_bale_kg)
    )
    hydrothermal_yields = {
        item.product: Fraction(item.kg_per_kg_feed) for item in recipe.hydrothermal_products
    }
    cracker_yields = {
        item.product: Fraction(item.kg_per_kg_feed) for item in recipe.cracker_products
    }
    cracker_fraction = processed_fraction * (
        hydrothermal_yields["naphtha"] + hydrothermal_yields["distillate_gas_oil"]
    )
    monomer_supply = tuple(
        chemical_total * cracker_fraction * cracker_yields[name] for name in monomers
    )
    headroom = tuple(
        sum((a * n for a, n in zip(coefficients, unrecovered_polymer, strict=True)), Fraction())
        - supply
        for coefficients, supply in zip(monomer_use, monomer_supply, strict=True)
    )
    try:
        mechanical = mechanical_allocation(
            waste=waste,
            recovery=eta,
            receiving_capacity=receiving,
            chemical_eligible=eligible,
            monomer_use=(monomer_use[0], monomer_use[1]),
            monomer_headroom=(headroom[0], headroom[1]),
            mechanical_total=mechanical_total,
            chemical_total=chemical_total,
        )
    except RecyclingCapacityError as error:
        constraint = error.constraint
        code = "optimization_material_recovery_capacity_infeasible"
        if constraint == "chemical feed":
            code = "optimization_chemical_feed_eligibility_infeasible"
        elif constraint.endswith(" absorption"):
            code = "optimization_recovered_monomer_surplus_infeasible"
        elif constraint.startswith("net primary imports "):
            code = "optimization_insufficient_paired_primary_trade_infeasible"
        if constraint.rsplit(" ", 1)[-1].isdigit():
            label, index = constraint.rsplit(" ", 1)
            constraint = f"{constituents[int(index)].constituent}: {label}"
        _capacity_failure(
            year=year,
            region=region,
            code=code,
            observed=error.required,
            capacity=error.maximum,
            equation=f"{constraint}: required <= maximum",
            coordinate=("recycling capacity", constraint),
        )

    remaining = tuple(w - m for w, m in zip(waste, mechanical, strict=True))
    chemical_capacity = sum(
        (value for value, use in zip(remaining, eligible, strict=True) if use), Fraction()
    )
    chemical = tuple(
        value * chemical_total / chemical_capacity if use and chemical_total else Fraction()
        for value, use in zip(remaining, eligible, strict=True)
    )
    terminal = tuple(value - c for value, c in zip(remaining, chemical, strict=True))
    terminal_total = incineration_total + landfill_total
    incineration = tuple(
        value * incineration_total / terminal_total if terminal_total else Fraction()
        for value in terminal
    )
    landfill = tuple(value - inc for value, inc in zip(terminal, incineration, strict=True))
    mechanical_output = tuple(value * factor for value, factor in zip(mechanical, eta, strict=True))
    waterborne = tuple(
        value * Fraction(function.waterborne_kg_per_input_kg) if function else Fraction()
        for value, function in zip(mechanical, functions, strict=True)
    )
    airborne = tuple(
        value * Fraction(function.airborne_kg_per_input_kg) if function else Fraction()
        for value, function in zip(mechanical, functions, strict=True)
    )
    managed = tuple(
        value - resin - water - air
        for value, resin, water, air in zip(
            mechanical, mechanical_output, waterborne, airborne, strict=True
        )
    )
    managed_landfill = tuple(
        value if function is not None and function.managed_treatment == "landfill" else Fraction()
        for value, function in zip(managed, functions, strict=True)
    )
    managed_incineration = tuple(a - b for a, b in zip(managed, managed_landfill, strict=True))

    # A paired family trade reduction preserves net trade and total new resin.
    # The allocation bounds guarantee enough paired trade for every constituent.
    reductions = [Fraction() for _ in materials]
    for family, share, output, required in zip(
        family_index, theta, mechanical_output, gross, strict=True
    ):
        reductions[family] = max(
            reductions[family], primary_imports[family] + (output - required) / share
        )
    residual_imports = tuple(i - d for i, d in zip(primary_imports, reductions, strict=True))
    residual_exports = tuple(x - d for x, d in zip(primary_exports, reductions, strict=True))
    domestic_polymer = tuple(
        amount - share * residual_imports[family] - output
        for amount, share, family, output in zip(
            gross, theta, family_index, mechanical_output, strict=True
        )
    )
    exported_polymer = tuple(
        share * residual_exports[family] for share, family in zip(theta, family_index, strict=True)
    )
    new_polymer = tuple(d + x for d, x in zip(domestic_polymer, exported_polymer, strict=True))
    processed = tuple(value * processed_fraction for value in chemical)
    preparation_reject = tuple(c - p for c, p in zip(chemical, processed, strict=True))
    hydrothermal = {
        name: tuple(value * factor for value in processed)
        for name, factor in hydrothermal_yields.items()
    }
    cracker_feed = tuple(c * cracker_fraction for c in chemical)
    cracker = {
        name: tuple(value * factor for value in cracker_feed)
        for name, factor in cracker_yields.items()
    }
    recovered: dict[str, np.ndarray] = {}
    virgin: dict[str, np.ndarray] = {}
    for name, coefficients, supply in zip(monomers, monomer_use, monomer_supply, strict=True):
        demand = tuple(a * n for a, n in zip(coefficients, new_polymer, strict=True))
        total_demand = sum(demand, Fraction())
        fraction = supply / total_demand if total_demand else Fraction()
        consumed = tuple(value * fraction for value in demand)
        recovered[name] = np.array([float(value) for value in consumed])
        virgin[name] = np.array(
            [float(value - used) for value, used in zip(demand, consumed, strict=True)]
        )
    amounts = {
        "collected_after_trade": waste,
        "mechanical_treatment": mechanical,
        "chemical_treatment": chemical,
        "incineration_treatment": incineration,
        "landfill_treatment": landfill,
        "mechanical_resin": mechanical_output,
        "mechanical_waterborne_loss": waterborne,
        "mechanical_airborne_loss": airborne,
        "mechanical_incinerated_reject": managed_incineration,
        "mechanical_landfilled_reject": managed_landfill,
        "chemical_processed_feed": processed,
        "chemical_preparation_reject": preparation_reject,
        "new_polymer_domestic": domestic_polymer,
        "new_polymer_export": exported_polymer,
        "manufacturing_principal_input": gross,
        "manufacturing_finished_output": finished,
        "manufacturing_principal_loss": tuple(g - f for g, f in zip(gross, finished, strict=True)),
    }
    return MaterialRealization(
        year,
        region,
        constituents,
        family_index,
        {name: np.array([float(value) for value in values]) for name, values in amounts.items()},
        {
            name: np.array([float(value) for value in values])
            for name, values in hydrothermal.items()
        },
        {name: np.array([float(value) for value in values]) for name, values in cracker.items()},
        recovered,
        virgin,
        np.array([float(value) for value in residual_imports]),
        np.array([float(value) for value in residual_exports]),
        np.array([float(value) for value in reductions]),
    )


def material_realization_table(state: MaterialRealization) -> pd.DataFrame:
    """Report constituent supplies and destinations using their known coordinates."""
    rows: list[tuple[object, ...]] = []

    def append(flow: str, product: str, amounts: np.ndarray, *, destination: bool) -> None:
        """Keep the known source or destination labels on each quantity."""
        for item, amount in zip(state.constituents, amounts, strict=True):
            source = (None, None) if destination else (item.material, item.constituent)
            target = (item.material, item.constituent) if destination else (None, None)
            rows.append(
                (state.year, state.region, flow, *source, *target, product, float(amount), "Mt")
            )

    for name, amounts in state.amounts.items():
        append(name, "polymer", amounts, destination=name.startswith("new_polymer"))
    for group, products, destination in (
        ("hydrothermal_product", state.hydrothermal_products, False),
        ("cracker_product", state.cracker_products, False),
        ("recovered_monomer_consumption", state.recovered_monomers, True),
        ("virgin_monomer_consumption", state.virgin_monomers, True),
    ):
        for product, amounts in products.items():
            append(group, product, amounts, destination=destination)
    table = pd.DataFrame.from_records(rows, columns=MATERIAL_REALIZATION_COLUMNS)
    return table.sort_values(
        list(MATERIAL_REALIZATION_COLUMNS[:8]), kind="stable", na_position="last"
    ).reset_index(drop=True)
