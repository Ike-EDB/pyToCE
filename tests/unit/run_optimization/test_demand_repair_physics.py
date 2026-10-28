"""Demand-only receiving repairs through real annual cohorts and material flows."""

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import pytest
from material_recipe_fixtures import material_source_members
from scipy.stats import lognorm

from pytoce.download_data.sources.ecoinvent_materials import (
    EcoinventMaterialRecipe,
    derive_material_recipe,
)
from pytoce.run_mfa.model.cohorts import (
    ProspectiveCohortResult,
    continue_cohort_stock,
    fixed_lifetime_stock_system,
)
from pytoce.run_mfa.model.flows import MFAFlowState, calculate_mfa_flows
from pytoce.run_mfa.model.materials import load_material_composition
from pytoce.run_mfa.remind_mfa.process_inputs import RemindMFAProcessInputs
from pytoce.run_mfa.remind_mfa.translation import NormalizedCohortState
from pytoce.run_mfa.scenario.scenario_contract import ScenarioFeasibilityError
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    demand_repair_neighbors,
    propose_annual_demand,
)
from pytoce.run_optimization.calculation.interventions import DemandLifetimeDecision


@dataclass(frozen=True)
class ReceivingCase:
    """One PS packaging cohort with a fixed, inactive Block I treatment schedule."""

    history: NormalizedCohortState
    inputs: RemindMFAProcessInputs
    recipe: EcoinventMaterialRecipe
    parent_stock: np.ndarray
    reference: DemandLifetimeDecision
    gross_factor: float
    recovery: float
    survival: tuple[float, float, float]


def _compiled(case: ReceivingCase, decision: DemandLifetimeDecision) -> pd.DataFrame:
    """Supply the real cohort owner with all prospective stock and lifetime rows."""
    rows = []
    for year in range(2020, 2101):
        target = case.parent_stock[year - 2020, 0, 0] * (
            1.0 - float(decision.demand_fraction[year - 2019])
        )
        for parameter, value, region, material in (
            ("stock_projection", target, "EUR", None),
            ("lifetime_mean", 1.0, None, None),
            ("lifetime_std", 0.2, None, None),
            ("plastic_family_share", 1.0, "EUR", "PS"),
        ):
            rows.append((parameter, year, region, material, "Packaging", value))
    return pd.DataFrame.from_records(
        rows,
        columns=("parameter_id", "year", "region", "material", "good", "resolved_value"),
    )


def _receiving_case(recipe: EcoinventMaterialRecipe, deficit_fraction: float) -> ReceivingCase:
    """Calculate the receiving deficit algebraically.

    The accepted PS recipe fixture has foam/solid/other finished pools 9.5/9.9/1.2
    and respective yields 1/.8/.6. For same-year survival f, existing survivors A
    and discard E, gross input I=(S-A)/f and waste W=E+(1-f)I. Thus the exact
    receiving inequality gI >= eta W is F=aS-aA-eta E >= 0, with
    a=(g-eta(1-f))/f. Only 2021 uses mechanical recycling; this prescribed parent
    control is inactive in Block I and remains identical for every tested path.
    """
    reference = DemandLifetimeDecision(50, 0)
    d0, d1 = (float(reference.demand_fraction[i]) for i in (1, 2))
    sigma = np.sqrt(np.log1p(0.2**2))
    survival = tuple(
        float(lognorm.sf(age, sigma, scale=np.exp(-(sigma**2) / 2))) for age in (0.5, 1.5, 2.5)
    )
    f, sf1, sf2 = survival
    ratio = sf1 / f
    inherited = sf2 / f + (1.0 - d0 - ratio) * ratio
    discard = 1.0 - d0 - inherited
    gross = float(
        (Fraction(9.5) + Fraction(9.9) * Fraction(1 / 0.8) + Fraction(1.2) * Fraction(1 / 0.6))
        / (Fraction(9.5) + Fraction(9.9) + Fraction(1.2))
    )
    eta = next(
        row.recovered_kg_per_input_kg
        for row in recipe.mechanical_recovery
        if row.name == "ps_granulate"
    )
    a = (gross - eta * (1.0 - f)) / f
    b = a * ratio + eta * (1.0 - ratio)
    early_change, binding_change = 0.02 - d0, d1 - 0.03
    # F(reference) = -k * (benefit of the early move + benefit of the binding move). Both benefits
    # are positive, giving an interior reference.
    binding_stock = (a * inherited + eta * discard - deficit_fraction * b * early_change) / (
        a * (1.0 - d1 + deficit_fraction * binding_change)
    )
    parent_stock = np.ones((81, 1, 1))
    parent_stock[1, 0, 0] = binding_stock
    years = tuple(range(2018, 2101))
    count = len(years)
    cohorts = np.zeros((count, count, 1, 1))
    cohorts[1, 1, 0, 0] = 1.0
    outflow = np.zeros_like(cohorts)
    outflow[1, 1, 0, 0] = 1.0 / f - 1.0
    inflow = np.zeros((count, 1, 1))
    inflow[1, 0, 0] = 1.0 / f
    state = NormalizedCohortState(
        years=years,
        interval_years=np.ones(count),
        regions=("EUR",),
        materials=("PS",),
        goods=("Packaging",),
        elements=("C", "Other Elements"),
        stock_by_cohort=cohorts,
        outflow_by_cohort=outflow,
        annual_stock=cohorts.sum(axis=1),
        annual_inflow=inflow,
        annual_outflow=outflow.sum(axis=1),
        material_shares=np.ones((1, 1, 1)),
        carbon_content=np.array([[0.25], [0.75]]),
        lifetime_mean=np.array([1.0]),
        lifetime_std=np.array([0.2]),
    )
    mechanical = np.zeros((count, 1))
    mechanical[years.index(2021), 0] = 1.0
    goods = np.zeros((count, 2, 1, 1, 1))
    pooled = np.zeros((count, 2, 1, 1))
    inputs = RemindMFAProcessInputs(
        final_imports=goods.copy(),
        final_exports=goods.copy(),
        intermediate_imports=pooled.copy(),
        intermediate_exports=pooled.copy(),
        primary_imports=pooled.copy(),
        primary_exports=pooled.copy(),
        waste_imports=goods.copy(),
        waste_exports=goods.copy(),
        collection_rate=np.ones((count, 1)),
        mechanical_recycling_rate=mechanical,
        chemical_recycling_rate=np.zeros((count, 1)),
        incineration_rate=1.0 - mechanical,
        bio_production_rate=np.zeros((count, 1)),
        daccu_production_rate=np.zeros((count, 1)),
        waste_trade_propensity_ledger=pd.DataFrame(),
    )
    return ReceivingCase(state, inputs, recipe, parent_stock, reference, gross, eta, survival)


def _cohort(case: ReceivingCase, decision: DemandLifetimeDecision) -> ProspectiveCohortResult:
    """Keep complete 2020--2100 stock admission, including the fixed endpoint tail."""
    return continue_cohort_stock(
        case.history, _compiled(case, decision), optimization_overlay=True, record_diagnostics=False
    )


def _flows(case: ReceivingCase, decision: DemandLifetimeDecision) -> MFAFlowState:
    """Evaluate all prospective years with actual receiving and monomer constraints."""
    return calculate_mfa_flows(
        _cohort(case, decision),
        case.inputs,
        recipe=case.recipe,
        composition=load_material_composition(),
        first_year=2020,
        last_year=2100,
        record_diagnostics=False,
    )


def _margin(case: ReceivingCase, decision: DemandLifetimeDecision) -> float:
    """Independently calculate the necessary 2021 mechanical-receiving inequality."""
    f, sf1, sf2 = case.survival
    first_stock = 1.0 - float(decision.demand_fraction[1])
    surviving = sf2 / f + (first_stock - sf1 / f) * sf1 / f
    old_discard = first_stock - surviving
    second_stock = case.parent_stock[1, 0, 0] * (1.0 - float(decision.demand_fraction[2]))
    incoming = (second_stock - surviving) / f
    waste = old_discard + (1.0 - f) * incoming
    return case.gross_factor * incoming - case.recovery * waste


def _stock_proposal(case: ReceivingCase) -> DemandLifetimeDecision:
    """Call the actual public SciPy solver and retain its exact transported controls."""
    system = fixed_lifetime_stock_system(case.history, _compiled(case, case.reference))
    proposal = propose_annual_demand(
        system,
        parent_stock=case.parent_stock,
        demand_reduction_pct=50,
        lifetime_extension_pct=0,
        lifetime_fraction=case.reference.lifetime_fraction,
        target_year=2050,
    )
    assert proposal is not None
    return proposal


def _changed(reference: DemandLifetimeDecision, **levels: float) -> DemandLifetimeDecision:
    """Set the chosen annual values and retain lifetime and endpoint settings."""
    values = list(reference.demand_fraction)
    for year, value in levels.items():
        values[int(year) - 2019] = Fraction.from_float(value)
    return replace(reference, demand_fraction=tuple(values))


@pytest.fixture(scope="module")
def recipe() -> EcoinventMaterialRecipe:
    """Reuse the existing invented EcoSpold-shaped recipe with all real recipe owners."""
    return derive_material_recipe(material_source_members(), source_identity="receiving_repair_xml")


def test_coordinated_demand_move_repairs_receiving_when_scalar_moves_fail(
    recipe: EcoinventMaterialRecipe,
) -> None:
    """The exact linear-anchor pool contains a pair absent from either scalar.

    The test evaluates the declared neighbors around the exact linear anchor.
    """
    case = _receiving_case(recipe, 0.75)
    assert _flows(case, DemandLifetimeDecision(0, 0)).years[-1] == 2100
    early = _changed(case.reference, **{"2020": 0.02})
    binding = _changed(case.reference, **{"2021": 0.03})
    pair = _changed(case.reference, **{"2020": 0.02, "2021": 0.03})
    pool = demand_repair_neighbors(case.reference, failure_year=2021)
    assert early in pool and binding in pool and pair in pool
    for failed in (case.reference, early, binding):
        assert _cohort(case, failed).annual_inflow.min() >= 0.0
        assert _margin(case, failed) < -0.0001
        with pytest.raises(ScenarioFeasibilityError) as error:
            _flows(case, failed)
        assert error.value.code == "optimization_material_recovery_capacity_infeasible"
        assert dict(error.value.diagnostic.coordinates)["year"] == 2021
    physical = _flows(case, pair)
    assert _margin(case, pair) > 0.0001
    assert physical.years == tuple(range(2020, 2101))
    assert pair.lifetime_fraction == case.reference.lifetime_fraction
    np.testing.assert_allclose(
        physical.flows["processing => fabrication"][1, 0, 0] * case.gross_factor
        - physical.flows["eol => collected"][1, 0, 0] * case.recovery,
        _margin(case, pair),
        rtol=1e-12,
        atol=1e-15,
    )


def test_actual_stock_lp_still_needs_full_receiving_repair(recipe: EcoinventMaterialRecipe) -> None:
    """A genuine stock-feasible LP path fails receiving and its own finite pool repairs it."""
    case = _receiving_case(recipe, 0.25)
    assert _flows(case, DemandLifetimeDecision(0, 0)).years[-1] == 2100
    proposal = _stock_proposal(case)
    assert _cohort(case, proposal).annual_inflow.min() >= 0.0
    with pytest.raises(ScenarioFeasibilityError) as error:
        _flows(case, proposal)
    assert error.value.code == "optimization_material_recovery_capacity_infeasible"
    pool = demand_repair_neighbors(proposal, failure_year=2021)
    repaired = _changed(proposal, **{"2021": 0.03})
    assert repaired in pool
    assert _flows(case, repaired).years[-1] == 2100
    assert repaired.lifetime_fraction == proposal.lifetime_fraction
    assert repaired.demand_fraction[31:] == (Fraction(1, 2),) * 51


def test_exhausted_repair_pool_does_not_prove_endpoint_impossible(
    recipe: EcoinventMaterialRecipe,
) -> None:
    """Compare the exhausted pool with every path in a declared two-year domain.

    This exhaustive comparison covers only fifteen monotone 2020/2021 pairs;
    the remaining demand path, lifetime and 2050 endpoint stay fixed. The result
    describes the paths evaluated in the declared finite pool.
    """
    case = _receiving_case(recipe, 2.0)
    assert _flows(case, DemandLifetimeDecision(0, 0)).years[-1] == 2100
    proposal = _stock_proposal(case)
    with pytest.raises(ScenarioFeasibilityError) as error:
        _flows(case, proposal)
    assert error.value.code == "optimization_material_recovery_capacity_infeasible"
    pool = demand_repair_neighbors(proposal, failure_year=2021)
    assert pool
    assert all(_margin(case, choice) < -0.0001 for choice in pool)
    farther = _changed(proposal, **{"2020": 0.03, "2021": 0.03})
    assert farther not in pool
    assert _margin(case, farther) > 0.0001
    assert _flows(case, farther).years[-1] == 2100
    assert farther.demand_reduction_pct == proposal.demand_reduction_pct
    assert farther.lifetime_fraction == proposal.lifetime_fraction
    assert farther.demand_fraction[31:] == proposal.demand_fraction[31:]

    levels = tuple(Fraction.from_float(value / 100) for value in range(5))
    outcomes = []
    for first, second in combinations_with_replacement(levels, 2):
        choice = _changed(proposal, **{"2020": float(first), "2021": float(second)})
        assert tuple(sorted(choice.demand_fraction)) == choice.demand_fraction
        try:
            _flows(case, choice)
        except ScenarioFeasibilityError as failure:
            assert failure.code == "optimization_material_recovery_capacity_infeasible"
            assert dict(failure.diagnostic.coordinates)["year"] == 2021
            outcomes.append((choice, failure.code, None))
        else:
            # Independent definition: mean absolute demand departure from the
            # exact linear reference in percentage points over 2020--2049.
            departure = Fraction(100, 30) * sum(
                (
                    abs(value - reference)
                    for value, reference in zip(
                        choice.demand_fraction[1:31],
                        case.reference.demand_fraction[1:31],
                        strict=True,
                    )
                ),
                Fraction(),
            )
            outcomes.append((choice, "physical_feasible", departure))

    assert len(outcomes) == 15
    feasible = [
        (choice, distance) for choice, status, distance in outcomes if status == "physical_feasible"
    ]
    assert feasible
    assert any(status != "physical_feasible" for _, status, _ in outcomes)
    assert all(choice not in pool for choice, _ in feasible)
    closest_distance = min(distance for _, distance in feasible)
    closest_pairs = {
        choice.demand_fraction[1:3] for choice, distance in feasible if distance == closest_distance
    }
    # Both equally close diagonal schedules are retained; no arbitrary single
    # minimizer conceals the nonuniqueness of this finite-domain result.
    assert closest_pairs == {(levels[2], levels[2]), (levels[3], levels[3])}
