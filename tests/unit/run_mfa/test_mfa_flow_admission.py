"""Demand admission and primary-market reconciliation at the flow owner."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from constituent_mfa_fixtures import empty_model
from material_recipe_fixtures import material_source_members

from pytoce.download_data.sources.ecoinvent_materials import derive_material_recipe
from pytoce.run_mfa.model.cohorts import accepted_default_cohort_result
from pytoce.run_mfa.model.flows import _manufacturing_demand, calculate_mfa_flows
from pytoce.run_mfa.model.materials import load_material_composition
from pytoce.run_mfa.scenario.scenario_contract import ScenarioFeasibilityError


def test_historical_element_capped_imports_report_the_negative_good_coordinate() -> None:
    """Preserve the strict diagnostic for retained historical element-product rounding."""
    source, inputs = empty_model("ABS_ASA_SAN")
    year = source.years.index(2019)
    material = source.materials.index("ABS_ASA_SAN")
    good = source.goods.index("Transportation")
    amount = 0.00234375
    # The source ABS nominal C/Other fractions produce a rounded sum above
    # this invented polymer mass. History retains independent products; the
    # prospective complementary partition has its own existing owner test.
    source.carbon_content[:, material] = (0.824528584806638, 0.175471415193362)
    source.stock_by_cohort[year:, year, 0, good] = amount
    source.annual_stock[:] = source.stock_by_cohort.sum(axis=1)
    source.annual_inflow[year, 0, good] = amount
    cohort = accepted_default_cohort_result(source)
    cap = cohort.carbon_inflow[year, :, 0, material, good]
    inputs.final_imports[year, :, 0, material, good] = cap
    inputs.final_exports[year, :, 1, material, good] = cap
    np.testing.assert_array_equal(inputs.final_imports[year, :, 0, material, good], cap)
    np.testing.assert_array_equal(
        inputs.final_imports[year].sum(axis=1), inputs.final_exports[year].sum(axis=1)
    )
    assert np.isfinite(cap).all() and (cap > 0).all()

    with pytest.raises(ScenarioFeasibilityError) as error:
        calculate_mfa_flows(
            cohort,
            inputs,
            recipe=derive_material_recipe(material_source_members(), source_identity="invented"),
            composition=load_material_composition(),
            first_year=2019,
            last_year=2019,
        )

    diagnostic = error.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.equation == "fabrication => use >= 0"
    assert diagnostic.observed == -4.336808689942018e-19
    assert diagnostic.shortfall == -diagnostic.observed
    assert diagnostic.bound == diagnostic.tolerance == 0.0
    assert diagnostic.count == 1
    assert dict(diagnostic.coordinates) == {
        "year": 2019,
        "region": source.regions[0],
        "material": "ABS_ASA_SAN",
        "good": "Transportation",
    }


def test_primary_trade_ledger_reports_both_sides_of_the_closed_market() -> None:
    """Map admitted source-rounding adjustments to their actual trade arrays."""
    source, inputs = empty_model("PUR")
    year = source.years.index(2020)
    material = source.materials.index("PUR")
    source.stock_by_cohort[year:, year, :2, 0] = 3.0
    source.annual_stock[:] = source.stock_by_cohort.sum(axis=1)
    source.annual_inflow[year, :2, 0] = 3.0
    # Two active regions trade below their 3 Mt receiving demand. Element-total differences follow
    # source balancing roundoff.
    imported = np.array([[0.175, 0.35], [0.525, 1.05]])
    exported = np.nextafter(imported, 0.0)
    inputs.primary_imports[year, :, :2, material] = imported
    inputs.primary_exports[year, :, :2, material] = exported
    state = calculate_mfa_flows(
        accepted_default_cohort_result(source),
        inputs,
        recipe=derive_material_recipe(material_source_members(), source_identity="invented"),
        composition=load_material_composition(),
        first_year=2020,
        last_year=2020,
    )
    ledger = state.reconciliation_ledger
    expected = []
    for region, name in enumerate(source.regions[:2]):
        for quantity, requested, flow in (
            ("primary_imports", imported[:, region].sum(), "primary_market => processing"),
            ("primary_exports", exported[:, region].sum(), "primary_market => sysenv"),
        ):
            admitted = state.flows[flow][0, region, material]
            expected.append(
                (2020, name, "PUR", quantity, requested, admitted, admitted - requested)
            )
    columns = (
        "year",
        "region",
        "material",
        "quantity",
        "requested_amount",
        "admitted_amount",
        "delta",
    )
    pd.testing.assert_frame_equal(
        ledger.loc[:, list(columns)],
        pd.DataFrame.from_records(expected, columns=columns),
        check_exact=True,
    )
    assert ledger["unit"].eq("Mt").all()
    assert ledger["method"].eq("closed_primary_market_and_constituent_paired_reduction").all()
    assert ledger.loc[ledger["quantity"].eq("primary_imports"), "delta"].lt(0.0).all()
    np.testing.assert_array_equal(
        state.flows["primary_market => processing"], state.flows["primary_market => sysenv"]
    )


@pytest.mark.parametrize(
    ("incoming_mass", "traded_mass"),
    [
        (0.0, 0.0),
        (-0.0, -0.0),
        (float(np.nextafter(0.0, 1.0)), 0.0),
        (float(np.nextafter(0.0, 1.0)), float(np.nextafter(0.0, 1.0))),
        (float(np.finfo(np.float64).max), float(np.finfo(np.float64).max)),
        (1.0, float(np.nextafter(1.0, 0.0))),
    ],
    ids=("zero", "signed_zero", "subnormal_local", "subnormal_trade", "finite_max", "residual"),
)
def test_sparse_manufacturing_trade_preserves_exact_receiving_requirements(
    incoming_mass: float, traded_mass: float
) -> None:
    """Retain exact zero and small residuals through paired finished/intermediate trade."""
    source, inputs = empty_model("HDPE")
    first = source.years.index(2020)
    material = source.materials.index("HDPE")
    shape = (1, len(source.regions), len(source.materials), len(source.goods))
    incoming = np.zeros(shape)
    incoming[0, 0, material, 0] = incoming_mass
    # Half plus its complement is an exact nominal partition, including at
    # the smallest subnormal. Region 2 supplies intermediates to region 1,
    # which supplies finished goods to region 0; both trade markets balance.
    half = traded_mass / 2
    elements = np.array([half, traded_mass - half])
    inputs.final_imports[first, :, 0, material, 0] = elements
    inputs.final_exports[first, :, 1, material, 0] = elements
    inputs.intermediate_imports[first, :, 1, material] = elements
    inputs.intermediate_exports[first, :, 2, material] = elements
    original_incoming = incoming.copy()
    original_trades = tuple(
        array.copy()
        for array in (
            inputs.final_imports,
            inputs.final_exports,
            inputs.intermediate_imports,
            inputs.intermediate_exports,
        )
    )

    fabrication, requirements = _manufacturing_demand(incoming, inputs, first)

    local = Fraction(incoming_mass) - Fraction(traded_mass)
    expected_fabrication = np.zeros(shape[:3])
    expected_fabrication[0, 0, material] = float(local)
    assert fabrication.tobytes() == expected_fabrication.tobytes()
    expected_requirements = [[Fraction() for _ in source.materials] for _ in source.regions]
    expected_requirements[0][material] = local
    expected_requirements[2][material] = Fraction(traded_mass)
    assert requirements == (tuple(tuple(row) for row in expected_requirements),)
    assert incoming.tobytes() == original_incoming.tobytes()
    for observed, original in zip(
        (
            inputs.final_imports,
            inputs.final_exports,
            inputs.intermediate_imports,
            inputs.intermediate_exports,
        ),
        original_trades,
        strict=True,
    ):
        assert observed.tobytes() == original.tobytes()
