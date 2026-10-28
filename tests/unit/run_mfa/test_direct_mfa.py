"""Direct polymer movements and accumulation from coherent cohort inputs."""

from dataclasses import replace
from math import fsum
from types import ModuleType

import flodym as fd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from material_recipe_fixtures import material_source_members
from scenario_fixtures import compile_scenario_payload

from pytoce.download_data.sources.ecoinvent_materials import derive_material_recipe
from pytoce.run_mfa.model.cohorts import (
    ProspectiveCohortResult,
    accepted_default_cohort_result,
    continue_cohort_stock,
)
from pytoce.run_mfa.model.flows import calculate_mfa_flows
from pytoce.run_mfa.model.materials import load_material_composition
from pytoce.run_mfa.model.mfa import _require_validity
from pytoce.run_mfa.model.result_tables import _derived_outputs, mfa_flow_table
from pytoce.run_mfa.model.stocks import accumulate_mfa_stocks, mfa_stock_table
from pytoce.run_mfa.remind_mfa.execution import (
    RemindMFAExecutionRequest,
    RemindMFAExecutionSession,
)
from pytoce.run_mfa.remind_mfa.process_inputs import (
    RemindMFAProcessInputs,
    _resolved_waste_trade,
)
from pytoce.run_mfa.remind_mfa.translation import NormalizedCohortState
from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue
from pytoce.run_mfa.scenario.scenario_contract import ScenarioFeasibilityError


def _cohort_and_process_inputs() -> tuple[ProspectiveCohortResult, RemindMFAProcessInputs]:
    """Provide balanced three-year cohorts and independently specified controls."""
    stock = np.array([[4.0, 0.0, 0.0], [3.0, 1.5, 0.0], [2.0, 1.0, 1.5]])[:, :, None, None]
    discard = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.5, 0.5]])[:, :, None, None]
    cohort = accepted_default_cohort_result(
        NormalizedCohortState(
            years=(1950, 1951, 1952),
            interval_years=np.ones(3),
            regions=("EUR",),
            goods=("Transportation",),
            materials=("HDPE", "BioPlastics"),
            elements=("C", "Other Elements"),
            stock_by_cohort=stock,
            outflow_by_cohort=discard,
            annual_stock=stock.sum(axis=1),
            annual_inflow=np.array([4.0, 2.0, 2.0])[:, None, None],
            annual_outflow=discard.sum(axis=1),
            material_shares=np.array([[[0.75], [0.25]]]),
            carbon_content=np.array([[0.8, 0.6], [0.2, 0.4]]),
            lifetime_mean=np.array([10.0]),
            lifetime_std=np.array([2.0]),
        )
    )
    fine = np.zeros((3, 2, 1, 2, 1))
    pooled = np.zeros((3, 2, 1, 2))
    rate = np.ones((3, 1))
    inputs = RemindMFAProcessInputs(
        final_imports=fine,
        final_exports=fine,
        intermediate_imports=pooled,
        intermediate_exports=pooled,
        primary_imports=pooled,
        primary_exports=pooled,
        waste_imports=fine,
        waste_exports=fine,
        collection_rate=rate * 0.6,
        mechanical_recycling_rate=rate * 0.2,
        chemical_recycling_rate=rate * 0.1,
        incineration_rate=rate * 0.3,
        bio_production_rate=rate * 0.1,
        daccu_production_rate=rate * 0.1,
        waste_trade_propensity_ledger=pd.DataFrame(),
    )
    return cohort, inputs


def test_complete_physical_accounts_preserve_scale_and_process_balance() -> None:
    """Keep all family/use coordinates and balances when a valid mass system is enlarged."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    results = []
    for scale in (1.0, 100000.0):
        source, inputs = empty_model()
        stock = np.array([[4, 0, 0, 0], [3, 1.5, 0, 0], [2, 1, 1.5, 0], [1, 0.5, 1, 1.5]])
        discard = np.array([[0, 0, 0, 0], [1, 0.5, 0, 0], [1, 0.5, 0.5, 0], [1, 0.5, 0.5, 0.5]])
        source.stock_by_cohort[:] = stock[:, :, None, None] * scale
        source.outflow_by_cohort[:] = discard[:, :, None, None] * scale
        source.annual_inflow[:] = np.array([4, 2, 2, 2])[:, None, None] * scale
        source.material_shares[:] = 1 / len(source.materials)
        inputs = replace(
            inputs,
            collection_rate=np.full_like(inputs.collection_rate, 0.6),
            mechanical_recycling_rate=np.full_like(inputs.mechanical_recycling_rate, 0.2),
            chemical_recycling_rate=np.full_like(inputs.chemical_recycling_rate, 0.05),
            incineration_rate=np.full_like(inputs.incineration_rate, 0.3),
        )
        result = evaluate_mfa(source, inputs, recipe)
        flows = result.model_flows
        amounts = {
            name: frame.groupby(["year", "region", "material"], sort=True)["amount"].sum()
            for name, frame in flows.groupby("source_flow_name", sort=False)
        }
        # Check balances across independently grouped reports, allowing rounding from constituent
        # and nominal-element sums.
        for process in ("fabrication", "processing", "eol", "collected", "reclmech", "reclchem"):
            incoming = sum(
                value for name, value in amounts.items() if name.endswith(" => " + process)
            )
            outgoing = sum(
                value for name, value in amounts.items() if name.startswith(process + " => ")
            )
            pd.testing.assert_series_equal(
                incoming, outgoing, check_exact=False, rtol=1e-14, atol=0
            )
        results.append(result)

    for name in (
        "model_flows",
        "material_realization",
        "nominal_feedstock_equivalents",
        "cohort_state",
    ):
        scaled = getattr(results[1], name).copy()
        scaled["amount"] /= 100000.0
        pd.testing.assert_frame_equal(
            getattr(results[0], name), scaled, check_exact=False, rtol=1e-14, atol=0
        )
    pd.testing.assert_frame_equal(
        results[0].cohort_material_identity, results[1].cohort_material_identity, check_exact=True
    )


def test_nonfinite_pooled_mass_fails_before_material_allocation() -> None:
    """Reject an overflowing good sum even though every incoming cohort cell is finite."""
    source, inputs = empty_model()
    source.stock_by_cohort[0, 0] = 1e308
    source.outflow_by_cohort[1, 0] = 1e308
    source.annual_inflow[0] = 1e308
    source = replace(
        source,
        annual_stock=source.stock_by_cohort.sum(axis=1),
        annual_outflow=source.outflow_by_cohort.sum(axis=1),
    )
    assert np.isfinite(source.stock_by_cohort).all()
    assert np.isfinite(source.outflow_by_cohort).all()
    cohort = accepted_default_cohort_result(source)
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    with (
        pytest.warns(RuntimeWarning),
        pytest.raises(ValueError, match="Derived eol => collected contains a nonfinite mass"),
    ):
        calculate_mfa_flows(
            cohort,
            inputs,
            recipe=recipe,
            composition=load_material_composition(),
            first_year=2019,
            last_year=2021,
        )


def test_derived_reports_preserve_physical_quantities_and_zero_treatment() -> None:
    """Report resin output, new production and undefined zero-treatment fractions."""
    cohort, inputs = _cohort_and_process_inputs()
    cohort = replace(cohort, years=(2019, 2020, 2021))
    state = calculate_mfa_flows(
        cohort,
        inputs,
        recipe=derive_material_recipe(material_source_members(), source_identity="invented_xml"),
        composition=load_material_composition(),
        first_year=2019,
        last_year=2021,
    )
    output = _derived_outputs(
        cohort, mfa_flow_table(state, cohort, foreground_scenario="report_fixture")
    )
    masses = output.loc[output["parameter_id"].ne("landfill_share")]
    totals = masses.groupby(["year", "parameter_id"], sort=True)["amount"].sum().unstack()
    expected = pd.DataFrame(
        {
            "plastics_in_use_stock": [4.0, 4.5, 4.5],
            "stock_inflow": [4.0, 2.0, 2.0],
            "stock_outflow": [0.0, 1.5, 2.0],
            "discard": [0.0, 1.5, 2.0],
            "collected_mass": [0.0, 0.9, 1.2],
            "mechanically_recycled_resin": state.flows["reclmech => processing"].sum(axis=(1, 2)),
            "new_polymer_production": (
                state.flows["new_polymer => processing"]
                + state.flows["new_polymer => primary_market"]
            ).sum(axis=(1, 2)),
        },
        index=pd.Index(np.asarray([2019, 2020, 2021], dtype=int), name="year"),
    ).rename_axis(columns="parameter_id")
    pd.testing.assert_frame_equal(
        totals, expected.sort_index(axis=1), check_exact=False, rtol=1e-14, atol=0
    )
    assert masses["unit"].eq("Mt").all()
    assert masses["region"].eq("EUR").all()
    assert masses["derivation_status"].eq("computed").all()
    landfill = output.loc[output["parameter_id"].eq("landfill_share")].reset_index(drop=True)
    assert landfill["year"].tolist() == [2019, 2020, 2021]
    assert landfill["region"].eq("EUR").all()
    assert landfill[["material", "good", "process"]].isna().all().all()
    assert landfill["route"].eq("landfill").all()
    assert landfill["unit"].eq("fraction").all()
    np.testing.assert_allclose(landfill["amount"], [np.nan, 0.4, 0.4], rtol=1e-14, atol=0)
    assert landfill["derivation_status"].tolist() == [
        "undefined_zero_total_treatment",
        "computed_residual_share",
        "computed_residual_share",
    ]


def test_direct_polymer_flows_and_stock_accumulation_keep_distinct_accounts() -> None:
    """Close material balances with chemical feed and recovered resin accounted for separately."""
    cohort, inputs = _cohort_and_process_inputs()
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    composition = load_material_composition()
    state = calculate_mfa_flows(
        cohort, inputs, recipe=recipe, composition=composition, first_year=1950, last_year=1952
    )
    flows = state.flows
    np.testing.assert_array_equal(flows["use => eol"], cohort.material_outflow)
    np.testing.assert_array_equal(flows["fabrication => use"], cohort.material_inflow)
    assert "reclchem => virgin" not in flows
    np.testing.assert_allclose(
        flows["reclchem => chemical_feedstock"] + flows["reclchem => chemical_preparation_reject"],
        flows["collected => reclchem"],
        rtol=1e-14,
        atol=1e-15,
    )
    for local in state.material_states:
        year = state.years.index(local.year)
        for quantity, name in (
            ("mechanical_resin", "reclmech => processing"),
            ("new_polymer_domestic", "new_polymer => processing"),
        ):
            for family, material in enumerate(state.materials):
                expected = sum(
                    local.amounts[quantity][i]
                    for i, item in enumerate(local.constituents)
                    if item.material == material
                )
                assert flows[name][year, 0, family] == pytest.approx(expected, abs=1e-15)
    opening = {name: np.zeros((1, 2)) for name in ("landfill", "uncontrolled", "atmospheric")}
    stocks = accumulate_mfa_stocks(cohort, state, opening=opening)
    assert stocks.validation["status"].eq("ok").all()
    # The scenario's regional route fractions are feasible; selective chemistry
    # changes constituent allocation but preserves each regional route total.
    regional_discard = np.array([0.0, 1.5, 2.0])
    np.testing.assert_allclose(
        stocks.quantities["landfill"]["stock"].sum(axis=(1, 2)),
        np.cumsum(regional_discard * 0.6 * 0.4 + flows["reclmech => landfill"].sum(axis=(1, 2))),
    )
    np.testing.assert_allclose(
        stocks.quantities["uncontrolled"]["stock"].sum(axis=(1, 2)),
        np.cumsum(regional_discard * 0.4),
    )
    table = mfa_stock_table(stocks, state, cohort, foreground_scenario="direct_fixture")
    atmosphere = table.loc[table["source_stock_name"].eq("atmospheric")]
    assert set(atmosphere["element"]) == {"C"}
    assert set(atmosphere["unit"]) == {"Mt nominal C"}
    assert atmosphere.loc[atmosphere["stock_quantity"].eq("stock"), "amount"].min() < 0
    non_atmosphere = table.loc[~table["source_stock_name"].eq("atmospheric")]
    use = non_atmosphere.loc[
        non_atmosphere["source_stock_name"].eq("in_use")
        & non_atmosphere["stock_quantity"].eq("stock")
    ]
    np.testing.assert_allclose(use.groupby("year")["amount"].sum(), [4.0, 4.5, 4.5])
    for quantity, expected in (
        ("stock", cohort.material_stock),
        ("inflow", cohort.material_inflow),
        ("outflow", cohort.material_outflow),
    ):
        np.testing.assert_array_equal(stocks.quantities["in_use"][quantity], expected)
    carbon = cohort.carbon_content[cohort.elements.index("C")]
    emitted = (flows["collected => incineration"] + flows["reclmech => incineration"]) * carbon
    withdrawn = (
        state.nominal_feedstock_equivalents["biogenic"]
        + state.nominal_feedstock_equivalents["daccu"]
    ) * carbon
    np.testing.assert_array_equal(stocks.quantities["atmospheric"]["inflow"], emitted)
    np.testing.assert_array_equal(stocks.quantities["atmospheric"]["outflow"], withdrawn)
    np.testing.assert_array_equal(
        stocks.quantities["atmospheric"]["stock"], np.cumsum(emitted - withdrawn, axis=0)
    )
    # Carry nonzero disposal and atmospheric stocks across the period boundary.
    prefix = calculate_mfa_flows(
        cohort, inputs, recipe=recipe, composition=composition, first_year=1950, last_year=1951
    )
    prefix_stocks = accumulate_mfa_stocks(cohort, prefix, opening=opening)
    suffix = calculate_mfa_flows(
        cohort, inputs, recipe=recipe, composition=composition, first_year=1952, last_year=1952
    )
    suffix_stocks = accumulate_mfa_stocks(cohort, suffix, opening=prefix_stocks.closing)
    for name in opening:
        np.testing.assert_allclose(
            suffix_stocks.quantities[name]["stock"],
            stocks.quantities[name]["stock"][2:],
            rtol=0.0,
            atol=1e-15,
        )


def test_finite_lifetime_driven_flows_reject_overflowing_accumulated_stock() -> None:
    """Reject unrepresentable disposal stocks and preserve admitted annual masses."""
    families = next(
        item.coordinates["material"]
        for item in load_parameter_catalogue()
        if item.parameter_id == "plastic_family_share"
    )
    changes = [
        {
            "parameter": "stock_projection",
            "region": "EUR",
            "good": "Packaging",
            "changes": [
                {"year": 2020, "mode": "absolute", "value": 0.0},
                {"year": 2095, "mode": "absolute", "value": 4e307},
            ],
        },
        {
            "parameter": "lifetime_mean",
            "good": "Packaging",
            "changes": [{"year": 2020, "mode": "absolute", "value": 1.0}],
        },
    ]
    changes.extend(
        {
            "parameter": "plastic_family_share",
            "region": "EUR",
            "material": material,
            "good": "Packaging",
            "changes": [{"year": 2020, "mode": "absolute", "value": float(material == "HDPE")}],
        }
        for material in families
    )
    compiled = compile_scenario_payload(
        {
            "schema_version": "pytoce_foreground_scenario_schema_v1",
            "foreground_scenario": "finite_stock_boundary",
            "ssp_scenario": "SSP2",
            "changes": changes,
        }
    ).compiled_parameters
    years = tuple(range(1950, 2101))
    count = len(years)
    source = NormalizedCohortState(
        years=years,
        interval_years=np.ones(count),
        regions=("EUR",),
        goods=("Packaging",),
        materials=("HDPE",),
        elements=("C", "Other Elements"),
        stock_by_cohort=np.zeros((count, count, 1, 1)),
        outflow_by_cohort=np.zeros((count, count, 1, 1)),
        annual_stock=np.zeros((count, 1, 1)),
        annual_inflow=np.zeros((count, 1, 1)),
        annual_outflow=np.zeros((count, 1, 1)),
        material_shares=np.ones((1, 1, 1)),
        carbon_content=np.array([[0.8], [0.2]]),
        lifetime_mean=np.array([0.5]),
        lifetime_std=np.array([0.1]),
    )
    # The compiler preserves Packaging's 20% coefficient of variation. The extreme,
    # finite Mt target exposes cumulative overflow only in the final year;
    # zero historical stock is consistent with the same lifetime law.
    cohort = continue_cohort_stock(source, compiled, optimization_overlay=False)
    fine = np.zeros((count, 2, 1, 1, 1))
    pooled = np.zeros((count, 2, 1, 1))
    rate = np.zeros((count, 1))
    inputs = RemindMFAProcessInputs(
        final_imports=fine,
        final_exports=fine,
        intermediate_imports=pooled,
        intermediate_exports=pooled,
        primary_imports=pooled,
        primary_exports=pooled,
        waste_imports=fine,
        waste_exports=fine,
        collection_rate=rate,
        mechanical_recycling_rate=rate,
        chemical_recycling_rate=rate,
        incineration_rate=np.ones_like(rate),
        bio_production_rate=rate,
        daccu_production_rate=rate,
        waste_trade_propensity_ledger=pd.DataFrame(),
    )
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    for final_year in (2099, 2100):
        flows = calculate_mfa_flows(
            cohort,
            inputs,
            recipe=recipe,
            composition=load_material_composition(),
            first_year=2095,
            last_year=final_year,
        )
        assert all(np.isfinite(values).all() for values in flows.flows.values())
        with np.errstate(over="ignore"):
            stocks = accumulate_mfa_stocks(
                cohort,
                flows,
                opening={
                    name: np.zeros((1, 1)) for name in ("landfill", "uncontrolled", "atmospheric")
                },
            )
        if final_year == 2099:
            _require_validity(stocks.validation)
            assert np.isfinite(stocks.validation["tolerance_mt"]).all()
        else:
            assert np.isinf(stocks.closing["uncontrolled"]).all()
            with pytest.raises(ValueError, match="uncontrolled_annual_stock_balance"):
                _require_validity(stocks.validation)


@pytest.mark.parametrize(("bio", "dac"), ((0.4, 0.3), (0.8, 0.2)))
def test_virgin_feed_controls_only_change_the_nominal_account_at_this_boundary(
    bio: float,
    dac: float,
) -> None:
    """Keep polymer realization independent of virgin-feed origin selection."""
    cohort, inputs = _cohort_and_process_inputs()
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    arguments = dict(
        recipe=recipe, composition=load_material_composition(), first_year=1950, last_year=1952
    )
    initial = calculate_mfa_flows(cohort, inputs, **arguments)
    changed = calculate_mfa_flows(
        cohort,
        replace(
            inputs,
            bio_production_rate=np.full((3, 1), bio),
            daccu_production_rate=np.full((3, 1), dac),
        ),
        **arguments,
    )
    for name in initial.flows:
        np.testing.assert_array_equal(initial.flows[name], changed.flows[name])
    for old, new in zip(initial.material_states, changed.material_states, strict=True):
        for quantity in old.amounts:
            np.testing.assert_array_equal(old.amounts[quantity], new.amounts[quantity])
    np.testing.assert_allclose(
        changed.nominal_feedstock_equivalents["biogenic"],
        initial.nominal_feedstock_equivalents["biogenic"] * (bio / 0.1),
    )
    assert np.all(changed.nominal_feedstock_equivalents["fossil"] >= 0.0)
    if bio + dac == 1.0:
        np.testing.assert_array_equal(changed.nominal_feedstock_equivalents["fossil"], 0.0)
    # At zero first-year recycling, polymer equivalent equals new polymer. Monomer requirements
    # retain their manufacturing bases.
    total_equivalent = sum(values[0] for values in initial.nominal_feedstock_equivalents.values())
    np.testing.assert_allclose(
        total_equivalent,
        cohort.material_inflow[0].sum(axis=-1)
        + initial.flows["processing => manufacturing_loss"][0],
    )

    # Mixed chemical recovery uses gross principal-monomer carbon as its denominator.
    state = initial.material_states[1]
    carbon = {item.product: item.carbon_kg_per_kg for item in recipe.cracker_products}
    equivalents = []
    fractions = []
    for index in range(len(state.family_index)):
        virgin = fsum(
            carbon[name] * state.virgin_monomers[name][index] for name in ("ethylene", "propylene")
        )
        recovered = fsum(
            carbon[name] * state.recovered_monomers[name][index]
            for name in ("ethylene", "propylene")
        )
        fraction = virgin / (virgin + recovered) if virgin + recovered else 1.0
        fractions.append(fraction)
        polymer = (
            state.amounts["new_polymer_domestic"][index]
            + state.amounts["new_polymer_export"][index]
        )
        equivalents.append(polymer * fraction)
    assert any(0 < fraction < 1 for fraction in fractions)
    expected = [
        fsum(
            value for index, value in enumerate(equivalents) if state.family_index[index] == family
        )
        for family in range(len(cohort.materials))
    ]
    observed = sum(values[1, 0] for values in initial.nominal_feedstock_equivalents.values())
    np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize(
    ("rates", "expected", "collection", "projected"),
    (
        ((0.8, 0.5, 0.3), (0.6, 0.3, 0.1, 0.0), 0.6, True),
        ((0.2, 0.1, 0.3), (0.2, 0.1, 0.3, 0.4), 0.6, False),
        ((0.2, 0.1, 5e-13), (0.2, 0.1, 5e-13, 1 - (0.2 + 0.1 + 5e-13)), 0.6, False),
        ((0.8, 0.5, 0.3), (0.6, 0.3, 0.1, 0.0), 5e-13, True),
    ),
)
def test_direct_routes_use_the_existing_least_change_projection(
    rates: tuple[float, float, float],
    expected: tuple[float, float, float, float],
    collection: float,
    projected: bool,
) -> None:
    """Project an infeasible source partition using the specified distance criterion."""
    cohort, inputs = _cohort_and_process_inputs()
    inputs = replace(
        inputs,
        mechanical_recycling_rate=np.full((3, 1), rates[0]),
        chemical_recycling_rate=np.full((3, 1), rates[1]),
        incineration_rate=np.full((3, 1), rates[2]),
        collection_rate=np.full((3, 1), collection),
    )
    state = calculate_mfa_flows(
        cohort,
        inputs,
        recipe=derive_material_recipe(material_source_members(), source_identity="invented_xml"),
        composition=load_material_composition(),
        first_year=1950,
        last_year=1952,
    )
    for route, share in zip(
        ("reclmech", "reclchem", "incineration", "landfill"), expected, strict=True
    ):
        np.testing.assert_allclose(
            state.flows["collected => " + route].sum(axis=(1, 2)),
            np.array([0.0, 1.5, 2.0]) * collection * share,
            rtol=1e-13,
            atol=0.0,
        )
        if share > 0:
            assert np.all(state.flows["collected => " + route][1:] > 0)
    assert state.reconciliation_ledger.empty is (not projected)
    assert state.reconciliation_ledger["method"].eq("constrained_least_change_projection").all()


def test_small_disposal_inflow_after_large_historical_accumulation() -> None:
    """Admit recurrence roundoff and retain small annual flows."""
    original, inputs = _cohort_and_process_inputs()
    stock = np.zeros((3, 3, 1, 1))
    stock[:, 0, 0, 0] = (200000.0, 100000.0, 99999.9)
    outflow = np.zeros_like(stock)
    outflow[:, 0, 0, 0] = (0.0, 100000.0, 0.1)
    cohort = accepted_default_cohort_result(
        NormalizedCohortState(
            years=original.years,
            interval_years=np.ones(3),
            regions=original.regions,
            goods=original.goods,
            materials=original.materials,
            elements=original.elements,
            stock_by_cohort=stock,
            outflow_by_cohort=outflow,
            annual_stock=stock.sum(axis=1),
            annual_inflow=np.array([200000.0, 0.0, 0.0])[:, None, None],
            annual_outflow=outflow.sum(axis=1),
            material_shares=np.array([[[0.75], [0.25]]]),
            carbon_content=original.carbon_content,
            lifetime_mean=np.array([10.0]),
            lifetime_std=np.array([2.0]),
        )
    )
    inputs = replace(
        inputs,
        collection_rate=np.ones((3, 1)),
        mechanical_recycling_rate=np.zeros((3, 1)),
        chemical_recycling_rate=np.zeros((3, 1)),
        incineration_rate=np.zeros((3, 1)),
    )
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    composition = load_material_composition()
    prefix = calculate_mfa_flows(
        cohort, inputs, recipe=recipe, composition=composition, first_year=1950, last_year=1951
    )
    opening = {name: np.zeros((1, 2)) for name in ("landfill", "uncontrolled", "atmospheric")}
    historical = accumulate_mfa_stocks(cohort, prefix, opening=opening)
    assert historical.closing["landfill"].sum() == 100000.0
    future = calculate_mfa_flows(
        cohort, inputs, recipe=recipe, composition=composition, first_year=1952, last_year=1952
    )
    stocks = accumulate_mfa_stocks(cohort, future, opening=historical.closing)
    np.testing.assert_array_equal(
        stocks.quantities["landfill"]["inflow"], future.flows["collected => landfill"]
    )
    assert stocks.quantities["landfill"]["inflow"].sum() == pytest.approx(0.1)
    assert stocks.validation["status"].eq("ok").all()


def test_source_balanced_waste_exports_cannot_exceed_regional_collected_waste(
    remind_source_request: RemindMFAExecutionRequest,
) -> None:
    """Reject a real regional waste deficit after native maximum trade balancing."""
    session = RemindMFAExecutionSession(remind_source_request)
    cohort = accepted_default_cohort_result(session.cohort_state)
    inputs = session.process_inputs(
        {
            "waste_imports": pd.DataFrame(
                {"year": [2020, 2020], "region": ["EUR", "USA"], "amount": [0.0, 1000000.0]}
            ),
            "waste_exports": pd.DataFrame(
                {"year": [2020, 2020], "region": ["EUR", "USA"], "amount": [1.0, 0.0]}
            ),
        },
        carbon_inflow=cohort.carbon_inflow,
        carbon_outflow=cohort.carbon_outflow,
        deferred_waste_trade=pd.DataFrame(),
    )
    year = cohort.years.index(2020)
    exported = inputs.waste_exports[year].sum(axis=(0, 2, 3))
    np.testing.assert_allclose(exported, [1000000.0, 0.0], rtol=1e-14, atol=0.0)
    local = cohort.material_outflow[year, 0].sum(axis=-1) * inputs.collection_rate[year, 0]
    available = (
        local
        + inputs.waste_imports[year, :, 0].sum(axis=(0, 2))
        - inputs.waste_exports[year, :, 0].sum(axis=(0, 2))
    )
    assert np.all(available < 0.0)
    with pytest.raises(ScenarioFeasibilityError) as error:
        calculate_mfa_flows(
            cohort,
            inputs,
            recipe=derive_material_recipe(
                material_source_members(), source_identity="invented_xml"
            ),
            composition=load_material_composition(),
            first_year=2020,
            last_year=2020,
        )
    failure = error.value.diagnostic
    assert failure is not None
    assert failure.equation == "available_collected_waste >= 0"
    assert failure.observed == available[0]
    assert failure.shortfall == -available[0]
    assert failure.bound == failure.tolerance == 0.0
    assert dict(failure.coordinates) == {"year": 2020, "region": "EUR", "material": "HDPE"}


@pytest.mark.parametrize("imported", (1.0, 1e-20))
def test_source_maximum_balance_rejects_unmatched_waste_without_a_diagnostic_ledger(
    remind_source_request: RemindMFAExecutionRequest,
    imported: float,
) -> None:
    """Reject zero-sided native trade, including below the source epsilon divisor."""
    session = RemindMFAExecutionSession(remind_source_request)
    cohort = accepted_default_cohort_result(session.cohort_state)
    updates = {
        "waste_imports": pd.DataFrame(
            {"year": [2020, 2020], "region": ["EUR", "USA"], "amount": [imported, 0.0]}
        ),
        "waste_exports": pd.DataFrame(
            {"year": [2020, 2020], "region": ["EUR", "USA"], "amount": [0.0, 0.0]}
        ),
    }
    failures = []
    for record_diagnostics in (True, False):
        with pytest.raises(ScenarioFeasibilityError) as error:
            session.process_inputs(
                updates,
                carbon_inflow=cohort.carbon_inflow,
                carbon_outflow=cohort.carbon_outflow,
                deferred_waste_trade=pd.DataFrame(),
                record_diagnostics=record_diagnostics,
            )
        diagnostic = error.value.diagnostic
        assert diagnostic is not None
        assert error.value.code == "optimization_source_trade_balance_infeasible"
        assert diagnostic.code == "optimization_source_trade_balance_infeasible"
        assert diagnostic.equation == "global_imports - global_exports = 0"
        assert dict(diagnostic.coordinates)["year"] == 2020
        assert dict(diagnostic.details)["trade_kind"] == "waste"
        assert diagnostic.observed is not None
        assert diagnostic.observed > diagnostic.tolerance
        failures.append(diagnostic)
    assert failures[0] == failures[1]


def test_native_waste_balance_preserves_maximum_and_regional_shares_at_small_scales(
    remind_trade_modules: tuple[ModuleType, ModuleType],
) -> None:
    """Retain two-sided trade below epsilon using the specified equation."""
    dims = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name=name, letter=letter, items=items)
            for name, letter, items in (
                ("Time", "t", [1950, 1951, 2020, 2021]),
                ("Element", "e", ["C"]),
                ("Region", "r", ["EUR", "USA"]),
                ("Material", "m", ["HDPE", "PP"]),
                ("Good", "g", ["Packaging"]),
            )
        ]
    )
    amounts = np.zeros((4, 1, 2, 2, 1))
    amounts[:, 0, :, :, 0] = [[3.0, 1.0], [1.0, 3.0]]
    # Each region has opposite 3/4 and 1/4 material shares. The four years
    # exercise both totals below epsilon, only one below, ordinary Mt and zero.
    units_mt = np.array([2.0**-60, 2.0**-53, 1.0, 0.0])
    parameters = {
        name: fd.Parameter(
            dims=dims[("t", "r")],
            name=name,
            values=units_mt[:, None] * np.array(regions)[None, :],
        )
        for name, regions in (
            ("waste_imports", [2.0, 1.0]),
            ("waste_exports", [1.0, 3.0]),
        )
    }
    trade, _ = _resolved_waste_trade(
        fd.FlodymArray(dims=dims, values=amounts, name="outflow"),
        parameters,
        dims,
        remind_trade_modules[0],
        pd.DataFrame(),
        pd.DataFrame(),
    )
    # Before balancing, HDPE imports/exports total 7/4 and 3/2; PP totals are 5/4 and 5/2.
    # Source maximum balancing raises the smaller total, retaining its two
    # regional shares. These are independent expected quantities in each unit.
    for values, expected in (
        (trade.imports, [[1.5, 1.0], [0.25, 1.5]]),
        (trade.exports, [[0.875, 0.25], [0.875, 2.25]]),
    ):
        actual = values.cast_values_to(dims)[:, 0, :, :, 0]
        np.testing.assert_allclose(
            actual, units_mt[:, None, None] * np.array(expected), rtol=2e-15, atol=0.0
        )
        np.testing.assert_array_equal(actual[-1], np.zeros((2, 2)))
        assert np.all(actual[:-1] > 0)


def test_native_balanced_trade_retains_the_same_small_physical_treatment_total(
    remind_trade_modules: tuple[ModuleType, ModuleType],
) -> None:
    """Close an almost exhausted regional waste account after real source trade admission."""
    cohort = accepted_default_cohort_result(
        NormalizedCohortState(
            years=(2020,),
            interval_years=np.ones(1),
            regions=("EUR", "USA"),
            goods=("Packaging",),
            materials=("HDPE",),
            elements=("C", "Other Elements"),
            stock_by_cohort=np.zeros((1, 1, 2, 1)),
            outflow_by_cohort=np.full((1, 1, 2, 1), 0.25),
            annual_stock=np.zeros((1, 2, 1)),
            annual_inflow=np.full((1, 2, 1), 0.25),
            annual_outflow=np.full((1, 2, 1), 0.25),
            material_shares=np.ones((2, 1, 1)),
            carbon_content=np.array([[0.73], [0.27]]),
            lifetime_mean=np.ones(1),
            lifetime_std=np.ones(1) * 0.2,
        )
    )
    dims = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name=name, letter=letter, items=list(values))
            for name, letter, values in (
                ("Time", "t", cohort.years),
                ("Element", "e", cohort.elements),
                ("Region", "r", cohort.regions),
                ("Material", "m", cohort.materials),
                ("Good", "g", cohort.goods),
            )
        ]
    )
    export = np.nextafter(0.25, 0)
    parameters = {
        name: fd.Parameter(dims=dims[("t", "r")], name=name, values=np.array([values]))
        for name, values in (
            ("collection_rate", [1.0, 1.0]),
            ("waste_imports", [0.0, export]),
            ("waste_exports", [export, 0.0]),
        )
    }
    trade, _ = _resolved_waste_trade(
        fd.FlodymArray(dims=dims, values=cohort.carbon_outflow, name="outflow"),
        parameters,
        dims,
        remind_trade_modules[0],
        pd.DataFrame(),
        pd.DataFrame(),
    )
    fine = np.zeros_like(cohort.carbon_outflow)
    pooled = fine.sum(axis=-1)
    zero = np.zeros((1, 2))
    inputs = RemindMFAProcessInputs(
        final_imports=fine,
        final_exports=fine,
        intermediate_imports=pooled,
        intermediate_exports=pooled,
        primary_imports=pooled,
        primary_exports=pooled,
        waste_imports=trade.imports.cast_values_to(dims),
        waste_exports=trade.exports.cast_values_to(dims),
        collection_rate=zero + 1,
        mechanical_recycling_rate=zero,
        chemical_recycling_rate=zero,
        incineration_rate=zero + 1,
        bio_production_rate=zero,
        daccu_production_rate=zero,
        waste_trade_propensity_ledger=pd.DataFrame(),
    )
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    composition = load_material_composition()
    outputs = []
    for diagnostics in (True, False):
        state = calculate_mfa_flows(
            cohort,
            inputs,
            recipe=recipe,
            composition=composition,
            first_year=2020,
            last_year=2020,
            record_diagnostics=diagnostics,
        )
        region = state.material_states[0]
        assert region.amounts["collected_after_trade"].sum() == 5.551115123125783e-17
        np.testing.assert_array_equal(
            region.amounts["incineration_treatment"], region.amounts["collected_after_trade"]
        )
        outputs.append(state)
    for name in outputs[0].flows:
        np.testing.assert_array_equal(outputs[0].flows[name], outputs[1].flows[name])


@pytest.mark.parametrize(
    ("material", "fractions", "amount"),
    (
        ("ABS_ASA_SAN", (0.824528584806638, 0.175471415193362), 0.00234375),
        ("PVC", (0.38, 0.62), 0.01),
    ),
)
def test_full_final_imports_preserve_the_physical_polymer_capacity(
    material: str, fractions: tuple[float, float], amount: float
) -> None:
    """Realize an imported final good with a closed element balance."""
    source, inputs = empty_model(material)
    polymer = source.materials.index(material)
    year = source.years.index(2020)
    # Invented Mt demands with the source's nominal ABS/PVC fractions expose
    # independent-product rounding. Imports cover the entire receiving demand;
    # a second region supplies exactly that quantity as finished-goods exports.
    source.carbon_content[:, polymer] = fractions
    source.stock_by_cohort[year:, year, 0, 0] = amount
    source.annual_stock[:] = source.stock_by_cohort.sum(axis=1)
    source.annual_inflow[year, 0, 0] = amount
    cohort = accepted_default_cohort_result(source)
    inputs.final_imports[year, :, 0, polymer, 0] = cohort.carbon_inflow[year, :, 0, polymer, 0]
    inputs.final_exports[year, :, 1, polymer, 0] = inputs.final_imports[year, :, 0, polymer, 0]
    state = calculate_mfa_flows(
        cohort,
        inputs,
        recipe=derive_material_recipe(material_source_members(), source_identity="invented_xml"),
        composition=load_material_composition(),
        first_year=2020,
        last_year=2021,
    )
    assert state.flows["fabrication => use"][0, 0, polymer, 0] == 0.0
    assert state.flows["good_market => use"][0, 0, polymer, 0] == amount
    assert state.flows["fabrication => good_market"][0, 1, polymer, 0] == amount
    assert state.flows["processing => fabrication"][0, 1, polymer] == amount


def test_full_import_supply_preserves_original_good_and_element_sums() -> None:
    """Equal underlying good demand and resin imports cannot become a false deficit."""
    # PUR uses the finished-product basis, so equal imports cover both demands. PS requires resin
    # for foam, injection, and film-forming losses.
    source, inputs = empty_model("PUR")
    polymer = source.materials.index("PUR")
    source.stock_by_cohort[:, 0, 0, :2] = (0.3, 0.6)
    source.annual_inflow[0, 0, :2] = (0.3, 0.6)
    inputs.primary_imports[0, :, 0, polymer] = (0.3, 0.6)
    inputs.primary_exports[0, :, 1, polymer] = (0.3, 0.6)
    recipe = derive_material_recipe(material_source_members(), source_identity="invented_xml")
    result = evaluate_mfa(source, inputs, recipe)
    local = result.material_realization.loc[
        result.material_realization["year"].eq(2018)
        & result.material_realization["region"].eq(source.regions[0])
        & result.material_realization["flow_name"].isin(
            ("new_polymer_domestic", "new_polymer_export")
        )
    ]
    np.testing.assert_array_equal(local["amount"], 0.0)
    exported = result.material_realization.loc[
        result.material_realization["year"].eq(2018)
        & result.material_realization["region"].eq(source.regions[1])
        & result.material_realization["flow_name"].eq("new_polymer_export")
        & result.material_realization["destination_material"].eq("PUR"),
        "amount",
    ]
    np.testing.assert_array_equal(exported, [fsum((0.3, 0.6))])
