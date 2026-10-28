"""Treatment and trade ownership through complete small constituent MFA results."""

from dataclasses import replace
from math import fsum

import flodym as fd
import numpy as np
import pandas as pd
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from infiltration_fixtures import resolved_infiltration_provider

from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    prepared_provider_authority,
)
from pytoce.run_lca.calculation.burdens import (
    build_burden_demands,
    mechanical_recycling_loss_result,
)
from pytoce.run_lca.calculation.custody import trace_carbon_custody
from pytoce.run_lca.calculation.foreground import compile_foreground_activations
from pytoce.run_lca.calculation.providers import ProviderRoutes
from pytoce.run_lca.calculation.result_diagnostics import mismanaged_result_tables


def test_mechanical_recycling_loss_result_preserves_flow_relationships(carbon_authorities) -> None:
    """Aggregate actual two-year input, resin and separately owned reject treatments."""
    _, recipe = carbon_authorities
    state, inputs = empty_model()
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    for t in (0, 1):
        state.annual_inflow[t, region, good] = 1
        state.stock_by_cohort[t, t, region, good] = 1
        state.outflow_by_cohort[t + 1, t, region, good] = 1
    state.annual_inflow[2, region, good] = 1
    state.stock_by_cohort[2:, 2, region, good] = 1
    inputs.mechanical_recycling_rate[1:3, region] = 0.5
    inputs.incineration_rate[1:3, region] = 0.5
    mfa = evaluate_mfa(state, inputs, recipe)
    result = mechanical_recycling_loss_result(mfa.model_flows)
    annual = result.loc[result.reporting_period.eq("annual")]
    cumulative = result.loc[result.reporting_period.eq("cumulative_2019_2100")]
    assert annual.year.tolist() == [2019, 2020]
    assert len(cumulative) == 1
    assert pd.isna(cumulative.iloc[0].year)
    # Independent target-PE sorting and mechanical-recycling stage retention; the invented
    # HDPE preparation emits 0.0001 kg per 2*0.8 kg dry feed, and its
    # finisher emits 0.001 kg principal dust per kg recovered output.
    target_sort = 31199.69 / 33540
    recovered = 0.5e9 * target_sort * 0.99 * 0.98 * 0.97
    waterborne = 0.5e9 * target_sort * 0.01
    airborne = 0.5e9 * 0.0001 / (2 * 0.8) + recovered * 0.001
    amounts = {
        "mechanical_recycling_input_kg": 0.5e9,
        "mechanical_recycling_output_kg": recovered,
        "mechanical_recycling_managed_reject_incinerated_kg": (
            0.5e9 - recovered - waterborne - airborne
        ),
        "mechanical_recycling_managed_reject_landfilled_kg": 0,
        "mechanical_recycling_waterborne_loss_kg": waterborne,
        "mechanical_recycling_airborne_loss_kg": airborne,
        "mechanical_recycling_total_loss_kg": 0.5e9 - recovered,
    }
    for column, expected in amounts.items():
        np.testing.assert_allclose(annual[column], expected, rtol=3e-15, atol=0)
        assert cumulative.iloc[0][column] == pytest.approx(2 * expected)
    for column, expected in (
        ("mechanical_recycling_output_share_of_input", recovered / 0.5e9),
        ("mechanical_recycling_total_loss_share_of_input", 1 - recovered / 0.5e9),
    ):
        np.testing.assert_allclose(result[column], expected, rtol=3e-15, atol=0)


@pytest.mark.parametrize(
    ("material", "sbr_share"),
    (
        ("Elastomers (tyres)", 1.0),
        ("Others", 0.108715495226123),
    ),
)
def test_sbr_treatment_composition_survives_activation_and_result_lineage(
    tmp_path,
    carbon_authorities,
    material,
    sbr_share,
) -> None:
    """Keep tire polymer proxies distinct from actual non-tire compound treatment."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model(material)
    region, good = state.regions.index("EUR"), state.goods.index("Transportation")
    state.annual_inflow[1, region, good] = 1
    state.stock_by_cohort[1, 1, region, good] = 1
    state.outflow_by_cohort[2, 1, region, good] = 1
    inputs.collection_rate[2, region] = 0
    mfa = evaluate_mfa(state, inputs, recipe)
    authority = resolved_infiltration_provider(tmp_path / "native_infiltration")
    routes = ProviderRoutes(
        authority.routes,
        identity=authority.identity,
        interface_database=authority.interface_database,
        material_carbon=carbon,
    )
    activations = compile_foreground_activations(mfa, routes, recipe).activations
    constituent = "non_tyre_elastomers" if material == "Others" else "elastomers"
    sbr = activations.loc[
        activations.constituent.eq(constituent)
        & activations.demand_type.isin(
            ("mismanaged_open_burning_treatment", "mismanaged_open_dump_treatment")
        )
    ]
    assert fsum(sbr.amount) == pytest.approx(sbr_share * 1e9)
    resolved = activations.rename(
        columns={
            "demand_type": "life_cycle_stage",
            "basis_flow_names": "source_flow_names",
            "provider_service_database": "provider_database",
            "provider_service_code": "provider_code",
        }
    )
    resolved["provider_anchor_year"] = 2025
    resolved["temporal_provider_share"] = 1.0
    resolved["provider_source_database"] = "invented_ecoinvent_2025"
    resolved["provider_source_role"] = "native_ecoinvent_provider"
    resolved["general_background_database"] = "invented_ecoinvent_2025"
    resolved["general_background_anchor_year"] = 2025
    resolved["general_background_role"] = "ordinary_ecoinvent"
    resolved["selected_background_member"] = "remind_ssp2_npi"

    class ProviderMetadata:
        """Provide the Brightway metadata read for already resolved fixture services."""

        def get_node(self, *, database: str, code: str) -> dict[str, str]:
            """Return metadata only; the package owns allocation and result construction."""
            return {
                "name": code,
                "reference product": "waste polymer",
                "unit": "kilogram",
                "location": "GLO",
            }

    fate, closure = mismanaged_result_tables(
        ProviderMetadata(),
        resolved,
        foreground_scenario="constituent_fixture",
        background_scenario="remind_ssp2_npi",
    )
    selected = fate.loc[fate.constituent.eq(constituent)]
    for rows in (sbr, selected):
        if material == "Others":
            # The constructed compound keeps its own principal/additive fate;
            # the PS/PE split applies only to the tire SBR-equivalent polymer.
            assert set(rows.analogue_status) == {
                "actual_material_with_transferred_burning_function",
                "filled_material_source_phase_reference",
            }
            assert not rows.sbr_proxy_component_role.isin(
                ("styrene_derived_fraction", "butadiene_derived_fraction")
            ).any()
        else:
            assert rows.styrene_component_share.eq(0.25).all()
            assert rows.butadiene_analogue_component_share.eq(0.75).all()
            assert rows.material_functional_unit.eq(
                "1 kg dry SBR-equivalent synthetic-rubber polymer"
            ).all()
            assert rows.whole_tyre_constituents_excluded.all()
            assert rows.no_treatment_sensitivity_executed.all()
    assert selected.provider_source_database.eq("invented_ecoinvent_2025").all()
    assert selected.general_background_role.eq("ordinary_ecoinvent").all()
    assert fate.provider_member.eq("remind_ssp2_npi").all()
    assert fate.provider_anchor_year.eq(2025).all()
    assert fsum(fate.final_provider_demand_amount_kg) == pytest.approx(1e9)
    assert fsum(closure.parent_mismanaged_mass_kg) == pytest.approx(1e9)
    # The source portfolios retain decimal rounding, admitted within 1e-12
    # of their input share. Check that scale rather than an absolute kg bound.
    np.testing.assert_allclose(
        closure.allocation_residual_kg / closure.parent_mismanaged_mass_kg, 0, atol=1e-12
    )
    np.testing.assert_allclose(fate.closure_residual_kg / fate.parent_mass_kg, 0, atol=1e-12)


def test_closed_intermediate_trade_assigns_conversion_to_the_producing_region(
    carbon_authorities,
) -> None:
    """Move processing with matched imports and exports and conserve global burden."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model()
    eur, usa = state.regions.index("EUR"), state.regions.index("USA")
    good, family = state.goods.index("Packaging"), state.materials.index("HDPE")
    state.annual_inflow[1, eur, good] = 1
    state.stock_by_cohort[1:, 1, eur, good] = 1
    baseline = evaluate_mfa(state, inputs, recipe)
    inputs.intermediate_exports[1, :, usa, family] = (0.1, 0.3)
    inputs.intermediate_imports[1, :, eur, family] = (0.1, 0.3)
    traded = evaluate_mfa(state, inputs, recipe)
    authority = prepared_provider_authority()
    routes = ProviderRoutes(
        authority.routes,
        identity=authority.identity,
        interface_database=authority.interface_database,
        material_carbon=carbon,
    )
    original = compile_foreground_activations(baseline, routes, recipe).activations
    changed = compile_foreground_activations(traded, routes, recipe).activations
    conversion = changed.loc[changed.demand_type.eq("processing_conversion")]
    assert conversion.groupby("region").amount.sum().to_dict() == pytest.approx(
        {"EUR": 0.6e9, "USA": 0.4e9}
    )
    assert original.loc[
        original.demand_type.eq("processing_conversion"), "amount"
    ].sum() == pytest.approx(1e9)
    # Both regions use the same fixture providers. Closed trade redistributes
    # regional burdens but leaves global inventory service demands unchanged.
    keys = ["year", "demand_type", "provider_service_code"]
    pd.testing.assert_series_equal(
        original.groupby(keys).amount.agg(fsum),
        changed.groupby(keys).amount.agg(fsum),
        check_exact=False,
        rtol=2e-15,
        atol=0,
    )
    fate, closure = mismanaged_result_tables(
        object(),
        original.rename(columns={"demand_type": "life_cycle_stage"}),
        foreground_scenario="constituent_fixture",
        background_scenario="remind_ssp2_npi",
    )
    assert fate.empty and closure.empty
    assert {"fate_share", "final_provider_demand_amount_kg"} <= set(fate.columns)
    assert {"parent_mismanaged_mass_kg", "allocation_residual_kg"} <= set(closure.columns)


def test_origin_attribution_underflow_excludes_zero_treatment_activations(
    carbon_authorities,
) -> None:
    """Exclude a treatment demand that rounds to zero after actual origin attribution.

    This direct-owner boundary case uses native lifetime cohorts and real MFA,
    custody and burden calculations. The extreme admitted treatment fraction
    exercises floating-point behavior.
    """
    carbon, recipe = carbon_authorities
    state, inputs = empty_model()
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    dimensions = fd.DimensionSet(
        dim_list=[
            fd.Dimension(name="Time", letter="t", items=list(state.years), dtype=int),
            fd.Dimension(name="Region", letter="r", items=list(state.regions), dtype=str),
            fd.Dimension(name="Good", letter="g", items=list(state.goods), dtype=str),
        ]
    )
    # A one-year lognormal mean with ten-percent spread retires at least 96%
    # between ages 0.5 and 1.5 years (Chebyshev's bound). Of 1.25 Mt entering
    # in 2020, therefore at least 1.2 Mt is collected in 2021. Each equal HDPE
    # carbon origin is above 0.5 Mt because its carbon mass fraction is 6/7.
    lifetime = fd.LogNormalLifetime(
        dims=dimensions,
        mean=np.full(dimensions.shape, 1.0),
        std=np.full(dimensions.shape, 0.1),
    )
    native = fd.InflowDrivenDSM(dims=dimensions, lifetime_model=lifetime)
    native.inflow.values[state.years.index(2020), region, good] = 1.25
    native.compute()
    state = replace(
        state,
        stock_by_cohort=native.get_stock_by_cohort().copy(),
        outflow_by_cohort=native.get_outflow_by_cohort().copy(),
        annual_stock=native.stock.values.copy(),
        annual_inflow=native.inflow.values.copy(),
        annual_outflow=native.outflow.values.copy(),
        lifetime_mean=np.full(len(state.goods), 1.0),
        lifetime_std=np.full(len(state.goods), 0.1),
    )
    inputs.bio_production_rate[state.years.index(2020), region] = 0.5
    # The smallest positive binary64 fraction is within the continuous [0, 1]
    # input domain. Treatment and each carbon component round to this
    # value, but splitting its service amount equally rounds both parts to zero
    # before the burden owner's Mt-to-kg conversion.
    smallest_positive = float.fromhex("0x0.0000000000001p-1022")
    inputs.incineration_rate[state.years.index(2021), region] = smallest_positive
    mfa = evaluate_mfa(state, inputs, recipe)
    physical = mfa.material_realization.loc[
        mfa.material_realization.year.eq(2021)
        & mfa.material_realization.region.eq("EUR")
        & mfa.material_realization.source_constituent.eq("hdpe")
    ]
    collected = physical.loc[physical.flow_name.eq("collected_after_trade"), "amount"].item()
    assert 1.2 <= collected <= 1.25
    assert (
        physical.loc[physical.flow_name.eq("incineration_treatment"), "amount"].item()
        == smallest_positive
    )
    trace = trace_carbon_custody(mfa, carbon, recipe)
    treatment = trace.calculation.loc[
        trace.calculation.year.eq(2021) & trace.calculation.custody_status.eq("incineration_route")
    ]
    assert treatment.set_index("ultimate_carbon_origin").carbon_amount_mt.to_dict() == {
        "fossil": smallest_positive,
        "biogenic": smallest_positive,
    }
    demands = build_burden_demands(mfa, carbon, trace).provider_demands
    selected = demands.loc[demands.year.eq(2021) & demands.demand_type.eq("incineration_treatment")]
    assert len(selected) == 2
    assert selected.origin_fraction.eq(0.5).all()
    assert selected.mfa_carbon_amount_kg.gt(0).all()
    assert selected.amount_kg.eq(0).all()
    authority = prepared_provider_authority()
    routes = ProviderRoutes(
        authority.routes,
        identity=authority.identity,
        interface_database=authority.interface_database,
        material_carbon=carbon,
    )
    activations = compile_foreground_activations(mfa, routes, recipe).activations
    assert not activations.empty
    assert activations.amount.ne(0).all()
    assert activations.loc[
        activations.year.eq(2021) & activations.demand_type.eq("incineration_treatment")
    ].empty
