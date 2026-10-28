"""Small positive current-cohort carbon survives historical and recycling feedback."""

from math import fsum

import numpy as np
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa

from pytoce.run_lca.calculation.custody import trace_carbon_custody
from pytoce.run_mfa.model.cohorts import conditional_survival_ratio
from pytoce.run_mfa.model.materials import load_material_composition


def test_tiny_current_discard_keeps_positive_carbon_through_recycling(carbon_authorities) -> None:
    """Retain a real positive lifetime tail while fossil history feeds the same-year loop."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model()
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    # Fixed packaging lifetime: mean 2.5 years, 20% relative dispersion, and half-year exposure of
    # new inflow. These inputs yield the positive survival tail.
    survival = float(
        conditional_survival_ratio(np.array([0.0]), np.array([0.5]), mean=2.5, std=0.5)[0]
    )
    current_discard = 1.0 - survival
    assert 0 < current_discard < 1e-12
    state.lifetime_mean[good], state.lifetime_std[good] = 2.5, 0.5
    state.annual_inflow[1, region, good] = 1.0
    state.stock_by_cohort[1, 1, region, good] = 1.0
    state.outflow_by_cohort[2, 1, region, good] = 1.0
    state.annual_inflow[2, region, good] = 1.0
    state.stock_by_cohort[2:, 2, region, good] = survival
    state.outflow_by_cohort[2, 2, region, good] = current_discard
    inputs.bio_production_rate[2, region] = 1.0
    inputs.mechanical_recycling_rate[2, region] = 0.5
    inputs.incineration_rate[2, region] = 0.5
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)

    annual = trace.calculation.loc[
        trace.calculation.year.eq(2020) & trace.calculation.region.eq("EUR")
    ]
    origins = ("fossil", "biogenic", "atmospheric_dac")
    flows = annual.pivot_table(
        index="source_flow_name",
        columns="ultimate_carbon_origin",
        values="carbon_amount_mt",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=origins, fill_value=0)
    incoming = flows.loc["fabrication => use"].to_numpy()
    # Half the collected discard is burned, giving an independently reported
    # view of the full discard-origin equation; the other half is recycled.
    discarded = 2 * flows.loc["collected => incineration"].to_numpy()
    historical = np.array((6 / 7, 0.0, 0.0))
    expected_discard = historical + current_discard * incoming
    assert 0 < discarded[1] < 1e-12
    assert discarded[2] == incoming[2] == 0.0
    np.testing.assert_allclose(discarded, expected_discard, rtol=3e-14, atol=0)
    np.testing.assert_allclose(fsum(discarded), (1 + current_discard) * 6 / 7, rtol=3e-14, atol=0)

    recovery_name = next(
        row.mechanical_recovery
        for row in load_material_composition().constituents
        if row.constituent == "hdpe"
    )
    recovery = next(
        row.recovered_kg_per_input_kg
        for row in recipe.mechanical_recovery
        if row.name == recovery_name
    )
    gross = mfa.material_realization.loc[
        mfa.material_realization.year.eq(2020)
        & mfa.material_realization.region.eq("EUR")
        & mfa.material_realization.source_constituent.eq("hdpe")
        & mfa.material_realization.flow_name.eq("manufacturing_principal_input")
    ].amount.sum()
    # The current mixture returns through recovery: g*f = eta/2 * (1 + q*f).
    fossil_fraction = 0.5 * recovery / (gross - 0.5 * recovery * current_discard)
    assert 0 < fossil_fraction < 1
    np.testing.assert_allclose(
        incoming,
        np.array((fossil_fraction, 1 - fossil_fraction, 0)) * 6 / 7,
        rtol=3e-14,
        atol=0,
    )
    recycled = flows.loc["reclmech => processing"].to_numpy()
    assert recycled[0] > 0 and recycled[1] > 0 and recycled[2] == 0
    np.testing.assert_allclose(recycled, 0.5 * recovery * expected_discard, rtol=3e-14, atol=0)
