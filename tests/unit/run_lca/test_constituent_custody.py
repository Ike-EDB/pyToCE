"""Track constituent carbon through trade, cohorts and recycling with synthetic MFA inputs."""

from math import fsum
from typing import cast

import numpy as np
import pandas as pd
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from infiltration_fixtures import resolved_infiltration_provider

from pytoce.run_lca.calculation.custody import (
    prepare_carbon_cohort_weights,
    trace_carbon_custody,
)
from pytoce.run_mfa.model.materials import (
    load_material_composition,
)


def test_final_product_trade_preserves_use_and_future_carbon_origin(carbon_authorities) -> None:
    """Keep fossil packaging and biogenic construction exports in their own global markets."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model()
    family = state.materials.index("HDPE")
    for supplier, consumer, good in (
        ("EUR", "CHA", "Packaging"),
        ("USA", "IND", "Buildings & Construction"),
    ):
        source, destination, use = (
            state.regions.index(supplier),
            state.regions.index(consumer),
            state.goods.index(good),
        )
        state.annual_inflow[1, destination, use] = 1
        state.stock_by_cohort[1, 1, destination, use] = 1
        state.outflow_by_cohort[2, 1, destination, use] = 1
        inputs.final_exports[1, :, source, family, use] = (0.25, 0.75)
        inputs.final_imports[1, :, destination, family, use] = (0.25, 0.75)
    inputs.bio_production_rate[1, state.regions.index("USA")] = 1
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    entry = trace.calculation.loc[trace.calculation["custody_status"].eq("entry_into_use")]
    assert set(zip(entry.region, entry.good, entry.ultimate_carbon_origin, strict=True)) == {
        ("CHA", "Packaging", "fossil"),
        ("IND", "Buildings & Construction", "biogenic"),
    }
    np.testing.assert_allclose(entry.carbon_amount_mt, 6 / 7, rtol=2e-15, atol=0)
    disposal = trace.calculation.loc[trace.calculation["custody_status"].eq("incineration_route")]
    assert set(zip(disposal.region, disposal.ultimate_carbon_origin, strict=True)) == {
        ("CHA", "fossil"),
        ("IND", "biogenic"),
    }
    assert set(disposal.year) == {2020}
    np.testing.assert_allclose(disposal.carbon_amount_mt, 6 / 7, rtol=2e-15, atol=0)
    # The nominal C coordinate uses 0.25; constituent carbon follows its own density.
    assert {entry[0] for entry in trace.history.entries} == {2019}


def test_ps_oil_carbon_survives_trade_recovery_and_historical_burning(
    carbon_authorities, tmp_path
) -> None:
    """Keep traded fossil resin and fixed oil when recovery mixes with new renewable PS."""
    from pytoce.prepare_backgrounds.preparation.providers.route_authorities import (
        mismanaged_fate_authorities,
    )
    from pytoce.run_lca.calculation.burdens import build_burden_demands
    from pytoce.run_lca.calculation.foreground import (
        OPTIMIZATION_ACTIVATION_COLUMNS,
        compile_foreground_activations,
        compile_optimization_activation_context,
    )
    from pytoce.run_lca.calculation.providers import ProviderRoutes

    carbon, recipe = carbon_authorities
    state, inputs = empty_model("PS")
    supplier, consumer = state.regions.index("EUR"), state.regions.index("CHA")
    family, good = state.materials.index("PS"), state.goods.index("Packaging")
    for time in (1, 2):
        state.annual_inflow[time, consumer, good] = 1
        state.stock_by_cohort[time, time, consumer, good] = 1
        state.outflow_by_cohort[time + 1, time, consumer, good] = 1
    # REMIND's illustrative .25/.75 element split controls trade; resin carbon density is prepared
    # separately.
    inputs.final_exports[1, :, supplier, family, good] = (0.25, 0.75)
    inputs.final_imports[1, :, consumer, family, good] = (0.25, 0.75)
    # The catalog's renewable PS production starts in 2020. Historical
    # 2019 traded resin is fossil; next year's new resin uses both alternatives.
    inputs.bio_production_rate[2, consumer] = 0.4
    inputs.daccu_production_rate[2, consumer] = 0.6
    inputs.mechanical_recycling_rate[2, consumer] = 0.5
    inputs.incineration_rate[2, consumer] = 0.5
    inputs.collection_rate[3, consumer] = 0
    inputs.incineration_rate[3, consumer] = 0
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    # ACC/Franklin nominal retained GPPS and HIPS, with their existing
    # finished-product portfolio. Oil is already inside each product kg.
    gpps, hips = 0.650485436893, 0.349514563107
    selected_c = gpps * 0.9862 * 12 / 13 + hips * (0.902 * 12 / 13 + 0.082 * 8 / 9)
    oil_c = (gpps * 0.0138 + hips * 0.016) * 6 / 7
    old = np.array((oil_c + selected_c, 0.0, 0.0))
    new = np.array((oil_c, 0.4 * selected_c, 0.6 * selected_c))
    physical = mfa.material_realization.loc[
        mfa.material_realization.year.eq(2020)
        & mfa.material_realization.region.eq("CHA")
        & mfa.material_realization.source_constituent.eq("ps")
    ]
    gross = physical.loc[physical.flow_name.eq("manufacturing_principal_input")].amount.sum()
    recovery = next(row for row in recipe.mechanical_recovery if row.name == "ps_granulate")
    recovered = 0.5 * recovery.recovered_kg_per_input_kg
    mixed = (recovered * old + (gross - recovered) * new) / gross
    assert mixed[0] == pytest.approx(oil_c + recovered / gross * selected_c)
    assert np.all(mixed > 0)
    for year, status, expected in (
        (2019, "entry_into_use", old),
        (2020, "mechanically_recycled_output", recovered * old),
        (2020, "entry_into_use", mixed),
    ):
        rows = trace.calculation.loc[
            trace.calculation.year.eq(year)
            & trace.calculation.region.eq("CHA")
            & trace.calculation.constituent.eq("ps")
            & trace.calculation.custody_status.eq(status)
        ]
        actual = rows.groupby("ultimate_carbon_origin").carbon_amount_mt.sum()
        np.testing.assert_allclose(
            actual.reindex(("fossil", "biogenic", "atmospheric_dac"), fill_value=0),
            expected,
            rtol=3e-14,
            atol=0,
        )
    resumed = trace_carbon_custody(
        mfa,
        carbon,
        recipe,
        history=trace.history,
        cohort_weights_by_year=prepare_carbon_cohort_weights(
            mfa.cohort_state, mfa.cohort_material_identity
        ),
    )
    pd.testing.assert_frame_equal(
        resumed.calculation,
        trace.calculation.loc[trace.calculation.year > 2019].reset_index(drop=True),
    )
    assert resumed.history == trace.history
    demands = build_burden_demands(mfa, carbon, trace).provider_demands
    fate = mismanaged_fate_authorities().set_index("region").loc["CHA"]
    for demand, share in (
        ("mismanaged_open_burning_treatment", float(cast(float, fate.open_burning_share))),
        ("mismanaged_open_dump_treatment", float(cast(float, fate.open_dump_share))),
    ):
        rows = demands.loc[demands.year.eq(2021) & demands.demand_type.eq(demand)]
        actual = rows.groupby("ultimate_carbon_origin").amount_kg.sum()
        np.testing.assert_allclose(
            actual.reindex(("fossil", "biogenic", "atmospheric_dac")),
            1e9 * share * mixed / fsum(mixed),
            rtol=3e-14,
        )
        assert fsum(rows.amount_kg) == pytest.approx(1e9 * share)
    authority = resolved_infiltration_provider(tmp_path / "ps_infiltration")
    routes = ProviderRoutes(
        authority.routes,
        identity=authority.identity,
        interface_database=authority.interface_database,
        material_carbon=carbon,
    )
    detailed = compile_foreground_activations(mfa, routes, recipe).activations
    for year, route in ((2019, "fossil"), (2020, "biogenic"), (2020, "atmospheric_dac")):
        virgin = detailed.loc[
            detailed.year.eq(year)
            & detailed.demand_type.eq("virgin_resin_production")
            & detailed.virgin_feedstock_route.eq(route)
        ]
        assert set(virgin.virgin_feedstock_route) == {route}
        assert set(virgin.ultimate_carbon_origin) == {"fossil", route}
        assert set(virgin.provider_service_id) == {f"ps_{route}"}
    burning = detailed.loc[
        detailed.year.eq(2021) & detailed.demand_type.eq("mismanaged_open_burning_treatment")
    ]
    for row in burning.itertuples():
        expected_provider = "mismanaged_open_burning_ps"
        if row.ultimate_carbon_origin != "fossil":
            expected_provider += "__non_fossil"
        assert row.provider_service_id == expected_provider
    assert fsum(burning.amount) == pytest.approx(1e9 * float(cast(float, fate.open_burning_share)))
    optimized = compile_optimization_activation_context(mfa, routes, recipe)
    expected = (
        detailed.groupby(list(OPTIMIZATION_ACTIVATION_COLUMNS[:-1]), observed=True, sort=True)
        .amount.agg(fsum)
        .reset_index()
    )
    pd.testing.assert_frame_equal(optimized.activations, expected)


def test_cohort_entry_density_and_resumed_history_remain_exact(carbon_authorities) -> None:
    """Preserve changing TPS carbon content and return identical full/resumed disposal rows."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("BioPlastics")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    for time in (0, 1):
        state.annual_inflow[time, region, good] = 1
        state.stock_by_cohort[time:2, time, region, good] = 1
        state.outflow_by_cohort[2, time, region, good] = 1
    inputs.bio_production_rate[1, region] = 1
    mfa = evaluate_mfa(state, inputs, recipe)
    full = trace_carbon_custody(mfa, carbon, recipe)
    assert full.calculation.year.min() == 2019
    assert {row[0] for row in full.history.entries} == {2018, 2019}
    history = {
        (year, region, material, constituent, good): np.array((f, b, d))
        for year, region, material, constituent, good, f, b, d in full.history.tps_route_entries
    }
    index = carbon.constituents.index(("BioPlastics", "starch_blends"))
    densities = np.asarray(carbon.fractions[index])
    assert fsum(densities[0]) != fsum(densities[1])
    for year, route in ((2018, 0), (2019, 1)):
        np.testing.assert_allclose(
            history[(year, "EUR", "BioPlastics", "starch_blends", "Packaging")],
            np.eye(3)[route],
            rtol=3e-15,
            atol=0,
        )
    tps_share = next(
        c.share
        for c in load_material_composition().constituents
        if c.constituent == "starch_blends"
    )
    disposal = full.calculation.loc[
        full.calculation["constituent"].eq("starch_blends")
        & full.calculation["custody_status"].eq("incineration_route")
    ]
    observed = (
        disposal.groupby("ultimate_carbon_origin")["carbon_amount_mt"]
        .sum()
        .reindex(("fossil", "biogenic", "atmospheric_dac"), fill_value=0)
    )
    np.testing.assert_allclose(
        observed, float(tps_share) * (densities[0] + densities[1]), rtol=3e-15, atol=0
    )
    resumed = trace_carbon_custody(
        mfa,
        carbon,
        recipe,
        history=full.history,
        cohort_weights_by_year=prepare_carbon_cohort_weights(
            mfa.cohort_state, mfa.cohort_material_identity
        ),
    )
    pd.testing.assert_frame_equal(
        resumed.calculation,
        full.calculation.loc[full.calculation.year > 2019].reset_index(drop=True),
    )
    assert resumed.history == full.history
    assert resumed.tps_route_amounts == {
        key: value for key, value in full.tps_route_amounts.items() if key[0] > 2019
    }
    assert not {row[3] for row in full.history.entries} & {row[3] for row in history}


@pytest.mark.parametrize(
    ("bio", "dac"),
    ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1e-14, 0.5)),
    ids=("fossil", "native", "dac", "small_native_share"),
)
def test_tps_formulation_amounts_preserve_endpoints_and_small_shares(
    carbon_authorities, bio, dac
) -> None:
    """Retain historical formulation masses in recovery and both landfill routes."""
    from pytoce.run_lca.calculation.burdens import build_burden_demands

    carbon, recipe = carbon_authorities
    composition = load_material_composition().constituents
    starch_share = float(next(c.share for c in composition if c.constituent == "starch_blends"))
    state, inputs = empty_model("BioPlastics")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    # The starch-blend principal is 0.1 Mt, whose inverse carbon calculation
    # would give a negative roundoff remainder for a pure synthetic route.
    amount = 0.1 / starch_share
    state.annual_inflow[1, region, good] = amount
    state.stock_by_cohort[1, 1, region, good] = amount
    state.outflow_by_cohort[2, 1, region, good] = amount
    state.annual_inflow[2, region, good] = amount
    state.stock_by_cohort[2:, 2, region, good] = amount
    inputs.bio_production_rate[1, region] = bio
    inputs.daccu_production_rate[1, region] = dac
    inputs.mechanical_recycling_rate[2, region] = 0.5
    inputs.incineration_rate[2, region] = 0.25
    # Landfill receives the remaining quarter of collected waste.
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    demands = build_burden_demands(mfa, carbon, trace).provider_demands
    shares = np.array((1 - bio - dac, bio, dac))
    for item in composition:
        if item.mechanical_recovery != "tps_compound":
            continue
        for year, flow, product_use, fraction in (
            (2019, "fabrication => use", "Packaging", 1.0),
            (2020, "collected => incineration", "all", 0.25),
            (2020, "collected => landfill", "all", 0.25),
            (2020, "collected => reclmech", "all", 0.5),
        ):
            expected = amount * float(item.share) * shares * fraction
            key = (year, "EUR", item.material, item.constituent, product_use, flow)
            routed = np.asarray(trace.tps_route_amounts[key])
            np.testing.assert_array_equal(routed[shares == 0], 0)
            assert np.all(routed[shares > 0] > 0)
            np.testing.assert_allclose(routed, expected, rtol=3e-14, atol=0)
            index = carbon.constituents.index((item.material, item.constituent))
            expected_carbon = expected @ np.asarray(carbon.fractions[index])
            selected = trace.calculation.loc[
                trace.calculation.year.eq(year)
                & trace.calculation.region.eq("EUR")
                & trace.calculation.constituent.eq(item.constituent)
                & trace.calculation.source_flow_name.eq(flow)
            ]
            observed = selected.set_index("ultimate_carbon_origin").carbon_amount_mt.reindex(
                ("fossil", "biogenic", "atmospheric_dac"), fill_value=0
            )
            np.testing.assert_allclose(observed, expected_carbon, rtol=3e-14, atol=0)
        tps = next(row for row in recipe.mechanical_recovery if row.name == "tps_compound")
        # Half the collected input enters recovery; the managed principal remainder goes to
        # landfill. Providers account for wet auxiliaries separately.
        managed = 1 - fsum(
            (
                tps.recovered_kg_per_input_kg,
                tps.waterborne_kg_per_input_kg,
                tps.airborne_kg_per_input_kg,
            )
        )
        for demand, fraction in (
            ("mechanical_recycling_operation", 0.5),
            ("landfill_treatment", 0.25),
            ("mechanical_recycling_reject_landfill_treatment", 0.5 * managed),
        ):
            selected = demands.loc[
                demands.year.eq(2020)
                & demands.constituent.eq(item.constituent)
                & demands.demand_type.eq(demand)
            ]
            observed_mass = (
                selected.groupby("virgin_feedstock_route")
                .amount_kg.agg(fsum)
                .reindex(("fossil", "biogenic", "atmospheric_dac"), fill_value=0)
            )
            # Convert principal Mt to kg; source routes and any tiny positive
            # share survive the same operation/landfill admission unchanged.
            np.testing.assert_allclose(
                observed_mass,
                amount * float(item.share) * shares * fraction * 1e9,
                rtol=3e-14,
                atol=0,
            )


def test_tps_burning_keeps_recycled_formulations_and_regional_fates(carbon_authorities) -> None:
    """Burn the retired mixture once after recovery and a change of virgin technology."""
    from pytoce.prepare_backgrounds.preparation.providers.route_authorities import (
        mismanaged_fate_authorities,
    )
    from pytoce.run_lca.calculation.burdens import build_burden_demands

    carbon, recipe = carbon_authorities
    state, inputs = empty_model("BioPlastics")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    for year in (1, 2, 3):
        state.annual_inflow[year, region, good] = 1
        state.stock_by_cohort[year, year, region, good] = 1
    state.outflow_by_cohort[2, 1, region, good] = 1
    state.outflow_by_cohort[3, 2, region, good] = 1
    # The first cohort mixes three technologies; half its collected principal enters recycling. New
    # production is biogenic in 2020 and DAC in 2021; recycled material retains its historical
    # formulation.
    inputs.bio_production_rate[1, region] = 0.2
    inputs.daccu_production_rate[1, region] = 0.3
    inputs.bio_production_rate[2, region] = 1
    inputs.daccu_production_rate[3, region] = 1
    inputs.mechanical_recycling_rate[2, region] = 0.5
    inputs.incineration_rate[2, region] = 0.5
    inputs.collection_rate[3, region] = 0.6
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    demands = build_burden_demands(mfa, carbon, trace).provider_demands
    fate = mismanaged_fate_authorities().set_index("region").loc["EUR"]
    origins = ("fossil", "biogenic", "atmospheric_dac")
    for item in load_material_composition().constituents:
        if item.mechanical_recovery != "tps_compound":
            continue
        routed = np.asarray(
            trace.tps_route_amounts[
                (2021, "EUR", item.material, item.constituent, "all", "eol => mismanaged")
            ]
        )
        # Forty percent of the one-Mt retired cohort is uncollected. Family
        # composition is applied here once; recovered fossil/DAC material is
        # still present even though all new 2020 principal used biogenic feed.
        assert fsum(routed) == pytest.approx(0.4 * float(item.share))
        assert np.all(routed > 0)
        for demand, fraction in (
            ("mismanaged_open_burning_treatment", float(cast(float, fate.open_burning_share))),
            ("mismanaged_open_dump_treatment", float(cast(float, fate.open_dump_share))),
        ):
            selected = demands.loc[
                demands.year.eq(2021)
                & demands.constituent.eq(item.constituent)
                & demands.demand_type.eq(demand)
            ]
            observed = (
                selected.groupby("virgin_feedstock_route").amount_kg.agg(fsum).reindex(origins)
            )
            np.testing.assert_allclose(observed, routed * fraction * 1e9, rtol=3e-14, atol=0)
            index = carbon.constituents.index((item.material, item.constituent))
            for r, origin in enumerate(origins):
                output = selected.loc[selected.virgin_feedstock_route.eq(origin)]
                origin_carbon = output.set_index("ultimate_carbon_origin").mfa_carbon_amount_kg
                np.testing.assert_allclose(
                    origin_carbon.reindex(origins, fill_value=0),
                    routed[r] * np.asarray(carbon.fractions[index][r]) * fraction * 1e9,
                    rtol=3e-14,
                    atol=0,
                )


def test_recovered_monomer_pools_use_physical_coefficients_and_retention(
    carbon_authorities,
) -> None:
    """Mix traded HDPE/PP waste once and retain unequal ethylene/propylene substitution."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model()
    recipient, good = state.regions.index("CHA"), state.goods.index("Packaging")
    state.material_shares[recipient] = 1 / len(state.materials)
    state.annual_inflow[2, recipient, good] = len(state.materials)
    state.stock_by_cohort[2:, 2, recipient, good] = len(state.materials)
    inputs.bio_production_rate[2, recipient] = 1
    inputs.chemical_recycling_rate[2, recipient] = 1
    inputs.incineration_rate[2, recipient] = 0
    for region, material, amount in (("EUR", "HDPE", 1.0), ("USA", "PP", 3.0)):
        source, family = state.regions.index(region), state.materials.index(material)
        state.material_shares[source] = 0
        state.material_shares[source, family] = 1
        state.annual_inflow[1, source, good] = amount
        state.stock_by_cohort[1, 1, source, good] = amount
        state.outflow_by_cohort[2, 1, source, good] = amount
        inputs.waste_exports[2, :, source, family, good] = np.array((0.25, 0.75)) * amount
        inputs.waste_imports[2, :, recipient, family, good] = np.array((0.25, 0.75)) * amount
    inputs.daccu_production_rate[1, state.regions.index("USA")] = 1
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    produced = trace.calculation.loc[
        trace.calculation.year.eq(2020)
        & trace.calculation.region.eq("CHA")
        & trace.calculation.source_flow_name.eq("new_polymer => processing")
    ]
    # The material-balance calculation supplies gross resin, which exceeds finished demand.
    # Recovered carbon retains the traded waste's 1:3 origin ratio; each resin
    # retains only its own stoichiometric fraction of the E/P inputs.
    physical = mfa.material_realization.loc[
        mfa.material_realization.year.eq(2020) & mfa.material_realization.region.eq("CHA")
    ]
    monomer_carbon = {row.product: row.carbon_kg_per_kg for row in recipe.cracker_products}
    recovered_carbon = []
    origins = ("fossil", "biogenic", "atmospheric_dac")
    requirements = {row.polymer: row for row in recipe.polymers}
    for item in load_material_composition().constituents:
        if item.chemical_polymer is None:
            continue
        requirement = requirements[item.chemical_polymer]
        selected = physical.loc[
            physical.destination_material.eq(item.material)
            & physical.destination_constituent.eq(item.constituent)
        ]
        resin = selected.loc[
            selected.flow_name.isin(("new_polymer_domestic", "new_polymer_export"))
        ].amount.sum()
        incoming = (
            monomer_carbon["ethylene"] * requirement.ethylene_kg_per_kg
            + monomer_carbon["propylene"] * requirement.propylene_kg_per_kg
        )
        retained = (6 / 7) / incoming
        recovered_input = selected.loc[selected.flow_name.eq("recovered_monomer_consumption")]
        recycled_carbon = retained * fsum(
            monomer_carbon[row.product] * row.amount for row in recovered_input.itertuples()
        )
        recovered_carbon.append(recycled_carbon)
        expected = (0.25 * recycled_carbon, resin * 6 / 7 - recycled_carbon, 0.75 * recycled_carbon)
        observed = produced.loc[
            produced.material.eq(item.material) & produced.constituent.eq(item.constituent)
        ].set_index("ultimate_carbon_origin")["carbon_amount_mt"]
        np.testing.assert_allclose(observed.reindex(origins), expected, rtol=2e-14, atol=0)
    recovered = trace.calculation.loc[
        trace.calculation.custody_status.eq("retained_recovered_monomer_carbon")
    ]
    assert set(recovered["product"]) == {"ethylene", "propylene"}
    assert set(recovered.ultimate_carbon_origin) == {"fossil", "atmospheric_dac"}
    np.testing.assert_allclose(
        recovered.groupby("ultimate_carbon_origin")["carbon_amount_mt"]
        .sum()
        .reindex(origins, fill_value=0),
        fsum(recovered_carbon) * np.array((0.25, 0, 0.75)),
        rtol=2e-14,
        atol=0,
    )
    pool_carbon = fsum(
        monomer_carbon[row.product] * row.amount
        for row in physical.loc[
            physical.flow_name.eq("cracker_product")
            & physical["product"].isin(("ethylene", "propylene"))
        ].itertuples()
    )
    assert recovered.carbon_amount_mt.sum() < pool_carbon


@pytest.mark.parametrize("mass_scale", [1.0, 1e155])
def test_current_cohort_mechanical_feedback_preserves_carbon_without_normalization(
    carbon_authorities,
    mass_scale: float,
) -> None:
    """Solve a same-year recycling loop using actual gross feed and recovery yield."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model()
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    state.annual_inflow[0, region, good] = mass_scale
    state.stock_by_cohort[:2, 0, region, good] = mass_scale
    state.outflow_by_cohort[2, 0, region, good] = mass_scale
    state.annual_inflow[2, region, good] = mass_scale
    state.stock_by_cohort[2:, 2, region, good] = 0.5 * mass_scale
    state.outflow_by_cohort[2, 2, region, good] = 0.5 * mass_scale
    inputs.mechanical_recycling_rate[2, region] = 0.5
    inputs.incineration_rate[2, region] = 0.5
    inputs.bio_production_rate[2, region] = 1
    mfa = evaluate_mfa(state, inputs, recipe)
    with np.errstate(over="raise", invalid="raise"):
        trace = trace_carbon_custody(mfa, carbon, recipe)
    assert np.isfinite(trace.calculation.carbon_amount_mt).all()
    assert (trace.calculation.carbon_amount_mt >= 0).all()
    gross = (
        mfa.material_realization.loc[
            mfa.material_realization.year.eq(2020)
            & mfa.material_realization.region.eq("EUR")
            & mfa.material_realization.source_constituent.eq("hdpe")
            & mfa.material_realization.flow_name.eq("manufacturing_principal_input")
        ].amount.sum()
        / mass_scale
    )
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
    # Per mass_scale Mt: old cohort 1 fossil, current retirement .5 with the
    # unknown mixture x. Half is recycled, so g*x = eta*(.5 + .25*x).
    fossil_share = 0.5 * recovery / (gross - 0.25 * recovery)
    annual = trace.calculation.loc[trace.calculation.year.eq(2020)]
    incoming = annual.loc[annual.custody_status.eq("entry_into_use")].set_index(
        "ultimate_carbon_origin"
    )["carbon_amount_mt"]
    np.testing.assert_allclose(
        incoming.reindex(("fossil", "biogenic")) / mass_scale,
        np.array((fossil_share, 1 - fossil_share)) * (6 / 7),
        rtol=3e-15,
        atol=0,
    )
    recycled = annual.loc[annual.custody_status.eq("mechanically_recycled_output")].set_index(
        "ultimate_carbon_origin"
    )["carbon_amount_mt"]
    np.testing.assert_allclose(
        recycled.reindex(("fossil", "biogenic")) / mass_scale,
        recovery * np.array((0.5 + 0.25 * fossil_share, 0.25 * (1 - fossil_share))) * (6 / 7),
        rtol=3e-15,
        atol=0,
    )


def test_others_conversion_and_treatment_preserve_each_constituent_origin(
    carbon_authorities, tmp_path
) -> None:
    """Keep intrinsic adhesive carbon out of unrelated coating and PA treatment."""
    from pytoce.run_lca.calculation.burdens import build_burden_demands
    from pytoce.run_lca.calculation.foreground import (
        OPTIMIZATION_ACTIVATION_COLUMNS,
        compile_foreground_activations,
        compile_optimization_activation_context,
    )
    from pytoce.run_lca.calculation.providers import ProviderRoutes

    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Others")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    state.annual_inflow[1, region, good] = 1
    state.stock_by_cohort[1, 1, region, good] = 1
    state.outflow_by_cohort[2, 1, region, good] = 1
    state.annual_inflow[2, region, good] = 2
    state.stock_by_cohort[2:, 2, region, good] = 2
    inputs.bio_production_rate[2, region] = 0.25
    inputs.daccu_production_rate[2, region] = 0.25
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    built = build_burden_demands(mfa, carbon, trace)
    assert built.pa_spinning_events == ()
    demands = built.provider_demands
    annual = demands.loc[demands.year.eq(2020)]
    conversion = annual.loc[annual.demand_type.eq("processing_conversion")]
    treatment = annual.loc[annual.demand_type.eq("incineration_treatment")]
    composition = load_material_composition().constituents
    origins = ("fossil", "biogenic", "atmospheric_dac")
    for i, item in enumerate(composition):
        if item.material != "Others":
            continue
        manufactured = conversion.loc[conversion.constituent.eq(item.constituent)]
        discarded = treatment.loc[treatment.constituent.eq(item.constituent)]
        assert fsum(manufactured.amount_kg) == pytest.approx(2e9 * item.share)
        assert fsum(discarded.amount_kg) == pytest.approx(1e9 * item.share)
        # This integration boundary must forward the principal carbon after
        # manufacturing. Its selective source equations have their own tests.
        for selected, status in (
            (manufactured, "manufactured_principal_output"),
            (discarded, "incineration_route"),
        ):
            traced = trace.calculation.loc[
                trace.calculation.year.eq(2020)
                & trace.calculation.region.eq("EUR")
                & trace.calculation.constituent.eq(item.constituent)
                & trace.calculation.custody_status.eq(status)
            ]
            expected = (
                traced.groupby("ultimate_carbon_origin")
                .carbon_amount_mt.sum()
                .reindex(origins, fill_value=0)
                * 1e9
            )
            actual = selected.groupby("ultimate_carbon_origin").mfa_carbon_amount_kg.sum()
            np.testing.assert_allclose(actual.reindex(origins, fill_value=0), expected, rtol=2e-14)
    assert set(treatment.loc[treatment.ultimate_carbon_origin.eq("biogenic"), "constituent"]) == {
        "adhesives",
        "sealants",
    }
    authority = resolved_infiltration_provider(tmp_path / "infiltration")
    routes = ProviderRoutes(
        authority.routes,
        identity=authority.identity,
        interface_database=authority.interface_database,
        material_carbon=carbon,
    )
    detailed = compile_foreground_activations(mfa, routes, recipe).activations
    optimized = compile_optimization_activation_context(mfa, routes, recipe)
    expected = (
        detailed.groupby(list(OPTIMIZATION_ACTIVATION_COLUMNS[:-1]), observed=True, sort=True)
        .amount.agg(fsum)
        .reset_index()
    )
    pd.testing.assert_frame_equal(optimized.activations, expected)
    incinerated = detailed.loc[
        detailed.year.eq(2020) & detailed.demand_type.eq("incineration_treatment")
    ]
    non_fossil = incinerated.loc[incinerated.ultimate_carbon_origin.eq("biogenic")]
    assert set(non_fossil.provider_service_id) == {"incineration_pur__non_fossil"}


def test_constituent_demands_preserve_provider_and_carbon_ownership(
    carbon_authorities, tmp_path
) -> None:
    """Resolve mixed virgin origins and selective recycling identically in both LCA paths."""
    from pytoce.run_lca.calculation.burdens import build_burden_demands
    from pytoce.run_lca.calculation.foreground import (
        OPTIMIZATION_ACTIVATION_COLUMNS,
        OPTIMIZATION_ACTIVATION_DELTA_COLUMNS,
        compile_foreground_activations,
        compile_optimization_activation_context,
        compile_optimization_activations_from_history,
        optimization_activation_delta,
    )
    from pytoce.run_lca.calculation.providers import ProviderRoutes

    carbon, recipe = carbon_authorities
    state, inputs = empty_model("BioPlastics")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    state.annual_inflow[1, region, good] = 1
    state.stock_by_cohort[1, 1, region, good] = 1
    state.outflow_by_cohort[2, 1, region, good] = 1
    state.annual_inflow[2, region, good] = 2
    state.stock_by_cohort[2:, 2, region, good] = 2
    inputs.bio_production_rate[2, region] = 0.25
    inputs.daccu_production_rate[2, region] = 0.25
    inputs.collection_rate[2, region] = 0.8
    inputs.mechanical_recycling_rate[2, region] = 0.2
    inputs.chemical_recycling_rate[2, region] = 0.05
    inputs.incineration_rate[2, region] = 0.45
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    demands = build_burden_demands(mfa, carbon, trace).provider_demands
    material = mfa.material_realization.loc[
        mfa.material_realization.year.eq(2020) & mfa.material_realization.region.eq("EUR")
    ]
    output = (
        material.loc[material.flow_name.isin(("new_polymer_domestic", "new_polymer_export"))]
        .groupby("destination_constituent")
        .amount.sum()
    )
    annual = demands.loc[demands.year.eq(2020)]
    for index, (family, constituent) in enumerate(carbon.constituents):
        if family != "BioPlastics" or constituent == "bio_pe":
            continue
        selected = annual.loc[
            annual.constituent.eq(constituent) & annual.demand_type.eq("virgin_resin_production")
        ]
        for r, (route, share) in enumerate(
            zip(("fossil", "biogenic", "atmospheric_dac"), (0.5, 0.25, 0.25), strict=True)
        ):
            routed = selected.loc[selected.virgin_feedstock_route.eq(route)]
            assert fsum(routed.amount_kg) == pytest.approx(output[constituent] * share * 1e9)
            observed = routed.set_index("ultimate_carbon_origin").mfa_carbon_amount_kg.reindex(
                ("fossil", "biogenic", "atmospheric_dac"), fill_value=0
            )
            np.testing.assert_allclose(
                observed,
                output[constituent] * share * np.asarray(carbon.fractions[index][r]) * 1e9,
                rtol=2e-14,
                atol=0,
            )
    pe = annual.loc[annual.constituent.eq("bio_pe")]
    assert fsum(
        pe.loc[pe.demand_type.eq("polymerization_operation"), "amount_kg"]
    ) == pytest.approx(output["bio_pe"] * 1e9)
    monomers = pe.loc[pe.demand_type.isin(("virgin_monomer_supply", "recovered_monomer_supply"))]
    np.testing.assert_allclose(
        monomers.groupby("product").amount_kg.sum(), output["bio_pe"] * 1e9, rtol=2e-14
    )
    physical_monomers = (
        material.loc[
            material.destination_constituent.eq("bio_pe")
            & material.flow_name.isin(
                ("virgin_monomer_consumption", "recovered_monomer_consumption")
            )
        ]
        .groupby(["product", "flow_name"])
        .amount.sum()
        .unstack(fill_value=0)
    )
    for product, quantities in physical_monomers.iterrows():
        virgin = quantities["virgin_monomer_consumption"]
        recovered = quantities["recovered_monomer_consumption"]
        pair = monomers.loc[monomers["product"].eq(cast(str, product))]
        for demand, amount in (
            ("virgin_monomer_supply", virgin),
            ("recovered_monomer_supply", recovered),
        ):
            assert fsum(pair.loc[pair.demand_type.eq(demand), "amount_kg"]) == pytest.approx(
                output["bio_pe"] * amount / (virgin + recovered) * 1e9
            )
    # Both input services use Q=0.04 Mt; processed carbon uses P.
    for demand in ("chemical_recycling_preparation", "chemical_recycling_conversion"):
        assert fsum(pe.loc[pe.demand_type.eq(demand), "amount_kg"]) == pytest.approx(0.04e9)
    processed = material.loc[material.flow_name.eq("chemical_processed_feed")].amount.sum()
    assert fsum(
        pe.loc[pe.demand_type.eq("chemical_recycling_conversion"), "mfa_carbon_amount_kg"]
    ) == pytest.approx(processed * 6 / 7 * 1e9)
    assert fsum(
        annual.loc[annual.demand_type.eq("processing_conversion"), "amount_kg"]
    ) == pytest.approx(2e9)
    converted = annual.loc[annual.demand_type.eq("processing_conversion")]
    weights = {
        c.constituent: c.share
        for c in load_material_composition().constituents
        if c.material == "BioPlastics"
    }
    assert converted.groupby("constituent").amount_kg.sum().to_dict() == pytest.approx(
        {constituent: 2e9 * float(share) for constituent, share in weights.items()}
    )
    mismanaged = annual.loc[annual.demand_type.str.startswith("mismanaged_")]
    assert fsum(mismanaged.amount_kg) == pytest.approx(0.2e9)
    authority = resolved_infiltration_provider(tmp_path / "native_infiltration")
    routes = ProviderRoutes(
        authority.routes,
        identity=authority.identity,
        interface_database=authority.interface_database,
        material_carbon=carbon,
    )
    detailed = compile_foreground_activations(mfa, routes, recipe).activations
    assert not detailed.empty
    conversion_activations = detailed.loc[
        detailed.demand_type.eq("processing_conversion")
        & detailed.provider_service_id.str.startswith("conversion_injection")
    ]
    assert set(
        conversion_activations.loc[
            conversion_activations.ultimate_carbon_origin.eq("fossil"), "provider_service_id"
        ]
    ) == {"conversion_injection"}
    assert set(
        conversion_activations.loc[
            conversion_activations.ultimate_carbon_origin.isin(("biogenic", "atmospheric_dac")),
            "provider_service_id",
        ]
    ) == {"conversion_injection__non_fossil"}
    historical_intrinsic_bio = detailed.loc[
        detailed.year.eq(2019)
        & detailed.constituent.eq("starch_blends")
        & detailed.demand_type.eq("virgin_resin_production")
        & detailed.ultimate_carbon_origin.eq("biogenic")
    ]
    assert set(historical_intrinsic_bio.provider_service_id) == {"tps_fossil"}
    # The retiring 2019 TPS retains the formulation for its fossil synthesis route,
    # including the biogenic primer; 2020 feed shares cannot alter that cohort.
    tps_recovery = detailed.loc[
        detailed.year.eq(2020)
        & detailed.constituent.isin(("starch_blends", "other_biodegradable"))
        & detailed.demand_type.eq("mechanical_recycling_operation")
    ]
    assert set(tps_recovery.provider_service_id) == {"mechanical_recycling_tps_synthetic"}
    assert set(tps_recovery.virgin_feedstock_route) == {"fossil"}
    assert set(tps_recovery.ultimate_carbon_origin) == {"fossil", "biogenic"}
    tps_burning = detailed.loc[
        detailed.year.eq(2020)
        & detailed.constituent.isin(("starch_blends", "other_biodegradable"))
        & detailed.demand_type.eq("mismanaged_open_burning_treatment")
    ]
    # The same historical fossil-synthesis mixture burns intact, including
    # its biogenic primer. Current virgin shares cannot select native starch.
    assert set(tps_burning.provider_service_id) == {"tps_open_burning_fossil"}
    assert set(tps_burning.virgin_feedstock_route) == {"fossil"}
    assert set(tps_burning.ultimate_carbon_origin) == {"fossil", "biogenic"}
    for demand, service_prefix in (
        ("incineration_treatment", "tps_native_incineration_"),
        ("mismanaged_open_dump_treatment", "tps_native_open_dump_"),
    ):
        disposed = detailed.loc[
            detailed.year.eq(2020)
            & detailed.constituent.isin(("starch_blends", "other_biodegradable"))
            & detailed.demand_type.eq(demand)
        ]
        # Retiring synthetic TPS retains its biogenic primer within the same
        # historical formulation, including when current production is mixed.
        assert not disposed.empty
        assert all(
            service.startswith(service_prefix) and service.endswith("_fossil")
            for service in disposed.provider_service_id
        )
        assert set(disposed.virgin_feedstock_route) == {"fossil"}
        assert set(disposed.ultimate_carbon_origin) == {"fossil", "biogenic"}
        if demand == "mismanaged_open_dump_treatment":
            assert disposed.provider_service_id.nunique() == 5
    for constituent in ("starch_blends", "other_biodegradable"):
        expected_input = material.loc[
            material.source_constituent.eq(constituent)
            & material.flow_name.eq("mechanical_treatment")
        ].amount.sum()
        assert fsum(
            tps_recovery.loc[tps_recovery.constituent.eq(constituent), "amount"]
        ) == pytest.approx(expected_input * 1e9)
    context = compile_optimization_activation_context(mfa, routes, recipe)
    expected = (
        detailed.groupby(list(OPTIMIZATION_ACTIVATION_COLUMNS[:-1]), observed=True, sort=True)
        .amount.agg(fsum)
        .reset_index()
    )
    pd.testing.assert_frame_equal(context.activations, expected)
    resumed = compile_optimization_activations_from_history(
        mfa,
        routes,
        recipe,
        context,
        cohort_weights_by_year=prepare_carbon_cohort_weights(
            mfa.cohort_state, mfa.cohort_material_identity
        ),
    ).activations
    pd.testing.assert_frame_equal(resumed, context.activations)
    unchanged = optimization_activation_delta(context.activations, resumed)
    assert unchanged.empty
    assert tuple(unchanged.columns) == OPTIMIZATION_ACTIVATION_DELTA_COLUMNS
    expected_dtypes = {column: "object" for column in OPTIMIZATION_ACTIVATION_DELTA_COLUMNS}
    expected_dtypes.update(
        year="int64",
        physical_event_year="int64",
        parent_amount_kg="float64",
        candidate_amount_kg="float64",
        delta_amount_kg="float64",
    )
    assert unchanged.dtypes.astype(str).to_dict() == expected_dtypes

    from pytoce.run_lca.calculation.result_diagnostics import mismanaged_result_tables

    class ProviderMetadata:
        """Expose the Brightway metadata read."""

        def get_node(self, *, database: str, code: str) -> dict[str, str]:
            """Return metadata for the actual service code requested by the owner."""
            return {
                "name": code,
                "reference product": "waste treatment",
                "unit": "kilogram",
                "location": "GLO",
            }

    resolved = detailed.rename(
        columns={
            "demand_type": "life_cycle_stage",
            "basis_flow_names": "source_flow_names",
            "provider_service_database": "provider_database",
            "provider_service_code": "provider_code",
        }
    )
    resolved["provider_anchor_year"] = 2025
    resolved["temporal_provider_share"] = 1.0
    resolved["provider_source_database"] = "native_2025"
    resolved["provider_source_role"] = "native_ecoinvent_provider"
    resolved["general_background_database"] = "native_2025"
    resolved["general_background_anchor_year"] = 2025
    resolved["general_background_role"] = "ordinary_ecoinvent"
    resolved["selected_background_member"] = "remind_ssp2_npi"
    fate, closure = mismanaged_result_tables(
        ProviderMetadata(),
        resolved,
        foreground_scenario="constructed_bioplastics",
        background_scenario="remind_ssp2_npi",
    )
    assert closure.constituent.nunique() == 10
    assert fsum(closure.parent_mismanaged_mass_kg) == pytest.approx(0.2e9)
    assert fsum(fate.final_provider_demand_amount_kg) == pytest.approx(0.2e9)
    np.testing.assert_allclose(closure.allocation_residual_kg, 0, atol=1e-7)
    np.testing.assert_allclose(fate.closure_residual_kg, 0, atol=1e-7)
