"""Historical polymer quantities and carbon origins through road-material disposal."""

from math import fsum
from typing import cast

import pandas as pd
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from infiltration_fixtures import resolved_infiltration_provider

from pytoce.prepare_backgrounds.preparation.providers.route_authorities import (
    mismanaged_fate_authorities,
)
from pytoce.run_lca.calculation.custody import prepare_carbon_cohort_weights
from pytoce.run_lca.calculation.foreground import (
    OPTIMIZATION_ACTIVATION_COLUMNS,
    compile_foreground_activations,
    compile_optimization_activation_context,
    compile_optimization_activations_from_history,
    optimization_activation_delta,
)
from pytoce.run_lca.calculation.providers import ProviderRoutes
from pytoce.run_mfa.model.materials import load_material_composition


def test_road_disposal_keeps_historical_principal_in_ordinary_and_candidate_paths(
    carbon_authorities, tmp_path
):
    """Retain historical polymer carbon origins through ordinary and candidate disposal."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Others")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    for entry, amount in ((1, 1.0), (2, 2.0)):
        state.annual_inflow[entry, region, good] = amount
        state.stock_by_cohort[entry, entry, region, good] = amount
        state.outflow_by_cohort[entry + 1, entry, region, good] = amount
    # The 2019 cohort uses fossil polymer; the next combines three carbon origins.
    # A later supply from direct air capture cannot change either discarded cohort.
    inputs.bio_production_rate[2, region] = 0.25
    inputs.daccu_production_rate[2, region] = 0.25
    inputs.daccu_production_rate[3, region] = 1.0
    inputs.collection_rate[2:, region] = 0.8
    inputs.incineration_rate[2:, region] = 0.5
    mfa = evaluate_mfa(state, inputs, recipe)
    authority = resolved_infiltration_provider(tmp_path / "infiltration")
    routes = ProviderRoutes(
        authority.routes,
        identity=authority.identity,
        interface_database=authority.interface_database,
        material_carbon=carbon,
    )
    detailed = compile_foreground_activations(mfa, routes, recipe).activations
    road = detailed.loc[detailed.constituent.eq("road_marking_coatings")]
    constituent = next(
        row
        for row in load_material_composition().constituents
        if row.constituent == "road_marking_coatings"
    )
    fate = cast(pd.Series, mismanaged_fate_authorities().set_index("region").loc["EUR"])
    fates = {
        "incineration_treatment": (0.4, ("municipal_incineration",)),
        "landfill_treatment": (0.4, ("landfill",)),
        "mismanaged_open_burning_treatment": (
            0.2 * float(fate.open_burning_share),
            ("open_burning",),
        ),
        "mismanaged_open_dump_treatment": (
            0.2 * float(fate.open_dump_share),
            (
                "open_dump_hyperarid_infiltration_class_250mm",
                "open_dump_dry_infiltration_class_100mm",
                "open_dump_moist_infiltration_class_300mm",
                "open_dump_wet_infiltration_class_500mm",
                "open_dump_very_wet_infiltration_class_1000mm",
            ),
        ),
    }
    for year, cohort_mt, origins in (
        (2020, 1.0, {"fossil": 1.0}),
        (2021, 2.0, {"fossil": 0.5, "biogenic": 0.25, "atmospheric_dac": 0.25}),
    ):
        retired_kg = cohort_mt * float(constituent.share) * 1e9
        annual = road.loc[road.year.eq(year) & road.demand_type.isin(fates)]
        assert fsum(annual.amount) == pytest.approx(retired_kg)
        assert set(annual.demand_type) == set(fates)
        assert set(annual.ultimate_carbon_origin) == set(origins)
        assert set(annual.physical_event_year) == {year}
        assert set(annual.region) == {"EUR"}
        assert annual.throughput_multiplier.eq(1).all()
        # MFA and service demands use polymer kg with PMMA at nominal 0.6 kg C/kg. Treatment adds
        # auxiliaries and geological carbon, giving 4.56 kg wet road material on its own reference
        # basis.
        assert annual.mfa_carbon_amount_kg.to_numpy() == pytest.approx(
            0.6 * annual.amount.to_numpy()
        )
        for demand, (share, suffixes) in fates.items():
            selected = annual.loc[annual.demand_type.eq(demand)]
            assert share > 0
            assert fsum(selected.amount) == pytest.approx(retired_kg * share)
            for origin, fraction in origins.items():
                by_origin = selected.loc[selected.ultimate_carbon_origin.eq(origin)]
                assert set(by_origin.provider_service_id) == {
                    f"road_{origin}_{suffix}" for suffix in suffixes
                }
                assert fsum(by_origin.amount) == pytest.approx(retired_kg * share * fraction)

    lean = compile_optimization_activation_context(mfa, routes, recipe)
    expected = (
        detailed.groupby(list(OPTIMIZATION_ACTIVATION_COLUMNS[:-1]), observed=True, sort=True)
        .amount.agg(fsum)
        .reset_index()
    )
    pd.testing.assert_frame_equal(lean.activations, expected)
    resumed = compile_optimization_activations_from_history(
        mfa,
        routes,
        recipe,
        lean,
        cohort_weights_by_year=prepare_carbon_cohort_weights(
            mfa.cohort_state, mfa.cohort_material_identity
        ),
    ).activations
    pd.testing.assert_frame_equal(resumed, lean.activations)
    assert optimization_activation_delta(lean.activations, resumed).empty
