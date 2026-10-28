"""Carry actual unsplit PA lifecycle quantities from MFA to receiving consumers."""

from math import fsum
from pathlib import Path

import pandas as pd
import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from infiltration_fixtures import resolved_infiltration_provider
from pa_custody_fixtures import two_cycle_pa_mfa

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.run_lca.calculation.burdens import build_burden_demands
from pytoce.run_lca.calculation.custody import (
    CarbonCustodyTrace,
    prepare_carbon_cohort_weights,
    trace_carbon_custody,
)
from pytoce.run_lca.calculation.foreground import (
    compile_foreground_activations,
    compile_optimization_activation_context,
    compile_optimization_activations_from_history,
)
from pytoce.run_lca.calculation.providers import MaterialCarbon, ProviderRoutes
from pytoce.run_mfa.model.mfa import MFAEvaluation


@pytest.fixture(scope="module")
def mixed_pa_cycles(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
) -> tuple[MFAEvaluation, CarbonCustodyTrace]:
    """Use two cycles, switching 2020 new principal to 40/35/25 origins."""
    carbon, recipe = carbon_authorities
    mfa = two_cycle_pa_mfa(recipe, biogenic_share=0.35, dac_share=0.25)
    return mfa, trace_carbon_custody(mfa, carbon, recipe)


def test_spinning_events_keep_actual_two_grade_mixtures_before_origin_splitting(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
    mixed_pa_cycles: tuple[MFAEvaluation, CarbonCustodyTrace],
) -> None:
    """Forward kg and dose kg once, retaining all three principal origins."""
    carbon, _ = carbon_authorities
    mfa, trace = mixed_pa_cycles
    demands = build_burden_demands(mfa, carbon, trace)
    events = {
        (event.year, event.region, event.constituent): event for event in demands.pa_spinning_events
    }
    assert len(events) == len(demands.pa_spinning_events) == 4
    assert tuple(events) == tuple(sorted(events))
    assert set(events) == {
        (year, "EUR", constituent)
        for year in (2019, 2020)
        for constituent in ("polyamide_6_resin", "polyamide_66_resin")
    }
    origins = ("fossil", "biogenic", "atmospheric_dac")
    for (year, region, constituent), event in events.items():
        realization = mfa.material_realization
        quantities = (
            realization.loc[
                realization.year.eq(year)
                & realization.region.eq(region)
                & realization["product"].eq("polymer")
                & (
                    realization.source_constituent.eq(constituent)
                    | realization.destination_constituent.eq(constituent)
                )
            ]
            .set_index("flow_name")
            .amount
        )
        # Source tables report Mt principal; receiving converts to actual kg.
        assert event.finished_principal_kg == pytest.approx(
            float(quantities["manufacturing_finished_output"]) * 1e9, rel=2e-13
        )
        assert event.gross_principal_kg == pytest.approx(
            float(quantities["manufacturing_principal_input"]) * 1e9, rel=2e-13
        )
        assert event.recovered_principal_kg == pytest.approx(
            float(quantities["mechanical_resin"]) * 1e9, rel=2e-13
        )
        key = year, region, "Fibres", constituent, "all"
        assert event.embedded_finish_dose_kg == pytest.approx(
            trace.pa_finish_amounts[(*key, "reclmech => processing")] * 1e9, rel=2e-13
        )
        assert event.maintenance_finish_dose_kg == pytest.approx(
            trace.pa_finish_amounts[(*key, "processing => spinning_equipment_deposits")] * 1e9,
            rel=2e-13,
        )
        assert 0 < event.maintenance_finish_dose_kg < event.embedded_finish_dose_kg
        rows = trace.calculation.loc[
            trace.calculation.year.eq(year)
            & trace.calculation.region.eq(region)
            & trace.calculation.material.eq("Fibres")
            & trace.calculation.constituent.eq(constituent)
        ]
        for flow, actual, carbon_fraction in (
            (
                "processing => finished_principal",
                event.finished_principal_carbon_by_origin_kg,
                1.0,
            ),
            (
                "processing => spinning_equipment_deposits",
                event.maintenance_principal_by_origin_kg,
                72 / 113,
            ),
            ("processing => caprolactam_air", event.caprolactam_air_by_origin_kg, 72 / 113),
            (
                "processing => caprolactam_collected_water",
                event.caprolactam_water_by_origin_kg,
                72 / 113,
            ),
        ):
            source = (
                rows.loc[rows.source_flow_name.eq(flow)]
                .groupby("ultimate_carbon_origin")
                .carbon_amount_mt.agg(fsum)
                .reindex(origins, fill_value=0.0)
            )
            # PA6 and PA66 use the same admitted nominal principal C fraction.
            # Destruction consumes the already separated exit; no
            # second equipment-loss fraction or newly applied finish belongs here.
            expected = tuple(float(value) * 1e9 / carbon_fraction for value in source)
            assert actual == pytest.approx(expected, rel=2e-13, abs=0)
            assert all(value >= 0 for value in actual)
        assert event.maintenance_principal_by_origin_kg[0] > 0
        if year == 2019:
            assert event.maintenance_principal_by_origin_kg[1:] == (0.0, 0.0)
        else:
            assert all(value > 0 for value in event.maintenance_principal_by_origin_kg)
        if constituent == "polyamide_66_resin":
            assert event.caprolactam_air_by_origin_kg == (0.0, 0.0, 0.0)
            assert event.caprolactam_water_by_origin_kg == (0.0, 0.0, 0.0)
        else:
            assert event.caprolactam_air_by_origin_kg[0] > 0
            assert event.caprolactam_water_by_origin_kg[0] > 0
            if year == 2020:
                assert all(value > 0 for value in event.caprolactam_air_by_origin_kg)
                assert all(value > 0 for value in event.caprolactam_water_by_origin_kg)
        assert fsum(event.finished_principal_carbon_by_origin_kg) == pytest.approx(
            event.finished_principal_kg * 72 / 113,
            rel=2e-13,
            abs=0,
        )
        if year == 2020:
            assert all(value > 0 for value in event.finished_principal_carbon_by_origin_kg)
    for constituent in ("polyamide_6_resin", "polyamide_66_resin"):
        assert (
            events[2020, "EUR", constituent].embedded_finish_dose_kg
            > events[2019, "EUR", constituent].embedded_finish_dose_kg
        )


def test_detailed_parent_and_history_compilation_keep_the_same_pa_events(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
    mixed_pa_cycles: tuple[MFAEvaluation, CarbonCustodyTrace],
    tmp_path: Path,
) -> None:
    """Preserve the same actual loads through ordinary and resumed candidate paths."""
    carbon, recipe = carbon_authorities
    mfa, trace = mixed_pa_cycles
    authority = resolved_infiltration_provider(tmp_path / "infiltration")
    routes = ProviderRoutes(
        authority.routes,
        identity=authority.identity,
        interface_database=authority.interface_database,
        material_carbon=carbon,
    )
    expected = build_burden_demands(mfa, carbon, trace)
    fixed = expected.provider_demands
    pa = fixed.material.eq("Fibres") & fixed.constituent.isin(
        ("polyamide_6_resin", "polyamide_66_resin")
    )
    retired = {
        "mechanical_recycling_operation",
        "incineration_treatment",
        "landfill_treatment",
        "mechanical_recycling_reject_incineration_treatment",
    }
    assert not fixed.loc[pa, "demand_type"].isin(retired).any()
    assert set(fixed.loc[pa, "demand_type"]) == {
        "virgin_resin_production",
        "processing_conversion",
    }
    # PP and PET fiber retain their fixed recycling and disposal functions.
    assert set(fixed.loc[~pa & fixed.demand_type.isin(retired), "constituent"]) == {
        "polypropylene",
        "polyethylene_terephthalate",
    }
    detailed = compile_foreground_activations(mfa, routes, recipe)
    parent = compile_optimization_activation_context(mfa, routes, recipe)
    resumed = compile_optimization_activations_from_history(
        mfa,
        routes,
        recipe,
        parent,
        cohort_weights_by_year=prepare_carbon_cohort_weights(
            mfa.cohort_state, mfa.cohort_material_identity
        ),
    )
    assert detailed.pa_spinning_events == parent.pa_spinning_events == expected.pa_spinning_events
    assert resumed.pa_spinning_events == expected.pa_spinning_events
    assert {event.year for event in expected.pa_spinning_events} == {2019, 2020}
    assert (
        detailed.pa_recycling_events == parent.pa_recycling_events == expected.pa_recycling_events
    )
    assert resumed.pa_recycling_events == expected.pa_recycling_events
    assert detailed.pa_disposal_events == parent.pa_disposal_events == expected.pa_disposal_events
    assert resumed.pa_disposal_events == expected.pa_disposal_events
    assert {event.year for event in expected.pa_disposal_events} == {2019, 2020, 2021}
    assert parent.carbon_history.through_year == 2019
    pd.testing.assert_frame_equal(resumed.activations, parent.activations)


def test_virgin_only_pa_events_have_no_recycled_or_inherited_load(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
) -> None:
    """A real first-use PA cohort has principal maintenance but no inherited finish."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Fibres")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    state.annual_inflow[1, region, good] = 1
    state.stock_by_cohort[1:, 1, region, good] = 1
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    demands = build_burden_demands(mfa, carbon, trace)
    events = demands.pa_spinning_events
    assert demands.pa_recycling_events == ()
    assert demands.pa_disposal_events == ()
    assert len(events) == 2
    for event in events:
        assert event.year == 2019
        assert event.region == "EUR"
        assert event.gross_principal_kg > event.finished_principal_kg > 0
        assert event.recovered_principal_kg == 0
        assert event.embedded_finish_dose_kg == 0
        assert event.maintenance_finish_dose_kg == 0
        assert event.maintenance_principal_by_origin_kg[0] > 0
        assert event.maintenance_principal_by_origin_kg[1:] == (0.0, 0.0)
        assert event.finished_principal_carbon_by_origin_kg == pytest.approx(
            (event.finished_principal_kg * 72 / 113, 0.0, 0.0), rel=2e-13, abs=0
        )


def _principal_exit_kg(
    trace: CarbonCustodyTrace, year: int, constituent: str, flow: str
) -> tuple[float, float, float]:
    """Convert EUR PA constituent carbon to kilograms of principal polymer by origin."""
    rows = trace.calculation
    carbon = (
        rows.loc[
            rows.year.eq(year)
            & rows.region.eq("EUR")
            & rows.material.eq("Fibres")
            & rows.constituent.eq(constituent)
            & rows.source_flow_name.eq(flow)
        ]
        .groupby("ultimate_carbon_origin")
        .carbon_amount_mt.agg(fsum)
    )
    # Constituent custody uses 72/113 kg C/kg principal polymer for both PA grades.
    # The custody table uses Mt C; receiving events use kg of principal polymer.
    return (
        float(carbon.get("fossil", 0.0)) * 1e9 * 113 / 72,
        float(carbon.get("biogenic", 0.0)) * 1e9 * 113 / 72,
        float(carbon.get("atmospheric_dac", 0.0)) * 1e9 * 113 / 72,
    )


def test_recycling_events_separate_inherited_input_from_washed_airborne_finish(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
    mixed_pa_cycles: tuple[MFAEvaluation, CarbonCustodyTrace],
) -> None:
    """Pass prewash inlet dose and actual postwash airborne dose to distinct duties."""
    carbon, _ = carbon_authorities
    mfa, trace = mixed_pa_cycles
    events = build_burden_demands(mfa, carbon, trace).pa_recycling_events
    assert len(events) == 4
    assert {(event.year, event.constituent) for event in events} == {
        (year, constituent)
        for year in (2019, 2020)
        for constituent in ("polyamide_6_resin", "polyamide_66_resin")
    }
    # Chain Yarn Table 2 gives paired postwash/prewash oil ratios, applied once to the designated
    # surface finish.
    remaining = {
        "polyamide_6_resin": (0.22 / 1.21 + 0.20 / 1.23 + 0.19 / 1.21) / 3,
        "polyamide_66_resin": (0.20 / 1.35 + 0.20 / 1.34 + 0.22 / 1.34) / 3,
    }
    for event in events:
        assert event.region == "EUR"
        key = event.year, event.region, "Fibres", event.constituent, "all"
        assert event.principal_by_origin_kg == pytest.approx(
            _principal_exit_kg(trace, event.year, event.constituent, "collected => reclmech"),
            rel=3e-13,
        )
        assert event.airborne_principal_by_origin_kg == pytest.approx(
            _principal_exit_kg(trace, event.year, event.constituent, "reclmech => airborne_loss"),
            rel=3e-13,
        )
        recovered = _principal_exit_kg(
            trace, event.year, event.constituent, "reclmech => processing"
        )
        assert event.recovered_principal_kg == pytest.approx(fsum(recovered), rel=3e-13)
        assert event.embedded_finish_dose_kg == pytest.approx(
            trace.pa_finish_amounts[(*key, "collected => reclmech")] * 1e9, rel=3e-13
        )
        principal = fsum(event.principal_by_origin_kg)
        airborne = fsum(event.airborne_principal_by_origin_kg)
        assert 0 < airborne < event.recovered_principal_kg < principal
        postwash = event.embedded_finish_dose_kg + remaining[event.constituent] * principal
        assert event.airborne_finish_dose_kg == pytest.approx(
            airborne / principal * postwash, rel=3e-13
        )
        assert event.airborne_finish_dose_kg == pytest.approx(
            trace.pa_finish_amounts[(*key, "reclmech => airborne_loss")] * 1e9, rel=3e-13
        )
        if event.year == 2019:
            assert event.embedded_finish_dose_kg == 0.0
        else:
            assert event.embedded_finish_dose_kg > 0.0


def test_disposal_events_keep_wet_retirements_distinct_from_dry_washed_rejects(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
    mixed_pa_cycles: tuple[MFAEvaluation, CarbonCustodyTrace],
) -> None:
    """Retain each outlet's origin and dose after the specified washing losses."""
    carbon, _ = carbon_authorities
    mfa, trace = mixed_pa_cycles
    events = build_burden_demands(mfa, carbon, trace).pa_disposal_events
    source_flows = {
        "incineration_treatment": "collected => incineration",
        "landfill_treatment": "collected => landfill",
        "mechanical_recycling_reject_incineration_treatment": "reclmech => incineration",
    }
    # The admitted PA mechanical source incinerates its managed rejects; it
    # has no landfilled reject stream to manufacture merely for test coverage.
    assert {event.demand_type for event in events} == set(source_flows)
    assert len({(e.year, e.region, e.constituent, e.demand_type) for e in events}) == len(events)
    regain = {"polyamide_6_resin": 0.044, "polyamide_66_resin": 0.043}
    for event in events:
        assert event.region == "EUR"
        flow = source_flows[event.demand_type]
        principal = _principal_exit_kg(trace, event.year, event.constituent, flow)
        assert event.principal_by_origin_kg == pytest.approx(principal, rel=3e-13)
        assert fsum(principal) > 0
        key = event.year, event.region, "Fibres", event.constituent, "all", flow
        embedded = trace.pa_finish_amounts[key] * 1e9
        if flow == "reclmech => incineration":
            assert event.water_kg == 0.0
            assert event.total_finish_dose_kg == pytest.approx(embedded, rel=3e-13)
        else:
            assert event.water_kg == pytest.approx(fsum(principal) * regain[event.constituent])
            assert event.total_finish_dose_kg == pytest.approx(fsum(principal) + embedded)
        if event.year == 2021:
            assert all(value > 0 for value in principal)
            assert embedded > 0


def test_mismanaged_events_allocate_the_complete_retired_mixture_once(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
) -> None:
    """Split mixed-origin principal mass, inherited/surface dose and water together."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Fibres")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    for time in (0, 1, 2):
        state.annual_inflow[time, region, good] = 1
        state.stock_by_cohort[time, time, region, good] = 1
        state.outflow_by_cohort[time + 1, time, region, good] = 1
    for time in (1, 2):
        inputs.mechanical_recycling_rate[time, region] = 0.5
        inputs.incineration_rate[time, region] = 0.25
        inputs.bio_production_rate[time, region] = 0.35
        inputs.daccu_production_rate[time, region] = 0.25
    # Collect 60% of the final mixed-origin cohort; its uncollected 40% has
    # the same origin, finish and principal-moisture composition.
    inputs.collection_rate[3, region] = 0.6
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    demands = build_burden_demands(mfa, carbon, trace)
    # The second collection now contains all three origins. The event keeps
    # them together instead of duplicating its whole receiving duty per origin.
    mixed_recycling = [event for event in demands.pa_recycling_events if event.year == 2020]
    assert len(mixed_recycling) == 2
    for event in mixed_recycling:
        assert all(value > 0 for value in event.principal_by_origin_kg)
        assert all(value > 0 for value in event.airborne_principal_by_origin_kg)
        assert event.principal_by_origin_kg == pytest.approx(
            _principal_exit_kg(trace, 2020, event.constituent, "collected => reclmech"),
            rel=3e-13,
        )
    unmanaged = {
        (e.constituent, e.demand_type): e
        for e in demands.pa_disposal_events
        if e.demand_type.startswith("mismanaged_")
    }
    assert len(unmanaged) == 4
    fixed_unmanaged = demands.provider_demands.loc[
        demands.provider_demands.demand_type.str.startswith("mismanaged_")
    ]
    assert set(fixed_unmanaged.constituent) == {
        "polypropylene",
        "polyethylene_terephthalate",
    }
    assert set(fixed_unmanaged.demand_type) == {
        "mismanaged_open_burning_treatment",
        "mismanaged_open_dump_treatment",
    }
    # Cottom 2024's admitted source-matched uncollected split is 16.9 Mt
    # burned and 18.7 Mt unburned out of 35.6 Mt; it is common to all the 12 world regions.
    for constituent, regain in (("polyamide_6_resin", 0.044), ("polyamide_66_resin", 0.043)):
        principal = _principal_exit_kg(trace, 2021, constituent, "eol => mismanaged")
        assert all(value > 0 for value in principal)
        embedded = (
            trace.pa_finish_amounts[
                (2021, "EUR", "Fibres", constituent, "all", "eol => mismanaged")
            ]
            * 1e9
        )
        assert embedded > 0
        pair = []
        for demand, share in (
            ("mismanaged_open_burning_treatment", 16.9 / 35.6),
            ("mismanaged_open_dump_treatment", 18.7 / 35.6),
        ):
            event = unmanaged[constituent, demand]
            pair.append(event)
            assert (event.year, event.region) == (2021, "EUR")
            assert event.principal_by_origin_kg == pytest.approx(
                tuple(p * share for p in principal)
            )
            assert event.total_finish_dose_kg == pytest.approx((fsum(principal) + embedded) * share)
            assert event.water_kg == pytest.approx(fsum(principal) * regain * share)
        assert tuple(
            fsum(e.principal_by_origin_kg[i] for e in pair) for i in range(3)
        ) == pytest.approx(principal)
        assert fsum(e.total_finish_dose_kg for e in pair) == pytest.approx(
            fsum(principal) + embedded
        )
        assert fsum(e.water_kg for e in pair) == pytest.approx(fsum(principal) * regain)
