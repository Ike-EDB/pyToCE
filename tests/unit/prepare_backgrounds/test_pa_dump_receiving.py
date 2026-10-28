"""Check complete unmanaged PA receiving with real cohorts and five source sites."""

import json
from collections import defaultdict
from dataclasses import asdict, replace
from math import exp, fsum, log

import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from pa_dump_receiving_fixtures import (
    DUMP_CLASSES,
    GROUND,
    LAND,
    LAND_ROWS,
    LATER_AIR,
    LATER_GROUND,
    RURAL,
    SOIL,
    WASTE,
    dump_source_boundary,
)

from pytoce.prepare_backgrounds.preparation.providers.pa_dump_receiving import (
    PaDumpReceivingReference,
    admit_pa_dump_receiving_reference,
    dump_reference_from_payload,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingFlow,
)
from pytoce.run_lca.calculation.burdens import build_burden_demands
from pytoce.run_lca.calculation.custody import trace_carbon_custody
from pytoce.run_lca.calculation.pa_dump_receiving import receive_pa_dump

_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")


@pytest.fixture(scope="module")
def dump_reference(tmp_path_factory: pytest.TempPathFactory) -> PaDumpReceivingReference:
    """Admit once through the real original-only boundary, then close its project."""
    with dump_source_boundary(tmp_path_factory.mktemp("dump_native")) as context:
        reference = admit_pa_dump_receiving_reference(context)
        payload = json.loads(json.dumps(asdict(reference), allow_nan=False))
    detached = dump_reference_from_payload(payload)
    assert detached == reference
    return detached


def test_dump_admission_keeps_original_climates_and_land_basis(
    dump_reference: PaDumpReceivingReference,
) -> None:
    """Use original dump calibration and the actual PA mixture."""
    assert len(dump_reference.sites) == 5
    for i, (site, (name, _, precipitation)) in enumerate(
        zip(dump_reference.sites, DUMP_CLASSES, strict=True)
    ):
        assert site.infiltration_class == name
        assert site.placement.activity.key == ("dump_original", f"site_{i}")
        assert site.placement.activity.reference_product == "waste polyurethane"
        assert site.placement.activity.unit == "kilogram"
        assert site.placement.activity.location == "GLO"
        assert site.placement.production_amount == -(i + 2.0)
        assert site.climate.mean_annual_temperature_c == 8
        assert site.climate.mean_annual_precipitation_mm == precipitation
        assert site.climate.methane_correction == pytest.approx(0.7056571428571429)
        assert site.upward_leachate == (i == 0)
        assert len(site.placement.rows) == 3
        for row, (flow_name, unit, per_kg) in zip(site.placement.rows, LAND_ROWS, strict=True):
            assert isinstance(row.endpoint, PaReceivingFlow)
            assert (row.endpoint.name, row.endpoint.unit, row.endpoint.categories) == (
                flow_name,
                unit,
                LAND,
            )
            assert row.amount == pytest.approx(per_kg, rel=2e-14)
            assert row.metadata["loc"] == pytest.approx(log(per_kg) + 0.03)
            assert row.metadata["scale"] == 0.12
    endpoints = {(f.name, f.categories): f for f in dump_reference.biosphere}
    for c in (RURAL, LATER_AIR, GROUND, LATER_GROUND):
        assert endpoints["Water", c].unit == "cubic meter"
    # Reconstructed actual P must resolve even though every source PU fixture
    # omits P. The compartments distinguish elemental P from aqueous phosphate.
    for c in (RURAL, LATER_AIR, SOIL):
        assert endpoints["Phosphorus", c].unit == "kilogram"
    for c in (GROUND, LATER_GROUND):
        assert endpoints["Phosphate", c].unit == "kilogram"
    assert len(endpoints) == len(dump_reference.biosphere)


def test_actual_wet_mismanaged_pa_closes_all_five_dump_functions(
    dump_reference: PaDumpReceivingReference, carbon_authorities
) -> None:
    """Keep each actual mixed-origin reused cohort intact through complete receiving."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Fibres")
    # From 2020 the allowed biological/DAC feedstocks enter the real cohort.
    # Its second mechanical cycle precedes a partially uncollected retirement.
    state = replace(state, years=(2019, 2020, 2021, 2022))
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
    inputs.collection_rate[3, region] = 0.6
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    events = tuple(
        e
        for e in build_burden_demands(mfa, carbon, trace).pa_disposal_events
        if e.demand_type == "mismanaged_open_dump_treatment"
    )
    assert len(events) == 2
    assert {e.constituent for e in events} == {"polyamide_6_resin", "polyamide_66_resin"}
    for event in events:
        principal = fsum(event.principal_by_origin_kg)
        assert all(p > 0 for p in event.principal_by_origin_kg)
        assert event.total_finish_dose_kg > principal > 0
        assert event.water_kg > 0
        grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
        finish = _pa_finish_reference(grade)
        expected_wet = (
            principal
            + event.total_finish_dose_kg * fsum(finish.retained.compounds_kg.values())
            + event.water_kg
        )
        for site in dump_reference.sites:
            actual = receive_pa_dump(
                constituent=event.constituent,
                principal_by_origin_kg=dict(
                    zip(_ORIGINS, event.principal_by_origin_kg, strict=True)
                ),
                total_finish_dose_kg=event.total_finish_dose_kg,
                water_kg=event.water_kg,
                infiltration_class=site.infiltration_class,
                reference=dump_reference,
            )
            load, physical = actual.load, actual.discharges
            assert load.wet_mass_kg == pytest.approx(expected_wet, rel=2e-14)
            amounts = defaultdict(float)
            for row in actual.amounts:
                assert row.source == site.placement.activity
                assert row.function == "pa_open_dump"
                assert isinstance(row.endpoint, PaReceivingFlow)
                assert row.amount >= 0
                amounts[row.endpoint.name, row.endpoint.categories] += row.amount
                if row.endpoint.categories == LAND:
                    assert row.metadata["loc"] == pytest.approx(log(row.amount) + 0.03)
                    assert row.metadata["scale"] == 0.12
                else:
                    assert row.metadata == {"uncertainty type": 0, "loc": row.amount}
            for name, _, per_kg in LAND_ROWS:
                assert amounts[name, LAND] == pytest.approx(per_kg * expected_wet)
            assert amounts["Waste mass, total, placed in landfill", WASTE] == pytest.approx(
                expected_wet
            )
            assert amounts["Organic carbon, placed in landfill", WASTE] == load.elements_kg["C"]
            # Source NOx and the complementary N2 partition the
            # dump's nitrogen pool.
            nox = load.phases["N"].gas_kg * 0.00852882427542164
            assert amounts["Nitrogen oxides", RURAL] == pytest.approx(nox, rel=2e-14)
            assert physical.nitrogen_as_nox_kg == pytest.approx(nox * 14 / 46)
            assert physical.dinitrogen_kg + physical.nitrogen_as_nox_kg == pytest.approx(
                load.phases["N"].gas_kg
            )
            assert amounts["Sulfur dioxide", RURAL] * 32 / 64 == pytest.approx(
                load.phases["S"].gas_kg
            )
            assert not any(row.endpoint.name == "Ammonia" for row in actual.amounts)
            for phase in load.phases.values():
                assert fsum(
                    (
                        phase.gas_kg,
                        phase.short_leachate_kg,
                        phase.later_leachate_kg,
                        phase.stored_kg,
                    )
                ) == pytest.approx(phase.input_kg, rel=2e-14)
            # Gas mapping retains each principal origin. Auxiliary carbon keeps its own origin, and
            # the synthetic donor methane value is excluded.
            principal_c = [0.0, 0.0, 0.0]
            emitted_c = 0.0
            for row in actual.amounts:
                if row.endpoint.name.startswith(("Carbon dioxide,", "Methane,")):
                    c = row.amount * (
                        12 / 44 if row.endpoint.name.startswith("Carbon dioxide,") else 12 / 16
                    )
                    emitted_c += c
                    assert row.principal_carbon_origin_fractions is not None
                    for i, share in enumerate(row.principal_carbon_origin_fractions):
                        principal_c[i] += c * share
            assert emitted_c == pytest.approx(load.phases["C"].gas_kg, rel=2e-14)
            for origin, c in zip(_ORIGINS, principal_c, strict=True):
                assert c == pytest.approx(
                    fsum(
                        p.phases.gas_kg for p in load.carbon if p.principal and p.origin == origin
                    ),
                    rel=2e-14,
                )
            # Evaluate source climate and the complete moisture return.
            lp = 60 * (
                1 - exp(-29.372386439548432 * site.climate.mean_annual_precipitation_mm / 1000)
            )
            lt = 60 * (1 - exp(-0.3 * 23))
            alpha = log(1 - lp / 220) * log(1 - lt / 220) / log(1 - 60 / 220) ** 2
            short_w = event.water_kg * 0.01 * alpha
            water_destinations = (
                (RURAL, LATER_AIR) if site.upward_leachate else (GROUND, LATER_GROUND)
            )
            assert amounts["Water", water_destinations[0]] == pytest.approx(
                short_w / 1000, rel=2e-14
            )
            assert amounts["Water", water_destinations[1]] == pytest.approx(
                (event.water_kg - short_w) / 1000, rel=2e-14
            )
            assert physical.short_water_kg + physical.later_water_kg == event.water_kg
            energy = load.gas.decomposition_trace
            assert amounts["Heat, waste", SOIL] == pytest.approx(
                energy["gas_allocated_hhv_mj"]
                + energy["short_leachate_hhv_mj"]
                - energy["raw_gas_hhv_mj"],
                rel=2e-14,
            )
            assert amounts["Heat, waste", LATER_GROUND] == energy["later_leachate_hhv_mj"]
            assert fsum(
                (
                    amounts["Heat, waste", SOIL],
                    amounts["Heat, waste", LATER_GROUND],
                    energy["raw_gas_hhv_mj"],
                )
            ) == pytest.approx(energy["input_hhv_mj"], rel=2e-14)
            if site.upward_leachate:
                assert ("Phosphate", GROUND) not in amounts
                for phase, compartment in (
                    ("short_leachate_kg", RURAL),
                    ("later_leachate_kg", LATER_AIR),
                ):
                    elements = {e: 0.1 * getattr(p, phase) for e, p in load.phases.items()}
                    assert amounts["Phosphorus", compartment] == pytest.approx(elements["P"])
                    assert amounts["Nitrate", compartment] * 14 / 62 == pytest.approx(elements["N"])
                    assert amounts["Sulfate", compartment] * 32 / 96 == pytest.approx(elements["S"])
                    na_gas = load.phases["Na"].gas_kg if compartment == RURAL else 0
                    assert amounts["Sodium I", compartment] == pytest.approx(
                        elements["Na"] + na_gas
                    )
                    total_pm = (
                        fsum(elements[e] for e in ("C", "H", "O", "P", "Na"))
                        + elements["N"] * 62 / 14
                        + elements["S"] * 96 / 32
                    )
                    for name, share in (
                        ("Particulate Matter, > 10 um", 0.5),
                        ("Particulate Matter, > 2.5 um and < 10um", 0.3),
                        ("Particulate Matter, < 2.5 um", 0.2),
                    ):
                        assert amounts[name, compartment] == pytest.approx(
                            total_pm * share, rel=2e-14
                        )
                for element, p in load.phases.items():
                    assert physical.short_air_elements_kg[element] + physical.later_air_elements_kg[
                        element
                    ] + physical.soil_elements_kg[element] == pytest.approx(
                        p.short_leachate_kg + p.later_leachate_kg, rel=2e-14
                    )
                for element, name in (
                    ("C", "Carbon"),
                    ("N", "Nitrogen"),
                    ("S", "Sulfur"),
                    ("P", "Phosphorus"),
                    ("Na", "Sodium I"),
                ):
                    assert amounts[name, SOIL] == physical.soil_elements_kg[element]
            else:
                assert ("Particulate Matter, < 2.5 um", RURAL) not in amounts
                assert physical.soil_elements_kg == {}
                for attr, compartment in (
                    ("short_leachate_kg", GROUND),
                    ("later_leachate_kg", LATER_GROUND),
                ):
                    expected = {e: getattr(p, attr) for e, p in load.phases.items()}
                    assert amounts["Phosphate", compartment] * 31 / 95 == pytest.approx(
                        expected["P"]
                    )
                    assert amounts["Sodium I", compartment] == expected["Na"]
                    assert amounts["TOC, Total Organic Carbon", compartment] == expected["C"]
                    assert amounts["Sulfate", compartment] / 3 + amounts[
                        "Hydrogen sulfide", compartment
                    ] * 32 / 34 == pytest.approx(expected["S"])
                    assert fsum(
                        amounts[n, compartment] / factor
                        for n, factor in (
                            ("Ammonium", 18 / 14),
                            ("Nitrogen, organic bound", 1),
                            ("Nitrite", 46 / 14),
                            ("Nitrate", 62 / 14),
                        )
                    ) == pytest.approx(expected["N"])


@pytest.mark.parametrize("grade", ("pa6", "pa66"))
@pytest.mark.parametrize("principal,dose", ((1.0, 0.0), (0.0, 1.0)))
def test_dump_endpoints_preserve_carbon_and_nonnegative_phase_energy(
    dump_reference: PaDumpReceivingReference, grade: str, principal: float, dose: float
) -> None:
    """Both actual component endpoints bound every nonnegative principal/finish mixture."""
    for site in dump_reference.sites:
        result = receive_pa_dump(
            constituent={"pa6": "polyamide_6_resin", "pa66": "polyamide_66_resin"}[grade],
            principal_by_origin_kg={"atmospheric_dac": principal},
            total_finish_dose_kg=dose,
            water_kg=0.0,
            infiltration_class=site.infiltration_class,
            reference=dump_reference,
        )
        assert all(row.amount >= 0 for row in result.amounts)
        energy = result.load.gas.decomposition_trace
        assert result.discharges.emissions["Heat, waste", SOIL] >= 0
        assert result.discharges.emissions["Heat, waste", LATER_GROUND] >= 0
        assert energy["stored_hhv_mj"] == 0
        assert fsum(
            (
                result.discharges.emissions["Heat, waste", SOIL],
                result.discharges.emissions["Heat, waste", LATER_GROUND],
                energy["raw_gas_hhv_mj"],
            )
        ) == pytest.approx(energy["input_hhv_mj"], rel=2e-14)
        assert result.discharges.short_water_kg == result.discharges.later_water_kg == 0
