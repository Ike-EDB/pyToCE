"""Connect actual PA sludge, complete markets and original/dated ash receiving."""

import json
from dataclasses import replace
from math import isnan, log
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from pa_landfill_receiving_fixtures import landfill_receiving_boundary
from pa_water_receiving_fixtures import water_receiving_boundary

from pytoce.prepare_backgrounds.preparation.providers.ash_treatment import (
    _PA_ASH_RELEASES,
    _ash_dataset,
    _ash_market_dataset,
    _AshReleaseProfile,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import provider_service_code
from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _CH_WASTEWATER_PROFILE,
    _ROW_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_treatment import (
    _sludge_market_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
    _pa_recycling_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_water import (
    _pa_finish_water_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_load import (
    _pa_landfill_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_receiving_sources import (
    admit_pa_landfill_receiving_reference,
    admit_pa_landfill_receiving_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    reference_from_payload,
    reference_to_payload,
    sources_from_payload,
    sources_to_payload,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    admit_pa_water_receiving_reference,
    admit_pa_water_receiving_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_incineration import (
    _sludge_dataset,
    _sludge_source,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_incineration_load import (
    calculate_sludge_incineration,
)
from pytoce.prepare_backgrounds.preparation.providers.swiss_sludge_incineration import (
    swiss_sludge_dataset,
    swiss_sludge_load,
)
from pytoce.run_lca.calculation.pa_receiving import PaReceivingAmount
from pytoce.run_lca.calculation.pa_sludge_receiving import receive_pa_incinerated_sludge
from pytoce.run_lca.calculation.pa_water_receiving import PaWaterReceiving
from tests.pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources


def _actual_native_rows(
    datasets: list[tuple[dict[str, Any], float]], children: set[tuple[str, str]]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Expand the six negative-reference functions with internal links handled separately."""
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for dataset, quantity in datasets:
        production = next(row for row in dataset["exchanges"] if row["type"] == "production")
        factor = quantity / -production["amount"]
        for row in dataset["exchanges"]:
            if row["type"] == "production" or row["input"] in children:
                continue
            actual = dict(row, amount=row["amount"] * factor)
            if "loc" in row:
                actual["loc"] = (
                    row["loc"] + log(factor)
                    if row.get("uncertainty type") == 2
                    else row["loc"] * factor
                )
            for field in ("minimum", "maximum"):
                if field in row:
                    actual[field] = row[field] * factor
            result.setdefault(row["input"], []).append(actual)
    return result


@pytest.mark.parametrize(
    "grade,receiving,future,inlet",
    [
        (grade, receiving, future, "manufacturing")
        for grade in ("pa6", "pa66")
        for receiving in ("swiss", "sanitary")
        for future in (False, True)
    ]
    + [("pa6", "sanitary", False, "landfill")]
    + [
        (grade, "swiss", future, "recycling")
        for grade in ("pa6", "pa66")
        for future in (False, True)
    ],
)
def test_actual_pa_sludge_preserves_complete_native_receiving_and_phosphorus(
    tmp_path: Path, future: bool, receiving: Literal["swiss", "sanitary"], grade: str, inlet: str
) -> None:
    """Keep all six functions, one collection share and separate actual-P ash pools.

    Manufacturing PA6 water has three principal origins; PA66 manufacturing
    water and both recycling waters contain auxiliary carbon only. The PA6
    landfill inlet has only fossil principal carbon. These synthetic native
    graphs retain individual ash/site rows and uncertainty.
    """
    reference = pa_receiving_reference()
    sources = pa_receiving_sources(2050 if future else 2025, background_scenario="member")
    finish = _pa_finish_reference(grade)
    leachate = _pa_finish_water_load(
        finish.collected,
        caprolactam_by_origin_kg={
            "fossil": finish.collected_caprolactam_kg * 0.2,
            "biogenic": finish.collected_caprolactam_kg * 0.3,
            "atmospheric_dac": finish.collected_caprolactam_kg * 0.5,
        },
    )
    incoming_water_kg = finish.collected_water_kg
    volume = finish.collected_solution_m3
    if inlet == "recycling":
        original = reference.recycling
        recycling = _pa_recycling_reference(
            grade,
            finish,
            principal_kg=1.0,
            embedded_dose_kg=0.5,
            recovery_fraction=0.8,
            wash_water_per_dry_kg=original.wash_water_kg_per_reference_kg
            / original.dry_feed_kg_per_reference_kg,
        )
        leachate = _pa_finish_water_load(recycling.removed_finish, caprolactam_by_origin_kg={})
        incoming_water_kg = recycling.collected_water_kg
        volume = recycling.collected_solution_m3
    if inlet == "landfill":
        # The accepted PA family admits zero retained finish. Its selected RoW
        # site reference is 16.74358 C / 1250.84109 mm/y and full methane
        # correction. This real pure handoff has no P/S or biogenic/DAC group.
        load = _pa_landfill_load(
            finish,
            principal_by_origin_kg={"fossil": 1.0},
            finish_dose_kg=0.0,
            water_kg=0.04,
            site=_LandfillSite(16.743579766537, 1250.84108527132, 1.0),
        )
        leachate = load.leachate
        # A bounded 0.9-liter hydraulic inlet keeps volume separate from the
        # incoming waste moisture. Site hydraulic scaling has its own tests.
        incoming_water_kg = 0.9
    boundary = water_receiving_boundary if receiving == "swiss" else landfill_receiving_boundary
    with boundary(tmp_path, future=future) as (_, context):
        if receiving == "swiss":
            selected = admit_pa_water_receiving_reference(context)
            dated = admit_pa_water_receiving_sources(context, selected)
            reference = replace(reference, water=selected)
            sources = replace(sources, water=dated)
            profile = _CH_WASTEWATER_PROFILE
            ids = {
                role: f"pa_phosphate_{role}_ch"
                for role in (
                    "sludge_incineration_market",
                    "sludge_incineration",
                    "bottom_ash_market",
                    "residual_ash_market",
                    "bottom_ash_treatment",
                    "residual_ash_treatment",
                )
            }
        else:
            selected = admit_pa_landfill_receiving_reference(context)
            dated = admit_pa_landfill_receiving_sources(context, selected)
            reference = replace(reference, landfill=selected)
            sources = replace(sources, landfill=dated)
            profile = _ROW_WASTEWATER_PROFILE
            ids = {
                role: f"sbr_filled_fossil_{role}"
                for role in (
                    "sludge_incineration_market",
                    "sludge_incineration",
                    "bottom_ash_market",
                    "residual_ash_market",
                    "bottom_ash_treatment",
                    "residual_ash_treatment",
                )
            }
        wastewater = _wastewater_load(
            leachate,
            volume_m3=volume,
            incoming_water_kg=incoming_water_kg,
            evaporation_fraction=selected.evaporation_fraction,
            profile=profile,
        )
        # The selected collector's own negative child coefficient is its real
        # treatment share. A second application would alter every expected row.
        treated = -sum(
            row.amount
            for row in dated.collection.rows
            if row.endpoint.key == dated.wastewater.activity.key
        )
        water = PaWaterReceiving((), wastewater, treated)
        services = {
            str(service["provider_service_id"]): service
            for service in cast(list[dict[str, object]], context.catalogue["services"])
        }
        furnace = services[ids["sludge_incineration"]]
        construction = cast(dict[str, object], furnace["construction"])
        if receiving == "swiss":
            physical = swiss_sludge_load(context, construction, wastewater)
            # Grade/formulation vectors scale with principal mass and retain their specified zero
            # metal/lime demands.
            assert physical.duties["CaO"] == physical.duties["FeCl3"] == 0
            assert physical.allocands["lime_before_purchase"] < 0
            native_furnace = swiss_sludge_dataset(
                context, furnace, physical, context.provider_database
            )
        else:
            physical = calculate_sludge_incineration(
                _sludge_source(context, construction),
                wastewater,
                "pa_actual_sanitary_sludge",
                source_carbon_origins=("fossil",),
                additional_apc_water_flows=("Calcium II",),
                phosphorus_phase_reference=True,
                # Selected plastic-leachate slag: 11.5628% water / 88.4372% dry.
                bottom_ash_water_to_dry_ratio=0.115628 / 0.884372,
            )
            # Compare the complete TPS/SBR supported vector, then add PA's separately admitted
            # phosphorus outputs.
            non_p = replace(
                physical,
                emissions={
                    key: value
                    for key, value in physical.emissions.items()
                    if key[0] not in {"Phosphorus", "Phosphate"}
                },
            )
            native_furnace = _sludge_dataset(context, furnace, non_p, context.provider_database)
        datasets = [
            (
                _sludge_market_dataset(
                    context,
                    services[ids["sludge_incineration_market"]],
                    wastewater,
                    context.provider_database,
                ),
                treated * physical.wet_mass_kg,
            ),
            (native_furnace, treated * physical.wet_mass_kg),
        ]
        for fraction, ash in (("bottom", physical.bottom_ash), ("residual", physical.residual_ash)):
            datasets.extend(
                (
                    (
                        _ash_market_dataset(
                            context,
                            services[ids[f"{fraction}_ash_market"]],
                            ash,
                            context.provider_database,
                        ),
                        treated * ash.treatment_mass_kg,
                    ),
                    (
                        _ash_dataset(
                            context,
                            services[ids[f"{fraction}_ash_treatment"]],
                            ash,
                            context.provider_database,
                            release_profile=_AshReleaseProfile(())
                            if receiving == "swiss"
                            else _PA_ASH_RELEASES[fraction],
                        ),
                        treated * ash.treatment_mass_kg,
                    ),
                )
            )
        children = {
            (context.provider_database, provider_service_code(service_id))
            for service_id in ids.values()
        }
        expected = _actual_native_rows(cast(list[tuple[dict[str, Any], float]], datasets), children)
        if receiving == "sanitary":
            # Wastewater2021 sludgeIncin G26:L26 gives air/direct-scrubber-water fractions. The
            # remaining phosphorus enters ash.
            p = physical.elements_kg["P"]
            for name, categories, amount in (
                ("Phosphorus", ("air", "urban air close to ground"), p * 0.00101624265046688),
                ("Phosphate", ("water", "surface water"), p * 0.000110461157659443 * 95 / 31),
            ):
                assert physical.emissions[name, "kilogram", categories] == pytest.approx(amount)
                if amount == 0:
                    continue
                endpoint = next(
                    flow
                    for flow in dated.biosphere
                    if flow.name == name and flow.categories == categories
                )
                expected.setdefault(endpoint.key, []).append(
                    {
                        "input": endpoint.key,
                        "amount": treated * amount,
                        "loc": treated * amount,
                        "uncertainty type": 0,
                    }
                )
        reference = reference_from_payload(
            json.loads(json.dumps(reference_to_payload(reference), allow_nan=False))
        )
        sources = sources_from_payload(
            json.loads(json.dumps(sources_to_payload(sources), allow_nan=False))
        )
    result = receive_pa_incinerated_sludge(water, reference, sources, receiving=receiving)
    assert replace(result.incineration, emissions={}) == replace(physical, emissions={})
    # Finite PA admission omits donor-only metal flows. The full native source
    # yields explicit zero quantities for those absent PA elements; both binders
    # omit them. All actual releases and every other field stay equal.
    assert {key: value for key, value in result.incineration.emissions.items() if value != 0} == {
        key: value for key, value in physical.emissions.items() if value != 0
    }
    actual: dict[tuple[str, str], list[PaReceivingAmount]] = {}
    for row in result.amounts:
        actual.setdefault(row.endpoint.key, []).append(row)
    assert actual.keys() == expected.keys(), {
        "unexpected": actual.keys() - expected.keys(),
        "missing": expected.keys() - actual.keys(),
    }
    for key, rows in actual.items():
        assert len(rows) == len(expected[key])
        for target, native in zip(
            sorted(rows, key=lambda row: row.amount),
            sorted(expected[key], key=lambda row: row["amount"]),
            strict=True,
        ):
            assert target.amount == pytest.approx(native["amount"], rel=2e-12, abs=1e-30)
            for field in ("loc", "scale", "shape", "minimum", "maximum"):
                if field in native and not isnan(native[field]):
                    assert target.metadata[field] == pytest.approx(native[field], rel=2e-12)
            for field in (
                "uncertainty type",
                "negative",
                "allocation",
                "formula",
                "activity",
                "flow",
                "swiss_hydroxide_cutoff",
            ):
                if field in native:
                    assert target.metadata[field] == native[field]
            if "principal_carbon_origin_fractions" in native:
                assert target.principal_carbon_origin_fractions == pytest.approx(
                    native["principal_carbon_origin_fractions"]
                )
            elif target.function in {
                f"pa_{receiving}_bottom_ash",
                f"pa_{receiving}_residual_ash",
            } and target.endpoint.name in {
                "DOC, Dissolved Organic Carbon",
                "TOC, Total Organic Carbon",
                "COD, Chemical Oxygen Demand",
                "BOD5, Biological Oxygen Demand",
                "Organic carbon, placed in landfill",
            }:
                # The furnace assigns carbon pools to ash, including fresh PAM. Principal-origin
                # fractions use actual ash-carbon quantities.
                ash = (
                    physical.bottom_ash
                    if target.function.endswith("bottom_ash")
                    else physical.residual_ash
                )
                expected_origins = tuple(
                    sum(
                        pool.carbon_kg
                        for pool in ash.carbon
                        if pool.principal and pool.origin == origin
                    )
                    / sum(pool.carbon_kg for pool in ash.carbon)
                    for origin in ("fossil", "biogenic", "atmospheric_dac")
                )
                assert target.principal_carbon_origin_fractions == pytest.approx(expected_origins)
                assert sum(expected_origins) < 1.0
            else:
                assert target.principal_carbon_origin_fractions is None
    if receiving == "sanitary":
        for ash, release, cumulative in (
            (physical.bottom_ash, result.bottom_ash, 0.013222518692302985),
            (physical.residual_ash, result.residual_ash, 0.22390423222123912),
        ):
            # These kg P precede the collector fraction. Native discharge rows include that
            # fraction.
            p = ash.waste_elements_kg["P"]
            assert release.retained_profile_elements_kg["P"] == pytest.approx(p * (1 - cumulative))
            released = sum(
                amount * 31 / 95 for name, _, amount in release.additions if name == "Phosphate"
            )
            assert released + release.retained_profile_elements_kg["P"] == pytest.approx(p)
    else:
        assert result.bottom_ash.retained_profile_elements_kg == {}
        assert result.residual_ash.retained_profile_elements_kg == {}
