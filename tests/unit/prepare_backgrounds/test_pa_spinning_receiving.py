"""Connect actual MFA spinning events to complete receiving and one site water balance."""

from dataclasses import replace
from math import fsum
from pathlib import Path

import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from pa_custody_fixtures import two_cycle_pa_mfa
from pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources
from pa_water_receiving_fixtures import water_receiving_boundary

from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    admit_pa_water_receiving_reference,
    admit_pa_water_receiving_sources,
)
from pytoce.run_lca.calculation.burdens import build_burden_demands
from pytoce.run_lca.calculation.custody import trace_carbon_custody
from pytoce.run_lca.calculation.pa_spinning_receiving import receive_pa_spinning


@pytest.mark.parametrize("future", (False, True))
def test_actual_mfa_spinning_has_one_complete_receiving_and_site_allocation(
    tmp_path: Path, carbon_authorities, future: bool
) -> None:
    """Use both grades and recycled cohorts with each material loss applied once.

    Small real MFA/carbon owners produce the actual activities. Real CH
    source admission supplies their water/sludge/ash portfolio; the existing
    finite chemical/gas fixtures supply the other individually tested receivers.
    All calculations below run after leaving the native source project.
    """
    carbon, recipe = carbon_authorities
    mfa = two_cycle_pa_mfa(recipe, biogenic_share=0.35, dac_share=0.25)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    events = build_burden_demands(mfa, carbon, trace).pa_spinning_events
    reference = pa_receiving_reference()
    sources = pa_receiving_sources(2050 if future else 2025, background_scenario="member")
    with water_receiving_boundary(tmp_path, future=future) as (_, context):
        water_reference = admit_pa_water_receiving_reference(context)
        water_sources = admit_pa_water_receiving_sources(context, water_reference)
        reference = replace(reference, water=water_reference)
        sources = replace(sources, water=water_sources)
    for event in events:
        grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
        # The receiving record retains conversion-source provenance.
        provider = PaReceivingActivity(
            ("frontier", f"fibres_{grade}_conversion"),
            f"{grade} spinning",
            f"{grade} fibre conversion",
            "kilogram",
            "RER",
        )
        actual = receive_pa_spinning(event, reference, sources, spinning_provider=provider)
        site_rows = {
            (row.endpoint.name, row.endpoint.unit, row.endpoint.categories): row
            for row in actual.amounts
            if row.function == "pa_spinning_site"
        }
        finish = _pa_finish_reference(grade)
        site = actual.water
        principal = event.finished_principal_kg
        assert (
            site.site_water_kg
            + principal * finish.supplier_water_kg
            + site.incoming_pellet_water_kg
            == pytest.approx(
                site.gate_water_kg
                + site.exhaust_water_kg
                + site.collected_water_kg
                + site.cooling_water_kg
                + site.maintenance_water_kg,
                rel=3e-13,
            )
        )
        assert site.maintenance_principal_kg == fsum(event.maintenance_principal_by_origin_kg)
        assert site.maintenance_embedded_dose_kg == event.maintenance_finish_dose_kg
        assert site.site_water_kg >= principal * finish.site_water_kg
        assert site.cooling_water_kg > 0
        assert site_rows[
            "NMVOC, non-methane volatile organic compounds", "kilogram", ("air",)
        ].amount == pytest.approx(principal * {"pa6": 0.000675, "pa66": 0.000200}[grade], rel=3e-13)
        assert (
            site_rows[
                "Water, well, in ground", "cubic meter", ("natural resource", "in water")
            ].amount
            == site.site_water_kg / 1000
        )
        assert site_rows["Water", "cubic meter", ("water",)].amount == site.cooling_water_kg / 1000
        assert site_rows["Water", "cubic meter", ("air",)].amount == site.exhaust_water_kg / 1000
        assert all(row.source == provider for row in site_rows.values())
        functions = {row.function for row in actual.amounts}
        assert {
            "pa_spinning_site",
            "pa_maintenance_incineration",
            "pa_maintenance_ash",
            "pa_swiss_collection",
            "pa_swiss_wastewater",
            "pa_swiss_sludge_landfarming",
        } <= functions
        # Full sludge/ash receiving expands the chosen children rather than
        # leaving another complete donor sludge, sewage or soil purchase.
        unexpanded = {
            water_sources.wastewater.activity.key,
            water_sources.sludge_incineration.activity.key,
            water_sources.sludge_landfarming.activity.key,
            water_sources.bottom_ash_treatment.activity.key,
            water_sources.residual_ash_treatment.activity.key,
        }
        assert not any(row.endpoint.key in unexpanded for row in actual.amounts)
