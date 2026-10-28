"""Test PA sludge fates using source coefficients and Doka phosphorus rates.

The synthetic original and future landfarming inventories include nonunit
negative production and spreading. Optional donor phosphorus rows are
independent of the modeled PA finish's phosphorus content.
"""

import json
from dataclasses import asdict
from math import fsum
from typing import cast

import pytest
from test_pa_finish_landfarming import landfarming_boundary as landfarming_boundary

from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _CH_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_load import (
    _pa_landfill_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingFlow,
    PaReceivingLandfarmingReference,
    _landfarming_reference_from_payload,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    _operation,
    _properties,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_landfarming import (
    _FLOW_ELEMENTS,
    _landfarming_dataset,
    calculate_landfarming,
)


@pytest.mark.parametrize("grade,origin", (("pa6", "fossil"), ("pa66", "atmospheric_dac")))
def test_actual_pa_sludge_calculates_from_portable_original_coefficients(
    landfarming_boundary, grade, origin
):
    """Check C/P fates, native nitrogen and water, and scaled source distributions."""
    _, context, database, donor_phosphorus = landfarming_boundary
    service = next(
        row
        for row in context.catalogue["services"]
        if row["provider_service_id"] == "tps_sludge_landfarming_biogenic"
    )
    source = context.source.activity(service["construction"]["source"], year=2025)
    properties = _properties(context, source)
    # This is the exact selected PA soil scope used by the shared water and
    # sanitary source admissions. Report P intentionally has no donor row/assay.
    element_names = {"C": "carbon", "N": "nitrogen", "S": "sulfur", "Fe": "iron", "Na": "sodium"}
    selected = {
        name for name, (element, _) in _FLOW_ELEMENTS.items() if element in {"N", "S", "Fe", "Na"}
    } | {
        "Carbon dioxide, fossil",
        "Carbon dioxide, non-fossil",
        "Water",
        "Heat, waste",
        "Non-hazardous waste disposed",
    }
    operation = _operation(context, source, products=set(), flows=selected)
    reference = PaReceivingLandfarmingReference(
        operation,
        float(cast(float, properties["dry mass"]["amount"])),
        float(cast(float, properties["water in wet mass"]["amount"])),
        {
            element: float(cast(float, properties[f"{name} content"]["amount"]))
            for element, name in element_names.items()
        },
    )
    decoded = _landfarming_reference_from_payload(
        json.loads(json.dumps(asdict(reference), allow_nan=False))
    )
    assert decoded.operation.production_amount == -8  # original, even for dated operation -11
    assert "P" not in decoded.element_contents
    assert all(
        cast(PaReceivingFlow, row.endpoint).name != "Phosphate" for row in decoded.operation.rows
    )
    assert ("phosphorus content" in properties) == donor_phosphorus
    finish = _pa_finish_reference(grade)
    mixture = _pa_landfill_load(
        finish,
        principal_by_origin_kg={origin: 2.0},
        finish_dose_kg=5.0,
        water_kg=0.4,
        site=_LandfillSite(16.743579766537, 1250.84108527132, 1.0),
    )
    water = _wastewater_load(
        mixture.leachate,
        volume_m3=0.002,
        incoming_water_kg=1.8,
        evaporation_fraction=0.01,
        profile=_CH_WASTEWATER_PROFILE,
    )
    result = calculate_landfarming(
        water, decoded, complete_carbon_oxidation=True, phosphorus_equilibrium=True
    )
    native = _landfarming_dataset(
        context,
        service,
        water,
        "actual_pa_sludge",
        "actual_pa_landfarming",
        complete_carbon_oxidation=True,
        phosphorus_equilibrium=True,
    )
    terms = json.loads(context.technology_rows[-1]["project_terms_json"])
    assert result.elements_kg == terms["actual_elements_kg"]
    assert result.empirical_heat_mj == terms["empirical_soil_heat_MJ"]
    # Retained WWcalc DU7 is a signed empirical bulk estimate. These PA test loads
    # are expected to give a negative DU7; the original DU18 expression emits zero
    # heat when DU7 is negative.
    # No signed lognormal or additional heat source is added.
    raw_heat = (
        -9.8324 * result.allocated_oxygen_kg
        + 124.265 * result.allocated_hydrogen_kg
        + 34.016 * result.elements_kg["C"]
        + 19.079 * result.elements_kg["S"]
        + 6.276 * result.elements_kg["N"]
    )
    assert raw_heat < 0
    assert result.empirical_unfloored_heat_mj == raw_heat
    assert terms["unfloored_empirical_soil_heat_MJ"] == raw_heat
    assert result.empirical_heat_mj == 0
    assert result.carbon_components == tuple(
        type(row)(**expected)
        for row, expected in zip(result.carbon_components, terms["carbon_components"], strict=True)
    )
    assert result.wet_kg == water.landfarmed_sludge_wet_kg
    assert result.carried_water_kg == water.water.landfarmed_sludge_water_kg
    assert result.dry_kg == water.landfarming_dry_share * water.final_sludge.dry_mass_kg
    assert fsum(row.carbon_kg for row in result.carbon_components) == pytest.approx(
        result.elements_kg["C"]
    )
    # Complete oxidation is explicit for actual PA; the original municipal
    # carbon-origin partition cannot relabel the actual principal or auxiliaries.
    co2_carbon = fsum(row.amount_kg for row in result.carbon_dioxide) * 12 / 44
    assert co2_carbon == pytest.approx(result.elements_kg["C"], rel=2e-12, abs=0)
    for row in result.carbon_dioxide:
        mass = result.carbon_by_origin_kg[row.origin]
        assert row.amount_kg == pytest.approx(mass * 44 / 12, rel=2e-12, abs=0)
        for index, principal_origin in enumerate(("fossil", "biogenic", "atmospheric_dac")):
            principal = fsum(
                pool.carbon_kg
                for pool in result.carbon_components
                if pool.principal
                and pool.origin == principal_origin
                and (pool.origin == "fossil") == (row.origin == "fossil")
            )
            assert row.principal_carbon_origin_fractions[index] == pytest.approx(principal / mass)
    phosphorus = result.elements_kg["P"]
    assert phosphorus > 0
    ground_p = phosphorus * 0.07 / 12
    surface_p = phosphorus * (0.175 + 1.0602) / 12
    assert result.phosphate_water_kg == pytest.approx(
        {
            ("water", "ground-"): ground_p * 94.97 / 30.97,
            ("water", "surface water"): surface_p * 94.97 / 30.97,
        }
    )
    assert result.phosphorus_crop_uptake_kg + ground_p + surface_p == pytest.approx(phosphorus)
    assert result.nitrogen_crop_uptake_kg == pytest.approx(0.6 * result.elements_kg["N"])
    assert result.phosphorus_crop_uptake_kg == terms["phosphorus_crop_uptake_kg"]
    rows = cast(list[dict[str, object]], native["exchanges"])
    assert not any(
        row["type"] == "biosphere" and context.input(row)["name"] == "Heat, waste" for row in rows
    )
    production = next(row for row in rows if row["type"] == "production")
    assert production["amount"] == (-11 if database == "future" else -8)
    scale = -float(cast(float, production["amount"])) / result.wet_kg
    for release in result.native_releases:
        template = decoded.operation.rows[release.source_row]
        emitted = next(
            row
            for row in rows
            if row["type"] == "biosphere" and row["input"] == template.endpoint.key
        )
        assert emitted["amount"] == pytest.approx(release.amount * scale, rel=2e-12, abs=0)
        if "scale" in template.metadata:
            assert emitted["scale"] == template.metadata["scale"]
    phosphate = [row for row in rows if row.get("name") == "Phosphate"]
    assert len(phosphate) == 2
    assert all(row["uncertainty type"] == 0 and "loc" not in row for row in phosphate)

    # Reject an ambiguous original inventory before normalizing its sludge load.
    extra_production = source.new_exchange(input=source.key, type="production", amount=1.0)
    try:
        extra_production.save()
        context.native_exchanges.pop(source.key, None)
        with pytest.raises(
            ValueError, match="sludge landfarming source must have one self-production exchange"
        ):
            _landfarming_dataset(
                context,
                service,
                water,
                "actual_pa_sludge",
                "actual_pa_landfarming",
                complete_carbon_oxidation=True,
                phosphorus_equilibrium=True,
            )
    finally:
        extra_production.delete()
        context.native_exchanges.pop(source.key, None)
