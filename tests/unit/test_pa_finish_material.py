"""Check polyamide finish and water quantities before wastewater treatment."""

from dataclasses import asdict
from math import fsum

import pytest

from pytoce.prepare_backgrounds.preparation.providers.pa_finish_concentrate import (
    _pa_blending_residue_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
    _pa_recycling_reference,
    _pa_spinning_water,
)


@pytest.mark.parametrize(
    ("grade", "expected_fraction", "expected_retained", "expected_gate_water"),
    [
        ("pa6", 0.0571223984097466, 0.012625131085293492, 0.044),
        ("pa66", 0.11789621748096138, 0.00823002829090263, 0.043),
    ],
)
def test_pa_finish_reproduces_independent_conditioned_yarn_reference(
    grade: str,
    expected_fraction: float,
    expected_retained: float,
    expected_gate_water: float,
) -> None:
    """Preserve finish oxygen demand and yarn moisture per kg of finished polyamide.

    The fixed expected values preserve the conditioned-yarn reference. The
    collected fraction uses COD minus caprolactam ThOD, divided by ThOD of finish
    remaining after air losses; retained finish is that remaining mass times
    one minus this fraction.
    """
    ref = _pa_finish_reference(grade)
    assert ref.collected_fraction == pytest.approx(expected_fraction)
    assert fsum(ref.retained.compounds_kg.values()) == pytest.approx(expected_retained)
    assert ref.gate_water_kg == pytest.approx(expected_gate_water)
    assert len(ref.ingredient_demands_kg) == 7
    assert set(ref.supplied.compounds_kg) == set(ref.ingredient_demands_kg)
    assert ref.supplied.elements_kg["Na"] == pytest.approx(
        ref.nominal_finish_kg * 0.03 * 22.98976928 / 272.37876928
    )
    assert ref.supplied.elements_kg["P"] == pytest.approx(
        ref.nominal_finish_kg * 0.04 * 30.973761998 / 497.653761998
    )

    # Every auxiliary atom and origin has exactly one air/water/product fate.
    for element, supplied in ref.supplied.elements_kg.items():
        assert fsum(
            load.elements_kg[element] for load in (ref.air, ref.collected, ref.retained)
        ) == pytest.approx(supplied)
    for field in ("fossil_carbon_kg", "biogenic_carbon_kg", "thod_kg_o2"):
        assert fsum(getattr(load, field) for load in (ref.air, ref.collected, ref.retained)) == (
            pytest.approx(getattr(ref.supplied, field))
        )
    for load in (ref.supplied, ref.air, ref.collected, ref.retained, ref.blending_residue):
        assert fsum(load.elements_kg.values()) == pytest.approx(fsum(load.compounds_kg.values()))
        assert load.fossil_carbon_kg + load.biogenic_carbon_kg == pytest.approx(
            load.elements_kg["C"]
        )


@pytest.mark.parametrize("grade", ["pa6", "pa66"])
def test_pa_water_and_blending_have_no_duplicate_carrier_or_rinse(grade: str) -> None:
    """Balance concentrate, site water and solution mass per kg of finished polyamide."""
    ref = _pa_finish_reference(grade)
    assert ref.concentrate_kg == pytest.approx(ref.dry_concentrate_kg + ref.supplier_water_kg)
    assert ref.drum_count == pytest.approx(ref.concentrate_kg / 100)
    assert ref.drum_rinse_water_kg == pytest.approx(
        3 * 0.25 * (55 * 3.785411784) * ref.concentrate_kg / 100
    )
    # Recovered drum rinse belongs to the same ten-percent emulsion, rather
    # than another water purchase or a separate wastewater product.
    assert (
        ref.concentrate_kg + ref.drum_rinse_water_kg + ref.direct_application_water_kg
    ) == pytest.approx(10 * ref.dry_concentrate_kg)
    assert ref.direct_application_water_kg > 0
    assert ref.collected_utility_water_kg > 0
    assert ref.exhaust_water_kg > 0
    assert ref.cooling_water_kg > 0
    assert ref.site_water_kg + ref.supplier_water_kg == pytest.approx(
        fsum(
            (ref.gate_water_kg, ref.exhaust_water_kg, ref.collected_water_kg, ref.cooling_water_kg)
        )
    )
    assert (
        ref.collected_water_kg
        + fsum(ref.collected.compounds_kg.values())
        + ref.collected_caprolactam_kg
    ) == pytest.approx(1000 * ref.collected_solution_m3)
    assert ref.collected_application_water_kg + ref.collected_utility_water_kg == pytest.approx(
        ref.collected_water_kg
    )
    assert fsum(ref.ingredient_demands_kg.values()) == pytest.approx(
        ref.nominal_finish_kg + fsum(ref.blending_residue.compounds_kg.values())
    )


def test_grade_references_survive_interleaved_actual_load_consumers() -> None:
    """Keep fixed finish quantities unchanged across grade-specific actual loads.

    Exercise spinning water, washing and blending-residue serialization
    with the admitted references used by their owner tests. Repeated interleaved
    grades must retain every scalar and nested compound/element quantity exactly.
    """
    expected = {"pa6": (0.014, 0.044, 0.000071), "pa66": (0.0095, 0.043, 0.0)}
    references = {grade: _pa_finish_reference(grade) for grade in expected}
    before = {grade: asdict(reference) for grade, reference in references.items()}
    results: dict[str, dict[str, object]] = {}
    for grade in ("pa6", "pa66", "pa6", "pa66"):
        reference = _pa_finish_reference(grade)
        nominal_finish, regain, caprolactam = expected[grade]
        assert reference.nominal_finish_kg == nominal_finish
        assert reference.collected_caprolactam_kg == caprolactam
        retained_mass = fsum(reference.retained.compounds_kg.values())
        loss = 0.00001
        gross = (1.0001035 if grade == "pa6" else 1.0) / (1 - loss)
        principal_deposit, inherited_deposit = loss * gross, loss * 0.9
        spinning = _pa_spinning_water(
            reference,
            finished_principal_kg=1.0,
            recovered_principal_kg=0.6,
            embedded_dose_kg=0.9,
            maintenance_principal_kg=principal_deposit,
            maintenance_embedded_dose_kg=inherited_deposit,
            maintenance_water_kg=5 * (principal_deposit + retained_mass * inherited_deposit),
        )
        washing = _pa_recycling_reference(
            grade,
            reference,
            principal_kg=3.2,
            embedded_dose_kg=1.8,
            recovery_fraction=0.98,
            wash_water_per_dry_kg=1.1105005972516775 / 1.681445533738159,
        )
        blending = _pa_blending_residue_load(reference)
        assert spinning.gate_water_kg == regain
        assert washing.incoming_water_kg == regain * 3.2
        assert blending.active_elements_kg == reference.blending_residue.elements_kg
        current = {
            "spinning": asdict(spinning),
            "washing": asdict(washing),
            "blending": asdict(blending),
        }
        if grade in results:
            assert current == results[grade]
        results[grade] = current
        assert {name: asdict(value) for name, value in references.items()} == before
