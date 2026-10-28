"""Polyamide fiber washing conserves inherited finish and dry and wet material quantities."""

from math import fsum

import pytest

from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
    _pa_recycling_reference,
)

# The original WEEE operation specifies tap-water kg per incoming dry kg. This material reference
# remains fixed across dated providers.
_WASH_WATER_PER_DRY_KG = 1.1105005972516775 / 1.681445533738159


@pytest.mark.parametrize(
    ("grade", "remaining", "regain"),
    [
        ("pa6", (0.22 / 1.21 + 0.20 / 1.23 + 0.19 / 1.21) / 3, 0.044),
        ("pa66", (0.20 / 1.35 + 0.20 / 1.34 + 0.22 / 1.34) / 3, 0.043),
    ],
)
@pytest.mark.parametrize(("principal", "embedded"), [(0.4, 0.0), (3.2, 1.8), (32000.0, 18000.0)])
def test_pa_washing_closes_components_water_and_actual_pellet_mass(
    grade: str, remaining: float, regain: float, principal: float, embedded: float
) -> None:
    """Apply paired patent oil-pickup ratios to finish at different batch masses.

    Ratios derived from US20200002854A1 represent surface-finish fractions;
    Hasanbeigi supplies moisture regain per kg of dry principal polymer. Inherited finish is
    expressed as equivalent kg of principal polymer associated with the fixed
    finish composition. The largest batch is exactly 10,000 times the smaller
    matching batch. Both retain their actual batch masses.
    """
    finish = _pa_finish_reference(grade)
    # The separately owned PA principal recovery remains 98%; retained finish
    # follows that same partition after its own surface washing.
    recovery = 0.98
    actual = _pa_recycling_reference(
        grade,
        finish,
        principal_kg=principal,
        embedded_dose_kg=embedded,
        recovery_fraction=recovery,
        wash_water_per_dry_kg=_WASH_WATER_PER_DRY_KG,
    )
    surface = finish.retained
    dose_mass = fsum(surface.compounds_kg.values())
    removed_dose = principal * (1 - remaining)
    continuing_dose = embedded + principal * remaining
    dry_in = principal + (principal + embedded) * dose_mass
    dry_washed = principal + continuing_dose * dose_mass
    dry_pellets = recovery * dry_washed

    assert actual.removed_finish.compounds_kg == pytest.approx(
        {name: removed_dose * mass for name, mass in surface.compounds_kg.items()}, rel=2e-13
    )
    # Every finish ingredient has one removed or retained
    # outlet. This also catches washing the inherited embedded pool again.
    for name, mass in surface.compounds_kg.items():
        assert actual.removed_finish.compounds_kg[name] + continuing_dose * mass == pytest.approx(
            (principal + embedded) * mass, rel=2e-13
        )
    assert actual.removed_finish.elements_kg == pytest.approx(
        {element: removed_dose * mass for element, mass in surface.elements_kg.items()},
        rel=2e-13,
    )
    assert actual.removed_finish.fossil_carbon_kg == pytest.approx(
        removed_dose * surface.fossil_carbon_kg, rel=2e-13
    )
    assert actual.removed_finish.biogenic_carbon_kg == pytest.approx(
        removed_dose * surface.biogenic_carbon_kg, rel=2e-13
    )
    assert actual.removed_finish.thod_kg_o2 == pytest.approx(
        removed_dose * surface.thod_kg_o2, rel=2e-13
    )
    assert actual.postwash_embedded_dose_kg == pytest.approx(continuing_dose, rel=2e-13)
    assert actual.incoming_dry_kg == pytest.approx(dry_in, rel=2e-13)
    assert actual.postwash_dry_kg == pytest.approx(dry_washed, rel=2e-13)
    assert actual.pellet_dry_kg == pytest.approx(dry_pellets, rel=2e-13)
    removed_mass = fsum(actual.removed_finish.compounds_kg.values())
    assert actual.postwash_dry_kg + removed_mass == pytest.approx(dry_in, rel=2e-13)

    fresh_water = _WASH_WATER_PER_DRY_KG * dry_in
    incoming_water = regain * principal
    # Patent water fractions refer to wet matter: 4% water/96% dry film,
    # then the selected 500 ppm midpoint/999,500 ppm dry pellets.
    film_water = dry_washed * 4 / 96
    pellet_water = dry_pellets * 500 / 999500
    collected_water = fresh_water + incoming_water - film_water
    assert actual.fresh_wash_water_kg == pytest.approx(fresh_water, rel=2e-13)
    assert actual.incoming_water_kg == pytest.approx(incoming_water, rel=2e-13)
    assert actual.film_water_kg == pytest.approx(film_water, rel=2e-13)
    assert actual.pellet_water_kg == pytest.approx(pellet_water, rel=2e-13)
    assert actual.collected_water_kg == pytest.approx(collected_water, rel=2e-13)
    assert actual.air_water_kg == pytest.approx(film_water - pellet_water, rel=2e-13)
    assert actual.collected_solution_m3 == pytest.approx(
        (collected_water + removed_mass) / 1000, rel=2e-13
    )
    assert fsum(
        (actual.collected_water_kg, actual.pellet_water_kg, actual.air_water_kg)
    ) == pytest.approx(fresh_water + incoming_water, rel=2e-13)

    # Freight carries wet pellets; managed and airborne rejects are dry. Each contains its retained
    # finish.
    wet_pellets = actual.pellet_dry_kg + actual.pellet_water_kg
    assert wet_pellets == pytest.approx(dry_pellets / 0.9995, rel=2e-13)
    dry_rejected = (1 - recovery) * dry_washed
    assert fsum(
        (
            wet_pellets,
            dry_rejected,
            1000 * actual.collected_solution_m3,
            actual.air_water_kg,
        )
    ) == pytest.approx(dry_in + incoming_water + fresh_water, rel=2e-13)


@pytest.mark.parametrize(
    ("grade", "remaining"),
    [
        ("pa6", (0.22 / 1.21 + 0.20 / 1.23 + 0.19 / 1.21) / 3),
        ("pa66", (0.20 / 1.35 + 0.20 / 1.34 + 0.22 / 1.34) / 3),
    ],
)
def test_inherited_finish_survives_washing_with_a_new_surface_dose(
    grade: str, remaining: float
) -> None:
    """Apply the wash fraction once to an inherited embedded dose.

    Compare equal principal polymer inputs with zero and positive finish retained
    from an earlier melt. Separate conversion calculations account for intervening
    material losses before these quantities enter washing.
    """
    finish = _pa_finish_reference(grade)
    principal = 2.5
    inherited_dose = 0.98 * remaining * 3.0
    surface_only = _pa_recycling_reference(
        grade,
        finish,
        principal_kg=principal,
        embedded_dose_kg=0.0,
        recovery_fraction=0.98,
        wash_water_per_dry_kg=_WASH_WATER_PER_DRY_KG,
    )
    repeated = _pa_recycling_reference(
        grade,
        finish,
        principal_kg=principal,
        embedded_dose_kg=inherited_dose,
        recovery_fraction=0.98,
        wash_water_per_dry_kg=_WASH_WATER_PER_DRY_KG,
    )
    assert repeated.removed_finish == surface_only.removed_finish
    # Moisture regain follows the fiber-class reference. Embedded finish determines its additional
    # wash duty.
    assert repeated.incoming_water_kg == surface_only.incoming_water_kg
    assert repeated.postwash_embedded_dose_kg - surface_only.postwash_embedded_dose_kg == (
        pytest.approx(inherited_dose, rel=2e-13)
    )
    inherited_mass = inherited_dose * fsum(finish.retained.compounds_kg.values())
    assert repeated.postwash_dry_kg - surface_only.postwash_dry_kg == pytest.approx(
        inherited_mass, rel=2e-13
    )
    assert repeated.pellet_dry_kg - surface_only.pellet_dry_kg == pytest.approx(
        0.98 * inherited_mass, rel=2e-13
    )
    assert repeated.fresh_wash_water_kg - surface_only.fresh_wash_water_kg == pytest.approx(
        _WASH_WATER_PER_DRY_KG * inherited_mass, rel=2e-13
    )
