"""Balance polyamide spinning water with recovered polymer and inherited finish."""

from math import fsum

import pytest

from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
    _pa_spinning_water,
)


@pytest.mark.parametrize("grade", ["pa6", "pa66"])
@pytest.mark.parametrize("recovered,dose", [(0.0, 0.0), (0.6, 0.9)])
def test_actual_spinning_water_closes_without_repeated_site_duty(
    grade: str, recovered: float, dose: float
) -> None:
    """Preserve the original water balance and add cleaning for inherited finish deposits.

    The synthetic recycled case combines 0.6 kg recovered principal polymer with
    finish from 0.9 kg of previously finished polymer. Cleaning uses a synthetic
    5 kg water/kg dry deposit coefficient, consistent with the thermal fixture.
    """
    finish = _pa_finish_reference(grade)
    loss = 0.00001
    gross = (1.0001035 if grade == "pa6" else 1.0) / (1 - loss)
    retained = fsum(finish.retained.compounds_kg.values())
    # Equipment treatment receives the calculated deposits and cleaning water, with spinning loss
    # counted once.
    principal_deposit, inherited_deposit = loss * gross, loss * dose
    maintenance_water = 5.0 * (principal_deposit + retained * inherited_deposit)
    actual = _pa_spinning_water(
        finish,
        finished_principal_kg=1.0,
        recovered_principal_kg=recovered,
        embedded_dose_kg=dose,
        maintenance_principal_kg=principal_deposit,
        maintenance_embedded_dose_kg=inherited_deposit,
        maintenance_water_kg=maintenance_water,
    )
    # Purchased water, concentrate carrier water and incoming pellet moisture
    # leave as yarn moisture, exhaust, collected water, cooling water or the
    # separate water input to equipment treatment.
    assert actual.site_water_kg + finish.supplier_water_kg + actual.incoming_pellet_water_kg == (
        pytest.approx(
            actual.gate_water_kg
            + actual.exhaust_water_kg
            + actual.collected_water_kg
            + actual.cooling_water_kg
            + actual.maintenance_water_kg
        )
    )
    assert actual.site_water_kg - finish.site_water_kg == pytest.approx(5 * loss * dose * retained)
    assert actual.cooling_water_kg + 5 * actual.maintenance_principal_kg == pytest.approx(
        finish.cooling_water_kg
    )
    assert actual.cooling_water_kg > 0
    assert actual.maintenance_dry_kg == pytest.approx(loss * (gross + retained * dose))
    assert actual.maintenance_embedded_dose_kg == loss * dose
    assert actual.maintenance_water_kg == 5 * actual.maintenance_dry_kg
    assert actual.incoming_pellet_water_kg == pytest.approx(
        0.0005 * (recovered + retained * dose + actual.incoming_pellet_water_kg)
    )
    assert actual.gate_water_kg == finish.gate_water_kg
    assert actual.collected_water_kg == finish.collected_water_kg
    assert actual.exhaust_water_kg - finish.exhaust_water_kg == pytest.approx(
        actual.incoming_pellet_water_kg
    )
