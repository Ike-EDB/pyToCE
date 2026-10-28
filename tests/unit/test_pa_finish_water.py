"""Keep principal polymer and auxiliary material conventions distinct in polyamide wastewater."""

from math import fsum

import pytest

from pytoce.prepare_backgrounds.preparation.providers.leachate_load import _CarbonOrigin
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_water import (
    _pa_finish_water_load,
)


@pytest.mark.parametrize(
    ("grade", "origins", "cod"),
    (
        ("pa6", {"fossil": 1.0}, 0.0023),
        ("pa6", {"fossil": 0.2, "biogenic": 0.3, "atmospheric_dac": 0.5}, 0.0023),
        ("pa66", {}, 0.0031),
    ),
)
def test_actual_collected_load_preserves_principal_basis_and_fixed_finish_origins(
    grade: str, origins: dict[_CarbonOrigin, float], cod: float
) -> None:
    """Preserve principal carbon accounting in each polyamide wastewater load.

    The mixture of three carbon origins is a synthetic test input.
    Joint treatment must preserve all seven finish components and their biogenic
    constituents. Polyamide 66 supplies no principal caprolactam.
    """
    reference = _pa_finish_reference(grade)
    load = _pa_finish_water_load(
        reference.collected,
        caprolactam_by_origin_kg={
            origin: fraction * reference.collected_caprolactam_kg
            for origin, fraction in origins.items()
        },
    )
    principal = [pool for pool in load.carbon if pool.principal]
    assert len(principal) == len(origins)
    assert all(pool.component == "caprolactam" for pool in principal)
    assert {pool.origin: pool.carbon_kg for pool in principal} == pytest.approx(
        {
            origin: fraction * reference.collected_caprolactam_kg * 72 / 113
            for origin, fraction in origins.items()
        },
        rel=2e-14,
    )
    auxiliary = [pool for pool in load.carbon if not pool.principal]
    assert {pool.component for pool in auxiliary} == set(reference.collected.compounds_kg)
    assert fsum(pool.carbon_kg for pool in auxiliary if pool.origin == "biogenic") == pytest.approx(
        reference.collected.biogenic_carbon_kg, rel=2e-14
    )
    assert fsum(pool.carbon_kg for pool in auxiliary if pool.origin == "fossil") == pytest.approx(
        reference.collected.fossil_carbon_kg, rel=2e-14
    )
    assert not any(pool.origin == "atmospheric_dac" for pool in auxiliary)
    assert fsum(pool.carbon_kg for pool in load.carbon) == pytest.approx(
        load.elements_kg["C"], rel=2e-14
    )
    # Solute masses plus separately supplied water give the same
    # selected 0.9 kg solution, while COD remains an overlapping O2 indicator.
    assert fsum(load.elements_kg.values()) + reference.collected_water_kg == pytest.approx(0.9)
    assert load.cod_kg_o2 == pytest.approx(cod, rel=2e-14)
    assert load.doc_kg < load.toc_kg
    assert load.organic_hydrogen_kg != load.elements_kg["H"]
    assert load.organic_oxygen_kg != load.elements_kg["O"]
