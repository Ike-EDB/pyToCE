"""Actual UPR formulation, dry phases, tracked carbon and homogeneous losses."""

from math import fsum
from typing import Literal

import pytest

from pytoce.download_data.sources.ecoinvent_materials import (
    ConversionRequirement,
    PrincipalExit,
)
from pytoce.prepare_backgrounds.preparation.providers.upr_material import (
    _upr_material,
    _upr_reference,
)


@pytest.fixture(scope="module")
def upr_conversion() -> ConversionRequirement:
    """Use nominal source-equation fractions at the typed material boundary.

    Exact fractions preserve the source equations in this calculation.
    Separate parser tests use synthetic UPR coefficients.
    """
    return ConversionRequirement(
        operation="others_upr_hand_lay_up_conversion",
        principal_feed_kg_per_finished_kg=43163838324174400 / 35680245530124351,
        finished_kg_per_native_service_kg=67075941681801 / 205053863772800,
        principal_exits=(
            PrincipalExit("styrene_air", 8208310784 / 84118311615),
            PrincipalExit("post_release_solid", 103474753 / 922564095),
        ),
    )


@pytest.mark.parametrize(
    ("origin", "index"),
    [("fossil", 0), ("biogenic", 1), ("atmospheric_dac", 2)],
)
def test_upr_actual_recipe_and_exits_match_the_admitted_witness(
    upr_conversion: ConversionRequirement,
    origin: Literal["fossil", "biogenic", "atmospheric_dac"],
    index: int,
) -> None:
    """One input charge closes across dry article, homogeneous trim and volatiles."""
    reference = _upr_reference(upr_conversion, origin)
    # The nominal reference uses source application kg/F, including glass sizing.
    expected_supplies = {
        "reinforcement_glass": 1.8160475072263038,
        "silica": 0.003944476195995197,
        "organic_peroxide": 0.008253057887005335,
        "dmp": 0.015462346688301171,
        "water": 0.0005582950923562433,
        "cobalt_salt": 0.0010923164850448238,
        "styrene": 0.0012136849833831375,
        "white_spirit": 0.0007282109900298825,
    }
    expected_retained = {
        "principal": 1.0,
        "reinforcement_glass": 1.6329013548044928,
        "silica": 0.003546680634072202,
        "organic_peroxide": 0.007420747172827992,
        "dmp": 0.01390298808556303,
        "cobalt_salt": 0.0009821577140507636,
        "styrene": 0.0008712219689321986,
    }
    assert reference.supply_amounts == pytest.approx(expected_supplies)
    assert reference.retained.components_kg == pytest.approx(expected_retained)
    assert reference.gross_principal_kg == pytest.approx(1.2097405071871423)
    assert reference.operation_kg == pytest.approx(3.057040402735562)
    assert reference.application_kg == pytest.approx(1 / 0.823937029535051)
    assert reference.input_mass_kg == pytest.approx(reference.operation_kg)
    assert reference.principal_styrene_air_kg == pytest.approx(0.0975805460952249)
    assert reference.accelerator_styrene_air_kg == pytest.approx(0.0002447467923130798)
    assert reference.white_spirit_air_kg == pytest.approx(expected_supplies["white_spirit"])
    assert reference.water_air_kg == pytest.approx(expected_supplies["water"])
    assert reference.retained.mass_kg == pytest.approx(2.659625150379939)
    assert reference.external_solid.components_kg["principal"] == pytest.approx(0.1121599610919174)
    assert reference.input_mass_kg == pytest.approx(
        fsum(
            (
                reference.retained.mass_kg,
                reference.external_solid.mass_kg,
                reference.principal_styrene_air_kg,
                reference.accelerator_styrene_air_kg,
                reference.white_spirit_air_kg,
                reference.water_air_kg,
            )
        )
    )
    ratio = 0.1121599610919174
    for name, amount in expected_retained.items():
        assert reference.external_solid.components_kg[name] == pytest.approx(ratio * amount)
    for material in (reference.retained, reference.external_solid):
        assert material.water_kg == 0.0
        assert fsum(
            (*material.burnable_elements_kg.values(), *material.inert_elements_kg.values())
        ) == pytest.approx(material.mass_kg)
        assert material.principal_carbon_kg[index] > 0
        assert fsum(material.principal_carbon_kg) == pytest.approx(
            material.principal_carbon_kg[index]
        )
        assert material.fossil_auxiliary_carbon_kg > 0
        assert material.burnable_elements_kg["C"] == pytest.approx(
            fsum(
                (
                    *material.principal_carbon_kg,
                    *material.principal_carbon_bridge_kg,
                    material.fossil_auxiliary_carbon_kg,
                )
            )
        )


def test_upr_nominal_atoms_and_heat_keep_glass_cobalt_and_carbon_bridge_distinct(
    upr_conversion: ConversionRequirement,
) -> None:
    """Keep nominal element masses distinct from tracked carbon and heat-eligible elements."""
    reference = _upr_reference(upr_conversion, "biogenic")
    material = reference.retained
    c = material.components_kg
    assert material.principal_elements_kg == pytest.approx(
        {
            "C": 0.7099482404733776,
            "H": 0.05917458203684342,
            "O": 0.23087717748977904,
        }
    )
    assert material.principal_carbon_kg == pytest.approx((0, 0.70405325131712, 0))
    assert material.principal_carbon_bridge_kg == pytest.approx((0, 0.005894989156257515, 0))

    # UNSW four-oxide E-glass and the admitted 1% dry sizing. Element masses
    # sum to the whole inert mass, with no inferred Mg/Na/Fe constituents.
    oxides = {
        "SiO2": 0.99 * c["reinforcement_glass"] * 0.54 + c["silica"],
        "Al2O3": 0.99 * c["reinforcement_glass"] * 0.14,
        "CaO": 0.99 * c["reinforcement_glass"] * 0.22,
        "B2O3": 0.99 * c["reinforcement_glass"] * 0.10,
    }
    inert = {
        "Si": oxides["SiO2"] * 28.085 / (28.085 + 32),
        "Al": oxides["Al2O3"] * (2 * 26.982) / (2 * 26.982 + 48),
        "Ca": oxides["CaO"] * 40.078 / (40.078 + 16),
        "B": oxides["B2O3"] * (2 * 10.81) / (2 * 10.81 + 48),
    }
    inert["O"] = fsum(oxides.values()) - fsum(inert.values())
    assert material.inert_elements_kg == pytest.approx(inert)

    # Independent actual formula table. Co2EH uses its entire admitted CIAAW
    # convention; the other nominal CHO formulas use C12/H1/O16 consistently.
    # The coupling's complete mass is separated from Michel (via Muñoz et al. (2007), Eq. 9)
    # eligibility.
    dry_formulae = (
        (c["organic_peroxide"] * 260 / 467, {"C": 48, "H": 10, "O": 64}),
        (c["organic_peroxide"] * 207 / 467, {"C": 96, "H": 18, "O": 96}),
        (c["dmp"], {"C": 120, "H": 10, "O": 64}),
        (c["styrene"], {"C": 96, "H": 8}),
        (c["cobalt_salt"], {"C": 192.176, "H": 30.24, "O": 63.996, "Co": 58.933}),
        (c["reinforcement_glass"] * 0.009, {"C": 48, "H": 6, "O": 32}),
        (c["reinforcement_glass"] * 0.0005, {"C": 36, "H": 8, "O": 48}),
    )
    auxiliary = {
        element: fsum(
            mass * atoms.get(element, 0) / fsum(atoms.values()) for mass, atoms in dry_formulae
        )
        for element in ("C", "H", "O", "Co")
    }
    coupling = c["reinforcement_glass"] * 0.0005
    coupling_atoms = {"C": 168, "H": 22, "O": 112, "Si": 56.17}
    expected_burnable = {
        element: material.principal_elements_kg.get(element, 0)
        + auxiliary.get(element, 0)
        + coupling * coupling_atoms.get(element, 0) / fsum(coupling_atoms.values())
        for element in ("C", "H", "O", "Si", "Co")
    }
    assert material.burnable_elements_kg == pytest.approx(expected_burnable)
    expected_eligible = {
        element: material.principal_elements_kg[element] + auxiliary[element]
        for element in ("C", "H", "O")
    }
    assert material.heat_eligible_elements_kg == pytest.approx(expected_eligible)
    assert material.coupling_heat_mj == pytest.approx(19.53 * coupling)
    assert material.heat_potential_mj == pytest.approx(
        34.016 * expected_eligible["C"]
        + 124.265 * expected_eligible["H"]
        - 9.8324 * expected_eligible["O"]
        + 19.53 * coupling
    )
    assert material.fossil_auxiliary_carbon_kg == pytest.approx(
        auxiliary["C"] + coupling * 168 / 358.17
    )

    # Reverse the homogeneous solid partition, then return the styrene withdrawal to reconstruct
    # delivered CHO. White spirit retains its specified mixture withdrawal.
    for element, delivered, styrene in (
        ("C", 0.7271397068361655, 96 / 104),
        ("H", 0.06060621784061919, 8 / 104),
        ("O", 0.2122540753232153, 0),
    ):
        assert fsum(
            (
                material.principal_elements_kg[element],
                reference.external_solid.principal_elements_kg[element],
                reference.principal_styrene_air_kg * styrene,
            )
        ) == pytest.approx(reference.gross_principal_kg * delivered)


def test_upr_retired_mixed_origin_material_preserves_fixed_auxiliaries_and_scaling(
    upr_conversion: ConversionRequirement,
) -> None:
    """Scale the fixed dry article to the mixed-origin cohort."""
    baseline = _upr_reference(upr_conversion, "fossil").retained
    mixed = _upr_material(upr_conversion, (0.2, 0.6, 1.2))
    assert mixed.principal_by_origin_kg == (0.2, 0.6, 1.2)
    for name in (
        "components_kg",
        "principal_elements_kg",
        "burnable_elements_kg",
        "inert_elements_kg",
        "heat_eligible_elements_kg",
    ):
        assert getattr(mixed, name) == pytest.approx(
            {key: 2 * value for key, value in getattr(baseline, name).items()}
        )
    assert mixed.principal_carbon_kg == pytest.approx(
        tuple(mass * 0.70405325131712 for mass in (0.2, 0.6, 1.2))
    )
    assert mixed.principal_carbon_bridge_kg == pytest.approx(
        tuple(mass * 0.005894989156257515 for mass in (0.2, 0.6, 1.2))
    )
    assert mixed.fossil_auxiliary_carbon_kg == pytest.approx(
        2 * baseline.fossil_auxiliary_carbon_kg
    )
    assert mixed.mass_kg == pytest.approx(2 * baseline.mass_kg)
    assert mixed.heat_potential_mj == pytest.approx(2 * baseline.heat_potential_mj)
    assert mixed.coupling_heat_mj == pytest.approx(2 * baseline.coupling_heat_mj)
    assert mixed.water_kg == 0.0
