"""Actual road formulation, selective MMA loss and receiving material partitions."""

from math import fsum
from typing import Literal

import pytest

from pytoce.download_data.sources.ecoinvent_materials import (
    ConversionRequirement,
    PrincipalExit,
)
from pytoce.prepare_backgrounds.preparation.providers.road_material import (
    _road_material,
    _road_reference,
)


@pytest.fixture(scope="module")
def road_conversion() -> ConversionRequirement:
    """Use the independent source witness at the admitted principal boundary.

    Per source application: 0.223 kg binder, 1% homogeneous container loss,
    then 0.00202 kg MMA evaporation leave 0.21875 kg retained principal.
    Exact fractions retain these declared quantities.
    """
    return ConversionRequirement(
        operation="others_road_marking_conversion",
        principal_feed_kg_per_finished_kg=892 / 875,
        finished_kg_per_native_service_kg=1.0,
        principal_exits=(
            PrincipalExit("container_residue", 223 / 21875),
            PrincipalExit("methyl_methacrylate_air", 202 / 21875),
        ),
    )


@pytest.mark.parametrize(
    ("origin", "index"),
    [("fossil", 0), ("biogenic", 1), ("atmospheric_dac", 2)],
)
def test_road_source_charge_closes_across_container_mma_and_retained_article(
    road_conversion: ConversionRequirement,
    origin: Literal["fossil", "biogenic", "atmospheric_dac"],
    index: int,
) -> None:
    """The pre-evaporation container and retained material use different ratios."""
    reference = _road_reference(road_conversion, origin)
    retained, container = reference.retained, reference.container_residue
    # Silikal source formulation and historical 50:50 hardener determine kg per application;
    # supplier proxies retain those quantities.
    application_supplies = {
        "rutile_titanium_dioxide": 0.075,
        "ground_calcium_carbonate": 0.15,
        "silica_sand": 0.10,
        "cristobalite_silica_analogue": 0.15,
        "corundum": 0.05,
        "others_recycled_glass_bead_manufacturing": 0.25,
        "fumed_silica_activated_silica_analogue": 0.002,
        "organic_peroxide_hardener_analogue": 0.005,
        "hardener_phlegmatizer_dotp_analogue": 0.005,
    }
    assert reference.supply_amounts == pytest.approx(
        {name: amount / 0.21875 for name, amount in application_supplies.items()}
    )
    assert reference.gross_principal_kg == pytest.approx(0.223 / 0.21875)
    assert reference.premix_kg == pytest.approx(1 / 0.21875)
    assert reference.mixing_kg == reference.premix_kg
    assert reference.input_mass_kg == pytest.approx(1.01 / 0.21875)
    assert reference.methyl_methacrylate_air_kg == pytest.approx(0.00202 / 0.21875)
    assert retained.mass_kg == pytest.approx(4.561737142857143)
    assert container.mass_kg == pytest.approx(0.0101 / 0.21875)
    assert reference.input_mass_kg == pytest.approx(
        fsum((retained.mass_kg, container.mass_kg, reference.methyl_methacrylate_air_kg))
    )
    assert retained.components_kg["principal"] == 1
    assert container.components_kg["principal"] == pytest.approx(0.00223 / 0.21875)
    assert retained.components_kg["limestone"] == pytest.approx(0.1485 / 0.21875)
    assert container.components_kg["limestone"] == pytest.approx(0.0015 / 0.21875)
    # Only auxiliaries share the 1:99 container/retained ratio. Scaling the
    # final article by the principal container ratio would over-treat fillers.
    for name, amount in retained.components_kg.items():
        if name != "principal":
            assert container.components_kg[name] == pytest.approx(amount / 99)
    assert container.components_kg["principal"] != pytest.approx(1 / 99)

    for material in (retained, container):
        assert fsum(material.principal_by_origin_kg) == material.components_kg["principal"]
        assert material.principal_carbon_kg[index] == pytest.approx(
            0.6 * material.components_kg["principal"]
        )
        assert fsum(material.principal_carbon_kg) == material.principal_carbon_kg[index]
        assert material.burnable_elements_kg["C"] == pytest.approx(
            fsum((*material.principal_carbon_kg, material.fossil_auxiliary_carbon_kg))
        )
        assert material.geological_carbon_kg == material.limestone_elements_kg["C"]
        assert material.geological_carbon_kg > 0
        assert "C" not in material.mineral_elements_kg
        assert fsum(
            (
                *material.burnable_elements_kg.values(),
                *material.mineral_elements_kg.values(),
                *material.limestone_elements_kg.values(),
                material.water_kg,
            )
        ) == pytest.approx(material.mass_kg)


def test_road_actual_glass_silicate_carbonate_and_organic_heat_stay_distinct(
    road_conversion: ConversionRequirement,
) -> None:
    """Keep burnable, mineral and limestone element masses distinct from water."""
    material = _road_reference(road_conversion, "biogenic").retained
    c = material.components_kg
    assert material.principal_elements_kg == pytest.approx({"C": 0.6, "H": 0.08, "O": 0.32})
    assert c["bpo_derived"] == pytest.approx(0.00495 / 0.21875)
    assert c["dchp"] == pytest.approx(0.00495 / 0.21875)
    assert c["glass"] == pytest.approx((0.2475 / 0.21875) * 500 / 500.240)
    assert c["silicate"] == pytest.approx((0.2475 / 0.21875) * 0.036 / 500.240)
    assert material.water_kg == pytest.approx(0.00032569509492586413)
    assert c["water"] == material.water_kg
    assert fsum(material.burnable_elements_kg.values()) == pytest.approx(1.0453928491466953)
    assert fsum(
        (*material.mineral_elements_kg.values(), *material.limestone_elements_kg.values())
    ) == pytest.approx(3.516018598615522)

    # Nominal formula vectors define composition. Silicate's 3.22 modulus is a weight ratio. JRC
    # glass retains sulfur with its oxygen.
    glass_oxides = (
        (72.6, {"Si": 28.085, "O": 32}),
        (13.6, {"Na": 45.98, "O": 16}),
        (8.6, {"Ca": 40.078, "O": 16}),
        (4.1, {"Mg": 24.305, "O": 16}),
        (0.7, {"Al": 53.964, "O": 48}),
        (0.3, {"K": 78.1966, "O": 16}),
        (0.17, {"S": 32, "O": 48}),
    )
    other_minerals = (
        (c["rutile"], {"Ti": 47.867, "O": 32}),
        (c["silica_sand"] + c["cristobalite"] + c["fumed_silica"], {"Si": 28.085, "O": 32}),
        (c["corundum"], {"Al": 53.964, "O": 48}),
        (c["silicate"] * 3.22 / 4.22, {"Si": 28.085, "O": 32}),
        (c["silicate"] / 4.22, {"Na": 45.98, "O": 16}),
    )
    expected_minerals = {
        element: fsum(
            c["glass"] * weight / 100.07 * formula.get(element, 0) / fsum(formula.values())
            for weight, formula in glass_oxides
        )
        + fsum(
            mass * formula.get(element, 0) / fsum(formula.values())
            for mass, formula in other_minerals
        )
        for element in ("Si", "Na", "Ca", "Mg", "Al", "K", "S", "Ti", "O")
    }
    assert material.mineral_elements_kg == pytest.approx(expected_minerals)
    assert material.limestone_elements_kg == pytest.approx(
        {
            element: c["limestone"] * mass / 100.078
            for element, mass in {"C": 12, "Ca": 40.078, "O": 48}.items()
        }
    )
    organic_formulae = (
        (1.0, {"C": 60, "H": 8, "O": 32}),
        (c["bpo_derived"], {"C": 168, "H": 10, "O": 64}),
        (c["dchp"], {"C": 240, "H": 26, "O": 64}),
    )
    eligible = {
        element: fsum(
            mass * formula[element] / fsum(formula.values()) for mass, formula in organic_formulae
        )
        for element in ("C", "H", "O")
    }
    pdms_formula = {"C": 24, "H": 6, "O": 16, "Si": 28.085}
    assert material.burnable_elements_kg == pytest.approx(
        {
            element: eligible.get(element, 0) + c["pdms"] * atom / fsum(pdms_formula.values())
            for element, atom in pdms_formula.items()
        }
    )
    assert material.fossil_auxiliary_carbon_kg == pytest.approx(
        eligible["C"] - 0.6 + c["pdms"] * 24 / 74.085
    )
    assert material.heat_potential_mj == pytest.approx(
        34.016 * eligible["C"]
        + 124.265 * eligible["H"]
        - 9.8324 * eligible["O"]
        + 19.53 * c["pdms"]
    )


def test_road_retired_mixed_origins_scale_one_complete_article(
    road_conversion: ConversionRequirement,
) -> None:
    """Retain auxiliary origins and chemistry when principal origin changes."""
    single = _road_reference(road_conversion, "fossil").retained
    mixed = _road_material(road_conversion, (0.2, 0.6, 1.2))
    assert mixed.principal_by_origin_kg == (0.2, 0.6, 1.2)
    assert mixed.principal_carbon_kg == pytest.approx((0.12, 0.36, 0.72))
    for name in (
        "components_kg",
        "principal_elements_kg",
        "burnable_elements_kg",
        "mineral_elements_kg",
        "limestone_elements_kg",
    ):
        assert getattr(mixed, name) == pytest.approx(
            {key: 2 * value for key, value in getattr(single, name).items()}
        )
    for name in (
        "mass_kg",
        "water_kg",
        "fossil_auxiliary_carbon_kg",
        "geological_carbon_kg",
        "heat_potential_mj",
    ):
        assert getattr(mixed, name) == pytest.approx(2 * getattr(single, name))
