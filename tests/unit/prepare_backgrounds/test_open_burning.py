"""Check PA and admitted burning equations against independent element mass balances."""

from math import fsum

import pytest

from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    load_provider_catalogue,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import _CarbonOrigin
from pytoce.prepare_backgrounds.preparation.providers.open_burning import (
    _open_burning_load,
    _OpenBurningLoad,
    _OpenBurningMaterial,
    _pa_open_burning_load,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.tps_burning_material import (
    _tps_burning_material,
)
from pytoce.prepare_backgrounds.preparation.providers.tps_residue_load import (
    _tps_material_load,
)

_AIR = ("air", "non-urban air or from high stacks")
_SOIL = ("soil", "industrial")
_INDICATOR = ("inventory indicator", "waste")
_PAH = "PAH, polycyclic aromatic hydrocarbons"
# Independent molecular ledger for the native emissions, nominal kg/kmol.
_GAS_ATOMS = {
    "Benzene": {"C": 72, "H": 6},
    "Phenol": {"C": 72, "H": 6, "O": 16},
    "Styrene": {"C": 96, "H": 8},
    "Toluene": {"C": 84, "H": 8},
    "Formaldehyde": {"C": 12, "H": 2, "O": 16},
    "Acetaldehyde": {"C": 24, "H": 4, "O": 16},
    "Ethyl benzene": {"C": 96, "H": 10},
    "Carbon dioxide": {"C": 12, "O": 32},
    "Carbon monoxide": {"C": 12, "O": 16},
    "Methane": {"C": 12, "H": 4},
    "Nitrogen oxides": {"N": 14, "O": 32},
    "Dinitrogen monoxide": {"N": 28, "O": 16},
    "Sulfur dioxide": {"S": 32, "O": 32},
}
# EPA1997 Table3-7/SIM: separately transcribed paired-run total and C/H
# mass fractions. The factor of two cancels when normalizing paired means.
_PAH_SOURCE = (
    (0.9578 + 0.5780, 144 / 154),
    (13.6424 + 8.9577, 144 / 152),
    (2.3724 + 1.2910, 168 / 178),
    (3.1364 + 1.3425, 216 / 228),
    (3.1275 + 1.1167, 240 / 252),
    (3.7585 + 1.5016, 240 / 252),
    (2.8148 + 1.1340, 264 / 276),
    (1.6424 + 0.4085, 240 / 252),
    (3.5588 + 1.5136, 216 / 228),
    (0.4861 + 0.1595, 264 / 278),
    (5.1917 + 2.9436, 192 / 202),
    (4.7756 + 2.5360, 156 / 166),
    (2.7997 + 0.9547, 264 / 276),
    (18.9598 + 16.1032, 120 / 128),
    (8.9946 + 5.6546, 168 / 178),
    (6.1419 + 3.5157, 192 / 202),
)


@pytest.mark.parametrize("grade", ["pa6", "pa66"])
@pytest.mark.parametrize(
    ("principal", "dose"),
    [
        ({"fossil": 1.0}, 0.0),
        ({"biogenic": 1.0}, 0.0),
        ({"atmospheric_dac": 1.0}, 0.0),
        ({}, 1.0),
        ({"fossil": 0.3, "biogenic": 0.8, "atmospheric_dac": 0.9}, 3.7),
    ],
)
def test_actual_pa_burning_closes_seven_elements_and_complete_native_scope(
    grade: str, principal: dict[_CarbonOrigin, float], dose: float
) -> None:
    """Check endpoints and a mixture through the shared calculation."""
    ref = _pa_finish_reference(grade)
    polymer = fsum(principal.values())
    moisture = polymer * ref.gate_water_kg
    load = _pa_open_burning_load(
        ref, principal_by_origin_kg=principal, finish_dose_kg=dose, water_kg=moisture
    )
    flows = load.biosphere_amounts
    expected_elements = {
        element: dose * mass
        + polymer * {"C": 72 / 113, "H": 11 / 113, "N": 14 / 113, "O": 16 / 113}.get(element, 0)
        for element, mass in ref.retained.elements_kg.items()
    }
    assert load.elements_kg == pytest.approx(expected_elements)
    wet = polymer + dose * fsum(ref.retained.compounds_kg.values()) + moisture
    assert load.wet_mass_kg == pytest.approx(wet)
    assert load.incoming_water_kg == pytest.approx(moisture)

    # Reconstruct the final elemental ledger from serialized native amounts.
    output = dict.fromkeys(expected_elements, 0.0)
    emitted_gas_mass = 0.0
    expected_keys = set()
    for name, atoms in _GAS_ATOMS.items():
        names = (
            (f"{name}, fossil", f"{name}, non-fossil")
            if name in ("Carbon dioxide", "Carbon monoxide", "Methane")
            else (name,)
        )
        mass = fsum(flows[part, "kilogram", _AIR] for part in names)
        expected_keys.update((part, "kilogram", _AIR) for part in names)
        emitted_gas_mass += mass
        for element, atomic_mass in atoms.items():
            output[element] += mass * atomic_mass / fsum(atoms.values())
    pah_carbon = fsum(mass * carbon for mass, carbon in _PAH_SOURCE) / fsum(
        mass for mass, _ in _PAH_SOURCE
    )
    pah = flows[_PAH, "kilogram", _AIR]
    output["C"] += pah * pah_carbon
    output["H"] += pah * (1 - pah_carbon)
    emitted_gas_mass += pah
    expected_keys.add((_PAH, "kilogram", _AIR))
    emitted_water = 1000 * flows["Water", "cubic meter", _AIR]
    output["H"] += emitted_water / 9
    output["O"] += emitted_water * 8 / 9
    # Source N2 is reported as a mass-balance trace.
    nitrogen = load.air_compounds_kg["Nitrogen"]
    output["N"] += nitrogen
    assert nitrogen > 0
    assert load.formed_water_kg > 0
    assert load.atmospheric_oxygen_kg > 0
    soil = {
        element: flows[name, "kilogram", _SOIL]
        for element, name in {
            "C": "Carbon",
            "N": "Nitrogen",
            "S": "Sulfur",
            "P": "Phosphorus",
            "Na": "Sodium I",
        }.items()
    }
    for element, mass in soil.items():
        output[element] += mass
    # Independent complete oxide masses: C+carbonate, SO3, NO3, P2O5, Na2O.
    soil_mass = (
        3 * soil["C"]
        + 2.5 * soil["S"]
        + 62 / 14 * soil["N"]
        + 70.97 / 30.97 * soil["P"]
        + 31 / 23 * soil["Na"]
    )
    output["O"] += soil_mass - fsum(soil.values())
    air_p = flows["Phosphorus", "kilogram", _AIR]
    air_na = flows["Sodium I", "kilogram", _AIR]
    output["P"] += air_p
    output["Na"] += air_na
    particulate = air_p * 70.97 / 30.97 + air_na * 31 / 23
    output["O"] += particulate - air_p - air_na
    input_elements = dict(expected_elements)
    input_elements["H"] += moisture / 9
    input_elements["O"] += (
        moisture * 8 / 9 + flows["Oxygen", "kilogram", ("natural resource", "in air")]
    )
    atmospheric_nitrogen = wet * (0.000810 * 14 / 46 + 0.00011625 * 28 / 44)
    input_elements["N"] += atmospheric_nitrogen
    assert output == pytest.approx(input_elements, abs=2e-15)
    assert emitted_gas_mass + emitted_water + nitrogen + soil_mass + particulate == pytest.approx(
        wet + load.atmospheric_oxygen_kg + load.atmospheric_nitrogen_kg, abs=3e-15
    )
    assert load.residue_mass_kg == pytest.approx(soil_mass)
    assert load.atmospheric_nitrogen_kg == pytest.approx(atmospheric_nitrogen)
    assert nitrogen == pytest.approx(
        (expected_elements["N"] - soil["N"]) * (1 - 0.37383241657031102)
    )
    assert load.air_oxide_elements_kg == pytest.approx(
        {"P": air_p, "Na": air_na, "O": particulate - air_p - air_na}
    )
    assert load.residue_elements_kg == pytest.approx({**soil, "O": soil_mass - fsum(soil.values())})

    # Source factors/compartments remain visible independently of closure.
    for name, factor in {
        "Benzene": 0.00097975,
        "Phenol": 0.000014799342788831124,
        "Styrene": 0.0005275,
        "Toluene": 0.000372,
        "Formaldehyde": 0.00044365,
        "Acetaldehyde": 0.0004284,
        "Ethyl benzene": 0.00032746373845053444,
        _PAH: 0.00034430210526037831,
        "Dinitrogen monoxide": 0.00011625,
    }.items():
        assert flows[name, "kilogram", _AIR] == pytest.approx(wet * factor)
    assert load.air_compounds_kg["Carbon monoxide"] == pytest.approx(wet * 0.0386)
    assert load.air_compounds_kg["Methane"] == pytest.approx(wet * 0.0059)
    for element, retained_fraction in {
        "C": 0.0050150854170955386,
        "S": 0.8564343815936636,
        "N": 0.010012170059468751,
        "P": 0.99312931599358296,
        "Na": 0.97536945812807858,
    }.items():
        assert soil[element] == pytest.approx(expected_elements[element] * retained_fraction)
    assert flows["Sulfur dioxide", "kilogram", _AIR] == pytest.approx(
        2 * (expected_elements["S"] - soil["S"])
    )
    assert flows["Nitrogen oxides", "kilogram", _AIR] == pytest.approx(
        (expected_elements["N"] - soil["N"]) * 46 / 14 * 0.37383241657031102 + wet * 0.000810
    )
    for name, share in (
        ("Particulate Matter, < 2.5 um", 0.52),
        ("Particulate Matter, > 10 um", 0.20),
        ("Particulate Matter, > 2.5 um and < 10um", 0.28),
    ):
        assert flows[name, "kilogram", _AIR] == pytest.approx(particulate * share)
        expected_keys.add((name, "kilogram", _AIR))
    heat = (
        34.016 * expected_elements["C"]
        + 124.265 * expected_elements["H"]
        - 9.8324 * expected_elements["O"]
        + 6.276 * expected_elements["N"]
        + 19.079 * expected_elements["S"]
    )
    assert flows["Heat, waste", "megajoule", _AIR] == pytest.approx(heat)
    assert load.heat_potential_mj == pytest.approx(heat)
    assert flows["Non-hazardous waste disposed", "kilogram", _INDICATOR] == pytest.approx(wet)
    assert flows["Waste mass, total, placed in landfill", "kilogram", _INDICATOR] == pytest.approx(
        soil_mass
    )
    assert flows["Organic carbon, placed in landfill", "kilogram", _INDICATOR] == pytest.approx(
        soil["C"]
    )
    expected_keys.update(
        (name, "kilogram", _SOIL)
        for name in ("Carbon", "Nitrogen", "Sulfur", "Phosphorus", "Sodium I")
    )
    expected_keys.update((name, "kilogram", _AIR) for name in ("Phosphorus", "Sodium I"))
    expected_keys.update(
        (name, "kilogram", _INDICATOR)
        for name in (
            "Non-hazardous waste disposed",
            "Waste mass, total, placed in landfill",
            "Organic carbon, placed in landfill",
        )
    )
    expected_keys.update(
        {
            ("Water", "cubic meter", _AIR),
            ("Heat, waste", "megajoule", _AIR),
            ("Oxygen", "kilogram", ("natural resource", "in air")),
        }
    )
    assert set(flows) == expected_keys  # No duplicate PAH16, halogens or captured-treatment duties.
    assert all(amount >= 0 for amount in flows.values())

    # The actual emitted group's denominator includes all auxiliary carbon.
    principal_c: dict[_CarbonOrigin, float] = {
        origin: mass * 72 / 113 for origin, mass in principal.items()
    }
    assert load.principal_carbon_kg == pytest.approx(principal_c)
    assert load.auxiliary_carbon_kg == pytest.approx(
        {
            "fossil": dose * ref.retained.fossil_carbon_kg,
            "biogenic": dose * ref.retained.biogenic_carbon_kg,
        }
    )
    for key, fractions in load.principal_origin_fractions.items():
        name = key[0]
        origins: tuple[_CarbonOrigin, ...]
        if name.endswith(", non-fossil"):
            origins = ("biogenic", "atmospheric_dac")
            denominator = (
                fsum(principal_c.get(origin, 0) for origin in origins)
                + dose * ref.retained.biogenic_carbon_kg
            )
        elif name.endswith(", fossil"):
            origins = ("fossil",)
            denominator = principal_c.get("fossil", 0) + dose * ref.retained.fossil_carbon_kg
        else:
            origins = tuple(principal_c)
            denominator = expected_elements["C"]
        assert fractions == pytest.approx(
            {origin: principal_c.get(origin, 0) / denominator for origin in origins}
            if denominator
            else {}
        )
        assert fsum(fractions.values()) <= 1 + 1e-15


@pytest.mark.parametrize("grade", ["pa6", "pa66"])
def test_actual_burning_scales_the_whole_receiving_event(grade: str) -> None:
    """Preserve scaling while per-output principal fractions stay fixed."""
    ref = _pa_finish_reference(grade)
    original = _pa_open_burning_load(
        ref,
        principal_by_origin_kg={"fossil": 0.4, "atmospheric_dac": 0.6},
        finish_dose_kg=2.5,
        water_kg=ref.gate_water_kg,
    )
    scale = 3.7
    scaled = _pa_open_burning_load(
        ref,
        principal_by_origin_kg={"fossil": scale * 0.4, "atmospheric_dac": scale * 0.6},
        finish_dose_kg=scale * 2.5,
        water_kg=scale * ref.gate_water_kg,
    )
    assert scaled.biosphere_amounts == pytest.approx(
        {key: scale * value for key, value in original.biosphere_amounts.items()}
    )
    assert scaled.air_compounds_kg == pytest.approx(
        {key: scale * value for key, value in original.air_compounds_kg.items()}
    )
    for key, fractions in original.principal_origin_fractions.items():
        assert scaled.principal_origin_fractions[key] == pytest.approx(fractions)


@pytest.mark.parametrize("water", [0.0, 0.05])
def test_process_nitrogen_closes_without_material_nitrogen(water: float) -> None:
    """Check the shared nitrogen ledger."""
    elements = {"C": 12 / 14, "H": 2 / 14, "O": 0.0, "N": 0.0, "P": 0.0, "S": 0.0, "Na": 0.0}
    load = _open_burning_load(
        _OpenBurningMaterial(
            wet_mass_kg=1 + water,
            water_kg=water,
            burnable_elements_kg=elements,
            inert_elements_kg={},
            principal_carbon_kg={"fossil": elements["C"]},
            auxiliary_carbon_kg={},
            heat_potential_mj=40.0,
        )
    )
    wet = 1 + water
    flows = load.biosphere_amounts
    nox = flows["Nitrogen oxides", "kilogram", _AIR]
    n2o = flows["Dinitrogen monoxide", "kilogram", _AIR]
    assert nox == pytest.approx(wet * 810 / 1e6)
    assert n2o == pytest.approx(wet * 116.25 / 1e6)
    assert load.air_compounds_kg["Nitrogen"] == 0.0
    assert flows["Nitrogen", "kilogram", _SOIL] == 0.0
    output_nitrogen = (
        nox * 14 / 46
        + n2o * 28 / 44
        + load.air_compounds_kg["Nitrogen"]
        + flows["Nitrogen", "kilogram", _SOIL]
    )
    assert load.atmospheric_nitrogen_kg == pytest.approx(
        wet * (810 * 14 / 46 + 116.25 * 28 / 44) / 1e6
    )
    assert elements["N"] + load.atmospheric_nitrogen_kg == pytest.approx(output_nitrogen)
    assert ("Nitrogen", "kilogram", _AIR) not in flows


def _check_reference_atom_balance(load: _OpenBurningLoad) -> None:
    """Reconcile final element masses and overlapping particulate indicators."""
    pcb_mass = 144 / 0.441533266
    pcb_x = (pcb_mass - 154) / 34.453
    molecules = {
        **_GAS_ATOMS,
        "Water": {"H": 2, "O": 16},
        "Nitrogen": {"N": 28},
        "Hydrochloric acid": {"H": 1, "Cl": 35.453},
        "Polychlorinated biphenyls": {"C": 144, "H": 10 - pcb_x, "Cl": 35.453 * pcb_x},
        "Dioxins, measured as 2,3,7,8-tetrachlorodibenzo-p-dioxin": {
            "C": 144,
            "H": 4,
            "Cl": 4 * 35.453,
            "O": 32,
        },
    }
    pah_c = fsum(mass * c for mass, c in _PAH_SOURCE) / fsum(mass for mass, _ in _PAH_SOURCE)
    molecules[_PAH] = {"C": pah_c, "H": 1 - pah_c}
    coordinates = set(load.elements_kg) | {"H", "O", "N"}
    for element in coordinates:
        incoming = load.elements_kg.get(element, 0.0)
        if element == "H":
            incoming += load.incoming_water_kg / 9
        if element == "O":
            incoming += load.incoming_water_kg * 8 / 9 + load.atmospheric_oxygen_kg
        if element == "N":
            incoming += load.atmospheric_nitrogen_kg
        outgoing = (
            fsum(
                amount * molecules[name].get(element, 0.0) / fsum(molecules[name].values())
                for name, amount in load.air_compounds_kg.items()
            )
            + load.air_oxide_elements_kg.get(element, 0.0)
            + load.residue_elements_kg.get(element, 0.0)
        )
        assert incoming == pytest.approx(outgoing, rel=3e-15, abs=3e-16), element
    total_output = (
        fsum(load.air_compounds_kg.values())
        + fsum(load.air_oxide_elements_kg.values())
        + load.residue_mass_kg
    )
    assert (
        load.wet_mass_kg + load.atmospheric_nitrogen_kg + load.atmospheric_oxygen_kg - total_output
        == pytest.approx(load.source_rounding_kg, rel=0, abs=2e-15)
    )
    assert load.residue_mass_kg == pytest.approx(fsum(load.residue_elements_kg.values()))


def test_actual_tps_chlorine_closes_without_extra_pm_oxygen() -> None:
    """Keep the element mass balance for an admitted chlorine-bearing TPS formulation."""
    services = {row["provider_service_id"]: row for row in load_provider_catalogue()["services"]}
    material = _tps_burning_material(_tps_material_load(services, "tps_fossil"))
    load = _open_burning_load(material)
    elements, wet = material.burnable_elements_kg, material.wet_mass_kg
    _check_reference_atom_balance(load)
    flows = load.biosphere_amounts
    pcb = "Polychlorinated biphenyls"
    dioxin = "Dioxins, measured as 2,3,7,8-tetrachlorodibenzo-p-dioxin"
    assert flows[pcb, "kilogram", _AIR] == pytest.approx(wet * 1.1189198274953115e-7)
    assert flows[dioxin, "kilogram", _AIR] == pytest.approx(
        wet * 0.001683 * (1e6 * elements["Cl"] / wet) ** 1.296881 / 1e12
    )
    assert flows[dioxin, "kilogram", _SOIL] == pytest.approx(flows[dioxin, "kilogram", _AIR] / 3.6)
    tcdd_mass = 144 + 4 + 4 * 35.453 + 32
    assert load.residue_elements_kg["C"] == pytest.approx(
        flows["Carbon", "kilogram", _SOIL] + flows[dioxin, "kilogram", _SOIL] * 144 / tcdd_mass
    )
    assert (
        flows["Organic carbon, placed in landfill", "kilogram", _INDICATOR]
        == load.residue_elements_kg["C"]
    )
    assert load.residue_elements_kg["Cl"] == pytest.approx(
        flows["Chloride", "kilogram", _SOIL]
        + flows[dioxin, "kilogram", _SOIL] * 4 * 35.453 / tcdd_mass
    )
    pm = fsum(load.air_oxide_elements_kg.values())
    pm += elements["Cl"] * (1 - 0.33288933288933293) * 1.4513017234084562
    assert flows["Particulate Matter, < 2.5 um", "kilogram", _AIR] == pytest.approx(0.52 * pm)
    assert flows["Particulate Matter, > 10 um", "kilogram", _AIR] == pytest.approx(0.20 * pm)
    assert flows["Particulate Matter, > 2.5 um and < 10um", "kilogram", _AIR] == pytest.approx(
        0.28 * pm
    )


@pytest.mark.parametrize("chlorine,water", [(1e-12, 0.0), (0.0, 50.0)])
def test_inadmissible_factor_complements_are_rejected(chlorine: float, water: float) -> None:
    """Reject deficient chlorine and water-diluted fuel at the source factors."""
    with pytest.raises(ValueError, match="inadmissible source complements"):
        _open_burning_load(
            _OpenBurningMaterial(
                wet_mass_kg=1 + water + chlorine,
                water_kg=water,
                burnable_elements_kg={"C": 12 / 14, "H": 2 / 14, "Cl": chlorine},
                inert_elements_kg={},
                principal_carbon_kg={"fossil": 12 / 14},
                auxiliary_carbon_kg={},
                heat_potential_mj=40.0,
            )
        )
