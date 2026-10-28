"""Municipal co-treatment air, treated-water and environmental-heat equations.

The municipal incineration calculation supplies phase masses and original
operating quantities. These equations access no native database or receiving load.
"""

_Emission = tuple[str, str, tuple[str, ...]]
_URBAN = ("air", "urban air close to ground")
_SURFACE = ("water", "surface water")

# Native element-dependent outputs for the admitted reacting plastic matrices. Target emissions use
# the actual material's elements.
_NATIVE_ELEMENT_FLOWS = {
    "Cyanide": "N",
    "Dinitrogen monoxide": "N",
    "Nitrate": "N",
    "Sulfur dioxide": "S",
    "Sulfate": "S",
    "Sodium I": "Na",
    # UPR accelerator Co uses the original MIXED furnace species factors.
    "Cobalt II": "Co",
    # Original MIXED furnace has positive air and water Zn outputs. Actual
    # filled SBR scales those source species factors by its retained Zn load.
    "Zinc II": "Zn",
}

# Doka (2013), sections 2.4/2.5, allocate these process traces to individual waste fractions. The
# other co-treated wastes supply that reactor chemistry. Count their nominal C/H once.
_TRACE_ORGANICS = {
    "Benzene": (6, 6, 0, 0),
    "Benzo(a)pyrene": (20, 12, 0, 0),
    "Hexachlorobenzene": (6, 0, 0, 6),
    "Pentachlorobenzene": (6, 1, 0, 5),
    "Pentachlorophenol": (6, 1, 1, 5),
    "Toluene": (7, 8, 0, 0),
}
_NMVOC = "NMVOC, non-methane volatile organic compounds"
_PROCESS_FLOWS = frozenset(_TRACE_ORGANICS) | {
    "Carbon monoxide, fossil",
    "Methane, fossil",
    _NMVOC,
    "Particulate Matter, < 2.5 um",
    "Particulate Matter, > 2.5 um and < 10um",
    "Dioxins, measured as 2,3,7,8-tetrachlorodibenzo-p-dioxin",
}


def municipal_emissions(
    source: dict[_Emission, float],
    ratio: dict[str, float],
    duties: dict[str, float],
    drivers: dict[str, float],
    quench: float,
    *,
    elements: dict[str, float],
    source_elements: dict[str, float],
    phase_air: dict[str, float],
    phase_water: dict[str, float],
    moisture: float,
    wet_mass: float,
    aqueous_auxiliary: dict[str, float],
    heat_potential_mj: float,
) -> tuple[dict[_Emission, float], dict[str, float], dict[_Emission, float]]:
    """Close municipal C, water and heat using the calculated operating quantities.

    Inputs are actual kg and original composition in kg/kg; phase_air/phase_water
    already contain actual elemental kg on the supplied material basis.
    Controlled co-treatment traces retain their source allocation. Reconstructed
    carbon/water/heat outputs have no donor-waste uncertainty distribution.
    """
    emissions = {
        key: value * ratio["process"] for key, value in source.items() if key[0] in _PROCESS_FLOWS
    }
    native_factors = {key: ratio["process"] for key in emissions}
    for name in ("Ammonia", "Nitrogen oxides"):
        key = (name, "kilogram", _URBAN)
        emissions[key] = source[key] * ratio["denox"]
        native_factors[key] = ratio["denox"]
    # Native factors already convert elements to named species. Actual PA
    # finish, epoxy hardener and UPR/SBR additives supply their own elements.
    # Cyanide and N2O use waste N; NOx/NH3 above also include air/reagent duty.
    for key, value in source.items():
        element = _NATIVE_ELEMENT_FLOWS.get(key[0])
        if element is not None and elements.get(element, 0.0) != 0:
            emissions[key] = value * elements[element] / source_elements[element]
            native_factors[key] = elements[element] / source_elements[element]
    if "P" in elements:
        # Direct MSWI phase outputs: PO4 is 95/31 kg per kg P; the native urban-air flow reports
        # elemental P.
        emissions["Phosphorus", "kilogram", _URBAN] = phase_air["P"]
        emissions["Phosphate", "kilogram", _SURFACE] = phase_water["P"] * 95 / 31
    # UPR/road organosiloxane Si follows source phases in elemental kg.
    # Bulk glass/silica bypasses these reactions and reaches bottom ash.
    if "Si" in elements:
        emissions["Silicon", "kilogram", _URBAN] = phase_air["Si"]
        emissions["Silicon", "kilogram", _SURFACE] = phase_water["Si"]
    co = emissions["Carbon monoxide, fossil", "kilogram", _URBAN]
    methane = emissions["Methane, fossil", "kilogram", _URBAN]
    voc = emissions[_NMVOC, "kilogram", _URBAN]
    ammonia = emissions["Ammonia", "kilogram", _URBAN]
    nox = emissions["Nitrogen oxides", "kilogram", _URBAN]
    cyanide = emissions.get(("Cyanide", "kilogram", _URBAN), 0.0)
    nitrous = emissions.get(("Dinitrogen monoxide", "kilogram", _URBAN), 0.0)
    trace_c = trace_h = trace_energy = 0.0
    for name, (c, h, o, cl) in _TRACE_ORGANICS.items():
        # Doka trace formula convention C 12/H 1/O 16/Cl 35.45 g/mol;
        # Michel (1938), via Doka (2003), Part II, p. 88:
        # C/H/O HHV coefficients are MJ/kg element.
        molar = 12 * c + h + 16 * o + 35.45 * cl
        amount = emissions[name, "kilogram", _URBAN]
        trace_c += amount * 12 * c / molar
        trace_h += amount * h / molar
        trace_energy += amount * (34.016 * 12 * c + 124.265 * h - 9.8324 * 16 * o) / molar
    carbon_air = phase_air["C"]
    # Other species retain C once: CO 12/28, CH4 12/16 and nominal
    # NMVOC CH1.85 (12/13.85). Remaining C becomes CO2 (44/12).
    other_carbon = co * 12 / 28 + methane * 12 / 16 + voc * 12 / 13.85 + trace_c + cyanide * 12 / 26
    emissions["Carbon dioxide, fossil", "kilogram", _URBAN] = (carbon_air - other_carbon) * 44 / 12
    carbon_water = phase_water["C"]
    for name, factor in (
        ("DOC, Dissolved Organic Carbon", 1),
        ("TOC, Total Organic Carbon", 1),
        ("COD, Chemical Oxygen Demand", 32 / 12),
        ("BOD5, Biological Oxygen Demand", 32 / 12),
    ):
        # Overlapping carbon/O2 indicators use elemental oxidation:
        # C + O2 -> CO2, or 32 kg O2 per 12 kg C.
        emissions[name, "kilogram", _SURFACE] = carbon_water * factor
    # Fresh scrubber HCl releases 35.45/36.45 kg Cl per kg active HCl to water.
    # The retained native process-water meter supplies the water balance below.
    emissions["Chloride", "kilogram", _SURFACE] = duties["HCl"] * 35.45 / 36.45
    for element, name in (("Na", "Sodium I"), ("Ca", "Calcium II")):
        mass = aqueous_auxiliary[element]
        if mass != 0:
            key = (name, "kilogram", _SURFACE)
            emissions[key] = emissions.get(key, 0.0) + mass
    # Retain the native flue-gas-cleaning water meter
    # and subtract water retained in the quenched bottom ash.
    process_water = duties["water"] - quench
    # NH3-H 3/17 joins the reacted-H pool; slip and residual organic H leave it. Reaction H makes 9
    # kg water/kg H. The 3 kg/kg NH3 dilution is already contained in the process meter.
    reacted_h = (
        elements["H"]
        + (duties["NH3"] - ammonia) * 3 / 17
        - methane * 4 / 16
        - voc * 1.85 / 13.85
        - trace_h
    )
    reaction_water = 9 * reacted_h
    # Original MSWI process-water partition 23/35 air, 12/35 surface;
    # density 1000 kg/m3 converts the existing elementary water quantities.
    emissions["Water", "cubic meter", _URBAN] = (
        moisture + reaction_water + process_water * 23 / 35
    ) / 1000
    emissions["Water", "cubic meter", _SURFACE] = process_water * 12 / 35 / 1000
    emissions["Oxygen", "kilogram", ("natural resource", "in air")] = drivers["oxygen"]
    emissions["Non-hazardous waste disposed", "kilogram", ("inventory indicator", "waste")] = (
        wet_mass
    )
    # The material owner supplies whole-input gross heat, including admitted
    # component references. This keeps UPR glass/silica oxygen out of Michel's (via Muñoz et al.
    # (2007), Eq. 9)
    # organic correlation. Municipal2020 energy!C95/C97 retain the nominal
    # 2.2 MJ/kg latent-water deduction.
    hhv = heat_potential_mj
    lhv = hhv - 2.2 * (moisture + 9 * elements["H"])
    # energy!I36:I42: fan, boiler, air, solid handling and aqueous operation,
    # in kWh per matching m3/MJ/kg source driver; I46 gives MJ reheat/m3air.
    internal_e = (
        0.00747901579308153 * drivers["flue"]
        + 0.0022407203755096163 * lhv
        + 0.0019809514121561524 * drivers["air"]
        + 0.009163023516161654 * wet_mass
        + 0.009163023516161654 * wet_mass
        + 0.0066177392061167505 * wet_mass
        + 0.5147153409931297 * drivers["aqueous"]
    )
    internal_h = 0.06684411052212387 * drivers["air"]
    # The source electricity efficiency 0.1584 and 3.6 MJ/kWh give internal
    # net energy only: no export, avoided product or recovered-energy credit.
    net_e = 0.1584 * lhv - 3.6 * internal_e
    # Chase (1998), via NIST Chemistry WebBook, NIST-JANAF formation enthalpies, kJ/mol divided by
    # g/mol
    # => MJ/kg. CO2−393.52,H2O(l)−285.83,CO−110.53,CH4−74.87,
    # NH3−45.90,NO2+33.10. Retain residual species energy unburned.
    residual_e = co * (393.52 - 110.53) / 28 + methane * (393.52 + 2 * 285.83 - 74.87) / 16
    residual_e += voc * (12 * 34.016 + 1.85 * 124.265) / 13.85 + trace_energy
    # CN retains nominal C/N calorific energy; N2O formation is +82.05 kJ/mol.
    residual_e += cyanide * (34.016 * 12 + 6.276 * 14) / 26
    ammonia_e = (duties["NH3"] - ammonia) * (1.5 * 285.83 - 45.90) / 17
    heat = (
        hhv * phase_air["C"] / elements["C"]
        - residual_e
        + ammonia_e
        - nox * 33.10 / 46
        - nitrous * 82.05 / 44
        + duties["reheat"]
        - net_e
    )
    # energy!F74:H75: empirical environmental-heat split using 90 -> 15 deg C air, 1005 J/kg/K, 6 m3
    # at 1.293 kg/m3; water 30 -> 7 deg C, 4182 J/kg/K, 1.39 kg. Use actual incoming water/kg wet
    # material. This heat split is nonlinear in mixture moisture.
    air_heat = (90 - 15) * 1005 * 6 * 1.293
    water_heat = (30 - 7) * 4182 * (1.39 + moisture / wet_mass)
    air_share = air_heat / (air_heat + water_heat)
    emissions["Heat, waste", "megajoule", _URBAN] = heat * air_share
    emissions["Heat, waste", "megajoule", _SURFACE] = heat * (1 - air_share)
    return (
        emissions,
        {
            "hhv_MJ": hhv,
            "lhv_MJ": lhv,
            "internal_electricity_kWh": internal_e,
            "internal_heat_MJ": internal_h,
            "net_electricity_MJ": net_e,
            "environmental_heat_MJ": heat,
            "heat_air_fraction": air_share,
            "reaction_water_kg": reaction_water,
            "process_water_kg": process_water,
        },
        native_factors,
    )
