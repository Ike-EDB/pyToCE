"""Calculate initial RoW sanitary placement and actual later element releases.

The material calculation supplies its phase quantities on a common basis.
This selected Landfill2020 site excludes degradation and leachate treatment,
combustion, support purchases and the five separate open-dump climates. The
calling inventory calculation assigns the native elementary-flow identities.
"""

from collections.abc import Mapping
from math import fsum

_AIR = ("air", "non-urban air or from high stacks")
_LONG = ("water", "ground-, long-term")
_WASTE = ("inventory indicator", "waste")
_LAND = ("natural resource", "land")


def _sanitary_site_emissions(
    *,
    wet_mass_kg: float,
    carbon_kg: float,
    later_elements_kg: Mapping[str, float],
    gas_elements_kg: Mapping[str, float],
) -> dict[tuple[str, tuple[str, ...]], float]:
    """Return initial site and later-release descriptors on the actual waste basis.

    All inputs are kg on one caller reference. Later C/N/S are required;
    the caller also supplies Zn for filled SBR or P/Na for PA finish. Gas Zn/Na
    are direct elemental releases, separate from combustible gas and particles.
    Results pair native flow names and compartments with kg species or indicator,
    m2-year occupation, or m2 transformation. Moisture belongs in wet
    placement mass, not in a second water purchase or a new degradation step.

    The original RoW sanitary site's fixed geometry and empirical species transfers apply uniformly
    across the 12 regions. Downstream sludge/ash
    disposal, short leachate, site utilities and reaction heat retain their own
    owners; this function supplies none of those duties a second time.
    """
    # Landfill2020 Infra: 90,000 m2 filled to 20 m at 1,000 kg/m3, with an
    # 18,000 m2 access road. Allocation is per kg actual wet waste placed.
    area, road, capacity = 90000, 18000, 90000 * 20 * 1000
    # Initial occupation lasts 5 years during construction, 30 during operation,
    # 5 as shrub land and 185 for the road. Transformation counts each area once.
    land = {
        "Occupation, construction site": 5 * area / capacity,
        "Occupation, dump site": 30 * area / capacity,
        "Occupation, shrub land, sclerophyllous": 5 * area / capacity,
        "Occupation, traffic area, road network": 185 * road / capacity,
        "Transformation, from dump site, sanitary landfill": area / capacity,
        "Transformation, from pasture, man made": (area + road) / capacity,
        "Transformation, from shrub land, sclerophyllous": area / capacity,
        "Transformation, to dump site, sanitary landfill": area / capacity,
        "Transformation, to forest, unspecified": area / capacity,
        "Transformation, to shrub land, sclerophyllous": area / capacity,
        "Transformation, to traffic area, road network": road / capacity,
    }
    later = _landfill_leachate_emissions(later_elements_kg, compartment=_LONG)
    # MSWLF T(row)=B(row)*L(row) is the actual gas-phase element release. Zn/Na bind directly to
    # native rural-air species. PA gas-phase phosphorus is zero.
    gas = {
        (name, _AIR): gas_elements_kg[element]
        for element, name in (
            ("Zn", "Zinc II"),
            ("Na", "Sodium I"),
            ("Al", "Aluminium III"),
            ("Co", "Cobalt II"),
            ("Si", "Silicon"),
            ("Ca", "Calcium II"),
            ("Ti", "Titanium ion"),
            ("K", "Potassium I"),
            ("Mg", "Magnesium"),
        )
        if element in gas_elements_kg
    }
    return {
        **{(name, _LAND): value * wet_mass_kg for name, value in land.items()},
        **later,
        ("Organic carbon, placed in landfill", _WASTE): carbon_kg,
        ("Waste mass, total, placed in landfill", _WASTE): wet_mass_kg,
        ("Non-hazardous waste disposed", _WASTE): wet_mass_kg,
        **gas,
    }


def _landfill_leachate_emissions(
    elements_kg: Mapping[str, float], *, compartment: tuple[str, ...]
) -> dict[tuple[str, tuple[str, ...]], float]:
    """Speciate actual Landfill2020 leachate at its short or later groundwater outlet.

    The sanitary site uses this calculation for untreated later leachate; open
    dumps with positive infiltration use it for both periods. C/N/S are actual
    kg element in that phase, with source-qualified Zn or P/Na supplied by the
    material owner. Analytical carbon indicators overlap and are not extra mass.
    Wastewater treatment, climate selection, and provider selection have separate owners.
    """
    carbon, nitrogen, sulfur = (elements_kg[element] for element in ("C", "N", "S"))
    # MSWLF C94:C97: mg N/L in ammonium, organic N, nitrite and nitrate.
    # Their normalized weights partition nitrogen in the selected phase.
    # Nominal g/mol ratios 18/14 (NH4), 46/14 (NO2) and 62/14 (NO3)
    # give kg emitted compound/kg N.
    nweights = (115.79732353778977, 243.33844006164554, 2.4662120743304703, 3.5355339059327378)
    n = tuple(value / fsum(nweights) for value in nweights)
    # MSWLF C90:C91: mg S/L in sulfate and sulfide. Nominal 96/32 and 34/32
    # convert their allocated kg S to kg SO4 and H2S, respectively.
    s = 40.00326505167857 / (40.00326505167857 + 2.9443208588856526)
    later = {
        "TOC, Total Organic Carbon": carbon,
        "DOC, Dissolved Organic Carbon": carbon,
        # MSWLF B86 and B85 are kg O2/kg TOC analytical ratios for COD and BOD, overlapping with
        # TOC/DOC.
        "COD, Chemical Oxygen Demand": 1.09375 * carbon,
        "BOD5, Biological Oxygen Demand": 0.25892857142857145 * carbon,
        "Ammonium": n[0] * nitrogen * 18 / 14,
        "Nitrogen, organic bound": n[1] * nitrogen,
        "Nitrite": n[2] * nitrogen * 46 / 14,
        "Nitrate": n[3] * nitrogen * 62 / 14,
        "Sulfate": s * sulfur * 96 / 32,
        "Hydrogen sulfide": (1 - s) * sulfur * 34 / 32,
    }
    # Existing SBR Zn and PA Na use elemental kg. PA P uses MSWLF G99/L99's
    # nominal phosphate/P molecular ratio (31+4*16)/31=95/31, distinct from
    # the Wastewater2021 landfarming receiver's 94.97/30.97 convention.
    for element, name, factor in (
        ("Zn", "Zinc II", 1.0),
        ("P", "Phosphate", 95 / 31),
        ("Na", "Sodium I", 1.0),
        # UPR's actual glass, silica and cobalt use native elemental kg.
        # Their phase amounts already contain the site-specific release.
        ("B", "Boron", 1.0),
        ("Si", "Silicon", 1.0),
        ("Ca", "Calcium II", 1.0),
        ("Ti", "Titanium ion", 1.0),
        ("K", "Potassium I", 1.0),
        ("Mg", "Magnesium", 1.0),
        ("Al", "Aluminium III", 1.0),
        ("Co", "Cobalt II", 1.0),
    ):
        if element in elements_kg:
            later[name] = elements_kg[element] * factor
    return {(name, compartment): value for name, value in later.items()}
