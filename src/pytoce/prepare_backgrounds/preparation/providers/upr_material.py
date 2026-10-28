"""Fixed UPR/glass article, selective formation losses and actual material pools.

The material recipe owns principal purchase and exits. This owner adds the
admitted formulation. Nominal atom carbon and native tracked principal carbon are distinct
references, linked by an explicit accounting bridge.
"""

from dataclasses import dataclass
from math import fsum

from pytoce.download_data.sources.ecoinvent_materials import ConversionRequirement

from .leachate_load import _CarbonOrigin
from .open_burning import _organic_heat_potential

# ACMA2012 Table23: 65:535 gelcoat/other application; Vezzetti2025 slide14:
# 3% fumed silica in gelcoat. The rest is delivered principal, kg/application kg.
_PRINCIPAL_APPLICATION = 0.97 * 65 / 600 + 535 / 600
# Native orthophthalic UPR source e8a68c...: kg tracked C/kg delivered resin. The nominal
# backbone/styrene composition is recorded separately.
_DELIVERED_TRACKED_C = 0.721720221523646
# Native .16 kg maleic and .24 kg phthalic anhydride define relative repeat moles only. C12/H1/O16
# gives C7H8O4/156 and C11H10O4/206. Normalization defines a long-chain composition.
_MALEATE_MOLES = 0.16 / 98
_PHTHALATE_MOLES = 0.24 / 148
_BACKBONE_MASS = 156 * _MALEATE_MOLES + 206 * _PHTHALATE_MOLES
_BACKBONE = {
    "C": (84 * _MALEATE_MOLES + 132 * _PHTHALATE_MOLES) / _BACKBONE_MASS,
    "H": (8 * _MALEATE_MOLES + 10 * _PHTHALATE_MOLES) / _BACKBONE_MASS,
    "O": 64 * (_MALEATE_MOLES + _PHTHALATE_MOLES) / _BACKBONE_MASS,
}
_STYRENE = {"C": 96 / 104, "H": 8 / 104, "O": 0.0}


@dataclass(frozen=True, slots=True)
class _UprMaterial:
    """Dry homogeneous retained/trimmed article on an actual principal basis.

    All masses and elements are kg, and heat is total gross MJ. The component
    map includes principal and the six retained auxiliary pools. Burnable atoms
    include dispersed cobalt and organosiloxane Si; inert atoms are bulk glass
    and fumed silica. Heat eligibility is a third, independent partition.
    Principal origin tuples use fossil, biogenic, atmospheric-DAC order.
    """

    principal_by_origin_kg: tuple[float, float, float]
    components_kg: dict[str, float]
    mass_kg: float
    water_kg: float
    principal_elements_kg: dict[str, float]
    burnable_elements_kg: dict[str, float]
    inert_elements_kg: dict[str, float]
    heat_eligible_elements_kg: dict[str, float]
    coupling_heat_mj: float
    principal_carbon_kg: tuple[float, float, float]
    principal_carbon_bridge_kg: tuple[float, float, float]
    fossil_auxiliary_carbon_kg: float
    heat_potential_mj: float


@dataclass(frozen=True, slots=True)
class _UprReference:
    """Complete formation reference per kg retained principal.

    ``application_kg`` is 1/F; ``operation_kg`` is T/F from the conversion
    record. Supply quantities are kg and exclude the separately owned principal.
    Styrene releases distinguish principal from fossil accelerator carbon.
    White spirit is withdrawn as a whole mixture; it and the incoming hardener
    water are absent from both dry material exits.
    """

    gross_principal_kg: float
    application_kg: float
    operation_kg: float
    input_mass_kg: float
    supply_amounts: dict[str, float]
    principal_styrene_air_kg: float
    accelerator_styrene_air_kg: float
    white_spirit_air_kg: float
    water_air_kg: float
    retained: _UprMaterial
    external_solid: _UprMaterial


def _upr_supplies(conversion: ConversionRequirement) -> dict[str, float]:
    """Recover fixed actual charges from the principal-owned normalization.

    Native resin:reinforcement is 421:632. METOX50 TDS2024 supplies 2 phr
    hardener; its nominal 34% organic peroxide and 2.3% water leave 63.7% DMP.
    ILT's admitted .25 phr accelerator is 36% Co2EH, 40% styrene and 24% white
    spirit. These kg inputs are material, independently of LCI proxies.
    """
    gross = conversion.principal_feed_kg_per_finished_kg
    application = gross / _PRINCIPAL_APPLICATION
    return {
        "reinforcement_glass": gross * 632 / 421,
        "silica": application * 0.03 * 65 / 600,
        "organic_peroxide": application * 0.02 * 0.34,
        "dmp": application * 0.02 * 0.637,
        "water": application * 0.02 * 0.023,
        "cobalt_salt": application * 0.0025 * 0.36,
        "styrene": application * 0.0025 * 0.40,
        "white_spirit": application * 0.0025 * 0.24,
    }


def _upr_material(
    conversion: ConversionRequirement,
    principal_by_origin_kg: tuple[float, float, float],
) -> _UprMaterial:
    """Assemble the fixed dry article at any homogeneous principal-origin mass.

    The selective principal styrene exit comes from ``conversion``. Its solid
    exit fixes the shared trim partition; the cured principal is nominally
    .60 copolyester/.40 styrene before that selective loss. No supplier formula,
    free carrier water, extra sizing purchase or second principal yield enters.
    """
    gross = conversion.principal_feed_kg_per_finished_kg
    exits = {row.name: row.kg_per_finished_kg for row in conversion.principal_exits}
    principal = fsum(principal_by_origin_kg)
    post_principal = gross - exits["styrene_air"]
    # One retained principal plus the admitted solid envelope share the same post-volatile
    # composition. This is the existing trim.
    retained_share = 1 / (1 + exits["post_release_solid"])
    supplies = _upr_supplies(conversion)
    styrene_fraction_released = exits["styrene_air"] / (0.40 * gross)
    components = {
        "principal": principal,
        "reinforcement_glass": supplies["reinforcement_glass"] * retained_share * principal,
        "silica": supplies["silica"] * retained_share * principal,
        "organic_peroxide": supplies["organic_peroxide"] * retained_share * principal,
        "dmp": supplies["dmp"] * retained_share * principal,
        "cobalt_salt": supplies["cobalt_salt"] * retained_share * principal,
        "styrene": supplies["styrene"]
        * (1 - styrene_fraction_released)
        * retained_share
        * principal,
    }
    principal_elements = {
        element: principal
        * (
            0.60 * gross * _BACKBONE[element]
            + (0.40 * gross - exits["styrene_air"]) * _STYRENE[element]
        )
        / post_principal
        for element in ("C", "H", "O")
    }
    tracked_carbon = (
        gross * _DELIVERED_TRACKED_C - exits["styrene_air"] * _STYRENE["C"]
    ) / post_principal
    atom_carbon = (
        0.60 * gross * _BACKBONE["C"] + (0.40 * gross - exits["styrene_air"]) * _STYRENE["C"]
    ) / post_principal

    # Retained reinforcement contains 99% UNSW nominal E-glass (54% SiO2,
    # 14% Al2O3, 22% CaO, 10% B2O3) and 1% dry sizing. The admitted sizing
    # reference is 90% PVAc, 5% condensed coupling residue, 5% glycerol.
    # No manufacturing sizing input or removed upstream water is added here.
    glass = 0.99 * components["reinforcement_glass"]
    pvac = 0.01 * 0.90 * components["reinforcement_glass"]
    coupling = 0.01 * 0.05 * components["reinforcement_glass"]
    glycerol = 0.01 * 0.05 * components["reinforcement_glass"]
    silica = glass * 0.54 + components["silica"]
    alumina, lime, boron_oxide = glass * 0.14, glass * 0.22, glass * 0.10
    # Nominal O16 and conventional Si28.085/Al26.982/Ca40.078/B10.81
    # are used in both atom numerators and oxide molecular denominators.
    inert = {
        "Si": silica * 28.085 / 60.085,
        "Al": alumina * 53.964 / 101.964,
        "Ca": lime * 40.078 / 56.078,
        "B": boron_oxide * 21.62 / 69.62,
        "O": fsum(
            (
                silica * 32 / 60.085,
                alumina * 48 / 101.964,
                lime * 16 / 56.078,
                boron_oxide * 48 / 69.62,
            )
        ),
    }
    # OSHA77's normalized 26:20.7 monomer/dimer reference represents the
    # aggregate fragments.
    peroxide = components["organic_peroxide"]
    peroxide_atoms = {
        "C": peroxide * ((26 / 46.7) * 48 / 122 + (20.7 / 46.7) * 96 / 210),
        "H": peroxide * ((26 / 46.7) * 10 / 122 + (20.7 / 46.7) * 18 / 210),
        "O": peroxide * ((26 / 46.7) * 64 / 122 + (20.7 / 46.7) * 96 / 210),
    }
    # Co2EH C16H30CoO4 uses the admitted complete C12.011/H1.008/O15.999/
    # Co58.933 convention, matching the Co-equivalent supplied-catalyst basis.
    # Its metal belongs to the burnable matrix but has no fuel coefficient.
    cobalt = components["cobalt_salt"]
    cobalt_molar_mass = 16 * 12.011 + 30 * 1.008 + 58.933 + 4 * 15.999
    cobalt_atoms = {
        "C": cobalt * 16 * 12.011 / cobalt_molar_mass,
        "H": cobalt * 30 * 1.008 / cobalt_molar_mass,
        "O": cobalt * 4 * 15.999 / cobalt_molar_mass,
    }
    # Nominal CHO formulas: DMP C10H10O4; PVAc C4H6O2; glycerol C3H8O3. Condensed coupling uses
    # C14H22O7Si2.
    dmp, styrene = components["dmp"], components["styrene"]
    auxiliary = {
        "C": fsum(
            (
                peroxide_atoms["C"],
                cobalt_atoms["C"],
                dmp * 120 / 194,
                styrene * 96 / 104,
                pvac * 48 / 86,
                glycerol * 36 / 92,
            )
        ),
        "H": fsum(
            (
                peroxide_atoms["H"],
                cobalt_atoms["H"],
                dmp * 10 / 194,
                styrene * 8 / 104,
                pvac * 6 / 86,
                glycerol * 8 / 92,
            )
        ),
        "O": fsum(
            (
                peroxide_atoms["O"],
                cobalt_atoms["O"],
                dmp * 64 / 194,
                pvac * 32 / 86,
                glycerol * 48 / 92,
            )
        ),
    }
    eligible = {
        element: principal_elements[element] + auxiliary[element] for element in ("C", "H", "O")
    }
    coupling_mass = 168 + 22 + 112 + 2 * 28.085
    burnable = {
        "C": eligible["C"] + coupling * 168 / coupling_mass,
        "H": eligible["H"] + coupling * 22 / coupling_mass,
        "O": eligible["O"] + coupling * 112 / coupling_mass,
        "Si": coupling * 2 * 28.085 / coupling_mass,
        "Co": cobalt * 58.933 / cobalt_molar_mass,
    }
    # FAA1998 measured PDMS gross19.53 MJ/kg is the admitted *whole* organosiloxane property proxy.
    # Michel (via Muñoz et al. (2007), Eq. 9) applies to the separately specified organic
    # components.
    coupling_heat = 19.53 * coupling
    fossil, biogenic, dac = principal_by_origin_kg
    carbon_bridge = atom_carbon - tracked_carbon
    return _UprMaterial(
        principal_by_origin_kg=principal_by_origin_kg,
        components_kg=components,
        mass_kg=fsum(components.values()),
        water_kg=0.0,
        principal_elements_kg=principal_elements,
        burnable_elements_kg=burnable,
        inert_elements_kg=inert,
        heat_eligible_elements_kg=eligible,
        coupling_heat_mj=coupling_heat,
        principal_carbon_kg=(
            fossil * tracked_carbon,
            biogenic * tracked_carbon,
            dac * tracked_carbon,
        ),
        principal_carbon_bridge_kg=(
            fossil * carbon_bridge,
            biogenic * carbon_bridge,
            dac * carbon_bridge,
        ),
        fossil_auxiliary_carbon_kg=auxiliary["C"] + coupling * 168 / coupling_mass,
        heat_potential_mj=_organic_heat_potential(eligible) + coupling_heat,
    )


def _upr_reference(conversion: ConversionRequirement, origin: _CarbonOrigin) -> _UprReference:
    """Attach actual auxiliary charges and withdrawals to the existing UPR exits.

    Complete white-spirit and hardener-water withdrawal precedes homogeneous
    trimming. Accelerator styrene shares the principal's fractional styrene
    loss. All input quantities are recovered from the actual recipe, not native
    supplier allocation or a second application loss.
    """
    gross = conversion.principal_feed_kg_per_finished_kg
    exits = {row.name: row.kg_per_finished_kg for row in conversion.principal_exits}
    supplies = _upr_supplies(conversion)
    retained = _upr_material(
        conversion,
        (
            1.0 if origin == "fossil" else 0.0,
            1.0 if origin == "biogenic" else 0.0,
            1.0 if origin == "atmospheric_dac" else 0.0,
        ),
    )
    solid = _upr_material(
        conversion,
        (
            exits["post_release_solid"] if origin == "fossil" else 0.0,
            exits["post_release_solid"] if origin == "biogenic" else 0.0,
            exits["post_release_solid"] if origin == "atmospheric_dac" else 0.0,
        ),
    )
    return _UprReference(
        gross_principal_kg=gross,
        application_kg=gross / _PRINCIPAL_APPLICATION,
        operation_kg=1 / conversion.finished_kg_per_native_service_kg,
        input_mass_kg=fsum((gross, *supplies.values())),
        supply_amounts=supplies,
        principal_styrene_air_kg=exits["styrene_air"],
        accelerator_styrene_air_kg=supplies["styrene"] * exits["styrene_air"] / (0.40 * gross),
        white_spirit_air_kg=supplies["white_spirit"],
        water_air_kg=supplies["water"],
        retained=retained,
        external_solid=solid,
    )
