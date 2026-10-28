"""Actual cold-plastic road marking, its container residue and retained material.

The admitted principal recipe owns the container withdrawal before MMA loss.
Silikal's 2016 formulation and the cold-coated bead reference supply the other
ingredients. Their actual chemistry is independent of manufacturing
proxies. The material record retains geological carbonate carbon separately from organic receiving
chemistry.
"""

from dataclasses import dataclass
from math import fsum

from pytoce.download_data.sources.ecoinvent_materials import ConversionRequirement

from .leachate_load import _CarbonOrigin
from .open_burning import _organic_heat_potential

# Silikal SM1.01.A R610HW (June 2016), pp.16-17: kg per kg premixed base.
# The 0.223 kg reactive binder is supplied separately by the resin stage.
# The additional 0.01 kg historical hardener is an admitted 50:50 BPO/DCHP
# reference (Silikal SDS v11, 2008, and its 2021 composition change notice).
_PREMIX_PRINCIPAL = 0.223
_SUPPLY_PER_PREMIX = {
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


@dataclass(frozen=True, slots=True)
class _RoadMaterial:
    """Wet road material with separate organic, mineral, limestone and water pools.

    Masses and element amounts are kg; gross heat is MJ. Burnable atoms include
    PDMS Si. Other minerals exclude limestone, whose C/Ca/O remain separate.
    Water is included in ``mass_kg``. Principal tuples follow fossil, biogenic,
    atmospheric-DAC order, while organic auxiliaries remain fossil.
    """

    principal_by_origin_kg: tuple[float, float, float]
    components_kg: dict[str, float]
    mass_kg: float
    water_kg: float
    principal_elements_kg: dict[str, float]
    burnable_elements_kg: dict[str, float]
    mineral_elements_kg: dict[str, float]
    limestone_elements_kg: dict[str, float]
    principal_carbon_kg: tuple[float, float, float]
    fossil_auxiliary_carbon_kg: float
    geological_carbon_kg: float
    heat_potential_mj: float


@dataclass(frozen=True, slots=True)
class _RoadReference:
    """Gross supplies and application exits per kg retained principal binder.

    All quantities are kg. ``mixing_kg`` is the factory premix throughput;
    separately dosed hardener enters ``input_mass_kg`` but not this mixing duty.
    Supplies retain the existing catalog component/service identifiers and
    exclude principal resin. Container residue leaves before the selective MMA
    release, so it has the original premix proportions, unlike the final article.
    """

    gross_principal_kg: float
    premix_kg: float
    mixing_kg: float
    input_mass_kg: float
    supply_amounts: dict[str, float]
    methyl_methacrylate_air_kg: float
    retained: _RoadMaterial
    container_residue: _RoadMaterial


def _road_supplies(conversion: ConversionRequirement) -> dict[str, float]:
    """Scale the source formulation to the principal owner's gross resin demand."""
    premix = conversion.principal_feed_kg_per_finished_kg / _PREMIX_PRINCIPAL
    return {name: amount * premix for name, amount in _SUPPLY_PER_PREMIX.items()}


def _road_load(
    principal_by_origin_kg: tuple[float, float, float],
    supplies: dict[str, float],
) -> _RoadMaterial:
    """Describe the material quantities reaching one exit.

    ``supplies`` contains only the quantities reaching this exit. Principal PMMA
    and MMA share nominal C5H8O2, so selective MMA loss preserves this elemental ratio. BPO-derived
    fragments retain their aggregate precursor atoms.
    """
    principal = fsum(principal_by_origin_kg)
    bead = supplies["others_recycled_glass_bead_manufacturing"]
    # CN101704637A Example 6: 500 kg beads + 0.060 kg PDMS + 0.180 kg 20%-active silicate solution.
    # Normalize the complete 500.240 kg wet bead, including retained carrier water.
    components = {
        "principal": principal,
        "rutile": supplies["rutile_titanium_dioxide"],
        "limestone": supplies["ground_calcium_carbonate"],
        "silica_sand": supplies["silica_sand"],
        "cristobalite": supplies["cristobalite_silica_analogue"],
        "corundum": supplies["corundum"],
        "fumed_silica": supplies["fumed_silica_activated_silica_analogue"],
        "bpo_derived": supplies["organic_peroxide_hardener_analogue"],
        "dchp": supplies["hardener_phlegmatizer_dotp_analogue"],
        "glass": bead * 500 / 500.240,
        "pdms": bead * 0.060 / 500.240,
        "silicate": bead * 0.036 / 500.240,
        "water": bead * 0.144 / 500.240,
    }
    # JRC Glass BREF (2013): normalize seven oxide weights totaling 100.07.
    # PQ N-type silicate uses a 3.22:1 SiO2:Na2O mass ratio.
    glass = components["glass"]
    silicate = components["silicate"]
    silica = fsum(
        (
            glass * 72.6 / 100.07,
            silicate * 3.22 / 4.22,
            components["silica_sand"],
            components["cristobalite"],
            components["fumed_silica"],
        )
    )
    sodium_oxide = glass * 13.6 / 100.07 + silicate / 4.22
    lime = glass * 8.6 / 100.07
    magnesia = glass * 4.1 / 100.07
    alumina = glass * 0.7 / 100.07 + components["corundum"]
    potassium_oxide = glass * 0.3 / 100.07
    sulfur_trioxide = glass * 0.17 / 100.07
    rutile = components["rutile"]
    # Nominal O16/S32 and conventional Si28.085, Na22.99, Ca40.078, Mg24.305, Al26.982, K39.0983 and
    # Ti47.867 define complete oxide masses. Glass oxide notation describes composition.
    oxides = (
        ("Si", silica, 28.085, 32),
        ("Na", sodium_oxide, 45.98, 16),
        ("Ca", lime, 40.078, 16),
        ("Mg", magnesia, 24.305, 16),
        ("Al", alumina, 53.964, 48),
        ("K", potassium_oxide, 78.1966, 16),
        ("S", sulfur_trioxide, 32, 48),
        ("Ti", rutile, 47.867, 32),
    )
    minerals = {
        element: mass * atom_mass / (atom_mass + oxygen)
        for element, mass, atom_mass, oxygen in oxides
    }
    minerals["O"] = fsum(
        mass * oxygen / (atom_mass + oxygen) for _, mass, atom_mass, oxygen in oxides
    )
    # CaCO3 keeps its actual Ca/C/O using the same Ca40.078/C12/O16 convention. Track inorganic
    # carbon separately from organic fuel and TOC (Doka 2023 pp.11,14; 2013 §2.10). This record
    # retains the carbonate's elemental composition.
    limestone = {
        element: components["limestone"] * mass / 100.078
        for element, mass in {"Ca": 40.078, "C": 12, "O": 48}.items()
    }
    principal_elements = {
        element: principal * mass / 100 for element, mass in {"C": 60, "H": 8, "O": 32}.items()
    }
    # Nominal C12/H1/O16: BPO aggregate C14H10O4/242 and DCHP C20H26O4/330 define retained atoms.
    auxiliary = {
        element: fsum(
            (
                components["bpo_derived"] * bpo / 242,
                components["dchp"] * dchp / 330,
            )
        )
        for element, bpo, dchp in (("C", 168, 240), ("H", 10, 26), ("O", 64, 64))
    }
    heat_eligible = {
        element: principal_elements[element] + auxiliary[element] for element in ("C", "H", "O")
    }
    # High-polymer PDMS C2H6OSi uses the same nominal CHO and Si28.085 weights.
    # Its Si stays in the burnable matrix, independent of the bulk mineral pool.
    pdms = {
        element: components["pdms"] * mass / 74.085
        for element, mass in {"C": 24, "H": 6, "O": 16, "Si": 28.085}.items()
    }
    burnable = {element: heat_eligible[element] + pdms[element] for element in ("C", "H", "O")}
    burnable["Si"] = pdms["Si"]
    fossil, biogenic, dac = principal_by_origin_kg
    return _RoadMaterial(
        principal_by_origin_kg=principal_by_origin_kg,
        components_kg=components,
        mass_kg=fsum(components.values()),
        water_kg=components["water"],
        principal_elements_kg=principal_elements,
        burnable_elements_kg=burnable,
        mineral_elements_kg=minerals,
        limestone_elements_kg=limestone,
        principal_carbon_kg=(fossil * 0.6, biogenic * 0.6, dac * 0.6),
        fossil_auxiliary_carbon_kg=auxiliary["C"] + pdms["C"],
        geological_carbon_kg=limestone["C"],
        # FAA's 1998 polymer calorimetry gives 19.53 MJ/kg gross for PDMS.
        # This whole-polymer transfer replaces, rather than supplements, Michel (via Muñoz et al.
        # (2007), Eq. 9)
        # for PDMS. Water and all bulk minerals have no oxidation-fuel term.
        heat_potential_mj=_organic_heat_potential(heat_eligible) + 19.53 * components["pdms"],
    )


def _road_material(
    conversion: ConversionRequirement,
    principal_by_origin_kg: tuple[float, float, float],
) -> _RoadMaterial:
    """Return retained road material at any homogeneous principal-origin mass.

    The principal owner has already removed homogeneous container residue and
    selective MMA air loss. Only the former reduces auxiliaries."""
    gross = conversion.principal_feed_kg_per_finished_kg
    exits = {row.name: row.kg_per_finished_kg for row in conversion.principal_exits}
    retained_share = 1 - exits["container_residue"] / gross
    principal = fsum(principal_by_origin_kg)
    return _road_load(
        principal_by_origin_kg,
        {
            name: amount * retained_share * principal
            for name, amount in _road_supplies(conversion).items()
        },
    )


def _road_reference(conversion: ConversionRequirement, origin: _CarbonOrigin) -> _RoadReference:
    """Attach source supplies and pre-MMA container material to admitted exits."""
    gross = conversion.principal_feed_kg_per_finished_kg
    exits = {row.name: row.kg_per_finished_kg for row in conversion.principal_exits}
    supplies = _road_supplies(conversion)
    premix = gross / _PREMIX_PRINCIPAL
    container_principal = exits["container_residue"]
    container_share = container_principal / gross
    retained_origin = (
        1.0 if origin == "fossil" else 0.0,
        1.0 if origin == "biogenic" else 0.0,
        1.0 if origin == "atmospheric_dac" else 0.0,
    )
    return _RoadReference(
        gross_principal_kg=gross,
        premix_kg=premix,
        mixing_kg=premix,
        input_mass_kg=fsum((gross, *supplies.values())),
        supply_amounts=supplies,
        methyl_methacrylate_air_kg=exits["methyl_methacrylate_air"],
        retained=_road_material(conversion, retained_origin),
        container_residue=_road_load(
            (
                container_principal * retained_origin[0],
                container_principal * retained_origin[1],
                container_principal * retained_origin[2],
            ),
            {name: amount * container_share for name, amount in supplies.items()},
        ),
    )
