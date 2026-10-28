"""Actual GY250/MXDA coating, application losses and thermal receiving loads.

The principal material recipe owns gross resin and its named exits. This owner
adds the fixed MXDA formulation and cleanup acetone. Retained MXDA is incorporated into the cured
network; its nominal atoms
and fossil carbon remain separate from the principal's ultimate carbon origin.
"""

from dataclasses import dataclass
from math import fsum

from pytoce.download_data.sources.ecoinvent_materials import ConversionRequirement

from ..chemical_residue_load import ChemicalResidueLoad
from .leachate_load import _CarbonOrigin
from .material_carbon import _EPOXY_MEAN_REPEAT_UNITS, _EPOXY_PRINCIPAL_CARBON
from .open_burning import _organic_heat_potential

# EP1838750B1 Table6, comparative3: kg MXDA per kg GY250. Both charges
# remain in the addition-cured coating; curing produces no water or CO2.
_EPOXY_HARDENER_RATIO = 15.44 / 84.56


@dataclass(frozen=True, slots=True)
class _EpoxyMaterial:
    """Coating components, CHON and thermal reference on one mass basis.

    Material and elemental quantities are kg. ``heat_potential_mj`` is Michel's (via Muñoz et al.
    (2007), Eq. 9)
    nominal gross heat. ``formed_water_kg`` describes complete oxidation of
    material hydrogen, separate from curing. The principal tuple follows
    fossil, biogenic and atmospheric-DAC order; MXDA and cleanup acetone have
    fixed fossil carbon.
    """

    principal_by_origin_kg: tuple[float, float, float]
    mxda_kg: float
    acetone_kg: float
    mass_kg: float
    elements_kg: dict[str, float]
    principal_carbon_kg: tuple[float, float, float]
    fossil_auxiliary_carbon_kg: float
    heat_potential_mj: float
    formed_water_kg: float


@dataclass(frozen=True, slots=True)
class _EpoxyReference:
    """Application and four material exits per kg retained principal GY250.

    Supply amounts are kg for MXDA and acetone, and native normal cubic meters
    for compressed air. Gross binder is the resin/MXDA mixing charge before
    preparation loss; cleanup solvent is outside that mixing throughput.
    Managed residue is the union of preparation residue and settled overspray,
    while their separate records preserve the actual manufacturing exits.
    """

    gross_principal_kg: float
    gross_binder_kg: float
    spray_volume_l: float
    supply_amounts: dict[str, float]
    retained: _EpoxyMaterial
    preparation_residue: _EpoxyMaterial
    settled_overspray: _EpoxyMaterial
    airborne_overspray: _EpoxyMaterial
    managed_residue: _EpoxyMaterial


def _epoxy_material(
    principal_by_origin_kg: tuple[float, float, float],
    mxda_kg: float,
    acetone_kg: float = 0.0,
) -> _EpoxyMaterial:
    """Calculate GY250, MXDA and cleanup-solvent elemental quantities.

    GY250 uses the EEW186 homologous-series reference C(21+18n)H(24+20n)O(4+3n),
    with n=(372-340)/284 and nominal C12/H1/O16/N14 masses. MXDA C8H12N2 and
    acetone C3H6O contribute their own elemental quantities.
    """
    principal = fsum(principal_by_origin_kg)
    n = _EPOXY_MEAN_REPEAT_UNITS
    molar_mass = 340 + 284 * n
    principal_carbon = (
        principal_by_origin_kg[0] * _EPOXY_PRINCIPAL_CARBON,
        principal_by_origin_kg[1] * _EPOXY_PRINCIPAL_CARBON,
        principal_by_origin_kg[2] * _EPOXY_PRINCIPAL_CARBON,
    )
    auxiliary_carbon = fsum((mxda_kg * 96 / 136, acetone_kg * 36 / 58))
    elements = {
        "C": fsum((*principal_carbon, auxiliary_carbon)),
        "H": fsum(
            (principal * (24 + 20 * n) / molar_mass, mxda_kg * 12 / 136, acetone_kg * 6 / 58)
        ),
        "O": fsum((principal * 16 * (4 + 3 * n) / molar_mass, acetone_kg * 16 / 58)),
        "N": mxda_kg * 28 / 136,
    }
    return _EpoxyMaterial(
        principal_by_origin_kg=principal_by_origin_kg,
        mxda_kg=mxda_kg,
        acetone_kg=acetone_kg,
        mass_kg=fsum((principal, mxda_kg, acetone_kg)),
        elements_kg=elements,
        principal_carbon_kg=principal_carbon,
        fossil_auxiliary_carbon_kg=auxiliary_carbon,
        heat_potential_mj=_organic_heat_potential(elements),
        # Nominal 2H+O -> H2O gives 18/2=9 kg formed water/kg organic H.
        formed_water_kg=9 * elements["H"],
    )


def _epoxy_residue_load(material: _EpoxyMaterial) -> ChemicalResidueLoad:
    """Pass actual dry managed residue to the complete external HWI function.

    Net heat subtracts 2.4417 MJ per kg newly formed water from the nominal
    Michel (via Doka) gross heat. Harvey (1998), NISTIR 5078 Table 1, supplies
    this pure-water vaporization reference at 25 degrees Celsius. There is no
    incoming carrier water or mineral filler.
    The receiver owns nitrogen treatment, fresh reagents, their ash, collection
    and fixed-site support. It preserves the principal subset of each C origin.
    """
    return ChemicalResidueLoad(
        wet_mass_kg=material.mass_kg,
        carrier_water_kg=0.0,
        active_elements_kg={**material.elements_kg, "S": 0.0, "Na": 0.0},
        net_energy_mj=material.heat_potential_mj - 2.4417 * material.formed_water_kg,
        nonfuel_salt_elements_kg={},
        waste_formed_water_kg=material.formed_water_kg,
        biogenic_carbon_kg=material.principal_carbon_kg[1],
        principal_carbon_kg=material.principal_carbon_kg,
    )


def _epoxy_reference(conversion: ConversionRequirement, origin: _CarbonOrigin) -> _EpoxyReference:
    """Add fixed coating auxiliaries to the principal material exits.

    The supplied conversion defines gross principal and preparation, settled
    and airborne losses per kg finished principal. Preparation loss and spray
    retention are applied once. The HMDA market supplies one kg of manufactured
    diamine per kg MXDA as an output-mass proxy.
    """
    gross = conversion.principal_feed_kg_per_finished_kg
    exits = {item.name: item.kg_per_finished_kg for item in conversion.principal_exits}
    ratio = _EPOXY_HARDENER_RATIO
    # GY250 TDS and Merck X1202 densities are 1.17 and 1.032 kg/L.
    # Calculate charge volume by ideal volume addition.
    gross_volume = gross / 1.17 + ratio * gross / 1.032
    # IERA Elmendorf1999 proportioning cleanup: 562.5 US gal acetone per
    # 1700 US gal coating. Their common volume unit cancels; .791 kg/L is
    # the admitted acetone density. All cleanup enters preparation residue.
    acetone = 0.791 * (562.5 / 1700) * gross_volume
    spray_principal = gross - exits["preparation_residue"]
    spray_volume = spray_principal / 1.17 + ratio * spray_principal / 1.032
    # EPA ETV2006 measured pump-only air: 2 SCFM at 380 psig and fluid
    # 468 g/min at 960 g/L. .028316846592 m3/ft3 converts the air volume.
    # Transfer constant air per hydraulic work to ETV2003's 2822 psig;
    # the source's separate 3 SCFM shaping air is outside airless application.
    air_per_litre = 2 * 0.028316846592 / ((468 / 960) * 380) * 2822
    # The source SCFM temperature is unspecified. The admitted engineering convention is 70F,
    # harmonized to the native 0C normal-volume reference at equal absolute pressure:
    # Ktarget/Ksource.
    normal_volume = 273.15 / (((70 - 32) * 5 / 9) + 273.15)

    def material(principal_kg: float, cleanup_kg: float = 0.0) -> _EpoxyMaterial:
        """Apply this principal origin and fixed MXDA ratio to one actual exit."""
        return _epoxy_material(
            (
                principal_kg if origin == "fossil" else 0.0,
                principal_kg if origin == "biogenic" else 0.0,
                principal_kg if origin == "atmospheric_dac" else 0.0,
            ),
            ratio * principal_kg,
            cleanup_kg,
        )

    return _EpoxyReference(
        gross_principal_kg=gross,
        gross_binder_kg=gross * (1 + ratio),
        spray_volume_l=spray_volume,
        supply_amounts={
            "mxda": ratio * gross,
            "acetone": acetone,
            "compressed_air": spray_volume * air_per_litre * normal_volume,
        },
        retained=material(1.0),
        preparation_residue=material(exits["preparation_residue"], acetone),
        settled_overspray=material(exits["settled_overspray"]),
        airborne_overspray=material(exits["airborne_overspray"]),
        managed_residue=material(
            fsum((exits["preparation_residue"], exits["settled_overspray"])), acetone
        ),
    )
