"""Keep PA fiber finish and application water on one basis.

US5472623A Table 1 Example 2 supplies the seven nominal finish fractions.
The selected molecular grades and feedstock origins are explicit engineering
references, completed by their ingredient manufacturing owners. Polymers BREF
2007 supplies the PA6/PA66 finished-principal doses and single COD/air envelope.
This module owns material, not extra manufacturing or treatment LCI.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import fsum

# RIPA/EY 2015 whole-container size, transferred to the selected 100 kg
# delivered PA concentrate payload. The exact US gallon is 3.785411784 L.
PA_FINISH_CONTAINER_CAPACITY_L = 55 * 3.785411784
# Installed proxy masses: 21 lb body, 6 oz closures and 1 oz gasket.
# One stationary reference owns both manufacturing demand and filled-cargo tare;
# dated supplier metadata cannot change these masses.
PA_FINISH_CONTAINER_PARTS_KG = {
    "hdpe_body": 21 * 0.45359237,
    "hdpe_closures": 6 * (0.45359237 / 16),
    "technical_seals": 0.45359237 / 16,
}
PA_FINISH_CONTAINER_TARE_KG = sum(PA_FINISH_CONTAINER_PARTS_KG.values())

# CIAAW atomic weights, g/mol, used throughout the admitted PA finish recipe.
# Principal PA carbon keeps its separate nominal 72/113 or 144/226 convention.
_ATOMS = {
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
    "N": 14.007,
    "P": 30.973761998,
    "S": 32.06,
    "Na": 22.98976928,
}
# Patent Example 2 proportions are nominal dry finish fractions.
_FINISH_FRACTIONS = {
    "tridecyl_stearate": 0.60,
    "diglycerol_dioleate": 0.05,
    "oleth10": 0.17,
    "peg40_hco": 0.06,
    "tea_oleyl_phosphate": 0.04,
    "sodium_dodecanesulfonate": 0.03,
    "hmda_dimeramide": 0.05,
}
# Target grades use nominal EO10/40, mono-TEA phosphate and C12 sulfonate
# formulas. The HMDA/dimer batch retains four excess C36 acids after two
# amidations at the patent's 1:6 molar feed.
_FORMULAS = {
    "tridecyl_stearate": {"C": 31, "H": 62, "O": 2},
    "diglycerol_dioleate": {"C": 42, "H": 78, "O": 7},
    "oleth10": {"C": 38, "H": 76, "O": 11},
    "peg40_hco": {"C": 137, "H": 270, "O": 49},
    "tea_oleyl_phosphate": {"C": 24, "H": 52, "N": 1, "O": 7, "P": 1},
    "sodium_dodecanesulfonate": {"C": 12, "H": 25, "Na": 1, "O": 3, "S": 1},
    "hmda_dimeramide": {"C": 222, "H": 396, "N": 2, "O": 22},
}
# Molecular descriptors depend only on the seven declared grades. Keep the
# original summation order while reusing them across independent actual loads.
_FORMULA_MOLAR_MASSES = {
    name: fsum(_ATOMS[element] * count for element, count in formula.items())
    for name, formula in _FORMULAS.items()
}
_FORMULA_OXYGEN_ATOMS = {
    name: fsum(
        factor * formula.get(element, 0)
        for element, factor in {
            "C": 4,
            "H": 1,
            "O": -2,
            "N": -3,
            "P": 5,
            "S": 6,
            "Na": 1,
        }.items()
    )
    for name, formula in _FORMULAS.items()
}
# Carbon atoms supplied by the admitted biological moieties: stearate; glycerol+oleate; oleyl;
# hydrogenated castor oil; oleyl phosphate; and dimer acid, respectively. Tridecanol, EO, TEA,
# sulfonate and HMDA retain fossil carbon. These ingredient moieties determine the fractions.
_BIOGENIC_CARBON_ATOMS = {
    "tridecyl_stearate": 18,
    "diglycerol_dioleate": 42,
    "oleth10": 18,
    "peg40_hco": 57,
    "tea_oleyl_phosphate": 18,
    "sodium_dodecanesulfonate": 0,
    "hmda_dimeramide": 216,
}
# BREF Table 9.4 textile midpoints, per kg finished principal PA:
# nominal finish kg, ester-equivalent air kg, principal caprolactam-water kg,
# inlet COD kg O2, total site water kg. Air excludes the separately accounted
# caprolactam and HTM parts, which share the parent's single native VOC flow.
_PA_APPLICATION = {
    "pa6": (0.014, 0.000610, 0.000071, 0.0023, 47.5),
    "pa66": (0.0095, 0.000170, 0.0, 0.0031, 15.0),
}
# Hasanbeigi (2010) nylon-fiber equilibrium regain at 20 C/65% RH.
# Apply kg water/kg dry principal PA; finish sorption is outside this estimate.
_MOISTURE_REGAIN = {"pa6": 0.044, "pa66": 0.043}
# Yantai EIA 2022 Figures 3.5-3/3.5-5: collected roller-application carrier water divided by
# incoming carrier water, both annual metric tons. Transfer this fraction.
_APPLICATION_COLLECTION = {"pa6": 55.121 / 1006.891, "pa66": 20.550 / 375.448}

# US20200002854A1 Table 2: equal means of three paired washed/incoming
# oil-pickup responses per grade. Apply the fixed ratio to newly applied
# surface material; embedded oil remains separate.
_WASH_REMAINING = {
    "pa6": (0.22 / 1.21 + 0.20 / 1.23 + 0.19 / 1.21) / 3,
    "pa66": (0.20 / 1.35 + 0.20 / 1.34 + 0.22 / 1.34) / 3,
}


@dataclass(frozen=True, slots=True)
class _PAFinishLoad:
    """Nonaqueous finish, element masses and carbon by origin, in kg.

    ThOD is kg O2 on the same reference. Water and principal
    caprolactam are separate loads supplied by the caller.
    """

    compounds_kg: dict[str, float]
    elements_kg: dict[str, float]
    fossil_carbon_kg: float
    biogenic_carbon_kg: float
    thod_kg_o2: float


def _pa_finish_load(compounds_kg: dict[str, float]) -> _PAFinishLoad:
    """Convert finish quantities to element masses and oxygen equivalents.

    Reference oxidation forms CO2, NH3, H3PO4, H2SO4, NaOH and water, giving
    mol O2 = (4C + H - 2O - 3N + 5P + 6S + Na)/4.
    This oxygen-equivalent descriptor is separate from effluent speciation.
    """
    elements = {element: 0.0 for element in _ATOMS}
    fossil, biogenic, thod = 0.0, 0.0, 0.0
    for name, amount in compounds_kg.items():
        formula = _FORMULAS[name]
        molar_mass = _FORMULA_MOLAR_MASSES[name]
        # kg/(g/mol) is numerically kmol; multiplying by g/mol gives kg.
        kmol = amount / molar_mass
        for element, count in formula.items():
            elements[element] += kmol * count * _ATOMS[element]
        biological_atoms = _BIOGENIC_CARBON_ATOMS[name]
        biogenic += kmol * biological_atoms * _ATOMS["C"]
        fossil += kmol * (formula["C"] - biological_atoms) * _ATOMS["C"]
        oxygen_atoms = _FORMULA_OXYGEN_ATOMS[name]
        thod += kmol * oxygen_atoms / 4 * (2 * _ATOMS["O"])
    return _PAFinishLoad(compounds_kg, elements, fossil, biogenic, thod)


@dataclass(frozen=True, slots=True)
class _PAFinishReference:
    """Concentrate and application quantities per kg finished principal PA.

    Ingredient demands are nominal kg and already include blending loss;
    the FAS supplier co-delivers its aqueous carrier. Finish loads exclude
    principal polymer and carrier water. All other masses are kg,
    except drum count and the named collected solution volume.
    """

    ingredient_demands_kg: dict[str, float]
    supplied: _PAFinishLoad
    air: _PAFinishLoad
    collected: _PAFinishLoad
    retained: _PAFinishLoad
    blending_residue: _PAFinishLoad
    nominal_finish_kg: float
    dry_concentrate_kg: float
    supplier_water_kg: float
    concentrate_kg: float
    blending_residue_water_kg: float
    drum_count: float
    drum_rinse_water_kg: float
    direct_application_water_kg: float
    collected_fraction: float
    collected_caprolactam_kg: float
    collected_solution_m3: float
    collected_water_kg: float
    collected_application_water_kg: float
    collected_utility_water_kg: float
    gate_water_kg: float
    exhaust_water_kg: float
    cooling_water_kg: float
    site_water_kg: float


@lru_cache(maxsize=len(_PA_APPLICATION))
def _pa_finish_reference(pa_grade: str) -> _PAFinishReference:
    """Balance the delivered concentrate, one COD partition and conditioned yarn.

    The seven declared ingredients supply the dry finish. The FAS supplier's
    carrier water belongs to the same application emulsion. Both PA grades
    keep their fixed principal dose, COD and fiber moisture references.
    Each grade's reference is shared within the process. Treat all its nested
    quantity mappings as read-only; actual-load consumers construct their own
    scaled quantities from this shared reference.
    """
    finish, ester_air, caprolactam, cod, site_water = _PA_APPLICATION[pa_grade]
    supplied = {name: finish * fraction for name, fraction in _FINISH_FRACTIONS.items()}
    dry = fsum(supplied.values())
    # The complete FAS manufacturing proxy supplies one kg active equivalent
    # in a 28% aqueous grade. Its own supply already includes this carrier.
    supplier_water = supplied["sodium_dodecanesulfonate"] * (1 / 0.28 - 1)
    concentrate = dry + supplier_water
    # Native alkyd handling loss, per dry output, applies homogeneously to the concentrate. PA-site
    # dilution water enters later.
    blending_loss = 0.01683
    residue = _pa_finish_load({name: blending_loss * kg for name, kg in supplied.items()})
    after_air = dict(supplied)
    after_air["tridecyl_stearate"] -= ester_air
    incoming = _pa_finish_load(after_air)
    # C6H11NO ->6CO2 + NH3 +4H2O requires 7.5 O2. Use CIAAW ThOD
    # while retaining the separate principal-carbon convention in custody.
    caprolactam_mass = 6 * _ATOMS["C"] + 11 * _ATOMS["H"] + _ATOMS["N"] + _ATOMS["O"]
    caprolactam_thod = 7.5 * 2 * _ATOMS["O"] / caprolactam_mass
    collected_fraction = (cod - caprolactam * caprolactam_thod) / incoming.thod_kg_o2
    collected = _pa_finish_load({name: collected_fraction * kg for name, kg in after_air.items()})
    retained = _pa_finish_load(
        {name: (1 - collected_fraction) * kg for name, kg in after_air.items()}
    )
    # Patent 10 wt% emulsion uses the actual nonaqueous finish mass.
    # Supply nine water parts; the sulfonate carrier is one of those parts.
    application_water = 9 * dry
    # Three quarter-capacity manual rinses use the selected whole drum volume
    # and 1 kg/L. ECCC's water-soluble-content guidance is an engineering
    # transfer to this emulsifiable finish; all rinsate returns to emulsion.
    drums = concentrate / 100
    rinse = 3 * 0.25 * PA_FINISH_CONTAINER_CAPACITY_L * drums
    # This reference has one kg finished principal PA. Retained finish keeps its actual dry mass
    # while hydration follows the fiber-class regain value.
    gate_water = _MOISTURE_REGAIN[pa_grade]
    collected_application = _APPLICATION_COLLECTION[pa_grade] * application_water
    # NDRC et al. (notice dated 2018-12-29, posted 2019-01-18), Table 1: class-II PA6 design ceiling
    # 0.9 m3/t, transferred to both grades per dry principal. Total solution uses 1000 kg/m3; water
    # and dry solutes occupy their respective parts of that mass.
    volume = 0.0009
    collected_water = 1000 * volume - fsum(collected.compounds_kg.values()) - caprolactam
    utility_water = collected_water - collected_application
    return _PAFinishReference(
        ingredient_demands_kg={
            name: (1 + blending_loss) * finish * fraction
            for name, fraction in _FINISH_FRACTIONS.items()
        },
        supplied=_pa_finish_load(supplied),
        air=_pa_finish_load({"tridecyl_stearate": ester_air}),
        collected=collected,
        retained=retained,
        blending_residue=residue,
        nominal_finish_kg=finish,
        dry_concentrate_kg=dry,
        supplier_water_kg=supplier_water,
        concentrate_kg=concentrate,
        blending_residue_water_kg=blending_loss * supplier_water,
        drum_count=drums,
        drum_rinse_water_kg=rinse,
        direct_application_water_kg=application_water - supplier_water - rinse,
        collected_fraction=collected_fraction,
        collected_caprolactam_kg=caprolactam,
        collected_solution_m3=volume,
        collected_water_kg=collected_water,
        collected_application_water_kg=collected_application,
        collected_utility_water_kg=utility_water,
        gate_water_kg=gate_water,
        exhaust_water_kg=application_water - collected_application - gate_water,
        cooling_water_kg=site_water - (application_water - supplier_water) - utility_water,
        site_water_kg=site_water,
    )


@dataclass(frozen=True, slots=True)
class _PASpinningWater:
    """Actual water and equipment-deposit kg for one PA spinning operation.

    Incoming pellet water accompanies recovered principal and inherited finish.
    Site water excludes that incoming water and the concentrate supplier's
    carrier. Equipment-control water is one part of the site demand, alongside
    cooling and the original application/collected-solution duties.
    """

    maintenance_principal_kg: float
    maintenance_embedded_dose_kg: float
    maintenance_dry_kg: float
    incoming_pellet_water_kg: float
    site_water_kg: float
    cooling_water_kg: float
    maintenance_water_kg: float
    gate_water_kg: float
    exhaust_water_kg: float
    collected_water_kg: float


def _pa_spinning_water(
    finish: _PAFinishReference,
    *,
    finished_principal_kg: float,
    recovered_principal_kg: float,
    embedded_dose_kg: float,
    maintenance_principal_kg: float,
    maintenance_embedded_dose_kg: float,
    maintenance_water_kg: float,
) -> _PASpinningWater:
    """Supply actual PA application and equipment-cleaning water exactly once.

    Custody supplies the already destroyed principal and inherited finish;
    their deposit fractions are not applied again. ``embedded_dose_kg`` is the incoming inherited
    finish dose before this operation; its fixed compound vector is
    ``finish.retained``. The actual dated maintenance receiver supplies its
    ordinary control-water quantity for this complete deposit within the site meter.

    Pristine-fiber BREF water is preserved. Principal-deposit cleaning is
    allocated from its cooling remainder; only cleaning of additional inherited
    finish increases the site meter. Principal moisture regain remains fixed, while
    actual recovered dry mixture determines incoming pellet water.
    """
    retained_mass = fsum(finish.retained.compounds_kg.values())
    inherited_deposit_mass = retained_mass * maintenance_embedded_dose_kg
    dry_deposit = maintenance_principal_kg + inherited_deposit_mass
    # Every actual spinning event has the recipe's positive equipment deposit.
    # The receiver owns its total ordinary water; recover only the per-dry-kg
    # coefficient needed to allocate that same water within the site meter.
    water_per_dry_kg = maintenance_water_kg / dry_deposit
    # The recycling patent's selected 500 ppm wet-pellet moisture reference
    # gives water/dry mass = 500/(10^6-500), including inherited dry finish.
    pellet_water = (0.0005 / 0.9995) * (recovered_principal_kg + retained_mass * embedded_dose_kg)
    return _PASpinningWater(
        maintenance_principal_kg=maintenance_principal_kg,
        maintenance_embedded_dose_kg=maintenance_embedded_dose_kg,
        maintenance_dry_kg=dry_deposit,
        incoming_pellet_water_kg=pellet_water,
        site_water_kg=finished_principal_kg * finish.site_water_kg
        + water_per_dry_kg * inherited_deposit_mass,
        cooling_water_kg=finished_principal_kg * finish.cooling_water_kg
        - water_per_dry_kg * maintenance_principal_kg,
        maintenance_water_kg=maintenance_water_kg,
        gate_water_kg=finished_principal_kg * finish.gate_water_kg,
        exhaust_water_kg=finished_principal_kg * finish.exhaust_water_kg + pellet_water,
        collected_water_kg=finished_principal_kg * finish.collected_water_kg,
    )


@dataclass(frozen=True, slots=True)
class _PARecyclingReference:
    """Actual PA fiber wash and pellet quantities on the caller's kg basis.

    An embedded dose multiplies the fixed retained finish vector per kg
    principal. All other masses are kg;
    collected solution volume is m3. Principal recovery and losses are supplied
    by their existing MFA owner.
    """

    incoming_dry_kg: float
    incoming_water_kg: float
    postwash_dry_kg: float
    postwash_embedded_dose_kg: float
    removed_finish: _PAFinishLoad
    fresh_wash_water_kg: float
    film_water_kg: float
    collected_water_kg: float
    collected_solution_m3: float
    pellet_dry_kg: float
    pellet_water_kg: float
    air_water_kg: float


def _pa_recycling_reference(
    pa_grade: str,
    finish: _PAFinishReference,
    *,
    principal_kg: float,
    embedded_dose_kg: float,
    recovery_fraction: float,
    wash_water_per_dry_kg: float,
) -> _PARecyclingReference:
    """Wash surface finish and conserve embedded material and water.

    ``finish.retained`` is the current grade's fixed fresh surface vector per
    finished principal kg. Incoming embedded material is that vector times
    ``embedded_dose_kg``. Previous-cycle material remains inaccessible under
    the selected reference; washing acts only on the newly applied surface.
    Its positive remainder joins the embedded pool during melt processing.

    ``wash_water_per_dry_kg`` comes from the complete original WEEE preparation
    owner, on actual incoming dry throughput. Existing preparation and pellet
    finishing supply the whole operating function. Dry managed residues and
    particulate losses are the stated aggregate outlet convention; recovered
    pellets alone retain the final water. Native dust releases are recorded
    for the aggregate recycling operation.
    """
    remaining = _WASH_REMAINING[pa_grade]
    retained = finish.retained.compounds_kg
    dose_mass = fsum(retained.values())
    incoming_dry = principal_kg + (principal_kg + embedded_dose_kg) * dose_mass
    # One scalar suffices because the selected wash fraction is common to all
    # surface components and inherited material retains the same fixed vector.
    embedded_after_wash = remaining * principal_kg + embedded_dose_kg
    postwash_dry = principal_kg + embedded_after_wash * dose_mass
    removed = _pa_finish_load(
        {name: (1 - remaining) * principal_kg * mass for name, mass in retained.items()}
    )
    # Stationary-use moisture follows principal PA. The actual dry mixture
    # still determines washer throughput and the later film/pellet water.
    incoming_water = _MOISTURE_REGAIN[pa_grade] * principal_kg
    fresh_water = wash_water_per_dry_kg * incoming_dry
    # The patent's 4 wt% film ceiling is selected as a design reference;
    # 0.04/0.96 converts wet fraction to water per kg actual dry washed matter.
    film_water = (0.04 / 0.96) * postwash_dry
    pellet_dry = recovery_fraction * postwash_dry
    # 500 ppm is the selected midpoint of the patent's optional 300-700 ppm
    # dry-pellet range. The wet-fraction convention gives 0.0005/0.9995.
    pellet_water = (0.0005 / 0.9995) * pellet_dry
    collected_water = fresh_water + incoming_water - film_water
    return _PARecyclingReference(
        incoming_dry_kg=incoming_dry,
        incoming_water_kg=incoming_water,
        postwash_dry_kg=postwash_dry,
        postwash_embedded_dose_kg=embedded_after_wash,
        removed_finish=removed,
        fresh_wash_water_kg=fresh_water,
        film_water_kg=film_water,
        collected_water_kg=collected_water,
        # Doka's nominal 1000 kg/m3 solution basis includes actual removed
        # solutes, while the complete receiving owner gets solvent kg separately.
        collected_solution_m3=(collected_water + fsum(removed.compounds_kg.values())) / 1000,
        pellet_dry_kg=pellet_dry,
        pellet_water_kg=pellet_water,
        air_water_kg=film_water - pellet_water,
    )
