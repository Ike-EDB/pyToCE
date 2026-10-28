"""Calculate wastewater loads for plastics residues and chemical manufacture.

Material calculations supply residue or ingredient loads with their
specified compositions, keeping gas and leachate phases separate. These include
thermoplastic starch, filled rubber, polyamide, resin and road-material residues.
Doka Wastewater2021 supplies the shared removal, digestion, chemical and
operating functions, applied to those actual loads.

Every mass shares the material or manufacturing-campaign reference declared
by the caller; quantities on different reference bases are not combined.
The site supplies hydraulic volume, water and climatic evaporation;
the collection market applies treatment share once outside this calculation.
Element masses, allocated sludge H/O, gas combustion and complete
disposal remain distinct quantities in the transferred municipal model.
"""

from dataclasses import dataclass
from math import fsum
from typing import Literal

from .water_treatment import _ROW_PROFILE, _WATER_PROFILES, _WaterTreatmentProfile

_CarbonOrigin = Literal["fossil", "biogenic", "atmospheric_dac"]


@dataclass(frozen=True, slots=True)
class _LeachateCarbon:
    """One named material pool's kg C at its caller's stated treatment boundary."""

    component: str
    principal: bool
    origin: _CarbonOrigin
    carbon_kg: float


@dataclass(frozen=True, slots=True)
class _LeachateLoad:
    """Incoming element masses and separate Doka organic descriptors, in kg.

    Callers supply residue phases or ingredient loads. Organic H/O are
    empirical oxidation descriptors; COD/BOD
    are overlapping kg O2 indicators. The calling site supplies hydraulic m3
    and carrier water independently. All fields share that site's
    material reference, before the collection market's treatment fraction.
    ``chloride_kg`` is the known ionic subset of total incoming Cl, supplied
    by the chemical owner. It changes only raw-bypass speciation; the plant
    receives the full elemental load. Zero retains the source's existing
    Chlorine (Cl2) exchange for callers without a designated ionic subset.
    """

    elements_kg: dict[str, float]
    carbon: tuple[_LeachateCarbon, ...]
    organic_hydrogen_kg: float
    organic_oxygen_kg: float
    toc_kg: float
    doc_kg: float
    cod_kg_o2: float
    bod5_kg_o2: float
    chloride_kg: float = 0.0


@dataclass(frozen=True, slots=True)
class _WastewaterProcessProfile:
    """Selected Doka treatment, digestion, sludge and internal-energy fractions.

    The hydraulic profile keeps the source's distinct plant-size and emission
    allocations. Biological/gas air uses its ``urban_treatment`` share, while
    pool/utility air uses ``urban_emissions``. These are fixed native-region
    references, separate from regional material custody and dated energy suppliers.
    """

    hydraulic: _WaterTreatmentProfile
    digestion_share: float
    incineration_dry_share: float
    chp_share: float
    electricity_efficiency: float
    heat_efficiency: float


# Central Repository 2020 GNIWWT row 337 underlies the selected RoW leachate
# plant. BB337 is digestion and BC337 is CHP. BJ337/BL337 give the retained
# agricultural/thermal dry-sludge allocation over those two outlets. Its
# delivered-energy efficiencies are zero, as in the original leachate owner.
_ROW_WASTEWATER_PROFILE = _WastewaterProcessProfile(
    hydraulic=_ROW_PROFILE,
    digestion_share=0.20525339975098886,
    incineration_dry_share=1 - 0.5859583315861807 / (0.5859583315861807 + 0.2525003587251051),
    chp_share=0.6045658030869254,
    electricity_efficiency=0.0,
    heat_efficiency=0.0,
)
# GNIWWT row 190 fixes Swiss municipal co-treatment for PA finish and ingredient
# manufacturing wastewater: AO:AQ stages, AY biological/gas and AZ
# hydraulic/utility air fractions, BB digestion, BC CHP and 90% dry sludge
# to thermal treatment. Doka 2021 section 10.2 derives the useful-energy
# efficiencies from 433/780 MJ output over (2295-665) MJ fuel input.
_CH_WASTEWATER_PROFILE = _WastewaterProcessProfile(
    hydraulic=_WATER_PROFILES["CH"],
    digestion_share=0.89,
    incineration_dry_share=0.9,
    chp_share=0.236,
    electricity_efficiency=433 / (2295 - 665),
    heat_efficiency=780 / (2295 - 665),
)

# avgWWT rows 23,24,25,26,28,36,48,56,57,59,60,61. Columns are respectively
# D (primary), N (primary+secondary), R (additional tertiary), K (biological
# air), AF (digestion gas); WWcalc FB supplies the empirical dry-mass factor.
# These are the admitted input elements plus fresh precipitant Fe. H and O
# have separate input and process-descriptor interfaces below.
_ELEMENT_TRANSFERS: dict[str, tuple[float, float, float, float, float, float]] = {
    "C": (0.20970000000000003, 0.6990000000000001, 0, 0.245, 0.603, 1.6280480551921013),
    "S": (
        0.012510581207774991,
        0.04170193735924997,
        0,
        0,
        0.04686997201050361,
        2.9956345494231367,
    ),
    "N": (0.0789, 0.263, 0, 0.207, 0.603, 4.426124197002141),
    "P": (0.15, 0.5, 0.42000000000000004, 0, 0, 3.0665159832095576),
    "Cl": (0, 0, 0, 0, 0, 1),
    # PDO's supplied cofactor and medium introduce Co/Ca/K. Original avgWWT rows36/57/59 retain
    # these metals in sludge or effluent. FB is an empirical dry-sludge descriptor.
    "Co": (0.25770080368472997, 0.85900267894909987, 0, 0, 0, 1.2715085694892245),
    "Zn": (0.27668668315083006, 0.9222889438361003, 0, 0, 0, 1.2447605935444392),
    "Fe": (0.29959905103816664, 0.9986635034605555, 0, 0, 0, 1.2864919066036384),
    "Ca": (0.062112441519308538, 0.20704147173102846, 0, 0, 0, 1.3992015968063873),
    "K": (0.015895638998058619, 0.052985463326862062, 0, 0, 0, 1),
    "Mg": (0.02401950262181472, 0.0800650087393824, 0, 0, 0, 1.6581112207963145),
    "Na": (0.002390945358977568, 0.00796981786325856, 0, 0, 0, 1),
    # Retained UPR glass introduces B/Si/Al. Wastewater2021 avgWWT rows27/55/58
    # and WWcalc FB give complementary liquid/sludge fates, with no gas.
    "B": (0.024019502621814723, 0.080065008739382409, 0, 0, 0, 3.2222222222222219),
    "Si": (0.16769428387925497, 0.55898094626418326, 0, 0, 0, 2.1393576871038951),
    "Al": (0.13960078417394403, 0.46533594724648014, 0, 0, 0, 1.8894983599873987),
    # Actual road rutile introduces Ti: Wastewater2021 avgWWT52 D/N/R/K/AF
    # and WWcalc FB52. No biological/digestion gas; FB sizes dry sludge.
    "Ti": (0.27891639648419198, 0.92972132161397325, 0, 0, 0, 1.6683375104427733),
}
_SLUDGE_HYDROGEN_PER_CARBON = 0.1359223300970875  # WWcalc C103 / avgWWT L78.
_SLUDGE_ORGANIC_OXYGEN_PER_CARBON = 0.628571428571428  # avgWWT L79, kg O/kg C.


@dataclass(frozen=True, slots=True)
class _SludgeLoad:
    """Sludge kg elements and separately allocated H/O on the caller's reference.

    ``elements_kg`` excludes H/O. Its C/N/S/P/metals follow the actual incoming
    load and supplied auxiliaries. ``allocated_*`` are Doka's sludge organic
    and oxide-equivalent descriptors, not recovered H/O atom totals.
    Their sum defines the source-model dry mass used by operating/disposal
    functions.
    ``organic_oxygen_kg`` separately follows Doka (2021), section 8.1.1,
    for calorific estimation with organic H and bulk N/S. The calculation keeps mineral
    oxide-equivalent O outside the calorific input and retains it in the dry-mass ledger. Undigested
    PAM retains its explicit
    H/O.
    """

    elements_kg: dict[str, float]
    allocated_hydrogen_kg: float
    allocated_oxygen_kg: float
    organic_oxygen_kg: float

    @property
    def dry_mass_kg(self) -> float:
        """Return source-model dry matter, including its separate H/O allocands."""
        return (
            fsum(self.elements_kg.values()) + self.allocated_hydrogen_kg + self.allocated_oxygen_kg
        )


@dataclass(frozen=True, slots=True)
class _WastewaterCarbon:
    """Carbon kg for one component/origin, including fresh auxiliary PAM.

    Biological air, digester gas, final sludge and effluent exhaust the input
    carbon. Raw sludge is an intermediate and is not another carbon exit.
    Digester gas precedes leakage, combustion and the separate gas compiler.
    """

    component: str
    principal: bool
    origin: _CarbonOrigin
    input_carbon_kg: float
    raw_sludge_carbon_kg: float
    biological_air_carbon_kg: float
    digester_gas_carbon_kg: float
    final_sludge_carbon_kg: float
    effluent_carbon_kg: float


@dataclass(frozen=True, slots=True)
class _TreatedEffluent:
    """Elemental liquid loads and overlapping source effluent indicators.

    Element masses exclude H/O and include fresh FeSO4 excess Fe and sulfate S.
    DOC, TOC, COD and BOD describe the same organic pool;
    COD/BOD are kg O2 equivalents and must not be added as matter.
    Native sulfate/nitrate/phosphate conversion belongs to flow binding.
    """

    elements_kg: dict[str, float]
    toc_kg: float
    doc_kg: float
    cod_kg_o2: float
    bod5_kg_o2: float


@dataclass(frozen=True, slots=True)
class _WastewaterDuties:
    """Actual-load WWT operations, excluding infrastructure and gas treatment.

    Chemical tables retain actual source-formula H/O separately from sludge
    descriptors. Grit/sand are co-treatment operating allocations, not further
    losses of the incoming plastic. Native reagent and terminal markets retain
    their own transport supplies.
    """

    electricity_kwh: float
    heat_mj: float
    biological_oxygen_kg: float
    iron_sulfate_kg: float
    iron_sulfate_elements_kg: dict[str, float]
    polyacrylamide_kg: float
    polyacrylamide_elements_kg: dict[str, float]
    secondary_raw_dry_mass_kg: float
    digestion_feed_dry_mass_kg: float
    grit_plastic_kg: float
    grit_biomass_kg: float
    sand_kg: float


@dataclass(frozen=True, slots=True)
class _WastewaterWater:
    """Carrier water and separately identified process-water terms.

    Biologically generated water uses the organic H descriptor and the
    hydrogen-to-water mass ratio.
    Gas-combustion water and downstream sludge-moisture releases are excluded.
    No clean-water purchase is inferred from any of these wastewater amounts.
    """

    incoming_kg: float
    hydraulic_evaporation_kg: float
    biological_generated_kg: float
    incinerated_sludge_water_kg: float
    landfarmed_sludge_water_kg: float
    effluent_kg: float


@dataclass(frozen=True, slots=True)
class _WastewaterLoad:
    """Conditional wastewater loads on the caller's common mass or campaign reference.

    Primary/secondary/tertiary sludge fields describe incoming-load removal
    before fresh PAM/Fe. Raw sludge includes both auxiliaries. Digested sludge
    contains only the treated digestion fraction; final sludge combines it
    with undigested material. The selected incineration/landfarming fractions
    apply to final dry matter and every final element/carbon pool alike.
    """

    volume_m3: float
    primary_sludge: _SludgeLoad
    secondary_sludge: _SludgeLoad
    tertiary_sludge: _SludgeLoad
    raw_sludge: _SludgeLoad
    digested_sludge: _SludgeLoad
    final_sludge: _SludgeLoad
    carbon: tuple[_WastewaterCarbon, ...]
    biological_air_elements_kg: dict[str, float]
    biological_n2o_nitrogen_kg: float
    biological_dinitrogen_kg: float
    digester_gas_elements_kg: dict[str, float]
    effluent: _TreatedEffluent
    duties: _WastewaterDuties
    water: _WastewaterWater
    incineration_dry_share: float
    landfarming_dry_share: float
    incinerated_sludge_wet_kg: float
    landfarmed_sludge_wet_kg: float


def _sludge_profile(elements: dict[str, float]) -> _SludgeLoad:
    """Keep WWcalc dry-mass H/O and the report's distinct calorific organic O."""
    return _SludgeLoad(
        elements_kg=elements,
        allocated_hydrogen_kg=_SLUDGE_HYDROGEN_PER_CARBON * elements["C"],
        allocated_oxygen_kg=fsum(
            amount * (_ELEMENT_TRANSFERS[element][5] - 1) for element, amount in elements.items()
        ),
        organic_oxygen_kg=_SLUDGE_ORGANIC_OXYGEN_PER_CARBON * elements["C"],
    )


def _raw_sludge(
    primary: _SludgeLoad,
    secondary: _SludgeLoad,
    tertiary: _SludgeLoad,
) -> tuple[_SludgeLoad, float, dict[str, float], float, dict[str, float]]:
    """Add Fe precipitate and neutral PAM before digestion.

    WWcalc D263:D278 supplies FeSO4/P and PAM/secondary-dry-matter doses.
    Apply FeSO4 to additional tertiary P removal, P3. Retain source mass
    conventions 55.85/30.97/32.07/16 and 12.01/1.008/14.01/16 for these compounds,
    separately from incoming principal and retained-reagent elemental bases.
    """
    phosphorus = tertiary.elements_kg["P"]
    iron_sulfate = 30.6456139178772 * phosphorus
    sulfate_mass = 55.85 + 32.07 + 4 * 16
    iron_sulfate_elements = {
        "Fe": iron_sulfate * 55.85 / sulfate_mass,
        "S": iron_sulfate * 32.07 / sulfate_mass,
        "O": iron_sulfate * 4 * 16 / sulfate_mass,
    }
    pam = 0.017265521465486582 * secondary.dry_mass_kg
    pam_mass = 3 * 12.01 + 5 * 1.008 + 14.01 + 16
    pam_elements = {
        "C": pam * 3 * 12.01 / pam_mass,
        "H": pam * 5 * 1.008 / pam_mass,
        "N": pam * 14.01 / pam_mass,
        "O": pam * 16 / pam_mass,
    }
    elements = {
        element: fsum(sludge.elements_kg[element] for sludge in (primary, secondary, tertiary))
        for element in _ELEMENT_TRANSFERS
    }
    # One mol precipitated Fe per mol P gives kg Fe = kg P * 55.85/30.97.
    # Assign fresh precipitant Fe by removed P.
    elements["Fe"] += phosphorus * 55.85 / 30.97
    before_pam = _sludge_profile(elements)
    elements = dict(elements)
    elements["C"] += pam_elements["C"]
    elements["N"] += pam_elements["N"]
    raw = _SludgeLoad(
        elements_kg=elements,
        allocated_hydrogen_kg=before_pam.allocated_hydrogen_kg + pam_elements["H"],
        allocated_oxygen_kg=before_pam.allocated_oxygen_kg + pam_elements["O"],
        organic_oxygen_kg=before_pam.organic_oxygen_kg + pam_elements["O"],
    )
    return raw, iron_sulfate, iron_sulfate_elements, pam, pam_elements


def _carbon_phases(
    carbon: tuple[_LeachateCarbon, ...], pam_carbon: float, profile: _WastewaterProcessProfile
) -> tuple[_WastewaterCarbon, ...]:
    """Keep component/principal/origin identity through shared removal and PAM.

    WWcalc BY23 includes new PAM carbon before DA23/DC23 digestion. The same source carbon-removal
    function applies separately to each incoming organic component and its attributed origin.
    """
    _, f2, f3 = profile.hydraulic.stages
    primary, total, tertiary, air, gas, _ = _ELEMENT_TRANSFERS["C"]
    raw_fraction = (
        fsum(profile.hydraulic.stages) * primary + (f2 + f3) * (total - primary) + f3 * tertiary
    )
    air_fraction = (f2 + f3) * air
    result: list[_WastewaterCarbon] = []
    for pool in carbon:
        incoming = pool.carbon_kg
        raw = incoming * raw_fraction
        gas_carbon = raw * profile.digestion_share * gas
        air_carbon = incoming * air_fraction
        result.append(
            _WastewaterCarbon(
                component=pool.component,
                principal=pool.principal,
                origin=pool.origin,
                input_carbon_kg=incoming,
                raw_sludge_carbon_kg=raw,
                biological_air_carbon_kg=air_carbon,
                digester_gas_carbon_kg=gas_carbon,
                final_sludge_carbon_kg=raw - gas_carbon,
                effluent_carbon_kg=fsum(profile.hydraulic.stages) * incoming - raw - air_carbon,
            )
        )
    pam_gas = pam_carbon * profile.digestion_share * gas
    result.append(
        _WastewaterCarbon(
            component="wastewater_polyacrylamide",
            principal=False,
            origin="fossil",
            input_carbon_kg=pam_carbon,
            raw_sludge_carbon_kg=pam_carbon,
            biological_air_carbon_kg=0,
            digester_gas_carbon_kg=pam_gas,
            final_sludge_carbon_kg=pam_carbon - pam_gas,
            effluent_carbon_kg=0,
        )
    )
    return tuple(result)


def _biological_allocation(
    leachate: _LeachateLoad, profile: _WastewaterProcessProfile
) -> tuple[float, float]:
    """Return kg biological oxygen and reaction water using Doka's allocations.

    Organic oxidation uses 32/12 C and 8 H, separate from raw-inlet COD.
    Net aeration demand is nonnegative. Mineralized hydrogen gives reaction
    water as H/1.008*18.016/2. Gas-combustion oxygen and water are separate.
    """
    _, f2, f3 = profile.hydraulic.stages
    biological = f2 + f3
    carbon_air = biological * _ELEMENT_TRANSFERS["C"][3]
    hydrogen_mineralized = carbon_air * leachate.organic_hydrogen_kg
    oxygen_mineralized = carbon_air * leachate.organic_oxygen_kg
    organic_oxygen = (
        32 / 12 * carbon_air * leachate.toc_kg + 8 * hydrogen_mineralized - oxygen_mineralized
    )
    nitrogen = leachate.elements_kg.get("N", 0.0)
    # C169:C176: source 95% NH4-N after sludge removal, 3% in effluent.
    after_sludge_nitrogen = biological * (1 - _ELEMENT_TRANSFERS["N"][1]) * nitrogen
    air_nitrogen = biological * _ELEMENT_TRANSFERS["N"][3] * nitrogen
    nitrified_nitrogen = 0.95 * after_sludge_nitrogen - 0.030000000000000027 * (
        after_sludge_nitrogen - air_nitrogen
    )
    nitrification_oxygen = nitrified_nitrogen * 48 / 14
    denitrification_oxygen = air_nitrogen * 48 / 14
    oxygen = max(organic_oxygen - denitrification_oxygen, 0.0) + nitrification_oxygen
    water = hydrogen_mineralized * 18.016 / (2 * 1.008)
    return oxygen, water


def _wastewater_load(
    leachate: _LeachateLoad,
    *,
    volume_m3: float,
    incoming_water_kg: float,
    evaporation_fraction: float,
    profile: _WastewaterProcessProfile,
) -> _WastewaterLoad:
    """Calculate one conditional wastewater operation on the caller reference.

    Parameters are admitted material loads from the actual material owner and
    material/site quantities on the same reference. The site compiler derives
    ``volume_m3`` and ``incoming_water_kg`` from its source water boundary and
    ``evaporation_fraction`` from the original plant climate via the existing
    water-evaporation owner. The supplied material composition determines incoming carrier water. No
    collection, incineration or gas-treatment
    fraction is applied by the caller before this calculation.
    """
    f1, f2, f3 = profile.hydraulic.stages
    all_stages = fsum(profile.hydraulic.stages)
    biological = f2 + f3
    # Unlisted H/O remain in the input table. Missing listed elements
    # are absent from the input formulation.
    incoming = {element: leachate.elements_kg.get(element, 0.0) for element in _ELEMENT_TRANSFERS}
    primary_elements: dict[str, float] = {}
    secondary_elements: dict[str, float] = {}
    tertiary_elements: dict[str, float] = {}
    air_elements: dict[str, float] = {}
    liquid_elements: dict[str, float] = {}
    for element, amount in incoming.items():
        primary, total, tertiary, air, _, _ = _ELEMENT_TRANSFERS[element]
        primary_elements[element] = all_stages * primary * amount
        # Secondary sludge receives the additional biological removal.
        secondary_elements[element] = biological * (total - primary) * amount
        tertiary_elements[element] = f3 * tertiary * amount
        air_elements[element] = biological * air * amount
        liquid_elements[element] = (
            all_stages * amount
            - primary_elements[element]
            - secondary_elements[element]
            - tertiary_elements[element]
            - air_elements[element]
        )
    primary_sludge = _sludge_profile(primary_elements)
    secondary_sludge = _sludge_profile(secondary_elements)
    tertiary_sludge = _sludge_profile(tertiary_elements)
    raw, sulfate, sulfate_elements, pam, pam_elements = _raw_sludge(
        primary_sludge, secondary_sludge, tertiary_sludge
    )
    # D267/D268: source excess Fe and all supplied sulfate S enter effluent.
    liquid_elements["Fe"] += sulfate_elements["Fe"] - tertiary_elements["P"] * 55.85 / 30.97
    liquid_elements["S"] += sulfate_elements["S"]

    # WWcalc DA/DC/DE/DL/DP: q is applied once, after both fresh auxiliaries.
    gas_elements = {
        element: profile.digestion_share * amount * _ELEMENT_TRANSFERS[element][4]
        for element, amount in raw.elements_kg.items()
    }
    digested = _sludge_profile(
        {
            element: profile.digestion_share * amount - gas_elements[element]
            for element, amount in raw.elements_kg.items()
        }
    )
    final = _SludgeLoad(
        elements_kg={
            element: (1 - profile.digestion_share) * amount + digested.elements_kg[element]
            for element, amount in raw.elements_kg.items()
        },
        allocated_hydrogen_kg=(
            (1 - profile.digestion_share) * raw.allocated_hydrogen_kg
            + digested.allocated_hydrogen_kg
        ),
        allocated_oxygen_kg=(
            (1 - profile.digestion_share) * raw.allocated_oxygen_kg + digested.allocated_oxygen_kg
        ),
        organic_oxygen_kg=(
            (1 - profile.digestion_share) * raw.organic_oxygen_kg + digested.organic_oxygen_kg
        ),
    )
    oxygen, biological_water = _biological_allocation(leachate, profile)
    digestion_feed_dry = profile.digestion_share * raw.dry_mass_kg
    # WWcalc B121:B126 sums these purchased-electricity duties. B111:D111
    # (avgWWT D114:F114) gives 0.057381... kWh/m3 for mechanical treatment.
    # B115 (avgWWT D118) and C115:D115 (E118:F118) give other kWh/m3 for
    # primary-only and biological plants, weighted by their treatment shares.
    # H112 (avgWWT L115) gives 1.327782... kWh/kg aeration oxygen (I112=C183).
    # H113 (L116) gives 0.179636... kWh/kg digestion feed dry matter (I113=DA62).
    # H114 (L117) gives 0.237603... kWh/kg final dry matter for dewatering
    # (I114=DP62). These are gross duties. The separate gas owner determines
    # internal generation; its plant compiler subtracts it once from purchases.
    electricity = (
        0.05738105911904385 * volume_m3
        + (0.0044590024915761835 * f1 + 0.019121199831055386 * biological) * volume_m3
        + 1.3277820299177079 * oxygen
        + 0.17963629273632015 * digestion_feed_dry
        + 0.23760323270872744 * final.dry_mass_kg
    )
    # WWcalc H151 (avgWWT L122) gives 2.725010... MJ/kg digestion-feed dry matter;
    # H152 (L123) gives 0.068932... MJ/m3 treated for other plant heating.
    heat = 2.7250108952894734 * digestion_feed_dry + 0.06893210265697261 * volume_m3
    # WWcalc C238/C236: incineration/agriculture sludge contains 70%/97% water
    # by wet mass, hence 30%/3% dry matter. Divide by the dry-matter fraction.
    incinerated_wet = final.dry_mass_kg * profile.incineration_dry_share / 0.30
    landfarmed_wet = final.dry_mass_kg * (1 - profile.incineration_dry_share) / 0.03
    incinerated_water = 0.70 * incinerated_wet
    landfarmed_water = 0.97 * landfarmed_wet
    # The source hydraulic reference is 1000 kg wet liquid per m3; the climate
    # fraction assigns its evaporation. Actual incoming water is supplied apart.
    evaporation = volume_m3 * evaporation_fraction * 1000
    # D244:D249 allocate final sludge (DP62) to both terminals.
    effluent_water = (
        incoming_water_kg - evaporation + biological_water - incinerated_water - landfarmed_water
    )
    effluent_carbon = liquid_elements["C"]
    # WWcalc C188/BF25: N2O-N is a share of biological nitrogen released to air.
    n2o_nitrogen = air_elements["N"] * 0.0067971710769745034
    return _WastewaterLoad(
        volume_m3=volume_m3,
        primary_sludge=primary_sludge,
        secondary_sludge=secondary_sludge,
        tertiary_sludge=tertiary_sludge,
        raw_sludge=raw,
        digested_sludge=digested,
        final_sludge=final,
        carbon=_carbon_phases(leachate.carbon, pam_elements["C"], profile),
        biological_air_elements_kg=air_elements,
        biological_n2o_nitrogen_kg=n2o_nitrogen,
        biological_dinitrogen_kg=air_elements["N"] - n2o_nitrogen,
        digester_gas_elements_kg=gas_elements,
        effluent=_TreatedEffluent(
            elements_kg=liquid_elements,
            toc_kg=effluent_carbon,
            # WWcalc B88/B86/B87: kg DOC/kg TOC, kg COD O2/kg TOC, and kg BOD5 O2/kg TOC describe
            # overlapping quantities.
            doc_kg=0.6114130434782609 * effluent_carbon,
            cod_kg_o2=3.7 * effluent_carbon,
            bod5_kg_o2=0.91 * effluent_carbon,
        ),
        duties=_WastewaterDuties(
            electricity_kwh=electricity,
            heat_mj=heat,
            biological_oxygen_kg=oxygen,
            iron_sulfate_kg=sulfate,
            iron_sulfate_elements_kg=sulfate_elements,
            polyacrylamide_kg=pam,
            polyacrylamide_elements_kg=pam_elements,
            secondary_raw_dry_mass_kg=secondary_sludge.dry_mass_kg,
            digestion_feed_dry_mass_kg=digestion_feed_dry,
            # WWcalc D291/D292 allocate 0.019585... kg grit and 0.004810... kg sand per m3 treated.
            # D297/D298 split grit equally between plastic and biomass treatment; these are
            # co-treatment duties.
            grit_plastic_kg=0.019585524957766807 * volume_m3 / 2,
            grit_biomass_kg=0.019585524957766807 * volume_m3 / 2,
            sand_kg=0.004810330964976001 * volume_m3,
        ),
        water=_WastewaterWater(
            incoming_kg=incoming_water_kg,
            hydraulic_evaporation_kg=evaporation,
            biological_generated_kg=biological_water,
            incinerated_sludge_water_kg=incinerated_water,
            landfarmed_sludge_water_kg=landfarmed_water,
            effluent_kg=effluent_water,
        ),
        incineration_dry_share=profile.incineration_dry_share,
        landfarming_dry_share=(1 - profile.incineration_dry_share),
        incinerated_sludge_wet_kg=incinerated_wet,
        landfarmed_sludge_wet_kg=landfarmed_wet,
    )
