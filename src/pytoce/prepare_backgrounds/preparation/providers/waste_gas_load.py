"""Calculate actual landfill and wastewater gas before native inventory binding.

Material owners supply gas carbon pools and N/S loads. These equations retain
the selected Doka leakage, flare/CHP, molecular and energy references.
Source rows and fuel quantity share one original or dated normalization basis;
only the fuel net heating value comes from the original source. The calculations use prepared source
coefficients and the actual gas load.
"""

from dataclasses import asdict, dataclass
from math import fsum
from typing import cast

from .leachate_load import _CarbonOrigin, _WastewaterLoad, _WastewaterProcessProfile
from .material_carbon import PrincipalCarbonOriginFractions

# Doka (2003), report 13-III, Tables 6.2-6.3 give reference landfill gas kg/m3, including waste N2
# and entrained air. V_eq expresses the source-equivalent volume.
_REFERENCE_WASTE_KMOL = 0.3357 / 16 + 0.7268 / 44 + 0.049 / 28
_REFERENCE_CARBON = 0.3357 * 12 / 16 + 0.7268 * 12 / 44
_NMVOC_NAME = "NMVOC, non-methane volatile organic compounds"
_PM_NAME = "Particulate Matter, < 2.5 um"
_RURAL = "non-urban air or from high stacks"
_URBAN = "urban air close to ground"
_CHP_FACTORS = {
    "co": "Carbon monoxide, non-fossil",
    "methane": "Methane, non-fossil",
    "nmvoc": _NMVOC_NAME,
    "nox": "Nitrogen oxides",
    "n2o": "Dinitrogen monoxide",
    "platinum": "Platinum",
}
# Chase (1998), via NIST Chemistry WebBook, 298.15 K, liquid-water HHV in MJ/kg. Rounded molecular
# masses match Doka's gas-species accounting.
_HHV_CH4 = (-74.87 + 393.52 + 2 * 285.83) / 16
_HHV_CO = (-110.53 + 393.52) / 28
_HHV_C = 393.52 / 12
_HHV_H2S = (-20.50 + 296.84 + 285.83) / 34
_HHV_NH3 = (-45.90 + 1.5 * 285.83) / 17
# Michel (via Doka) empirical organic HHV for the selected anonymous CH1.85 NMVOC.
_HHV_NMVOC = (12 * 34.016 + 1.85 * 124.265) / 13.85


@dataclass(frozen=True, slots=True)
class _GasPool:
    """One component/origin's actual inlet carbon, in kg C rather than species."""

    component: str
    principal: bool
    origin: _CarbonOrigin
    methane_carbon_kg: float
    carbon_dioxide_carbon_kg: float


@dataclass(frozen=True, slots=True)
class _LandfillGasLoad:
    """Gas inlet and matrix-decomposition heat on the calling material basis.

    The material owner supplies gas species carbon, N/S and decomposition
    heat. SBR uses its climate-adjusted matrix release; TPS uses its biodegradable-class equations.
    """

    carbon: tuple[_GasPool, ...]
    nitrogen_kg: float
    sulfur_kg: float
    decomposition_heat_mj: float
    decomposition_trace: dict[str, float]


@dataclass(frozen=True, slots=True)
class _GasOperation:
    """Admitted gas emission factors and original fuel net heating value.

    ``rows`` contain ordinary dictionaries with native amounts and metadata;
    ``fuel_amount`` uses the same native-reference or normalized basis as every
    row. Their ratio cancels production exactly once. ``fuel_lhv`` is MJ per native fuel unit from
    the original source. ``trace`` retains the preparation provenance.
    """

    rows: dict[str, dict[str, object]]
    fuel_amount: float
    fuel_lhv: float
    trace: dict[str, object]


@dataclass(frozen=True, slots=True)
class _GasBranch:
    """One actual gas combustion branch and its native factor templates."""

    pools: tuple[_GasPool, ...]
    products: dict[str, float]
    templates: dict[str, dict[str, object]]
    water_kg: float
    oxygen_kg: float
    heat_mj: float
    trace: dict[str, object]


@dataclass(frozen=True, slots=True)
class _GasEmission:
    """One actual air release before native rural/urban compartment binding."""

    name: str
    amount: float
    unit: str
    template: dict[str, object] | None = None
    principal_carbon_origin_fractions: PrincipalCarbonOriginFractions | None = None


@dataclass(frozen=True, slots=True)
class _GasLeakage:
    """Uncollected gas releases and their separate atmospheric reaction ledger."""

    emissions: tuple[_GasEmission, ...]
    trace: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LandfillGasReference:
    """Actual gas operation and energy allocation before native-flow binding.

    Branch products retain their own units and source templates.
    Multipliers scale dated combustion factors; they are not gas yields. The
    equations need no database resolver; templates retain the rows supplied
    by preparation or by its completed source record.
    ``volume_m3`` is Doka source-equivalent volume; ``physical_dry_volume_m3``
    is actual gas under the study's rounded normal-volume convention.
    """

    branches: tuple[_GasBranch, ...]
    volume_m3: float
    physical_dry_volume_m3: float
    energy_mj: float
    chp_multiplier: float
    flare_multiplier: float
    pump_kwh: float
    generation_kwh: float
    export_kwh: float
    heat_air_mj: float
    useful_heat_mj: float


@dataclass(frozen=True, slots=True)
class _WastewaterGasReference:
    """Conditional digester operation on the actual incoming wastewater reference.

    Pools and N/S/NH3-N are kg before leakage, volume is dry normal m3, and
    branch multipliers use admitted source rows. Internal kWh/MJ reduce plant
    purchases once; air heat is the existing HHV consuming-meter allocation.
    No collection fraction or second digestion step is applied here.
    """

    pools: tuple[_GasPool, ...]
    branches: tuple[_GasBranch, ...]
    nitrogen_kg: float
    sulfur_kg: float
    ammonia_nitrogen_kg: float
    volume_m3: float
    chp_multiplier: float
    flare_multiplier: float
    electricity_kwh: float
    useful_heat_mj: float
    heat_air_mj: float


def _combustion_branch(
    pools: tuple[_GasPool, ...],
    nitrogen: float,
    sulfur: float,
    ammonia_nitrogen: float,
    volume: float,
    operation: _GasOperation,
    multiplier: float,
    *,
    flare: bool,
    particulate_per_m3: float,
    entrained_oxygen: float = 0.0,
    entrained_nitrogen: float = 0.0,
) -> _GasBranch:
    """Close actual C/H/N/S with one empirical minor-emission profile.

    Flare carbon/NOx factors are Doka2003 Table6.3; N2O uses the admitted
    industrial burner factor.
    Engine factors already include thermal NOx: no old fuel-N factor is added.
    Molecular masses use Doka's rounded basis. The admitted TPS/SBR profiles
    and selected native factors leave positive C/H/N and heat remainders.
    Supported premise pollutant updates only reduce the relevant factors;
    no user-supplied gas composition or combustion factor enters this owner.
    """
    templates: dict[str, dict[str, object]]
    if flare:
        templates = {"n2o": operation.rows["Dinitrogen monoxide"]}
        products = {
            "co": 0.00014 * volume,
            "methane": 0.000014 * (0.0018 / (0.0018 + 0.00042)) * volume,
            "nmvoc": 0.000014 * (0.00042 / (0.0018 + 0.00042)) * volume,
            "nox": 0.000419 * volume,
            "n2o": float(cast(float, templates["n2o"]["amount"])) * multiplier,
            "platinum": 0.0,
        }
    else:
        templates = {name: operation.rows[source] for name, source in _CHP_FACTORS.items()}
        products = {
            name: float(cast(float, ex["amount"])) * multiplier for name, ex in templates.items()
        }
    products["particles"] = particulate_per_m3 * volume
    products["so2"] = 2 * sulfur
    methane_c = fsum(pool.methane_carbon_kg for pool in pools)
    minor_c = (
        products["co"] * 12 / 28
        + products["methane"] * 12 / 16
        + products["nmvoc"] * 12 / 13.85
        + products["particles"]
    )
    remaining_c = methane_c - minor_c
    methane = methane_c * 16 / 12
    h2s = sulfur * 34 / 32
    ammonia = ammonia_nitrogen * 17 / 14
    water_kmol = (
        4 * methane / 16
        - 4 * products["methane"] / 16
        - 1.85 * products["nmvoc"] / 13.85
        + 2 * h2s / 34
        + 3 * ammonia / 17
    ) / 2
    water = 18 * water_kmol
    oxygen = 16 * (
        2 * remaining_c / 12
        + products["co"] / 28
        + water_kmol
        + 2 * products["so2"] / 64
        + 2 * products["nox"] / 46
        + products["n2o"] / 44
    )
    additional_oxygen = oxygen - entrained_oxygen
    # Doka combustion air is 76% N2 and 24% O2 by mass. Both admitted gas
    # branches contain methane, so atmospheric air supplies positive N2.
    air_nitrogen = entrained_nitrogen + additional_oxygen * 0.76 / 0.24
    oxide_n = products["nox"] * 14 / 46 + products["n2o"] * 28 / 44
    total_n = nitrogen + air_nitrogen
    gas_n_to_oxides = oxide_n * nitrogen / total_n
    fuel_hhv = _HHV_CH4 * methane + _HHV_H2S * h2s + _HHV_NH3 * ammonia
    residual_hhv = (
        _HHV_CH4 * products["methane"]
        + _HHV_CO * products["co"]
        + _HHV_NMVOC * products["nmvoc"]
        + _HHV_C * products["particles"]
    )
    # Chase (1998), via NIST Chemistry WebBook formation enthalpies: NO2 33.10 and N2O 82.05
    # MJ/kmol;
    # divide by their 46/44 kg/kmol molecular masses for the actual gas kg.
    oxide_energy = products["nox"] * 33.10 / 46 + products["n2o"] * 82.05 / 44
    heat = fuel_hhv - residual_hhv - oxide_energy
    carbon = []
    for pool in pools:
        share = pool.methane_carbon_kg / methane_c
        carbon.append(
            {
                **asdict(pool),
                "co_carbon_kg": products["co"] * 12 / 28 * share,
                "methane_slip_carbon_kg": products["methane"] * 12 / 16 * share,
                "nmvoc_carbon_kg": products["nmvoc"] * 12 / 13.85 * share,
                "particulate_carbon_kg": products["particles"] * share,
                "carbon_dioxide_output_carbon_kg": pool.carbon_dioxide_carbon_kg
                + remaining_c * share,
            }
        )
    trace: dict[str, object] = {
        "carbon": carbon,
        "input_species_kg": {
            "CH4": methane,
            "H2S": h2s,
            "NH3": ammonia,
            "N2": nitrogen - ammonia_nitrogen,
        },
        "products_kg": products,
        "gas_volume_m3": volume,
        "native_multiplier": multiplier,
        "plant_water_kg": water,
        "plant_oxygen_kg": oxygen,
        "entrained_oxygen_kg": entrained_oxygen,
        "additional_oxygen_kg": additional_oxygen,
        "gas_nitrogen_to_oxides_kg": gas_n_to_oxides,
        "gas_dinitrogen_out_kg": nitrogen - gas_n_to_oxides,
        "air_nitrogen_to_oxides_kg": oxide_n - gas_n_to_oxides,
        "air_dinitrogen_out_kg": air_nitrogen - oxide_n + gas_n_to_oxides,
        "input_hhv_mj": fuel_hhv,
        "residual_hhv_mj": residual_hhv,
        "oxide_formation_mj": oxide_energy,
        "reaction_hhv_mj": heat,
        "water_vapour_reference_difference_mj": water / 18 * (285.83 - 241.83),
    }
    return _GasBranch(pools, products, templates, water, oxygen, heat, trace)


def _gas_principal_origins(pool: _GasPool) -> PrincipalCarbonOriginFractions:
    """Tag principal carbon in a gas pool using its native flow identity.

    Protein, retained reagents and digester PAM keep zero principal fractions;
    their complete gas emissions remain in the inventory. TPS principal glucan
    primer remains biogenic even in the atmospheric-DAC formulation.
    """
    return (
        float(pool.principal and pool.origin == "fossil"),
        float(pool.principal and pool.origin == "biogenic"),
        float(pool.principal and pool.origin == "atmospheric_dac"),
    )


def _gas_fraction(pools: tuple[_GasPool, ...], fraction: float) -> tuple[_GasPool, ...]:
    """Pass one gas share while preserving component, principal and origin."""
    return tuple(
        _GasPool(
            pool.component,
            pool.principal,
            pool.origin,
            fraction * pool.methane_carbon_kg,
            fraction * pool.carbon_dioxide_carbon_kg,
        )
        for pool in pools
    )


def _landfill_gas_reference(
    load: _LandfillGasLoad, chp: _GasOperation, furnace: _GasOperation
) -> _LandfillGasReference:
    """Calculate the actual mixed landfill gas using admitted dated sources.

    Source lookup belongs to preparation. Ordinary inventory construction
    and PA candidate projection can use this same arithmetic after admission,
    including origin-dependent combustion and the station energy balance.
    The caller has already resolved a nonzero gas load.
    """
    pools = load.carbon
    nitrogen, sulfur = load.nitrogen_kg, load.sulfur_kg
    methane = fsum(item.methane_carbon_kg for item in pools) * 16 / 12
    carbon_dioxide = fsum(item.carbon_dioxide_carbon_kg for item in pools) * 44 / 12
    waste_kmol = methane / 16 + carbon_dioxide / 44 + nitrogen / 28 + sulfur / 32
    volume = waste_kmol / _REFERENCE_WASTE_KMOL
    # Gas combustion uses 50 MJ/kg CH4 and 518.17 MJ/kmol H2S.
    # Dated fuel quantities scale operating inventories; fuel heating values
    # retain their original source reference.
    energy = 50 * methane + 518.17 * sulfur / 32
    k_chp = 0.3498 * energy / (chp.fuel_amount * chp.fuel_lhv)
    k_flare = 0.1802 * energy / (furnace.fuel_amount * furnace.fuel_lhv)
    # Doka2003 Table6.3: 47 mg particulate per reference m3, with 0.036 kg
    # entrained O2 and 0.114 kg entrained N2 per reference m3. The declared
    # 18.02% flare/34.98% CHP shares apply once to that same actual gas.
    branches = tuple(
        _combustion_branch(
            _gas_fraction(pools, fraction),
            fraction * nitrogen,
            fraction * sulfur,
            0.0,
            fraction * volume,
            operation,
            multiplier,
            flare=is_flare,
            particulate_per_m3=0.000047,
            entrained_oxygen=fraction * volume * 0.036,
            entrained_nitrogen=fraction * volume * 0.114,
        )
        for fraction, operation, multiplier, is_flare in (
            (0.1802, furnace, k_flare, True),
            (0.3498, chp, k_chp, False),
        )
    )
    # Doka landfill extraction kWh/kg reference gas-C. The 53% collected gas
    # serves both flare and CHP; the existing station balance owns its power.
    pump = 0.53 * volume * _REFERENCE_CARBON * 0.046647169657919804
    # The study's dry-normal convention is 22.4 m3/kmol, including source entrained air. V_eq uses
    # the source-equivalent basis.
    dry_volume = 22.4 * volume * (_REFERENCE_WASTE_KMOL + 0.114 / 28 + 0.036 / 32)
    # Doka pumping is the changed gas load. Native site supplies remain with
    # the landfill owner; the emission-factor donor adds no furnace operation.
    # The source engine converts 27.8% of the CHP feed energy to electricity;
    # 3.6 MJ/kWh converts its generation to the same unit as the station load.
    generation = 0.278 * 0.3498 * energy / 3.6
    # All admitted TPS/SBR/sanitary-PA gas has at least half its carbon as
    # methane; admitted N/C is at most 0.032207. These source compositions
    # generate over 37 times the Doka pumping demand. No purchased gas power
    # occurs in this model; site administration remains with its native owner.
    export = generation - pump
    # The source engine reports a 13.5% useful-heat allocation.
    return _LandfillGasReference(
        branches,
        volume,
        dry_volume,
        energy,
        k_chp,
        k_flare,
        pump,
        generation,
        export,
        fsum(branch.heat_mj for branch in branches) - export * 3.6,
        0.135 * 0.3498 * energy,
    )


def _branch_emissions(branch: _GasBranch) -> tuple[_GasEmission, ...]:
    """Allocate actual carbon species by methane pool and retain source templates.

    Incoming CO2 stays with its own pool; newly formed CO2 and incomplete
    combustion products follow methane carbon. Principal fractions describe
    each resulting origin pool, excluding auxiliary carbon from the numerator.
    Oxygen remains the branch's separate atmospheric resource quantity.
    """
    emissions: list[_GasEmission] = []
    methane_c = fsum(pool.methane_carbon_kg for pool in branch.pools)
    carbon = cast(list[dict[str, object]], branch.trace["carbon"])
    for pool, ledger in zip(branch.pools, carbon, strict=True):
        origin = "fossil" if pool.origin == "fossil" else "non-fossil"
        share = pool.methane_carbon_kg / methane_c
        principal_origins = _gas_principal_origins(pool)
        emissions.append(
            _GasEmission(
                f"Carbon dioxide, {origin}",
                float(cast(float, ledger["carbon_dioxide_output_carbon_kg"])) * 44 / 12,
                "kilogram",
                principal_carbon_origin_fractions=principal_origins,
            )
        )
        for species, name in (("co", "Carbon monoxide"), ("methane", "Methane")):
            emissions.append(
                _GasEmission(
                    f"{name}, {origin}",
                    branch.products[species] * share,
                    "kilogram",
                    branch.templates.get(species),
                    principal_origins,
                )
            )
    for species, name in (
        ("nmvoc", _NMVOC_NAME),
        ("particles", _PM_NAME),
        ("nox", "Nitrogen oxides"),
        ("n2o", "Dinitrogen monoxide"),
        ("platinum", "Platinum"),
        ("so2", "Sulfur dioxide"),
    ):
        emissions.append(
            _GasEmission(name, branch.products[species], "kilogram", branch.templates.get(species))
        )
    emissions.append(_GasEmission("Water", branch.water_kg / 1000, "cubic meter"))
    return tuple(emissions)


def _gas_leakage_reference(
    pools: tuple[_GasPool, ...],
    nitrogen: float,
    sulfur: float,
    ammonia_nitrogen: float,
    fraction: float,
) -> _GasLeakage:
    """Release uncollected gas once, keeping atmospheric H2S chemistry outside plant."""
    emissions: list[_GasEmission] = []
    carbon = []
    for pool in pools:
        origin = "fossil" if pool.origin == "fossil" else "non-fossil"
        methane_c = fraction * pool.methane_carbon_kg
        dioxide_c = fraction * pool.carbon_dioxide_carbon_kg
        principal_origins = _gas_principal_origins(pool)
        emissions.extend(
            (
                _GasEmission(
                    f"Methane, {origin}",
                    methane_c * 16 / 12,
                    "kilogram",
                    principal_carbon_origin_fractions=principal_origins,
                ),
                _GasEmission(
                    f"Carbon dioxide, {origin}",
                    dioxide_c * 44 / 12,
                    "kilogram",
                    principal_carbon_origin_fractions=principal_origins,
                ),
            )
        )
        carbon.append(
            {
                "component": pool.component,
                "principal": pool.principal,
                "origin": pool.origin,
                "methane_carbon_kg": methane_c,
                "carbon_dioxide_carbon_kg": dioxide_c,
            }
        )
    escaped_nh3 = fraction * ammonia_nitrogen * 17 / 14
    emissions.append(_GasEmission("Ammonia", escaped_nh3, "kilogram"))
    emissions.append(_GasEmission("Sulfur dioxide", 2 * fraction * sulfur, "kilogram"))
    return _GasLeakage(
        tuple(emissions),
        {
            "carbon": carbon,
            "ammonia_kg": escaped_nh3,
            "dinitrogen_kg_physical": fraction * (nitrogen - ammonia_nitrogen),
            "h2s_kg_before_atmospheric_oxidation": fraction * sulfur * 34 / 32,
            "so2_kg_inventory": 2 * fraction * sulfur,
            "atmospheric_water_kg_physical": 18 * fraction * sulfur / 32,
            "atmospheric_oxygen_kg_physical": 48 * fraction * sulfur / 32,
        },
    )


def _wastewater_gas_reference(
    load: _WastewaterLoad,
    chp: _GasOperation,
    furnace: _GasOperation,
    profile: _WastewaterProcessProfile,
) -> _WastewaterGasReference:
    """Calculate actual digester combustion and useful energy before native binding.

    Carbon/N/S already include the selected digestion and fresh PAM. Source
    biogas/furnace normalization transfers their admitted emission functions;
    internal energy uses the actual species' LHV and regional efficiencies.
    The complete WWT parent applies collection and utility subtraction once.
    """
    pools = tuple(
        _GasPool(
            item.component,
            item.principal,
            item.origin,
            0.65 * item.digester_gas_carbon_kg,
            0.35 * item.digester_gas_carbon_kg,
        )
        for item in load.carbon
    )
    nitrogen = load.digester_gas_elements_kg["N"]
    sulfur = load.digester_gas_elements_kg["S"]
    # avgWWT AK10 NH3-N; molecular N2 uses 28, correcting DG25's atomic 14.
    ammonia_n = 0.015136275611828884 * nitrogen
    gas_carbon = fsum(item.digester_gas_carbon_kg for item in load.carbon)
    # Doka2021 p33 / avgWWT AL7: 22.4138 dry normal m3/kmol; molecular
    # gas amounts below use the source's rounded kg/kmol species masses.
    volume = 22.4138 * (
        gas_carbon / 12 + ammonia_n / 14 + (nitrogen - ammonia_n) / 28 + sulfur / 32
    )
    chp_fraction = (1 - 0.0075) * profile.chp_share
    flare_fraction = (1 - 0.0075) * (1 - profile.chp_share)
    k_chp = chp_fraction * volume / chp.fuel_amount
    # Source-equivalent biogas LHV transfers only this burner N2O factor;
    # it supplies no furnace, additional fuel or calorimetric heat estimate.
    k_flare = flare_fraction * volume * chp.fuel_lhv / (furnace.fuel_amount * furnace.fuel_lhv)
    branches = []
    for fraction, operation, multiplier, is_flare in (
        (flare_fraction, furnace, k_flare, True),
        (chp_fraction, chp, k_chp, False),
    ):
        branches.append(
            _combustion_branch(
                _gas_fraction(pools, fraction),
                fraction * nitrogen,
                fraction * sulfur,
                fraction * ammonia_n,
                fraction * volume,
                operation,
                multiplier,
                flare=is_flare,
                # Doka2021 WWcalc EC14, mg particles/m3 gas, converted to kg/m3;
                # the report p35 rounds this WWT-specific factor to 46.75.
                particulate_per_m3=46.74857142857142 / 1e6,
            )
        )
    # Chase (1998), via NIST Chemistry WebBook: HHV at 298.15 K, less 44 kJ/mol of product water,
    # gives the actual-species LHV input to the selected CHP. Doka's 19.079*S and 6.276*N describe
    # the source elemental heat references. Methane has 16/12 kg per kg methane C; NH3 and H2S use
    # 17/14 and 34/32. CO2/N2 have zero fuel-energy contributions. Efficiencies act before
    # incomplete combustion.
    fuel_lhv = (
        0.65 * gas_carbon * 16 / 12 * (_HHV_CH4 - (44 / 18) * (36 / 16))
        + ammonia_n * 17 / 14 * (_HHV_NH3 - (44 / 18) * (27 / 17))
        + sulfur * 34 / 32 * (_HHV_H2S - (44 / 18) * (18 / 34))
    )
    electricity = chp_fraction * fuel_lhv * profile.electricity_efficiency / 3.6
    useful_heat = chp_fraction * fuel_lhv * profile.heat_efficiency
    # Gas HHV waste heat uses the specified inventory allocation; electricity is assigned at its
    # consuming meter.
    heat_air = fsum(branch.heat_mj for branch in branches) - 3.6 * electricity
    return _WastewaterGasReference(
        pools,
        tuple(branches),
        nitrogen,
        sulfur,
        ammonia_n,
        volume,
        k_chp,
        k_flare,
        electricity,
        useful_heat,
        heat_air,
    )
