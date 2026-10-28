"""Compile sludge landfarming for plastics and chemical wastewater.

Doka (2021) supplies integrated agricultural element fates, carried-water
allocation and vacuum-tanker spreading. Each source defines the municipal or
plastic-leachate reference and carbon oxidation rule.
"""

import json
from dataclasses import asdict, dataclass
from math import fsum
from typing import Literal, cast

from .catalogue import provider_service_code
from .leachate_load import _CarbonOrigin, _WastewaterLoad
from .material_carbon import CARBON_ORIGINS, PrincipalCarbonOriginFractions
from .native_recipe import _self_production_amount, _service_trace
from .pa_receiving_records import (
    PaReceivingFlow,
    PaReceivingLandfarmingReference,
    PaReceivingOperation,
    _activity,
    _flow,
    _receiving_row,
)
from .recipe_context import _copied_native_exchange, _RecipeContext, _scaled_native_exchange

_RURAL = ("air", "non-urban air or from high stacks")
_SURFACE = ("water", "surface water")
_GROUND = ("water", "ground-")
_SOIL = ("soil", "agricultural")

# Native landfarming flow names and production-property elements. Soil emissions use gross input
# (Doka 2021 section 17.2). N/P fate coefficients include compound conversion.
_FLOW_ELEMENTS = {
    "Beryllium II": ("Be", "beryllium"),
    "Strontium II": ("Sr", "strontium"),
    "Thallium I": ("Tl", "thallium"),
    "Titanium ion": ("Ti", "titanium"),
    "Ammonia": ("N", "nitrogen"),
    "Dinitrogen monoxide": ("N", "nitrogen"),
    "Nitrogen oxides": ("N", "nitrogen"),
    "Nitrate": ("N", "nitrogen"),
    "Phosphate": ("P", "phosphorus"),
    "Aluminium III": ("Al", "aluminium"),
    "Antimony ion": ("Sb", "antimony"),
    "Arsenic ion": ("As", "arsenic"),
    "Barium II": ("Ba", "barium"),
    "Boron": ("B", "boron"),
    "Cadmium II": ("Cd", "cadmium"),
    "Calcium II": ("Ca", "calcium"),
    "Chromium III": ("Cr", "chromium"),
    "Cobalt II": ("Co", "cobalt"),
    "Copper ion": ("Cu", "copper"),
    "Iron ion": ("Fe", "iron"),
    "Lead II": ("Pb", "lead"),
    "Magnesium": ("Mg", "magnesium"),
    "Manganese II": ("Mn", "manganese"),
    "Mercury II": ("Hg", "mercury"),
    "Molybdenum VI": ("Mo", "molybdenum"),
    "Nickel II": ("Ni", "nickel"),
    "Potassium I": ("K", "potassium"),
    # The average municipal-sludge donor also contains Sc, Ag and W. Resolve
    # their native names before selecting the actual plastics/chemical sludge
    # elements; an absent target element receives no donor-metal release.
    "Scandium": ("Sc", "scandium"),
    "Selenium IV": ("Se", "selenium"),
    "Silicon": ("Si", "silicon"),
    "Silver I": ("Ag", "silver"),
    "Sodium I": ("Na", "sodium"),
    "Sulfur": ("S", "sulfur"),
    "Tin ion": ("Sn", "tin"),
    "Tungsten": ("W", "tungsten"),
    "Vanadium V": ("V", "vanadium"),
    "Zinc II": ("Zn", "zinc"),
}
# The completed WWT owner admits these elements. H/O remain its
# allocated dry descriptors; phosphorus and carbon have local-fate roles.
_TARGET_ELEMENTS = {
    "C": "carbon",
    "N": "nitrogen",
    "S": "sulfur",
    "P": "phosphorus",
    "Cl": "chlorine",
    "Zn": "zinc",
    "Fe": "iron",
    "Mg": "magnesium",
    "Na": "sodium",
    "Co": "cobalt",
    "Ca": "calcium",
    "K": "potassium",
    "B": "boron",
    "Si": "silicon",
    "Al": "aluminium",
    "Ti": "titanium",
}


@dataclass(frozen=True, slots=True)
class LandfarmingCarbon:
    """One actual component's kg C entering soil after the agricultural allocation."""

    component: str
    principal: bool
    origin: _CarbonOrigin
    carbon_kg: float


@dataclass(frozen=True, slots=True)
class LandfarmingRelease:
    """A nonzero actual release in its indexed original flow's native unit.

    ``source_row`` indexes ``reference.operation.rows``. The binder scales that
    row's uncertainty to ``amount``; the pure calculation holds no native node.
    """

    source_row: int
    amount: float


@dataclass(frozen=True, slots=True)
class LandfarmingCarbonDioxide:
    """Actual kg CO2 with principal fractions relative to this entire origin group.

    Auxiliary C remains in the denominator. Biogenic and DAC principal carbon share the existing
    non-fossil flow, with fractions determined from the actual incoming material.
    """

    origin: Literal["fossil", "non-fossil"]
    amount_kg: float
    principal_carbon_origin_fractions: PrincipalCarbonOriginFractions


@dataclass(frozen=True, slots=True)
class LandfarmingLoad:
    """Agricultural sludge quantities and releases on the caller's basis.

    Mass and element/crop pools are kg; empirical heat is MJ. Signed bulk heat
    remains separate from emitted heat. Indexed releases retain native units.
    Phosphate additions use kg PO4 in each water compartment; mineral-soil
    additions use kg element. Donor rows remain in the indexed releases.
    Spreading applies to wet sludge once. Crop uptake is recorded as a fate.
    """

    wet_kg: float
    dry_kg: float
    carried_water_kg: float
    elements_kg: dict[str, float]
    allocated_hydrogen_kg: float
    allocated_oxygen_kg: float
    carbon_components: tuple[LandfarmingCarbon, ...]
    carbon_by_origin_kg: dict[str, float]
    empirical_unfloored_heat_mj: float
    empirical_heat_mj: float
    native_releases: tuple[LandfarmingRelease, ...]
    carbon_dioxide: tuple[LandfarmingCarbonDioxide, ...]
    phosphate_water_kg: dict[tuple[str, ...], float]
    mineral_soil_kg: dict[str, float]
    nitrogen_crop_uptake_kg: float
    phosphorus_crop_uptake_kg: float


def calculate_landfarming(
    wastewater: _WastewaterLoad,
    reference: PaReceivingLandfarmingReference,
    *,
    complete_carbon_oxidation: bool,
    phosphorus_equilibrium: bool,
) -> LandfarmingLoad:
    """Calculate actual agricultural sludge from admitted original source coefficients.

    Dry elements and C receive the agricultural share once; wet sludge and
    carried moisture already include it. Original reference rows are normalized
    per kg wet donor; intensive dry/water/content properties are not divided by
    production again. Native municipal routes retain their selected CO2 and
    phosphate factors. Reconstructed PA selects both complete carbon
    oxidation and report-equilibrium phosphorus, independently of donor P rows.
    """
    fraction = wastewater.landfarming_dry_share
    sludge = wastewater.final_sludge
    wet = wastewater.landfarmed_sludge_wet_kg
    water = wastewater.water.landfarmed_sludge_water_kg
    dry = fraction * sludge.dry_mass_kg
    elements = {element: fraction * mass for element, mass in sludge.elements_kg.items()}
    hydrogen = fraction * sludge.allocated_hydrogen_kg
    oxygen = fraction * sludge.allocated_oxygen_kg
    carbon = tuple(
        LandfarmingCarbon(
            item.component, item.principal, item.origin, fraction * item.final_sludge_carbon_kg
        )
        for item in wastewater.carbon
    )
    origins = {
        origin: fsum(
            fraction * item.final_sludge_carbon_kg
            for item in wastewater.carbon
            if (item.origin == "fossil") == (origin == "fossil")
        )
        for origin in ("fossil", "non-fossil")
    }
    # WWcalc DU7: MJ from allocated bulk O/H and actual C/S/N (kg).
    unfloored_heat = (
        -9.8324 * oxygen
        + 124.265 * hydrogen
        + 34.016 * elements["C"]
        + 19.079 * elements["S"]
        + 6.276 * elements["N"]
    )
    # Original Wastewater 2021.xlsm, WWcalc DU18=IF(DU7<0,0,DU7).
    # Its empirical allocation emits no negative soil heat. For negative DU7,
    # retain the signed estimate in the ledger and emit no heat.
    heat = 0.0 if unfloored_heat < 0 else unfloored_heat
    # Original 531730/531731 assume complete sludge-carbon oxidation after deposition. Combine the
    # native total, then attribute it by actual carbon origin, retaining biological/fossil auxiliary
    # identities.
    source_co2 = fsum(
        row.amount
        for row in reference.operation.rows
        if cast(PaReceivingFlow, row.endpoint).name
        in {"Carbon dioxide, fossil", "Carbon dioxide, non-fossil"}
    )
    source_carbon = reference.dry_fraction * reference.element_contents["C"]
    emissions: dict[tuple[str, tuple[str, ...]], float] = {}
    releases = []
    for index, row in enumerate(reference.operation.rows):
        flow = cast(PaReceivingFlow, row.endpoint)
        name, categories = flow.name, flow.categories
        coefficient = row.amount
        if name in {"Carbon dioxide, fossil", "Carbon dioxide, non-fossil"}:
            continue  # Actual carbon groups are constructed once below.
        if phosphorus_equilibrium and name == "Phosphate":
            continue  # The selected actual-P equation owns both outlets.
        if name == "Water":
            # Section 17.2.3: fixed original site evaporation/groundwater split of carried moisture.
            amount = coefficient * water / reference.water_fraction
        elif name == "Heat, waste":
            amount = heat
        elif name == "Non-hazardous waste disposed":
            amount = coefficient * wet
        else:
            element, _ = _FLOW_ELEMENTS[name]
            # No unlisted metal is present in the admitted WWT load.
            if element not in elements:
                continue
            donor = reference.dry_fraction * reference.element_contents[element]
            amount = coefficient * elements[element] / donor
        emissions[name, categories] = amount
        if amount != 0:
            releases.append(LandfarmingRelease(index, amount))
    carbon_dioxide = []
    for origin, mass in origins.items():
        # A fossil-only manufacturing solvent supplies an empty non-fossil carbon pool.
        if mass == 0:
            continue
        # Plastic-leachate landfarming uses Doka's complete C oxidation,
        # 44 kg CO2/12 kg C; municipal sludge keeps its admitted native ratio.
        amount = (44 / 12 if complete_carbon_oxidation else source_co2 / source_carbon) * mass
        fractions = cast(
            PrincipalCarbonOriginFractions,
            tuple(
                fsum(
                    fraction * pool.final_sludge_carbon_kg
                    for pool in wastewater.carbon
                    if pool.principal
                    and pool.origin == actual_origin
                    and (pool.origin == "fossil") == (origin == "fossil")
                )
                / mass
                for actual_origin in CARBON_ORIGINS
            ),
        )
        carbon_dioxide.append(
            LandfarmingCarbonDioxide(
                cast(Literal["fossil", "non-fossil"], origin), amount, fractions
            )
        )
    # Doka 2021 Eq. 17.5 and Table 17.1: uptake complements emitted nutrient fractions. Nominal
    # source compound masses (g/mol) convert existing kg emissions only for this mass-balance
    # ledger.
    nitrogen_emitted = (
        emissions["Ammonia", _RURAL] * 14 / 17
        + emissions["Dinitrogen monoxide", _RURAL] * 28 / 44
        + emissions["Nitrogen oxides", _RURAL] * 14 / 46
        + emissions["Nitrate", _SURFACE] * 14 / 62
    )
    phosphate: dict[tuple[str, ...], float] = {}
    if phosphorus_equilibrium:
        # Doka 2021 pp. 56-58, Eq. 17.6/Table 17.1: groundwater 0.07 and surface
        # run-off 0.175 kg P/(ha year). Erosion uses 3000 kg eroded soil/(ha year)
        # times 0.00095 kg P/kg soil, 1.86 enrichment and 0.2 delivery fraction
        # to surface water. Normalize by 12 kg P/(ha year), keeping unrounded
        # erosion factors and combining run-off and erosion once.
        phosphorus_ground = elements["P"] * 0.07 / 12
        phosphorus_surface = elements["P"] * (0.175 + 3000 * 0.00095 * 1.86 * 0.2) / 12
        phosphorus_emitted = phosphorus_ground + phosphorus_surface
        # Wastewater2021 WWcalc C408:C409 converts compounds; the crop ledger retains elemental
        # phosphorus.
        phosphate = {
            compartment: mass * (30.97 + 4 * 16) / 30.97
            for compartment, mass in ((_GROUND, phosphorus_ground), (_SURFACE, phosphorus_surface))
            if mass != 0
        }
    else:
        # Established native routes sum only supplied species, with the
        # existing 31/95 kg P/kg nominal phosphate ledger convention.
        phosphorus_emitted = (
            fsum(
                amount
                for (name, compartment), amount in emissions.items()
                if name == "Phosphate" and compartment in (_SURFACE, _GROUND)
            )
            * 31
            / 95
        )
    # Doka 2021 section 17.2 inventories gross elemental input to agricultural
    # soil. Actual UPR glass and road minerals need this fate when absent from
    # the selected donor properties;
    # existing soil rows retain their native coefficients and uncertainty.
    mineral_soil = {
        name: elements[element]
        for name, element in (
            ("Boron", "B"),
            ("Silicon", "Si"),
            ("Calcium II", "Ca"),
            ("Aluminium III", "Al"),
            ("Titanium ion", "Ti"),
            ("Potassium I", "K"),
            ("Magnesium", "Mg"),
            ("Sodium I", "Na"),
            ("Sulfur", "S"),
        )
        if (name, _SOIL) not in emissions and elements[element] != 0
    }
    return LandfarmingLoad(
        wet,
        dry,
        water,
        elements,
        hydrogen,
        oxygen,
        carbon,
        origins,
        unfloored_heat,
        heat,
        tuple(releases),
        tuple(carbon_dioxide),
        phosphate,
        mineral_soil,
        elements["N"] - nitrogen_emitted,
        elements["P"] - phosphorus_emitted,
    )


def _landfarming_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    wastewater: _WastewaterLoad,
    load_owner_service_id: str,
    database_name: str,
    *,
    complete_carbon_oxidation: bool,
    phosphorus_equilibrium: bool = False,
) -> dict[str, object]:
    """Supply negative-kg landfarming from one completed wastewater load.

    The caller supplies the load per kg plastic or chemical campaign, before
    collection. Sludge includes residual PAM/Fe. Apply the agricultural share
    to dry elements and carbon once; wet sludge and carried water already
    include that share and 97% moisture. The separate sludge market owns
    collection and haulage.

    Native negative production and distributions are retained. Source properties
    are per kg product. Crop N/P uptake carries no credit; water describes
    carried moisture and heat uses Doka's empirical allocation.

    Reconstructed PA selects ``phosphorus_equilibrium``: Doka's rates allocate
    elemental P to two water outlets and crop uptake, omitting the negligible
    generic slurry adjustments. Other callers retain native phosphorus factors.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    selector = cast(dict[str, object], construction["source"])
    _service_trace(context, service)
    original = context.source.activity(selector, year=2025)
    original_rows = context.exchanges(original)
    source_p = -_self_production_amount(original, original_rows, method="sludge landfarming source")
    source_production = next(ex for ex in original_rows if ex["type"] == "production")
    source_properties = cast(dict[str, dict[str, object]], source_production["properties"])
    original_biosphere = [ex for ex in original_rows if ex["type"] == "biosphere"]
    # Admit only intensive denominators actually used by this selected load.
    # A missing donor P row cannot supply the phosphorus denominator.
    required = {"C": "carbon"}
    for ex in original_biosphere:
        name = str(context.input(ex)["name"])
        if name in _FLOW_ELEMENTS:
            element, property_name = _FLOW_ELEMENTS[name]
            if element in wastewater.final_sludge.elements_kg and not (
                phosphorus_equilibrium and element == "P"
            ):
                required[element] = property_name
    source_reference = PaReceivingLandfarmingReference(
        PaReceivingOperation(
            _activity(original),
            -source_p,
            tuple(
                _receiving_row(_flow(context.input(ex)), ex, 1 / source_p)
                for ex in original_biosphere
            ),
        ),
        float(cast(float, source_properties["dry mass"]["amount"])),
        float(cast(float, source_properties["water in wet mass"]["amount"])),
        {
            element: float(cast(float, source_properties[f"{name} content"]["amount"]))
            for element, name in required.items()
        },
    )
    load = calculate_landfarming(
        wastewater,
        source_reference,
        complete_carbon_oxidation=complete_carbon_oxidation,
        phosphorus_equilibrium=phosphorus_equilibrium,
    )
    fraction = wastewater.landfarming_dry_share
    wet, dry, water = load.wet_kg, load.dry_kg, load.carried_water_kg
    elements = load.elements_kg
    hydrogen, oxygen = load.allocated_hydrogen_kg, load.allocated_oxygen_kg
    carbon = [asdict(item) for item in load.carbon_components]
    origins, heat = load.carbon_by_origin_kg, load.empirical_heat_mj
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="actual_sludge_landfarming",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    p = _self_production_amount(activity, native, method="leachate sludge landfarming")
    scale = -p / wet
    production = next(ex for ex in native if ex["type"] == "production")
    properties: dict[str, object] = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": dry / wet, "unit": "kg"},
        "water in wet mass": {"amount": water / wet, "unit": "kg"},
        "carbon allocation": {"amount": elements["C"] / wet, "unit": "kg"},
        "heating value, gross": {
            "amount": heat / wet,
            "unit": "MJ",
            "comment": "WWcalc DU18 allocation from DU7 using separately allocated H/O.",
        },
    }
    properties.update(
        {
            f"{_TARGET_ELEMENTS[element]} content": {"amount": mass / dry, "unit": "dimensionless"}
            for element, mass in elements.items()
        }
    )
    properties.update(
        {
            f"carbon content, {origin}": {"amount": mass / dry, "unit": "dimensionless"}
            for origin, mass in origins.items()
        }
    )
    reference = _copied_native_exchange(production)
    for field in ("flow", "activity", "production volume", "classifications"):
        reference.pop(field, None)
    reference.update(
        input=(database_name, provider_service_code(service_id)),
        name=str(service["reference_product"]),
        properties=properties,
        comment=(
            "Actual leachate sludge elements and water; dry mass includes "
            "separately traced nominal H/O."
        ),
    )
    generated = [reference]
    # Doka 2021 section 17.2.5: the complete spreading service covers machinery and operating
    # inputs, excludes sludge emissions, and uses 1030 kg/m3. Retain its actual dated rows once.
    retained = [ex for ex in native if ex["type"] == "technosphere"]
    generated.extend(_copied_native_exchange(ex) for ex in retained)
    for release in load.native_releases:
        ex = original_biosphere[release.source_row]
        generated.append(_scaled_native_exchange(ex, release.amount * scale / float(ex["amount"])))
    for name, amount in load.mineral_soil_kg.items():
        flow = context.biosphere.flow({"name": name, "unit": "kilogram", "categories": list(_SOIL)})
        generated.append(
            {
                "input": cast(tuple[str, str], flow.key),
                "name": name,
                "unit": "kilogram",
                "type": "biosphere",
                "amount": amount * scale,
                "uncertainty type": 0,
                "comment": (
                    "Doka2021 section17.2 gross input of actual cured-material "
                    "sludge mineral to soil."
                ),
            }
        )
    templates = cast(dict[str, str], construction["carbon_templates"])
    for carbon_release in load.carbon_dioxide:
        origin, amount = carbon_release.origin, carbon_release.amount_kg
        name = f"Carbon dioxide, {origin}"
        template_name = f"Carbon dioxide, {templates[origin]}"
        template = next(
            ex
            for ex in original_rows
            if ex["type"] == "biosphere" and context.input(ex)["name"] == template_name
        )
        row = _scaled_native_exchange(template, amount * scale / float(template["amount"]))
        if template_name != name:
            node = context.biosphere.flow(
                {"name": name, "unit": "kilogram", "categories": list(_RURAL)}
            )
            row.update(input=cast(tuple[str, str], node.key), name=name)
            for field in ("flow", "activity", "properties"):
                row.pop(field, None)
        row["principal_carbon_origin_fractions"] = carbon_release.principal_carbon_origin_fractions
        generated.append(row)
    for compartment, amount in load.phosphate_water_kg.items():
        flow = context.biosphere.flow(
            {"name": "Phosphate", "unit": "kilogram", "categories": list(compartment)}
        )
        generated.append(
            {
                "input": cast(tuple[str, str], flow.key),
                "type": "biosphere",
                "name": "Phosphate",
                "unit": "kilogram",
                "amount": amount * scale,
                "uncertainty type": 0,
                "comment": "Doka2021 report equilibrium P fate for reconstructed PA sludge.",
            }
        )
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = "[]"
    trace["retained_biosphere_json"] = "[]"
    trace["removed_biosphere_json"] = context.exchange_json(
        ex for ex in native if ex["type"] == "biosphere"
    )
    trace["project_terms_json"] = json.dumps(
        {
            "load_owner_service_id": load_owner_service_id,
            "reference": "caller load before collection treatment fraction",
            "landfarming_dry_share": fraction,
            "wet_kg": wet,
            "dry_reference_kg": dry,
            "carried_water_kg": water,
            "actual_elements_kg": elements,
            "additional_mineral_soil_kg": load.mineral_soil_kg,
            "carbon_components": carbon,
            "allocated_hydrogen_kg": hydrogen,
            "allocated_oxygen_kg": oxygen,
            "empirical_soil_heat_MJ": heat,
            "unfloored_empirical_soil_heat_MJ": load.empirical_unfloored_heat_mj,
            "source_production_properties": source_properties,
            "nitrogen_crop_uptake_kg": load.nitrogen_crop_uptake_kg,
            "phosphorus_crop_uptake_kg": load.phosphorus_crop_uptake_kg,
            "phosphorus_allocation": (
                "Doka2021 report equilibrium applied to actual P"
                if phosphorus_equilibrium
                else "Established native phosphate species and 31/95 elemental-ledger conversion"
            ),
            "nutrient_allocation": "Doka local fates; uptake uncharacterized, no fertilizer credit",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }
