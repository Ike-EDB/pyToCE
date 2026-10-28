"""Allocate municipal hydraulics to chemical-manufacturing water.

PA finish esters, hydrogenated castor oil and PBS/PBAT polycondensation use
the admitted source-region profiles for collected formation water or condensed
inlet moisture. The actual target effluent composition replaces the native average sewage
composition.
Sewers, volume-related operation and infrastructure remain.

The actual-load TPS leachate calculation also uses the source evaporation equation.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, cast

from .catalogue import provider_service_code
from .native_recipe import _service_trace
from .recipe_context import _copied_native_exchange, _RecipeContext, _scaled_native_exchange


@dataclass(frozen=True, slots=True)
class _WaterTreatmentProfile:
    """Doka stage shares and distinct plant-size and emission-location fractions."""

    stages: tuple[float, float, float]
    urban_treatment: float
    urban_emissions: float


# Central Repository 2020, GNIWWT row 337 (Global underlying RoW): AO:AQ
# are one/two/three-stage treatment shares; AY weights urban plant size,
# whereas AZ allocates emissions to urban air. These are distinct source
# proportions applied to the selected hydraulic function in each region.
_ROW_PROFILE = _WaterTreatmentProfile(
    (0.08203366382544722, 0.4115654223641656, 0.5064009138103872),
    0.6808092816046385,
    0.7217062553373826,
)
# Doka Central Repository 2020 GNIWWT rows 99/363/337/190: Canada, Europe, Global and Switzerland
# source profiles. CA-QC and Europe without Switzerland use their native source's larger
# geographical model. Canada's rounded stage shares total 1.0001; the admitted new hydraulic
# construction normalizes those fixed source shares once. Other source shares already sum to one.
_WATER_PROFILES = {
    "CA-QC": _WaterTreatmentProfile(
        (0.1905 / 1.0001, 0.631 / 1.0001, 0.1786 / 1.0001),
        0.8674556018533911,
        0.8674556028920686,
    ),
    "Europe without Switzerland": _WaterTreatmentProfile(
        (0.01892642746691165, 0.2279501594273743, 0.7531234131057141),
        0.8243136538013532,
        0.8277305113123367,
    ),
    "RoW": _ROW_PROFILE,
    "CH": _WaterTreatmentProfile((0.0, 0.1122, 0.8878), 0.7518243149997053, 0.7518241277622788),
}
_LAND_FLOWS = {
    "Occupation, construction site",
    "Occupation, industrial area",
    "Occupation, urban/industrial fallow (non-use)",
    "Transformation, from pasture, man made",
    "Transformation, to industrial area",
    "Transformation, to urban/industrial fallow (non-use)",
}
# Premise 2.4.6 heat.py relinks the original gas-heat markets to its generic
# industrial-heat product. Both names identify this plant's one heating duty;
# retain the dated suppliers and their shares when applying the hydraulic load.
_HEAT_PRODUCTS = {"heat, district or industrial, natural gas", "heat, district or industrial"}


def _water_reference(
    exchange: Any, key: tuple[str, str], reference_product: str
) -> dict[str, object]:
    """Retain the signed water-service amount on its specified reference basis.

    The service carries 1000 kg water per m3. Any actual brine ions belong to
    its separate parent; dry-basis ratios and original sewage identities have
    no meaning for this hydraulic function. Original records stay in provenance.
    """
    copied = _copied_native_exchange(exchange)
    for field in ("flow", "activity", "production volume", "classifications"):
        copied.pop(field, None)
    copied.update(
        input=key,
        name=reference_product,
        properties={
            "wet mass": {"amount": 1000.0, "unit": "kg"},
            "dry mass": {"amount": 0.0, "unit": "kg"},
        },
        comment="Nominal pure-water hydraulic service; target solutes are owned by its parent.",
    )
    return copied


def _water_emission(
    context: _RecipeContext, name: str, unit: str, categories: list[str], amount: float
) -> dict[str, object]:
    """Bind a deterministic target water, heat or ion quantity to its native flow."""
    flow = context.biosphere.flow({"name": name, "unit": unit, "categories": categories})
    return {
        "input": cast(tuple[str, str], flow.key),
        "name": name,
        "unit": unit,
        "amount": amount,
        "loc": amount,
        "uncertainty type": 0,
        "type": "biosphere",
    }


def _water_evaporation(comment: str, profile: _WaterTreatmentProfile) -> float:
    """Calculate source-allocated m3 evaporated per m3 treated water.

    Doka's passive-pool and aeration terms allocate evaporation in a mixed
    municipal plant.
    Original ecoinvent MAT/MAP/EAT metadata supply the fixed source climate;
    premise's omission of those annotations cannot replace it with future weather.
    """
    climate = {
        label: float(value)
        for label, value in re.findall(r"\((MAT|MAP|EAT)\) of ([0-9.]+)", comment)
    }
    # Doka (2021) assumes wastewater temperature midway between mean annual
    # ambient temperature and household wastewater at 20 degrees C.
    temperature = (climate["MAT"] + 20.0) / 2.0
    # Infra O19/P19 use 1.6/120 million m3/year rural/urban plants. Their
    # difference is 118.4 million; the source urban plant share selects the
    # mixture and 365.25 converts the annual reference to m3/day (WWcalc C204).
    daily_volume = (1_600_000.0 + 118_400_000.0 * profile.urban_treatment) / 365.25
    # WWcalc C203: 17 h retention, 24 h/day and 3.5 m depth give exposed m2.
    pool_area = daily_volume * 17.0 / 24.0 / 3.5
    # Eq.13.4 / C212:C213 estimate relative humidity from actual annual
    # evapotranspiration / precipitation, capped at saturation (fraction 1).
    # 0.4562 and -0.8267 are the source's empirical coefficients.
    humidity = min(1.0, 0.4562 * (climate["EAT"] / climate["MAP"]) ** -0.8267)
    # Eq.13.2 / C214: Tetens saturation pressure in kPa; 0.61078 kPa,
    # 17.27 and 237.3 degrees C are the published temperature-fit constants.
    pressure = 0.61078 * math.exp(17.27 * temperature / (temperature + 237.3))
    # Eq.13.1 / C205/C216: McJannet open-water coefficients 2.36, 1.67 and
    # -0.05 use area in m2 and the assumed 2 m/s wind at 2 m above the surface.
    # Multiply kg/(m2 day) by pool area; divide by 1000 kg/m3 and m3/day
    # throughput to obtain evaporated m3 per treated m3.
    passive = (
        (2.36 + 1.67 * 2.0)
        * pool_area**-0.05
        * (1.0 - humidity)
        * pressure
        * pool_area
        / (1000.0 * daily_volume)
    )
    # WWcalc C221 gives kg water/kg saturated air using the coefficient
    # 0.0044 and exponential temperature factor 0.0591 per degree C.
    saturated_water = 0.0044 * math.exp(0.0591 * temperature)
    # C222 / report footnote 35: 6000 m3 air/h * 1.2 kg air/m3 divided by (16000 m3 wastewater/day /
    # 24 h/day) gives 10.8 kg air/m3 wastewater. C224 assumes 80% average/max aeration loading; only
    # two/three-stage plants aerate (L11). Divide water kg by 1000 kg/m3 for volume fraction.
    aeration = (
        10.8
        * saturated_water
        * (1.0 - humidity)
        / 1000.0
        * 0.8
        * (profile.stages[1] + profile.stages[2])
    )
    return passive + aeration


def _water_market_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Retain complete sewers and the signed plant handoff for nominal water.

    The untreated share is one minus plant demand divided by market production.
    The child's own negative production is normalized later by Brightway, once.
    Sewage solute emissions are removed; a saline parent owns its actual ions.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    plant_service_id = str(cast(dict[str, object], construction["plant"])["service_id"])
    _service_trace(context, service)
    activity, trace = context.activity(
        cast(dict[str, object], construction["selector"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="water_collection",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    production = next(ex for ex in native if ex["type"] == "production")
    generated = [
        _water_reference(
            production,
            (database_name, provider_service_code(service_id)),
            str(service["reference_product"]),
        )
    ]
    plant = next(
        ex
        for ex in native
        if ex["type"] == "technosphere"
        and context.input(ex)["reference product"] == "wastewater, average"
    )
    generated.append(
        _water_reference(
            plant,
            (database_name, provider_service_code(plant_service_id)),
            str(service["reference_product"]),
        )
    )
    retained = [
        ex
        for ex in native
        if ex["type"] == "technosphere"
        and context.input(ex)["reference product"] in {"sewer grid, rural", "sewer grid, urban"}
    ]
    generated.extend(_copied_native_exchange(ex) for ex in retained)
    volume = -float(production["amount"])
    treated = float(plant["amount"]) / float(production["amount"])
    generated.append(
        _water_emission(
            context, "Water", "cubic meter", ["water", "surface water"], volume * (1.0 - treated)
        )
    )
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = context.exchange_json([plant])
    trace["retained_biosphere_json"] = "[]"
    trace["removed_biosphere_json"] = context.exchange_json(
        ex for ex in native if ex["type"] == "biosphere"
    )
    trace["project_terms_json"] = json.dumps(
        {"treated_fraction": treated, "plant_service_id": plant_service_id},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "cubic meter",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }


def _water_plant_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
    profile: _WaterTreatmentProfile,
) -> dict[str, object]:
    """Allocate hydraulic operation, screenings and infrastructure to pure water.

    Zero incoming C/N/P gives zero biological, sludge and treatment-reagent
    duties under Doka's equations. Keep volume-allocated screenings and all
    infrastructure. Existing supplier shares and utility uncertainty are retained.
    Heat and evaporation are allocated life cycle inventory indicators. Conservative salts remain at
    an actual brine parent.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    selector = cast(dict[str, object], construction["selector"])
    _service_trace(context, service)
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="hydraulic_water_treatment",
        semantic_class="service_normalization",
    )
    original = context.source.activity(selector, year=2025)
    evaporation = _water_evaporation(str(original["comment"]), profile)
    native = context.exchanges(activity)
    production = next(ex for ex in native if ex["type"] == "production")
    volume = -float(production["amount"])
    technosphere = [ex for ex in native if ex["type"] == "technosphere"]
    limestone = math.fsum(
        float(ex["amount"])
        for ex in technosphere
        if context.input(ex)["reference product"] == "limestone, crushed, washed"
    )
    # Doka avgWWT C119 is average three-stage electricity in kWh/m3;
    # B114/B115/B118 are its mechanical, biological and other operating shares.
    # For the nominal pure-water load, the mechanical duty remains but the
    # incoming C/N/P-dependent biological and sludge duties are zero.
    total_source_electricity = 0.33306535778266716
    mechanical = 0.17228167919068399
    biological = 0.56650068777551799
    other = 0.057409752723465199
    first, second, third = profile.stages
    # avgWWT D118 allocates other electricity in a one-stage plant in the
    # mechanical / (mechanical + biological) ratio; E118/F118 retain all other
    # electricity in two/three-stage plants. Weight by the selected source profile.
    hydraulic = total_source_electricity * (
        mechanical + other * (first * mechanical / (mechanical + biological) + second + third)
    )
    # Infra G28 = 84150 MJ / 3.6 = 23375 kWh annual infrastructure electricity; G33 = 13250 kg
    # annual limestone. Shared capacity, lifetime, and stage allocation let limestone demand carry
    # the corresponding electricity.
    electricity = volume * hydraulic + limestone * 23375.0 / 13250.0
    # avgWWT B123 allocates 10% of heating to water volume rather than digestion;
    # C124 supplies 0.191478062936035 kWh thermal/m3, converted by 3.6 MJ/kWh.
    heat = volume * 3.6 * 0.1 * 0.191478062936035
    source_electricity = math.fsum(
        float(ex["amount"])
        for ex in technosphere
        if context.input(ex)["reference product"] == "electricity, medium voltage"
    )
    source_heat = math.fsum(
        float(ex["amount"])
        for ex in technosphere
        if context.input(ex)["reference product"] in _HEAT_PRODUCTS
    )
    generated = [
        _water_reference(
            production,
            (database_name, provider_service_code(service_id)),
            str(service["reference_product"]),
        )
    ]
    # The original Europe average-sewage plant's digester covers its heat duty. The constructed
    # pure-water operation supplies the source-model heat deficit through the selected complete
    # industrial furnace. Other regional owners rescale their existing heat purchases below.
    if "source_heat" in construction:
        heat_supplier, _ = context.activity(
            cast(dict[str, object], construction["source_heat"]),
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id="nominal_water_external_heating",
            semantic_class="service_normalization",
        )
        generated.append(
            {
                "input": cast(tuple[str, str], heat_supplier.key),
                "type": "technosphere",
                "amount": heat,
                "loc": heat,
                "uncertainty type": 0,
                "comment": "Doka WWcalc B152/B155 pure-water heat deficit, MJ delivered.",
            }
        )
    retained: list[Any] = []
    removed: list[Any] = []
    retained_land: list[Any] = []
    for ex in native:
        if ex["type"] == "technosphere":
            product = str(context.input(ex)["reference product"])
            if product == "electricity, medium voltage":
                generated.append(_scaled_native_exchange(ex, electricity / source_electricity))
                removed.append(ex)
            elif product in _HEAT_PRODUCTS:
                generated.append(_scaled_native_exchange(ex, heat / source_heat))
                removed.append(ex)
            elif product in {"iron sulfate", "polyacrylamide"} or product.startswith(
                "sewage sludge,"
            ):
                removed.append(ex)
            else:
                generated.append(_copied_native_exchange(ex))
                retained.append(ex)
        elif ex["type"] == "biosphere" and context.input(ex)["name"] in _LAND_FLOWS:
            generated.append(_copied_native_exchange(ex))
            retained_land.append(ex)
    # Doka allocates electricity dissipation to waste heat (3.6 MJ/kWh).
    # Delivered external heat, already in MJ, is included once.
    dissipated_heat = 3.6 * electricity + heat
    for air, share in (
        ("urban air close to ground", profile.urban_emissions),
        ("non-urban air or from high stacks", 1.0 - profile.urban_emissions),
    ):
        generated.append(
            _water_emission(
                context, "Heat, waste", "megajoule", ["air", air], dissipated_heat * share
            )
        )
        generated.append(
            _water_emission(
                context, "Water", "cubic meter", ["air", air], volume * evaporation * share
            )
        )
    generated.append(
        _water_emission(
            context,
            "Water",
            "cubic meter",
            ["water", "surface water"],
            volume * (1.0 - evaporation),
        )
    )
    generated.append(
        _water_emission(
            context,
            "Non-hazardous waste disposed",
            "kilogram",
            ["inventory indicator", "waste"],
            1000.0 * volume,
        )
    )
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = context.exchange_json(removed)
    trace["retained_biosphere_json"] = context.exchange_json(retained_land)
    trace["removed_biosphere_json"] = context.exchange_json(
        ex
        for ex in native
        if ex["type"] == "biosphere" and context.input(ex)["name"] not in _LAND_FLOWS
    )
    trace["project_terms_json"] = json.dumps(
        {
            "electricity_kwh": electricity,
            "delivered_heat_mj": heat,
            "allocated_heat_mj": dissipated_heat,
            "evaporated_fraction": evaporation,
            "source_profile": construction["source_profile"],
            "target_dry_mass_kg_per_m3": 0.0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "cubic meter",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }
