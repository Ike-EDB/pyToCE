"""Admit the sanitary-landfill receiving chain for PA fiber residues.

The RoW plastic-waste site supplies its hydraulic reference. RoW water/soil
and GLO sludge/ash inventories provide original calibration and dated
operations. Material-balance calculations supply PA principal, finish and water;
the parent's gas section owns combustion factors and fuel calibration.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from .ash_treatment import _ELEMENTS as _ASH_ELEMENTS
from .ash_treatment import _FLOW_ELEMENTS as _ASH_FLOW_ELEMENTS
from .ash_treatment import _MASS_INDICATORS
from .landfill_phases import _LandfillSite
from .leachate_load import _ROW_WASTEWATER_PROFILE
from .leachate_treatment import _CARBON_INDICATORS, _RAW_ELEMENTS
from .native_recipe import _self_production_amount
from .pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingAshReference,
    PaReceivingFlow,
    PaReceivingLandfarmingReference,
    PaReceivingOperation,
    _activity,
    _activity_from_payload,
    _ash_reference_from_payload,
    _flow,
    _flow_from_payload,
    _landfarming_reference_from_payload,
    _operation_from_payload,
    _receiving_row,
)
from .recipe_context import _RecipeContext
from .sludge_incineration import _ELEMENT_NAMES
from .sludge_incineration_load import _CARBON_INDICATORS as _SLUDGE_CARBON_INDICATORS
from .sludge_incineration_load import _FLOW_ELEMENTS, _PROCESS_FLOWS, _SUPPLY
from .sludge_landfarming import _FLOW_ELEMENTS as _SOIL_FLOW_ELEMENTS
from .water_treatment import _LAND_FLOWS, _water_evaporation

# These services select eleven native processes, supplying fixed source selections and complete
# receiving functions for PA.
_SERVICES = {
    role: f"sbr_filled_fossil_{suffix}"
    for role, suffix in (
        ("landfill", "landfill"),
        ("collection", "leachate_collection"),
        ("wastewater", "wastewater_treatment"),
        ("sludge_incineration_market", "sludge_incineration_market"),
        ("sludge_landfarming_market", "sludge_landfarming_market"),
        ("sludge_incineration", "sludge_incineration"),
        ("sludge_landfarming", "sludge_landfarming"),
        ("bottom_ash_market", "bottom_ash_market"),
        ("residual_ash_market", "residual_ash_market"),
        ("bottom_ash_treatment", "bottom_ash_treatment"),
        ("residual_ash_treatment", "residual_ash_treatment"),
    )
}
_OWNER = "pa_actual_landfill_receiving"
_SURFACE = ("water", "surface water")
_LONG = ("water", "ground-, long-term")
_RURAL = ("air", "non-urban air or from high stacks")
_URBAN = ("air", "urban air close to ground")
_GAS_CARBON = {
    f"{species}, {origin}"
    for species in ("Carbon dioxide", "Carbon monoxide", "Methane")
    for origin in ("fossil", "non-fossil")
}
_SLUDGE_FLOWS = (
    _PROCESS_FLOWS
    | _GAS_CARBON
    | set(_SLUDGE_CARBON_INDICATORS)
    | {name for name, element in _FLOW_ELEMENTS.items() if element in {"N", "S", "P", "Na", "Fe"}}
    | {
        "Ammonia",
        "Nitrogen oxides",
        "Cyanide",
        "Dinitrogen monoxide",
        "Water",
        "Heat, waste",
        "Oxygen",
    }
)


@dataclass(frozen=True, slots=True)
class PaLandfillSludgeReference:
    """Original GLO sludge calibration per wet kg, including donor causal elements.

    Elements are kg element/kg wet sludge, using only specified original dry
    properties. Source metals contribute to reagent denominators even though
    they are absent from PA. Actual target transfers and reagent chemistry stay
    at the sludge calculation owner; missing donor phosphorus is not target zero.
    """

    operation: PaReceivingOperation
    elements_kg: dict[str, float]


@dataclass(frozen=True, slots=True)
class PaLandfillReceivingReference:
    """Shared original hydraulic, sludge, soil and ash calibration for PA landfill.

    Leachate volume is m3/kg wet waste from the selected original site's actual
    negative leachate and waste rows. Evaporation is m3/m3 from the original
    plant's climate. The site's original MAT/MAP drive waste degradation;
    they are distinct from the wastewater plant's evaporation climate.
    Intensive sludge and ash properties are not scaled by
    source production or replaced by dated properties removed by premise.
    """

    landfill: PaReceivingActivity
    climate: _LandfillSite
    leachate_volume_m3_per_wet_kg: float
    wastewater: PaReceivingActivity
    evaporation_fraction: float
    sludge: PaLandfillSludgeReference
    landfarming: PaReceivingLandfarmingReference
    bottom_ash: PaReceivingAshReference
    residual_ash: PaReceivingAshReference


@dataclass(frozen=True, slots=True)
class PaLandfillReceivingSources:
    """Eleven dated functions and their additional complete receiving endpoints.

    Rows preserve native units, signs and portfolios per absolute production
    unit. Internal child rows identify the separately calculated functions;
    receiving must replace those demands, not purchase both whole chains.
    Ordinary-background and member identities belong to the containing record.
    Catalyst rows retain native suppliers; gas has its separate source section.
    """

    landfill: PaReceivingOperation
    collection: PaReceivingOperation
    wastewater: PaReceivingOperation
    sludge_incineration_market: PaReceivingOperation
    sludge_landfarming_market: PaReceivingOperation
    sludge_incineration: PaReceivingOperation
    sludge_landfarming: PaReceivingOperation
    bottom_ash_market: PaReceivingOperation
    residual_ash_market: PaReceivingOperation
    bottom_ash_treatment: PaReceivingOperation
    residual_ash_treatment: PaReceivingOperation
    pam: PaReceivingActivity
    biosphere: tuple[PaReceivingFlow, ...]


def _services(context: _RecipeContext) -> dict[str, dict[str, object]]:
    """Select the eleven declared RoW/GLO treatment functions."""
    declared = {
        str(service["provider_service_id"]): service
        for service in cast(list[dict[str, object]], context.catalogue["services"])
    }
    return {role: declared[service_id] for role, service_id in _SERVICES.items()}


def _selector(service: dict[str, object]) -> dict[str, object]:
    """Return one selected native activity's exact product/unit/location declaration."""
    return cast(dict[str, object], cast(dict[str, object], service["construction"])["source"])


def _operation(
    context: _RecipeContext,
    activity: Any,
    *,
    products: set[str] | None,
    flows: set[str] | frozenset[str] | None,
) -> PaReceivingOperation:
    """Detach required native rows, preserving their order and signed source basis.

    None retains one complete side; an empty selection excludes that side when
    actual-load calculations replace it. The common row owner alone normalizes
    quantities and uncertainty metadata by absolute native production.
    """
    native = context.exchanges(activity)
    production = _self_production_amount(activity, native, method="actual PA landfill source")
    rows = []
    for exchange in native:
        if exchange["type"] == "production":
            continue
        node = context.input(exchange)
        if exchange["type"] == "technosphere":
            if products is not None and node["reference product"] not in products:
                continue
            endpoint: PaReceivingActivity | PaReceivingFlow = _activity(node)
        else:
            if flows is not None and node["name"] not in flows:
                continue
            endpoint = _flow(node)
        rows.append(_receiving_row(endpoint, dict(exchange), 1 / abs(production)))
    return PaReceivingOperation(_activity(activity), production, tuple(rows))


def _properties(context: _RecipeContext, activity: Any) -> dict[str, dict[str, object]]:
    """Read original per-product properties."""
    production = next(row for row in context.exchanges(activity) if row["type"] == "production")
    return cast(dict[str, dict[str, object]], production["properties"])


def _ash_reference(
    context: _RecipeContext, service: dict[str, object], *, residual: bool
) -> PaReceivingAshReference:
    """Read ash release coefficients for the elements in PA residues.

    PA sludge contains C/N/P/S/Na/Fe; fresh APC and spent SCR add Ca/Ti/V to
    residual ash. Separately specified equations supply P and Ca release.
    """
    source = context.source.activity(_selector(service), year=2025)
    elements = {"C", "N", "P", "S", "Na", "Fe"}
    if residual:
        elements.update(("Ca", "Ti", "V"))
    flows = {name for name, element in _ASH_FLOW_ELEMENTS.items() if element in elements}
    operation = _operation(context, source, products=set(), flows=flows | _MASS_INDICATORS)
    properties = _properties(context, source)
    required = {
        _ASH_FLOW_ELEMENTS[row.endpoint.name]
        for row in operation.rows
        if isinstance(row.endpoint, PaReceivingFlow) and row.endpoint.name not in _MASS_INDICATORS
    }
    return PaReceivingAshReference(
        operation,
        float(cast(float, properties["dry mass"]["amount"])),
        {
            element: float(cast(float, properties[f"{_ASH_ELEMENTS[element]} content"]["amount"]))
            for element in sorted(required)
        },
    )


def admit_pa_landfill_receiving_reference(context: _RecipeContext) -> PaLandfillReceivingReference:
    """Admit original site hydraulics and complete selected thermal/soil/ash references."""
    services = _services(context)
    site = context.source.activity(_selector(services["landfill"]), year=2025)
    climate = {
        label: float(value)
        for label, value in re.findall(r"\((MAT|MAP)\) of ([0-9.]+)", str(site["comment"]))
    }
    site_rows = context.exchanges(site)
    production = _self_production_amount(site, site_rows, method="PA sanitary landfill source")
    liquid = next(
        row
        for row in site_rows
        if row["type"] == "technosphere"
        and context.input(row)["reference product"]
        == _selector(services["wastewater"])["reference_product"]
    )
    wastewater = context.source.activity(_selector(services["wastewater"]), year=2025)
    sludge = context.source.activity(_selector(services["sludge_incineration"]), year=2025)
    properties = _properties(context, sludge)
    dry = float(cast(float, properties["dry mass"]["amount"]))
    sludge_reference = PaLandfillSludgeReference(
        _operation(
            context,
            sludge,
            products={product for product, _ in _SUPPLY.values()},
            flows=_SLUDGE_FLOWS,
        ),
        {
            element: dry * float(cast(float, properties[f"{name} content"]["amount"]))
            for element, name in _ELEMENT_NAMES.items()
            if f"{name} content" in properties
        },
    )
    soil = context.source.activity(_selector(services["sludge_landfarming"]), year=2025)
    soil_properties = _properties(context, soil)
    soil_flows = {
        name
        for name, (element, _) in _SOIL_FLOW_ELEMENTS.items()
        if element in {"N", "S", "Na", "Fe"}
    }
    soil_operation = _operation(
        context,
        soil,
        products=set(),
        flows=soil_flows
        | {
            "Carbon dioxide, fossil",
            "Carbon dioxide, non-fossil",
            "Water",
            "Heat, waste",
            "Non-hazardous waste disposed",
        },
    )
    soil_required = {"C": "carbon"}
    for row in soil_operation.rows:
        if isinstance(row.endpoint, PaReceivingFlow) and row.endpoint.name in _SOIL_FLOW_ELEMENTS:
            element, name = _SOIL_FLOW_ELEMENTS[row.endpoint.name]
            soil_required[element] = name
    return PaLandfillReceivingReference(
        _activity(site),
        # Selected RoW sanitary-site metadata supply degrees C and mm/year.
        # Landfill2020's managed anaerobic site has methane correction 1.
        _LandfillSite(climate["MAT"], climate["MAP"], 1.0),
        # The source has negative m3 liquid and negative kg waste production. Their ratio is
        # positive m3/kg wet waste; incoming moisture is already included in this placed-waste
        # reference.
        float(liquid["amount"]) / production,
        _activity(wastewater),
        _water_evaporation(str(wastewater["comment"]), _ROW_WASTEWATER_PROFILE.hydraulic),
        sludge_reference,
        PaReceivingLandfarmingReference(
            soil_operation,
            float(cast(float, soil_properties["dry mass"]["amount"])),
            float(cast(float, soil_properties["water in wet mass"]["amount"])),
            {
                element: float(cast(float, soil_properties[f"{name} content"]["amount"]))
                for element, name in soil_required.items()
            },
        ),
        _ash_reference(context, services["bottom_ash_treatment"], residual=False),
        _ash_reference(context, services["residual_ash_treatment"], residual=True),
    )


def admit_pa_landfill_receiving_sources(
    context: _RecipeContext, reference: PaLandfillReceivingReference
) -> PaLandfillReceivingSources:
    """Admit eleven dated functions with their original catalyst suppliers."""
    services = _services(context)
    operations = {}
    reheat = (
        _SUPPLY["reheat"][0]
        if context.ordinary.database_name == reference.wastewater.key[0]
        else "heat, district or industrial"
    )
    for role, service in services.items():
        activity, _ = context.activity(
            _selector(service),
            source_role="native_ecoinvent_provider",
            service_id=_OWNER,
            component_id=role,
            semantic_class="service_normalization",
        )
        products = None
        flows: set[str] | frozenset[str] | None = None
        if role == "landfill":
            flows = set()
        elif role == "collection":
            flows = (
                {
                    name
                    for name, element in _RAW_ELEMENTS.items()
                    if element in {"H", "O", "N", "P", "S", "Na"}
                }
                | set(_CARBON_INDICATORS)
                | {"Water"}
            )
        elif role == "wastewater":
            flows = _LAND_FLOWS
        elif role == "sludge_incineration":
            products = {
                reheat if duty == "reheat" else product for duty, (product, _) in _SUPPLY.items()
            }
            flows = set()
        elif role in {"sludge_landfarming", "bottom_ash_treatment", "residual_ash_treatment"}:
            flows = set()
        operations[role] = _operation(context, activity, products=products, flows=flows)
    thermal = cast(dict[str, object], services["sludge_incineration"]["construction"])
    pam, _ = context.activity(
        cast(dict[str, object], thermal["flocculant"]),
        source_role="native_ecoinvent_provider",
        service_id=_OWNER,
        component_id="pam",
        semantic_class="service_normalization",
    )
    biosphere = _biosphere(context)
    return PaLandfillReceivingSources(
        *(operations[role] for role in _SERVICES),
        pam=_activity(pam),
        biosphere=biosphere,
    )


def _biosphere(context: _RecipeContext) -> tuple[PaReceivingFlow, ...]:
    """Bind the finite PA site, water, gas-origin and added-element native endpoints.

    Actual PA phosphorus and sodium follow the supplied material composition. Binding their existing
    flow identities is separate from admitting their release equations. Gas
    combustion factors stay in the parent's gas record.
    """
    descriptors: set[tuple[str, str, tuple[str, ...]]] = {
        (name, "kilogram", _SURFACE)
        for name in (
            # The raw PA leachate calculation assigns elemental P under _RAW_ELEMENTS. Treated water
            # and ash use phosphate.
            "Phosphorus",
            "Sodium I",
            "Chloride",
            "Iron ion",
            "Calcium II",
            "Sulfate",
            "Nitrate",
            "Phosphate",
            *_CARBON_INDICATORS,
        )
    }
    descriptors.update(
        (name, "kilogram", _LONG)
        for name in (
            *_CARBON_INDICATORS,
            "Ammonium",
            "Nitrogen, organic bound",
            "Nitrite",
            "Nitrate",
            "Sulfate",
            "Hydrogen sulfide",
            "Phosphate",
            "Sodium I",
            "Calcium II",
        )
    )
    descriptors.update(
        (name, "kilogram", ("inventory indicator", "waste"))
        for name in (
            "Organic carbon, placed in landfill",
            "Waste mass, total, placed in landfill",
            "Non-hazardous waste disposed",
        )
    )
    descriptors.update(
        (name, "kilogram", categories)
        for categories, carbon_species in (
            (_URBAN, _GAS_CARBON),
            (_RURAL, {"Carbon dioxide, fossil", "Carbon dioxide, non-fossil"}),
        )
        for name in (*carbon_species, "Dinitrogen monoxide")
    )
    descriptors.update(
        (
            ("Phosphorus", "kilogram", _URBAN),
            ("Sodium I", "kilogram", _RURAL),
            ("Oxygen", "kilogram", ("natural resource", "in air")),
            ("Phosphate", "kilogram", ("water", "ground-")),
        )
    )
    descriptors.update(
        (name, unit, categories)
        for name, unit in (("Water", "cubic meter"), ("Heat, waste", "megajoule"))
        for categories in (_SURFACE, _URBAN, _RURAL)
    )
    descriptors.add(("Heat, waste", "megajoule", ("soil", "industrial")))
    for name in (
        "Occupation, construction site",
        "Occupation, dump site",
        "Occupation, shrub land, sclerophyllous",
        "Occupation, traffic area, road network",
        "Transformation, from dump site, sanitary landfill",
        "Transformation, from pasture, man made",
        "Transformation, from shrub land, sclerophyllous",
        "Transformation, to dump site, sanitary landfill",
        "Transformation, to forest, unspecified",
        "Transformation, to shrub land, sclerophyllous",
        "Transformation, to traffic area, road network",
    ):
        descriptors.add(
            (
                name,
                "square meter-year" if name.startswith("Occupation") else "square meter",
                ("natural resource", "land"),
            )
        )
    return tuple(
        _flow(context.biosphere.flow({"name": name, "unit": unit, "categories": list(categories)}))
        for name, unit, categories in sorted(descriptors)
    )


def landfill_reference_from_payload(payload: Mapping[str, object]) -> PaLandfillReceivingReference:
    """Restore trusted shared landfill calibration."""
    sludge = cast(Mapping[str, object], payload["sludge"])
    climate = cast(Mapping[str, float], payload["climate"])
    return PaLandfillReceivingReference(
        _activity_from_payload(cast(Mapping[str, object], payload["landfill"])),
        _LandfillSite(**climate),
        float(cast(float, payload["leachate_volume_m3_per_wet_kg"])),
        _activity_from_payload(cast(Mapping[str, object], payload["wastewater"])),
        float(cast(float, payload["evaporation_fraction"])),
        PaLandfillSludgeReference(
            _operation_from_payload(cast(Mapping[str, object], sludge["operation"])),
            dict(cast(Mapping[str, float], sludge["elements_kg"])),
        ),
        _landfarming_reference_from_payload(cast(Mapping[str, object], payload["landfarming"])),
        _ash_reference_from_payload(cast(Mapping[str, object], payload["bottom_ash"])),
        _ash_reference_from_payload(cast(Mapping[str, object], payload["residual_ash"])),
    )


def landfill_sources_from_payload(payload: Mapping[str, object]) -> PaLandfillReceivingSources:
    """Restore complete dated functions and portable endpoint identities from the parent."""
    return PaLandfillReceivingSources(
        *(_operation_from_payload(cast(Mapping[str, object], payload[role])) for role in _SERVICES),
        pam=_activity_from_payload(cast(Mapping[str, object], payload["pam"])),
        biosphere=tuple(
            _flow_from_payload(row)
            for row in cast(list[Mapping[str, object]], payload["biosphere"])
        ),
    )
