"""Admit complete Swiss water receiving sources for actual PA loads.

Original wastewater climate and sludge/soil/ash calibrations remain shared.
Dated records retain actual collection, treatment and terminal supplier rows.
Gas operations have their separate prepared owner. These records supply
the existing wastewater and Swiss sludge calculations.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from .ash_treatment import _ELEMENTS as _ASH_ELEMENTS
from .ash_treatment import _FLOW_ELEMENTS as _ASH_FLOW_ELEMENTS
from .ash_treatment import _MASS_INDICATORS
from .leachate_load import _CH_WASTEWATER_PROFILE
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
from .sludge_landfarming import _FLOW_ELEMENTS as _SOIL_FLOW_ELEMENTS
from .swiss_sludge_load import (
    CARBON_INDICATORS as _SLUDGE_CARBON_INDICATORS,
)
from .swiss_sludge_load import (
    ELEMENT_NAMES,
    FLOW_ELEMENTS,
    PROCESS_FLOWS,
    SUPPLIES,
    SwissSludgeSource,
)
from .water_treatment import _LAND_FLOWS, _water_evaporation

# Existing PA phosphate-water declarations select the complete Swiss
# collection, plant, sludge, FLUWA and ash functions. Each material supplies its
# own load.
_SERVICES = {
    "collection": "pa_phosphate_water_collection_ch",
    "wastewater": "pa_phosphate_water_treatment_ch",
    "sludge_incineration_market": "pa_phosphate_sludge_incineration_market_ch",
    "sludge_landfarming_market": "pa_phosphate_sludge_landfarming_market_ch",
    "sludge_incineration": "pa_phosphate_sludge_incineration_ch",
    "sludge_landfarming": "pa_phosphate_sludge_landfarming_ch",
    "bottom_ash_market": "pa_phosphate_bottom_ash_market_ch",
    "residual_ash_market": "pa_phosphate_residual_ash_market_ch",
    "bottom_ash_treatment": "pa_phosphate_bottom_ash_treatment_ch",
    "residual_ash_treatment": "pa_phosphate_residual_ash_treatment_ch",
}
_OWNER = "pa_actual_water_receiving"
_SURFACE = ("water", "surface water")
_GAS_CARBON = {
    f"{species}, {origin}"
    for species in ("Carbon dioxide", "Carbon monoxide", "Methane")
    for origin in ("fossil", "non-fossil")
}
_SLUDGE_FLOWS = (
    PROCESS_FLOWS
    | set(FLOW_ELEMENTS)
    | set(_SLUDGE_CARBON_INDICATORS)
    | _GAS_CARBON
    | {
        "Ammonia",
        "Nitrogen oxides",
        "Cyanide",
        "Dinitrogen monoxide",
        "Non-hazardous waste disposed",
        "Water",
        "Heat, waste",
    }
)


@dataclass(frozen=True, slots=True)
class PaWaterSludgeReference:
    """Original Swiss furnace calibration per wet source kg and known donor elements.

    Element quantities are kg element/kg wet reference product; they retain all
    specified source properties used to allocate original inputs and emissions. Rows
    contain used original input magnitudes and emission/template coefficients.
    Dated supplier shares must not replace these original magnitudes.
    """

    operation: PaReceivingOperation
    elements_kg: dict[str, float]

    def source(self) -> SwissSludgeSource:
        """Construct the existing pure Swiss calculation input from admitted rows."""
        emissions: dict[tuple[str, str, tuple[str, ...]], float] = {}
        for row in self.operation.rows:
            if isinstance(row.endpoint, PaReceivingFlow):
                key = row.endpoint.name, row.endpoint.unit, row.endpoint.categories
                emissions[key] = emissions.get(key, 0.0) + row.amount
        return SwissSludgeSource(self.elements_kg, self.operation.supplies(), emissions)


@dataclass(frozen=True, slots=True)
class PaWaterReceivingReference:
    """Shared original calibration for the complete selected Swiss water chain.

    Evaporation is m3 evaporated/m3 treated water, calculated once from original
    plant climate by its existing owner. The Swiss process profile, FLUWA fractions,
    empirical organic H/O and report-equilibrium P equations remain at their
    scientific owners rather than being copied into these source records.
    The PA caller selects ``phosphorus_equilibrium=True``; donor
    phosphate rows/P denominators are omitted. Swiss ash uses its native Ti/V
    coefficients.
    """

    wastewater: PaReceivingActivity
    evaporation_fraction: float
    sludge: PaWaterSludgeReference
    landfarming: PaReceivingLandfarmingReference
    bottom_ash: PaReceivingAshReference
    residual_ash: PaReceivingAshReference


@dataclass(frozen=True, slots=True)
class PaWaterReceivingSources:
    """Ten dated Swiss functions and complete additional endpoints in their native units.

    Each operation keeps signed production and rows already normalized once by
    its magnitude. Actual-load evaluation replaces internal treatment children;
    their native rows specify source signs and collection/haulage responsibilities,
    not additional whole sludge or ash purchases. Source gas remains separate.

    The parent record owns member, anchor and ordinary-background identity.
    Native catalyst rows and PAM retain their selected ordinary suppliers. No
    node integer or live native object survives source admission.
    """

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
    """Read the existing complete Swiss declarations from the admitted catalog."""
    declared = {
        str(service["provider_service_id"]): service
        for service in cast(list[dict[str, object]], context.catalogue["services"])
    }
    return {role: declared[service_id] for role, service_id in _SERVICES.items()}


def _selector(service: dict[str, object]) -> dict[str, object]:
    """Return one water function's already admitted exact native selector."""
    return cast(dict[str, object], cast(dict[str, object], service["construction"])["source"])


def _operation(
    context: _RecipeContext,
    activity: Any,
    *,
    products: set[str] | None,
    flows: set[str] | frozenset[str] | None,
) -> PaReceivingOperation:
    """Admit only the selected water function's required native rows and production.

    A None selection retains the complete corresponding source side; an empty
    selection omits the side replaced by another owner. Distribution normalization
    and native missing-parameter semantics belong to the common row admission.
    """
    native = context.exchanges(activity)
    production = _self_production_amount(activity, native, method="actual PA CH water source")
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
    """Read original intensive production properties."""
    production = next(row for row in context.exchanges(activity) if row["type"] == "production")
    return cast(dict[str, dict[str, object]], production["properties"])


def _ash_reference(
    context: _RecipeContext, service: dict[str, object], *, residual: bool
) -> PaReceivingAshReference:
    """Admit the selected Swiss element-release functions for actual sludge and spent SCR."""
    activity = context.source.activity(_selector(service), year=2025)
    # PA finish, caprolactam, and wastewater iron/PAM determine sludge composition. Spent SCR adds
    # Ti/V to residual ash; terminal cement remains a native supply.
    elements = {"C", "N", "S", "P", "Na", "Fe"}
    if residual:
        elements.update(("Ti", "V"))
    flows = {name for name, element in _ASH_FLOW_ELEMENTS.items() if element in elements}
    operation = _operation(context, activity, products=set(), flows=flows | _MASS_INDICATORS)
    properties = _properties(context, activity)
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


def admit_pa_water_receiving_reference(context: _RecipeContext) -> PaWaterReceivingReference:
    """Admit original climate and Swiss sludge/soil/ash calibration.

    Preserve normalized reference coefficients and intensive source properties.
    Absent properties remain unspecified.
    """
    services = _services(context)
    wastewater = context.source.activity(_selector(services["wastewater"]), year=2025)
    sludge = context.source.activity(_selector(services["sludge_incineration"]), year=2025)
    properties = _properties(context, sludge)
    dry = float(cast(float, properties["dry mass"]["amount"]))
    sludge_reference = PaWaterSludgeReference(
        _operation(
            context,
            sludge,
            products={product for product, _ in SUPPLIES.values()},
            flows=_SLUDGE_FLOWS,
        ),
        {
            element: dry * float(cast(float, properties[f"{name} content"]["amount"]))
            for element, name in ELEMENT_NAMES.items()
            if f"{name} content" in properties
        },
    )
    soil = context.source.activity(_selector(services["sludge_landfarming"]), year=2025)
    soil_properties = _properties(context, soil)
    # Select phosphorus from the report equilibrium.
    soil_elements = {"N", "S", "Fe", "Na"}
    soil_flows = {
        name for name, (element, _) in _SOIL_FLOW_ELEMENTS.items() if element in soil_elements
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
    return PaWaterReceivingReference(
        wastewater=_activity(wastewater),
        evaporation_fraction=_water_evaporation(
            str(wastewater["comment"]), _CH_WASTEWATER_PROFILE.hydraulic
        ),
        sludge=sludge_reference,
        landfarming=PaReceivingLandfarmingReference(
            soil_operation,
            float(cast(float, soil_properties["dry mass"]["amount"])),
            float(cast(float, soil_properties["water in wet mass"]["amount"])),
            {
                element: float(cast(float, soil_properties[f"{name} content"]["amount"]))
                for element, name in soil_required.items()
            },
        ),
        bottom_ash=_ash_reference(context, services["bottom_ash_treatment"], residual=False),
        residual_ash=_ash_reference(context, services["residual_ash_treatment"], residual=True),
    )


def admit_pa_water_receiving_sources(
    context: _RecipeContext, reference: PaWaterReceivingReference
) -> PaWaterReceivingSources:
    """Admit complete dated Swiss portfolios with their original catalyst suppliers.

    Resolve original or matching dated ordinary native supplies once during preparation. The shared
    reference is not copied per anchor.
    Actual PA landfarming must call its existing owner with report-equilibrium P.
    """
    services = _services(context)
    operations = {}
    reheat_product = (
        SUPPLIES["reheat"][0]
        if context.ordinary.database_name == reference.wastewater.key[0]
        else "heat, district or industrial"
    )
    raw_flows = {
        name
        for name, element in _RAW_ELEMENTS.items()
        if element in {"H", "O", "N", "P", "S", "Na", "Cl"}
    }
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
        if role == "collection":
            flows = raw_flows | set(_CARBON_INDICATORS) | {"Water"}
        elif role == "wastewater":
            flows = _LAND_FLOWS
        elif role == "sludge_incineration":
            products = {
                reheat_product if duty == "reheat" else product
                for duty, (product, _) in SUPPLIES.items()
            }
            products.add("metalliferous hydroxide sludge")
            flows = set()
        elif role in {"sludge_landfarming", "bottom_ash_treatment", "residual_ash_treatment"}:
            flows = set()
        operations[role] = _operation(context, activity, products=products, flows=flows)
    construction = cast(dict[str, object], services["sludge_incineration"]["construction"])
    pam, _ = context.activity(
        cast(dict[str, object], construction["flocculant"]),
        source_role="native_ecoinvent_provider",
        service_id=_OWNER,
        component_id="pam",
        semantic_class="service_normalization",
    )
    # Existing native endpoints for wastewater discharges, report-equilibrium soil
    # P and fresh Swiss HCl/NaOH receiving. Gas has its independent source record.
    descriptors: list[tuple[str, str, tuple[str, ...]]] = [
        (name, "kilogram", _SURFACE)
        for name in (
            "Chloride",
            "Sodium I",
            "Iron ion",
            "Sulfate",
            "Nitrate",
            "Phosphate",
            *_CARBON_INDICATORS,
        )
    ]
    descriptors.extend((("Water", "cubic meter", _SURFACE), ("Heat, waste", "megajoule", _SURFACE)))
    for compartment in ("urban air close to ground", "non-urban air or from high stacks"):
        descriptors.extend(
            (name, unit, ("air", compartment))
            for name, unit in (
                ("Carbon dioxide, fossil", "kilogram"),
                ("Carbon dioxide, non-fossil", "kilogram"),
                ("Dinitrogen monoxide", "kilogram"),
                ("Water", "cubic meter"),
                ("Heat, waste", "megajoule"),
            )
        )
    descriptors.extend(
        (
            ("Oxygen", "kilogram", ("natural resource", "in air")),
            ("Non-hazardous waste disposed", "kilogram", ("inventory indicator", "waste")),
            ("Phosphate", "kilogram", ("water", "ground-")),
        )
    )
    biosphere = tuple(
        _flow(context.biosphere.flow({"name": name, "unit": unit, "categories": list(categories)}))
        for name, unit, categories in descriptors
    )
    return PaWaterReceivingSources(
        operations["collection"],
        operations["wastewater"],
        operations["sludge_incineration_market"],
        operations["sludge_landfarming_market"],
        operations["sludge_incineration"],
        operations["sludge_landfarming"],
        operations["bottom_ash_market"],
        operations["residual_ash_market"],
        operations["bottom_ash_treatment"],
        operations["residual_ash_treatment"],
        _activity(pam),
        biosphere,
    )


def water_reference_from_payload(payload: Mapping[str, object]) -> PaWaterReceivingReference:
    """Restore the nested shared water calibration from its completed parent v1 record."""
    sludge = cast(Mapping[str, object], payload["sludge"])
    return PaWaterReceivingReference(
        _activity_from_payload(cast(Mapping[str, object], payload["wastewater"])),
        float(cast(float, payload["evaporation_fraction"])),
        PaWaterSludgeReference(
            _operation_from_payload(cast(Mapping[str, object], sludge["operation"])),
            dict(cast(Mapping[str, float], sludge["elements_kg"])),
        ),
        _landfarming_reference_from_payload(cast(Mapping[str, object], payload["landfarming"])),
        _ash_reference_from_payload(cast(Mapping[str, object], payload["bottom_ash"])),
        _ash_reference_from_payload(cast(Mapping[str, object], payload["residual_ash"])),
    )


def water_sources_from_payload(payload: Mapping[str, object]) -> PaWaterReceivingSources:
    """Restore the nested dated water operations."""
    return PaWaterReceivingSources(
        *(_operation_from_payload(cast(Mapping[str, object], payload[role])) for role in _SERVICES),
        pam=_activity_from_payload(cast(Mapping[str, object], payload["pam"])),
        biosphere=tuple(
            _flow_from_payload(row)
            for row in cast(list[Mapping[str, object]], payload["biosphere"])
        ),
    )
