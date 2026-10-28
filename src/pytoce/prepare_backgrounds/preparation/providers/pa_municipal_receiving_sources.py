"""Admit municipal functions for PA fiber incineration.

Original mixed-plastic properties and release factors calibrate the municipal
function. Dated inventories supply operating inputs. Additional ash P/Ca
functions apply to PA inputs, while native ash factors retain their precision.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from .ash_treatment import _ELEMENTS, _FLOW_ELEMENTS, _MASS_INDICATORS
from .municipal_incineration_emissions import _NATIVE_ELEMENT_FLOWS, _PROCESS_FLOWS
from .municipal_incineration_load import (
    _MUNICIPAL_ELEMENTS,
    _MUNICIPAL_SUPPLIES,
    MunicipalIncinerationSource,
)
from .native_recipe import _self_production_amount
from .pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingAshReference,
    PaReceivingFlow,
    PaReceivingOperation,
    _activity,
    _activity_from_payload,
    _ash_reference_from_payload,
    _flow,
    _flow_from_payload,
    _operation_from_payload,
    _receiving_row,
)
from .recipe_context import _RecipeContext

_SERVICES = {
    "incineration": "phb_fossil_incineration",
    "bottom_ash_market": "phb_fossil_bottom_ash_market",
    "residual_ash_market": "phb_fossil_residual_ash_market",
    "bottom_ash_treatment": "phb_fossil_bottom_ash_treatment",
    "residual_ash_treatment": "phb_fossil_residual_ash_treatment",
}
_AIR = ("air", "urban air close to ground")
_WATER = ("water", "surface water")
_LONG = ("water", "ground-, long-term")
_SOURCE_FLOWS = _PROCESS_FLOWS | set(_NATIVE_ELEMENT_FLOWS) | {"Ammonia", "Nitrogen oxides"}


@dataclass(frozen=True, slots=True)
class PaMunicipalReceivingReference:
    """Original furnace and ash calibration, independent of supplier dates.

    Furnace elements/water are kg per wet source kg. Normalized native rows
    retain original uncertainty; no duplicate stored pure calculation is needed.
    The two ash references retain their own intensive dry-element denominators.
    """

    incineration: PaReceivingOperation
    elements_kg: dict[str, float]
    water_kg: float
    bottom_ash: PaReceivingAshReference
    residual_ash: PaReceivingAshReference

    def source(self) -> MunicipalIncinerationSource:
        """Return the original quantities for the material balance calculation."""
        emissions: dict[tuple[str, str, tuple[str, ...]], float] = {}
        for row in self.incineration.rows:
            if isinstance(row.endpoint, PaReceivingFlow):
                key = row.endpoint.name, row.endpoint.unit, row.endpoint.categories
                emissions[key] = emissions.get(key, 0.0) + row.amount
        return MunicipalIncinerationSource(
            self.elements_kg, self.water_kg, self.incineration.supplies(), emissions
        )


@dataclass(frozen=True, slots=True)
class PaMunicipalReceivingSources:
    """Dated municipal operation, two ash markets/terminals and complete auxiliaries.

    All native quantities are per positive source unit, with signed production
    retained separately. Incineration stores selected causal supply portfolios;
    actual ash calculations replace its two native waste demands. Markets retain
    each haulage row; terminals retain complete support including existing
    hydrated cement. Catalyst rows retain native suppliers; the selected PAM
    supplier represents the actual flocculant.
    """

    incineration: PaReceivingOperation
    bottom_ash_market: PaReceivingOperation
    residual_ash_market: PaReceivingOperation
    bottom_ash_treatment: PaReceivingOperation
    residual_ash_treatment: PaReceivingOperation
    pam: PaReceivingActivity
    reheat_product: str
    biosphere: tuple[PaReceivingFlow, ...]


def _municipal_operation(
    context: _RecipeContext, activity: Any, *, products: set[str] | None, flows: set[str]
) -> PaReceivingOperation:
    """Detach selected rows once on the source's absolute production basis."""
    native = context.exchanges(activity)
    production = _self_production_amount(activity, native, method="PA municipal source")
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
            if node["name"] not in flows:
                continue
            endpoint = _flow(node)
        rows.append(_receiving_row(endpoint, dict(exchange), 1 / abs(production)))
    return PaReceivingOperation(_activity(activity), production, tuple(rows))


def _municipal_services(context: _RecipeContext) -> dict[str, dict[str, object]]:
    """Select the existing complete municipal source declarations from the catalog."""
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], context.catalogue["services"])
    }
    return {role: services[identifier] for role, identifier in _SERVICES.items()}


def _municipal_selector(service: dict[str, object]) -> dict[str, object]:
    """Read the selected native function; the PHB formulation is not transferred."""
    return cast(dict[str, object], cast(dict[str, object], service["construction"])["source"])


def _municipal_properties(context: _RecipeContext, activity: Any) -> dict[str, dict[str, float]]:
    """Read original intensive product properties."""
    production = next(ex for ex in context.exchanges(activity) if ex["type"] == "production")
    return cast(dict[str, dict[str, float]], production["properties"])


def _municipal_ash_reference(
    context: _RecipeContext, service: dict[str, object], *, residual: bool
) -> PaReceivingAshReference:
    """Admit native ash releases for actual PA elements and fresh residual APC/SCR."""
    activity = context.source.activity(_municipal_selector(service), year=2025)
    required = {"C", "N", "S", "Na", "P"}
    if residual:
        required.update(("Ca", "Ti", "V"))
    operation = _municipal_operation(
        context,
        activity,
        products=set(),
        flows={name for name, element in _FLOW_ELEMENTS.items() if element in required}
        | _MASS_INDICATORS,
    )
    properties = _municipal_properties(context, activity)
    present = {
        _FLOW_ELEMENTS[row.endpoint.name]
        for row in operation.rows
        if row.endpoint.name not in _MASS_INDICATORS
    }
    return PaReceivingAshReference(
        operation,
        properties["dry mass"]["amount"],
        {element: properties[f"{_ELEMENTS[element]} content"]["amount"] for element in present},
    )


def admit_pa_municipal_receiving_reference(
    context: _RecipeContext,
) -> PaMunicipalReceivingReference:
    """Admit the original municipal five-function source calibration for PA receiving."""
    services = _municipal_services(context)
    activity = context.source.activity(_municipal_selector(services["incineration"]), year=2025)
    properties = _municipal_properties(context, activity)
    names = {f"{item.property_name} content": e for e, item in _MUNICIPAL_ELEMENTS.items()}
    names.update({"hydrogen content": "H", "oxygen content": "O"})
    dry = properties["dry mass"]["amount"]
    return PaMunicipalReceivingReference(
        _municipal_operation(
            context,
            activity,
            products={product for product, _ in _MUNICIPAL_SUPPLIES.values()},
            flows=set(_SOURCE_FLOWS),
        ),
        {names[name]: dry * value["amount"] for name, value in properties.items() if name in names},
        properties["water in wet mass"]["amount"],
        _municipal_ash_reference(context, services["bottom_ash_treatment"], residual=False),
        _municipal_ash_reference(context, services["residual_ash_treatment"], residual=True),
    )


def admit_pa_municipal_receiving_sources(
    context: _RecipeContext, reference: PaMunicipalReceivingReference
) -> PaMunicipalReceivingSources:
    """Admit dated operations with their original catalyst suppliers."""
    services = _municipal_services(context)
    reheat = (
        "heat, district or industrial, natural gas"
        if context.year <= 2025
        else "heat, district or industrial"
    )
    operations: dict[str, PaReceivingOperation] = {}
    for role, service in services.items():
        activity, _ = context.activity(
            _municipal_selector(service),
            source_role="native_ecoinvent_provider",
            service_id="pa_actual_municipal_receiving",
            component_id=role,
            semantic_class="service_normalization",
        )
        products = (
            {
                reheat if duty == "reheat" else product
                for duty, (product, _) in _MUNICIPAL_SUPPLIES.items()
            }
            if role == "incineration"
            else None
        )
        operations[role] = _municipal_operation(context, activity, products=products, flows=set())
    construction = cast(dict[str, object], services["incineration"]["construction"])
    pam, _ = context.activity(
        cast(dict[str, object], construction["flocculant"]),
        source_role="native_ecoinvent_provider",
        service_id="pa_actual_municipal_receiving",
        component_id="pam",
        semantic_class="service_normalization",
    )
    descriptors = {
        (row.endpoint.name, row.endpoint.unit, row.endpoint.categories)
        for operation in (
            reference.incineration,
            reference.bottom_ash.operation,
            reference.residual_ash.operation,
        )
        for row in operation.rows
        if isinstance(row.endpoint, PaReceivingFlow)
    }
    descriptors.update(
        (f"{species}, {origin}", "kilogram", _AIR)
        for species in ("Carbon dioxide", "Carbon monoxide", "Methane")
        for origin in ("fossil", "non-fossil")
    )
    descriptors.update(
        (name, "kilogram", _WATER)
        for name in (
            "DOC, Dissolved Organic Carbon",
            "TOC, Total Organic Carbon",
            "COD, Chemical Oxygen Demand",
            "BOD5, Biological Oxygen Demand",
            "Chloride",
            "Sodium I",
            "Calcium II",
            "Phosphate",
        )
    )
    descriptors.update((name, "kilogram", _LONG) for name in ("Phosphate", "Calcium II"))
    descriptors.update(
        (name, unit, compartment)
        for name, unit in (("Water", "cubic meter"), ("Heat, waste", "megajoule"))
        for compartment in (_AIR, _WATER)
    )
    descriptors.update(
        {
            ("Phosphorus", "kilogram", _AIR),
            ("Oxygen", "kilogram", ("natural resource", "in air")),
            ("Non-hazardous waste disposed", "kilogram", ("inventory indicator", "waste")),
        }
    )
    flows = tuple(
        _flow(context.biosphere.flow({"name": name, "unit": unit, "categories": list(categories)}))
        for name, unit, categories in sorted(descriptors)
    )
    return PaMunicipalReceivingSources(
        *(operations[role] for role in _SERVICES),
        pam=_activity(pam),
        reheat_product=reheat,
        biosphere=flows,
    )


def municipal_reference_from_payload(
    payload: Mapping[str, object],
) -> PaMunicipalReceivingReference:
    """Restore the shared calibration from its completed record."""
    return PaMunicipalReceivingReference(
        _operation_from_payload(cast(Mapping[str, object], payload["incineration"])),
        dict(cast(Mapping[str, float], payload["elements_kg"])),
        float(cast(float, payload["water_kg"])),
        _ash_reference_from_payload(cast(Mapping[str, object], payload["bottom_ash"])),
        _ash_reference_from_payload(cast(Mapping[str, object], payload["residual_ash"])),
    )


def municipal_sources_from_payload(payload: Mapping[str, object]) -> PaMunicipalReceivingSources:
    """Restore five dated operations and their native endpoints."""
    return PaMunicipalReceivingSources(
        *(_operation_from_payload(cast(Mapping[str, object], payload[role])) for role in _SERVICES),
        pam=_activity_from_payload(cast(Mapping[str, object], payload["pam"])),
        reheat_product=str(payload["reheat_product"]),
        biosphere=tuple(
            _flow_from_payload(row)
            for row in cast(list[Mapping[str, object]], payload["biosphere"])
        ),
    )
