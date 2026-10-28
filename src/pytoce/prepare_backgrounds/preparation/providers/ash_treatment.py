"""Compile disposal of composition-specific incineration residues.

Each donor supplies its dry composition and existing release functions.
The municipal-source TPS chain adds Ti; the plastic-source SBR chain retains
native Ti and adds fresh APC Ca. Doka (2003), report 13-III, Tables A7/A8
supply the additional elemental release fractions.
"""

import json
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from math import fsum
from typing import Any, Literal, cast

from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _copied_native_exchange, _RecipeContext, _scaled_native_exchange


@dataclass(frozen=True, slots=True)
class _TreatmentCarbon:
    """Carbon kg in one treatment pool, retaining principal and origin custody."""

    component: str
    principal: bool
    origin: Literal["fossil", "biogenic", "atmospheric_dac"]
    carbon_kg: float


@dataclass(frozen=True, slots=True)
class _AshLoad:
    """Actual residue atoms and the separately calculated treatment service mass.

    Quantities share the caller reference. Waste elements exclude the separately
    allocated H/O; auxiliary elements include actual fresh reagents and spent
    SCR. The originating furnace defines oxide-equivalent service mass and
    quench water. The selected terminal retains its native material supplies.
    ``geological_carbon_kg`` belongs to road bulk material and contributes to
    reference mass once. It is excluded from ``waste_elements_kg``
    organic C and ``carbon``; the selected source supplies no inorganic-C fate.
    """

    treatment_mass_kg: float
    water_kg: float
    waste_elements_kg: dict[str, float]
    allocated_hydrogen_kg: float
    allocated_oxygen_kg: float
    auxiliary_elements_kg: dict[str, float]
    carbon: tuple[_TreatmentCarbon, ...]
    geological_carbon_kg: float = dataclass_field(default=0.0, kw_only=True)


# Exact ecoinvent 3.12 ash descriptors and their incoming elemental allocands.
# Native compound/indicator amounts already include their source conversions;
# multiplying by a second nitrate/phosphate/sulfate molecular factor is wrong.
_ELEMENTS = {
    "C": "carbon",
    "N": "nitrogen",
    "S": "sulfur",
    "P": "phosphorus",
    "B": "boron",
    "Cl": "chlorine",
    "Al": "aluminium",
    "Sb": "antimony",
    "As": "arsenic",
    "Ba": "barium",
    "Cd": "cadmium",
    "Ca": "calcium",
    "Cr": "chromium",
    "Co": "cobalt",
    "Cu": "copper",
    "Fe": "iron",
    "Pb": "lead",
    "Mg": "magnesium",
    "Mn": "manganese",
    "Hg": "mercury",
    "Mo": "molybdenum",
    "Ni": "nickel",
    "K": "potassium",
    "Se": "selenium",
    "Si": "silicon",
    "Na": "sodium",
    "Sn": "tin",
    "Ti": "titanium",
    "V": "vanadium",
    "Zn": "zinc",
}
_FLOW_ELEMENTS = {
    "Beryllium II": "Be",
    "Strontium II": "Sr",
    "Thallium I": "Tl",
    "Titanium ion": "Ti",
    "Aluminium III": "Al",
    "Antimony ion": "Sb",
    "Arsenic ion": "As",
    "Barium II": "Ba",
    "Boron": "B",
    "Bromine": "Br",
    "Cadmium II": "Cd",
    "Calcium II": "Ca",
    "Chloride": "Cl",
    "Chromium VI": "Cr",
    "Cobalt II": "Co",
    "Copper ion": "Cu",
    "Fluoride": "F",
    "Iron ion": "Fe",
    "Lead II": "Pb",
    "Magnesium": "Mg",
    "Manganese II": "Mn",
    "Mercury II": "Hg",
    "Molybdenum VI": "Mo",
    "Nickel II": "Ni",
    "Nitrate": "N",
    "Phosphate": "P",
    "Potassium I": "K",
    "Scandium": "Sc",
    "Selenium IV": "Se",
    "Silicon": "Si",
    "Silver I": "Ag",
    "Sodium I": "Na",
    "Sulfate": "S",
    "Tin ion": "Sn",
    "Tungsten": "W",
    "Vanadium V": "V",
    "Zinc II": "Zn",
    "DOC, Dissolved Organic Carbon": "C",
    "TOC, Total Organic Carbon": "C",
    "COD, Chemical Oxygen Demand": "C",
    "BOD5, Biological Oxygen Demand": "C",
    "Organic carbon, placed in landfill": "C",
}
_MASS_INDICATORS = {
    "Non-hazardous waste disposed",
    "Hazardous waste disposed",
    "Waste mass, total, placed in landfill",
}


# Doka (2003), report 13-III, Table A8: kg Ti released/kg incoming raw-residue Ti. Eq. 5.22
# subtracts short release from cumulative long release. Native water identities: surface
# ff36578b-f403-4656-b934-81d8d4e02dc8 and long groundwater 2a46e7f4-c66c-43e5-8a8a-d119c4098aff.
@dataclass(frozen=True, slots=True)
class _AshReleaseProfile:
    """Printed source functions for actual elements absent from the selected ash.

    Additions give flow, element, kg element/kg compound, and short and
    cumulative kg elemental release/kg incoming element.
    The selected source chains retain their declared native Ti/Ca coverage.
    """

    additions: tuple[tuple[str, str, float, float, float], ...]


_MUNICIPAL_ASH_RELEASES = {
    "bottom": _AshReleaseProfile(()),
    "residual": _AshReleaseProfile((("Titanium ion", "Ti", 1, 0.0004956, 0.2973),)),
}

# Actual UPR glass/silica introduces B/Si/Ca in both MIXED municipal and
# sludge ash. Doka2003 III Tables A7/A8 give short/cumulative elemental
# fractions; Eq5.22 assigns cumulative-minus-short to later groundwater.
# Retain native Al/Co/Ti/V releases and supplied infrastructure unchanged.
_UPR_ASH_RELEASES = {
    "bottom": _AshReleaseProfile(
        (
            ("Boron", "B", 1, 0.007835, 0.8399),
            ("Silicon", "Si", 1, 0.0000369, 0.008593),
            ("Calcium II", "Ca", 1, 0.004363, 1.0),
        ),
    ),
    "residual": _AshReleaseProfile(
        (
            ("Boron", "B", 1, 0.007835, 0.9911),
            ("Silicon", "Si", 1, 0.002249, 1.0),
            ("Calcium II", "Ca", 1, 0.0001508, 0.09051),
        ),
    ),
}
# Native N/S/Zn/Ti release factors remain as supplied. Only actual fresh APC Ca lacks a function in
# the selected residual ash; Doka2003 Table A8 supplies its short/cumulative transfer. Existing
# native coefficients retain their source values.
_PLASTIC_ASH_RELEASES = {
    "bottom": _AshReleaseProfile(()),
    "residual": _AshReleaseProfile(
        (("Calcium II", "Ca", 1, 0.0001508, 0.09051),),
    ),
}

# PA sanitary sludge uses plastic-source ash functions, including Ti and
# APC calcium. Municipal2020 and Doka (2003) supply phosphorus fractions.
# The phosphate conversion uses 31 kg P per 95 kg PO4.
_PA_ASH_RELEASES = {
    "bottom": _AshReleaseProfile(
        (("Phosphate", "P", 31 / 95, 0.000022037531153838305, 0.013222518692302985),),
    ),
    "residual": _AshReleaseProfile(
        (
            *_PLASTIC_ASH_RELEASES["residual"].additions,
            ("Phosphate", "P", 31 / 95, 0.00037317372036873187, 0.22390423222123912),
        ),
    ),
}

# The selected paint municipal and paint-sludge terminals have no inlet or discharge rows for these
# actual road minerals. Doka2003 III Tables A7/A8, pp117-118, supply elemental short/cumulative
# fractions. Sulfate contains 32/96 kg S/kg SO4. Existing residual fresh catalyst V retains its
# independently required A8 transfer as well.
_ROAD_ASH_RELEASES = {
    "bottom": _AshReleaseProfile(
        (
            ("Titanium ion", "Ti", 1, 0.000004824, 0.2999),
            ("Silicon", "Si", 1, 0.0000369, 0.008593),
            ("Calcium II", "Ca", 1, 0.004363, 1.0),
            ("Aluminium III", "Al", 1, 0.00001314, 0.8169),
            ("Potassium I", "K", 1, 0.1208, 1.0),
            ("Magnesium", "Mg", 1, 0.003799, 1.0),
            ("Sodium I", "Na", 1, 0.122, 1.0),
            ("Sulfate", "S", 32 / 96, 0.09119, 1.0),
        )
    ),
    "residual": _AshReleaseProfile(
        (
            ("Titanium ion", "Ti", 1, 0.0004956, 0.2973),
            ("Silicon", "Si", 1, 0.002249, 1.0),
            ("Calcium II", "Ca", 1, 0.0001508, 0.09051),
            ("Aluminium III", "Al", 1, 0.0004956, 0.2973),
            ("Potassium I", "K", 1, 0.2819, 1.0),
            ("Magnesium", "Mg", 1, 0.0001897, 0.1138),
            ("Sodium I", "Na", 1, 0.3753, 1.0),
            ("Sulfate", "S", 32 / 96, 0.1073, 1.0),
            ("Vanadium V", "V", 1, 0.002454, 0.7711),
        )
    ),
}


@dataclass(frozen=True, slots=True)
class AshSource:
    """Original ash calibration data.

    Dry fraction is kg dry/kg treatment; contents are kg element/kg dry.
    Emissions preserve original row order as name, compartment and native flow
    amount per positive kg treatment. Source identities and uncertainty remain
    in the admitted native records at the binding owner.
    """

    dry_fraction: float
    element_contents: dict[str, float]
    emissions: tuple[tuple[str, tuple[str, ...], float], ...]


@dataclass(frozen=True, slots=True)
class AshNativeDischarge:
    """One original row's actual discharge and amount-preserving source scale.

    Amount is in the native flow unit per caller's actual load. The multiplier scales a source
    coefficient per kg treatment to the actual treatment amount.
    Keeping the row index and multiplier preserves distinct native uncertainty
    distributions, including zero native coefficients, at compilation.
    """

    source_index: int
    amount: float
    multiplier: float


@dataclass(frozen=True, slots=True)
class AshReceiving:
    """Ash discharges and elemental kg retained under the selected release profile.

    Added emissions contain name, compartment and compound kg. Retained
    elements are pools, separate from emissions.
    """

    native: tuple[AshNativeDischarge, ...]
    additions: tuple[tuple[str, tuple[str, ...], float], ...]
    retained_profile_elements_kg: dict[str, float]


def calculate_ash_receiving(
    load: _AshLoad, source: AshSource, *, release_profile: _AshReleaseProfile
) -> AshReceiving:
    """Apply the selected ash release functions to the caller's actual atoms.

    Native element releases scale by actual kg element divided by original
    kg element/kg treatment. Mass indicators use treatment service kg. Native
    short and later coefficients retain their source precision independently.
    Explicit additions use elemental fractions and convert once to compound kg.
    DOC/TOC/COD/BOD remain overlapping source indicators, not additive C exits.
    """
    actual = dict(load.waste_elements_kg)
    for element, mass in load.auxiliary_elements_kg.items():
        actual[element] = actual.get(element, 0.0) + mass
    native: list[AshNativeDischarge] = []
    for index, (name, categories, amount) in enumerate(source.emissions):
        if name in _MASS_INDICATORS:
            multiplier = load.treatment_mass_kg
        else:
            element = _FLOW_ELEMENTS[name]
            # The admitted actual formulation has no unspecified elements.
            # A source function for an absent target element has no discharge.
            if element not in actual or actual[element] == 0:
                continue
            multiplier = actual[element] / (source.dry_fraction * source.element_contents[element])
        native.append(AshNativeDischarge(index, amount * multiplier, multiplier))
    additions: list[tuple[str, tuple[str, ...], float]] = []
    retained: dict[str, float] = {}
    for name, element, element_fraction, short, cumulative in release_profile.additions:
        # Road's bulk-only minerals enter bottom ash, whereas its sanitary
        # sludge can carry the same elements into both selected terminals.
        mass = actual.get(element, 0.0)
        for categories, fraction in (
            (("water", "surface water"), short),
            (("water", "ground-, long-term"), cumulative - short),
        ):
            additions.append((name, categories, mass * fraction / element_fraction))
        # Eq.5.22 uses cumulative release through the source horizon. Keep the unreleased element in
        # ash.
        retained[element] = mass * (1 - cumulative)
    return AshReceiving(tuple(native), tuple(additions), retained)


def _ash_reference(
    exchange: Any,
    key: tuple[str, str],
    product: str,
    load: _AshLoad,
) -> dict[str, object]:
    """Replace donor composition while preserving signed reference and uncertainty.

    Properties are per kg treatment, irrespective of nonunit native production.
    Reconstructed dry mass includes separately allocated H/O. Carbon components
    describe the same material already included in elemental masses.
    """
    actual = dict(load.waste_elements_kg)
    for element, mass in load.auxiliary_elements_kg.items():
        actual[element] = actual.get(element, 0.0) + mass
    dry = (
        fsum(actual.values())
        + load.allocated_hydrogen_kg
        + load.allocated_oxygen_kg
        + load.geological_carbon_kg
    )
    basis = load.treatment_mass_kg
    properties: dict[str, object] = {
        "wet mass": {"amount": (dry + load.water_kg) / basis, "unit": "kg"},
        "dry mass": {"amount": dry / basis, "unit": "kg"},
        "water in wet mass": {"amount": load.water_kg / basis, "unit": "kg"},
        "carbon allocation": {"amount": actual["C"] / basis, "unit": "kg"},
    }
    properties.update(
        {
            f"{_ELEMENTS[element]} content": {"amount": mass / dry, "unit": "dimensionless"}
            for element, mass in actual.items()
            if element not in {"H", "O"}
        }
    )
    # Furnace H/O and auxiliary H/O remain separate in the trace.
    for origin in ("fossil", "non-fossil"):
        carbon = fsum(
            item.carbon_kg
            for item in load.carbon
            if (item.origin == "fossil") == (origin == "fossil")
        )
        properties[f"carbon content, {origin}"] = {
            "amount": carbon / dry,
            "unit": "dimensionless",
        }
    copied = _copied_native_exchange(exchange)
    for field in ("flow", "activity", "production volume", "classifications"):
        copied.pop(field, None)
    copied.update(
        input=key,
        name=product,
        properties=properties,
        comment=(
            "Incineration ash: actual waste/APC/SCR elements and carried water. Dry mass "
            "contains separately traced nominal H/O. Treatment mass is an oxide-equivalent "
            "service allocand plus any actual bulk material. Geological C remains a separate "
            "reference outside organic-C properties and releases. "
            "Native properties remain in provenance."
        ),
    )
    return copied


def _ash_dataset_record(
    service: dict[str, object], exchanges: list[dict[str, object]]
) -> dict[str, object]:
    """Return the existing package service record for one ash market or terminal."""
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": str(service["provider_service_id"]),
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": exchanges,
    }


def _ash_market_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    load: _AshLoad,
    database_name: str,
    *,
    original_physical_reference: bool = False,
) -> dict[str, object]:
    """Retain the complete ash haul and replace its municipal terminal once.

    RoW bottom/residual markets each specify a 77 km lorry transport reference.
    Copy their dated transport rows, metadata, and uncertainty once. The target terminal uses its
    own negative production basis.
    Constructed CH wastewater keeps its original terminal-to-market ratio;
    dated production and complete haulage still belong to this market.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    selector = cast(dict[str, object], construction["source"])
    child = str(cast(dict[str, object], construction["treatment"])["service_id"])
    _service_trace(context, service)
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="ash_collection",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    production = next(ex for ex in native if ex["type"] == "production")
    reference_terms: dict[str, object] = {}
    rows = native
    if original_physical_reference:
        original = context.source.activity(selector, year=2025)
        reference = context.exchanges(original)
        p0 = float(next(ex["amount"] for ex in reference if ex["type"] == "production"))
        # Original kg terminal/kg collection is a fixed CH handoff. Rescale to dated collection
        # production.
        original_children = [
            ex
            for ex in reference
            if ex["type"] == "technosphere"
            and context.input(ex)["reference product"] == selector["reference_product"]
        ]
        rows = tuple(
            ex
            for ex in native
            if ex["type"] == "production"
            or context.input(ex)["reference product"] != selector["reference_product"]
        ) + tuple(
            _scaled_native_exchange(ex, float(production["amount"]) / p0)
            for ex in original_children
        )
        reference_terms = {
            "physical_reference_source": list(original.key),
            "terminal_amount_per_collection_unit": (
                fsum(float(ex["amount"]) for ex in original_children) / p0
            ),
        }
    generated = [
        _ash_reference(
            production,
            (database_name, provider_service_code(service_id)),
            str(service["reference_product"]),
            load,
        )
    ]
    retained: list[Any] = []
    for ex in rows:
        if ex["type"] == "production":
            continue
        supplier = context.input(ex)
        if supplier["reference product"] == selector["reference_product"]:
            generated.append(
                _ash_reference(
                    ex,
                    (database_name, provider_service_code(child)),
                    str(service["reference_product"]),
                    load,
                )
            )
        else:
            generated.append(_copied_native_exchange(ex))
            retained.append(ex)
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = context.exchange_json(
        ex
        for ex in native
        if ex["type"] == "technosphere"
        and context.input(ex)["reference product"] == selector["reference_product"]
    )
    trace["retained_biosphere_json"] = "[]"
    trace["removed_biosphere_json"] = "[]"
    trace["project_terms_json"] = json.dumps(
        {
            "ash_load_per_caller_reference": asdict(load),
            "treatment_service_id": child,
            **reference_terms,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _ash_dataset_record(service, generated)


def _ash_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    load: _AshLoad,
    database_name: str,
    *,
    release_profile: _AshReleaseProfile,
) -> dict[str, object]:
    """Apply complete source ash treatment to the calculated incoming ash payload.

    Source biosphere releases per kg native reference are normalized by its
    independently specified dry mass and elemental mass fractions. Apply actual
    waste plus auxiliary elements once, then convert from the caller's load
    to this provider's signed, possibly nonunit kg production. Dated cement,
    support, capital and hydrated-cement treatment remain linked unchanged.

    DOC/TOC/COD/BOD are overlapping source indicators. Organic carbon placed retains its source
    coefficient. The target carbon tuple records the attributed carbon origins.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    selector = cast(dict[str, object], construction["source"])
    _service_trace(context, service)
    original = context.source.activity(selector, year=2025)
    original_rows = context.exchanges(original)
    source_p = -_self_production_amount(original, original_rows, method="incineration ash source")
    original_production = next(ex for ex in original_rows if ex["type"] == "production")
    properties = cast(dict[str, dict[str, object]], original_production["properties"])
    source_dry = float(cast(float, properties["dry mass"]["amount"]))
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="ash_terminal",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    p = _self_production_amount(activity, native, method="incineration ash treatment")
    production = next(ex for ex in native if ex["type"] == "production")
    generated = [
        _ash_reference(
            production,
            (database_name, provider_service_code(service_id)),
            str(service["reference_product"]),
            load,
        )
    ]
    retained = [ex for ex in native if ex["type"] == "technosphere"]
    generated.extend(_copied_native_exchange(ex) for ex in retained)
    # Keep the selected terminal's cement, hydrated-waste treatment and other native supplies
    # unchanged. The actual-ash adaptation changes its incoming composition and releases.
    source_emissions = tuple(ex for ex in original_rows if ex["type"] == "biosphere")
    source = AshSource(
        dry_fraction=source_dry,
        element_contents={
            element: float(cast(float, properties[f"{name} content"]["amount"]))
            for element, name in _ELEMENTS.items()
            if f"{name} content" in properties
        },
        emissions=tuple(
            (
                str(context.input(ex)["name"]),
                tuple(context.input(ex)["categories"]),
                float(ex["amount"]) / source_p,
            )
            for ex in source_emissions
        ),
    )
    receiving = calculate_ash_receiving(load, source, release_profile=release_profile)
    for discharge in receiving.native:
        # Source rows carry their native production magnitude. The pure owner
        # supplies actual-load scales; only this binder normalizes to the
        # provider's signed treatment reference and preserves row uncertainty.
        generated.append(
            _scaled_native_exchange(
                source_emissions[discharge.source_index],
                discharge.multiplier * -p / source_p / load.treatment_mass_kg,
            )
        )
    for name, categories, actual_amount in receiving.additions:
        flow = context.biosphere.flow(
            {"name": name, "unit": "kilogram", "categories": list(categories)}
        )
        amount = actual_amount * -p / load.treatment_mass_kg
        generated.append(
            {
                "input": cast(tuple[str, str], flow.key),
                "name": name,
                "unit": "kilogram",
                "type": "biosphere",
                "amount": amount,
                "loc": amount,
                "uncertainty type": 0,
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
            "ash_fraction": construction["ash_fraction"],
            "ash_load_per_caller_reference": asdict(load),
            "source_dry_kg_per_reference_kg": source_dry,
            "source_properties": properties,
            "source_negative_reference_kg": -source_p,
            "target_negative_reference_kg": p,
            "terminal_supply_boundary": "unchanged native inventories",
            "explicit_profile_retained_elements_kg": receiving.retained_profile_elements_kg,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _ash_dataset_record(service, generated)
