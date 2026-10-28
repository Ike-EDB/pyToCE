"""Admit native sources for actual PA fiber-residue receiving.

Original hazardous-waste incineration (HWI) and ash composition calibrate shared
residue equations. Matching general-background rows supply dated operations and native endpoints.
Original fuel heating values calibrate the shared landfill and wastewater gas
calculations; their operations retain the matching dated emissions and support.
The Swiss water section supplies collection, wastewater, sludge and ash
receiving; the sanitary section separately retains the selected RoW/GLO site
and leachate-treatment chain. These records remain usable after source preparation. Municipal PA
incineration retains its own complete
five-function chain. Open burning and the five dump climates use original
source identities and native flow bindings.
Off-site ingredient and heat-transfer-fluid treatments remain separate providers.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import cast

from ..chemical_residue_incineration import (
    CHEMICAL_RESIDUE_ASH_SOURCE,
    CHEMICAL_RESIDUE_FIXED_SOURCE,
    CHEMICAL_RESIDUE_PAM_SOURCE,
    CHEMICAL_RESIDUE_SOURCE,
    _chemical_residue_ash_source,
    _chemical_residue_source,
    _native_treatment,
    _NativeTreatment,
)
from ..chemical_residue_load import (
    _AMMONIA,
    _CARBON_INDICATORS,
    _FERRIC,
    _GAS_HEAT,
    _HCL,
    _LONG,
    _NAOH,
    _PEROXIDE,
    _SURFACE,
    _TMT,
    _URBAN,
    ChemicalResidueAshSource,
    ChemicalResidueSource,
)
from .catalogue import provider_service_code
from .pa_dump_receiving import (
    PaDumpReceivingReference,
    admit_pa_dump_receiving_reference,
    dump_reference_from_payload,
)
from .pa_landfill_receiving_sources import (
    PaLandfillReceivingReference,
    PaLandfillReceivingSources,
    admit_pa_landfill_receiving_reference,
    admit_pa_landfill_receiving_sources,
    landfill_reference_from_payload,
    landfill_sources_from_payload,
)
from .pa_municipal_receiving_sources import (
    PaMunicipalReceivingReference,
    PaMunicipalReceivingSources,
    admit_pa_municipal_receiving_reference,
    admit_pa_municipal_receiving_sources,
    municipal_reference_from_payload,
    municipal_sources_from_payload,
)
from .pa_open_burning_sources import (
    PaOpenBurningReference,
    admit_pa_open_burning_reference,
    open_burning_reference_from_payload,
)
from .pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
    PaReceivingOperation,
    PaReceivingRow,
    _activity,
    _activity_from_payload,
    _flow,
    _flow_from_payload,
    _operation_from_payload,
    _receiving_row,
)
from .pa_recycling_receiving_sources import (
    PaRecyclingReceivingReference,
    PaRecyclingReceivingSources,
    admit_pa_recycling_receiving_reference,
    admit_pa_recycling_receiving_sources,
    recycling_reference_from_payload,
    recycling_sources_from_payload,
)
from .pa_water_receiving_sources import (
    PaWaterReceivingReference,
    PaWaterReceivingSources,
    admit_pa_water_receiving_reference,
    admit_pa_water_receiving_sources,
    water_reference_from_payload,
    water_sources_from_payload,
)
from .recipe_context import _RecipeContext
from .waste_gas_load import (
    _CHP_FACTORS,
    _NMVOC_NAME,
    _PM_NAME,
    _RURAL,
    _GasOperation,
)

_OWNER = "pa_actual_integrated_maintenance"
_REFERENCE_SCHEMA = "pytoce_pa_receiving_reference_schema_v1"
_SOURCES_SCHEMA = "pytoce_pa_receiving_sources_schema_v1"


@dataclass(frozen=True, slots=True)
class PaReceivingFuelReference:
    """Original source and fuel identity with its intensive net heating value.

    Heating value is MJ per unit of the identified fuel. The record retains the original source
    heating-value property. The actual PA-derived gas mixture has its own calculated heating value.
    """

    activity: PaReceivingActivity
    fuel: PaReceivingActivity
    net_heat_mj_per_fuel_unit: float


@dataclass(frozen=True, slots=True)
class PaReceivingGasReference:
    """Original biogas-engine and natural-gas-furnace fuel calibration."""

    chp: PaReceivingFuelReference
    furnace: PaReceivingFuelReference


@dataclass(frozen=True)
class PaReceivingGasSources:
    """Dated combustion calibration for actual PA waste gas.

    CHP means combined heat and power. Combustion rows, including fuel, use the
    same positive MJ source-output basis. Their row/fuel ratios therefore retain
    the native production normalization exactly once. Fuel rows calibrate gas
    handling; the actual waste gas replaces purchased fuel.
    Native parent landfill and wastewater inventories retain their own
    operating support.
    All added elementary flows already exist in the source biosphere.
    """

    chp: PaReceivingOperation
    furnace: PaReceivingOperation
    biosphere: tuple[PaReceivingFlow, ...]

    @cached_property
    def _chp_rows(self) -> dict[str, dict[str, object]]:
        """Prepare source-only CHP templates once; event scaling copies metadata."""
        return _gas_calculation_rows(self.chp)

    @cached_property
    def _furnace_rows(self) -> dict[str, dict[str, object]]:
        """Prepare source-only furnace templates outside the serialized fields."""
        return _gas_calculation_rows(self.furnace)

    @cached_property
    def flow_index(self) -> dict[tuple[str, tuple[str, ...]], PaReceivingFlow]:
        """Index immutable native flow endpoints once for this dated source."""
        return {(flow.name, flow.categories): flow for flow in self.biosphere}

    def chp_operation(self, reference: PaReceivingGasReference) -> _GasOperation:
        """Pair dated engine rows with the original intensive biogas heating value."""
        return _gas_calculation_source(self._chp_rows, reference.chp)

    def furnace_operation(self, reference: PaReceivingGasReference) -> _GasOperation:
        """Pair dated flare-factor rows with the original natural-gas heating value."""
        return _gas_calculation_source(self._furnace_rows, reference.furnace)


@dataclass(frozen=True, slots=True)
class PaReceivingReference:
    """Original calibration shared across PA receiving anchors.

    Donor elements use kg element/kg wet HWI input. Ash dry fraction uses kg
    dry/kg received ash, and contents use kg element/kg dry ash. These are
    intensive properties, independent of native production amount. Original
    properties are retained before premise 2.4.6 extraction.
    """

    hwi: PaReceivingActivity
    donor_elements_kg: dict[str, float]
    ash: PaReceivingActivity
    ash_dry_fraction: float
    ash_element_contents: dict[str, float]
    gas: PaReceivingGasReference
    water: PaWaterReceivingReference
    landfill: PaLandfillReceivingReference
    recycling: PaRecyclingReceivingReference
    municipal: PaMunicipalReceivingReference
    open_burning: PaOpenBurningReference
    dump: PaDumpReceivingReference


@dataclass(frozen=True, slots=True)
class PaReceivingSources:
    """Dated ordinary sources and complete endpoints for PA fiber residues.

    The provider anchor is distinct from the general-background anchor: ordinary
    sources remain original ecoinvent through 2025, including early premise
    technology frontiers. Native Ti/Cr supplier rows retain their inventories;
    actual receiving scales their doses while residue chemistry uses
    Ti/V oxides. The inorganic flocculant row supplies the complete PAM dose.
    Native reheat rows document the dated source role; the integrated site's energy supplies are
    recorded by the site calculation. Ash rows retain full outgoing support.
    ``spinning_providers`` identifies the two prepared grade converters only
    as provenance for actual site water and air quantities. Their fixed inputs remain separate
    purchases.
    """

    background_scenario: str | None
    provider_anchor_year: int
    general_background_database: str
    general_background_anchor_year: int
    hwi: PaReceivingOperation
    fixed: PaReceivingOperation
    ash: PaReceivingOperation
    reheat_product: str
    pam: PaReceivingActivity
    added_biosphere: tuple[PaReceivingFlow, ...]
    gas: PaReceivingGasSources
    water: PaWaterReceivingSources
    landfill: PaLandfillReceivingSources
    recycling: PaRecyclingReceivingSources
    municipal: PaMunicipalReceivingSources
    spinning_providers: dict[str, PaReceivingActivity]

    def thermal_source(self, reference: PaReceivingReference) -> ChemicalResidueSource:
        """Provide the existing pure thermal boundary for integrated PA maintenance.

        Call ``_residue_reference`` with ``integrated_pa_site=True``. The existing site heat
        convention uses zero supporting-oil and boiler-electricity coefficients.
        """
        return ChemicalResidueSource(
            donor_elements_kg=reference.donor_elements_kg,
            waste_supplies=_thermal_supplies(self.hwi, self.reheat_product),
            waste_releases=self.hwi.releases(),
            fixed_supplies=_thermal_supplies(self.fixed, self.reheat_product),
            fixed_releases=self.fixed.releases(),
            oil_kg_per_mj=0.0,
            boiler_electricity_kwh_per_mj=0.0,
        )

    def ash_source(self, reference: PaReceivingReference) -> ChemicalResidueAshSource:
        """Pair original ash composition with this anchor's release rows."""
        return ChemicalResidueAshSource(
            dry_fraction=reference.ash_dry_fraction,
            element_contents=reference.ash_element_contents,
            releases=self.ash.releases(),
        )


def _thermal_supplies(operation: PaReceivingOperation, reheat_product: str) -> dict[str, float]:
    """Name dated industrial reheat consistently for the existing thermal equation.

    The native product identity remains in each prepared row for actual supply
    binding. The label identifies the existing heating calculation.
    """
    amounts: dict[str, float] = {}
    for product, amount in operation.supplies().items():
        label = _GAS_HEAT if product == reheat_product else product
        amounts[label] = amounts.get(label, 0.0) + amount
    return amounts


def _constructions(context: _RecipeContext) -> dict[str, dict[str, object]]:
    """Read the selected spinning and shared gas-source declarations."""
    return {
        str(service["provider_service_id"]): service
        for service in cast(list[dict[str, object]], context.catalogue["services"])
        if service["provider_service_id"]
        in {
            "sbr_filled_fossil_landfill",
            "fibres_pa6_conversion",
            "fibres_pa66_conversion",
        }
    }


def _gas_fuel_reference(
    context: _RecipeContext, selector: dict[str, object], fuel_product: str
) -> PaReceivingFuelReference:
    """Admit the original fuel identity and net heating value before premise extraction."""
    original = context.source.activity(selector, year=2025)
    fuel_row = next(
        row
        for row in context.exchanges(original)
        if row["type"] == "technosphere" and context.input(row)["reference product"] == fuel_product
    )
    return PaReceivingFuelReference(
        activity=_activity(original),
        fuel=_activity(context.input(fuel_row)),
        net_heat_mj_per_fuel_unit=float(fuel_row["properties"]["heating value, net"]["amount"]),
    )


def _gas_calculation_rows(operation: PaReceivingOperation) -> dict[str, dict[str, object]]:
    """Prepare read-only gas templates from one completed dated operation.

    These source dictionaries are shared only by pure gas calculations. Native
    uncertainty scaling creates separate event dictionaries before any mutation.
    """
    rows: dict[str, dict[str, object]] = {}
    for row in operation.rows:
        endpoint = row.endpoint
        name = (
            endpoint.reference_product
            if isinstance(endpoint, PaReceivingActivity)
            else endpoint.name
        )
        rows[name] = {
            **row.metadata,
            "amount": row.amount,
            "input": endpoint.key,
            "name": name,
            "unit": endpoint.unit,
            "type": "technosphere" if isinstance(endpoint, PaReceivingActivity) else "biosphere",
        }
    return rows


def _gas_calculation_source(
    rows: dict[str, dict[str, object]], reference: PaReceivingFuelReference
) -> _GasOperation:
    """Pair prepared dated coefficients with the supplied original fuel reference.

    Both row and fuel amounts already use the same source-output normalization.
    Reference-dependent heating values and preparation trace are calculated for each reference;
    shared source templates remain immutable.
    """
    return _GasOperation(
        rows=rows,
        fuel_amount=float(cast(float, rows[reference.fuel.reference_product]["amount"])),
        fuel_lhv=reference.net_heat_mj_per_fuel_unit,
        trace={},
    )


def admit_pa_receiving_reference(context: _RecipeContext) -> PaReceivingReference:
    """Admit original intensive properties during shared source preparation.

    The caller supplies the completed original source context. The established
    chemical-residue selectors and property readers own the source conventions;
    this record stores their original calibration once for all later anchors.
    """
    services = _constructions(context)
    hwi_selector = CHEMICAL_RESIDUE_SOURCE
    ash_selector = CHEMICAL_RESIDUE_ASH_SOURCE
    hwi = _native_treatment(context, hwi_selector, _OWNER)
    fixed = _native_treatment(context, CHEMICAL_RESIDUE_FIXED_SOURCE, _OWNER)
    ash = _native_treatment(context, ash_selector, _OWNER)
    thermal = _chemical_residue_source(
        context, hwi, fixed, hwi_selector, _GAS_HEAT, integrated_pa_site=True
    )
    terminal = _chemical_residue_ash_source(context, ash, ash_selector)
    # PA gas uses the equipment selections for filled SBR and Swiss chemical
    # wastewater, with the PA input composition.
    gas_construction = cast(
        dict[str, object], services["sbr_filled_fossil_landfill"]["construction"]
    )
    gas_selectors = cast(dict[str, dict[str, object]], gas_construction["gas_sources"])
    return PaReceivingReference(
        hwi=_activity(context.source.activity(hwi_selector, year=2025)),
        donor_elements_kg=thermal.donor_elements_kg,
        ash=_activity(context.source.activity(ash_selector, year=2025)),
        ash_dry_fraction=terminal.dry_fraction,
        ash_element_contents=terminal.element_contents,
        gas=PaReceivingGasReference(
            chp=_gas_fuel_reference(context, gas_selectors["chp"], "biogas"),
            furnace=_gas_fuel_reference(
                context, gas_selectors["furnace"], "natural gas, high pressure"
            ),
        ),
        water=admit_pa_water_receiving_reference(context),
        landfill=admit_pa_landfill_receiving_reference(context),
        recycling=admit_pa_recycling_receiving_reference(context),
        municipal=admit_pa_municipal_receiving_reference(context),
        open_burning=admit_pa_open_burning_reference(context),
        dump=admit_pa_dump_receiving_reference(context),
    )


def _operation(
    context: _RecipeContext, source: _NativeTreatment, role: str, reheat_product: str
) -> PaReceivingOperation:
    """Normalize used HWI/gas rows and complete fixed-control/ash operations."""
    hwi_products = {
        _AMMONIA,
        reheat_product,
        _HCL,
        _NAOH,
        _PEROXIDE,
        _FERRIC,
        _TMT,
        "calcium chloride",
        "quicklime, milled, packed",
        "chemical, inorganic",
        "titanium dioxide",
        "chromium",
    }
    hwi_releases = _CARBON_INDICATORS | {
        "Carbon dioxide, fossil",
        "Carbon monoxide, fossil",
        "Cyanide",
        "Ammonia",
        "Dinitrogen monoxide",
        "Nitrogen oxides",
        "Sulfur dioxide",
        "Nickel II",
        "Silicon",
        "Nitrate",
        "Phosphorus",
        "Phosphate",
        "Sulfate",
        "Hazardous waste disposed",
    }
    if role == "hwi":
        products, releases = hwi_products, hwi_releases
    elif role == "chp":
        products = {"biogas"}
        releases = set(_CHP_FACTORS.values())
    elif role == "furnace":
        products = {"natural gas, high pressure"}
        releases = {"Dinitrogen monoxide"}
    else:
        products = releases = None
    rows: list[PaReceivingRow] = []
    for native in source.rows:
        if native["type"] == "production":
            continue
        node = context.input(native)
        if native["type"] == "technosphere":
            if products is not None and node["reference product"] not in products:
                continue
            endpoint: PaReceivingActivity | PaReceivingFlow = _activity(node)
        else:
            if releases is not None and node["name"] not in releases:
                continue
            endpoint = _flow(node)
        rows.append(_receiving_row(endpoint, native, 1 / source.reference))
    return PaReceivingOperation(
        _activity(source.activity), float(source.production["amount"]), tuple(rows)
    )


def _admit_gas_sources(
    context: _RecipeContext, construction: dict[str, object]
) -> PaReceivingGasSources:
    """Admit dated PA landfill and wastewater gas factors and existing flow endpoints.

    The landfill and wastewater owners retain their native site and plant support.
    Donor combustion rows calibrate actual gas emissions. No purchased fuel or
    new characterization flow is introduced.
    """
    selectors = cast(dict[str, dict[str, object]], construction["gas_sources"])
    operations = {
        role: _native_treatment(context, selectors[role], "pa_actual_receiving_gas")
        for role in ("chp", "furnace")
    }
    # Landfill emits at the rural site; the selected Swiss water plant apportions
    # actual gas to both compartments. Carbon keeps each incoming origin through
    # the existing fossil/non-fossil CO2, CO and CH4 identities, including DAC.
    species = (
        "Carbon dioxide, fossil",
        "Carbon dioxide, non-fossil",
        "Carbon monoxide, fossil",
        "Carbon monoxide, non-fossil",
        "Methane, fossil",
        "Methane, non-fossil",
        _NMVOC_NAME,
        _PM_NAME,
        "Nitrogen oxides",
        "Dinitrogen monoxide",
        "Platinum",
        "Sulfur dioxide",
        "Ammonia",
        "Water",
    )
    flows = [
        _flow(
            context.biosphere.flow(
                {
                    "name": name,
                    "unit": "cubic meter" if name == "Water" else "kilogram",
                    "categories": ["air", compartment],
                }
            )
        )
        for name in species
        for compartment in (_RURAL, _URBAN[1])
    ]
    flows.append(
        _flow(
            context.biosphere.flow(
                {"name": "Oxygen", "unit": "kilogram", "categories": ["natural resource", "in air"]}
            )
        )
    )
    return PaReceivingGasSources(
        chp=_operation(context, operations["chp"], "chp", ""),
        furnace=_operation(context, operations["furnace"], "furnace", ""),
        biosphere=tuple(flows),
    )


def admit_pa_receiving_sources(
    context: _RecipeContext, reference: PaReceivingReference
) -> PaReceivingSources:
    """Bind actual ordinary source rows and complete support at one provider frontier.

    ``reference`` supplies the admitted shared calibration. The caller admits these sources after
    the provider frontier
    has been successfully written. Its two spinning providers are retained as
    provenance. Native operations, catalyst rows
    and PAM resolve in the selected ordinary source.
    """
    services = _constructions(context)
    original = context.ordinary.database_name == reference.hwi.key[0]
    reheat_product = _GAS_HEAT if original else "heat, district or industrial"
    hwi = _native_treatment(context, CHEMICAL_RESIDUE_SOURCE, _OWNER)
    fixed = _native_treatment(context, CHEMICAL_RESIDUE_FIXED_SOURCE, _OWNER)
    ash = _native_treatment(context, CHEMICAL_RESIDUE_ASH_SOURCE, _OWNER)
    pam, _ = context.activity(
        CHEMICAL_RESIDUE_PAM_SOURCE,
        source_role="native_ecoinvent_provider",
        service_id=_OWNER,
        component_id="complete_PAM_supply",
        semantic_class="service_normalization",
    )
    frontier_activities: dict[str, PaReceivingActivity] = {}
    for service_id in (
        "fibres_pa6_conversion",
        "fibres_pa66_conversion",
    ):
        service = services[service_id]
        frontier_activities[service_id] = PaReceivingActivity(
            key=(context.provider_database, provider_service_code(service_id)),
            name=str(service["name"]),
            reference_product=str(service["reference_product"]),
            unit=str(service["unit"]),
            location=str(service["location"]),
        )
    # Added species belong to the actual PA site and its principal/finish
    # deposits. The BREF total organic-air envelope uses the existing native
    # unspecified-air NMVOC identity (d3260d0e-8203-4cbb-a45a-6a13131a5108),
    # once; caprolactam and ester masses are components of that envelope.
    added = (
        (_NMVOC_NAME, "kilogram", ("air",)),
        ("Carbon dioxide, non-fossil", "kilogram", _URBAN),
        ("Carbon monoxide, non-fossil", "kilogram", _URBAN),
        ("Water", "cubic meter", ("air",)),
        ("Water", "cubic meter", ("water",)),
        ("Sodium I", "kilogram", _URBAN),
        ("Sodium I", "kilogram", _SURFACE),
        ("Chloride", "kilogram", _SURFACE),
        ("Nitrogen", "kilogram", ("air",)),
        ("Heat, waste", "megajoule", _URBAN),
        ("Heat, waste", "megajoule", _SURFACE),
        ("Titanium ion", "kilogram", _SURFACE),
        ("Titanium ion", "kilogram", _LONG),
    )
    return PaReceivingSources(
        background_scenario=context.background_scenario,
        provider_anchor_year=context.year,
        general_background_database=context.ordinary.database_name,
        general_background_anchor_year=2025 if original else context.year,
        hwi=_operation(context, hwi, "hwi", reheat_product),
        fixed=_operation(context, fixed, "fixed", reheat_product),
        ash=_operation(context, ash, "ash", reheat_product),
        reheat_product=reheat_product,
        pam=_activity(pam),
        added_biosphere=tuple(
            _flow(
                context.biosphere.flow({"name": name, "unit": unit, "categories": list(categories)})
            )
            for name, unit, categories in added
        ),
        gas=_admit_gas_sources(
            context,
            cast(dict[str, object], services["sbr_filled_fossil_landfill"]["construction"]),
        ),
        water=admit_pa_water_receiving_sources(context, reference.water),
        landfill=admit_pa_landfill_receiving_sources(context, reference.landfill),
        recycling=admit_pa_recycling_receiving_sources(context),
        municipal=admit_pa_municipal_receiving_sources(context, reference.municipal),
        spinning_providers={
            grade: frontier_activities[f"fibres_{grade}_conversion"] for grade in ("pa6", "pa66")
        },
    )


def reference_to_payload(reference: PaReceivingReference) -> dict[str, object]:
    """Encode the shared original calibration under its single current schema."""
    return {"schema": _REFERENCE_SCHEMA, **asdict(reference)}


def _fuel_reference_from_payload(payload: Mapping[str, object]) -> PaReceivingFuelReference:
    """Restore original fuel calibration and its exact product unit and identity."""
    return PaReceivingFuelReference(
        activity=_activity_from_payload(cast(dict[str, object], payload["activity"])),
        fuel=_activity_from_payload(cast(dict[str, object], payload["fuel"])),
        net_heat_mj_per_fuel_unit=float(cast(float, payload["net_heat_mj_per_fuel_unit"])),
    )


def reference_from_payload(payload: Mapping[str, object]) -> PaReceivingReference:
    """Read a completed package calibration."""
    gas = cast(dict[str, dict[str, object]], payload["gas"])
    return PaReceivingReference(
        hwi=_activity_from_payload(cast(dict[str, object], payload["hwi"])),
        donor_elements_kg=cast(dict[str, float], payload["donor_elements_kg"]),
        ash=_activity_from_payload(cast(dict[str, object], payload["ash"])),
        ash_dry_fraction=float(cast(float, payload["ash_dry_fraction"])),
        ash_element_contents=cast(dict[str, float], payload["ash_element_contents"]),
        gas=PaReceivingGasReference(
            chp=_fuel_reference_from_payload(gas["chp"]),
            furnace=_fuel_reference_from_payload(gas["furnace"]),
        ),
        water=water_reference_from_payload(cast(dict[str, object], payload["water"])),
        landfill=landfill_reference_from_payload(cast(dict[str, object], payload["landfill"])),
        recycling=recycling_reference_from_payload(cast(dict[str, object], payload["recycling"])),
        municipal=municipal_reference_from_payload(cast(dict[str, object], payload["municipal"])),
        open_burning=open_burning_reference_from_payload(
            cast(dict[str, object], payload["open_burning"])
        ),
        dump=dump_reference_from_payload(cast(dict[str, object], payload["dump"])),
    )


def sources_to_payload(sources: PaReceivingSources) -> dict[str, object]:
    """Encode normalized dated rows with their shared-calibration reference."""
    return {"schema": _SOURCES_SCHEMA, **asdict(sources)}


def sources_from_payload(payload: Mapping[str, object]) -> PaReceivingSources:
    """Read a completed dated product."""
    gas = cast(dict[str, object], payload["gas"])
    return PaReceivingSources(
        background_scenario=cast(str | None, payload["background_scenario"]),
        provider_anchor_year=int(cast(int, payload["provider_anchor_year"])),
        general_background_database=str(payload["general_background_database"]),
        general_background_anchor_year=int(cast(int, payload["general_background_anchor_year"])),
        hwi=_operation_from_payload(cast(dict[str, object], payload["hwi"])),
        fixed=_operation_from_payload(cast(dict[str, object], payload["fixed"])),
        ash=_operation_from_payload(cast(dict[str, object], payload["ash"])),
        reheat_product=str(payload["reheat_product"]),
        pam=_activity_from_payload(cast(dict[str, object], payload["pam"])),
        added_biosphere=tuple(
            _flow_from_payload(row)
            for row in cast(list[dict[str, object]], payload["added_biosphere"])
        ),
        gas=PaReceivingGasSources(
            chp=_operation_from_payload(cast(dict[str, object], gas["chp"])),
            furnace=_operation_from_payload(cast(dict[str, object], gas["furnace"])),
            biosphere=tuple(
                _flow_from_payload(row) for row in cast(list[dict[str, object]], gas["biosphere"])
            ),
        ),
        water=water_sources_from_payload(cast(dict[str, object], payload["water"])),
        landfill=landfill_sources_from_payload(cast(dict[str, object], payload["landfill"])),
        recycling=recycling_sources_from_payload(cast(dict[str, object], payload["recycling"])),
        municipal=municipal_sources_from_payload(cast(dict[str, object], payload["municipal"])),
        spinning_providers={
            grade: _activity_from_payload(row)
            for grade, row in cast(
                dict[str, dict[str, object]], payload["spinning_providers"]
            ).items()
        },
    )
