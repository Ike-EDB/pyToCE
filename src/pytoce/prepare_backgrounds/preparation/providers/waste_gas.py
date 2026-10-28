"""Bind complete gas receiving for plastics residues and chemical wastewater.

Thermoplastic starch (TPS), filled styrene-butadiene rubber (SBR) residues and
constructed chemical wastewater supply their own gas quantities and origins.
The pure gas owner applies selected Doka leakage, flare/CHP and oxygen/water/heat equations. This
owner binds gas emissions to native flows
with their preparation traces. Donor gas is replaced;
no recovered-energy credits or gas coproduct services are added.
"""

import json
from dataclasses import asdict, dataclass
from math import fsum
from typing import cast

from .leachate_load import _WastewaterLoad, _WastewaterProcessProfile
from .material_carbon import PrincipalCarbonOriginFractions
from .recipe_context import _RecipeContext, _scaled_native_exchange
from .waste_gas_load import (
    _CHP_FACTORS,
    _RURAL,
    _URBAN,
    _branch_emissions,
    _gas_leakage_reference,
    _GasBranch,
    _GasOperation,
    _GasPool,
    _landfill_gas_reference,
    _LandfillGasLoad,
    _wastewater_gas_reference,
)


@dataclass(frozen=True, slots=True)
class _GasTreatment:
    """Gas duties on the caller reference; wastewater loads are conditional on treatment.

    Biosphere rows already include plant reaction water and atmospheric oxygen.
    Heat fields are unbound MJ: the parent combines them with its own heat
    allocation. Atmospheric H2S oxidation water/O2 are trace only.
    Internal electricity/heat are generated from wastewater gas and consumed
    within its plant; they reduce purchases once and create no export credit.
    The gas calculation supplies these rows once to the consuming parent.
    TPS/SBR supply a nonzero load per kg principal plastic; PA ingredient
    manufacture supplies a load per kg product. Callers handle zero demand before gas activation.
    """

    biosphere_exchanges: tuple[dict[str, object], ...]
    heat_air_mj: float
    heat_soil_mj: float
    internal_electricity_kwh: float
    internal_heat_mj: float
    trace: dict[str, object]


def _gas_operation(
    context: _RecipeContext,
    selector: dict[str, object],
    fuel_product: str,
    service_id: str,
    role: str,
    *,
    original_physical_reference: bool = False,
) -> _GasOperation:
    """Read matched fuel/emission rows and original LHV from the selected sources.

    Premise can omit fuel properties, so LHV comes from the original 2025
    source. Constructed CH wastewater also keeps original fuel and emission
    coefficients together; other gas receiving uses dated rows. Dividing a
    row by its same-source fuel input cancels nonunit reference production.
    """
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id=f"gas_{role}",
        semantic_class="service_normalization",
    )
    rows: dict[str, dict[str, object]] = {
        str(context.input(ex).get("reference product", context.input(ex)["name"])): dict(ex)
        for ex in context.exchanges(activity)
        if ex["type"] != "production"
    }
    dated_fuel = float(cast(float, rows[fuel_product]["amount"]))
    original = context.source.activity(selector, year=2025)
    reference_terms: dict[str, object] = {}
    if original_physical_reference:
        rows = {
            str(context.input(ex).get("reference product", context.input(ex)["name"])): dict(ex)
            for ex in context.exchanges(original)
            if ex["type"] != "production"
        }
        reference_terms = {
            "physical_reference_source": list(original.key),
            "physical_reference_fuel_input": float(cast(float, rows[fuel_product]["amount"])),
        }
    source_fuel = next(
        ex
        for ex in context.exchanges(original)
        if ex["type"] == "technosphere"
        and context.input(ex).get("reference product") == fuel_product
    )
    lhv = float(source_fuel["properties"]["heating value, net"]["amount"])
    trace["project_terms_json"] = json.dumps(
        {
            "fuel_property_source": list(original.key),
            "fuel_product": fuel_product,
            "net_heat_mj_per_fuel_unit": lhv,
            "dated_fuel_input": dated_fuel,
            "purchased_fuel_omitted": True,
            **reference_terms,
        },
        sort_keys=True,
    )
    return _GasOperation(rows, float(cast(float, rows[fuel_product]["amount"])), lhv, trace)


def _gas_source_trace(
    context: _RecipeContext,
    operation: _GasOperation,
    retained_names: set[str],
    scale: float,
) -> None:
    """Record selected operating rows separately from replaced donor chemistry.

    The serialized source trace records input identity, type, amount, unit and
    categories. Copied native exchange templates
    retain uncertainty metadata through the native scaling owner. The gas ledger states the actual
    normalization and the material-specific carbon properties.
    """
    terms = cast(dict[str, object], json.loads(str(operation.trace["project_terms_json"])))
    terms.update(
        actual_gas_multiplier=scale,
        retained_row_names=sorted(retained_names),
        omitted_row_names=sorted(set(operation.rows) - retained_names),
    )
    for kind in ("technosphere", "biosphere"):
        for disposition, selected in (("retained", True), ("removed", False)):
            operation.trace[f"{disposition}_{kind}_json"] = context.exchange_json(
                ex
                for name, ex in operation.rows.items()
                if ex["type"] == kind and (name in retained_names) == selected
            )
    operation.trace["project_terms_json"] = json.dumps(terms, sort_keys=True)


def _gas_air_rows(
    context: _RecipeContext,
    name: str,
    amount: float,
    unit: str,
    urban_share: float,
    template: dict[str, object] | None = None,
    *,
    principal_origins: PrincipalCarbonOriginFractions | None = None,
) -> list[dict[str, object]]:
    """Bind existing 3.12 air flows and retain positive-scaled native uncertainty.

    Rural/urban identities follow ecoinvent's elementary-flow master data,
    including Pt and escaped NH3. Carbon callers provide the actual
    principal-origin fractions. Geographic and native-reference scaling
    preserve these fractions.
    """
    result: list[dict[str, object]] = []
    if amount == 0:
        return result
    for compartment, share in ((_RURAL, 1 - urban_share), (_URBAN, urban_share)):
        if share == 0:
            continue
        node = context.biosphere.flow(
            {"name": name, "unit": unit, "categories": ["air", compartment]}
        )
        value = amount * share
        row = (
            {"amount": value, "loc": value, "uncertainty type": 0}
            if template is None
            else _scaled_native_exchange(template, value / float(cast(float, template["amount"])))
        )
        for field in ("properties", "activity", "production volume", "classifications"):
            row.pop(field, None)
        row.update(
            input=cast(tuple[str, str], node.key),
            flow=str(node.key[1]),
            name=name,
            unit=unit,
            type="biosphere",
            comment=(
                "Actual residue gas load; selected Doka/engine/burner function. Carbon pool "
                "and origin remain in the gas ledger; donor composition is replaced."
            ),
        )
        if principal_origins is not None:
            row["principal_carbon_origin_fractions"] = principal_origins
        result.append(row)
    return result


def _gas_oxygen(context: _RecipeContext, amount: float) -> list[dict[str, object]]:
    """Bind total plant oxygen to the atmospheric resource flow."""
    node = context.biosphere.flow(
        {"name": "Oxygen", "unit": "kilogram", "categories": ["natural resource", "in air"]}
    )
    return [
        {
            "input": cast(tuple[str, str], node.key),
            "name": "Oxygen",
            "unit": "kilogram",
            "type": "biosphere",
            "amount": amount,
            "loc": amount,
            "uncertainty type": 0,
        }
    ]


def _branch_exchanges(
    context: _RecipeContext, branch: _GasBranch, urban_share: float
) -> list[dict[str, object]]:
    """Bind calculated species/origins and atmospheric oxygen once."""
    result: list[dict[str, object]] = []
    for emission in _branch_emissions(branch):
        result.extend(
            _gas_air_rows(
                context,
                emission.name,
                emission.amount,
                emission.unit,
                urban_share,
                emission.template,
                principal_origins=emission.principal_carbon_origin_fractions,
            )
        )
    result.extend(_gas_oxygen(context, branch.oxygen_kg))
    return result


def _gas_leakage(
    context: _RecipeContext,
    pools: tuple[_GasPool, ...],
    nitrogen: float,
    sulfur: float,
    ammonia_nitrogen: float,
    fraction: float,
    urban_share: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Bind the calculated leakage releases with their source air allocation."""
    ref = _gas_leakage_reference(pools, nitrogen, sulfur, ammonia_nitrogen, fraction)
    result: list[dict[str, object]] = []
    for emission in ref.emissions:
        result.extend(
            _gas_air_rows(
                context,
                emission.name,
                emission.amount,
                emission.unit,
                urban_share,
                principal_origins=emission.principal_carbon_origin_fractions,
            )
        )
    return result, ref.trace


def _landfill_gas(
    context: _RecipeContext,
    construction: dict[str, object],
    load: _LandfillGasLoad,
    service_id: str,
) -> _GasTreatment:
    """Compile residue landfill gas per kg principal, replacing municipal gas duties.

    Doka's selected RoW fractions are .47 uncollected, .1802 flare and .3498
    CHP. Each material owner supplies its actual gas-phase carbon; SBR uses
    its 1%/century matrix profile and TPS its component degradation profile.
    Combustion factors use methane 50 MJ/kg and H2S 518.17 MJ/kmol. The heat
    inventory uses the stated HHV references."""
    sources = cast(dict[str, dict[str, object]], construction["gas_sources"])
    chp = _gas_operation(context, sources["chp"], "biogas", service_id, "CHP")
    furnace = _gas_operation(
        context, sources["furnace"], "natural gas, high pressure", service_id, "flare"
    )
    ref = _landfill_gas_reference(load, chp, furnace)
    pools = load.carbon
    nitrogen, sulfur = load.nitrogen_kg, load.sulfur_kg
    biosphere, leakage = _gas_leakage(context, pools, nitrogen, sulfur, 0.0, 0.47, 0.0)
    for branch in ref.branches:
        biosphere.extend(_branch_exchanges(context, branch, 0.0))
    heat_soil, heat_trace = load.decomposition_heat_mj, load.decomposition_trace
    _gas_source_trace(context, chp, set(_CHP_FACTORS.values()), ref.chp_multiplier)
    _gas_source_trace(
        context,
        furnace,
        {"Dinitrogen monoxide"},
        ref.flare_multiplier,
    )
    trace: dict[str, object] = {
        "basis": "per kg principal material",
        "gas_carbon_inputs": [asdict(item) for item in pools],
        "source_equivalent_volume_m3": ref.volume_m3,
        "physical_dry_volume_m3": ref.physical_dry_volume_m3,
        "combustion_reference_input_mj": ref.energy_mj,
        "flare": ref.branches[0].trace,
        "chp": ref.branches[1].trace,
        "leakage": leakage,
        "decomposition": heat_trace,
        "plant_water_kg_already_bound": fsum(branch.water_kg for branch in ref.branches),
        "plant_oxygen_kg_already_bound": fsum(branch.oxygen_kg for branch in ref.branches),
        "extraction_electricity_kwh": ref.pump_kwh,
        "gross_generation_kwh": ref.generation_kwh,
        "net_electricity_export_kwh_no_credit": ref.export_kwh,
        "useful_heat_mj_no_credit": ref.useful_heat_mj,
        "heat_air_mj_unbound": ref.heat_air_mj,
        "heat_soil_mj_unbound": heat_soil,
    }
    return _GasTreatment(
        tuple(biosphere),
        ref.heat_air_mj,
        heat_soil,
        0.0,
        0.0,
        trace,
    )


def _wastewater_gas(
    context: _RecipeContext,
    construction: dict[str, object],
    load: _WastewaterLoad,
    service_id: str,
    profile: _WastewaterProcessProfile,
    *,
    original_physical_reference: bool = False,
) -> _GasTreatment:
    """Compile actual digester gas on the caller reference before collection allocation.

    Doka2021 WWcalc uses 65/35 methane/CO2, .0075 homogeneous leakage, and
    the selected regional CHP fraction. RoW leachate has zero
    delivered efficiencies; CH co-treatment retains internal generation. Actual gas C/N/S already
    include digestion and fresh PAM: no repeated transfer or biological gas.
    The parent retains whole-plant meters, capital and operating supplies.
    Old EC25 fuel-N NOx, C184 oxygen
    and EC18 water are replaced by the actual gas emissions.
    Constructed CH chemical wastewater retains original gas-normalized factors;
    the default keeps the dated factors used by other plastics receiving.
    """
    sources = cast(dict[str, dict[str, object]], construction["gas_sources"])
    chp = _gas_operation(
        context,
        sources["chp"],
        "biogas",
        service_id,
        "CHP",
        original_physical_reference=original_physical_reference,
    )
    furnace = _gas_operation(
        context,
        sources["furnace"],
        "natural gas, high pressure",
        service_id,
        "flare",
        original_physical_reference=original_physical_reference,
    )
    ref = _wastewater_gas_reference(load, chp, furnace, profile)
    biosphere, leakage = _gas_leakage(
        context,
        ref.pools,
        ref.nitrogen_kg,
        ref.sulfur_kg,
        ref.ammonia_nitrogen_kg,
        0.0075,
        profile.hydraulic.urban_treatment,
    )
    for branch in ref.branches:
        biosphere.extend(_branch_exchanges(context, branch, profile.hydraulic.urban_treatment))
    # The retained WWT function owns digestion/utilization and its operating supplies.
    _gas_source_trace(
        context,
        chp,
        set(_CHP_FACTORS.values()),
        ref.chp_multiplier,
    )
    _gas_source_trace(context, furnace, {"Dinitrogen monoxide"}, ref.flare_multiplier)
    trace: dict[str, object] = {
        "basis": "conditional treatment on the caller reference, before collection share",
        "gas_carbon_inputs": [asdict(item) for item in ref.pools],
        "raw_dry_normal_volume_m3": ref.volume_m3,
        "urban_share": profile.hydraulic.urban_treatment,
        "flare": ref.branches[0].trace,
        "chp": ref.branches[1].trace,
        "leakage": leakage,
        "plant_water_kg_already_bound": fsum(branch.water_kg for branch in ref.branches),
        "plant_oxygen_kg_already_bound": fsum(branch.oxygen_kg for branch in ref.branches),
        "delivered_electricity_kwh": ref.electricity_kwh,
        "delivered_heat_mj": ref.useful_heat_mj,
        "heat_air_mj_unbound": ref.heat_air_mj,
    }
    return _GasTreatment(
        tuple(biosphere),
        ref.heat_air_mj,
        0.0,
        ref.electricity_kwh,
        ref.useful_heat_mj,
        trace,
    )
