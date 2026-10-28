"""Bind actual PA landfill and wastewater gas to admitted native endpoints.

The shared gas owner calculates the unsplit mixture. This module binds
its emissions.
The consuming site or water plant accounts for unbound heat and internal energy.
"""

from dataclasses import dataclass
from typing import cast

from ...prepare_backgrounds.preparation.providers.leachate_load import (
    _WastewaterLoad,
    _WastewaterProcessProfile,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingGasReference,
    PaReceivingGasSources,
)
from ...prepare_backgrounds.preparation.providers.recipe_context import _scaled_native_exchange
from ...prepare_backgrounds.preparation.providers.waste_gas_load import (
    _RURAL,
    _URBAN,
    _branch_emissions,
    _gas_leakage_reference,
    _GasBranch,
    _GasEmission,
    _landfill_gas_reference,
    _LandfillGasLoad,
    _wastewater_gas_reference,
)
from .pa_receiving import PaReceivingAmount


@dataclass(frozen=True, slots=True)
class PaGasReceiving:
    """Complete actual gas quantities and consuming-site energy in kWh and MJ.

    ``amounts`` already includes atmospheric oxygen and plant reaction water.
    Internal electricity and heat reduce the wastewater plant's purchases once.
    Heat to air and soil is unbound MJ for the parent. Atmospheric oxidation
    after leakage remains trace in the shared gas equations.
    """

    amounts: tuple[PaReceivingAmount, ...]
    heat_air_mj: float
    heat_soil_mj: float
    internal_electricity_kwh: float
    internal_heat_mj: float


def _air_amounts(
    function: str,
    source: PaReceivingActivity,
    emissions: tuple[_GasEmission, ...],
    flows: dict[tuple[str, tuple[str, ...]], PaReceivingFlow],
    urban_share: float,
    comment: str,
) -> list[PaReceivingAmount]:
    """Bind each total carbon pool and source species in its actual air compartment.

    A zero emission has no distribution to scale. Positive emissions retain the
    native template's uncertainty while replacing donor carbon attribution.
    Per-pool principal fractions remain valid because pools are not aggregated
    before their individual quantities and origins reach the inventory owner.
    """
    amounts = []
    for emission in emissions:
        if emission.amount == 0:
            continue
        for compartment, share in ((_RURAL, 1 - urban_share), (_URBAN, urban_share)):
            if share == 0:
                continue
            endpoint = flows[emission.name, ("air", compartment)]
            value = emission.amount * share
            row = (
                {"amount": value, "loc": value, "uncertainty type": 0}
                if emission.template is None
                else _scaled_native_exchange(
                    emission.template,
                    value / float(cast(float, emission.template["amount"])),
                )
            )
            for field in ("properties", "activity", "production volume", "classifications"):
                row.pop(field, None)
            row.update(
                input=endpoint.key,
                flow=endpoint.key[1],
                name=endpoint.name,
                unit=endpoint.unit,
                type="biosphere",
                comment=comment,
            )
            row.pop("amount")
            amounts.append(
                PaReceivingAmount(
                    function,
                    source,
                    endpoint,
                    value,
                    row,
                    emission.principal_carbon_origin_fractions,
                )
            )
    return amounts


def _combustion_amounts(
    function: str,
    branches: tuple[_GasBranch, ...],
    sources: PaReceivingGasSources,
    flows: dict[tuple[str, tuple[str, ...]], PaReceivingFlow],
    urban_share: float,
) -> list[PaReceivingAmount]:
    """Bind flare then CHP products and their atmospheric oxygen once."""
    amounts = []
    oxygen = flows["Oxygen", ("natural resource", "in air")]
    # Leakage, flare, and CHP are separate gas destinations. Take each destination's absolute direct
    # contribution before aggregating flows.
    for branch, source, branch_function in zip(
        branches,
        (sources.furnace.activity, sources.chp.activity),
        (f"{function}_flare", f"{function}_chp"),
        strict=True,
    ):
        amounts.extend(
            _air_amounts(
                branch_function,
                source,
                _branch_emissions(branch),
                flows,
                urban_share,
                "Actual PA gas; selected Doka/engine/burner function replaces donor composition.",
            )
        )
        amounts.append(
            PaReceivingAmount(
                branch_function,
                source,
                oxygen,
                branch.oxygen_kg,
                {
                    "input": oxygen.key,
                    "name": oxygen.name,
                    "unit": oxygen.unit,
                    "type": "biosphere",
                    "loc": branch.oxygen_kg,
                    "uncertainty type": 0,
                },
            )
        )
    return amounts


def receive_pa_landfill_gas(
    load: _LandfillGasLoad,
    reference: PaReceivingGasReference,
    sources: PaReceivingGasSources,
) -> PaGasReceiving:
    """Receive one nonzero actual PA/finish landfill gas mixture on its caller basis.

    Doka's .47 leakage, .1802 flare and .3498 CHP allocation is owned by the
    shared gas equations. Electricity generation covers Doka pumping throughout the
    supported PA sanitary-landfill domain. The parent retains native site
    support. The caller handles absent gas before invoking this function.
    """
    function = "pa_landfill_gas"
    chp, furnace = sources.chp_operation(reference), sources.furnace_operation(reference)
    ref = _landfill_gas_reference(load, chp, furnace)
    flows = sources.flow_index
    leakage = _gas_leakage_reference(load.carbon, load.nitrogen_kg, load.sulfur_kg, 0.0, 0.47)
    amounts = _air_amounts(
        f"{function}_leakage",
        reference.chp.activity,
        leakage.emissions,
        flows,
        0.0,
        "Landfill gas leakage follows Doka; the CHP record identifies the shared gas reference.",
    )
    amounts.extend(_combustion_amounts(function, ref.branches, sources, flows, 0.0))
    return PaGasReceiving(tuple(amounts), ref.heat_air_mj, load.decomposition_heat_mj, 0.0, 0.0)


def receive_pa_wastewater_gas(
    load: _WastewaterLoad,
    profile: _WastewaterProcessProfile,
    reference: PaReceivingGasReference,
    sources: PaReceivingGasSources,
) -> PaGasReceiving:
    """Receive actual digester gas before the parent's collection-share allocation.

    The shared gas equations use a 65/35 methane/CO2 carbon split and a 0.0075
    leakage fraction. The selected wastewater profile supplies the CHP share
    and electricity and heat efficiencies for internal generation. The parent
    retains complete whole-plant support. Digestion, hydraulic load and incoming
    collection are applied by their own owners.
    """
    function = "pa_wastewater_gas"
    chp, furnace = sources.chp_operation(reference), sources.furnace_operation(reference)
    ref = _wastewater_gas_reference(load, chp, furnace, profile)
    flows = sources.flow_index
    urban = profile.hydraulic.urban_treatment
    leakage = _gas_leakage_reference(
        ref.pools, ref.nitrogen_kg, ref.sulfur_kg, ref.ammonia_nitrogen_kg, 0.0075
    )
    amounts = _air_amounts(
        f"{function}_leakage",
        reference.chp.activity,
        leakage.emissions,
        flows,
        urban,
        "Wastewater gas leakage follows Doka; the CHP record identifies the shared gas reference.",
    )
    amounts.extend(_combustion_amounts(function, ref.branches, sources, flows, urban))
    return PaGasReceiving(
        tuple(amounts), ref.heat_air_mj, 0.0, ref.electricity_kwh, ref.useful_heat_mj
    )
