"""Calculate PA fiber disposal inventories for one unmanaged dump class."""

from dataclasses import dataclass

from ...prepare_backgrounds.preparation.providers.landfill_phases import (
    _dump_discharges,
    _DumpDischarges,
    _LandfillLoad,
)
from ...prepare_backgrounds.preparation.providers.leachate_load import _CarbonOrigin
from ...prepare_backgrounds.preparation.providers.pa_dump_receiving import PaDumpReceivingReference
from ...prepare_backgrounds.preparation.providers.pa_finish_material import _pa_finish_reference
from ...prepare_backgrounds.preparation.providers.pa_landfill_load import _pa_landfill_load
from .pa_receiving import PaReceivingAmount


@dataclass(frozen=True, slots=True)
class PaDumpReceiving:
    """Actual native amounts and separate complete elemental/origin/water ledgers.

    ``load`` retains actual component phases, including unremoved phosphorus.
    ``discharges`` distinguishes direct gas, overlapping evaporite descriptors,
    soil return, carrier water and complementary gas nitrogen. The event's material and
    infiltration-class quantities determine the receiving load.
    """

    amounts: tuple[PaReceivingAmount, ...]
    load: _LandfillLoad
    discharges: _DumpDischarges


def receive_pa_dump(
    *,
    constituent: str,
    principal_by_origin_kg: dict[_CarbonOrigin, float],
    total_finish_dose_kg: float,
    water_kg: float,
    infiltration_class: str,
    reference: PaDumpReceivingReference,
) -> PaDumpReceiving:
    """Receive the actual mixed PA load allocated to one dump class.

    The caller applies the class share once to all incoming mass coordinates.
    The actual dose already includes retained surface and embedded finish.
    Water belongs to the event. Original-only dump identities supply source climate
    and native flow keys. The caller retains the activity year and world region.
    """
    grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[constituent]
    finish = _pa_finish_reference(grade)
    site = next(item for item in reference.sites if item.infiltration_class == infiltration_class)
    load = _pa_landfill_load(
        finish,
        principal_by_origin_kg=principal_by_origin_kg,
        finish_dose_kg=total_finish_dose_kg,
        water_kg=water_kg,
        site=site.climate,
    )
    physical = _dump_discharges(load, site.climate, upward_leachate=site.upward_leachate)
    function = "pa_open_dump"
    source = site.placement.activity
    amounts = [
        PaReceivingAmount(function, source, row.endpoint, row.amount, row.metadata).scaled(
            load.wet_mass_kg
        )
        for row in site.placement.rows
    ]
    flows = {(flow.name, flow.categories): flow for flow in reference.biosphere}
    for emission in physical.gas:
        # Actual origin pools are kept separate until the inventory owner groups identical endpoints
        # within this process.
        amount = emission.amount
        amounts.append(
            PaReceivingAmount(
                function,
                source,
                flows[emission.name, ("air", "non-urban air or from high stacks")],
                amount,
                {"uncertainty type": 0, "loc": amount},
                emission.principal_carbon_origin_fractions,
            )
        )
    amounts.extend(
        PaReceivingAmount(
            function, source, flows[key], amount, {"uncertainty type": 0, "loc": amount}
        )
        for key, amount in physical.emissions.items()
    )
    return PaDumpReceiving(tuple(amounts), load, physical)
