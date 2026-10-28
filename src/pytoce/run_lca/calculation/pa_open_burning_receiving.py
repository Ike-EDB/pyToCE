"""Calculate emissions and residues from uncontrolled PA fiber burning."""

from dataclasses import dataclass

from ...prepare_backgrounds.preparation.providers.leachate_load import _CarbonOrigin
from ...prepare_backgrounds.preparation.providers.open_burning import (
    _OpenBurningLoad,
    _pa_open_burning_load,
)
from ...prepare_backgrounds.preparation.providers.pa_finish_material import _pa_finish_reference
from ...prepare_backgrounds.preparation.providers.pa_open_burning_sources import (
    PaOpenBurningReference,
)
from .pa_receiving import PaDisposalEvent, PaReceivingAmount


@dataclass(frozen=True, slots=True)
class PaOpenBurningReceiving:
    """Actual native outputs and the separate uncontrolled-burning mass-balance ledger.

    Air, on-site residue, water and atmospheric inputs stay in ``load``.
    Inventory indicators overlap those material pools; they describe the residue retained at the
    burning site. Masses are kg;
    native outputs retain their admitted units and compartments.
    """

    amounts: tuple[PaReceivingAmount, ...]
    load: _OpenBurningLoad


def receive_pa_open_burning(
    event: PaDisposalEvent,
    reference: PaOpenBurningReference,
) -> PaOpenBurningReceiving:
    """Receive the routed PA end-of-life mixture before origin or date attribution.

    The event already contains the open-burning fate share of principal kg,
    total retained-finish dose and water. Apply none of those shares
    again. The finish owner's seven ingredient fractions remain stationary
    across years and backgrounds.
    The uncontrolled treatment emits directly at the disposal location and
    leaves its residue there.
    """
    grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
    finish = _pa_finish_reference(grade)
    principal: dict[_CarbonOrigin, float] = {
        "fossil": event.principal_by_origin_kg[0],
        "biogenic": event.principal_by_origin_kg[1],
        "atmospheric_dac": event.principal_by_origin_kg[2],
    }
    load = _pa_open_burning_load(
        finish,
        principal_by_origin_kg=principal,
        finish_dose_kg=event.total_finish_dose_kg,
        water_kg=event.water_kg,
    )
    flows = {(flow.name, flow.unit, flow.categories): flow for flow in reference.biosphere}
    amounts: list[PaReceivingAmount] = []
    for key, amount in load.biosphere_amounts.items():
        fractions = load.principal_origin_fractions.get(key)
        # The material-balance calculation supplies each carbon-bearing output's own
        # denominator. Auxiliary carbon remains outside its principal tuple;
        # DAC shares a native non-fossil gas, retaining distinct attribution.
        principal_fractions = (
            (
                fractions.get("fossil", 0.0),
                fractions.get("biogenic", 0.0),
                fractions.get("atmospheric_dac", 0.0),
            )
            if fractions is not None
            else None
        )
        amounts.append(
            PaReceivingAmount(
                "pa_open_burning",
                reference.activity,
                flows[key],
                amount,
                {"uncertainty type": 0, "loc": amount},
                principal_fractions,
            )
        )
    return PaOpenBurningReceiving(tuple(amounts), load)
