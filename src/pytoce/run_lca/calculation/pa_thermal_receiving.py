"""Receive actual PA spinning-equipment deposits inside the integrated site.

Prepared native rows supply dated operations and original calibration. The
chemical-residue owner calculates material fates once; this module binds those
actual quantities to portable endpoints.
"""

from dataclasses import dataclass, replace
from typing import cast

from ...prepare_backgrounds.preparation.chemical_residue_load import (
    _AMMONIA,
    _FERRIC,
    _HCL,
    _NAOH,
    _PEROXIDE,
    _TMT,
    _URBAN,
    _residue_added_releases,
    _residue_ash_additions,
    _residue_ash_release,
    _residue_reference,
    _residue_waste_release,
)
from ...prepare_backgrounds.preparation.providers.material_carbon import (
    PrincipalCarbonOriginFractions,
)
from ...prepare_backgrounds.preparation.providers.pa_finish_material import _pa_finish_reference
from ...prepare_backgrounds.preparation.providers.pa_finish_residues import (
    _pa_maintenance_residue_load,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
    PaReceivingRow,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from .pa_receiving import PaReceivingAmount, PaSpinningEvent


@dataclass(frozen=True, slots=True)
class PaMaintenanceReceiving:
    """Actual maintenance LCI and ordinary water allocated from the spinning site.

    ``site_water_kg`` is part of the parent's existing water meter. Final discharges are already in
    ``amounts``. The spinning owner supplies the site's energy and capital quantities.
    """

    amounts: tuple[PaReceivingAmount, ...]
    site_water_kg: float


def _scaled_maintenance_row(
    function: str,
    source: PaReceivingActivity,
    row: PaReceivingRow,
    multiplier: float,
    endpoint: PaReceivingActivity | PaReceivingFlow,
    principal: PrincipalCarbonOriginFractions | None = None,
) -> PaReceivingAmount:
    """Scale an admitted individual row and remove remapped endpoint identities."""
    scaled = PaReceivingAmount(
        function, source, row.endpoint, row.amount, row.metadata, principal
    ).scaled(multiplier)
    if endpoint.key != row.endpoint.key:
        # Native activity/flow identifiers no longer describe the complete PAM
        # supplier or the exact non-fossil gas counterpart. Allocation and the
        # scaled distribution still describe the unchanged source coefficient.
        scaled.metadata.pop("activity", None)
        scaled.metadata.pop("flow", None)
    return replace(scaled, endpoint=endpoint)


def receive_pa_maintenance(
    event: PaSpinningEvent,
    reference: PaReceivingReference,
    sources: PaReceivingSources,
) -> PaMaintenanceReceiving:
    """Bind complete integrated treatment of an event's already destroyed deposits.

    Custody has already partitioned principal and inherited finish to maintenance.
    Its three principal masses retain fossil/biogenic/DAC order; finish keeps
    the seven-ingredient finish reference.
    Source rows are already per positive native unit; apply the recorded quantities directly. Ash
    quantities instead follow actual ash kg.
    """
    grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
    finish = _pa_finish_reference(grade)
    load = _pa_maintenance_residue_load(
        finish,
        principal_by_origin_kg=event.maintenance_principal_by_origin_kg,
        embedded_dose_kg=event.maintenance_finish_dose_kg,
    )
    source = sources.thermal_source(reference)
    ash_source = sources.ash_source(reference)
    physical = _residue_reference(load, source, integrated_pa_site=True)
    mass, duties = load.wet_mass_kg, physical["supplies"]
    supplies = {
        _AMMONIA: duties["NH3_waste"],
        _HCL: duties["HCl"],
        _NAOH: duties["NaOH"],
        _PEROXIDE: duties["H2O2"],
        _FERRIC: duties["FeCl3"],
        _TMT: duties["TMT"],
        "calcium chloride": duties["CaCl2"],
        "quicklime, milled, packed": duties["CaO"],
        "chemical, inorganic": duties["PAM"],
        "titanium dioxide": (
            source.waste_supplies["titanium dioxide"] * physical["nitrogen_ratio"]
        ),
        "chromium": source.waste_supplies["chromium"] * physical["nitrogen_ratio"],
    }
    flows = {
        (row.endpoint.name, row.endpoint.unit, row.endpoint.categories): row.endpoint
        for row in (*sources.hwi.rows, *sources.fixed.rows, *sources.ash.rows)
        if isinstance(row.endpoint, PaReceivingFlow)
    }
    flows.update(
        {(flow.name, flow.unit, flow.categories): flow for flow in sources.added_biosphere}
    )
    amounts: list[PaReceivingAmount] = []
    for row in sources.hwi.rows:
        endpoint = row.endpoint
        if isinstance(endpoint, PaReceivingActivity):
            product = endpoint.reference_product
            if product not in supplies or supplies[product] == 0:
                continue
            # Preserve each dated supplier's share in the original function. The inorganic
            # flocculant coefficient purchases complete PAM. Native Ti/Cr supply quantities
            # represent actual Ti/V oxide doses.
            multiplier = supplies[product] / source.waste_supplies[product]
            target = sources.pam if product == "chemical, inorganic" else endpoint
            function = (
                "pa_maintenance_scr"
                if product in {"titanium dioxide", "chromium"}
                else "pa_maintenance_incineration"
            )
            amounts.append(
                _scaled_maintenance_row(function, sources.hwi.activity, row, multiplier, target)
            )
            continue
        name, categories = endpoint.name, endpoint.categories
        # Admission excludes native Water rows, the generic helper's sole
        # omitted-output case; actual water is emitted from the mass balance.
        amount = cast(
            float, _residue_waste_release(name, categories, row.amount, load, ash_source, physical)
        )
        if name in {"Carbon dioxide, fossil", "Carbon monoxide, fossil"}:
            counterpart = name.replace("fossil", "non-fossil")
            nonfossil = physical["gases"][counterpart]
            if nonfossil != 0:
                amounts.append(
                    _scaled_maintenance_row(
                        "pa_maintenance_incineration",
                        sources.hwi.activity,
                        row,
                        nonfossil / row.amount,
                        flows[counterpart, endpoint.unit, categories],
                        physical["principal_gas_origin_fractions"].get(counterpart),
                    )
                )
        if amount == 0:
            continue
        principal = (
            physical["principal_gas_origin_fractions"].get(name) if categories == _URBAN else None
        )
        amounts.append(
            _scaled_maintenance_row(
                "pa_maintenance_incineration",
                sources.hwi.activity,
                row,
                amount / row.amount,
                endpoint,
                principal,
            )
        )
    for row in sources.fixed.rows:
        endpoint = row.endpoint
        if isinstance(endpoint, PaReceivingActivity):
            if endpoint.reference_product == sources.reheat_product:
                # The spinning site's existing energy meter covers DeNOx reheating.
                continue
        elif endpoint.name in {"Water", "Water, well, in ground"}:
            # The parent site owns the water withdrawal; the mass balance
            # supplies actual discharges below.
            continue
        amounts.append(
            _scaled_maintenance_row(
                "pa_maintenance_scr", sources.fixed.activity, row, mass, endpoint
            )
        )
    ash_mass = physical["ash_mass_kg"]
    for row in sources.ash.rows:
        endpoint = row.endpoint
        if isinstance(endpoint, PaReceivingActivity):
            amount = row.amount
        else:
            amount = _residue_ash_release(
                endpoint.name, endpoint.categories, row.amount, ash_source, physical
            )
            if amount is None:
                continue
        if amount != 0:
            amounts.append(
                _scaled_maintenance_row(
                    "pa_maintenance_ash",
                    sources.ash.activity,
                    row,
                    amount * ash_mass / row.amount,
                    endpoint,
                )
            )
    for name, categories, per_kg in _residue_ash_additions(physical):
        amount = per_kg * ash_mass
        amounts.append(
            PaReceivingAmount(
                "pa_maintenance_ash",
                sources.ash.activity,
                flows[name, "kilogram", categories],
                amount,
                {"uncertainty type": 0, "loc": amount},
            )
        )
    for name, unit, categories, amount in _residue_added_releases(load, physical):
        amounts.append(
            PaReceivingAmount(
                "pa_maintenance_incineration",
                sources.hwi.activity,
                flows[name, unit, categories],
                amount,
                {"uncertainty type": 0, "loc": amount},
            )
        )
    return PaMaintenanceReceiving(tuple(amounts), physical["site_water_kg"])
