"""Receive actual PA principal, retained finish and water at a municipal furnace."""

from dataclasses import dataclass, replace
from math import fsum
from typing import Literal, cast

from ...prepare_backgrounds.preparation.providers.ash_treatment import (
    AshReceiving,
    _AshReleaseProfile,
    _TreatmentCarbon,
)
from ...prepare_backgrounds.preparation.providers.material_carbon import (
    CARBON_ORIGINS,
    PrincipalCarbonOriginFractions,
)
from ...prepare_backgrounds.preparation.providers.municipal_incineration_emissions import (
    _NMVOC,
    _TRACE_ORGANICS,
)
from ...prepare_backgrounds.preparation.providers.municipal_incineration_load import (
    _MUNICIPAL_SUPPLIES,
    MunicipalIncinerationLoad,
    MunicipalIncinerationMaterial,
    calculate_municipal_incineration,
)
from ...prepare_backgrounds.preparation.providers.open_burning import _organic_heat_potential
from ...prepare_backgrounds.preparation.providers.pa_finish_material import (
    _pa_finish_load,
    _pa_finish_reference,
)
from ...prepare_backgrounds.preparation.providers.pa_municipal_receiving_sources import (
    PaMunicipalReceivingReference,
    PaMunicipalReceivingSources,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
    PaReceivingRow,
)
from .pa_receiving import PaDisposalEvent, PaReceivingAmount
from .pa_sludge_receiving import _ash_receiving, _market_support

# Municipal2020 slag/residual compartments: initial and cumulative P releases. Native S/N/Na release
# factors remain as supplied. Ca is fresh half-solid lime, with the residual-source release
# reference.
_PA_MUNICIPAL_ASH_RELEASES = {
    "bottom": _AshReleaseProfile(
        (("Phosphate", "P", 31 / 95, 0.000022037531153838305, 0.013222518692302985),),
    ),
    "residual": _AshReleaseProfile(
        (
            ("Phosphate", "P", 31 / 95, 0.00037317372036873187, 0.22390423222123912),
            ("Calcium II", "Ca", 1.0, 0.00015084836425899547, 0.090509018555397294),
        ),
    ),
}
_CARBON_OUTPUTS = set(_TRACE_ORGANICS) | {
    _NMVOC,
    "Cyanide",
    "DOC, Dissolved Organic Carbon",
    "TOC, Total Organic Carbon",
    "COD, Chemical Oxygen Demand",
    "BOD5, Biological Oxygen Demand",
}


@dataclass(frozen=True, slots=True)
class PaMunicipalReceiving:
    """Five-function LCI and separate furnace/ash mass-balance ledgers, actual event basis."""

    amounts: tuple[PaReceivingAmount, ...]
    incineration: MunicipalIncinerationLoad
    bottom_ash: AshReceiving
    residual_ash: AshReceiving


def _pa_municipal_material(event: PaDisposalEvent) -> MunicipalIncinerationMaterial:
    """Assemble routed principal/finish/water once, preserving PA nominal CHNO.

    PA6 and PA66 share C6H11NO per 113 g (PA66 is twice that unit). The
    retained-finish owner supplies the seven ingredients' CIAAW vector.
    The event already supplies its fate share and post-washing state; dry rejects retain their
    outgoing embedded finish and dry mass.
    """
    grade = {"polyamide_6_resin": "pa6", "polyamide_66_resin": "pa66"}[event.constituent]
    finish = _pa_finish_reference(grade)
    auxiliary = _pa_finish_load(
        {name: event.total_finish_dose_kg * kg for name, kg in finish.retained.compounds_kg.items()}
    )
    principal = fsum(event.principal_by_origin_kg)
    elements = dict(auxiliary.elements_kg)
    for element, nominal in (("C", 72), ("H", 11), ("N", 14), ("O", 16)):
        elements[element] += principal * nominal / 113
    pools = tuple(
        _TreatmentCarbon(
            "PA",
            True,
            cast(Literal["fossil", "biogenic", "atmospheric_dac"], origin),
            kg * 72 / 113,
        )
        for origin, kg in zip(CARBON_ORIGINS, event.principal_by_origin_kg, strict=True)
    )
    pools += (
        _TreatmentCarbon("PA_finish", False, "fossil", auxiliary.fossil_carbon_kg),
        _TreatmentCarbon("PA_finish", False, "biogenic", auxiliary.biogenic_carbon_kg),
    )
    return MunicipalIncinerationMaterial(
        principal + fsum(auxiliary.compounds_kg.values()) + event.water_kg,
        event.water_kg,
        elements,
        pools,
        # Retain the admitted PA/finish CHONS heat convention at its material owner.
        _organic_heat_potential(elements),
    )


def receive_pa_municipal_incineration(
    event: PaDisposalEvent,
    reference: PaMunicipalReceivingReference,
    sources: PaMunicipalReceivingSources,
) -> PaMunicipalReceiving:
    """Bind one actual municipal event through all five selected source functions.

    Equations run before origin attribution and dated supply shares.

    This receiver retains the internal ash markets and treatment activities.
    Native catalyst suppliers retain their actual doses; selected PAM replaces
    the source's inorganic flocculant proxy exactly once.
    """
    material = _pa_municipal_material(event)
    load = calculate_municipal_incineration(material, reference.source())
    function = "pa_municipal_incineration"
    operation = sources.incineration
    grouped: dict[str, list[PaReceivingRow]] = {}
    for row in operation.rows:
        grouped.setdefault(cast(PaReceivingActivity, row.endpoint).reference_product, []).append(
            row
        )
    amounts: list[PaReceivingAmount] = []
    for duty, (product, _) in _MUNICIPAL_SUPPLIES.items():
        # Native TiO2/chromium proxies supply oxide-equivalent doses. Residue chemistry retains
        # Ti/V.
        quantity = load.duties[duty]
        rows = grouped[sources.reheat_product if duty == "reheat" else product]
        total = fsum(row.amount for row in rows)
        for row in rows:
            actual = PaReceivingAmount(
                function, operation.activity, row.endpoint, row.amount, row.metadata
            ).scaled(quantity / total)
            if duty == "PAM":
                actual.metadata.pop("activity", None)
                actual.metadata.pop("flow", None)
                actual = replace(actual, endpoint=sources.pam)
            amounts.append(actual)
    flows = {(flow.name, flow.unit, flow.categories): flow for flow in sources.biosphere}
    templates: dict[tuple[str, str, tuple[str, ...]], list[PaReceivingRow]] = {}
    for row in reference.incineration.rows:
        if isinstance(row.endpoint, PaReceivingFlow):
            templates.setdefault(
                (row.endpoint.name, row.endpoint.unit, row.endpoint.categories), []
            ).append(row)
    total_carbon = fsum(pool.carbon_kg for pool in material.carbon)
    for key, quantity in load.emissions.items():
        name, unit, categories = key
        gas = name.startswith(("Carbon dioxide,", "Carbon monoxide,", "Methane,"))
        groups = ("fossil", "non-fossil") if gas else ("all",)
        for origin_group in groups:
            pools = tuple(
                pool
                for pool in material.carbon
                if origin_group == "all" or (pool.origin == "fossil") == (origin_group == "fossil")
            )
            carbon = fsum(pool.carbon_kg for pool in pools)
            # Every routed positive PA fiber load retains the fixed finish:
            # surface plus inherited material at EOL, or rho>0 washed material
            # in dry rejects. It supplies positive fossil and biological pools.
            share = carbon / total_carbon
            target_name = name.rsplit(", ", 1)[0] + ", " + origin_group if gas else name
            target = flows[target_name, unit, categories]
            fractions = (
                cast(
                    PrincipalCarbonOriginFractions,
                    tuple(
                        fsum(
                            pool.carbon_kg
                            for pool in pools
                            if pool.principal and pool.origin == origin
                        )
                        / carbon
                        for origin in CARBON_ORIGINS
                    ),
                )
                if gas or name in _CARBON_OUTPUTS
                else None
            )
            factor = load.native_emission_factors.get(key)
            if factor is not None:
                for row in templates[key]:
                    actual = PaReceivingAmount(
                        function,
                        reference.incineration.activity,
                        target,
                        row.amount,
                        row.metadata,
                        fractions,
                    ).scaled(factor * share)
                    if target.key != row.endpoint.key:
                        actual.metadata.pop("activity", None)
                        actual.metadata.pop("flow", None)
                    amounts.append(actual)
                # Native Na has its original uncertainty; fresh alkali sodium is an additional
                # deterministic aqueous quantity.
                if name != "Sodium I" or categories != ("water", "surface water"):
                    continue
                quantity -= fsum(row.amount for row in templates[key]) * factor
            amounts.append(
                PaReceivingAmount(
                    function,
                    reference.incineration.activity,
                    target,
                    quantity * share,
                    {"uncertainty type": 0, "loc": quantity * share},
                    fractions,
                )
            )
    ash_results = []
    for fraction, ash in (("bottom", load.bottom_ash), ("residual", load.residual_ash)):
        market = sources.bottom_ash_market if fraction == "bottom" else sources.residual_ash_market
        terminal = (
            sources.bottom_ash_treatment if fraction == "bottom" else sources.residual_ash_treatment
        )
        calibrated = reference.bottom_ash if fraction == "bottom" else reference.residual_ash
        amounts.extend(
            _market_support(
                market,
                terminal.activity,
                ash.treatment_mass_kg,
                f"pa_municipal_{fraction}_ash_market",
            )
        )
        received, physical = _ash_receiving(
            ash,
            calibrated,
            terminal,
            flows,
            _PA_MUNICIPAL_ASH_RELEASES[fraction],
            f"pa_municipal_{fraction}_ash_treatment",
        )
        amounts.extend(received)
        ash_results.append(physical)
    return PaMunicipalReceiving(tuple(amounts), load, ash_results[0], ash_results[1])
