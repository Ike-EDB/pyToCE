"""Bind actual PA sludge and ash through complete Swiss or sanitary receiving."""

from dataclasses import asdict, dataclass, replace
from math import fsum
from typing import Literal, cast

from ...prepare_backgrounds.preparation.providers.ash_treatment import (
    _FLOW_ELEMENTS,
    _PA_ASH_RELEASES,
    AshReceiving,
    AshSource,
    _AshLoad,
    _AshReleaseProfile,
    calculate_ash_receiving,
)
from ...prepare_backgrounds.preparation.providers.material_carbon import (
    CARBON_ORIGINS,
    PrincipalCarbonOriginFractions,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingAshReference,
    PaReceivingFlow,
    PaReceivingOperation,
    PaReceivingRow,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from ...prepare_backgrounds.preparation.providers.sludge_incineration_load import (
    _SUPPLY,
    SludgeIncinerationSource,
    _IncinerationLoad,
    calculate_sludge_incineration,
)
from ...prepare_backgrounds.preparation.providers.swiss_sludge_load import (
    AIR_OXYGEN,
    SUPPLIES,
    SURFACE,
    SwissSludgeLoad,
    calculate_swiss_sludge,
)
from .pa_receiving import PaReceivingAmount
from .pa_water_receiving import PaWaterReceiving


@dataclass(frozen=True, slots=True)
class PaSludgeReceiving:
    """Actual LCI and conditional furnace/ash fates before collection scaling.

    ``amounts`` already include the incoming water's treatment fraction once.
    The other three fields retain the wastewater caller's conditional basis,
    including its existing sludge-incineration share. Ash retained phosphorus
    is a pool.
    """

    amounts: tuple[PaReceivingAmount, ...]
    incineration: SwissSludgeLoad | _IncinerationLoad
    bottom_ash: AshReceiving
    residual_ash: AshReceiving


def _native_sludge_amount(
    function: str, operation: PaReceivingOperation, row: PaReceivingRow, multiplier: float
) -> PaReceivingAmount:
    """Scale one normalized native row while retaining its individual distribution."""
    return PaReceivingAmount(
        function, operation.activity, row.endpoint, row.amount, row.metadata
    ).scaled(multiplier)


def _market_support(
    operation: PaReceivingOperation,
    child: PaReceivingActivity,
    mass_kg: float,
    function: str,
) -> list[PaReceivingAmount]:
    """Retain native haulage and reference the separately calculated child."""
    return [
        _native_sludge_amount(function, operation, row, mass_kg)
        for row in operation.rows
        if row.endpoint.key != child.key
    ]


def _ash_receiving(
    load: _AshLoad,
    reference: PaReceivingAshReference,
    operation: PaReceivingOperation,
    flows: dict[tuple[str, str, tuple[str, ...]], PaReceivingFlow],
    profile: _AshReleaseProfile,
    function: str,
) -> tuple[list[PaReceivingAmount], AshReceiving]:
    """Bind original ash release equations and complete dated terminal support once."""
    # Both ash admission owners retain only original elementary rows here.
    native = reference.operation.rows
    source = AshSource(
        reference.dry_fraction,
        reference.element_contents,
        tuple(
            (row.endpoint.name, cast(PaReceivingFlow, row.endpoint).categories, row.amount)
            for row in native
        ),
    )
    physical = calculate_ash_receiving(load, source, release_profile=profile)
    # Dated sources supply operating support. Original elementary rows
    # supply release distributions.
    amounts = [
        _native_sludge_amount(function, operation, row, load.treatment_mass_kg)
        for row in operation.rows
    ]
    # The furnace assigns principal and auxiliary carbon to the ash fraction, including fresh
    # flocculant carbon. Release functions use that mixture. DOC, TOC, COD, and BOD describe
    # overlapping quantities, attributed by these carbon fractions.
    total_carbon = fsum(pool.carbon_kg for pool in load.carbon)
    principal = cast(
        PrincipalCarbonOriginFractions,
        tuple(
            fsum(pool.carbon_kg for pool in load.carbon if pool.principal and pool.origin == origin)
            / total_carbon
            for origin in CARBON_ORIGINS
        ),
    )
    for release in physical.native:
        row = native[release.source_index]
        actual = _native_sludge_amount(function, reference.operation, row, release.multiplier)
        if _FLOW_ELEMENTS.get(row.endpoint.name) == "C":
            actual = replace(actual, principal_carbon_origin_fractions=principal)
        amounts.append(actual)
    for name, categories, amount in physical.additions:
        amounts.append(
            PaReceivingAmount(
                function,
                reference.operation.activity,
                flows[name, "kilogram", categories],
                amount,
                {"uncertainty type": 0, "loc": amount},
            )
        )
    return amounts, physical


def receive_pa_incinerated_sludge(
    water: PaWaterReceiving,
    reference: PaReceivingReference,
    sources: PaReceivingSources,
    *,
    receiving: Literal["swiss", "sanitary"],
) -> PaSludgeReceiving:
    """Receive actual conditional PA sludge through its fixed complete source chain.

    The wastewater owner already selected incineration versus landfarming and
    supplied their distinct wet masses. The input amounts already use the receiving activity basis.
    The collection treatment fraction scales final
    rows only; returned furnace and ash records keep their conditional basis.
    """
    if receiving == "swiss":
        calibrated, dated = reference.water, sources.water
        incineration: SwissSludgeLoad | _IncinerationLoad = calculate_swiss_sludge(
            reference.water.sludge.source(), water.wastewater
        )
        supplies = SUPPLIES
    else:
        calibrated, dated = reference.landfill, sources.landfill
        original = reference.landfill.sludge
        emissions: dict[tuple[str, str, tuple[str, ...]], float] = {}
        for row in original.operation.rows:
            if isinstance(row.endpoint, PaReceivingFlow):
                key = row.endpoint.name, row.endpoint.unit, row.endpoint.categories
                emissions[key] = emissions.get(key, 0.0) + row.amount
        incineration = calculate_sludge_incineration(
            SludgeIncinerationSource(
                original.elements_kg, original.operation.supplies(), emissions
            ),
            water.wastewater,
            "pa_actual_sanitary_sludge",
            # The selected plastic-leachate source contains fossil gas templates
            # only. Actual PA pools retain all three origins. Its missing Ca/P
            # rows use the already admitted APC and six-phase report functions.
            source_carbon_origins=("fossil",),
            additional_apc_water_flows=("Calcium II",),
            phosphorus_phase_reference=True,
            # Selected plastic-leachate slag: 11.5628% water / 88.4372% dry.
            bottom_ash_water_to_dry_ratio=0.115628 / 0.884372,
        )
        supplies = _SUPPLY
    function = f"pa_{receiving}_sludge_incineration"
    operation = dated.sludge_incineration
    amounts = _market_support(
        dated.sludge_incineration_market,
        operation.activity,
        incineration.wet_mass_kg,
        f"pa_{receiving}_sludge_collection",
    )
    grouped: dict[str, list[PaReceivingRow]] = {}
    for row in operation.rows:
        # Dated furnace admission retains the selected supplies only; original
        # emission templates have their separate calibrated operation below.
        product = cast(PaReceivingActivity, row.endpoint).reference_product
        if product == "heat, district or industrial":
            product = supplies["reheat"][0]
        grouped.setdefault(product, []).append(row)
    for duty, (product, _) in supplies.items():
        # Native TiO2/chromium proxies supply oxide-equivalent doses. Residue chemistry retains
        # Ti/V.
        quantity = incineration.duties[duty]
        if quantity == 0:
            continue
        rows = grouped[product]
        coefficient = fsum(row.amount for row in rows)
        for row in rows:
            actual = _native_sludge_amount(function, operation, row, quantity / coefficient)
            if duty == "PAM":
                actual.metadata.pop("activity", None)
                actual.metadata.pop("flow", None)
                actual = replace(actual, endpoint=dated.pam)
            amounts.append(actual)
    if isinstance(incineration, SwissSludgeLoad):
        rows = grouped["metalliferous hydroxide sludge"]
        coefficient = fsum(row.amount for row in rows)
        for row in rows:
            actual = _native_sludge_amount(
                function, operation, row, -incineration.hydroxide_kg / coefficient
            )
            # The positive cut-off recipient takes a negative outgoing input.
            # This is no zinc substitution or separate Waelz-treatment purchase.
            actual.metadata["swiss_hydroxide_cutoff"] = {
                "mass_kg": incineration.hydroxide_kg,
                "water_kg": incineration.hydroxide_water_kg,
                "elements_kg": incineration.phases["hydroxide"],
                "carbon": [asdict(pool) for pool in incineration.hydroxide_carbon],
            }
            amounts.append(actual)
    original_operation = calibrated.sludge.operation
    templates: dict[tuple[str, str, tuple[str, ...]], list[PaReceivingRow]] = {}
    for row in original_operation.rows:
        if isinstance(row.endpoint, PaReceivingFlow):
            key = row.endpoint.name, row.endpoint.unit, row.endpoint.categories
            templates.setdefault(key, []).append(row)
    flows = {(flow.name, flow.unit, flow.categories): flow for flow in dated.biosphere}
    added: set[tuple[str, str, tuple[str, ...]]] = {("Chloride", "kilogram", SURFACE)}
    if receiving == "swiss":
        added.add(("Oxygen", "kilogram", AIR_OXYGEN))
    else:
        added.update(
            {
                ("Calcium II", "kilogram", SURFACE),
                ("Phosphate", "kilogram", SURFACE),
                ("Phosphorus", "kilogram", ("air", "urban air close to ground")),
            }
        )
    for key, quantity in incineration.emissions.items():
        if quantity == 0:
            continue
        if key in added:
            amounts.append(
                PaReceivingAmount(
                    function,
                    original_operation.activity,
                    flows[key],
                    quantity,
                    {"uncertainty type": 0, "loc": quantity},
                )
            )
            continue
        name, unit, categories = key
        template = key
        gas = name.startswith(("Carbon dioxide,", "Carbon monoxide,", "Methane,"))
        if receiving == "sanitary" and gas:
            template = name.rsplit(", ", 1)[0] + ", fossil", unit, categories
        rows = templates[template]
        coefficient = fsum(row.amount for row in rows)
        principal = None
        if gas:
            fossil = name.endswith(", fossil")
            total = fsum(
                pool.carbon_kg
                for pool in incineration.carbon
                if (pool.origin == "fossil") == fossil
            )
            principal = cast(
                PrincipalCarbonOriginFractions,
                tuple(
                    fsum(
                        pool.carbon_kg
                        for pool in incineration.carbon
                        if pool.principal
                        and pool.origin == origin
                        and (pool.origin == "fossil") == fossil
                    )
                    / total
                    for origin in CARBON_ORIGINS
                ),
            )
        for row in rows:
            actual = _native_sludge_amount(
                function, original_operation, row, quantity / coefficient
            )
            if template != key:
                actual.metadata.pop("activity", None)
                actual.metadata.pop("flow", None)
                actual = replace(actual, endpoint=flows[key])
            amounts.append(replace(actual, principal_carbon_origin_fractions=principal))
    ash_results = []
    for fraction, load, ash_reference, market, terminal in (
        (
            "bottom",
            incineration.bottom_ash,
            calibrated.bottom_ash,
            dated.bottom_ash_market,
            dated.bottom_ash_treatment,
        ),
        (
            "residual",
            incineration.residual_ash,
            calibrated.residual_ash,
            dated.residual_ash_market,
            dated.residual_ash_treatment,
        ),
    ):
        amounts.extend(
            _market_support(
                market,
                terminal.activity,
                load.treatment_mass_kg,
                f"pa_{receiving}_{fraction}_ash_collection",
            )
        )
        ash_amounts, physical = _ash_receiving(
            load,
            ash_reference,
            terminal,
            flows,
            _AshReleaseProfile(()) if receiving == "swiss" else _PA_ASH_RELEASES[fraction],
            f"pa_{receiving}_{fraction}_ash",
        )
        amounts.extend(ash_amounts)
        ash_results.append(physical)
    return PaSludgeReceiving(
        tuple(amount.scaled(water.treatment_fraction) for amount in amounts),
        incineration,
        ash_results[0],
        ash_results[1],
    )
