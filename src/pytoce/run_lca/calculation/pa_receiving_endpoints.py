"""Finite native supplier and elementary-flow axes for actual PA receiving.

Each scientific owner declares its retained operating rows and replaced children.
Enumeration applies those same local boundaries to completed source records; it enumerates their
prepared native endpoint identities.
"""

from ...prepare_backgrounds.preparation.chemical_residue_load import (
    _AMMONIA,
    _FERRIC,
    _HCL,
    _NAOH,
    _PEROXIDE,
    _TMT,
)
from ...prepare_backgrounds.preparation.providers.municipal_incineration_load import (
    _MUNICIPAL_SUPPLIES,
)
from ...prepare_backgrounds.preparation.providers.pa_landfill_receiving_sources import (
    PaLandfillReceivingSources,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
)
from ...prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    PaWaterReceivingSources,
)
from ...prepare_backgrounds.preparation.providers.sludge_incineration_load import _SUPPLY
from ...prepare_backgrounds.preparation.providers.swiss_sludge_load import SUPPLIES


def _water_suppliers(
    dated: PaWaterReceivingSources | PaLandfillReceivingSources,
    sludge_products: set[str],
) -> list[PaReceivingActivity]:
    """Retain one complete collection, plant, sludge, soil and two-ash chain.

    Each market replaces only its own calculated child. The same
    activity may remain a complete supplier elsewhere, for example the recycler's
    independent cooling-water treatment. PAM replaces its source proxy; catalyst
    suppliers and negative waste recipients retain their native identities.
    """

    native = [dated.pam]
    pairs = (
        (dated.collection, {dated.wastewater.activity.key}),
        (
            dated.wastewater,
            {
                dated.sludge_incineration_market.activity.key,
                dated.sludge_landfarming_market.activity.key,
            },
        ),
        (dated.sludge_incineration_market, {dated.sludge_incineration.activity.key}),
        (dated.sludge_landfarming_market, {dated.sludge_landfarming.activity.key}),
        (dated.bottom_ash_market, {dated.bottom_ash_treatment.activity.key}),
        (dated.residual_ash_market, {dated.residual_ash_treatment.activity.key}),
        (dated.sludge_landfarming, set()),
        (dated.bottom_ash_treatment, set()),
        (dated.residual_ash_treatment, set()),
    )
    for operation, children in pairs:
        native.extend(
            row.endpoint
            for row in operation.rows
            if isinstance(row.endpoint, PaReceivingActivity) and row.endpoint.key not in children
        )
    native.extend(
        row.endpoint
        for row in dated.sludge_incineration.rows
        if isinstance(row.endpoint, PaReceivingActivity)
        and row.endpoint.reference_product in sludge_products
    )
    return native


def pa_receiving_endpoints(
    reference: PaReceivingReference,
    sources: PaReceivingSources,
) -> tuple[tuple[PaReceivingActivity, ...], tuple[PaReceivingFlow, ...]]:
    """Enumerate native suppliers and elementary flows for one source anchor.

    Supplier identities are deduplicated only within this reference/source pair;
    the caller supplies the Brightway matrix context and resolves each activity's
    actual reference product. Calibration activities document the source equations; receiving uses
    the supplied operating and internal-child quantities. Conditional electricity,
    support and receiving suppliers remain in the enumeration even
    when one particular event has no demand for them.

    Elementary flows are included independently of event amounts and native
    characterization factors. Original calibration rows and dated support have
    different roles, but both can supply an elementary-flow identity.
    Sorted native keys give a deterministic order.
    """

    native = [sources.pam]
    # Integrated spinning maintenance inherits site fuel/power/capital. Its
    # actual thermal owner retains native reagents and catalyst suppliers, with
    # complete PAM replacing the generic inorganic dose, plus full ash support.
    maintenance_reagents = {
        _AMMONIA,
        _HCL,
        _NAOH,
        _PEROXIDE,
        _FERRIC,
        _TMT,
        "calcium chloride",
        "quicklime, milled, packed",
        "titanium dioxide",
        "chromium",
    }
    native.extend(
        row.endpoint
        for row in sources.hwi.rows
        if isinstance(row.endpoint, PaReceivingActivity)
        and row.endpoint.reference_product in maintenance_reagents
    )
    native.extend(
        row.endpoint
        for row in sources.fixed.rows
        if isinstance(row.endpoint, PaReceivingActivity)
        and row.endpoint.reference_product != sources.reheat_product
    )
    native.extend(
        row.endpoint for row in sources.ash.rows if isinstance(row.endpoint, PaReceivingActivity)
    )
    gas = sources.gas
    swiss_products = {product for duty, (product, _) in SUPPLIES.items() if duty != "PAM"} | {
        "heat, district or industrial",
        "metalliferous hydroxide sludge",
    }
    sanitary_products = {product for duty, (product, _) in _SUPPLY.items() if duty != "PAM"} | {
        "heat, district or industrial"
    }
    native.extend(_water_suppliers(sources.water, swiss_products))
    native.extend(_water_suppliers(sources.landfill, sanitary_products))
    native.extend(
        row.endpoint
        for row in sources.landfill.landfill.rows
        if isinstance(row.endpoint, PaReceivingActivity)
        and row.endpoint.key != sources.landfill.collection.activity.key
    )
    municipal = sources.municipal
    municipal_products = {
        product
        for duty, (product, _) in _MUNICIPAL_SUPPLIES.items()
        if duty not in {"PAM", "reheat"}
    } | {municipal.reheat_product}
    native.append(municipal.pam)
    native.extend(
        row.endpoint
        for row in municipal.incineration.rows
        if isinstance(row.endpoint, PaReceivingActivity)
        and row.endpoint.reference_product in municipal_products
    )
    for market, terminal in (
        (municipal.bottom_ash_market, municipal.bottom_ash_treatment),
        (municipal.residual_ash_market, municipal.residual_ash_treatment),
    ):
        native.extend(
            row.endpoint
            for row in market.rows
            if isinstance(row.endpoint, PaReceivingActivity)
            and row.endpoint.key != terminal.activity.key
        )
        native.extend(
            row.endpoint for row in terminal.rows if isinstance(row.endpoint, PaReceivingActivity)
        )
    recycling = sources.recycling
    replaced_preparation = (*recycling.preparation_pm_rows, *recycling.preparation_wastewater_rows)
    native.extend(
        row.endpoint
        for index, row in enumerate(recycling.preparation.rows)
        if index not in replaced_preparation and isinstance(row.endpoint, PaReceivingActivity)
    )
    # Finishing's cooling-water treatment is a complete retained supplier, separate from the
    # replaced pre-wash wastewater.
    native.extend(
        row.endpoint
        for index, row in enumerate(recycling.finishing.rows)
        if index not in recycling.finishing_pm_rows
        and isinstance(row.endpoint, PaReceivingActivity)
    )
    native.append(recycling.pellet_delivery)
    flows = [
        *sources.added_biosphere,
        *gas.biosphere,
        *sources.water.biosphere,
        *sources.landfill.biosphere,
        *municipal.biosphere,
        *reference.open_burning.biosphere,
        *reference.dump.biosphere,
        recycling.evaporated_water,
    ]
    operations = [
        sources.hwi,
        sources.fixed,
        sources.ash,
        gas.chp,
        gas.furnace,
        sources.landfill.landfill,
        recycling.preparation,
        recycling.finishing,
        reference.municipal.incineration,
        reference.municipal.bottom_ash.operation,
        reference.municipal.residual_ash.operation,
        *(site.placement for site in reference.dump.sites),
    ]
    for calibrated, dated in (
        (reference.water, sources.water),
        (reference.landfill, sources.landfill),
    ):
        operations.extend(
            (
                calibrated.sludge.operation,
                calibrated.landfarming.operation,
                calibrated.bottom_ash.operation,
                calibrated.residual_ash.operation,
                dated.collection,
                dated.wastewater,
                dated.sludge_incineration_market,
                dated.sludge_landfarming_market,
                dated.sludge_incineration,
                dated.sludge_landfarming,
                dated.bottom_ash_market,
                dated.residual_ash_market,
                dated.bottom_ash_treatment,
                dated.residual_ash_treatment,
            )
        )
    flows.extend(
        row.endpoint
        for operation in operations
        for row in operation.rows
        if isinstance(row.endpoint, PaReceivingFlow)
    )
    native_by_key = {endpoint.key: endpoint for endpoint in native}
    flows_by_key = {endpoint.key: endpoint for endpoint in flows}
    return (
        tuple(native_by_key[key] for key in sorted(native_by_key)),
        tuple(flows_by_key[key] for key in sorted(flows_by_key)),
    )
