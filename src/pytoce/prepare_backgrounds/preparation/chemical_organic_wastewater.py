"""Connect complete Swiss treatment for chemical-manufacturing organic wastewater.

The actual chemical owner supplies its elements, carbon origins, water,
solution volume and qualified analytical indicators. This owner joins
Doka's municipal co-treatment, gas, sludge, FLUWA and ash functions on that same
reference. The chemical wastewater service treats the declared organic load, while the hydraulic
nitrate-water service retains its separate role.
"""

from typing import cast

from .providers.ash_treatment import _ash_dataset, _ash_market_dataset, _AshReleaseProfile
from .providers.catalogue import provider_service_code
from .providers.leachate_load import _CH_WASTEWATER_PROFILE, _LeachateLoad, _wastewater_load
from .providers.leachate_treatment import (
    _leachate_market_dataset,
    _sludge_market_dataset,
    _wastewater_dataset,
)
from .providers.recipe_context import _RecipeContext
from .providers.sludge_landfarming import _landfarming_dataset
from .providers.swiss_sludge_incineration import swiss_sludge_dataset, swiss_sludge_load
from .providers.waste_gas import _wastewater_gas
from .providers.water_treatment import _water_evaporation


def chemical_organic_wastewater_datasets(
    context: _RecipeContext,
    *,
    service_id: str,
    load: _LeachateLoad,
    solution_volume_m3: float,
    water_kg: float,
    complete_carbon_oxidation: bool,
    phosphorus_equilibrium: bool,
) -> dict[tuple[str, str], dict[str, object]]:
    """Build one complete CH receiving chain in the dated provider frontier.

    Inputs share the chemical caller's reference, such as one kg PA phosphate-ester
    product. Volume is total solution m3; water is solvent kg, kept
    separate from dissolved solutes. Native negative production and signed
    collection-to-plant ratios remain. The caller demands minus the solution volume; Brightway
    applies the source reference production once.

    All ten records have fixed catalog owners and share one actual load.
    Original signed collection ratios and treatment coefficients
    remain fixed while complete supporting suppliers follow the dated
    background. Native source activities are not modified.
    The selected municipal H/O, sludge and biological indicators retain their
    empirical meaning.

    The caller declares its landfarming reference: PDO uses complete carbon
    oxidation and phosphorus equilibrium; the PA phosphate wash preserves its
    previously selected partial-carbon and source-phosphorus treatment.
    """
    services = {
        str(item["provider_service_id"]): item
        for item in cast(list[dict[str, object]], context.catalogue["services"])
    }
    collection = services[service_id]
    collection_construction = cast(dict[str, object], collection["construction"])
    plant_id = str(cast(dict[str, object], collection_construction["plant"])["service_id"])
    plant = services[plant_id]
    construction = cast(dict[str, object], plant["construction"])
    original_plant = context.source.activity(
        cast(dict[str, object], construction["source"]), year=2025
    )
    profile = _CH_WASTEWATER_PROFILE
    wastewater = _wastewater_load(
        load,
        volume_m3=solution_volume_m3,
        incoming_water_kg=water_kg,
        evaporation_fraction=_water_evaporation(str(original_plant["comment"]), profile.hydraulic),
        profile=profile,
    )
    gas = _wastewater_gas(
        context, construction, wastewater, plant_id, profile, original_physical_reference=True
    )
    incineration_market_id = str(
        cast(dict[str, object], construction["incineration"])["service_id"]
    )
    landfarming_market_id = str(cast(dict[str, object], construction["landfarming"])["service_id"])
    incineration_market = services[incineration_market_id]
    landfarming_market = services[landfarming_market_id]
    incineration_construction = cast(dict[str, object], incineration_market["construction"])
    landfarming_construction = cast(dict[str, object], landfarming_market["construction"])
    incineration_id = str(
        cast(dict[str, object], incineration_construction["treatment"])["service_id"]
    )
    landfarming_id = str(
        cast(dict[str, object], landfarming_construction["treatment"])["service_id"]
    )
    thermal_service = services[incineration_id]
    thermal_construction = cast(dict[str, object], thermal_service["construction"])
    thermal = swiss_sludge_load(context, thermal_construction, wastewater)
    database = context.provider_database
    records = {
        service_id: _leachate_market_dataset(
            context, collection, load, wastewater, database, original_physical_reference=True
        ),
        plant_id: _wastewater_dataset(
            context,
            plant,
            load,
            wastewater,
            gas,
            database,
            profile,
            original_physical_reference=True,
        ),
        incineration_market_id: _sludge_market_dataset(
            context, incineration_market, wastewater, database, original_physical_reference=True
        ),
        landfarming_market_id: _sludge_market_dataset(
            context, landfarming_market, wastewater, database, original_physical_reference=True
        ),
        incineration_id: swiss_sludge_dataset(context, thermal_service, thermal, database),
        landfarming_id: _landfarming_dataset(
            context,
            services[landfarming_id],
            wastewater,
            service_id,
            database,
            complete_carbon_oxidation=complete_carbon_oxidation,
            phosphorus_equilibrium=phosphorus_equilibrium,
        ),
    }
    # Swiss ash retains its original source-element functions. The thermal calculation separates
    # FLUWA matrix, hydroxide, and residue.
    for phase, ash in (("bottom_ash", thermal.bottom_ash), ("residual_ash", thermal.residual_ash)):
        market_id = str(cast(dict[str, object], thermal_construction[phase])["service_id"])
        market = services[market_id]
        ash_construction = cast(dict[str, object], market["construction"])
        terminal_id = str(cast(dict[str, object], ash_construction["treatment"])["service_id"])
        records[market_id] = _ash_market_dataset(
            context, market, ash, database, original_physical_reference=True
        )
        records[terminal_id] = _ash_dataset(
            context,
            services[terminal_id],
            ash,
            database,
            release_profile=_AshReleaseProfile(()),
        )
    return {
        (database, provider_service_code(sid)): {
            **dataset,
            "database": database,
            "code": provider_service_code(sid),
        }
        for sid, dataset in records.items()
    }
