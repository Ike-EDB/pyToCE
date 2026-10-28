"""Construct PTT chemical steam by replacing only the native heating duty."""

import json
from math import fsum
from typing import cast

from .providers.catalogue import provider_service_code
from .providers.recipe_context import (
    _copied_native_exchange,
    _RecipeContext,
    _scaled_native_exchange,
)


def ptt_steam_dataset(context: _RecipeContext) -> dict[str, object]:
    """Return one kg of PTT steam with native operation and biogas heating.

    The original converter for the rest of the world supplies q MJ/kg. The
    steam producer supplies H MJ heating/MJ steam; its non-heat amounts and
    uncertainty remain on that same fixed reference. Only q*H is
    supplied by the biogas-engine heat function. Technosphere inputs link to
    the ordinary dated suppliers, retaining their complete upstream functions.
    Original biosphere identities are retained; their amounts are normalized
    to one kilogram of steam.

    The prepared steam inventory leaves the original native converter,
    steam producer and supplying inventories unchanged. Original quantities
    and their sources are recorded separately from the dated suppliers.
    """
    service = "ptt_renewable_steam"
    converter = context.source.activity(
        {
            "name": "steam production, in chemical industry",
            "reference_product": "steam, in chemical industry",
            "unit": "kilogram",
            "exact_location": "RoW",
        },
        year=2025,
    )
    conversion_rows = context.exchanges(converter)
    reference = next(row for row in conversion_rows if str(row["type"]) == "production")
    energy = next(row for row in conversion_rows if str(row["type"]) == "technosphere")
    # The source converter contains production and its MJ input. Its 2.75 MJ/kg ratio defines the
    # energy-carrier conversion.
    q = float(energy["amount"]) / float(reference["amount"])
    steam = context.source.activity(
        {
            "name": "steam production, as energy carrier, in chemical industry",
            "reference_product": "heat, from steam, in chemical industry",
            "unit": "megajoule",
            "exact_location": "RoW",
        },
        year=2025,
    )
    source_rows = context.exchanges(steam)
    production = next(
        float(row["amount"]) for row in source_rows if str(row["type"]) == "production"
    )
    # The selected native producer's MJ inputs are its three industrial-furnace heat suppliers (gas,
    # oil and coal). Electricity is kWh, makeup is kg. Read all three delivered-heat amounts.
    heating = [
        row
        for row in source_rows
        if str(row["type"]) == "technosphere" and str(context.input(row)["unit"]) == "megajoule"
    ]
    heat_keys = {cast(tuple[str, str], row["input"]) for row in heating}
    h = fsum(float(row["amount"]) for row in heating) / production
    biogas, heat_trace = context.activity(
        {
            "name": "heat and power co-generation, biogas, gas engine",
            "reference_product": "heat, central or small-scale, other than natural gas",
            "unit": "megajoule",
            "exact_location": "RoW",
        },
        source_role="native_ecoinvent_provider",
        service_id=service,
        component_id="complete_biogas_engine_heat",
        semantic_class="ptt_renewable_heat_supply",
    )
    key = (context.provider_database, provider_service_code(service))
    output = _scaled_native_exchange(reference, 1.0 / float(reference["amount"]))
    output["input"] = key
    # Preserve the original water, electricity, releases and other non-heat
    # duties on the converter's kg basis, with their source metadata and order.
    retained = []
    for index, row in enumerate(source_rows):
        if str(row["type"]) == "production" or row["input"] in heat_keys:
            continue
        scaled = _scaled_native_exchange(row, q / production)
        if str(row["type"]) == "technosphere":
            original_supplier = context.input(row)
            supplier, trace = context.activity(
                {
                    "name": str(original_supplier["name"]),
                    "reference_product": str(original_supplier["reference product"]),
                    "unit": str(original_supplier["unit"]),
                    "exact_location": str(original_supplier["location"]),
                },
                source_role="native_ecoinvent_provider",
                service_id=service,
                component_id=f"complete_non_heat_supply_{index}",
                semantic_class="ptt_complete_non_heat_supply",
            )
            scaled["input"] = cast(tuple[str, str], supplier.key)
            trace["project_terms_json"] = json.dumps(
                {
                    "original_steam_key": steam.key,
                    "original_exchange": _copied_native_exchange(row),
                    "fixed_reference_multiplier": q / production,
                    "dated_supply_exchange": scaled,
                },
                sort_keys=True,
            )
        retained.append(scaled)
    heat = {"input": cast(tuple[str, str], biogas.key), "amount": q * h, "type": "technosphere"}
    quantities = {
        "basis": "per kg PTT chemical steam",
        "native_conversion_mj_per_kg": q,
        "native_heating_mj_per_mj": h,
        "biogas_heating_mj_per_kg": q * h,
        "conversion_source_key": cast(tuple[str, str], converter.key),
        "complete_steam_source_key": cast(tuple[str, str], steam.key),
        "complete_source_reference_mj": production,
        "complete_non_heat_reference_multiplier": q / production,
    }
    dataset = {field: value for field, value in dict(converter).items() if field != "id"}
    dataset.update(
        {
            "database": key[0],
            "code": key[1],
            "provider_service_id": service,
            "name": "PTT chemical steam production, biogas heat",
            "exchanges": [output, *retained, heat],
            "source_balance_outputs": quantities,
            "comment": str(converter.get("comment", ""))
            + "\nPTT steam uses biogas-engine heat. The native RoW conversion q "
            "(MJ/kg steam) and heating requirement H (MJ/MJ steam) give q*H MJ "
            "of biogas heat per kg steam. Other native steam inputs are scaled "
            "by q and use the corresponding dated suppliers. Native steam source: "
            + str(steam.get("comment", "")),
        }
    )
    # These original records define quantities. Keep them in the replacement's reference terms,
    # separately from each actual dated supplying activity and its complete native rows.
    heat_trace["project_terms_json"] = json.dumps(
        {
            **quantities,
            "native_conversion_exchange": _copied_native_exchange(energy),
            "native_reference_kg": float(reference["amount"]),
            "original_heating_exchanges": [_copied_native_exchange(row) for row in heating],
            "original_non_heat_exchanges": [
                _copied_native_exchange(row)
                for row in source_rows
                if str(row["type"]) != "production" and row["input"] not in heat_keys
            ],
            "scaled_complete_non_heat_exchanges": retained,
            "replacement_heat_exchange": heat,
            "water_boundary": "Preserve every native water row at q per MJ of source operation",
        },
        sort_keys=True,
    )
    return dataset
