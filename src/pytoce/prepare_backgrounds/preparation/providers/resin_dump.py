"""Compile actual cured polymers and auxiliaries at five unmanaged dump sites.

Each site supplies original climate, methane correction and native placement
rows. The actual cured material supplies its atoms and carbon origins. There is
no sanitary collection, gas capture, wastewater treatment or operating supplier.
"""

import json
import re
from dataclasses import asdict
from typing import cast

from .catalogue import provider_service_code
from .epoxy_landfill_load import _epoxy_landfill_load
from .landfill_phases import _dump_discharges, _LandfillSite
from .leachate_load import _CarbonOrigin
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _RecipeContext, _scaled_native_exchange
from .resin_landfill import _resin_disposal_reference
from .road_landfill_load import _road_landfill_load
from .sbr_landfill_load import _sbr_dump_load
from .treatment_compilation import _direct_principal_carbon_owner
from .upr_landfill_load import _upr_landfill_load


def _resin_dump_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Receive one principal kg plus its actual cured auxiliaries in one selected dump.

    The five original MIXED selectors retain their own climates and initial
    land records. Subsequent source dates only relink unchanged flow identities;
    donor plastic composition and its .153 kg moisture are not transferred.
    """
    construction = cast(dict[str, object], service["construction"])
    material_name = str(construction["kind"]).split("_", 1)[0]
    selector = cast(dict[str, object], construction["source"])
    origin = cast(_CarbonOrigin, construction["principal_carbon_origin"])
    service_id = str(service["provider_service_id"])
    original = context.source.activity(selector, year=2025)
    climate = {
        name: float(value)
        for name, value in re.findall(r"\((MAT|MAP)\) of ([0-9.]+)", str(original["comment"]))
    }
    # Landfill2020 control board E133/E135: 2.0604m aerobic top layer in these 7m dumps. MCF alters
    # CH4/CO2 formation.
    site = _LandfillSite(climate["MAT"], climate["MAP"], 1 - 2.0604 / 7)
    if material_name == "sbr":
        load = _sbr_dump_load(origin, site)
    elif material_name == "road":
        load = _road_landfill_load(
            next(
                row
                for row in context.material_recipe.conversion
                if row.operation == "others_road_marking_conversion"
            ),
            origin,
            site,
            carbonate_end_year=float(cast(float, construction["carbonate_end_year"])),
        )
    elif material_name == "upr":
        load = _upr_landfill_load(
            next(
                row
                for row in context.material_recipe.conversion
                if row.operation == "others_upr_hand_lay_up_conversion"
            ),
            origin,
            site,
            carbonate_end_year=float(cast(float, construction["carbonate_end_year"])),
        )
    else:
        load = _epoxy_landfill_load(origin, site)
    discharges = _dump_discharges(
        load, site, upward_leachate=cast(bool, construction["upward_leachate"])
    )
    _service_trace(context, service)
    activity, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id=f"actual_cured_{material_name}_open_dump",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    production = _self_production_amount(activity, native, method=f"{material_name} open dump")
    key = database_name, provider_service_code(service_id)
    generated = [
        _resin_disposal_reference(load, key, str(service["reference_product"]), material_name)
    ]
    land = [
        row
        for row in native
        if row["type"] == "biosphere"
        and tuple(context.input(row)["categories"]) == ("natural resource", "land")
    ]
    generated.extend(_scaled_native_exchange(row, load.wet_mass_kg / -production) for row in land)
    for emission in discharges.gas:
        target = context.biosphere.flow(
            {
                "name": emission.name,
                "unit": emission.unit,
                "categories": ["air", "non-urban air or from high stacks"],
            }
        )
        row: dict[str, object] = {
            "input": cast(tuple[str, str], target.key),
            "name": emission.name,
            "unit": emission.unit,
            "type": "biosphere",
            "amount": emission.amount,
            "loc": emission.amount,
            "uncertainty type": 0,
        }
        if emission.principal_carbon_origin_fractions is not None:
            row["principal_carbon_origin_fractions"] = emission.principal_carbon_origin_fractions
        generated.append(row)
    for (name, categories), amount in discharges.emissions.items():
        if amount == 0.0:
            continue  # An absent actual element or moisture has no emission.
        unit = (
            "cubic meter"
            if name == "Water"
            else "megajoule"
            if name == "Heat, waste"
            else "kilogram"
        )
        target = context.biosphere.flow(
            {"name": name, "unit": unit, "categories": list(categories)}
        )
        generated.append(
            {
                "input": cast(tuple[str, str], target.key),
                "name": name,
                "unit": unit,
                "type": "biosphere",
                "amount": amount,
                "loc": amount,
                "uncertainty type": 0,
            }
        )
    trace["retained_technosphere_json"] = "[]"
    trace["removed_technosphere_json"] = "[]"
    trace["retained_biosphere_json"] = context.exchange_json(land)
    trace["removed_biosphere_json"] = context.exchange_json(
        row for row in native if row["type"] == "biosphere" and row not in land
    )
    trace["project_terms_json"] = json.dumps(
        {
            "schema": f"pytoce_{material_name}_dump_reference_schema_v1",
            "actual_matrix_per_principal_kg": asdict(load),
            "site": asdict(site),
            "upward_leachate": construction["upward_leachate"],
            "native_production": production,
            "short_air_elements_kg": discharges.short_air_elements_kg,
            "later_air_elements_kg": discharges.later_air_elements_kg,
            "soil_elements_kg": discharges.soil_elements_kg,
            "dinitrogen_kg": discharges.dinitrogen_kg,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    data: dict[str, object] = {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }
    data["direct_treatment_carbon_owners_json"] = json.dumps(
        [_direct_principal_carbon_owner(data, database_name, origin, material_name)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return data
