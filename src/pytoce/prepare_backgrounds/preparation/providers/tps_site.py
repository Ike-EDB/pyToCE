"""Compile the sanitary-landfill site for actual TPS processing residues.

Thermoplastic starch (TPS) manufacturing and mechanical-recovery residues retain
their original formulation, including carried water and synthetic-starch
auxiliaries. The selected RoW site supplies its complete infrastructure and
operating functions. Actual component fates replace municipal waste chemistry;
its leachate, gas and sludge descendants have their own calculation owners.
"""

import json
from dataclasses import asdict
from math import fsum
from typing import cast

from .catalogue import provider_service_code
from .leachate_load import _WastewaterLoad
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _copied_native_exchange, _RecipeContext, _scaled_native_exchange
from .tps_residue_load import _TPSResidueLoad
from .waste_gas import _GasTreatment

_ELEMENT_NAMES = {
    "C": "carbon",
    "H": "hydrogen",
    "O": "oxygen",
    "N": "nitrogen",
    "S": "sulfur",
    "P": "phosphorus",
    "Na": "sodium",
    "Mg": "magnesium",
    "Zn": "zinc",
    "Cl": "chlorine",
}
# Doka Landfill2020, MSWLF calculation C89:C92: mg S/L in sulfate/sulfide. D90/D91 normalize this
# fixed source speciation.
_SULFATE_SHARE = 40.00326505167857 / (40.00326505167857 + 2.9443208588856526)
_SULFIDE_SHARE = 2.9443208588856526 / (40.00326505167857 + 2.9443208588856526)


def _tps_landfill_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    residue: _TPSResidueLoad,
    wastewater: _WastewaterLoad,
    gas: _GasTreatment,
    database_name: str,
) -> dict[str, object]:
    """Treat actual wet TPS residue while retaining the native negative reference.

    Inputs are per one kg principal glucan/glycerol before any loss or treatment
    share. The parent supplies the actual signed wet residue quantity; Brightway
    divides by this site's negative production once. Initial residue haulage is
    a separate complete freight service, not supplied again inside the site.

    The gas owner already binds its species, reaction water/oxygen and support.
    This site binds its heat and adds consuming-meter electricity heat once.
    The later protein-sulfur speciation follows its declared phase reference. Initial placement
    indicators
    exclude the sludge/ash descendants that have their own disposal indicators.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    collection_id = str(cast(dict[str, object], construction["leachate"])["service_id"])
    _service_trace(context, service)
    activity, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="TPS_actual_residue_landfill",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    p = _self_production_amount(activity, native, method="TPS residue landfill")
    scale = -p / residue.wet_mass_kg
    dry = fsum(item.mass_kg for item in residue.components)
    properties: dict[str, object] = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": dry / residue.wet_mass_kg, "unit": "kg"},
        "water in wet mass": {
            "amount": residue.water_kg / residue.wet_mass_kg,
            "unit": "kg",
        },
        "carbon allocation": {
            "amount": residue.elements_kg["C"] / residue.wet_mass_kg,
            "unit": "kg",
        },
    }
    properties.update(
        {
            f"{_ELEMENT_NAMES[element]} content": {
                "amount": mass / dry,
                "unit": "dimensionless",
            }
            for element, mass in residue.elements_kg.items()
        }
    )
    for origin in ("fossil", "non-fossil"):
        carbon = fsum(
            pool.input_carbon_kg
            for pool in residue.carbon
            if (pool.origin == "fossil") == (origin == "fossil")
        )
        properties[f"carbon content, {origin}"] = {
            "amount": carbon / dry,
            "unit": "dimensionless",
        }
    production = next(ex for ex in native if ex["type"] == "production")
    reference = _copied_native_exchange(production)
    for field in ("flow", "activity", "production volume", "classifications"):
        reference.pop(field, None)
    reference.update(
        input=(database_name, provider_service_code(service_id)),
        name=str(service["reference_product"]),
        properties=properties,
        comment="Actual wet TPS formulation residue; no municipal waste composition is retained.",
    )
    generated = [reference]
    technosphere = [ex for ex in native if ex["type"] == "technosphere"]
    leachate = next(
        ex
        for ex in technosphere
        if context.input(ex)["reference product"] == "leachate, SLF, municipal solid waste"
    )
    collection = _scaled_native_exchange(
        leachate, -scale * wastewater.volume_m3 / float(leachate["amount"])
    )
    for field in ("flow", "activity", "properties"):
        collection.pop(field, None)
    collection.update(
        input=(database_name, provider_service_code(collection_id)),
        name=collection_id,
    )
    generated.append(collection)
    retained = [ex for ex in technosphere if ex is not leachate]
    generated.extend(_copied_native_exchange(ex) for ex in retained)
    generated.extend(_scaled_native_exchange(ex, scale) for ex in gas.biosphere_exchanges)
    # Original rows501876:501936 serve site administration. They
    # are independent of gas-station generation/pumping and remain purchased.
    admin_electricity = fsum(
        float(ex["amount"])
        for ex in retained
        if context.input(ex)["reference product"] == "electricity, low voltage"
    )
    # Doka2003 partII3.2.10 places electricity heat at its consuming owner: 3.6 MJ/kWh once for site
    # administration. Returned gas heat already subtracts net electricity export. Purchased
    # administrative heat remains with its complete native producer under the site convention.
    heat_air = scale * gas.heat_air_mj + 3.6 * admin_electricity
    later_s = residue.protein_sulfur.later_leachate_kg
    # MSWLF G90/G91 and L90/L91: 96/32 kg SO4 and34/32 kg H2S per kg S.
    # The associated O/H are descriptors; they add no oxidation reaction,
    # water purchase or heat duty.
    sulfate = 3 * _SULFATE_SHARE * later_s
    sulfide = (34 / 32) * _SULFIDE_SHARE * later_s
    initial = {
        "Organic carbon, placed in landfill": scale * residue.elements_kg["C"],
        "Waste mass, total, placed in landfill": -p,
        "Non-hazardous waste disposed": -p,
    }
    retained_land = []
    for ex in native:
        if ex["type"] != "biosphere":
            continue
        flow = context.input(ex)
        name = str(flow["name"])
        categories = tuple(flow["categories"])
        if name.startswith(("Occupation,", "Transformation,")):
            generated.append(_copied_native_exchange(ex))
            retained_land.append(ex)
            continue
        if name in initial:
            # air & energy B204:B213 includes possible higher-order sludge/ash
            # terms. Reconstruct initial TPS placement independently; our explicit
            # children own their own indicators.
            amount = initial[name]
        elif name == "Heat, waste" and categories[0] == "air":
            amount = heat_air
        elif name == "Heat, waste" and categories[0] == "soil":
            amount = scale * gas.heat_soil_mj
        elif categories == ("water", "ground-, long-term") and name == "Sulfate":
            amount = scale * sulfate
        elif categories == ("water", "ground-, long-term") and name == "Hydrogen sulfide":
            amount = scale * sulfide
        else:
            # Actual gas and short leachate have separate owners. Principal and
            # protein carbon already exhaust the admitted phase-integrated fate;
            # no municipal carbon/heat tail or absent donor metal is copied.
            continue
        if amount != 0:
            row = _scaled_native_exchange(ex, amount / float(ex["amount"]))
            row.pop("properties", None)
            generated.append(row)
    trace["retained_technosphere_json"] = context.exchange_json(retained)
    trace["removed_technosphere_json"] = context.exchange_json([leachate])
    trace["retained_biosphere_json"] = context.exchange_json(retained_land)
    trace["removed_biosphere_json"] = context.exchange_json(
        ex for ex in native if ex["type"] == "biosphere" and ex not in retained_land
    )
    trace["project_terms_json"] = json.dumps(
        {
            "residue_per_principal_kg": asdict(residue),
            "leachate_m3_per_principal_kg": wastewater.volume_m3,
            "gas": gas.trace,
            "administration_kwh_per_native_reference": admin_electricity,
            "later_sulfur_kg_per_principal_kg": later_s,
            "later_sulfate_kg_per_principal_kg": sulfate,
            "later_hydrogen_sulfide_kg_per_principal_kg": sulfide,
            "later_sulfate_oxygen_descriptor_kg": 2 * _SULFATE_SHARE * later_s,
            "later_sulfide_hydrogen_descriptor_kg": (2 / 32) * _SULFIDE_SHARE * later_s,
            "placement_indicators": "Initial site only; each actual sludge/ash child owns itself.",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }
