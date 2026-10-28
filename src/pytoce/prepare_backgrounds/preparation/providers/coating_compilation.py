"""Compile actual coating application and its separately managed material losses.

The material recipe supplies gross principal resin separately. Epoxy application
uses complete diamine, acetone and air supplies with an alkyd mixing operation.
Road application uses complete manufactured auxiliaries and coated glass beads
with an acrylic-filler mixing operation. Actual managed residues enter their
selected hazardous-waste incineration (HWI) and ash functions; retained and
airborne material keep separate custody.
"""

import json
from dataclasses import asdict
from math import fsum
from typing import cast

from ..chemical_residue_incineration import chemical_residue_datasets
from ..chemical_residue_load import ChemicalResidueLoad
from .catalogue import provider_service_code
from .epoxy_material import _epoxy_reference, _epoxy_residue_load
from .leachate_load import _CarbonOrigin
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _RecipeContext, _scaled_native_exchange
from .road_material import _road_reference, _RoadMaterial


def epoxy_coating_datasets(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[tuple[str, str], dict[str, object]]:
    """Build application and managed treatment per kg retained principal GY250.

    Mixing rows retain source production, dated links and uncertainty under
    gross-binder scaling. Chemical and air supplies retain their native
    inventories. Material quantities use MXDA composition; HMDA supplies its
    manufacturing proxy. HWI and ash records are written with the application
    root to the dated provider frontier.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    conversion = next(
        item
        for item in context.material_recipe.conversion
        if item.operation == str(construction["operation"])
    )
    origin = cast(_CarbonOrigin, construction["principal_carbon_origin"])
    reference = _epoxy_reference(conversion, origin)
    _service_trace(context, service)
    key = database_name, provider_service_code(service_id)
    rows: list[dict[str, object]] = [{"input": key, "type": "production", "amount": 1.0}]
    supplies = cast(dict[str, dict[str, object]], construction["supplies"])
    for label, amount in reference.supply_amounts.items():
        supplier, trace = context.activity(
            supplies[label],
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id=label,
            semantic_class="source_backed_portfolio",
        )
        rows.append(
            {
                "input": cast(tuple[str, str], supplier.key),
                "type": "technosphere",
                "amount": amount,
                "unit": str(supplier["unit"]),
            }
        )
        trace["project_terms_json"] = json.dumps(
            {"amount_per_retained_principal_kg": amount, "unit": supplier["unit"]},
            sort_keys=True,
        )

    mixing, trace = context.activity(
        cast(dict[str, object], construction["mixing_source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="complete_local_mixing_and_facility",
        semantic_class="operation_only_extraction",
    )
    native = context.exchanges(mixing)
    production = _self_production_amount(mixing, native, method="epoxy coating mixing")
    retained = []
    for exchange in native:
        if exchange["type"] == "production":
            continue
        node = context.input(exchange)
        keep = (
            str(node["reference product"])
            in {"chemical factory, organics", "electricity, medium voltage", "nitrogen, liquid"}
            if exchange["type"] == "technosphere"
            else str(node["name"]) == "Nitrogen"
        )
        if keep:
            retained.append(exchange)
            # Alkyd's native dry-output kg defines the operation. Keep .07133 kWh, .019 kg N2 and
            # 4e-10 factory-unit source rows with the dated vector and uncertainty, per gross mixed
            # binder kg.
            rows.append(_scaled_native_exchange(exchange, reference.gross_binder_kg / production))
    trace["retained_technosphere_json"] = context.exchange_json(
        row for row in retained if row["type"] == "technosphere"
    )
    trace["retained_biosphere_json"] = context.exchange_json(
        row for row in retained if row["type"] == "biosphere"
    )
    trace["removed_technosphere_json"] = context.exchange_json(
        row for row in native if row["type"] == "technosphere" and row not in retained
    )
    trace["removed_biosphere_json"] = context.exchange_json(
        row for row in native if row["type"] == "biosphere" and row not in retained
    )
    trace["project_terms_json"] = json.dumps(
        {
            "gross_mixed_binder_kg_per_retained_principal_kg": reference.gross_binder_kg,
            "native_dry_output_kg": production,
            "operation_multiplier": reference.gross_binder_kg / production,
        },
        sort_keys=True,
    )

    incineration_key = database_name, f"epoxy_coating__{service_id}__managed_incineration"
    ash_key = database_name, f"epoxy_coating__{service_id}__managed_ash"
    load = _epoxy_residue_load(reference.managed_residue)
    datasets = chemical_residue_datasets(
        context,
        load,
        incineration_key=incineration_key,
        ash_key=ash_key,
        residue_name=f"epoxy preparation and settled overspray, {origin}",
    )
    # Complete external HWI produces -1 kg; collection is already included in that function. Only
    # preparation+settled mass is demanded.
    rows.append(
        {
            "input": incineration_key,
            "type": "technosphere",
            "amount": -load.wet_mass_kg,
            "unit": "kilogram",
        }
    )
    datasets[key] = {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": rows,
        "epoxy_reference_schema": "pytoce_epoxy_coating_reference_schema_v1",
        "epoxy_reference_per_retained_principal_kg": asdict(reference),
        "comment": (
            "GY250/MXDA application per kg retained principal resin. Principal resin supply "
            "and manufacturing exits belong to the material recipe. RER HMDA supply is an "
            "output-mass proxy for fossil MXDA. Alkyd mixing and its facility are scaled to "
            "gross binder throughput. Pump air uses hydraulic work and standard temperature. "
            "Cleanup acetone, preparation residue and settled overspray receive external HWI "
            "and ash treatment. Airborne overspray leaves the inventory as uncharacterized "
            "material. Retained hardener stays in the cured article for subsequent "
            "treatment."
        ),
    }
    return datasets


def _road_residue_load(material: _RoadMaterial) -> ChemicalResidueLoad:
    """Pass road-container residue to the selected direct HWI function.

    Mineral elements use HWI average transfers. Geological carbonate C stays
    outside organic gases and heat. Terminal treatment uses oxide-equivalent
    service mass.
    """
    elements = {
        element: fsum(
            (
                material.burnable_elements_kg.get(element, 0.0),
                material.mineral_elements_kg.get(element, 0.0),
                material.limestone_elements_kg.get(element, 0.0) if element != "C" else 0.0,
            )
        )
        for element in (
            material.burnable_elements_kg.keys()
            | material.mineral_elements_kg.keys()
            | material.limestone_elements_kg.keys()
        )
    }
    # The declared road ingredients contain no nitrogen. Doka's nominal
    # H1/O16 convention gives 9 kg newly formed water per kg organic hydrogen.
    # Harvey (1998), NISTIR 5078 Table 1: 2441.7 kJ/kg / 1000 gives the
    # 2.4417 MJ/kg vaporization deduction at 25 C for that water and moisture.
    # Bulk minerals supply no oxidation heat or additional calcination duty.
    elements["N"] = 0.0
    formed_water = 9 * elements["H"]
    return ChemicalResidueLoad(
        wet_mass_kg=material.mass_kg,
        carrier_water_kg=material.water_kg,
        active_elements_kg=elements,
        net_energy_mj=material.heat_potential_mj - 2.4417 * (formed_water + material.water_kg),
        nonfuel_salt_elements_kg={},
        waste_formed_water_kg=formed_water,
        biogenic_carbon_kg=material.principal_carbon_kg[1],
        principal_carbon_kg=material.principal_carbon_kg,
        geological_carbon_kg=material.geological_carbon_kg,
    )


def road_coating_datasets(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[tuple[str, str], dict[str, object]]:
    """Compile road application per kg retained principal PMMA-equivalent binder.

    The material recipe separately supplies gross principal resin. This owner
    supplies the actual auxiliaries, bead manufacturing and premix operation,
    treats the container withdrawal before MMA evaporation, and emits the
    source MMA quantity through the admitted existing NMVOC identity. The
    retained article has its own later receiving owner.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    conversion = next(
        item
        for item in context.material_recipe.conversion
        if item.operation == str(construction["operation"])
    )
    origin = cast(_CarbonOrigin, construction["principal_carbon_origin"])
    reference = _road_reference(conversion, origin)
    _service_trace(context, service)
    key = database_name, provider_service_code(service_id)
    rows: list[dict[str, object]] = [{"input": key, "type": "production", "amount": 1.0}]
    supplies = cast(dict[str, dict[str, object]], construction["supplies"])
    for label, amount in reference.supply_amounts.items():
        selected = supplies[label]
        if "service_id" in selected:
            # Complete bead manufacturing already owns its furnace, coating,
            # mixing and upstream suppliers. Buy it once on gross wet bead kg.
            supplier_key = database_name, provider_service_code(str(selected["service_id"]))
        else:
            supplier, trace = context.activity(
                cast(dict[str, object], selected["selector"]),
                source_role="native_ecoinvent_provider",
                service_id=service_id,
                component_id=label,
                semantic_class="source_backed_portfolio",
            )
            supplier_key = cast(tuple[str, str], supplier.key)
            trace["project_terms_json"] = json.dumps(
                {"amount_per_retained_principal_kg": amount, "unit": supplier["unit"]},
                sort_keys=True,
            )
        rows.append(
            {"input": supplier_key, "type": "technosphere", "amount": amount, "unit": "kilogram"}
        )
    # The selected acrylic-filler operation supplies premix mixing and its existing site duties.
    mixing_id = str(cast(dict[str, object], construction["mixing"])["service_id"])
    rows.append(
        {
            "input": (database_name, provider_service_code(mixing_id)),
            "type": "technosphere",
            "amount": reference.mixing_kg,
            "unit": "kilogram",
        }
    )
    flow = context.biosphere.flow(cast(dict[str, object], construction["mma_air_proxy"]))
    rows.append(
        {
            "input": cast(tuple[str, str], flow.key),
            "type": "biosphere",
            "amount": reference.methyl_methacrylate_air_kg,
            "unit": "kilogram",
        }
    )
    incineration_key = database_name, f"road_coating__{service_id}__container_incineration"
    ash_key = database_name, f"road_coating__{service_id}__container_ash"
    load = _road_residue_load(reference.container_residue)
    datasets = chemical_residue_datasets(
        context,
        load,
        incineration_key=incineration_key,
        ash_key=ash_key,
        residue_name=f"road-marking container residue, {origin}",
        include_collection=False,
    )
    # Direct CH HWI has a -1 kg reference and includes its operating and
    # terminal treatment chain. No incoming collection haulage is added.
    rows.append(
        {
            "input": incineration_key,
            "type": "technosphere",
            "amount": -load.wet_mass_kg,
            "unit": "kilogram",
        }
    )
    datasets[key] = {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": rows,
        "road_reference_schema": "pytoce_road_coating_reference_schema_v1",
        "road_reference_per_retained_principal_kg": asdict(reference),
        "comment": (
            "Cold-plastic road-marking application and pre-evaporation container residue per "
            "retained principal kg. Manufactured auxiliaries and wet coated beads are "
            "supplied once; principal resin is supplied separately. Acrylic-filler mixing "
            "provides the operation and site proxy. MMA emissions use the urban-air NMVOC "
            "flow and its EF factors. Direct CH HWI supplies operating and terminal "
            "treatment without incoming collection haulage. Geological carbonate carbon is "
            "recorded separately from organic-carbon fates."
        ),
    }
    return datasets
