"""Bind composition-specific uncontrolled burning to native elementary flows."""

import json
from math import fsum
from typing import cast

from .catalogue import provider_service_code
from .epoxy_material import _EPOXY_HARDENER_RATIO, _epoxy_material
from .leachate_load import _CarbonOrigin
from .native_recipe import _service_trace
from .open_burning import _open_burning_load, _OpenBurningMaterial
from .polyester_material import _pbat_material, _pbs_material, _ptt_material
from .recipe_context import _RecipeContext
from .road_material import _road_material
from .sbr_material import _sbr_compound_material
from .tps_burning_material import _tps_burning_material
from .tps_residue_load import _tps_material_load
from .treatment_compilation import _direct_carbon_owners_json
from .upr_material import _upr_material

_ORIGINS: tuple[_CarbonOrigin, ...] = ("fossil", "biogenic", "atmospheric_dac")


def _open_burning_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Bind actual plastic burning to existing ecoinvent elementary flows.

    Production is positive one kg principal treated. The original native mixed-plastic activity
    supplies the source reference. The foreground burden calculation applies regional fate shares
    and historical formulation quantities.
    Calculated gas emissions include auxiliary carbon;
    the separate principal fractions serve only the carbon diagnostic.
    """
    service_id = str(service["provider_service_id"])
    key = (database_name, provider_service_code(service_id))
    construction = cast(dict[str, object], service["construction"])
    if construction["kind"] == "tps_open_burning":
        formulation_id = str(construction["formulation_service_id"])
        material = _tps_burning_material(_tps_material_load(services, formulation_id))
        material_terms: dict[str, object] = {"formulation_service_id": formulation_id}
        origin_status = "actual_tps_principal_origin"
        exact_once = (
            "One historical TPS mixture per principal kg; source activity is not purchased."
        )
    elif construction["kind"] == "upr_open_burning":
        origin = cast(_CarbonOrigin, construction["principal_carbon_origin"])
        conversion = next(
            row
            for row in context.material_recipe.conversion
            if row.operation == "others_upr_hand_lay_up_conversion"
        )
        composite = _upr_material(
            conversion,
            cast(tuple[float, float, float], tuple(float(origin == item) for item in _ORIGINS)),
        )
        # Glass and fumed silica remain inert; sizing and dispersed Co belong
        # to the burnable material. The nominal/tracked principal-C difference
        # keeps its actual origin but is outside the principal diagnostic.
        auxiliary_carbon = dict(zip(_ORIGINS, composite.principal_carbon_bridge_kg, strict=True))
        auxiliary_carbon["fossil"] += composite.fossil_auxiliary_carbon_kg
        material = _OpenBurningMaterial(
            composite.mass_kg,
            composite.water_kg,
            composite.burnable_elements_kg,
            composite.inert_elements_kg,
            dict(zip(_ORIGINS, composite.principal_carbon_kg, strict=True)),
            auxiliary_carbon,
            composite.heat_potential_mj,
        )
        material_terms = {
            "principal_carbon_origin": origin,
            "actual_components_kg": composite.components_kg,
            "principal_reference_difference_kg": composite.principal_carbon_bridge_kg,
        }
        origin_status = "actual_upr_principal_origin"
        exact_once = (
            "One kg retained principal with actual glass, silica and cured auxiliaries; "
            "uncontrolled burning and on-site residues without another material purchase "
            "or an added ash disposal service."
        )
    elif construction["kind"] == "road_open_burning":
        origin = cast(_CarbonOrigin, construction["principal_carbon_origin"])
        conversion = next(
            row
            for row in context.material_recipe.conversion
            if row.operation == "others_road_marking_conversion"
        )
        road = _road_material(
            conversion,
            cast(tuple[float, float, float], tuple(float(origin == item) for item in _ORIGINS)),
        )
        # Retained fillers stay outside the burning matrix, whereas the PDMS coating's Si follows
        # its organic phase. Limestone C is a separate source-model exclusion.
        inert = {
            element: fsum(
                (
                    road.mineral_elements_kg.get(element, 0.0),
                    road.limestone_elements_kg.get(element, 0.0),
                )
            )
            for element in road.mineral_elements_kg.keys() | road.limestone_elements_kg.keys()
            if element != "C"
        }
        material = _OpenBurningMaterial(
            wet_mass_kg=road.mass_kg,
            water_kg=road.water_kg,
            burnable_elements_kg=road.burnable_elements_kg,
            inert_elements_kg=inert,
            principal_carbon_kg=dict(zip(_ORIGINS, road.principal_carbon_kg, strict=True)),
            auxiliary_carbon_kg={"fossil": road.fossil_auxiliary_carbon_kg},
            heat_potential_mj=road.heat_potential_mj,
            geological_carbon_kg=road.geological_carbon_kg,
        )
        material_terms = {
            "principal_carbon_origin": origin,
            "actual_components_kg": road.components_kg,
            "geological_carbon_limitation": (
                "Carbonate carbon is recorded outside the source organic-carbon fate. "
                "No calcination, soil release or permanent storage is inferred."
            ),
        }
        origin_status = "actual_road_principal_origin"
        exact_once = (
            "One retained-principal kg with its actual fillers, hardener and wet coated beads. "
            "Complete uncontrolled burning and on-site mineral/organic residue; geological "
            "carbon remains an explicit fate limitation. No second ingredient purchase, "
            "collection, treatment plant or ash-disposal service."
        )
    elif construction["kind"] == "epoxy_open_burning":
        origin = cast(_CarbonOrigin, construction["principal_carbon_origin"])
        # The foreground supplies one kg principal entering this fate. Fixed incorporated MXDA
        # accompanies that resin and retains its own fossil carbon origin.
        inlet_epoxy = _epoxy_material(
            (
                1.0 if origin == "fossil" else 0.0,
                1.0 if origin == "biogenic" else 0.0,
                1.0 if origin == "atmospheric_dac" else 0.0,
            ),
            _EPOXY_HARDENER_RATIO,
        )
        material = _OpenBurningMaterial(
            wet_mass_kg=inlet_epoxy.mass_kg,
            water_kg=0.0,
            burnable_elements_kg=inlet_epoxy.elements_kg,
            inert_elements_kg={},
            principal_carbon_kg=dict(zip(_ORIGINS, inlet_epoxy.principal_carbon_kg, strict=True)),
            auxiliary_carbon_kg={"fossil": inlet_epoxy.fossil_auxiliary_carbon_kg},
            heat_potential_mj=inlet_epoxy.heat_potential_mj,
        )
        material_terms = {
            "principal_carbon_origin": origin,
            "incorporated_mxda_kg": _EPOXY_HARDENER_RATIO,
        }
        origin_status = "actual_epoxy_principal_origin"
        exact_once = (
            "One kg principal with its fixed cured MXDA; complete uncontrolled burning "
            "and on-site residues. No new resin, hardener, treatment plant or ash disposal."
        )
    elif construction["kind"] == "sbr_open_burning":
        origin = cast(_CarbonOrigin, construction["principal_carbon_origin"])
        # Compound 22 carries fixed retained auxiliaries per principal kg. The complete load follows
        # the product.
        sbr = _sbr_compound_material(origin, principal_kg=1.0)
        material = _OpenBurningMaterial(
            wet_mass_kg=sbr.mass_kg,
            water_kg=0.0,
            burnable_elements_kg=sbr.elements_kg,
            inert_elements_kg={},
            principal_carbon_kg=sbr.principal_carbon_kg,
            auxiliary_carbon_kg=sbr.auxiliary_carbon_kg,
            heat_potential_mj=fsum((sbr.carbon_heat_mj, sbr.sulfur_heat_mj)),
        )
        material_terms = {"principal_carbon_origin": origin}
        origin_status = "actual_sbr_principal_origin"
        exact_once = (
            "One kg principal with the declared retained material; complete uncontrolled "
            "burning and on-site residue. Fixed auxiliaries keep their origins, with "
            "no new compound purchase, infrastructure or ash disposal."
        )
    else:
        material_id = str(construction["formulation_service_id"])
        material_name = material_id.removesuffix("_biogenic")
        inlet = {
            "pbat_biogenic": _pbat_material,
            "pbs_biogenic": _pbs_material,
            "ptt_biogenic": _ptt_material,
        }[material_id](services)
        origin = cast(_CarbonOrigin, construction["ultimate_carbon_origin"])
        # Finished-resin demands already split kg by actual carbon origins,
        # including polyester TBOT and PBAT's HDI. Each provider attributes all
        # its carbon to that receiving origin. Repeating the manufacturing
        # fossil fraction here would count that origin twice. Dry chemistry,
        # included residues and heat are identical for the three profiles.
        material = _OpenBurningMaterial(
            wet_mass_kg=1.0,
            water_kg=0.0,
            burnable_elements_kg=inlet.elements_kg,
            inert_elements_kg={},
            principal_carbon_kg={
                origin: fsum((inlet.selected_carbon_kg, inlet.fixed_fossil_carbon_kg))
            },
            auxiliary_carbon_kg={},
            heat_potential_mj=inlet.heat_potential_mj,
        )
        material_terms = {
            "material_service_id": material_id,
            "ultimate_carbon_origin": origin,
        }
        origin_status = f"actual_{material_name}_principal_origin"
        exact_once = (
            f"One kg finished {material_name.upper()}; "
            "ultimate-origin demand owns the carbon split once. "
            "Neither formation nor the original mixed-plastic source is purchased."
        )
    load = _open_burning_load(material)
    original = context.source.activity(cast(dict[str, object], construction["source"]), year=2025)
    _service_trace(context, service)
    trace = next(
        row
        for row in context.semantic_rows
        if row["ledger_entry_type"] == "service_construction"
        and row["provider_service_id"] == service_id
    )
    trace.update(
        bundled_stage_coverage="Complete uncontrolled burning; residue remains on site.",
        exact_once_decision=exact_once,
        project_terms_json=json.dumps(
            {
                **material_terms,
                "original_source_key": original.key,
                "wet_mass_kg_per_principal_kg": load.wet_mass_kg,
                "incoming_water_kg": load.incoming_water_kg,
                "input_elements_kg": load.elements_kg,
                "principal_carbon_kg": load.principal_carbon_kg,
                "auxiliary_carbon_kg": load.auxiliary_carbon_kg,
                "air_compounds_kg": load.air_compounds_kg,
                "air_oxide_elements_kg": load.air_oxide_elements_kg,
                "residue_elements_kg": load.residue_elements_kg,
                "soil_compounds_kg": load.soil_compounds_kg,
                "atmospheric_oxygen_kg": load.atmospheric_oxygen_kg,
                "atmospheric_nitrogen_kg": load.atmospheric_nitrogen_kg,
                "gross_heat_potential_mj": load.heat_potential_mj,
                "source_rounding_kg": load.source_rounding_kg,
                "geological_carbon_kg": load.geological_carbon_kg,
                "indicator_boundary": (
                    "PM, heat and disposal indicators overlap the mass-balance ledgers; "
                    "PCB/dioxin atoms are qualified source-equivalent bookkeeping."
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    exchanges: list[dict[str, object]] = [{"input": key, "amount": 1.0, "type": "production"}]
    principal_fractions: list[dict[str, object]] = []
    for (name, unit, categories), amount in load.biosphere_amounts.items():
        flow = context.biosphere.flow({"name": name, "unit": unit, "categories": list(categories)})
        row: dict[str, object] = {
            "input": flow.key,
            "amount": amount,
            "type": "biosphere",
            "uncertainty type": 0,
            "loc": amount,
        }
        fractions = load.principal_origin_fractions.get((name, unit, categories))
        if fractions is not None:
            row["principal_carbon_origin_fractions"] = tuple(
                fractions.get(origin, 0.0) for origin in _ORIGINS
            )
            # These services distinguish principal from retained auxiliaries.
            # The ordinary diagnostic reads the sum of principal
            # fractions per native flow; full auxiliary emissions stay in LCI.
            if construction["kind"] in {
                "epoxy_open_burning",
                "upr_open_burning",
                "sbr_open_burning",
                "road_open_burning",
            }:
                principal_fractions.append(
                    {
                        "flow_database": flow.key[0],
                        "flow_code": flow.key[1],
                        "fraction": fsum(fractions.values()),
                    }
                )
        exchanges.append(row)
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "direct_treatment_carbon_owners_json": _direct_carbon_owners_json(
            (
                {
                    "owner_database": key[0],
                    "owner_code": key[1],
                    "owner_scale_per_provider_unit": 1.0,
                    "owner_scope": "direct_biosphere_only",
                    "source_role": "member_local_frontier_service",
                    "origin_mapping_status": origin_status,
                    **(
                        {"principal_carbon_fractions": principal_fractions}
                        if construction["kind"]
                        in {
                            "epoxy_open_burning",
                            "upr_open_burning",
                            "sbr_open_burning",
                            "road_open_burning",
                        }
                        else {}
                    ),
                },
            )
        ),
        "exchanges": exchanges,
    }
