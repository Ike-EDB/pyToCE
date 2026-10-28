"""Receive actual resins, coatings and PBS finishing rejects in municipal co-treatment."""

from math import fsum
from typing import Literal, cast

from .ash_treatment import (
    _ROAD_ASH_RELEASES,
    _UPR_ASH_RELEASES,
    _ash_dataset,
    _ash_market_dataset,
    _AshReleaseProfile,
    _TreatmentCarbon,
)
from .epoxy_material import _epoxy_material
from .municipal_incineration import (
    _municipal_bottom_ash_water_fraction,
    _municipal_incineration_dataset,
    _municipal_incineration_source,
)
from .municipal_incineration_load import (
    _MUNICIPAL_BOTTOM_WATER_FRACTION,
    MunicipalIncinerationLoad,
    MunicipalIncinerationMaterial,
    calculate_municipal_incineration,
)
from .polyester_material import _pbs_material
from .recipe_context import _RecipeContext
from .road_material import _road_material
from .sbr_material import _sbr_compound_material
from .upr_material import _upr_material


def _resin_incineration_load(
    context: _RecipeContext,
    construction: dict[str, object],
    *,
    bottom_ash_water_fraction: float = _MUNICIPAL_BOTTOM_WATER_FRACTION,
) -> MunicipalIncinerationLoad:
    """Apply the municipal operation to one principal kg and its actual cured material.

    Epoxy incorporates its fixed MXDA; UPR retains its actual glass, silica
    and organic auxiliaries; SBR retains its compound22 additives and zinc.
    UPR and road marking separate their reacting matrix from bulk minerals.
    PBS finishing rejects retain the finished backbone/catalyst proportions;
    selected-origin backbone and fossil catalyst carbon remain separate pools.
    No receiving function buys principal or auxiliaries again.
    The selected original source composition calibrates operating duties; the actual coating
    supplies its own moisture and element quantities.
    """
    origin = cast(
        Literal["fossil", "biogenic", "atmospheric_dac"], construction["principal_carbon_origin"]
    )
    principal = (
        float(origin == "fossil"),
        float(origin == "biogenic"),
        float(origin == "atmospheric_dac"),
    )
    if str(construction["kind"]).startswith("road_"):
        conversion = next(
            row
            for row in context.material_recipe.conversion
            if row.operation == "others_road_marking_conversion"
        )
        road = _road_material(conversion, principal)
        # Doka2003 II section3.2.1 sends bulk glass, pigment, silica, alumina and limestone to slag.
        # Only the organic matrix and its PDMS Si use the furnace element phases. Geological C stays
        # outside organic chemistry.
        bulk = dict(road.mineral_elements_kg)
        for element, amount in road.limestone_elements_kg.items():
            if element != "C":
                bulk[element] = bulk.get(element, 0.0) + amount
        material = MunicipalIncinerationMaterial(
            road.mass_kg,
            road.water_kg,
            road.burnable_elements_kg,
            (
                _TreatmentCarbon("road_principal", True, origin, fsum(road.principal_carbon_kg)),
                _TreatmentCarbon(
                    "road_auxiliaries", False, "fossil", road.fossil_auxiliary_carbon_kg
                ),
            ),
            road.heat_potential_mj,
            bulk_inert_elements_kg=bulk,
            geological_carbon_kg=road.geological_carbon_kg,
        )
    elif str(construction["kind"]).startswith("upr_"):
        conversion = next(
            row
            for row in context.material_recipe.conversion
            if row.operation == "others_upr_hand_lay_up_conversion"
        )
        composite = _upr_material(conversion, principal)
        # Doka2023 sections3.1.1/3.1.3 keep glass/silica outside furnace
        # reactions. Their complete mass reaches bottom ash; dispersed Co
        # and organosiloxane Si remain in the admitted reacting matrix.
        material = MunicipalIncinerationMaterial(
            composite.mass_kg,
            composite.water_kg,
            composite.burnable_elements_kg,
            (
                _TreatmentCarbon("UPR", True, origin, sum(composite.principal_carbon_kg)),
                _TreatmentCarbon(
                    "UPR_principal_reference_difference",
                    False,
                    origin,
                    sum(composite.principal_carbon_bridge_kg),
                ),
                _TreatmentCarbon(
                    "UPR_auxiliaries", False, "fossil", composite.fossil_auxiliary_carbon_kg
                ),
            ),
            composite.heat_potential_mj,
            bulk_inert_elements_kg=composite.inert_elements_kg,
        )
    elif str(construction["kind"]).startswith("sbr_"):
        compound = _sbr_compound_material(origin, principal_kg=1.0)
        material = MunicipalIncinerationMaterial(
            compound.mass_kg,
            0.0,  # The retained compound22 material is dry.
            compound.elements_kg,
            (
                _TreatmentCarbon(
                    "principal_sbr", True, origin, compound.principal_carbon_kg[origin]
                ),
                _TreatmentCarbon(
                    "fossil_compounding_auxiliaries",
                    False,
                    "fossil",
                    compound.auxiliary_carbon_kg["fossil"],
                ),
                _TreatmentCarbon(
                    "tallow_compounding_auxiliaries",
                    False,
                    "biogenic",
                    compound.auxiliary_carbon_kg["biogenic"],
                ),
            ),
            compound.carbon_heat_mj + compound.sulfur_heat_mj,
        )
    elif str(construction["kind"]).startswith("pbs_"):
        formulation = next(
            row
            for row in cast(list[dict[str, object]], context.catalogue["services"])
            if row["provider_service_id"] == "pbs_biogenic"
        )
        polyester = _pbs_material({"pbs_biogenic": formulation})
        material = MunicipalIncinerationMaterial(
            1.0,
            0.0,
            polyester.elements_kg,
            (
                _TreatmentCarbon("PBS_backbone", True, origin, polyester.selected_carbon_kg),
                _TreatmentCarbon(
                    "PBS_catalyst_residue", False, "fossil", polyester.fixed_fossil_carbon_kg
                ),
            ),
            polyester.heat_potential_mj,
        )
    else:
        # Fixed dry MXDA/resin ratio belongs to the admitted epoxy material.
        coating = _epoxy_material(principal, 15.44 / 84.56)
        material = MunicipalIncinerationMaterial(
            coating.mass_kg,
            0.0,
            coating.elements_kg,
            (
                _TreatmentCarbon("GY250", True, origin, sum(coating.principal_carbon_kg)),
                _TreatmentCarbon("MXDA", False, "fossil", coating.fossil_auxiliary_carbon_kg),
            ),
            coating.heat_potential_mj,
        )
    return calculate_municipal_incineration(
        material,
        _municipal_incineration_source(context, construction),
        bottom_ash_water_fraction=bottom_ash_water_fraction,
    )


def _resin_treatment_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Bind a principal-kg furnace or its bottom/residual ash treatment.

    Controlled co-treatment uses the European paint source for road material
    and RoW sources for the other materials across 12 world regions.  Recovered heat and electricity
    receive
    no credit.
    """
    construction = cast(dict[str, object], service["construction"])
    kind = str(construction["kind"])
    material_name = kind.split("_", 1)[0]
    thermal = (
        construction
        if kind == f"{material_name}_incineration"
        else cast(
            dict[str, object],
            services[str(construction["incineration_service_id"])]["construction"],
        )
    )
    water_fraction = _MUNICIPAL_BOTTOM_WATER_FRACTION
    if material_name == "road":
        market_id = str(cast(dict[str, object], thermal["bottom_ash"])["service_id"])
        market = cast(dict[str, object], services[market_id]["construction"])
        water_fraction = _municipal_bottom_ash_water_fraction(
            context, cast(dict[str, object], market["source"])
        )
    load = _resin_incineration_load(context, thermal, bottom_ash_water_fraction=water_fraction)
    if kind == f"{material_name}_incineration":
        if material_name == "sbr":
            # Retained principal rubber is a hydrocarbon: its one-kg unit contains the admitted
            # carbon and complementary hydrogen. The complete load carries fixed additives
            # separately.
            carbon = fsum(pool.carbon_kg for pool in load.material.carbon if pool.principal)
            principal_elements = {"C": carbon, "H": 1.0 - carbon}
        elif material_name == "road":
            principal_elements = _road_material(
                next(
                    row
                    for row in context.material_recipe.conversion
                    if row.operation == "others_road_marking_conversion"
                ),
                (1.0, 0.0, 0.0),
            ).principal_elements_kg
        elif material_name == "upr":
            principal_elements = _upr_material(
                next(
                    row
                    for row in context.material_recipe.conversion
                    if row.operation == "others_upr_hand_lay_up_conversion"
                ),
                (1.0, 0.0, 0.0),
            ).principal_elements_kg
        elif material_name == "pbs":
            # A manufacturing reject is one kg of the complete dry PBS product,
            # including its product-associated catalyst residue, with no repurchase.
            principal_elements = load.material.elements_kg
        else:
            principal_elements = _epoxy_material((1.0, 0.0, 0.0), 0.0).elements_kg
        return _municipal_incineration_dataset(
            context,
            service,
            load,
            database_name,
            material_name=material_name,
            principal_origin=str(thermal["principal_carbon_origin"]),
            principal_elements_kg=principal_elements,
            production_comment=(
                "One kg homogeneous dry PBS finishing reject, including proportional "
                "catalyst-derived residue; selected backbone and fossil catalyst carbon "
                "retain separate origins. No second polymer or catalyst purchase."
                if material_name == "pbs"
                else "One kg retained principal with its actual cured auxiliaries; UPR includes "
                "reinforcement glass and silica; SBR includes retained compound22 additives. "
                "Road marking includes its actual wet fillers and coating; geological C stays "
                "outside the selected organic fate model. "
                "Nominal total carbon and tracked "
                "principal carbon remain explicit separate references."
            ),
        )
    fraction = str(construction["ash_fraction"])
    ash = load.bottom_ash if fraction == "bottom" else load.residual_ash
    if kind == f"{material_name}_ash_market":
        return _ash_market_dataset(context, service, ash, database_name)
    return _ash_dataset(
        context,
        service,
        ash,
        database_name,
        # Both selected MIXED ash functions already contain positive Zn rows.
        # Preserve their native release fractions and uncertainty for SBR.
        release_profile=(
            _ROAD_ASH_RELEASES[fraction]
            if material_name == "road"
            else _UPR_ASH_RELEASES[fraction]
            if material_name == "upr"
            else _AshReleaseProfile(())
        ),
    )
