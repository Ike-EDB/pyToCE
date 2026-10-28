"""Connect actual PHB municipal treatment, fresh auxiliaries and ash disposal."""

from typing import Literal, cast

from .ash_treatment import _ash_dataset, _ash_market_dataset, _AshReleaseProfile, _TreatmentCarbon
from .municipal_incineration import _municipal_incineration_dataset, _municipal_incineration_source
from .municipal_incineration_load import (
    MunicipalIncinerationLoad,
    MunicipalIncinerationMaterial,
    calculate_municipal_incineration,
)
from .open_burning import _organic_heat_potential
from .recipe_context import _RecipeContext

# Harding PHB1 is modeled dry and pure. Kookos (2019) supplement gives C4H6O2, 86.09 g/mol; CIAAW H
# 1.008/O 15.999 give kg H/O per kg repeat unit. Preserve the existing rounded principal C 0.558067;
# report its 1.39e-7 kg rounding remainder separately.
_PHB_ELEMENTS = {"C": 0.558067, "H": 6 * 1.008 / 86.09, "O": 2 * 15.999 / 86.09}


def _phb_incineration_load(
    context: _RecipeContext,
    construction: dict[str, object],
) -> MunicipalIncinerationLoad:
    """Read the original selected municipal source and calculate one kg dry PHB.

    Source product properties are per kg, independent of negative/nonunit
    reference production. Prospective relinking changes suppliers, not this
    admitted composition or original treatment quantities. Source metadata and
    coefficients remain available in the compiled service's provenance.
    """
    source = _municipal_incineration_source(context, construction)
    origin = cast(
        Literal["fossil", "biogenic", "atmospheric_dac"], construction["principal_origin"]
    )
    material = MunicipalIncinerationMaterial(
        1.0,
        0.0,
        _PHB_ELEMENTS,
        (_TreatmentCarbon("PHB", True, origin, _PHB_ELEMENTS["C"]),),
        _organic_heat_potential(_PHB_ELEMENTS),
    )
    return calculate_municipal_incineration(material, source)


def _phb_treatment_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Connect PHB furnace and actual ash services to the same municipal load.

    The ash construction obtains its quantities from the furnace reference. Native source reads are
    already owned by the recipe context.
    """
    construction = cast(dict[str, object], service["construction"])
    kind = str(construction["kind"])
    thermal = (
        construction
        if kind == "phb_incineration"
        else cast(
            dict[str, object],
            services[str(construction["incineration_service_id"])]["construction"],
        )
    )
    load = _phb_incineration_load(context, thermal)
    if kind == "phb_incineration":
        return _municipal_incineration_dataset(
            context,
            service,
            load,
            database_name,
            material_name="PHB",
            principal_origin=str(thermal["principal_origin"]),
            principal_elements_kg=_PHB_ELEMENTS,
            production_comment=(
                "One kg dry P(3HB) treated; H/O follow the nominal repeat-unit composition."
            ),
        )
    fraction = str(construction["ash_fraction"])
    ash = load.bottom_ash if fraction == "bottom" else load.residual_ash
    if kind == "phb_ash_market":
        return _ash_market_dataset(context, service, ash, database_name)
    return _ash_dataset(
        context, service, ash, database_name, release_profile=_AshReleaseProfile(())
    )
