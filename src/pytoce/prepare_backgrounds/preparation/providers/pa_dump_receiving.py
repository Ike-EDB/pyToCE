"""Admit five original dump references and PA receiving flow identities.

Landfill2020 supplies climate, gas and evaporite conventions; the PA receiving
calculation supplies actual principal, retained finish and water.
These unmanaged sites have no treatment, gas capture or operating suppliers.
Original native land rows retain their own uncertainty and product basis.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .landfill_phases import _LandfillSite
from .native_recipe import _self_production_amount
from .pa_receiving_records import (
    PaReceivingFlow,
    PaReceivingOperation,
    _activity,
    _flow,
    _flow_from_payload,
    _operation_from_payload,
    _receiving_row,
)
from .recipe_context import _RecipeContext

_AIR = ("air", "non-urban air or from high stacks")
_LATER_AIR = ("air", "low population density, long-term")
_GROUND = ("water", "ground-")
_LATER_GROUND = ("water", "ground-, long-term")
_SOIL = ("soil", "industrial")
_WASTE = ("inventory indicator", "waste")
_LAND = ("natural resource", "land")
_CLASSES = (
    ("hyperarid infiltration class (-250mm)", "hyperarid_infiltration_class_250mm", True),
    ("dry infiltration class (100mm)", "dry_infiltration_class_100mm", False),
    ("moist infiltration class (300mm)", "moist_infiltration_class_300mm", False),
    ("wet infiltration class (500mm)", "wet_infiltration_class_500mm", False),
    ("very wet infiltration class (1000mm)", "very_wet_infiltration_class_1000mm", False),
)


@dataclass(frozen=True, slots=True)
class PaDumpSite:
    """One original 7 m dump climate and normalized initial land occupation.

    ``upward_leachate`` selects the hyperarid evaporite reference; even the dry
    positive-infiltration class releases to groundwater. Production is signed
    native kg, while land rows are already per positive kg placed. Climate is
    native site geography, independent of the material's world region.
    """

    infiltration_class: str
    placement: PaReceivingOperation
    climate: _LandfillSite
    upward_leachate: bool


@dataclass(frozen=True, slots=True)
class PaDumpReceivingReference:
    """Five fixed original biosphere-only sites and exact receiving endpoints.

    Shared preparation resolves all original native identities once. There is
    no dated dump operation, manufactured support, sanitary child or copied
    polyurethane composition in the actual PA calculation.
    """

    sites: tuple[PaDumpSite, ...]
    biosphere: tuple[PaReceivingFlow, ...]


def _dump_flow_descriptors() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Name the finite native outputs of PA gas, leachate and evaporite receiving."""
    descriptors: set[tuple[str, str, tuple[str, ...]]] = {
        (f"{species}, {origin}", "kilogram", _AIR)
        for species in ("Methane", "Carbon dioxide")
        for origin in ("fossil", "non-fossil")
    }
    descriptors.update(
        (name, "kilogram", _AIR) for name in ("Sulfur dioxide", "Nitrogen oxides", "Sodium I")
    )
    descriptors.update(
        (name, "kilogram", compartment)
        for name in (
            "TOC, Total Organic Carbon",
            "DOC, Dissolved Organic Carbon",
            "COD, Chemical Oxygen Demand",
            "BOD5, Biological Oxygen Demand",
            "Ammonium",
            "Nitrogen, organic bound",
            "Nitrite",
            "Nitrate",
            "Sulfate",
            "Hydrogen sulfide",
            "Phosphate",
            "Sodium I",
        )
        for compartment in (_GROUND, _LATER_GROUND)
    )
    descriptors.update(
        (name, "kilogram", compartment)
        for name in (
            "Sulfate",
            "Nitrate",
            "Phosphorus",
            "Sodium I",
            "Particulate Matter, > 10 um",
            "Particulate Matter, > 2.5 um and < 10um",
            "Particulate Matter, < 2.5 um",
        )
        for compartment in (_AIR, _LATER_AIR)
    )
    descriptors.update(
        (name, "kilogram", _SOIL)
        for name in ("Carbon", "Nitrogen", "Sulfur", "Phosphorus", "Sodium I")
    )
    descriptors.update(
        (name, "kilogram", _WASTE)
        for name in (
            "Non-hazardous waste disposed",
            "Organic carbon, placed in landfill",
            "Waste mass, total, placed in landfill",
        )
    )
    descriptors.update(("Heat, waste", "megajoule", c) for c in (_SOIL, _LATER_GROUND))
    descriptors.update(
        ("Water", "cubic meter", c) for c in (_AIR, _LATER_AIR, _GROUND, _LATER_GROUND)
    )
    return tuple(sorted(descriptors))


def admit_pa_dump_receiving_reference(context: _RecipeContext) -> PaDumpReceivingReference:
    """Read selected original sites and bind their existing elementary identities."""
    services = {
        str(item["provider_service_id"]): item
        for item in cast(list[dict[str, object]], context.catalogue["services"])
    }
    sites = []
    for infiltration_class, suffix, upward in _CLASSES:
        service = services[f"mismanaged_open_dump_pur_{suffix}"]
        construction = cast(dict[str, object], service["construction"])
        source = context.source.activity(
            cast(dict[str, object], construction["selector"]), year=2025
        )
        climate = {
            name: float(value)
            for name, value in re.findall(r"\((MAT|MAP)\) of ([0-9.]+)", str(source["comment"]))
        }
        native = context.exchanges(source)
        production = _self_production_amount(source, native, method="PA open-dump source")
        land = []
        for row in native:
            if row["type"] != "biosphere":
                continue
            endpoint = context.input(row)
            if tuple(endpoint["categories"]) == _LAND:
                land.append(_receiving_row(_flow(endpoint), dict(row), 1 / abs(production)))
        sites.append(
            PaDumpSite(
                infiltration_class,
                PaReceivingOperation(_activity(source), production, tuple(land)),
                # Landfill2020 control board E133/E135: the 2.0604 m aerobic top layer reduces
                # methane formation in the selected 7 m dump. This changes gas speciation.
                _LandfillSite(climate["MAT"], climate["MAP"], 1 - 2.0604 / 7),
                upward,
            )
        )
    biosphere = tuple(
        _flow(context.biosphere.flow({"name": name, "unit": unit, "categories": list(categories)}))
        for name, unit, categories in _dump_flow_descriptors()
    )
    return PaDumpReceivingReference(tuple(sites), biosphere)


def dump_reference_from_payload(payload: Mapping[str, object]) -> PaDumpReceivingReference:
    """Restore completed original dump records."""
    sites = []
    for item in cast(list[dict[str, object]], payload["sites"]):
        climate = cast(dict[str, float], item["climate"])
        sites.append(
            PaDumpSite(
                str(item["infiltration_class"]),
                _operation_from_payload(cast(dict[str, object], item["placement"])),
                _LandfillSite(**climate),
                bool(item["upward_leachate"]),
            )
        )
    return PaDumpReceivingReference(
        tuple(sites),
        tuple(
            _flow_from_payload(item) for item in cast(list[dict[str, object]], payload["biosphere"])
        ),
    )
