"""Admit the original uncontrolled-burning identity and existing PA output flows.

``open_burning`` reconstructs source quantities for the actual PA mixture. This shared source record
binds the original native identities.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .open_burning import _AIR, _INDICATOR, _ORGANICS, _PAH, _SOIL, _OpenBurningFlow
from .pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
    _activity,
    _activity_from_payload,
    _flow,
    _flow_from_payload,
)
from .recipe_context import _RecipeContext

# These are the admitted PA function's existing elementary outputs. Carbon-origin flows remain
# available even when a particular event has a zero origin pool.
_OUTPUTS: tuple[_OpenBurningFlow, ...] = (
    *(
        (name, "kilogram", _AIR)
        for name in (*_ORGANICS, _PAH, "Nitrogen oxides", "Dinitrogen monoxide", "Sulfur dioxide")
    ),
    *(
        (f"{name}, {origin}", "kilogram", _AIR)
        for origin in ("fossil", "non-fossil")
        for name in ("Carbon dioxide", "Carbon monoxide", "Methane")
    ),
    *(
        (name, "kilogram", _SOIL)
        for name in ("Carbon", "Nitrogen", "Sulfur", "Phosphorus", "Sodium I")
    ),
    *(
        (name, "kilogram", _AIR)
        for name in (
            "Phosphorus",
            "Sodium I",
            "Particulate Matter, > 10 um",
            "Particulate Matter, > 2.5 um and < 10um",
            "Particulate Matter, < 2.5 um",
        )
    ),
    ("Water", "cubic meter", _AIR),
    ("Oxygen", "kilogram", ("natural resource", "in air")),
    ("Heat, waste", "megajoule", _AIR),
    *(
        (name, "kilogram", _INDICATOR)
        for name in (
            "Non-hazardous waste disposed",
            "Waste mass, total, placed in landfill",
            "Organic carbon, placed in landfill",
        )
    ),
)


@dataclass(frozen=True, slots=True)
class PaOpenBurningReference:
    """Original GLO source provenance and complete existing PA biosphere bindings.

    The activity is the native mixed-plastic source, not its non-fossil provider
    adaptation. It supplies the uncontrolled-function provenance; no activity
    demand or negative native production scales the actual elementary amounts.
    Source geography describes the transferred function, independently of the 12 world regions.
    """

    activity: PaReceivingActivity
    biosphere: tuple[PaReceivingFlow, ...]


def admit_pa_open_burning_reference(context: _RecipeContext) -> PaOpenBurningReference:
    """Bind the declared original donor and all PA outputs once during preparation."""
    services = cast(list[dict[str, object]], context.catalogue["services"])
    service = next(
        item
        for item in services
        if item["provider_service_id"] == "plastic_waste_open_burning_glo_non_fossil"
    )
    construction = cast(dict[str, object], service["construction"])
    base = cast(dict[str, object], construction["base"])
    original = context.source.activity(cast(dict[str, object], base["selector"]), year=2025)
    return PaOpenBurningReference(
        _activity(original),
        tuple(
            _flow(
                context.biosphere.flow({"name": name, "unit": unit, "categories": list(categories)})
            )
            for name, unit, categories in _OUTPUTS
        ),
    )


def open_burning_reference_from_payload(payload: Mapping[str, object]) -> PaOpenBurningReference:
    """Restore the trusted shared identities."""
    return PaOpenBurningReference(
        _activity_from_payload(cast(Mapping[str, object], payload["activity"])),
        tuple(
            _flow_from_payload(row)
            for row in cast(list[Mapping[str, object]], payload["biosphere"])
        ),
    )
