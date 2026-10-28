"""Invented original/future open-burning sources shared by admission tests."""

from typing import Any

_AIR = ("air", "non-urban air or from high stacks")
_SOIL = ("soil", "industrial")
_INDICATOR = ("inventory indicator", "waste")
# Synthetic flow keys and exchange values represent the complete declared PA output domain.
_FLOWS = (
    *(
        (name, "kilogram", _AIR)
        for name in (
            "Benzene",
            "Phenol",
            "Styrene",
            "Toluene",
            "Formaldehyde",
            "Acetaldehyde",
            "Ethyl benzene",
            "PAH, polycyclic aromatic hydrocarbons",
            "Nitrogen oxides",
            "Dinitrogen monoxide",
            "Sulfur dioxide",
            "Carbon dioxide, fossil",
            "Carbon monoxide, fossil",
            "Methane, fossil",
            "Carbon dioxide, non-fossil",
            "Carbon monoxide, non-fossil",
            "Methane, non-fossil",
            "Phosphorus",
            "Sodium I",
            "Particulate Matter, > 10 um",
            "Particulate Matter, > 2.5 um and < 10um",
            "Particulate Matter, < 2.5 um",
        )
    ),
    *(
        (name, "kilogram", _SOIL)
        for name in ("Carbon", "Nitrogen", "Sulfur", "Phosphorus", "Sodium I")
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


def open_burning_biosphere(
    database: str = "burning_biosphere",
) -> dict[tuple[str, str], dict[str, Any]]:
    """Keep all admitted output descriptors plus the two deliberate unselected controls."""
    descriptors = (
        *_FLOWS,
        ("PCB", "kilogram", _AIR),
        ("Carbon dioxide, fossil", "kilogram", ("air",)),
    )
    return {
        (database, f"f{index}"): {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "natural resource" if categories[0] == "natural resource" else "emission",
        }
        for index, (name, unit, categories) in enumerate(descriptors)
    }


def open_burning_native_inventory(
    database: str, *, original: bool, biosphere_database: str = "burning_biosphere"
) -> dict[tuple[str, str], dict[str, Any]]:
    """Retain the two existing signed source references and their illustrative flow rows."""
    key = database, "burning"
    production, coefficient = (-1.0, 0.01) if original else (-2.0, 0.02)
    return {
        key: {
            "name": "treatment of waste plastic, mixture, open burning",
            "reference product": "waste plastic, mixture",
            "unit": "kilogram",
            "location": "GLO",
            "exchanges": [
                {"input": key, "type": "production", "amount": production},
                *(
                    {"input": flow, "type": "biosphere", "amount": coefficient}
                    for flow in open_burning_biosphere(biosphere_database)
                ),
            ],
        }
    }
