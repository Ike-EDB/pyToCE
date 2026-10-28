"""Small invented native formulations with real Brightway exchange properties."""

from pathlib import Path
from typing import Any


def material_carbon_sources(
    database: str, biosphere: str
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Supply six kg-based formulations, duplicate suppliers and conflicting master metadata."""
    inputs = {
        "long": {"ethylene oxide": 0.2, "propylene oxide, liquid": 0.7, "glycerine": 0.1},
        "short": {
            "ethylene oxide": 0.05,
            "propylene oxide, liquid": 0.6,
            "glycerine": 0.1,
            "glucose": 0.15,
            "sugar, from sugarcane": 0.1,
        },
        "flexible": {
            "polyether polyols, long chain": 0.7,
            "toluene diisocyanate": 0.3,
            "styrene-acrylonitrile copolymer": 0.01,
            "waste polyurethane": -0.02,
        },
        "rigid": {
            "polyether polyols, short chain": 0.4,
            "methylene diphenyl diisocyanate": 0.6,
            "pentane": 0.05,
            "waste polyurethane": -0.03,
        },
        "adhesive": {
            "polyether polyols, long chain": 0.4,
            "methylene diphenyl diisocyanate": 0.55,
            "chemical, organic, basic precursors and materials": 0.05,
        },
        "upr": {},
    }
    identities = {
        "long": (
            "polyether polyols production, long chain",
            "polyether polyols, long chain",
            "RER",
        ),
        "short": (
            "polyether polyols production, short chain",
            "polyether polyols, short chain",
            "RER",
        ),
        "flexible": (
            "polyurethane production, flexible foam, TDI-based, high density",
            "polyurethane, flexible foam",
            "RER",
        ),
        "rigid": ("polyurethane production, rigid foam", "polyurethane, rigid foam", "RER"),
        "adhesive": ("polyurethane adhesive production", "polyurethane adhesive", "GLO"),
        "upr": (
            "orthophthalic acid based unsaturated polyester resin production",
            "orthophthalic acid based unsaturated polyester resin",
            "RER",
        ),
    }
    data: dict[tuple[str, str], dict[str, Any]] = {}
    for role, (name, product, location) in identities.items():
        key = (database, role)
        exchanges = [
            {
                "input": key,
                "type": "production",
                "name": product,
                "amount": 1.0,
                "unit": "kilogram",
                "properties": {
                    "dry mass": {"amount": 1.0, "unit": "kg"},
                    "carbon content": {
                        "amount": {"long": 0.6, "short": 0.5, "upr": 0.7}.get(role, 0.99),
                        "unit": "dimensionless",
                    },
                },
            }
        ]
        for index, (ingredient, amount) in enumerate(inputs[role].items()):
            # Split adhesive MDI/polyol between two suppliers; selecting a first
            # supplier loses a substantive fraction of the source formulation.
            fractions = (0.25, 0.75) if role == "adhesive" and index < 2 else (1.0,)
            for supplier, fraction in enumerate(fractions):
                input_key = (database, f"{role}_ingredient_{index}_{supplier}")
                data[input_key] = {
                    "name": f"market for {ingredient}",
                    "reference product": ingredient,
                    "unit": "kilogram",
                    "location": ("RER", "RoW")[supplier],
                    "type": "process",
                    "carbon content": 0.99,
                    "exchanges": [{"input": input_key, "type": "production", "amount": 1.0}],
                }
                exchange = {
                    "input": input_key,
                    "type": "technosphere",
                    "name": ingredient,
                    "unit": "kilogram",
                    "amount": amount * fraction,
                }
                if ingredient == "chemical, organic, basic precursors and materials":
                    exchange["properties"] = {
                        "carbon content": {"amount": 0.5, "unit": "dimensionless"},
                        "dry mass": {"amount": 0.8, "unit": "kg"},
                    }
                exchanges.append(exchange)
        if role in {"flexible", "rigid"}:
            exchanges.append(
                {
                    "input": (biosphere, role),
                    "type": "biosphere",
                    "unit": "kilogram",
                    "amount": 0.04 if role == "flexible" else 0.01,
                }
            )
        data[key] = {
            "name": name,
            "reference product": product,
            "location": location,
            "unit": "kilogram",
            "type": "process",
            "carbon content": 0.01,
            "properties": {"carbon content": 0.01},
            "exchanges": exchanges,
        }
    flows = {
        (biosphere, role): {
            "name": name,
            "categories": ("air", "urban air close to ground"),
            "unit": "kilogram",
            "type": "emission",
        }
        for role, name in (("flexible", "Carbon dioxide, fossil"), ("rigid", "Pentane"))
    }
    return data, flows


def prepared_material_carbon(directory: Path) -> tuple[dict[str, object], ...]:
    """Build a complete carbon table once for a bounded stored-product fixture."""
    from pytoce.prepare_backgrounds.preparation.ecoinvent import (
        BrightwayNames,
        selected_brightway_project,
    )
    from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
        load_provider_catalogue,
    )
    from pytoce.prepare_backgrounds.preparation.providers.material_carbon import (
        prepare_material_carbon,
    )

    names = BrightwayNames(directory, "carbon_source_fixture", "carbon_biosphere", "carbon_native")
    data, flows = material_carbon_sources(names.source_database, names.biosphere_database)
    with selected_brightway_project(names) as bd:
        bd.Database(names.biosphere_database).write(flows, searchable=False)
        bd.Database(names.source_database).write(data, searchable=False)
        return prepare_material_carbon(names.source_database, load_provider_catalogue())
