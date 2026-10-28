"""Small registered inventories with one characterized emission per activity."""

from collections.abc import Sequence
from types import ModuleType

import bw2calc as bc


def diagonal_lca(
    bd: ModuleType,
    factors: Sequence[float],
    *,
    database: str = "background",
    amounts: Sequence[float] | None = None,
    method: tuple[str, ...] = ("EF v3.1", "climate change"),
    unit: str = "kg CO2-Eq",
) -> bc.LCA:
    """Register one emission per activity and load its actual Brightway matrices.

    Each activity produces one kilogram and exchanges ``amounts[i]`` kilograms
    of its own elementary flow, characterized by ``factors[i]``. Different
    databases can extend the same registered method with disjoint flow sets.
    """
    biosphere = f"{database}_biosphere"
    quantities = tuple(amounts) if amounts is not None else (1.0,) * len(factors)
    bd.Database(biosphere).write(
        {
            (biosphere, str(index)): {
                "name": f"emission {index}",
                "unit": "kilogram",
                "type": "emission",
                "categories": ("air",),
            }
            for index in range(len(factors))
        }
    )
    bd.Database(database).write(
        {
            (database, str(index)): {
                "name": f"activity {index}",
                "reference product": f"product {index}",
                "unit": "kilogram",
                "location": "GLO",
                "type": "process",
                "exchanges": [
                    {
                        "input": (database, str(index)),
                        "amount": 1.0,
                        "type": "production",
                    },
                    {
                        "input": (biosphere, str(index)),
                        "amount": amount,
                        "type": "biosphere",
                    },
                ],
            }
            for index, amount in enumerate(quantities)
        }
    )
    registered = bd.Method(method)
    previous = registered.load() if method in bd.methods else []
    if method not in bd.methods:
        registered.register(unit=unit)
    registered.write(
        [*previous, *[((biosphere, str(index)), factor) for index, factor in enumerate(factors)]]
    )
    lca = bc.LCA({bd.get_node(database=database, code="0"): 1.0}, method=method)
    lca.lci()
    return lca
