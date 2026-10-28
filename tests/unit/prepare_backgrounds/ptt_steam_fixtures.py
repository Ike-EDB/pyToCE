"""Small native PTT conversion and biogas suppliers for source-boundary tests."""

from math import log
from typing import Any


def ptt_steam_native_inventory(database: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Add a nonunit kg converter and its selected native biogas heat supplier.

    The companion chemical-steam fixture supplies the RoW MJ market/producer.
    The invented conversion is 8 MJ per 2 kg, deliberately unlike ecoinvent's
    2.75 MJ/kg. Both activities use synthetic source quantities.
    """
    return {
        (database, "ptt_kg_converter"): {
            "name": "steam production, in chemical industry",
            "reference product": "steam, in chemical industry",
            "unit": "kilogram",
            "location": "RoW",
            "type": "process",
            "comment": "Invented massless energy conversion with a nonunit output.",
            "native_note": "retain converter metadata",
            "exchanges": [
                {
                    "input": (database, "ptt_kg_converter"),
                    "amount": 2.0,
                    "type": "production",
                    "comment": "Source reference amount differs from one kg.",
                },
                {
                    "input": (database, "steam_row_market"),
                    "amount": 8.0,
                    "type": "technosphere",
                    "uncertainty type": 2,
                    "loc": log(8.0),
                    "scale": 0.15,
                },
            ],
        },
        (database, "ptt_biogas"): {
            "name": "heat and power co-generation, biogas, gas engine",
            "reference product": "heat, central or small-scale, other than natural gas",
            "unit": "megajoule",
            "location": "RoW",
            "type": "process",
            "exchanges": [{"input": (database, "ptt_biogas"), "amount": 1.0, "type": "production"}],
        },
    }
