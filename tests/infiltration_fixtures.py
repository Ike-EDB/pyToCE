"""Invented native market records for the conditional open-dump class calculation."""

from dataclasses import replace
from pathlib import Path

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.route_authorities import (
    _csv_rows,
    resolve_open_dump_routes,
)


def infiltration_source_data(database: str) -> dict[tuple[str, str], dict[str, object]]:
    """Make five class inputs, other treatment and transport with nonunit production."""
    data: dict[tuple[str, str], dict[str, object]] = {}
    inputs = []
    selectors = [
        row
        for row in _csv_rows("provider_selector_catalogue.csv")
        if row["fate"] == "open_dump" and row["provider_service_family"] == "PE"
    ]
    for row, amount in zip(selectors, (-1.0, -2.0, -3.0, -4.0, -7.0), strict=True):
        key = (database, row["provider_service_id"])
        data[key] = {
            "name": row["activity_name"],
            "reference product": row["reference_product"],
            "unit": row["unit"],
            "location": row["location"],
            "type": "process",
            "exchanges": [{"input": key, "amount": -1.0, "type": "production"}],
        }
        inputs.append({"input": key, "amount": amount, "type": "technosphere"})
    for code, unit, amount in (
        ("other_treatment", "kilogram", -3.0),
        ("transport", "ton kilometer", 0.5),
    ):
        key = (database, code)
        data[key] = {
            "name": code,
            "reference product": code,
            "unit": unit,
            "location": "GLO",
            "type": "process",
            "exchanges": [{"input": key, "amount": 1.0, "type": "production"}],
        }
        inputs.append({"input": key, "amount": amount, "type": "technosphere"})
    key = (database, "pe_waste_market")
    data[key] = {
        "name": "market for waste polyethylene",
        "reference product": "waste polyethylene",
        "unit": "kilogram",
        "location": "RoW",
        "type": "process",
        "exchanges": [
            {"input": key, "amount": -20.0, "type": "production"},
            *inputs,
        ],
    }
    return data


def resolved_infiltration_provider(workspace: Path) -> PreparedProviderAuthority:
    """Resolve the catalog using a small real Brightway source database."""
    names = BrightwayNames(
        store=workspace / "brightway",
        project="infiltration_fixture",
        source_database="infiltration_source",
        biosphere_database="biosphere",
    )
    provider = prepared_provider_authority()
    with selected_brightway_project(names) as bd:
        bd.Database(names.source_database).write(
            infiltration_source_data(names.source_database), searchable=False
        )
        routes, source = resolve_open_dump_routes(names.source_database, provider.routes)
    return replace(
        provider,
        routes=routes,
        source_authorities=tuple(
            {**row, **source} if row["source_id"] == source["source_id"] else row
            for row in provider.source_authorities
        ),
    )
