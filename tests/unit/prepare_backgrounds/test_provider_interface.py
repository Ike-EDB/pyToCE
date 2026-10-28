"""Brightway provider interface writing tests."""

from dataclasses import replace
from pathlib import Path

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    prepared_provider_authority,
    provider_interface_database_name,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.interface_database import (
    PROVIDER_TIMELINE_YEARS,
    provider_timeline_databases,
    provider_timeline_identity,
    write_provider_interface,
)


def test_provider_interface_is_written_with_all_dated_databases(tmp_path: Path) -> None:
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="provider_interface_test",
        biosphere_database="biosphere",
        source_database="ecoinvent",
    )
    provider = replace(
        prepared_provider_authority(),
        interface_database=provider_interface_database_name(
            source_identity="sources::interface_fixture",
            preparation_method_identity="preparation::interface_fixture",
            dependency_authorities=(),
        ),
    )
    services = provider.catalogue["services"]
    foreground_services = tuple(
        service for service in services if bool(service.get("foreground_service", False))
    )

    with selected_brightway_project(names) as bd:
        bd.Database(names.biosphere_database).write(
            {
                (names.biosphere_database, "co2"): {
                    "name": "Carbon dioxide, fossil",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
            },
            searchable=False,
        )
        bd.geomapping.add({str(service["location"]) for service in foreground_services})
        write_provider_interface(bd, names.biosphere_database, provider)
        database_by_year = provider_timeline_databases(provider.interface_database)

        assert tuple(database_by_year) == PROVIDER_TIMELINE_YEARS
        assert database_by_year[2020] == provider.interface_database
        assert provider_timeline_identity(
            provider.identity, interface_database=provider.interface_database
        ).startswith("pytoce_provider_timeline_v1::")
        assert all(
            len(bd.Database(database)) == len(foreground_services)
            for database in database_by_year.values()
        )

        biosphere_exchanges = [
            exchange
            for activity in bd.Database(provider.interface_database)
            for exchange in activity.biosphere()
        ]
        assert len(biosphere_exchanges) == 1
        assert float(biosphere_exchanges[0]["amount"]) == 0.0
        assert all(database in bd.databases for database in database_by_year.values())
        code = provider_service_code(str(foreground_services[0]["provider_service_id"]))
        for database in database_by_year.values():
            activity = bd.get_node(database=database, code=code)
            production = list(activity.production())
            assert len(production) == 1
            assert production[0].input.key == activity.key
            assert production[0]["amount"] == 1.0
        assert set(bd.databases) == {names.biosphere_database, *database_by_year.values()}
