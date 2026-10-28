"""Stable background-independent provider-service interface."""

from typing import Any, cast

from pytoce.shared.runtime.brightway import write_brightway_database
from pytoce.shared.runtime.filesystem import payload_sha256

from ..premise import (
    GENERAL_BACKGROUND_ANCHOR_YEARS,
    PROVIDER_TECHNOLOGY_ANCHOR_YEARS,
)
from .catalogue import (
    PreparedProviderAuthority,
    provider_service_code,
    provider_timeline_database_name,
)

PROVIDER_TIMELINE_YEARS = (
    *PROVIDER_TECHNOLOGY_ANCHOR_YEARS,
    *GENERAL_BACKGROUND_ANCHOR_YEARS,
)
PROVIDER_TIMELINE_METHOD = "bw_timex_semantic_twin_interpolation_v1"
_TIMEX_INTERFACE = "inventory_neutral_native_biosphere_v2"


def _interface_data(
    catalogue: dict[str, object],
    biosphere_key: tuple[str, str],
    database: str,
) -> dict[tuple[str, str], dict[str, object]]:
    """Build stable provider-service activities for the Brightway interface database."""
    services = sorted(
        (
            service
            for service in cast(list[dict[str, object]], catalogue["services"])
            if bool(service.get("foreground_service", False))
        ),
        key=lambda service: str(service["provider_service_id"]),
    )
    result: dict[tuple[str, str], dict[str, object]] = {}
    for index, service in enumerate(services):
        code = provider_service_code(str(service["provider_service_id"]))
        exchanges = [
            {
                "input": (database, code),
                "amount": 1.0,
                "type": "production",
            }
        ]
        if index == 0:
            exchanges.append(
                {
                    "input": biosphere_key,
                    "amount": 0.0,
                    "type": "biosphere",
                }
            )
        result[(database, code)] = {
            "name": str(service["name"]),
            "reference product": str(service["reference_product"]),
            "unit": str(service["unit"]),
            "location": str(service["location"]),
            "type": "process",
            "provider_service_id": str(service["provider_service_id"]),
            "exchanges": exchanges,
        }
    return result


def provider_timeline_databases(interface_database: str) -> dict[int, str]:
    """Return the shared technical date coordinates used by Timex."""

    return {
        year: provider_timeline_database_name(year, interface_database=interface_database)
        for year in PROVIDER_TIMELINE_YEARS
    }


def provider_timeline_identity(provider_identity: str, *, interface_database: str) -> str:
    """Return the identity for the dated provider database map."""

    payload = {
        "method": PROVIDER_TIMELINE_METHOD,
        "provider_identity": provider_identity,
        "database_by_year": provider_timeline_databases(interface_database),
    }
    return "pytoce_provider_timeline_v1::" + payload_sha256(payload)


def _timeline_metadata(
    provider: PreparedProviderAuthority,
    biosphere_key: tuple[str, str],
    year: int,
) -> dict[str, object]:
    """Return the metadata required for one dated provider database."""

    return {
        "pytoce_provider_identity": provider.identity,
        "pytoce_timex_interface": _TIMEX_INTERFACE,
        "pytoce_timex_biosphere_key": ":".join(biosphere_key),
        "pytoce_provider_timeline_year": year,
    }


def write_provider_interface(
    bd: Any,
    biosphere_database: str,
    provider: PreparedProviderAuthority,
) -> None:
    """Write the provider interface and its Timex date coordinates once."""

    biosphere_key = min(
        (str(node["code"]), cast(tuple[str, str], node.key))
        for node in bd.Database(biosphere_database)
    )[1]
    write_brightway_database(
        bd,
        provider.interface_database,
        _interface_data(provider.catalogue, biosphere_key, provider.interface_database),
        _timeline_metadata(provider, biosphere_key, PROVIDER_TIMELINE_YEARS[0]),
    )
    for year in PROVIDER_TIMELINE_YEARS[1:]:
        database = provider_timeline_database_name(
            year, interface_database=provider.interface_database
        )
        write_brightway_database(
            bd,
            database,
            _interface_data(provider.catalogue, biosphere_key, database),
            _timeline_metadata(provider, biosphere_key, year),
        )
