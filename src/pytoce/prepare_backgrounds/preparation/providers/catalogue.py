"""Shared provider catalog and its stable source record."""

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import cast

import pandas as pd

from pytoce.shared.runtime.filesystem import payload_sha256

from .route_catalogue import augment_provider_catalogue

PROVIDER_INTERFACE_DATABASE = "pytoce__provider_interface"
_PROVIDER_METHOD = "pytoce_provider_services_v85"
_SOURCE_ROLE_AUTHORITIES = {
    "native_ecoinvent_provider": "ecoinvent_3_12_cutoff",
    "premise_only_provider": "premise_2_4_6",
}
_SOURCE_PROVENANCE_ONLY_FIELDS = {
    "database_release",
    "data_repository_commit",
    "process_url",
    "publication_source_uuid",
    "publication_source_url",
    "retained_source_record_sha256",
}


@dataclass(frozen=True, slots=True)
class PreparedProviderAuthority:
    """One provider route catalog and its stable service interface."""

    identity: str
    interface_database: str
    catalogue: dict[str, object]
    routes: pd.DataFrame
    source_authorities: tuple[dict[str, object], ...] = ()
    method_authorities: tuple[dict[str, object], ...] = ()


def load_provider_catalogue() -> dict[str, object]:
    """Load the package provider catalog."""

    resource = files("pytoce.prepare_backgrounds.data.provider_recipes").joinpath(
        "provider_catalogue.json"
    )
    return augment_provider_catalogue(json.loads(resource.read_text(encoding="utf-8")))


def provider_service_code(service_id: str) -> str:
    """Return one stable readable provider-service code."""

    name = re.sub(r"[^a-z0-9_]+", "_", service_id.casefold()).strip("_")
    return f"provider_service__{name}"


def provider_interface_database_name(
    *,
    source_identity: str,
    preparation_method_identity: str,
    dependency_authorities: tuple[dict[str, object], ...],
) -> str:
    """Bind package database names to admitted inputs before any output is written."""

    payload = {
        "schema_version": "pytoce_provider_database_namespace_schema_v1",
        "source_identity": source_identity,
        "preparation_method_identity": preparation_method_identity,
        "dependency_authorities": dependency_authorities,
    }
    return f"{PROVIDER_INTERFACE_DATABASE}__{payload_sha256(payload)}"


def provider_frontier_database_name(
    background_scenario: str,
    year: int,
    *,
    interface_database: str,
) -> str:
    """Return one dated frontier name within its admitted provider namespace."""

    return f"{interface_database}__services__{background_scenario}__{year}"


def provider_timeline_database_name(year: int, *, interface_database: str) -> str:
    """Return one shared inventory-neutral Timex interpolation database name."""

    if year == 2020:
        return interface_database
    return f"{interface_database}__timeline__{year}"


def _catalogue_identity(catalogue: dict[str, object]) -> str:
    """Hash the catalog and method, excluding designated provenance-only fields."""

    inventory_catalogue = {
        **catalogue,
        "source_authorities": [
            {
                key: value
                for key, value in source.items()
                if key not in _SOURCE_PROVENANCE_ONLY_FIELDS
            }
            for source in cast(
                list[dict[str, object]],
                catalogue["source_authorities"],
            )
        ],
    }
    payload = {
        "method": _PROVIDER_METHOD,
        "catalogue": inventory_catalogue,
    }
    return f"{_PROVIDER_METHOD}::{payload_sha256(payload)}"


def _route_table(catalogue: dict[str, object]) -> pd.DataFrame:
    """Add stable provider service codes and return the route table."""
    rows = []
    for raw in cast(list[dict[str, object]], catalogue["routes"]):
        service_id = str(raw["provider_service_id"])
        rows.append(
            {
                **raw,
                "provider_service_code": provider_service_code(service_id),
            }
        )
    return pd.DataFrame(rows)


def _method_authority(
    catalogue: dict[str, object],
    method_authority_id: str,
) -> dict[str, object]:
    """Resolve one method citation to its direct source records."""
    registry = cast(dict[str, object], catalogue["method_authority_registry"])
    exact = cast(dict[str, list[str]], registry["exact_source_bindings"])
    if method_authority_id in exact:
        defaults = cast(dict[str, object], registry["project_recipe_defaults"])
        details = cast(
            dict[str, dict[str, object]],
            registry.get("exact_method_authorities", {}),
        ).get(method_authority_id, defaults)
        return {
            "method_authority_id": method_authority_id,
            "source_authority_ids_json": json.dumps(
                exact[method_authority_id],
                separators=(",", ":"),
            ),
            **details,
        }
    prefix_rule = next(
        rule
        for rule in cast(list[dict[str, object]], registry["prefix_rules"])
        if method_authority_id.startswith(str(rule["method_authority_prefix"]))
    )
    return {
        "method_authority_id": method_authority_id,
        "source_authority_ids_json": json.dumps(
            prefix_rule["source_authority_ids"],
            separators=(",", ":"),
        ),
        **{
            key: value
            for key, value in prefix_rule.items()
            if key not in {"method_authority_prefix", "source_authority_ids"}
        },
    }


def _method_authority_rows(
    catalogue: dict[str, object],
) -> tuple[dict[str, object], ...]:
    """Collect source records for every method used by a service or route."""
    method_ids = sorted(
        {
            str(service["citation"])
            for service in cast(list[dict[str, object]], catalogue["services"])
        }
        | {
            str(route["method_authority_id"])
            for route in cast(list[dict[str, object]], catalogue["routes"])
            if route.get("method_authority_id")
        }
    )
    return tuple(_method_authority(catalogue, method_id) for method_id in method_ids)


def _source_authority_closure(
    catalogue: dict[str, object],
    source_ids: object,
) -> tuple[str, ...]:
    """Follow direct source links and return the complete source identifier set."""
    authorities = {
        str(row["source_id"]): row
        for row in cast(list[dict[str, object]], catalogue["source_authorities"])
    }
    pending = [str(value) for value in cast(list[object], source_ids)]
    resolved: set[str] = set()
    while pending:
        source_id = pending.pop()
        if source_id in resolved:
            continue
        resolved.add(source_id)
        pending.extend(
            str(value)
            for value in cast(
                list[object],
                authorities[source_id].get("direct_source_authority_ids", []),
            )
        )
    return tuple(sorted(resolved))


def _construction_source_authority_ids(value: object) -> tuple[str, ...]:
    """Return the construction source authority identifiers."""
    source_ids: set[str] = set()

    def visit(item: object) -> None:
        """Collect source authorities from each nested construction entry."""
        if isinstance(item, dict):
            source_role = item.get("source_role")
            if source_role is not None:
                source_ids.add(_SOURCE_ROLE_AUTHORITIES[str(source_role)])
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(sorted(source_ids))


def _service_direct_source_ids(
    catalogue: dict[str, object],
    service: dict[str, object],
) -> tuple[str, ...]:
    """Return the service direct source identifiers."""
    method = _method_authority(catalogue, str(service["citation"]))
    source_ids = {str(value) for value in json.loads(str(method["source_authority_ids_json"]))}
    source_ids.update(_construction_source_authority_ids(service["construction"]))
    return tuple(sorted(source_ids))


def _service_source_closures(
    catalogue: dict[str, object],
) -> dict[str, tuple[str, ...]]:
    """Collect all supply and formulation sources, including actual supply cycles.

    Treatment can buy a catalyst whose manufacture buys the same treatment.
    Source attribution visits each reachable service once; Brightway solves the inventory
    coefficients for the complete supply cycle.
    """
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], catalogue["services"])
    }
    dependencies: dict[str, tuple[str, ...]] = {}
    direct_sources: dict[str, tuple[str, ...]] = {}
    for service_id, service in services.items():
        dependencies[service_id] = _service_dependencies(service)
        construction = cast(dict[str, object], service["construction"])
        # Recovery and treatment reference the original formulation quantities. Ash references its
        # incinerator's composition calculation.
        for reference in ("formulation_service_id", "incineration_service_id"):
            if quantity_owner := construction.get(reference):
                dependencies[service_id] += (str(quantity_owner),)
        direct_sources[service_id] = _source_authority_closure(
            catalogue, list(_service_direct_source_ids(catalogue, service))
        )
    resolved: dict[str, tuple[str, ...]] = {}
    for service_id in services:
        pending = [service_id]
        visited: set[str] = set()
        source_ids: set[str] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            source_ids.update(direct_sources[current])
            pending.extend(dependencies[current])
        resolved[service_id] = tuple(sorted(source_ids))
    return resolved


def _source_uses_json(
    catalogue: dict[str, object],
    *,
    direct_source_ids: tuple[str, ...],
    complete_source_ids: tuple[str, ...],
) -> str:
    """Serialize each source's direct or component use for one service."""
    authorities = {
        str(row["source_id"]): row
        for row in cast(list[dict[str, object]], catalogue["source_authorities"])
    }
    direct = set(direct_source_ids)
    return json.dumps(
        [
            {
                "source_id": source_id,
                "inheritance": "direct" if source_id in direct else "component",
                "role": str(authorities[source_id].get("role", "")),
                "locations": authorities[source_id].get("source_locations", []),
            }
            for source_id in complete_source_ids
        ],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def prepared_provider_authority() -> PreparedProviderAuthority:
    """Return the package provider catalog and its source record."""

    catalogue = load_provider_catalogue()
    return PreparedProviderAuthority(
        identity=_catalogue_identity(catalogue),
        interface_database=PROVIDER_INTERFACE_DATABASE,
        catalogue=catalogue,
        routes=_route_table(catalogue),
        source_authorities=tuple(cast(list[dict[str, object]], catalogue["source_authorities"])),
        method_authorities=_method_authority_rows(catalogue),
    )


def _service_dependencies(service: dict[str, object]) -> tuple[str, ...]:
    """Find provider services referenced by one service construction recipe."""
    construction = cast(dict[str, object], service["construction"])
    dependencies: set[str] = set()

    def visit(value: object) -> None:
        """Collect provider service references from a nested recipe entry."""
        if isinstance(value, dict):
            if "service_id" in value:
                dependencies.add(str(value["service_id"]))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(construction)
    return tuple(sorted(dependencies))
