"""Read-only access to prepared provider services."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    PreparedDataError,
    PreparedManifest,
)
from pytoce.prepare_backgrounds.preparation.providers.material_carbon import (
    CARBON_ORIGINS,
    MATERIAL_CARBON_SCHEMA,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
    reference_from_payload,
    sources_from_payload,
)
from pytoce.run_mfa.model.materials import load_material_composition

_ROUTE_TEXT_COLUMNS = (
    "provider_class",
    "analogue_status",
    "selection_reason",
    "method_authority_id",
    "material_functional_unit",
    "sbr_proxy_component_role",
    "source_role",
    "resolution_status",
    "functional_equivalence_claim",
    "limitation",
    "treatment_carbon_mapping_status",
    "lca_fate",
    "infiltration_class",
)
_ROUTE_SHARE_COLUMNS = (
    "material_component_share",
    "infiltration_share",
)
_ROUTE_ZERO_COLUMNS = (
    "styrene_component_share",
    "butadiene_analogue_component_share",
    "sbr_share_in_material",
)
_ROUTE_BOOLEAN_COLUMNS = (
    "whole_tyre_constituents_excluded",
    "no_treatment_sensitivity_executed",
)


@dataclass(frozen=True, slots=True)
class PreparedProviderProduct:
    """Provider databases and routes prepared for one background scenario.

    ``database_by_year`` maps provider anchor years to dated service databases;
    ``timeline_database_by_year`` maps the same anchors to Timex temporal-market
    databases. Foreground compilation consumes the shared provider routes separately.
    PA cleaning uses one shared original calibration and dated source records;
    consumers read these completed products directly.
    """

    database_by_year: dict[int, str]
    timeline_database_by_year: dict[int, str]
    pa_receiving_reference: PaReceivingReference
    pa_receiving_sources_by_year: dict[int, PaReceivingSources]


@dataclass(frozen=True, slots=True)
class MaterialCarbon:
    """Prepared carbon in kg C/kg constituent, with immutable numerical values.

    ``constituents`` follows the packaged (material, constituent) order.
    ``fractions`` has axes constituent, virgin feedstock route, ultimate origin;
    both origin axes follow ``CARBON_ORIGINS``. The content identity includes
    source provenance and calculation conventions as well as numerical values.
    """

    identity: str
    constituents: tuple[tuple[str, str], ...]
    fractions: tuple[tuple[tuple[float, float, float], ...], ...]


@dataclass(frozen=True, slots=True)
class ProviderRoute:
    """One material component of an MFA-to-LCA provider route.

    The record identifies material, valid years, service activity and component
    share. Treatment routes additionally retain their fate, infiltration, tire
    SBR, and proxy-component information so foreground
    burdens are assigned once to the intended provider activity.

    Attributes:
        route_id: Identifier shared by the additive components of one logical
            MFA-to-LCA route.
        route_component_id: Identifier of this additive component. It also
            determines the stable order returned by :meth:`ProviderRoutes.resolve`.
        provider_demand_type: Provider demand represented by the route, such as
            resin production, conversion, recycling, or treatment.
        material: MFA plastic material to which the route applies.
        constituent: Exact constituent for production, conversion and treatment.
        product: Polymer, ethylene or propylene reference quantity.
        mapping_basis: Mass basis that connects the MFA demand to the
            provider service.
        valid_from_year: First activity year for which the route is valid,
            inclusive.
        valid_to_year: Last activity year for which the route is valid,
            inclusive.
        component_role: Life-cycle responsibility assigned to this provider
            component.
        component_material: Material represented by the provider component,
            including a disclosed analog material where applicable.
        virgin_feedstock_route: Source production route, retained for recovered
            TPS to select its original formulation; ``"all"`` for
            operations independent of that selection.
        ultimate_carbon_origin: Carbon origin selecting treatment or recovered-feed
            inventories; ``"all"`` for operations independent of carbon origin.
        share: Additive route share multiplied by the MFA demand.
        throughput_multiplier: Throughput factor applied after
            the route share.
        provider_service_id: Stable identifier of the prepared provider service.
        provider_service_code: Brightway activity code of that service in the
            prepared interface database.
        lca_fate: Treatment fate represented by the component, or an empty value
            when no fate distinction applies.
        infiltration_class: Open-dump infiltration class, or an empty value for
            routes without an infiltration portfolio.
        material_component_share: Share of the MFA material mapped
            to the provider material before any infiltration weighting.
        infiltration_share: Share of the fixed infiltration
            portfolio assigned to this component. The same portfolio applies
            to all 12 world regions and years.
        provider_class: Methodological classification of the selected provider,
            including whether it is direct or an analog.
        analogue_status: Specific analog disclosure for the selected provider,
            or an empty value for routes without an analog qualification.
        resolution_status: Source-specific statement of how the provider route
            was resolved.
        material_functional_unit: Declared material basis for a source-specific
            treatment method, or an empty value when no additional basis applies.
        styrene_component_share: Styrene-derived mass share recorded
            for an SBR treatment method.
        butadiene_analogue_component_share: Butadiene-derived mass
            share represented by the disclosed treatment analog.
        sbr_share_in_material: Dry styrene-butadiene rubber (SBR)
            mass share of the MFA material.
        sbr_proxy_component_role: Role of the provider component in the disclosed
            SBR treatment proxy.
        whole_tyre_constituents_excluded: Whether constituents outside the dry SBR
            fraction are excluded from this route.
        no_treatment_sensitivity_executed: Whether the source-specific treatment
            record declares that no alternative treatment sensitivity was run.
        source_role: Declared inventory-source role from which the provider
            service was prepared.
    """

    route_id: str
    route_component_id: str
    provider_demand_type: str
    material: str
    constituent: str
    product: str
    mapping_basis: str
    valid_from_year: int
    valid_to_year: int
    component_role: str
    component_material: str
    virgin_feedstock_route: str
    ultimate_carbon_origin: str
    share: float
    throughput_multiplier: float
    provider_service_id: str
    provider_service_code: str
    lca_fate: str
    infiltration_class: str
    material_component_share: float
    infiltration_share: float
    provider_class: str
    analogue_status: str
    resolution_status: str
    material_functional_unit: str
    styrene_component_share: float
    butadiene_analogue_component_share: float
    sbr_share_in_material: float
    sbr_proxy_component_role: str
    whole_tyre_constituents_excluded: bool
    no_treatment_sensitivity_executed: bool
    source_role: str


class ProviderRoutes:
    """Indexed mappings from MFA burdens to provider services."""

    def __init__(
        self,
        routes: pd.DataFrame,
        *,
        identity: str,
        interface_database: str,
        material_carbon: MaterialCarbon,
    ) -> None:
        """Index the provider routes written by ``prepare_backgrounds``.

        Args:
            routes: Provider route table. Each row links an MFA demand and validity period
                to one additive provider service component.
            identity: Content identifier of the complete prepared provider route table.
            interface_database: Brightway database containing the provider services.
            material_carbon: Constituent carbon admitted from the same shared product.

        Notes:
            Required text, share, numeric, and Boolean columns may contain null
            values. Those values receive their documented defaults before rows are
            converted to :class:`ProviderRoute`. Routes are then
            indexed by demand type, material, constituent, product and mapping basis.
            The two origin selections remain independent. Constituent coordinates use their declared
            prepared provider routes.
        """
        normalized = routes.copy()
        for column in _ROUTE_TEXT_COLUMNS:
            normalized[column] = normalized[column].fillna("")
        # All-null Parquet numeric columns arrive as object dtype. Convert them
        # instead of relying on pandas' deprecated implicit downcast.
        for column in _ROUTE_SHARE_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column]).fillna(1.0)
        for column in _ROUTE_ZERO_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column]).fillna(0.0)
        for column in _ROUTE_BOOLEAN_COLUMNS:
            normalized[column] = normalized[column].astype("boolean").fillna(False).astype(bool)
        self.routes = normalized
        self.identity = identity
        self.interface_database = interface_database
        self.material_carbon = material_carbon
        groups: dict[tuple[str, str, str, str, str], list[ProviderRoute]] = {}
        for raw in normalized.itertuples(index=False):
            row = cast(Any, raw)
            route = ProviderRoute(
                route_id=str(row.route_id),
                route_component_id=str(row.route_component_id),
                provider_demand_type=str(row.provider_demand_type),
                material=str(row.material),
                constituent=str(row.constituent),
                product=str(row.product),
                mapping_basis=str(row.mapping_basis),
                valid_from_year=int(row.valid_from_year),
                valid_to_year=int(row.valid_to_year),
                component_role=str(row.component_role),
                component_material=str(row.component_material),
                virgin_feedstock_route=str(row.virgin_feedstock_route),
                ultimate_carbon_origin=str(row.ultimate_carbon_origin),
                share=float(row.share),
                throughput_multiplier=float(row.throughput_multiplier),
                provider_service_id=str(row.provider_service_id),
                provider_service_code=str(row.provider_service_code),
                lca_fate=str(row.lca_fate),
                infiltration_class=str(row.infiltration_class),
                material_component_share=float(row.material_component_share),
                infiltration_share=float(row.infiltration_share),
                provider_class=str(row.provider_class),
                analogue_status=str(row.analogue_status),
                resolution_status=str(row.resolution_status),
                material_functional_unit=str(row.material_functional_unit),
                styrene_component_share=float(row.styrene_component_share),
                butadiene_analogue_component_share=float(row.butadiene_analogue_component_share),
                sbr_share_in_material=float(row.sbr_share_in_material),
                sbr_proxy_component_role=str(row.sbr_proxy_component_role),
                whole_tyre_constituents_excluded=bool(row.whole_tyre_constituents_excluded),
                no_treatment_sensitivity_executed=bool(row.no_treatment_sensitivity_executed),
                source_role=str(row.source_role),
            )
            key = (
                route.provider_demand_type,
                route.material,
                route.constituent,
                route.product,
                route.mapping_basis,
            )
            groups.setdefault(key, []).append(route)
        self._groups = {key: tuple(values) for key, values in groups.items()}
        self._resolved: dict[tuple[object, ...], tuple[ProviderRoute, ...]] = {}

    def open_dump_shares(self) -> dict[str, float]:
        """Return the original source's fixed conditional infiltration portfolio.

        Shared preparation applies this same five-class portfolio to every
        material, region and year. Read one copy of each completed class weight;
        actual PA events apply it to their full mixture before receiving.
        The stored source-market shares are applied across the modeled regions.
        """
        dumping = self.routes.loc[
            self.routes["provider_demand_type"].eq("mismanaged_open_dump_treatment"),
            ["infiltration_class", "infiltration_share"],
        ].drop_duplicates("infiltration_class")
        return {
            str(name): float(share) for name, share in dumping.itertuples(index=False, name=None)
        }

    def resolve(
        self,
        *,
        provider_demand_type: str,
        material: str,
        constituent: str,
        product: str,
        year: int,
        mapping_basis: str,
        virgin_feedstock_route: str,
        ultimate_carbon_origin: str,
    ) -> tuple[ProviderRoute, ...]:
        """Return the applicable additive service components.

        Args:
            provider_demand_type: Provider demand represented by the MFA burden.
            material: MFA plastic material of the burden.
            constituent: Actual constituent, including for processing conversion.
            product: Polymer or named monomer.
            year: Activity year used against the inclusive route validity
                interval.
            mapping_basis: Mass basis linking the burden to its provider
                service.
            virgin_feedstock_route: Source production technology, including the
                historical TPS formulation, independently of ultimate carbon origin.
            ultimate_carbon_origin: Actual origin controlling treatment/recovered supply.

        Returns:
            Matching route components ordered by ``route_component_id``.

        Raises:
            LookupError: If an internal demand has no matching prepared provider route.

        Notes:
            Prepared routes apply to all 12 world regions and goods. Demand
            records retain that attribution. Resolved tuples are cached by every
            remaining argument; the prepared-route content identifier is unchanged.
        """

        key = (
            provider_demand_type,
            material,
            constituent,
            product,
            year,
            mapping_basis,
            virgin_feedstock_route,
            ultimate_carbon_origin,
        )
        cached = self._resolved.get(key)
        if cached is not None:
            return cached
        rows = tuple(
            row
            for row in self._groups.get(
                (
                    provider_demand_type,
                    material,
                    constituent,
                    product,
                    mapping_basis,
                ),
                (),
            )
            if row.valid_from_year <= year <= row.valid_to_year
            and row.virgin_feedstock_route in (virgin_feedstock_route, "all")
            and row.ultimate_carbon_origin in (ultimate_carbon_origin, "all")
        )
        if not rows:
            raise LookupError(f"No provider route for {key!r}")
        result = tuple(sorted(rows, key=lambda row: row.route_component_id))
        self._resolved[key] = result
        return result


def _material_carbon(table: pa.Table, *, identity: str) -> MaterialCarbon:
    """Arrange prepared constituent carbon on the calculation's constituent/origin axes."""
    numerical_columns = [f"{origin}_carbon" for origin in CARBON_ORIGINS]
    frame = table.to_pandas()
    constituents = tuple(
        (item.material, item.constituent) for item in load_material_composition().constituents
    )
    expected = [(*item, route) for item in constituents for route in CARBON_ORIGINS]
    coordinates = frame.set_index(["material", "constituent", "virgin_feedstock_route"])
    values = coordinates.loc[expected, numerical_columns].to_numpy(dtype=float)
    return MaterialCarbon(
        identity=f"{MATERIAL_CARBON_SCHEMA}::{identity}",
        constituents=constituents,
        fractions=tuple(
            tuple((float(row[0]), float(row[1]), float(row[2])) for row in matrix)
            for matrix in values.reshape(len(constituents), 3, 3)
        ),
    )


def load_provider_routes(shared: PreparedManifest) -> ProviderRoutes:
    """Read prepared routes and arrange constituent carbon for foreground calculation."""
    root = shared.path.parent
    manifest = json.loads((root / "provider_service_manifest.json").read_bytes())
    artifacts = cast(dict[str, dict[str, object]], shared.payload["artifacts"])
    carbon = _material_carbon(
        pq.read_table(root / "material_carbon.parquet"),
        identity=str(artifacts["material_carbon.parquet"]["scientific_identity"]),
    )
    return ProviderRoutes(
        pq.read_table(root / "provider_service_routes.parquet").to_pandas(),
        identity=str(manifest["identity"]),
        interface_database=str(manifest["interface_database"]),
        material_carbon=carbon,
    )


def prepared_store_directory(path: Path) -> Path:
    """Require the recorded Brightway store, allowing an intentional directory link."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PreparedDataError(
            f"Prepared Brightway store {path} is unavailable; run prepare_backgrounds first."
        ) from exc
    if not path.is_dir() or not resolved.is_dir():
        raise PreparedDataError(
            f"Prepared Brightway store {path} is not a directory. "
            "Preserve the product and prepare backgrounds in a new workspace."
        )
    return path


def prepared_brightway_names(workspace: Path, shared: PreparedManifest) -> BrightwayNames:
    """Locate the Brightway store and database names recorded by shared preparation."""
    data = shared.payload
    return BrightwayNames(
        store=prepared_store_directory(workspace / str(data["brightway_store_relative_path"])),
        project=str(data["brightway_project"]),
        biosphere_database=str(data["biosphere_database"]),
        source_database=str(data["source_database"]),
    )


def load_prepared_provider_product(
    shared: PreparedManifest,
    member: PreparedManifest,
) -> PreparedProviderProduct:
    """Read provider mappings and admitted PA sources from completed preparation."""
    frontiers = pq.read_table(
        member.path.parent / "provider_technology_crosswalk.parquet"
    ).to_pandas()
    timeline = cast(dict[str, str], shared.payload["provider_timeline_database_by_year"])
    pa_sources = tuple(
        sources_from_payload(payload)
        for payload in json.loads((member.path.parent / "pa_receiving_sources.json").read_bytes())
    )
    return PreparedProviderProduct(
        database_by_year={
            int(year): str(database)
            for year, database in frontiers.loc[:, ["anchor_year", "database_name"]].itertuples(
                index=False, name=None
            )
        },
        timeline_database_by_year={int(year): str(database) for year, database in timeline.items()},
        pa_receiving_reference=reference_from_payload(
            json.loads((shared.path.parent / "pa_receiving_reference.json").read_bytes())
        ),
        pa_receiving_sources_by_year={source.provider_anchor_year: source for source in pa_sources},
    )
