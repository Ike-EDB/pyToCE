"""Background-independent regional foreground construction."""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import fsum
from operator import attrgetter
from typing import Any, cast

import numpy as np
import pandas as pd
from bw_temporalis import TemporalDistribution  # type: ignore[import-untyped]

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.run_mfa.model.mfa import MFACandidateEvaluation, MFAEvaluation
from pytoce.shared.runtime.brightway import database_matches, write_brightway_database
from pytoce.shared.runtime.file_lock import writer_lock
from pytoce.shared.runtime.filesystem import canonical_table_sha256, payload_sha256

from .burdens import (
    BURDEN_METHOD_IDENTITY,
    build_burden_demands,
    mechanical_recycling_loss_result,
)
from .custody import (
    CARBON_CUSTODY_METHOD_IDENTITY,
    CarbonCustodyHistory,
    CohortWeightsByYear,
    trace_carbon_custody,
)
from .flow_coverage import FLOW_OWNERSHIP_METHOD_IDENTITY, mfa_to_lca_flow_coverage
from .pa_receiving import PaDisposalEvent, PaRecyclingEvent, PaSpinningEvent
from .providers import ProviderRoute, ProviderRoutes

H12_REGIONS = (
    "CAZ",
    "CHA",
    "EUR",
    "IND",
    "JPN",
    "LAM",
    "MEA",
    "NEU",
    "OAS",
    "REF",
    "SSA",
    "USA",
)

FOREGROUND_ACTIVATION_METHOD_IDENTITY = "pytoce_constituent_foreground_v1"
OPTIMIZATION_ACTIVATION_METHOD_IDENTITY = "pytoce_constituent_optimization_activation_v2"
_FOREGROUND_IDENTITY_FIELD = "pytoce_foreground_identity"

_DEMAND_ACTIVATION_COLUMNS = (
    "foreground_scenario",
    "year",
    "region",
    "material",
    "constituent",
    "product",
    "virgin_feedstock_route",
    "ultimate_carbon_origin",
    "origin_fraction",
    "good",
    "custody",
    "demand_type",
    "provider_demand_type",
    "material_cycle_source",
    "basis_flow_names",
    "custody_status",
    "source_parent_amount_kg",
    "source_allocation_share",
    "mapping_basis",
    "lca_boundary",
    "burden_attribution_basis",
    "stage_group",
    "cutoff_cycle_role",
    "physical_event_basis",
    "physical_event_year",
    "lca_fate",
    "fate_origin",
    "allocation_system_model",
)

_ACTIVATION_COLUMNS = (
    *_DEMAND_ACTIVATION_COLUMNS,
    "route_id",
    "route_component_id",
    "provider_service_id",
    "provider_role",
    "provider_material",
    "infiltration_class",
    "material_component_share",
    "infiltration_share",
    "provider_class",
    "analogue_status",
    "throughput_multiplier",
    "resolution_status",
    "material_functional_unit",
    "styrene_component_share",
    "butadiene_analogue_component_share",
    "sbr_share_in_material",
    "sbr_proxy_component_role",
    "whole_tyre_constituents_excluded",
    "no_treatment_sensitivity_executed",
    "declared_provider_source_role",
    "branch_lineage_id",
    "provider_service_database",
    "provider_service_code",
)

OPTIMIZATION_ACTIVATION_COLUMNS = (
    "year",
    "region",
    "demand_type",
    "provider_service_id",
    "provider_service_code",
    "stage_group",
    "cutoff_cycle_role",
    "material_cycle_source",
    "physical_event_basis",
    "physical_event_year",
    "amount",
)
OPTIMIZATION_ACTIVATION_DELTA_COLUMNS = (
    *(
        "life_cycle_stage" if column == "demand_type" else column
        for column in OPTIMIZATION_ACTIVATION_COLUMNS[:-1]
    ),
    "parent_amount_kg",
    "candidate_amount_kg",
    "delta_amount_kg",
)


@dataclass(frozen=True, slots=True)
class ForegroundActivations:
    """Provider activations and complete PA process/fate mixtures before attribution."""

    activations: pd.DataFrame
    pa_spinning_events: tuple[PaSpinningEvent, ...]
    pa_recycling_events: tuple[PaRecyclingEvent, ...]
    pa_disposal_events: tuple[PaDisposalEvent, ...]


@dataclass(frozen=True, slots=True)
class ForegroundProduct:
    """Compiled foreground data for one completed MFA result.

    Attributes:
        foreground_identity: Content identifier of the compiled foreground data.
        database_name: Brightway database containing the 12 regional roots.
        root_keys: Brightway database and code for each regional root.
        activations: Annual provider demands in their provider functional units.
        mfa_to_lca_flow_coverage: MFA burden flows and their assigned LCA stages.
        mechanical_recycling_loss: Mechanical recycling input, output and loss in kg.
        pa_spinning_events: Actual PA manufacturing quantities before origin attribution.
        pa_recycling_events: Actual PA washing, recovery and airborne exits.
        pa_disposal_events: Wet end-of-life fiber and dry recycling-reject mixtures.
    """

    foreground_identity: str
    database_name: str
    root_keys: tuple[tuple[str, str], ...]
    activations: pd.DataFrame
    mfa_to_lca_flow_coverage: pd.DataFrame
    mechanical_recycling_loss: pd.DataFrame
    pa_spinning_events: tuple[PaSpinningEvent, ...]
    pa_recycling_events: tuple[PaRecyclingEvent, ...]
    pa_disposal_events: tuple[PaDisposalEvent, ...]


@dataclass(frozen=True, slots=True)
class OptimizationActivationContext:
    """Lean parent endpoint plus immutable historical carbon-origin state."""

    identity: str
    activations: pd.DataFrame
    carbon_history: CarbonCustodyHistory
    pa_spinning_events: tuple[PaSpinningEvent, ...]
    pa_recycling_events: tuple[PaRecyclingEvent, ...]
    pa_disposal_events: tuple[PaDisposalEvent, ...]


def optimization_activation_context_identity(
    activations: pd.DataFrame,
    carbon_history: CarbonCustodyHistory,
    pa_spinning_events: tuple[PaSpinningEvent, ...],
    pa_recycling_events: tuple[PaRecyclingEvent, ...],
    pa_disposal_events: tuple[PaDisposalEvent, ...],
) -> str:
    """Return a content identifier for the baseline activations and their history."""

    return "pytoce_optimization_activation_context_v1::" + payload_sha256(
        {
            "method_identity": OPTIMIZATION_ACTIVATION_METHOD_IDENTITY,
            "carbon_custody_method_identity": CARBON_CUSTODY_METHOD_IDENTITY,
            "activation_identity": canonical_table_sha256(activations),
            "pa_spinning_events": [asdict(event) for event in pa_spinning_events],
            "pa_recycling_events": [asdict(event) for event in pa_recycling_events],
            "pa_disposal_events": [asdict(event) for event in pa_disposal_events],
            "history_through_year": carbon_history.through_year,
            "history_entries": [list(entry) for entry in carbon_history.entries],
            "tps_route_history_entries": [
                list(entry) for entry in carbon_history.tps_route_entries
            ],
            "pa_finish_history_entries": [
                list(entry) for entry in carbon_history.pa_finish_entries
            ],
        }
    )


def foreground_database_name(foreground_scenario: str, foreground_identity: str) -> str:
    """Keep distinct scientific foregrounds for the same readable scenario name."""

    digest = foreground_identity.split("::", maxsplit=1)[1]
    return f"pytoce_foreground_{foreground_scenario}__{digest}"


def _root_code(database: str, region: str) -> str:
    """Build the Brightway code for one foreground root activity for a world region."""
    scenario = database.removeprefix("pytoce_foreground_")
    return f"root_{scenario}_{region.casefold()}"


def _foreground_identity(
    mfa_result_identity: str,
    provider_identity: str,
    material_carbon_identity: str,
    provider_interface_database: str,
) -> str:
    """Identify the foreground database from its complete input records."""
    payload = {
        "method": FOREGROUND_ACTIVATION_METHOD_IDENTITY,
        "mfa_result_identity": mfa_result_identity,
        "provider_service_identity": provider_identity,
        "provider_interface_database": provider_interface_database,
        "material_carbon_identity": material_carbon_identity,
        "burden_method_identity": BURDEN_METHOD_IDENTITY,
        "flow_ownership_method_identity": FLOW_OWNERSHIP_METHOD_IDENTITY,
        "carbon_custody_method_identity": CARBON_CUSTODY_METHOD_IDENTITY,
        "regions": H12_REGIONS,
    }
    digest = sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{FOREGROUND_ACTIVATION_METHOD_IDENTITY}::{digest}"


def _production(database: str, code: str) -> dict[str, object]:
    """Build the reference production exchange for one foreground activity."""
    return {
        "input": (database, code),
        "amount": 1.0,
        "type": "production",
    }


def _stable_sum(values: pd.Series) -> float:
    """Sum each group's values with accurate floating-point accumulation."""

    return fsum(float(value) for value in values)


def _branch_lineage_id(demand: object, component: object) -> str:
    """Identify a constituent/provider branch independently of year and amount."""
    demand_item = cast(Any, demand)
    component_item = cast(Any, component)
    payload = (
        str(demand_item.foreground_scenario),
        str(demand_item.region),
        str(demand_item.material),
        str(demand_item.constituent),
        str(demand_item.product),
        str(demand_item.virgin_feedstock_route),
        str(demand_item.ultimate_carbon_origin),
        str(demand_item.good),
        str(demand_item.demand_type),
        str(demand_item.material_cycle_source),
        str(component_item.route_id),
        str(component_item.route_component_id),
        str(component_item.provider_service_code),
    )
    return sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _provider_allocations(
    demands: pd.DataFrame,
    routes: ProviderRoutes,
) -> Iterator[tuple[Any, ProviderRoute, float, float]]:
    """Resolve attributed demands once for both detailed and optimization results.

    Provider shares and throughput scale the service amount. Carbon is already
    attributed to the event and origin, so only its additive provider share is
    applied. Native provider production signs stay inside the prepared activity.
    """
    for raw_demand in demands.itertuples(index=False):
        demand = cast(Any, raw_demand)
        components = routes.resolve(
            provider_demand_type=str(demand.provider_demand_type),
            material=str(demand.material),
            constituent=str(demand.constituent),
            product=str(demand.product),
            year=int(demand.year),
            mapping_basis=str(demand.mapping_basis),
            virgin_feedstock_route=str(demand.virgin_feedstock_route),
            ultimate_carbon_origin=str(demand.ultimate_carbon_origin),
        )
        for component in components:
            amount = float(demand.amount_kg) * component.share * component.throughput_multiplier
            if amount != 0:
                yield (
                    demand,
                    component,
                    amount,
                    float(demand.mfa_carbon_amount_kg) * component.share,
                )


def _activation_ledger(
    demands: pd.DataFrame,
    routes: ProviderRoutes,
    *,
    provider_service_database: str,
) -> pd.DataFrame:
    """Retain constituent, technology and carbon attribution in provider activations."""
    rows: list[dict[str, object]] = []
    for demand, component, amount, carbon in _provider_allocations(demands, routes):
        rows.append(
            {
                **{column: getattr(demand, column) for column in _DEMAND_ACTIVATION_COLUMNS},
                "route_id": component.route_id,
                "route_component_id": component.route_component_id,
                "provider_service_id": component.provider_service_id,
                "provider_role": component.component_role,
                "provider_material": component.component_material,
                "infiltration_class": component.infiltration_class,
                "material_component_share": component.material_component_share,
                "infiltration_share": component.infiltration_share,
                "provider_class": component.provider_class,
                "analogue_status": component.analogue_status,
                "throughput_multiplier": component.throughput_multiplier,
                "resolution_status": component.resolution_status,
                "material_functional_unit": component.material_functional_unit,
                "styrene_component_share": component.styrene_component_share,
                "butadiene_analogue_component_share": component.butadiene_analogue_component_share,
                "sbr_share_in_material": component.sbr_share_in_material,
                "sbr_proxy_component_role": component.sbr_proxy_component_role,
                "whole_tyre_constituents_excluded": component.whole_tyre_constituents_excluded,
                "no_treatment_sensitivity_executed": component.no_treatment_sensitivity_executed,
                "declared_provider_source_role": component.source_role,
                "branch_lineage_id": _branch_lineage_id(demand, component),
                "provider_service_database": provider_service_database,
                "provider_service_code": component.provider_service_code,
                "amount": amount,
                "mfa_carbon_amount_kg": carbon,
            }
        )
    return (
        pd.DataFrame(rows, columns=[*_ACTIVATION_COLUMNS, "amount", "mfa_carbon_amount_kg"])
        .groupby(
            list(_ACTIVATION_COLUMNS),
            observed=True,
            sort=True,
            dropna=False,
        )[["amount", "mfa_carbon_amount_kg"]]
        .agg(_stable_sum)
        .reset_index()
    )


def _optimization_activation_ledger(demands: pd.DataFrame, routes: ProviderRoutes) -> pd.DataFrame:
    """Aggregate resolved provider allocations for candidate evaluation and reporting."""
    amount_lists: dict[tuple[object, ...], list[float]] = {}
    for demand, component, amount, _ in _provider_allocations(demands, routes):
        key = (
            int(demand.year),
            str(demand.region),
            str(demand.demand_type),
            component.provider_service_id,
            component.provider_service_code,
            str(demand.stage_group),
            str(demand.cutoff_cycle_role),
            str(demand.material_cycle_source),
            str(demand.physical_event_basis),
            int(demand.physical_event_year),
        )
        amount_lists.setdefault(key, []).append(amount)
    rows = [(*key, fsum(values)) for key, values in sorted(amount_lists.items())]
    return pd.DataFrame.from_records(rows, columns=OPTIMIZATION_ACTIVATION_COLUMNS)


def optimization_activation_delta(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    """Return candidate-minus-parent endpoints at the exact LCIA/stage grain."""

    keys = OPTIMIZATION_ACTIVATION_COLUMNS[:-1]
    output_columns = list(OPTIMIZATION_ACTIVATION_DELTA_COLUMNS)
    joined = parent.rename(columns={"amount": "parent_amount_kg"}).merge(
        candidate.rename(columns={"amount": "candidate_amount_kg"}),
        on=list(keys),
        how="outer",
    )
    joined[["parent_amount_kg", "candidate_amount_kg"]] = joined[
        ["parent_amount_kg", "candidate_amount_kg"]
    ].fillna(0.0)
    joined["delta_amount_kg"] = joined["candidate_amount_kg"] - joined["parent_amount_kg"]
    result = joined.loc[joined["delta_amount_kg"].ne(0.0)].copy()
    result = result.rename(columns={"demand_type": "life_cycle_stage"})
    ordered_keys = tuple(
        "life_cycle_stage" if column == "demand_type" else column for column in keys
    )
    return result.loc[:, output_columns].sort_values(
        list(ordered_keys),
        kind="stable",
        na_position="first",
        ignore_index=True,
    )


def _stage_distribution(annual: pd.DataFrame) -> TemporalDistribution:
    """Create a temporal distribution from annual event years and amounts."""
    return TemporalDistribution(
        date=np.array([np.datetime64(f"{int(year)}-01-01T00:00:00") for year in annual["year"]]),
        amount=annual["amount"].to_numpy(dtype=np.float64),
    )


def _foreground_data(
    *,
    database: str,
    foreground_scenario: str,
    mfa_result_identity: str,
    provider_identity: str,
    activations: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Compile exactly twelve regional roots and their annual service demands."""

    data: dict[tuple[str, str], dict[str, Any]] = {}
    for region in H12_REGIONS:
        code = _root_code(database, region)
        exchanges: list[dict[str, object]] = [_production(database, code)]
        selected = activations.loc[activations["region"].eq(region)]
        for raw_key, annual in selected.groupby(
            ["provider_service_database", "provider_service_code"],
            observed=True,
            sort=True,
        ):
            service_database, service_code = cast(tuple[str, str], raw_key)
            annual_amounts = (
                annual.groupby("year", observed=True, sort=True)["amount"]
                .agg(_stable_sum)
                .reset_index()
            )
            exchanges.append(
                {
                    "input": (service_database, service_code),
                    "amount": 1.0,
                    "type": "technosphere",
                    "temporal_distribution": _stage_distribution(annual_amounts),
                    "exchange_role": "physical_life_cycle_stage",
                    "provider_service_code": service_code,
                }
            )
        data[(database, code)] = {
            "name": f"Regional plastics system, {region}",
            "reference product": "regional plastics system",
            "unit": "unit",
            "location": region,
            "type": "process",
            "activity_kind": "regional_root",
            "foreground_scenario": foreground_scenario,
            "region": region,
            "mfa_result_identity": mfa_result_identity,
            "provider_service_identity": provider_identity,
            "exchanges": exchanges,
        }
    return data


def compile_foreground_activations(
    mfa: MFAEvaluation | MFACandidateEvaluation,
    routes: ProviderRoutes,
    recipe: EcoinventMaterialRecipe,
) -> ForegroundActivations:
    """Compile detailed provider demands and actual PA mixtures."""

    trace = trace_carbon_custody(mfa, routes.material_carbon, recipe)
    demands = build_burden_demands(mfa, routes.material_carbon, trace)
    return ForegroundActivations(
        _activation_ledger(
            demands.provider_demands,
            routes,
            provider_service_database=routes.interface_database,
        ),
        demands.pa_spinning_events,
        demands.pa_recycling_events,
        demands.pa_disposal_events,
    )


def compile_optimization_activation_context(
    mfa: MFAEvaluation | MFACandidateEvaluation,
    routes: ProviderRoutes,
    recipe: EcoinventMaterialRecipe,
) -> OptimizationActivationContext:
    """Compile the lean parent LCIA endpoint and reusable exact 2019 history."""

    trace = trace_carbon_custody(mfa, routes.material_carbon, recipe)
    demands = build_burden_demands(mfa, routes.material_carbon, trace)
    activations = _optimization_activation_ledger(demands.provider_demands, routes)
    return OptimizationActivationContext(
        identity=optimization_activation_context_identity(
            activations,
            trace.history,
            demands.pa_spinning_events,
            demands.pa_recycling_events,
            demands.pa_disposal_events,
        ),
        activations=activations,
        carbon_history=trace.history,
        pa_spinning_events=demands.pa_spinning_events,
        pa_recycling_events=demands.pa_recycling_events,
        pa_disposal_events=demands.pa_disposal_events,
    )


def compile_optimization_activations_from_history(
    mfa: MFAEvaluation | MFACandidateEvaluation,
    routes: ProviderRoutes,
    recipe: EcoinventMaterialRecipe,
    parent: OptimizationActivationContext,
    *,
    cohort_weights_by_year: CohortWeightsByYear,
) -> ForegroundActivations:
    """Compile prospective candidate LCIA endpoints."""

    trace = trace_carbon_custody(
        mfa,
        routes.material_carbon,
        recipe,
        history=parent.carbon_history,
        cohort_weights_by_year=cohort_weights_by_year,
    )
    demands = build_burden_demands(mfa, routes.material_carbon, trace)
    future = _optimization_activation_ledger(demands.provider_demands, routes)
    historical = parent.activations.loc[
        parent.activations["year"] <= parent.carbon_history.through_year
    ]
    activations = (
        historical.copy() if future.empty else pd.concat((historical, future), ignore_index=True)
    )
    historical_events = tuple(
        event
        for event in parent.pa_spinning_events
        if event.year <= parent.carbon_history.through_year
    )
    return ForegroundActivations(
        activations.sort_values(
            list(OPTIMIZATION_ACTIVATION_COLUMNS[:-1]),
            kind="stable",
            na_position="first",
            ignore_index=True,
        ),
        tuple(
            sorted(
                (*historical_events, *demands.pa_spinning_events),
                key=attrgetter("year", "region", "constituent"),
            )
        ),
        tuple(
            sorted(
                (
                    *(
                        event
                        for event in parent.pa_recycling_events
                        if event.year <= parent.carbon_history.through_year
                    ),
                    *demands.pa_recycling_events,
                ),
                key=attrgetter("year", "region", "constituent"),
            )
        ),
        tuple(
            sorted(
                (
                    *(
                        event
                        for event in parent.pa_disposal_events
                        if event.year <= parent.carbon_history.through_year
                    ),
                    *demands.pa_disposal_events,
                ),
                key=attrgetter("year", "region", "constituent", "demand_type"),
            )
        ),
    )


def build_foreground_database(
    foreground_scenario: str,
    *,
    names: BrightwayNames,
    routes: ProviderRoutes,
    mfa_result_identity: str,
    mfa: MFAEvaluation,
    recipe: EcoinventMaterialRecipe,
) -> ForegroundProduct:
    """Create or reuse twelve roots from admitted MFA, provider routes and project names."""

    mfa_identity = mfa_result_identity
    foreground_identity = _foreground_identity(
        mfa_identity, routes.identity, routes.material_carbon.identity, routes.interface_database
    )
    database = foreground_database_name(foreground_scenario, foreground_identity)
    roots = tuple((database, _root_code(database, region)) for region in H12_REGIONS)

    flows = mfa.model_flows
    flow_coverage = mfa_to_lca_flow_coverage(
        mfa,
        foreground_scenario=foreground_scenario,
        background_scenario="",
    )
    mechanical_recycling_loss = mechanical_recycling_loss_result(flows)
    compiled = compile_foreground_activations(mfa, routes, recipe)
    activations = compiled.activations
    data = _foreground_data(
        database=database,
        foreground_scenario=foreground_scenario,
        mfa_result_identity=mfa_identity,
        provider_identity=routes.identity,
        activations=activations,
    )
    del flows
    names.store.mkdir(parents=True, exist_ok=True)
    with writer_lock(names.writer_lock_path), selected_brightway_project(names) as bd:
        if not database_matches(
            bd,
            database,
            {_FOREGROUND_IDENTITY_FIELD: foreground_identity},
        ):
            write_brightway_database(
                bd,
                database,
                data,
                {_FOREGROUND_IDENTITY_FIELD: foreground_identity},
            )
    return ForegroundProduct(
        foreground_identity=foreground_identity,
        database_name=database,
        root_keys=roots,
        activations=activations,
        mfa_to_lca_flow_coverage=flow_coverage,
        mechanical_recycling_loss=mechanical_recycling_loss,
        pa_spinning_events=compiled.pa_spinning_events,
        pa_recycling_events=compiled.pa_recycling_events,
        pa_disposal_events=compiled.pa_disposal_events,
    )
