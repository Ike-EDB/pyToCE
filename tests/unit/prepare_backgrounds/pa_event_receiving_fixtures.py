"""Complete native PA receiving graph shared by admission and actual-event tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import log
from pathlib import Path
from typing import Any, cast

import pytest
from constituent_mfa_fixtures import empty_model, evaluate_mfa
from material_recipe_fixtures import physical_material_recipe

from pytoce.download_data.sources.ecoinvent_materials import EcoinventMaterialRecipe
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.member_preparation import (
    _prepare_pa_receiving_sources,
)
from pytoce.prepare_backgrounds.preparation.prepared_state.shared_preparation import (
    SharedAuthority,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
    admit_pa_receiving_reference,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.run_lca.calculation.burdens import BurdenDemands, build_burden_demands
from pytoce.run_lca.calculation.custody import trace_carbon_custody
from pytoce.run_lca.calculation.providers import MaterialCarbon
from tests.pa_municipal_fixtures import pa_municipal_biosphere, pa_municipal_native_inventory
from tests.unit.prepare_backgrounds.chemical_residue_fixtures import (
    apply_premise_chemical_receiving_metadata,
)
from tests.unit.prepare_backgrounds.native_chemical_fixtures import (
    native_chemical_biosphere,
    native_chemical_inventory,
)
from tests.unit.prepare_backgrounds.pa_dump_receiving_fixtures import (
    dump_biosphere,
    dump_native_inventory,
)
from tests.unit.prepare_backgrounds.pa_landfill_receiving_fixtures import (
    _biosphere as landfill_biosphere_inventory,
)
from tests.unit.prepare_backgrounds.pa_landfill_receiving_fixtures import (
    _native as landfill_native_inventory,
)
from tests.unit.prepare_backgrounds.pa_open_burning_fixtures import (
    open_burning_biosphere,
    open_burning_native_inventory,
)
from tests.unit.prepare_backgrounds.pa_recycling_receiving_fixtures import (
    RECYCLING_SERVICES,
    recycling_biosphere,
    recycling_native_inventory,
)
from tests.unit.prepare_backgrounds.pa_water_receiving_fixtures import _dated_water_inventory
from tests.unit.prepare_backgrounds.tps_gas_fixtures import (
    biosphere_inventory as gas_biosphere_inventory,
)
from tests.unit.prepare_backgrounds.tps_gas_fixtures import (
    native_inventory as gas_native_inventory,
)


def _future_inventory(
    catalogue: dict[str, Any], database: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """Double native reference sizes and change one dated peroxide requirement.

    All quantities come from the invented chemical fixture. Doubling every
    exchange preserves its intensive duty; the additional 50% peroxide demand
    distinguishes dated operation from original composition calibration.
    """
    records = _dated_water_inventory(catalogue, database)
    for code in ("mc_hwi", "mc_fixed", "mc_ash"):
        for exchange in records[database, code]["exchanges"]:
            factor = 2.0
            if code == "mc_hwi" and exchange["input"][1] == "cr_peroxide":
                factor *= 1.5
            exchange["amount"] *= factor
            if "loc" in exchange:
                exchange["loc"] = (
                    exchange["loc"] + log(factor)
                    if exchange.get("uncertainty type") == 2
                    else exchange["loc"] * factor
                )
            for bound in ("minimum", "maximum"):
                if bound in exchange:
                    exchange[bound] *= factor
    return records


def _receiving_biosphere() -> tuple[
    dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], tuple[str, str]]
]:
    """Join valid chemical/gas fixtures with one key per elementary-flow descriptor."""
    records: dict[tuple[str, str], dict[str, Any]] = {}
    selected = {}
    flow_keys = {}
    for prefix, source in (
        (
            "",
            {
                ("biosphere", "pa_spinning_nmvoc"): {
                    "name": "NMVOC, non-methane volatile organic compounds",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
            },
        ),
        ("", native_chemical_biosphere()),
        ("", gas_biosphere_inventory()),
        ("", landfill_biosphere_inventory()[0]),
        ("recycling_source_", recycling_biosphere()),
        ("municipal_source_", pa_municipal_biosphere()),
        ("burning_source_", open_burning_biosphere("biosphere")),
        ("dump_source_", dump_biosphere("biosphere")),
    ):
        for key, row in source.items():
            key = key[0], prefix + key[1]
            descriptor = row["name"], row["unit"], tuple(row["categories"])
            selected_key = selected.setdefault(descriptor, key)
            flow_keys[key] = selected_key
            if selected_key == key:
                records[key] = row
    return records, flow_keys


def _receiving_inventory(
    catalogue: dict[str, Any],
    database: str,
    *,
    future: bool,
    gas_keys: dict[tuple[str, str], tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Combine complete invented receivers and their shared gas-support activities."""
    records = (
        _future_inventory(catalogue, database)
        if future
        else native_chemical_inventory(catalogue, database)
    )
    gas = gas_native_inventory(database, future=future)
    recycling = recycling_native_inventory(catalogue, database, future=future)
    for activity in gas.values():
        for row in activity["exchanges"]:
            if row["type"] == "biosphere":
                row["input"] = gas_keys[row["input"]]
    # The water fixture already has complete original gas functions. Replace
    # their matched external supplier identities with the gas test's dated
    # functions, keeping one exact descriptor and every existing graph link.
    gas_descriptors = {
        (row["name"], row["reference product"], row["unit"], row["location"]): key
        for key, row in gas.items()
    }
    aliases = {
        key: gas_descriptors[descriptor]
        for key, row in records.items()
        if (descriptor := (row["name"], row["reference product"], row["unit"], row["location"]))
        in gas_descriptors
    }
    records = {key: row for key, row in records.items() if key not in aliases}
    for activity in records.values():
        for row in activity["exchanges"]:
            row["input"] = (
                gas_keys[row["input"]]
                if row["type"] == "biosphere"
                else aliases.get(row["input"], row["input"])
            )
    records.update(gas)
    # Added fixtures retain complete native functions. Their matching
    # suppliers reuse this parent's completed graph; prefixes avoid collisions
    # between independent invented native-supply keys.
    for prefix, flow_prefix, additions in (
        ("landfill_source_", "", landfill_native_inventory(catalogue, database, future=future)),
        ("recycling_source_", "recycling_source_", recycling),
        (
            "municipal_source_",
            "municipal_source_",
            pa_municipal_native_inventory(catalogue, database, original=not future),
        ),
        *(
            (
                (
                    "burning_source_",
                    "burning_source_",
                    open_burning_native_inventory(
                        database, original=True, biosphere_database="biosphere"
                    ),
                ),
                (
                    "dump_source_",
                    "dump_source_",
                    dump_native_inventory(catalogue, database, biosphere_database="biosphere"),
                ),
            )
            if not future
            else ()
        ),
    ):
        existing = {
            (row["name"], row["reference product"], row["unit"], row["location"]): key
            for key, row in records.items()
        }
        added_aliases = {
            key: existing[descriptor]
            for key, row in additions.items()
            if (descriptor := (row["name"], row["reference product"], row["unit"], row["location"]))
            in existing
        }
        added_keys = {key: added_aliases.get(key, (database, prefix + key[1])) for key in additions}
        for key, activity in additions.items():
            if key in added_aliases:
                continue
            for row in activity["exchanges"]:
                row["input"] = (
                    gas_keys[row["input"][0], flow_prefix + row["input"][1]]
                    if row["type"] == "biosphere"
                    else added_keys[row["input"]]
                )
            records[added_keys[key]] = activity
    return records


@dataclass(frozen=True, slots=True)
class NativePaReceivingBoundary:
    """Complete disposable source/project state plus its detached admission records."""

    bd: Any
    context: _RecipeContext
    original_context: _RecipeContext
    reference: PaReceivingReference
    sources: PaReceivingSources
    services: dict[str, dict[str, object]]
    gas_keys: dict[tuple[str, str], tuple[str, str]]


@contextmanager
def pa_event_receiving_boundary(
    root: Path, *, original: bool
) -> Iterator[NativePaReceivingBoundary]:
    """Compile the complete PA source and prepared graphs from synthetic inputs.

    Shared original calibration, actual general-background operations, complete
    native HWI, water and terminal suppliers remain separate. No PA
    treatment result is manufactured or substituted by this fixture. Callers
    may retain its typed reference/sources after leaving the disposable project.
    """
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    recipe = physical_material_recipe()
    names = BrightwayNames(root / "brightway", "pa_receiving", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        biosphere, gas_keys = _receiving_biosphere()
        bd.Database("biosphere").write(biosphere, searchable=False)
        bd.Database("original").write(
            _receiving_inventory(catalogue, "original", future=False, gas_keys=gas_keys),
            searchable=False,
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                _receiving_inventory(catalogue, database, future=True, gas_keys=gas_keys),
                searchable=False,
            )
        else:
            # Early-provider ordinary PA receiving uses the original general-background peroxide
            # dose.
            bd.Database("technology").write(
                _receiving_inventory(catalogue, "technology", future=True, gas_keys=gas_keys),
                searchable=False,
            )
        if not original:
            apply_premise_chemical_receiving_metadata(bd, database)
            # This future fixture sets four optional uncertainty fields to NaN
            # while retaining the deterministic dose and negative=False.
            # Their presence is optional.
            ammonia = next(
                row
                for row in bd.get_node(database=database, code="mc_hwi").technosphere()
                if row.input["code"] == "mc_ammonia"
            )
            ammonia["uncertainty type"] = 0
            ammonia["loc"] = ammonia["amount"]
            ammonia["negative"] = False
            for field in ("scale", "shape", "minimum", "maximum"):
                ammonia[field] = float("nan")
            ammonia.save()
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario=None if original else "remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="frontier",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=recipe,
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        original_context = replace(
            context,
            background_scenario=None,
            year=2025,
            provider_database="original",
            ordinary=context.source,
            technology=context.source,
        )
        reference = admit_pa_receiving_reference(original_context)
        services = {
            str(service["provider_service_id"]): service
            for service in cast(list[dict[str, object]], catalogue["services"])
        }

        frontier_services = (
            "hydraulic_water_collection_row",
            "hydraulic_water_treatment_row",
            *RECYCLING_SERVICES,
        )
        frontier_records = {
            ("frontier", provider_service_code(service)): _service_dataset(
                context, services[service], services, "frontier"
            )
            for service in frontier_services
        }
        bd.Database("frontier").write(frontier_records, searchable=False)
        shared = SharedAuthority(
            names=names,
            identity="shared::fixture",
            source_identity="source::fixture",
            preparation_method_identity="method::fixture",
            lcia_identity="lcia::fixture",
            provider_identity=provider.identity,
            provider_interface_database=provider.interface_database,
            provider_timeline_identity="timeline::fixture",
            provider_timeline_database_by_year={},
            provider_routes=(),
            impact_categories=(),
            database_rows=(),
            matrix_rows=(),
            biosphere_rows=(),
            pa_receiving_reference=reference,
        )
        sources = _prepare_pa_receiving_sources(
            bd,
            shared=shared,
            provider=provider,
            material_recipe=recipe,
            background_scenario="remind_ssp2_npi",
            year=2025 if original else 2050,
            ordinary_database=database,
            technology_database="technology" if original else database,
            provider_database="frontier",
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )

        yield NativePaReceivingBoundary(
            bd, context, original_context, reference, sources, services, gas_keys
        )


@pytest.fixture(scope="module")
def mixed_pa_events(
    carbon_authorities: tuple[MaterialCarbon, EcoinventMaterialRecipe],
) -> BurdenDemands:
    """Retire three annual fiber cohorts, including a partly uncollected final cohort."""
    carbon, recipe = carbon_authorities
    state, inputs = empty_model("Fibres")
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    for time in (0, 1, 2):
        state.annual_inflow[time, region, good] = 1
        state.stock_by_cohort[time, time, region, good] = 1
        state.outflow_by_cohort[time + 1, time, region, good] = 1
    for time in (1, 2):
        inputs.mechanical_recycling_rate[time, region] = 0.5
        inputs.incineration_rate[time, region] = 0.25
    # New origin routes become eligible in 2020; the existing cohort retains
    # its historical origin rather than being relabeled by the new source mix.
    inputs.bio_production_rate[2, region] = 0.35
    inputs.daccu_production_rate[2, region] = 0.25
    inputs.collection_rate[3, region] = 0.6
    mfa = evaluate_mfa(state, inputs, recipe)
    trace = trace_carbon_custody(mfa, carbon, recipe)
    return build_burden_demands(mfa, carbon, trace)
