"""Transfer complete native treatment proxies to historical TPS wet quantities.

Municipal incineration and open dumping retain their selected MIXED inventories.
Only the functional quantity and direct waste-carbon origins change. Native emissions retain their
proxy values.
"""

import json
from math import fsum
from typing import cast

from .catalogue import provider_service_code
from .material_carbon import CARBON_ORIGINS
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _RecipeContext, _scaled_native_exchange
from .tps_residue_load import _tps_material_load
from .treatment_compilation import (
    _direct_carbon_owners_json,
    _matches_treatment_carbon_selector,
    _replace_carbon_identity,
)

# These are the native gas identities read by the principal-carbon diagnostic. A gas outside the
# declared waste-carbon mappings remains in the inventory but receives no principal fraction.
_CARBON_GASES = {
    f"{species}, {origin}"
    for species in ("Carbon dioxide", "Carbon monoxide", "Methane")
    for origin in ("fossil", "non-fossil")
}


def _tps_native_treatment_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Treat the whole TPS formulation with its complete native waste proxy.

    The service supplies one kg principal glucan plus glycerol. Carried water,
    protein and reagents increase the wet waste quantity. All native exchanges scale by wet kg
    divided by the
    magnitude of native waste production. Direct waste-gas amounts retain the
    native proxy yield and split proportionally to incoming carbon origins.
    Principal fractions attribute the separate diagnostic to principal carbon.
    """
    service_id = str(service["provider_service_id"])
    key = database_name, provider_service_code(service_id)
    construction = cast(dict[str, object], service["construction"])
    inlet = _tps_material_load(services, str(construction["formulation_service_id"]))
    rules = cast(list[dict[str, object]], construction["direct_treatment_carbon_mappings"])
    _service_trace(context, service)
    activity, trace = context.activity(
        cast(dict[str, object], construction["source"]),
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="complete_native_tps_treatment_proxy",
        semantic_class="service_normalization",
    )
    native = context.exchanges(activity)
    production = _self_production_amount(activity, native, method="TPS native treatment proxy")
    multiplier = inlet.wet_mass_kg / abs(production)
    principal = {
        origin: fsum(
            pool.input_carbon_kg
            for pool in inlet.carbon
            if pool.principal and pool.origin == origin
        )
        for origin in CARBON_ORIGINS
    }
    total = {
        origin: fsum(pool.input_carbon_kg for pool in inlet.carbon if pool.origin == origin)
        for origin in CARBON_ORIGINS
    }
    total_carbon = fsum(total.values())
    # EF uses the same native non-fossil flow for biogenic and DAC carbon. Keep
    # their distinct principal shares in metadata rather than inventing flows.
    groups = (("fossil",), ("biogenic", "atmospheric_dac"))
    exchanges: list[dict[str, object]] = [{"input": key, "amount": 1.0, "type": "production"}]
    for row in native:
        if row["type"] == "production":
            continue
        node = context.input(row)
        rule = next(
            (
                item
                for item in rules
                if _matches_treatment_carbon_selector(str(row["type"]), node, item)
            ),
            None,
        )
        if rule is None:
            copied = _scaled_native_exchange(row, multiplier)
            if row["type"] == "biosphere" and node.get("name") in _CARBON_GASES:
                copied["principal_carbon_origin_fractions"] = (0.0, 0.0, 0.0)
            exchanges.append(copied)
            continue
        for origins in groups:
            group_carbon = fsum(total[origin] for origin in origins)
            if group_carbon == 0.0:
                continue
            # Both portions retain the source gas yield; their sum is exactly the whole wet-quantity
            # proxy.
            copied = _scaled_native_exchange(row, multiplier * group_carbon / total_carbon)
            if origins[0] != "fossil":
                flow = context.biosphere.flow(cast(dict[str, object], rule["replacement_selector"]))
                _replace_carbon_identity(copied, flow)
            copied["principal_carbon_origin_fractions"] = tuple(
                principal[origin] / group_carbon if origin in origins else 0.0
                for origin in CARBON_ORIGINS
            )
            exchanges.append(copied)
    terms = {
        "formulation_service_id": inlet.formulation_service_id,
        "native_source_key": activity.key,
        "native_reference_production": production,
        "wet_mass_kg_per_principal_kg": inlet.wet_mass_kg,
        "water_kg_per_principal_kg": inlet.water_kg,
        "native_inventory_multiplier": multiplier,
        "principal_carbon_kg": principal,
        "auxiliary_carbon_kg": {
            origin: fsum(
                pool.input_carbon_kg
                for pool in inlet.carbon
                if not pool.principal and pool.origin == origin
            )
            for origin in CARBON_ORIGINS
        },
        "emission_basis": "Native MIXED treatment proxy per whole wet kg; unchanged source yield.",
        "carbon_attribution_basis": (
            "Native waste-gas yields allocated in proportion to the complete formulation's "
            "incoming carbon pools; not component-specific degradation or mass balance."
        ),
    }
    trace["project_terms_json"] = json.dumps(terms, sort_keys=True, separators=(",", ":"))
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "direct_treatment_carbon_owners_json": _direct_carbon_owners_json(
            (
                {
                    "owner_database": key[0],
                    "owner_code": key[1],
                    "owner_scale_per_provider_unit": 1.0,
                    "owner_scope": "direct_biosphere_only",
                    "source_role": "member_local_frontier_service",
                    "origin_mapping_status": "actual_tps_principal_origin",
                },
            )
        ),
        "exchanges": exchanges,
    }
