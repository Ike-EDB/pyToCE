"""Treatment-origin compilation and direct-carbon attribution."""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, cast

from .catalogue import provider_service_code
from .leachate_load import _CarbonOrigin
from .material_carbon import CARBON_ORIGINS
from .native_recipe import _self_production_amount, _service_trace
from .native_resolution import _exact_exchange_matches
from .recipe_context import _copied_native_exchange, _RecipeContext

# Nominal molar masses convert kg emitted gas to kg carbon: C12 divided by
# CO2 44, CH4 16 or CO 28 g/mol. Origin changes preserve the chemical species.
_DIRECT_TREATMENT_CARBON_FRACTIONS = {
    ("exact_non_fossil_counterpart", "Carbon dioxide, fossil", "Carbon dioxide, non-fossil"): 12.0
    / 44.0,
    ("exact_non_fossil_counterpart", "Methane, fossil", "Methane, non-fossil"): 12.0 / 16.0,
    ("exact_non_fossil_counterpart", "Carbon monoxide, fossil", "Carbon monoxide, non-fossil"): 12.0
    / 28.0,
    ("exact_fossil_counterpart", "Carbon dioxide, non-fossil", "Carbon dioxide, fossil"): 12.0
    / 44.0,
    ("exact_fossil_counterpart", "Methane, non-fossil", "Methane, fossil"): 12.0 / 16.0,
    ("exact_fossil_counterpart", "Carbon monoxide, non-fossil", "Carbon monoxide, fossil"): 12.0
    / 28.0,
}


@dataclass(frozen=True, slots=True)
class _WastewaterCarbon:
    """Native kg-per-activity inputs and their Doka waste-carbon fractions."""

    source_database: str
    source_code: str
    effluent_toc: float
    polyacrylamide: float
    sludge: float
    carbon_dioxide: float


def _wastewater_carbon(
    context: _RecipeContext, construction: dict[str, object]
) -> _WastewaterCarbon | None:
    """Separate wastewater carbon from fossil flocculant using actual source inputs.

    The native effluent equation gives influent carbon. Source transfers use the original
    calibration; dated methane emissions follow the matching premise background and its GAINS
    changes.
    """
    if "wastewater_source" not in construction:
        return None
    source = cast(dict[str, object], construction["wastewater_source"])
    activity = context.resolver(str(source["source_role"])).activity(
        cast(dict[str, object], source["selector"]), year=context.year
    )
    semantic = cast(dict[str, object], context.catalogue["semantic_contract"])
    authority = cast(dict[str, object], semantic["landfill_carbon_origin"])
    constants = cast(dict[str, float], authority["source_constants"])
    toc_selector = {"selector": authority["toc_selector"]}
    native = context.exchanges(activity)
    toc_rows = [
        float(row["amount"])
        for row in native
        if _matches_treatment_carbon_selector(str(row["type"]), context.input(row), toc_selector)
    ]
    flocculant_rows = [
        float(row["amount"])
        for row in native
        if _exact_exchange_matches(
            row, context.input(row), cast(dict[str, object], authority["flocculant_selector"])
        )
    ]
    if len(toc_rows) != 1 or len(flocculant_rows) != 1:
        raise ValueError(
            f"Wastewater source {activity.key!r} has {len(toc_rows)} effluent TOC "
            f"and {len(flocculant_rows)} polyacrylamide rows. Exactly one of each "
            "declared source input is required; prepare compatible native inputs "
            "in a new workspace."
        )
    toc, flocculant = toc_rows[0], flocculant_rows[0]
    if not math.isfinite(toc + flocculant) or toc <= 0.0 or flocculant <= 0.0:
        raise ValueError(
            f"Wastewater source {activity.key!r} has TOC {toc!r} or polyacrylamide "
            f"{flocculant!r}. Both source inputs must be positive for the declared "
            "Doka carbon partition; prepare compatible native inputs in a new workspace."
        )
    first = constants["one_stage_fraction"]
    later = 1.0 - first
    sludge_transfer = (
        first * constants["one_stage_sludge_carbon_transfer"]
        + later * constants["later_stage_sludge_carbon_transfer"]
    )
    biological_transfer = later * constants["later_stage_biological_carbon_transfer"]
    influent = toc / (1.0 - sludge_transfer - biological_transfer)
    sludge = influent * sludge_transfer
    flocculant_carbon = (
        flocculant
        * constants["flocculant_carbon_molar_mass"]
        / constants["flocculant_repeat_molar_mass"]
    )
    principal = sludge / (sludge + flocculant_carbon)
    gas_carbon = (
        (sludge + flocculant_carbon)
        * constants["raw_sludge_digestion_fraction"]
        * constants["digested_carbon_gas_transfer"]
    )
    # Convert kg carbon from the biological and gas transfers to kg CO2 using
    # the same nominal molar masses (44 g CO2 per 12 g C).
    biological_co2 = influent * biological_transfer * 44.0 / 12.0
    gas_co2 = (
        gas_carbon
        * (1.0 - constants["gas_methane_volume_fraction"] * constants["methane_leak_fraction"])
        * 44.0
        / 12.0
    )
    return _WastewaterCarbon(
        str(activity.key[0]),
        str(activity.key[1]),
        toc,
        flocculant,
        principal,
        (biological_co2 + principal * gas_co2) / (biological_co2 + gas_co2),
    )


def _waste_fraction(rule: dict[str, object], wastewater: _WastewaterCarbon | None) -> float:
    """Return the declared principal fraction, or the unmixed waste boundary."""
    if wastewater is None:
        return 1.0
    return (
        wastewater.carbon_dioxide
        if rule["wastewater_carbon_fraction"] == "carbon_dioxide"
        else wastewater.sludge
    )


def _scaled_carbon_exchange(exchange: Any, fraction: float) -> dict[str, object]:
    """Scale native gas amounts and distribution locations, retaining dispersion."""
    copied = _copied_native_exchange(exchange)
    if fraction == 1.0:
        return copied
    copied["amount"] = float(exchange["amount"]) * fraction
    uncertainty = int(exchange.get("uncertainty type", 0))
    if uncertainty == 2:
        copied["loc"] = float(exchange["loc"]) + math.log(fraction)
    elif uncertainty in {0, 1}:
        copied["loc"] = float(exchange.get("loc", exchange["amount"])) * fraction
    else:
        raise ValueError(
            f"Landfill gas {exchange['input']!r} has uncertainty type {uncertainty}. "
            "The selected source requires deterministic or lognormal gas rows; "
            "prepare the supported ecoinvent/premise source in a new workspace."
        )
    return copied


def _replace_carbon_identity(generated: dict[str, object], flow: Any) -> None:
    """Change gas identity and per-kg origin properties while preserving total carbon."""
    generated["input"] = cast(tuple[str, str], flow.key)
    generated["name"] = str(flow["name"])
    generated["unit"] = str(flow["unit"])
    if "flow" in generated:
        generated["flow"] = str(flow.key[1])
    if "categories" in generated:
        generated["categories"] = tuple(flow["categories"])
    if "properties" in generated:
        properties = {
            name: dict(value)
            for name, value in cast(dict[str, dict[str, object]], generated["properties"]).items()
        }
        # Native gas properties contain this pair; premise omits properties altogether.
        fossil = properties["carbon content, fossil"]
        non_fossil = properties["carbon content, non-fossil"]
        carbon = float(cast(float, fossil["amount"])) + float(cast(float, non_fossil["amount"]))
        is_non_fossil = "non-fossil" in str(flow["name"])
        fossil["amount"] = 0.0 if is_non_fossil else carbon
        non_fossil["amount"] = carbon if is_non_fossil else 0.0
        generated["properties"] = properties


def _waste_market_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[str, object]:
    """Copy a declared plastic-waste market with origin-specific treatment links.

    Keep its native signed production, transport, amounts and uncertainty.
    Replace only the waste-market or treatment inputs identified by the
    declared source selectors.
    """

    service_id = str(service["provider_service_id"])
    code = provider_service_code(service_id)
    construction = cast(dict[str, object], service["construction"])
    _service_trace(context, service)
    activity, resolution_row = context.activity(
        cast(dict[str, object], construction["selector"]),
        source_role=str(construction["source_role"]),
        service_id=service_id,
        component_id="native_plastic_waste_market",
        semantic_class="operation_only_extraction",
    )
    replacements = cast(
        list[dict[str, object]],
        construction["treatment_replacements"],
    )
    generated: list[dict[str, object]] = []
    retained_technosphere: list[Any] = []
    replaced_technosphere: list[Any] = []
    replacement_rows: list[dict[str, object]] = []
    for exchange in context.exchanges(activity):
        copied = _copied_native_exchange(exchange)
        if str(exchange["type"]) == "production":
            copied["input"] = (database_name, code)
            generated.append(copied)
            continue
        input_node = context.input(exchange)
        matching = [
            replacement
            for replacement in replacements
            if _exact_exchange_matches(
                exchange,
                input_node,
                cast(dict[str, object], replacement["selector"]),
            )
        ]
        if matching:
            replacement = matching[0]
            copied["input"] = (
                database_name,
                provider_service_code(str(replacement["service_id"])),
            )
            replaced_technosphere.append(exchange)
            replacement_rows.append(
                {
                    "native_input_database": str(input_node.key[0]),
                    "native_input_code": str(input_node.key[1]),
                    "project_input_database": database_name,
                    "project_input_code": provider_service_code(str(replacement["service_id"])),
                    "amount": float(exchange["amount"]),
                    "exchange_type": str(exchange["type"]),
                    "metadata_preserved": True,
                }
            )
        elif str(exchange["type"]) == "technosphere":
            retained_technosphere.append(exchange)
        generated.append(copied)
    resolution_row["retained_technosphere_json"] = context.exchange_json(retained_technosphere)
    resolution_row["removed_technosphere_json"] = context.exchange_json(replaced_technosphere)
    resolution_row["project_terms_json"] = json.dumps(
        replacement_rows,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": generated,
    }


def _waste_carbon_owners(
    context: _RecipeContext,
    activity: Any,
    service_id: str,
    *,
    scale: float,
    non_fossil: bool,
    database_name: str,
    root_production: float | None = None,
) -> tuple[dict[str, object], ...]:
    """Follow declared waste children and record principal-only gas duties.

    ``scale`` is the signed incoming coefficient per provider unit. Divide by
    each current production before passing its child exchange; the diagnostic
    alone divides the terminal coefficient by the terminal production. Public
    copied landfill services supply ``root_production=1`` while their native
    source and internal children retain negative reference production.
    """
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], context.catalogue["services"])
    }
    amounts: dict[tuple[str, str], list[float]] = {}
    owners: dict[tuple[str, str], dict[str, object]] = {}

    def visit(
        node: Any, current_service_id: str, amount: float, production: float | None = None
    ) -> None:
        """Accumulate direct waste gas and continue through the same waste chain."""
        construction = cast(dict[str, object], services[current_service_id]["construction"])
        native = context.exchanges(node)
        if production is None:
            production = _self_production_amount(node, native, method="waste carbon owner")
        rules = cast(
            list[dict[str, object]],
            construction.get(
                "replace_biosphere", construction.get("direct_treatment_carbon_mappings", [])
            ),
        )
        if rules:
            key = (
                (database_name, provider_service_code(current_service_id))
                if non_fossil
                else cast(tuple[str, str], node.key)
            )
            amounts.setdefault(key, []).append(amount)
            owner: dict[str, object] = {
                "owner_database": key[0],
                "owner_code": key[1],
                "source_role": "member_local_frontier_service"
                if non_fossil
                else "native_ecoinvent_provider",
                "owner_scope": "direct_biosphere_only",
                "origin_mapping_status": str(rules[0]["mapping_status"])
                if non_fossil
                else "native_identity",
            }
            wastewater = _wastewater_carbon(context, construction)
            if wastewater is not None:
                fractions = []
                for rule in rules:
                    flow = context.biosphere.flow(
                        cast(
                            dict[str, object],
                            rule["replacement_selector" if non_fossil else "selector"],
                        )
                    )
                    fractions.append(
                        {
                            "flow_database": str(flow.key[0]),
                            "flow_code": str(flow.key[1]),
                            "fraction": 1.0 if non_fossil else _waste_fraction(rule, wastewater),
                        }
                    )
                    if non_fossil:
                        auxiliary = context.biosphere.flow(
                            cast(dict[str, object], rule["selector"])
                        )
                        fractions.append(
                            {
                                "flow_database": str(auxiliary.key[0]),
                                "flow_code": str(auxiliary.key[1]),
                                "fraction": 0.0,
                            }
                        )
                owner["principal_carbon_fractions"] = fractions
                owner["wastewater_carbon_source"] = asdict(wastewater)
            owners[key] = owner
        replacements = cast(list[dict[str, object]], construction.get("treatment_replacements", []))
        for exchange in native:
            if str(exchange["type"]) != "technosphere":
                continue
            child = context.input(exchange)
            for replacement in replacements:
                if _exact_exchange_matches(
                    exchange, child, cast(dict[str, object], replacement["selector"])
                ):
                    visit(
                        child,
                        str(replacement["service_id"]),
                        amount * float(exchange["amount"]) / production,
                    )

    visit(activity, service_id, scale, root_production)
    return tuple(
        {**owners[key], "owner_scale_per_provider_unit": math.fsum(values)}
        for key, values in sorted(amounts.items())
    )


def _matches_treatment_carbon_selector(
    exchange_type: str,
    input_node: Any,
    rule: dict[str, object],
) -> bool:
    """Test whether one biosphere exchange matches a treatment carbon selector."""
    selector = cast(dict[str, object], rule["selector"])
    return (
        exchange_type == "biosphere"
        and str(input_node.get("name", "")) == str(selector["name"])
        and str(input_node.get("unit", "")) == str(selector["unit"])
        and tuple(str(value) for value in input_node.get("categories", ()))
        == tuple(str(value) for value in cast(list[object], selector["categories"]))
    )


def _direct_carbon_owners_json(
    owners: tuple[dict[str, object], ...],
) -> str:
    """Serialize direct carbon owners in stable provider metadata."""
    return json.dumps(
        owners,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _direct_principal_carbon_owner(
    data: dict[str, object], database_name: str, origin: _CarbonOrigin, material_name: str
) -> dict[str, object]:
    """Attribute direct gas to tracked principal, excluding auxiliaries and reference differences.

    Multiple actual pools can share a native gas identity. Aggregate their
    amount-weighted principal fractions only for this fixed-origin provider;
    upstream utility carbon is outside the selected direct receiving rows.
    """
    origin_index = CARBON_ORIGINS.index(origin)
    total: dict[tuple[str, str], list[float]] = {}
    principal: dict[tuple[str, str], list[float]] = {}
    for row in cast(list[dict[str, object]], data["exchanges"]):
        if row["type"] != "biosphere" or "principal_carbon_origin_fractions" not in row:
            continue
        key = cast(tuple[str, str], row["input"])
        amount = float(cast(float, row["amount"]))
        fractions = cast(tuple[float, float, float], row["principal_carbon_origin_fractions"])
        total.setdefault(key, []).append(amount)
        principal.setdefault(key, []).append(amount * fractions[origin_index])
    return {
        "owner_database": database_name,
        "owner_code": provider_service_code(str(data["provider_service_id"])),
        "owner_scale_per_provider_unit": 1.0,
        "owner_scope": "direct_biosphere_only",
        "source_role": "member_local_frontier_service",
        "origin_mapping_status": f"actual_{material_name}_principal_origin",
        "principal_carbon_fractions": [
            {
                "flow_database": key[0],
                "flow_code": key[1],
                # A retained zero emission has no principal gas contribution.
                "fraction": math.fsum(principal[key]) / math.fsum(values)
                if math.fsum(values)
                else 0.0,
            }
            for key, values in sorted(total.items())
        ],
    }


def attach_landfill_carbon_owners(
    data: dict[tuple[str, str], dict[str, object]],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> None:
    """Connect direct principal-carbon owners through actual compiled signed purchases.

    Only the sanitary site, leachate collection, WWT and sludge chain participates.
    Source utilities, reagents, ash support and manufacture remain outside this
    principal diagnostic. Native production is divided once at each child.
    """
    children = {
        "landfill": ("leachate",),
        "residue_landfill": ("leachate",),
        "sbr_compound_landfill": ("treatment",),
        "leachate_collection": ("plant",),
        "wastewater_treatment": ("incineration", "landfarming"),
        "sludge_collection": ("treatment",),
        "sludge_incineration": (),
        "sludge_landfarming": (),
    }

    def visit(service_id: str, amount: float) -> list[dict[str, object]]:
        """Keep one direct gas owner and follow only this node's declared waste roles."""
        key = database_name, provider_service_code(service_id)
        dataset = data[key]
        construction = cast(dict[str, object], services[service_id]["construction"])
        exchanges = cast(list[dict[str, object]], dataset["exchanges"])
        production = next(
            float(cast(float, row["amount"]))
            for row in exchanges
            if row["type"] == "production" and row["input"] == key
        )
        kind = str(construction["kind"])
        role = (
            kind.removeprefix("epoxy_")
            .removeprefix("upr_")
            .removeprefix("road_")
            .removeprefix("sbr_filled_")
        )
        # SBR receiving rows contain principal fractions. Read those rows for the end-of-life
        # diagnostic.
        if kind.startswith("sbr_filled_") and role in {
            "residue_landfill",
            "wastewater_treatment",
            "sludge_incineration",
            "sludge_landfarming",
        }:
            direct = [
                _direct_principal_carbon_owner(
                    dataset,
                    database_name,
                    cast(_CarbonOrigin, construction["principal_carbon_origin"]),
                    "sbr",
                )
            ]
        else:
            direct = cast(
                list[dict[str, object]],
                json.loads(str(dataset.get("direct_treatment_carbon_owners_json", "[]"))),
            )
        owners = [{**owner, "owner_scale_per_provider_unit": amount} for owner in direct]
        for child_role in children[role]:
            child_id = str(cast(dict[str, object], construction[child_role])["service_id"])
            child_key = database_name, provider_service_code(child_id)
            coefficient = math.fsum(
                float(cast(float, row["amount"]))
                for row in exchanges
                if row["type"] == "technosphere" and row["input"] == child_key
            )
            owners.extend(visit(child_id, amount * coefficient / production))
        return owners

    for service_id, service in services.items():
        if cast(dict[str, object], service["construction"])["kind"] in {
            "sbr_compound_landfill",
            "epoxy_landfill",
            "upr_landfill",
            "road_landfill",
        }:
            owners = visit(service_id, 1.0)
            data[database_name, provider_service_code(service_id)][
                "direct_treatment_carbon_owners_json"
            ] = json.dumps(owners, sort_keys=True, separators=(",", ":"))


def attach_tps_carbon_owners(
    data: dict[tuple[str, str], dict[str, object]],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> None:
    """Record actual TPS gas owners along the compiled waste-treatment chain.

    The principal landfill service supplies one kg principal glucan/glycerol,
    together with its historical formulation's water and auxiliaries. Follow
    only its landfill, leachate and sludge treatment purchases; equipment,
    fuels, catalyst manufacture and ash support remain
    outside this principal-carbon diagnostic. Amounts are signed incoming
    demands. The reader divides each terminal demand by its native production
    once, preserving negative waste reference products and nonunit production.
    """
    child_roles = {
        "tps_residue_treatment": ("treatment",),
        "tps_residue_landfill": ("leachate",),
        "tps_leachate_collection": ("plant",),
        "tps_wastewater_treatment": ("incineration", "landfarming"),
        "tps_sludge_collection": ("treatment",),
        "tps_sludge_incineration": (),
        "tps_sludge_landfarming": (),
    }
    gas_owners = {
        "tps_residue_landfill",
        "tps_wastewater_treatment",
        "tps_sludge_incineration",
        "tps_sludge_landfarming",
    }

    def visit(service_id: str, amount: float) -> list[dict[str, object]]:
        """Pass compiled signed purchases and retain the four direct gas owners."""
        key = (database_name, provider_service_code(service_id))
        construction = cast(dict[str, object], services[service_id]["construction"])
        kind = str(construction["kind"])
        exchanges = cast(list[dict[str, Any]], data[key]["exchanges"])
        production = next(
            float(row["amount"])
            for row in exchanges
            if row["type"] == "production" and row["input"] == key
        )
        owners: list[dict[str, object]] = []
        if kind in gas_owners:
            owners.append(
                {
                    "owner_database": key[0],
                    "owner_code": key[1],
                    "owner_scale_per_provider_unit": amount,
                    "owner_scope": "direct_biosphere_only",
                    "source_role": "member_local_frontier_service",
                    "origin_mapping_status": "actual_tps_principal_origin",
                }
            )
        for role in child_roles[kind]:
            child_id = str(cast(dict[str, object], construction[role])["service_id"])
            child_key = (database_name, provider_service_code(child_id))
            coefficient = math.fsum(
                float(row["amount"])
                for row in exchanges
                if row["type"] == "technosphere" and row["input"] == child_key
            )
            owners.extend(visit(child_id, amount * coefficient / production))
        return owners

    for service_id, service in services.items():
        if cast(dict[str, object], service["construction"])["kind"] == "tps_residue_treatment":
            data[database_name, provider_service_code(service_id)][
                "direct_treatment_carbon_owners_json"
            ] = _direct_carbon_owners_json(tuple(visit(service_id, 1.0)))


def _treatment_origin_operation_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[str, object]:
    """Retain native treatment duties and attribute the declared waste-carbon fraction.

    Public services produce one kilogram. Internal leachate and sludge owners
    retain negative native kg or m3 production and their signed child duties.
    Fractional gases retain lognormal dispersion and shift location by log(fraction);
    auxiliary fossil carbon and all unrelated exchanges remain in the LCI.
    """

    service_id = str(service["provider_service_id"])
    code = provider_service_code(service_id)
    construction = cast(dict[str, object], service["construction"])
    rules = cast(list[dict[str, object]], construction["replace_biosphere"])
    wastewater = _wastewater_carbon(context, construction)
    treatment_replacements = cast(
        list[dict[str, object]], construction.get("treatment_replacements", [])
    )
    ultimate_origins = tuple(
        str(value) for value in cast(list[object], service.get("ultimate_carbon_origins", []))
    )
    mapping_status = str(rules[0]["mapping_status"])
    _service_trace(context, service)
    base = cast(dict[str, object], construction["base"])
    activity, resolution_row = context.activity(
        cast(dict[str, object], base["selector"]),
        source_role=str(base["source_role"]),
        service_id=service_id,
        component_id="native_treatment_operation",
        semantic_class="operation_only_extraction",
    )
    native_exchanges = context.exchanges(activity)
    native_production = _self_production_amount(
        activity,
        native_exchanges,
        method="treatment operation",
    )
    if not math.isclose(native_production, -1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"The selected native treatment operation must have a -1 "
            f"{activity.get('unit')} reference production."
        )

    matches = [0 for _rule in rules]
    retained_technosphere: list[Any] = []
    replaced_technosphere: list[Any] = []
    retained_biosphere: list[Any] = []
    removed_biosphere: list[Any] = []
    replacement_rows: list[dict[str, object]] = []
    generated_exchanges: list[dict[str, object]] = []

    for exchange in native_exchanges:
        exchange_type = str(exchange["type"])
        if exchange_type == "production":
            continue
        input_node = context.input(exchange)
        matching_rules = [
            (index, rule)
            for index, rule in enumerate(rules)
            if _matches_treatment_carbon_selector(exchange_type, input_node, rule)
        ]
        if not matching_rules:
            copied = _copied_native_exchange(exchange)
            for replacement in treatment_replacements:
                if _exact_exchange_matches(
                    exchange, input_node, cast(dict[str, object], replacement["selector"])
                ):
                    copied["input"] = (
                        database_name,
                        provider_service_code(str(replacement["service_id"])),
                    )
                    replaced_technosphere.append(exchange)
            generated_exchanges.append(copied)
            if exchange_type == "technosphere" and copied["input"] == exchange["input"]:
                retained_technosphere.append(exchange)
            elif exchange_type != "technosphere":
                retained_biosphere.append(exchange)
            continue
        replacement_index, rule = matching_rules[0]
        matches[replacement_index] += 1
        removed_biosphere.append(exchange)
        replacement_flow = context.biosphere.flow(
            cast(dict[str, object], rule["replacement_selector"])
        )
        source_name = str(input_node.get("name", ""))
        replacement_name = str(replacement_flow.get("name", ""))
        carbon_fraction = _DIRECT_TREATMENT_CARBON_FRACTIONS[
            (mapping_status, source_name, replacement_name)
        ]
        source_unit = str(input_node.get("unit", ""))
        replacement_unit = str(replacement_flow.get("unit", ""))
        source_categories = tuple(str(value) for value in input_node.get("categories", ()))
        replacement_categories = tuple(
            str(value) for value in replacement_flow.get("categories", ())
        )
        attribution_basis = str(rule["attribution_basis"]).strip()
        limitation = str(rule["limitation"]).strip()

        fraction = _waste_fraction(rule, wastewater)
        generated = _scaled_carbon_exchange(exchange, fraction)
        _replace_carbon_identity(generated, replacement_flow)
        generated_exchanges.append(generated)
        amount = float(exchange["amount"])
        auxiliary_amount = amount - float(cast(float, generated["amount"]))
        if fraction < 1.0:
            auxiliary = _scaled_carbon_exchange(exchange, 1.0 - fraction)
            auxiliary["amount"] = auxiliary_amount
            source_partition = json.dumps(
                {
                    "schema_version": "pytoce_treatment_carbon_partition_schema_v1",
                    "activity": list(activity.key),
                    "flow": list(input_node.key),
                    "source_amount": amount,
                    "principal_fraction": fraction,
                    "wastewater_source": asdict(cast(_WastewaterCarbon, wastewater)),
                    "uncertainty_scope": ("Deterministic fractions of one source."),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            generated["pytoce_carbon_partition"] = source_partition
            auxiliary["pytoce_carbon_partition"] = source_partition
            generated_exchanges.append(auxiliary)
        source_carbon = amount * carbon_fraction
        replacement_carbon = float(cast(float, generated["amount"])) * carbon_fraction
        replacement_row = {
            "source_flow_database": str(input_node.key[0]),
            "source_flow_code": str(input_node.key[1]),
            "source_flow_name": source_name,
            "source_flow_unit": source_unit,
            "source_flow_categories_json": json.dumps(
                list(source_categories), separators=(",", ":")
            ),
            "replacement_flow_database": str(replacement_flow.key[0]),
            "replacement_flow_code": str(replacement_flow.key[1]),
            "replacement_flow_name": replacement_name,
            "replacement_flow_unit": replacement_unit,
            "replacement_flow_categories_json": json.dumps(
                list(replacement_categories), separators=(",", ":")
            ),
            "source_amount": amount,
            "generated_amount": float(cast(float, generated["amount"])),
            "retained_auxiliary_amount": auxiliary_amount,
            "principal_carbon_fraction": fraction,
            "source_carbon_kg_per_provider_unit": source_carbon,
            "generated_carbon_kg_per_provider_unit": replacement_carbon,
            "retained_auxiliary_carbon_kg_per_provider_unit": auxiliary_amount * carbon_fraction,
            "carbon_difference_kg_per_provider_unit": (
                replacement_carbon + auxiliary_amount * carbon_fraction - source_carbon
            ),
            "uncertainty_metadata_preserved": fraction == 1.0,
            "uncertainty_scaling": (
                "unchanged"
                if fraction == 1.0
                else "native_location_plus_log_fraction"
                if exchange.get("uncertainty type", 0) == 2
                else "deterministic_location_times_fraction"
            ),
            "allocation_metadata_preserved": True,
            "attribution_basis": attribution_basis,
            "carbon_attribution_status": str(rule["carbon_attribution_status"]),
            "mapping_status": str(rule["mapping_status"]),
            "limitation": limitation,
            "reason": str(rule["reason"]),
        }
        replacement_rows.append(replacement_row)
        for ultimate_origin in ultimate_origins:
            context.semantic_rows.append(
                {
                    "background_scenario": context.background_scenario,
                    "anchor_year": context.year,
                    "provider_service_id": service_id,
                    "provider_demand_type": str(service["provider_demand_type"]),
                    "provider_stage_owner": str(service["stage_owner"]),
                    "ultimate_carbon_origin": ultimate_origin,
                    "semantic_class": "operation_only_extraction",
                    "ledger_entry_type": "treatment_carbon_mapping",
                    "native_activity_database": str(activity.key[0]),
                    "native_activity_code": str(activity.key[1]),
                    "native_activity_name": str(activity.get("name", "")),
                    "native_reference_product": str(activity.get("reference product", "")),
                    "native_unit": str(activity.get("unit", "")),
                    "native_location": str(activity.get("location", "")),
                    "native_reference_production_amount": native_production,
                    "direct_treatment_carbon_owner_database": database_name,
                    "direct_treatment_carbon_owner_code": code,
                    "direct_treatment_carbon_owner_scope": "direct_biosphere_only",
                    **replacement_row,
                }
            )

    if any(count != 1 for count in matches):
        raise ValueError(
            "Each source-backed direct treatment-carbon replacement must match "
            "exactly one native operation exchange."
        )
    resolution_row["retained_technosphere_json"] = context.exchange_json(retained_technosphere)
    resolution_row["removed_technosphere_json"] = context.exchange_json(replaced_technosphere)
    resolution_row["retained_biosphere_json"] = context.exchange_json(retained_biosphere)
    resolution_row["removed_biosphere_json"] = context.exchange_json(removed_biosphere)
    resolution_row["project_terms_json"] = json.dumps(
        replacement_rows,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    preserve_native_production = bool(construction.get("preserve_native_production", False))
    if preserve_native_production:
        native_self_production = next(
            exchange
            for exchange in native_exchanges
            if str(exchange["type"]) == "production"
            and cast(tuple[str, str], exchange["input"]) == cast(tuple[str, str], activity.key)
        )
        production_exchange = _copied_native_exchange(native_self_production)
        production_exchange["input"] = (database_name, code)
    else:
        production_exchange = {
            "input": (database_name, code),
            "amount": 1.0,
            "type": "production",
        }
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "direct_treatment_carbon_owners_json": _direct_carbon_owners_json(
            _waste_carbon_owners(
                context,
                activity,
                service_id,
                scale=1.0,
                non_fossil=True,
                database_name=database_name,
                root_production=native_production if preserve_native_production else 1.0,
            )
        ),
        "exchanges": [
            production_exchange,
            *generated_exchanges,
        ],
    }


def _native_treatment_carbon_owner(
    context: _RecipeContext,
    service: dict[str, object],
) -> dict[str, object]:
    """Record direct fossil carbon flows emitted by one native treatment activity.

    Args:
        context: Prepared activity and biosphere resolvers for one background scenario
            and anchor year.
        service: Provider service with an optional native direct carbon declaration.

    Returns:
        An empty mapping when the service has no native direct carbon declaration;
        otherwise, JSON metadata locating the native treatment activity and its scale of
        one provider functional unit.

    Raises:
        ValueError: If a declared direct carbon selector does not match exactly one
            native biosphere exchange.

    Notes:
        The native fossil flow identity and exchange amount are retained. The trace record uses the
        species-specific carbon fraction; Brightway uses the native exchange quantity.
        The stored owner scale is the signed native technosphere coefficient per
        provider unit. The diagnostic reader divides by native reference production;
        recording an activity supply multiplier here would normalize it twice.
    """

    construction = cast(dict[str, object], service["construction"])
    if construction.get("direct_treatment_carbon_owner") != "native_activity":
        return {}
    rules = cast(list[dict[str, object]], construction["direct_treatment_carbon_mappings"])
    activity = context.resolver(str(construction["source_role"])).activity(
        cast(dict[str, object], construction["selector"]), year=context.year
    )
    native_exchanges = context.exchanges(activity)
    for rule in rules:
        matching = []
        for exchange in native_exchanges:
            input_node = context.input(exchange)
            if _matches_treatment_carbon_selector(
                str(exchange["type"]),
                input_node,
                rule,
            ):
                matching.append((exchange, input_node))
        if len(matching) != 1:
            raise ValueError(
                "Each declared fossil treatment-carbon flow must match exactly "
                "one native operation exchange."
            )
        exchange, input_node = matching[0]
        source_name = str(input_node.get("name", ""))
        replacement_name = str(cast(dict[str, object], rule["replacement_selector"])["name"])
        carbon_fraction = _DIRECT_TREATMENT_CARBON_FRACTIONS[
            ("exact_non_fossil_counterpart", source_name, replacement_name)
        ]
        amount = float(exchange["amount"])
        categories = tuple(str(value) for value in input_node.get("categories", ()))
        source_unit = str(input_node.get("unit", ""))
        carbon_amount = amount * carbon_fraction
        context.semantic_rows.append(
            {
                "background_scenario": context.background_scenario,
                "anchor_year": context.year,
                "provider_service_id": str(service["provider_service_id"]),
                "provider_demand_type": str(service["provider_demand_type"]),
                "provider_stage_owner": str(service["stage_owner"]),
                "ultimate_carbon_origin": "fossil",
                "semantic_class": "unchanged_same_product",
                "ledger_entry_type": "treatment_carbon_mapping",
                "native_activity_database": str(activity.key[0]),
                "native_activity_code": str(activity.key[1]),
                "native_activity_name": str(activity.get("name", "")),
                "native_reference_product": str(activity.get("reference product", "")),
                "native_unit": str(activity.get("unit", "")),
                "native_location": str(activity.get("location", "")),
                "source_flow_database": str(input_node.key[0]),
                "source_flow_code": str(input_node.key[1]),
                "source_flow_name": source_name,
                "source_flow_unit": source_unit,
                "source_flow_categories_json": json.dumps(list(categories), separators=(",", ":")),
                "replacement_flow_database": str(input_node.key[0]),
                "replacement_flow_code": str(input_node.key[1]),
                "replacement_flow_name": source_name,
                "replacement_flow_unit": source_unit,
                "replacement_flow_categories_json": json.dumps(
                    list(categories), separators=(",", ":")
                ),
                "source_amount": amount,
                "generated_amount": amount,
                "source_carbon_kg_per_provider_unit": carbon_amount,
                "generated_carbon_kg_per_provider_unit": carbon_amount,
                "carbon_difference_kg_per_provider_unit": 0.0,
                "uncertainty_metadata_preserved": True,
                "allocation_metadata_preserved": True,
                "carbon_attribution_status": str(rule["carbon_attribution_status"]),
                "attribution_basis": str(rule["attribution_basis"]),
                "mapping_status": "native_fossil_identity",
                "limitation": str(rule["limitation"]),
                "reason": "Retain the native fossil direct-carbon flow identity.",
                "direct_treatment_carbon_owner_database": str(activity.key[0]),
                "direct_treatment_carbon_owner_code": str(activity.key[1]),
                "direct_treatment_carbon_owner_scope": "direct_biosphere_only",
            }
        )
    return {
        "direct_treatment_carbon_owners_json": _direct_carbon_owners_json(
            _waste_carbon_owners(
                context,
                activity,
                str(service["provider_service_id"]),
                scale=float(cast(float | int | str, construction.get("amount", 1.0))),
                non_fossil=False,
                database_name=context.provider_database,
            )
        )
    }
