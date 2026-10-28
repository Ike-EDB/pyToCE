"""Compile native operations with explicit feed and direct-emission substitutions."""

import json
import math
from typing import Any, cast

from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .native_resolution import _exact_exchange_matches, _input_semantics, _matches
from .recipe_context import _copied_native_exchange, _RecipeContext
from .treatment_compilation import (
    _direct_carbon_owners_json,
    _replace_carbon_identity,
    _waste_carbon_owners,
)

# These feed substitutions select the supplied grade and its specified composition.
_CHANGED_FEED_PRODUCTS = {
    ("electricity, medium voltage", "electricity, high voltage"),
    (
        "heat, district or industrial, natural gas",
        "heat, central or small-scale, other than natural gas",
    ),
    (
        "heat, district or industrial, other than natural gas",
        "heat, central or small-scale, other than natural gas",
    ),
    ("heat, district or industrial", "heat, central or small-scale, other than natural gas"),
    ("natural gas, high pressure", "biomethane, high pressure"),
    ("natural gas, high pressure", "methane at stated standard conditions"),
    # Acetylene's DAC feed uses the complete biomethane network as a delivery
    # reference, with the separate stated methane-volume convention.
    ("biomethane, high pressure", "methane at stated standard conditions"),
    ("maize starch", "synthetic amylose feed, 14% water"),
    ("soybean oil, crude", "castor oil, crude, dry mass basis"),
    ("sodium hydroxide, without water, in 50% solution state", "potassium hydroxide"),
    ("sodium dihydrogen phosphate", "potassium dihydrogen phosphate"),
    ("enzymes", "cyanocobalamin"),
    ("non-ionic surfactant", "Oleth-10 nominal finish ingredient"),
    ("propylene glycol, liquid", "1,3-propanediol, 99.92% solution"),
    # UPR's actual cobalt salt uses native acetate only as its Co-equivalent manufacturing proxy.
    ("cobalt acetate", "cobalt(II) 2-ethylhexanoate"),
    # PA6 transfers dry chemical-cargo distribution from the HMDA market; caprolactam manufacture
    # supplies the material.
    ("hexamethylenediamine", "caprolactam"),
    # PBAT/PBS/PTT buy TBOT through a complete TEOS supply proxy.
    ("tetraethyl orthosilicate", "titanium butoxide"),
}

# These supplied products retain the donor's carbon origin. Other declared
# replacements use their own origin or mixture fractions, independent of the
# service-name suffix.
_UNCHANGED_CARBON_ORIGIN_SERVICES = {
    # Polyol's native glucose purchase retains its biogenic carbon properties.
    "glucose_delivered_biogenic",
    "styrene_delivered_fossil",
    "styrene_fossil",
    "acrylonitrile_delivered_fossil",
    "acrylonitrile_fossil",
    # These net polyester purchases keep the selected native fossil compound.
    # Preserve incomplete native properties as supplied, including absence.
    "butanediol_fossil",
    "butanediol_delivered_fossil",
    "succinic_acid_fossil",
    "succinic_acid_delivered_fossil",
    "adipic_acid_fossil",
    "adipic_acid_delivered_fossil",
    "pta_fossil",
    "purified_terephthalic_acid_delivered_fossil",
    "lactic_acid_fossil",
    "lactic_acid_delivered_fossil",
    "ptt_renewable_steam",
    # Selected peracetic acid manufacture changes technology and retains fossil carbon origin.
    # Preserve same-product market metadata for the UPR proxy.
    "upr_organic_peroxide_factory",
    "upr_organic_peroxide_delivered",
}


def _operation_exchange_metadata(
    record: dict[str, object],
    source_product: str,
    service: dict[str, object],
    *,
    preserve_carbon_origin: bool,
) -> None:
    """Bind the optional product metadata.

    EcoSpold activity UUIDs identify the replaced producer. Same-compound flow,
    CAS, molecular formula and material properties remain valid, including
    route-specific feed synonyms. A changed production reference and the finite
    changed feed grades cannot inherit donor product identity, chemistry or
    product classifications (such as soybean-specific CPC and HS codes).
    The caller preserves an admitted unchanged origin; otherwise omit the
    unsupported origin pair.
    Mixed-product carbon accounts remain at their material-balance calculation. An exchange's
    amount formula is distinct from its chemical formula.
    """
    record.pop("activity", None)
    product = str(service["reference_product"])
    for field, value in (
        ("name", product),
        ("reference product", product),
        # Premise exchanges use this label instead of EcoSpold product metadata.
        ("product", product),
        ("unit", str(service["unit"])),
        # An existing supplier label must describe the selected service, while
        # the original producer geography remains in source provenance.
        ("location", str(service["location"])),
    ):
        if field in record:
            record[field] = value
    if (record["type"] == "production" and source_product != product) or (
        source_product,
        product,
    ) in _CHANGED_FEED_PRODUCTS:
        for field in ("flow", "CAS number", "chemical formula", "properties", "classifications"):
            record.pop(field, None)
    elif not preserve_carbon_origin and "properties" in record:
        properties = dict(cast(dict[str, object], record["properties"]))
        properties.pop("carbon content, fossil", None)
        properties.pop("carbon content, non-fossil", None)
        if service["provider_service_id"] in {
            "san_biogenic",
            "san_atmospheric_dac",
            "san_delivered_biogenic",
            "san_delivered_atmospheric_dac",
        }:
            # PUR buys the selected 75:25 styrene/acrylonitrile SAN grade. The selected grade has
            # its own carbon and heat properties. Keep its dry/wet mass and water metadata.
            for field in ("carbon allocation", "carbon content", "heating value, net"):
                properties.pop(field, None)
        record["properties"] = properties


def _scale_feed(record: dict[str, object], multiplier: float, *, service_id: str) -> None:
    """Scale an admitted starch or phosphate-salt feed on its original basis.

    The catalog supplies a positive fixed dry-glucan or molar-mass correction. Preserve
    the native deterministic or lognormal distribution under multiplication:
    amount and material bounds scale by the correction; lognormal location
    shifts by log(multiplier), with dispersion and the sign flag unchanged.
    The deterministic location, if present, scales directly.

    Raises:
        ValueError: If the external source uses an unsupported distribution or
            a live formula, its lognormal parameters are invalid, or scaling
            exceeds the finite numerical range.
    """
    distribution = int(
        cast(int | str, record.get("uncertainty_type", record.get("uncertainty type", 0)))
    )
    if distribution not in {0, 1, 2} or record.get("formula"):
        raise ValueError(
            f"Feed {record['input']!r} for {service_id!r} has uncertainty "
            f"type {distribution!r} or a formula. Composition scaling requires "
            "a deterministic or lognormal exchange without a live formula; "
            "use the supported source inventory in a new workspace."
        )
    if distribution == 2:
        loc = float(cast(float | int | str, record.get("loc", math.nan)))
        spread = float(cast(float | int | str, record.get("scale", math.nan)))
        if not math.isfinite(loc) or not math.isfinite(spread) or spread <= 0.0:
            raise ValueError(
                f"Lognormal feed {record['input']!r} for {service_id!r} "
                f"has loc={loc!r}, scale={spread!r}. A finite location and finite "
                "positive logarithmic spread are required; use a valid source "
                "inventory in a new workspace."
            )
    scaled = {"amount": float(cast(float | int | str, record["amount"])) * multiplier}
    if "loc" in record:
        loc = float(cast(float | int | str, record["loc"]))
        scaled["loc"] = loc + math.log(multiplier) if distribution == 2 else loc * multiplier
    for field in ("minimum", "maximum"):
        if field in record:
            scaled[field] = float(cast(float | int | str, record[field])) * multiplier
    for field, value in scaled.items():
        original = float(cast(float | int | str, record[field]))
        if math.isfinite(original) and not math.isfinite(value):
            raise ValueError(
                f"Feed {record['input']!r} for {service_id!r} has "
                f"{field}={original!r}; multiplying by {multiplier!r} exceeds "
                "the finite numerical range. Use source values whose composition "
                "correction remains finite in a new workspace."
            )
    record.update(scaled)


def _operation_only_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    services: dict[str, dict[str, object]],
    database_name: str,
) -> dict[str, object]:
    """Retain a native operation and replace only its declared input boundaries.

    Preserve the sole positive native production and each retained exchange,
    including duplicate records, explicit zeros, signs and uncertainty. Each
    substituted feed links to its published service in the same member/anchor.
    Declared feed composition corrections scale their original demands and
    distributions. A retained phosphate-acid feed keeps each dated supplier;
    its molar quantity changes with the target salt. Biosphere substitutions
    change only the flow identity.
    Declared plastic-waste inputs retain their signed native demand and select
    the corresponding treatment origin. Direct waste-carbon owners are expressed
    per finished output, with terminal production normalization left to the
    carbon diagnostic.
    Brightway resolves parent demands against the operation's native production basis.

    Args:
        context: Admitted native sources and provenance for one member and year.
        service: Validated operation, replacement and omission declarations.
        services: Admitted services indexed by their immutable provider identifiers.
        database_name: Project database receiving this operation and its feeds.

    Returns:
        One dataset whose exchange and provenance quantities retain the native
        production basis.

    Raises:
        ValueError: If the source production is not positive and singular, a
            declared input is absent, or a specified material balance fails.
    """
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    _service_trace(context, service)
    base = cast(dict[str, object], construction["base"])
    activity, resolution_row = context.activity(
        cast(dict[str, object], base["selector"]),
        source_role=str(base["source_role"]),
        service_id=service_id,
        component_id="native_operation",
        semantic_class="operation_only_extraction",
    )
    native_exchanges = context.exchanges(activity)
    native_production_amount = _self_production_amount(
        activity, native_exchanges, method="native operation"
    )
    if native_production_amount <= 0.0:
        raise ValueError(
            f"Native operation {activity.key!r} has production {native_production_amount!r}. "
            "A positive self-production amount is required; use a valid source "
            "operation in a new workspace."
        )
    production = _copied_native_exchange(
        next(exchange for exchange in native_exchanges if str(exchange["type"]) == "production")
    )
    production["input"] = (database_name, provider_service_code(service_id))
    _operation_exchange_metadata(
        production,
        str(activity["reference product"]),
        service,
        preserve_carbon_origin=(
            service_id in _UNCHANGED_CARBON_ORIGIN_SERVICES
            or (
                not construction.get("replacements")
                and construction.get("ultimate_carbon_class") != "non_fossil"
            )
        ),
    )
    replacements = {
        str(item["replacement_id"]): item
        for item in cast(list[dict[str, object]], construction.get("replacements", []))
    }
    matched_replacements: set[str] = set()
    exchanges: list[dict[str, object]] = [production]
    resolution_row["native_production_amount"] = native_production_amount
    retained_technosphere: list[Any] = []
    removed_technosphere: list[Any] = []
    retained_biosphere: list[Any] = []
    removed_biosphere: list[Any] = []
    replacement_rows: list[dict[str, object]] = []
    waste_owners: list[dict[str, object]] = []
    waste_rules = cast(list[dict[str, object]], construction.get("waste_replacements", []))
    technosphere_omission_rules = cast(
        list[dict[str, object]],
        construction.get("omit_technosphere", []),
    )
    technosphere_omission_matches = [0 for _rule in technosphere_omission_rules]
    technosphere_omission_amounts = [0.0 for _rule in technosphere_omission_rules]
    biosphere_replacement_rules = cast(
        list[dict[str, object]],
        construction.get("replace_biosphere", []),
    )
    biosphere_replacement_matches = [0 for _rule in biosphere_replacement_rules]
    biosphere_omission_rules = cast(
        list[dict[str, object]],
        construction.get("omit_biosphere", []),
    )
    biosphere_omission_matches = [0 for _rule in biosphere_omission_rules]
    feed_scales = cast(list[dict[str, object]], construction.get("scale_technosphere", []))
    feed_scale_matches = [0 for _rule in feed_scales]
    for exchange in native_exchanges:
        exchange_type = str(exchange["type"])
        if exchange_type == "production":
            continue
        input_node = context.input(exchange)
        waste_replacement = next(
            (
                rule
                for rule in waste_rules
                if _exact_exchange_matches(
                    exchange, input_node, cast(dict[str, object], rule["selector"])
                )
            ),
            None,
        )
        replaced_technosphere = next(
            (
                item
                for item in cast(
                    list[dict[str, object]],
                    construction.get("replace_technosphere", []),
                )
                if exchange_type == "technosphere"
                and _matches(input_node, cast(dict[str, object], item["selector"]))
            ),
            None,
        )
        omitted_technosphere = next(
            (
                (index, item)
                for index, item in enumerate(technosphere_omission_rules)
                if exchange_type == "technosphere"
                and _matches(input_node, cast(dict[str, object], item["selector"]))
            ),
            None,
        )
        replaced_biosphere = next(
            (
                (index, item)
                for index, item in enumerate(biosphere_replacement_rules)
                if exchange_type == "biosphere"
                and str(input_node.get("name", ""))
                == str(cast(dict[str, object], item["selector"])["name"])
                and str(input_node.get("unit", ""))
                == str(cast(dict[str, object], item["selector"])["unit"])
                and tuple(str(value) for value in input_node.get("categories", ()))
                == tuple(
                    str(value)
                    for value in cast(
                        list[object],
                        cast(dict[str, object], item["selector"])["categories"],
                    )
                )
            ),
            None,
        )
        omitted_biosphere = next(
            (
                (index, item)
                for index, item in enumerate(biosphere_omission_rules)
                if exchange_type == "biosphere"
                and str(input_node.get("name", ""))
                == str(cast(dict[str, object], item["selector"])["name"])
                and str(input_node.get("unit", ""))
                == str(cast(dict[str, object], item["selector"])["unit"])
                and tuple(str(value) for value in input_node.get("categories", ()))
                == tuple(
                    str(value)
                    for value in cast(
                        list[object],
                        cast(dict[str, object], item["selector"])["categories"],
                    )
                )
            ),
            None,
        )
        if waste_replacement is not None:
            waste_service_id = str(waste_replacement["service_id"])
            non_fossil = str(construction["ultimate_carbon_class"]) == "non_fossil"
            waste_owners.extend(
                _waste_carbon_owners(
                    context,
                    input_node,
                    waste_service_id,
                    scale=float(exchange["amount"]) / native_production_amount,
                    non_fossil=non_fossil,
                    database_name=database_name,
                )
            )
            record = _copied_native_exchange(exchange)
            if non_fossil:
                record["input"] = (database_name, provider_service_code(waste_service_id))
                _operation_exchange_metadata(
                    record,
                    str(input_node["reference product"]),
                    services[waste_service_id],
                    preserve_carbon_origin=False,
                )
                removed_technosphere.append(exchange)
            else:
                retained_technosphere.append(exchange)
            exchanges.append(record)
            project_input = cast(tuple[str, str], record["input"])
            replacement_rows.append(
                {
                    "native_input_database": str(input_node.key[0]),
                    "native_input_code": str(input_node.key[1]),
                    "project_input_database": project_input[0],
                    "project_input_code": project_input[1],
                    "amount": float(exchange["amount"]),
                    "exchange_type": "technosphere",
                    "quantity_and_uncertainty_preserved": True,
                }
            )
        elif (
            replaced_technosphere is None
            and omitted_technosphere is None
            and replaced_biosphere is None
            and omitted_biosphere is None
        ):
            if exchange_type == "technosphere":
                retained_technosphere.append(exchange)
            else:
                retained_biosphere.append(exchange)
            record = _copied_native_exchange(exchange)
            feed_scale = next(
                (
                    (index, rule)
                    for index, rule in enumerate(feed_scales)
                    if exchange_type == "technosphere"
                    and _matches(input_node, cast(dict[str, object], rule["selector"]))
                ),
                None,
            )
            if feed_scale is not None:
                scale_index, scale_rule = feed_scale
                feed_scale_matches[scale_index] += 1
                multiplier = float(cast(float | int | str, scale_rule["multiplier"]))
                _scale_feed(record, multiplier, service_id=service_id)
                context.semantic_rows.append(
                    {
                        "background_scenario": context.background_scenario,
                        "anchor_year": context.year,
                        "provider_service_id": service_id,
                        "semantic_class": "operation_only_extraction",
                        "ledger_entry_type": "native_feed_quantity_scaling",
                        "native_activity_database": str(activity.key[0]),
                        "native_activity_code": str(activity.key[1]),
                        "input_database": str(input_node.key[0]),
                        "input_code": str(input_node.key[1]),
                        "source_amount": float(exchange["amount"]),
                        "multiplier": multiplier,
                        "project_amount": record["amount"],
                        "reason": str(scale_rule["reason"]),
                    }
                )
            exchanges.append(record)
        elif omitted_technosphere is not None:
            removed_technosphere.append(exchange)
            omission_index, omission = cast(
                tuple[int, dict[str, object]],
                omitted_technosphere,
            )
            technosphere_omission_matches[omission_index] += 1
            source_unit_amount = float(exchange["amount"])
            technosphere_omission_amounts[omission_index] += source_unit_amount
            omission_id = str(omission["omission_id"])
            context.semantic_rows.append(
                {
                    "background_scenario": context.background_scenario,
                    "anchor_year": context.year,
                    "provider_service_id": service_id,
                    "semantic_class": "operation_only_extraction",
                    "ledger_entry_type": "native_exchange_omission",
                    "native_activity_database": str(activity.key[0]),
                    "native_activity_code": str(activity.key[1]),
                    "removed_input_name": _input_semantics(input_node)[0],
                    "removed_input_reference_product": _input_semantics(input_node)[1],
                    "removed_input_unit": _input_semantics(input_node)[2],
                    "removed_input_location": _input_semantics(input_node)[3],
                    "omission_id": omission_id,
                    "source_unit_amount": source_unit_amount,
                    "reason": str(omission["reason"]),
                }
            )
        elif replaced_technosphere is not None:
            removed_technosphere.append(exchange)
            replacement_id = str(replaced_technosphere["replacement_id"])
            matched_replacements.add(replacement_id)
            replacement = replacements[replacement_id]
            record = _copied_native_exchange(exchange)
            multiplier = float(cast(float | int | str, replacement.get("multiplier", 1.0)))
            if multiplier != 1.0:
                _scale_feed(record, multiplier, service_id=service_id)
            record["input"] = (database_name, provider_service_code(str(replacement["service_id"])))
            _operation_exchange_metadata(
                record,
                str(input_node["reference product"]),
                services[str(replacement["service_id"])],
                preserve_carbon_origin=(
                    str(replacement["service_id"]) in _UNCHANGED_CARBON_ORIGIN_SERVICES
                ),
            )
            exchanges.append(record)
            context.semantic_rows.append(
                {
                    "background_scenario": context.background_scenario,
                    "anchor_year": context.year,
                    "provider_service_id": service_id,
                    "semantic_class": "operation_only_extraction",
                    "ledger_entry_type": "native_exchange",
                    "native_activity_database": str(activity.key[0]),
                    "native_activity_code": str(activity.key[1]),
                    "removed_input_name": _input_semantics(input_node)[0],
                    "removed_input_reference_product": _input_semantics(input_node)[1],
                    "removed_input_unit": _input_semantics(input_node)[2],
                    "removed_input_location": _input_semantics(input_node)[3],
                    "removed_amount": float(exchange["amount"]),
                    "replacement_id": replacement_id,
                    "replacement_service_id": str(replacement["service_id"]),
                    "replacement_multiplier": multiplier,
                    "replacement_amount": record["amount"],
                }
            )
        elif replaced_biosphere is not None:
            removed_biosphere.append(exchange)
            replacement_index, biosphere_replacement = cast(
                tuple[int, dict[str, object]],
                replaced_biosphere,
            )
            biosphere_replacement_matches[replacement_index] += 1
            replacement_flow = context.biosphere.flow(
                cast(dict[str, object], biosphere_replacement["replacement_selector"])
            )
            amount = float(exchange["amount"])
            record = _copied_native_exchange(exchange)
            _replace_carbon_identity(record, replacement_flow)
            exchanges.append(record)
            replacement_row = {
                "source_flow_database": str(input_node.key[0]),
                "source_flow_code": str(input_node.key[1]),
                "source_flow_name": _input_semantics(input_node)[0],
                "source_flow_categories_json": json.dumps(list(input_node["categories"])),
                "replacement_flow_database": str(replacement_flow.key[0]),
                "replacement_flow_code": str(replacement_flow.key[1]),
                "replacement_flow_name": _input_semantics(replacement_flow)[0],
                "replacement_flow_categories_json": json.dumps(
                    list(replacement_flow["categories"])
                ),
                "source_amount": amount,
                "reason": str(biosphere_replacement["reason"]),
            }
            replacement_rows.append(replacement_row)
            context.semantic_rows.append(
                {
                    "background_scenario": context.background_scenario,
                    "anchor_year": context.year,
                    "provider_service_id": service_id,
                    "semantic_class": "operation_only_extraction",
                    "ledger_entry_type": "native_biosphere_replacement",
                    "native_activity_database": str(activity.key[0]),
                    "native_activity_code": str(activity.key[1]),
                    **replacement_row,
                }
            )
        else:
            removed_biosphere.append(exchange)
            omission_index, omission = cast(
                tuple[int, dict[str, object]],
                omitted_biosphere,
            )
            biosphere_omission_matches[omission_index] += 1
            context.semantic_rows.append(
                {
                    "background_scenario": context.background_scenario,
                    "anchor_year": context.year,
                    "provider_service_id": service_id,
                    "semantic_class": "operation_only_extraction",
                    "ledger_entry_type": "native_biosphere_omission",
                    "native_activity_database": str(activity.key[0]),
                    "native_activity_code": str(activity.key[1]),
                    "removed_flow_name": _input_semantics(input_node)[0],
                    "removed_flow_unit": _input_semantics(input_node)[2],
                    "removed_flow_categories_json": json.dumps(
                        list(input_node.get("categories", ())),
                        separators=(",", ":"),
                    ),
                    "source_unit_amount": float(exchange["amount"]),
                    "reason": str(omission["reason"]),
                }
            )
    for label, rules, matches in (
        ("feed quantity scaling", feed_scales, feed_scale_matches),
        (
            "biosphere origin replacement",
            biosphere_replacement_rules,
            biosphere_replacement_matches,
        ),
        ("biosphere boundary omission", biosphere_omission_rules, biosphere_omission_matches),
        ("operation-boundary omission", technosphere_omission_rules, technosphere_omission_matches),
    ):
        missing_rules = [
            rule["selector"] for rule, count in zip(rules, matches, strict=True) if count == 0
        ]
        if missing_rules:
            raise ValueError(
                f"Each {label} must match at least one native exchange in "
                f"{activity.key!r} for {service_id!r}; absent selectors: {missing_rules!r}. "
                "Use a source inventory containing the declared inputs in a new workspace."
            )
    omission_rule_index = {
        str(rule["omission_id"]): index for index, rule in enumerate(technosphere_omission_rules)
    }
    for group in cast(
        list[dict[str, object]],
        construction.get("omission_balance_groups", []),
    ):
        absolute_tolerance = float(cast(float | int | str, group.get("absolute_tolerance", 1e-12)))
        source_net_amount = math.fsum(
            technosphere_omission_amounts[omission_rule_index[str(omission_id)]]
            for omission_id in cast(list[object], group["omission_ids"])
        )
        if not math.isclose(
            source_net_amount,
            native_production_amount,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                "The declared operation-boundary omission does not preserve "
                f"the native net product for {service_id!r} in {activity.key!r}: "
                f"net omitted amount {source_net_amount!r}, required production "
                f"{native_production_amount!r}, absolute tolerance {absolute_tolerance!r}. "
                "Use a source inventory satisfying the declared material balance "
                "in a new workspace."
            )
        context.semantic_rows.append(
            {
                "background_scenario": context.background_scenario,
                "anchor_year": context.year,
                "provider_service_id": service_id,
                "semantic_class": "operation_only_extraction",
                "ledger_entry_type": "native_omission_balance",
                "native_activity_database": str(activity.key[0]),
                "native_activity_code": str(activity.key[1]),
                "balance_group_id": str(group["group_id"]),
                "source_net_omitted_amount": source_net_amount,
                "native_production_amount": native_production_amount,
                "source_balance_difference": (source_net_amount - native_production_amount),
                "absolute_tolerance": absolute_tolerance,
                "reason": str(group["reason"]),
            }
        )
    resolution_row["retained_technosphere_json"] = context.exchange_json(retained_technosphere)
    resolution_row["removed_technosphere_json"] = context.exchange_json(removed_technosphere)
    resolution_row["retained_biosphere_json"] = context.exchange_json(retained_biosphere)
    resolution_row["removed_biosphere_json"] = context.exchange_json(removed_biosphere)
    resolution_row["project_terms_json"] = json.dumps(
        replacement_rows,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    missing = sorted(replacements.keys() - matched_replacements)
    if missing:
        raise ValueError(
            f"Native operation {activity.key!r} lacks declared replacement inputs {missing!r}. "
            "Each replacement must match a native exchange; use the required "
            "source database in a new workspace."
        )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": str(service["unit"]),
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        **(
            {"direct_treatment_carbon_owners_json": _direct_carbon_owners_json(tuple(waste_owners))}
            if waste_rules
            else {}
        ),
        "exchanges": exchanges,
    }
