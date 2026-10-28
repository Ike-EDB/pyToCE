"""Provider authority and external-inventory guard contracts."""

from typing import cast

import pytest

from pytoce.prepare_backgrounds.preparation.providers import (
    recipe_compilation,
)


def test_reject_market_preserves_unmatched_native_biosphere_exchange(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext,
        dict[str, dict[str, object]],
    ],
) -> None:
    """Retain native direct emissions unrelated to the replaced treatment children."""
    context, services = chemical_compilation_case
    service = services["plastic_waste_market_gb_non_fossil"]
    construction = cast(dict[str, object], service["construction"])
    market = context.technology.activity(
        cast(dict[str, object], construction["selector"]), year=context.year
    )
    market.new_exchange(
        input=("biosphere", "water"),
        amount=0.004,
        type="biosphere",
    ).save()
    context.native_exchanges.clear()

    dataset = recipe_compilation._service_dataset(
        context,
        service,
        services,
        "provider_frontier",
    )

    assert {
        (row["input"], row["amount"], row["type"])
        for row in cast(list[dict[str, object]], dataset["exchanges"])
        if row["input"] == ("biosphere", "water")
    } == {(("biosphere", "water"), 0.004, "biosphere")}


@pytest.mark.parametrize(
    ("missing_rule", "message"),
    [
        (
            {
                "omit_biosphere": [
                    {
                        "selector": {
                            "name": "absent direct emission",
                            "unit": "kilogram",
                            "categories": ["air"],
                        },
                        "reason": "the declared boundary owns this direct emission",
                    }
                ]
            },
            "biosphere boundary omission must match at least one",
        ),
        (
            {
                "omit_technosphere": [
                    {
                        "omission_id": "absent_feed",
                        "selector": {
                            "name": "absent feed",
                            "reference_product": "absent feed",
                            "unit": "kilogram",
                        },
                        "reason": "the declared boundary owns this feed",
                    }
                ]
            },
            "operation-boundary omission must match at least one",
        ),
    ],
)
def test_operation_authority_requires_each_declared_external_exchange(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext,
        dict[str, dict[str, object]],
    ],
    missing_rule: dict[str, object],
    message: str,
) -> None:
    """Fail when an exact source-backed omission is absent from the selected inventory."""
    context, services = chemical_compilation_case
    construction = {
        "kind": "operation_only",
        "base": {
            "source_role": "native_ecoinvent_provider",
            "selector": {
                "name": "utility supply",
                "reference_product": "utility",
                "unit": "kilogram",
                "exact_location": "GB",
            },
        },
        **missing_rule,
    }

    with pytest.raises(ValueError, match=message):
        recipe_compilation._service_dataset(
            context,
            {**services["chemical_recycling_preparation_fossil"], "construction": construction},
            services,
            context.provider_database,
        )


def test_operation_omissions_must_preserve_native_net_product(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext,
        dict[str, dict[str, object]],
    ],
) -> None:
    """Reject a source boundary whose omitted net amount differs from production."""
    context, services = chemical_compilation_case
    base = {
        "source_role": "native_ecoinvent_provider",
        "selector": {
            "name": "treatment of waste mixed polyolefins, processed, hydrothermal treatment",
            "reference_product": "naphtha",
            "unit": "kilogram",
            "exact_location": "GB",
        },
    }
    activity = context.technology.activity(
        cast(dict[str, object], base["selector"]), year=context.year
    )
    utility_exchange = next(
        exchange
        for exchange in context.exchanges(activity)
        if str(exchange["type"]) == "technosphere"
        and context.input(exchange).key == ("technology", "utility")
    )
    utility = context.input(utility_exchange)
    construction = {
        "kind": "operation_only",
        "base": base,
        "omit_technosphere": [
            {
                "omission_id": "utility",
                "selector": {
                    "name": utility["name"],
                    "reference_product": utility["reference product"],
                    "unit": utility["unit"],
                },
                "reason": "bounded balance authority",
            }
        ],
        "omission_balance_groups": [
            {
                "group_id": "net_boundary",
                "omission_ids": ["utility"],
                "absolute_tolerance": 1e-12,
                "reason": "the omitted net amount must equal native production",
            }
        ],
    }

    with pytest.raises(ValueError, match="does not preserve the native net product"):
        recipe_compilation._service_dataset(
            context,
            {**services["chemical_recycling_preparation_fossil"], "construction": construction},
            services,
            context.provider_database,
        )
