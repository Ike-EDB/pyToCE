"""Chemical recycling and treatment admission using synthetic source inventories."""

from typing import Any, cast

import pytest

from pytoce.prepare_backgrounds.preparation.providers import recipe_compilation


def test_preparation_requires_its_declared_plastic_reject(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext, dict[str, dict[str, object]]
    ],
) -> None:
    """Require the preparation inventory's declared reject denominator."""
    context, services = chemical_compilation_case
    sorting = context.technology.node(("technology", "sorting"))
    reject = next(
        exchange
        for exchange in sorting.technosphere()
        if exchange["input"] == ("technology", "reject_market")
    )
    reject.delete()
    with pytest.raises(ValueError, match="one source-defined plastic reject"):
        recipe_compilation._service_dataset(
            context,
            services["chemical_recycling_preparation_non_fossil"],
            services,
            "provider_frontier",
        )


def test_conversion_requires_the_declared_native_direct_carbon_flow(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext, dict[str, dict[str, object]]
    ],
) -> None:
    """Reject a common hydrothermal operation lacking its declared direct carbon emission."""
    context, services = chemical_compilation_case
    for index in range(4):
        activity = context.technology.node(("technology", f"htt_{index}"))
        next(iter(activity.biosphere())).delete()
    with pytest.raises(
        ValueError, match="Each declared conversion carbon flow must resolve exactly once"
    ):
        recipe_compilation._service_dataset(
            context,
            services["chemical_recycling_conversion_non_fossil"],
            services,
            "provider_frontier",
        )


@pytest.mark.parametrize("case", ["semantic_vector", "outside_tolerance", "within_tolerance"])
def test_hydrothermal_siblings_require_the_same_normalized_inventory(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext,
        dict[str, dict[str, object]],
    ],
    case: str,
) -> None:
    """Admit native sibling differences only within the declared premise tolerance."""
    context, services = chemical_compilation_case
    service = services["chemical_recycling_conversion_non_fossil"]
    construction = service["construction"]
    assert isinstance(construction, dict)
    activity = context.technology.activity(
        {
            "name": construction["activity_name"],
            "reference_product": "distillate gas oil",
            "unit": construction["unit"],
            "exact_location": construction["exact_location"],
        },
        year=context.year,
    )
    if case == "semantic_vector":
        activity.new_exchange(
            input=("technology", "metal"),
            amount=0.01,
            type="technosphere",
        ).save()
        message = "do not retain one semantic operation vector"
    else:
        utility = next(
            exchange
            for exchange in activity.technosphere()
            if exchange.input.key == ("technology", "utility")
        )
        utility["amount"] = float(utility["amount"]) * (
            1.0 + (2e-7 if case == "within_tolerance" else 1e-3)
        )
        utility.save()
        message = "sibling operations differ beyond the accepted tolerance"
    context.native_exchanges.clear()

    if case != "within_tolerance":
        with pytest.raises(ValueError, match=message):
            recipe_compilation._service_dataset(
                context,
                service,
                services,
                "provider_frontier",
            )
    else:
        dataset = recipe_compilation._service_dataset(
            context,
            service,
            services,
            "provider_frontier",
        )
        retained_utility = next(
            exchange
            for exchange in cast(list[dict[str, Any]], dataset["exchanges"])
            if exchange["input"] == ("technology", "utility")
        )
        assert retained_utility["amount"] == 2.0
        alignment = next(
            row
            for row in context.semantic_rows
            if row.get("ledger_entry_type") == "chemical_recycling_conversion_alignment"
        )
        assert 1e-12 < cast(float, alignment["maximum_relative_sibling_difference"]) < 5e-7


@pytest.mark.parametrize("failure", ["production", "missing", "duplicate"])
def test_treatment_clone_requires_native_production_and_carbon_occurrences(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext, dict[str, dict[str, object]]
    ],
    failure: str,
) -> None:
    """Admit the actual reference basis and each source carbon exchange once."""
    context, services = chemical_compilation_case
    activity = context.technology.node(("technology", "incineration"))
    if failure == "production":
        production = next(iter(activity.production()))
        production["amount"] = -0.5
        production.save()
        message = "must have a -1 kilogram reference production"
    else:
        if failure == "missing":
            next(
                exchange
                for exchange in activity.biosphere()
                if exchange["input"] == ("biosphere", "co2_urban_f")
            ).delete()
        else:
            activity.new_exchange(
                input=("biosphere", "co2_urban_f"), amount=0.5, type="biosphere"
            ).save()
        message = "must match exactly one native operation exchange"
    with pytest.raises(ValueError, match=message):
        recipe_compilation._service_dataset(
            context,
            services["plastic_waste_incineration_gb_non_fossil"],
            services,
            "provider_frontier",
        )


@pytest.mark.parametrize("duplicate", [False, True])
def test_native_fossil_treatment_requires_each_direct_carbon_exchange_once(
    chemical_compilation_case: tuple[
        recipe_compilation._RecipeContext, dict[str, dict[str, object]]
    ],
    duplicate: bool,
) -> None:
    """Reject missing or repeated native direct carbon and preserve provider selectors."""
    context, services = chemical_compilation_case
    activity = context.technology.node(("technology", "open_burning"))
    if duplicate:
        activity.new_exchange(
            input=("biosphere", "co2_high_f"), amount=0.5, type="biosphere"
        ).save()
    else:
        next(
            exchange
            for exchange in activity.biosphere()
            if exchange["input"] == ("biosphere", "co2_high_f")
        ).delete()
    with pytest.raises(ValueError, match="must match exactly one native operation exchange"):
        recipe_compilation._service_dataset(
            context, services["mismanaged_open_burning_mixed"], services, "provider_frontier"
        )
