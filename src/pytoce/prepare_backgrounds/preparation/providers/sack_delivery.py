"""Construct sealed LDPE sacks and delivery for moist TPS pellets."""

from math import fsum
from typing import cast

from .native_recipe import _self_production_amount
from .native_resolution import _matches
from .recipe_context import _RecipeContext


def _tps_sack_delivery_recipe(
    context: _RecipeContext, construction: dict[str, object]
) -> dict[str, object]:
    """Supply protected delivery per kg moist technical TPS, including its sacks.

    Each 100 g sack carries 24.9 kg TPS. Full film and bag operations retain
    their own waste, utilities and capital. Gross bag input follows the original native signed
    waste complement, while the admitted material recipe supplies film yield.
    Filling and closure use the measured wood-pelletchips bagging operation
    as a proxy per kg moist TPS. Both native operations keep their production
    basis and exchange uncertainty; Brightway normalizes them once.

    Empty packing and filled sacks each receive the selected freight function.
    Used packing leaves once at the native PE recycling cut-off;
    this creates no avoided-material credit or second-cycle recycling burden.
    """
    bag = cast(dict[str, object], construction["bag_making"])
    source = context.source.activity(cast(dict[str, object], bag["selector"]), year=2025)
    exchanges = context.exchanges(source)
    production = _self_production_amount(source, exchanges, method="TPS sack making")
    waste_selectors = cast(list[dict[str, object]], construction["bag_waste_selectors"])
    retention = (
        1.0
        + fsum(
            float(exchange["amount"])
            for exchange in exchanges
            if exchange["type"] == "technosphere"
            and any(_matches(context.input(exchange), selector) for selector in waste_selectors)
        )
        / production
    )
    film_basis = next(
        item for item in context.material_recipe.conversion if item.operation == "conversion_film"
    )
    # Ferplast's 100 g empty LDPE sack packages 24.9 kg moist TPS. Divide sack mass by cargo mass.
    packing = float(cast(float, construction["sack_mass_kg"])) / float(
        cast(float, construction["sack_payload_kg"])
    )
    bag_input = packing / retention
    film_input = bag_input / film_basis.finished_kg_per_native_service_kg
    # NORSUS OR.23.14 gives 3.65 kWh/t, or 0.00365 kWh/kg bagged pelletchips.
    # Apply this whole bagging operation per kg moist TPS.
    bagging = float(cast(float, construction["bagging_electricity_kwh_per_kg"]))
    return {
        "components": [
            {**cast(dict[str, object], construction[role]), "amount": amount}
            for role, amount in (
                ("ldpe", film_input),
                ("film", film_input),
                ("bag_making", bag_input),
                ("bagging_electricity", bagging),
                # One kg cargo plus its sack travels out; the empty sack
                # travels in separately, through the same delivery proxy.
                ("freight", fsum((packing, 1.0, packing))),
                ("packing_cutoff", -packing),
            )
        ]
    }
