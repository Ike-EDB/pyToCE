"""Supply the dry supported-nickel catalyst for PA-finish castor-oil hardening."""

import json
from typing import cast

from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _RecipeContext

# US3207702A Examples VI/VIII describe this dry activated oil-hardening grade. Its 13% Ni content is
# a nominal product specification. The HCO ingredient owner separately applies PBL's
# supplied-catalyst dose.
_HCO_NICKEL_SHARE = 0.13


def _hco_nickel_dataset(
    context: _RecipeContext,
    service: dict[str, object],
    database_name: str,
) -> dict[str, object]:
    """Bind one kg dry Ni/silica to a catalyst-manufacturing proxy.

    The ANL (2015) GREET Pt/alumina preparation inventory supplies the operating
    and environmental quantities. Ni/silica replaces its metal and support.
    Native suppliers retain upstream manufacture and delivery. Fuel input
    scales the furnace inventory through its fuel/output ratio. The HCO
    ingredient calculation treats the spent catalyst cake.
    """
    _service_trace(context, service)
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    selectors = cast(dict[str, dict[str, object]], construction["supplies"])
    # One US short ton is 2000 international pounds. One million Btu is 1055.05585262 MJ; Table 5
    # prints 5.0 with a footnote marker.
    short_ton_kg = 907.18474
    mmbtu_mj = 1055.05585262
    fuel_mj = 5 * mmbtu_mj / short_ton_kg
    # The source's 120 US gal/t is consumed during drying/calcination. Its separate 240 gal/t
    # aqueous circuit is recycled.
    water_kg = 120 * 3.785411784 / short_ton_kg
    # Section 2.3 reports 17,237 g process NOx/t on its NOx mass basis.
    nox_kg = 17.237 / short_ton_kg
    supplies = {
        "nickel": _HCO_NICKEL_SHARE,
        "silica": 1 - _HCO_NICKEL_SHARE,
        # Source kg/kg catalyst: environmental-operation coefficients.
        "nitric_acid": 0.04,
        "hydrochloric_acid": 0.08,
        "sodium_hydroxide": 0.003,
        "water": water_kg,
        "heat": fuel_mj,
        # 3.6 MJ/kWh converts electrical energy.
        "electricity": 0.07 * mmbtu_mj / (3.6 * short_ton_kg),
    }
    key = database_name, provider_service_code(service_id)
    rows: list[dict[str, object]] = [{"input": key, "amount": 1.0, "type": "production"}]
    native_supplies: dict[str, dict[str, object]] = {}
    for label, amount in supplies.items():
        node, trace = context.activity(
            selectors[label],
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id=label,
            semantic_class="service_normalization",
        )
        if label == "heat":
            # Original gas properties retain the net heating value omitted by
            # premise. Dated gas demand and furnace production determine the
            # current MJ fuel/MJ heat; buy that complete furnace once.
            original = context.source.activity(selectors[label], year=2025)
            original_gas = next(
                row
                for row in context.exchanges(original)
                if row["type"] == "technosphere"
                and context.input(row)["reference product"] == "natural gas, high pressure"
            )
            properties = cast(dict[str, dict[str, object]], original_gas["properties"])
            net_heat = float(cast(float, properties["heating value, net"]["amount"]))
            native = context.exchanges(node)
            production = _self_production_amount(node, native, method="HCO catalyst furnace")
            gas = next(
                row
                for row in native
                if row["type"] == "technosphere"
                and context.input(row)["reference product"] == "natural gas, high pressure"
            )
            amount = fuel_mj / (float(gas["amount"]) * net_heat / production)
        unit = str(node["unit"])
        rows.append(
            {
                "input": cast(tuple[str, str], node.key),
                "amount": amount,
                "type": "technosphere",
                "unit": unit,
            }
        )
        native_supplies[label] = {"amount": amount, "unit": unit}
        trace["project_terms_json"] = json.dumps(
            {"demand_per_kg_prepared_powder": amount, "unit": unit}, sort_keys=True
        )
    for name, unit, amount in (
        ("Nitrogen oxides", "kilogram", nox_kg),
        # Consumed water evaporates on the source drying/calcination basis.
        # Its existing elementary flow uses m3, at the 1000 kg/m3 convention.
        ("Water", "cubic meter", water_kg / 1000),
    ):
        node = context.biosphere.flow({"name": name, "unit": unit, "categories": ["air"]})
        rows.append(
            {
                "input": cast(tuple[str, str], node.key),
                "amount": amount,
                "type": "biosphere",
                "unit": unit,
            }
        )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": rows,
        "catalyst_manufacturing_reference": {
            "powder_kg": {"Ni": _HCO_NICKEL_SHARE, "SiO2": 1 - _HCO_NICKEL_SHARE},
            "native_supplies": native_supplies,
            "fuel_input_mj": fuel_mj,
            "NOx_kg": nox_kg,
            "evaporated_water_kg": water_kg,
        },
        "comment": (
            "Dry 13% Ni/silica catalyst for PA-finish HCO. The ANL Pt/alumina preparation "
            "inventory provides a manufacturing proxy with Ni/silica replacing the metal and "
            "support. The dry activated catalyst uses protected handling. The HCO ingredient "
            "calculation supplies its use rate and treats the whole spent cake."
        ),
    }
