"""Protect ethoxylate supplies, source emissions and auxiliary material quantities."""

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from material_recipe_fixtures import physical_material_recipe
from pa_ethoxylation_fixtures import (
    PA_ETHOXYLATION_SERVICES,
    ethoxylation_biosphere,
    ethoxylation_native_inventory,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_source_closures,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_compilation import (
    _service_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


@pytest.mark.parametrize("original", [True, False])
def test_ethoxylation_uses_actual_dated_supplies_and_complete_environmental_function(
    tmp_path: Path, original: bool
) -> None:
    """Compile both ingredient recipes from synthetic original and future inventories.

    This test checks retained source exchanges, signed wastewater demand and
    oleyl alcohol, hydrogenated castor oil and ethylene oxide inputs; separate tests
    cover hydrogenated castor oil manufacture. No substitute process is created.
    """
    catalogue = load_provider_catalogue()
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], catalogue["services"])
    }
    names = BrightwayNames(tmp_path / "brightway", "pa_ethoxylation", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(ethoxylation_biosphere(), searchable=False)
        bd.Database("original").write(
            ethoxylation_native_inventory(catalogue, "original", original=True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                ethoxylation_native_inventory(catalogue, database, original=False), searchable=False
            )
        before = {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        }
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        compiled = {
            ("providers", provider_service_code(sid)): _service_dataset(
                context, services[sid], services, "providers"
            )
            for sid in PA_ETHOXYLATION_SERVICES
        }
        carbon, hydrogen, oxygen = 12.011, 1.008, 15.999
        for sid, (c, h, o, n) in zip(
            PA_ETHOXYLATION_SERVICES, ((18, 36, 1, 10), (57, 110, 9, 40)), strict=True
        ):
            key = ("providers", provider_service_code(sid))
            node = compiled[key]
            record = json.loads(str(node["pa_ethoxylation_reference_json"]))
            exchanges = cast(list[dict[str, object]], node["exchanges"])
            inputs = {
                cast(tuple[str, str], ex["input"]): ex
                for ex in exchanges
                if ex["type"] == "technosphere"
            }
            emissions = {
                cast(tuple[str, str], ex["input"])[1]: ex
                for ex in exchanges
                if ex["type"] == "biosphere"
            }
            precursor_mass = c * carbon + h * hydrogen + o * oxygen
            eo_mass = 2 * carbon + 4 * hydrogen + oxygen
            product_mass = precursor_mass + n * eo_mass
            precursor = 1.08 * precursor_mass / product_mass
            eo = 1.08 * n * eo_mass / product_mass
            bio_c = c * carbon / product_mass
            fossil_c = 2 * n * carbon / product_mass
            production = next(ex for ex in exchanges if ex["type"] == "production")
            assert production["amount"] == 1.0
            assert record["main_gross"] == pytest.approx(1.08)
            for field, expected in {
                "precursor_kg": precursor,
                "ethylene_oxide_kg": eo,
                "nominal_fossil_kg_c": fossil_c,
                "nominal_biogenic_kg_c": bio_c,
                "main_aerosol_kg": 0.006,
                "precursor_water_kg": 0.001,
                "ethylene_oxide_air_kg": 0.002,
                "original_co2_kg": 0.04,
            }.items():
                assert record[field] == pytest.approx(expected, rel=2e-14)
            precursor_key = (
                (database, "oleyl")
                if sid == "pa_oleth10"
                else ("providers", provider_service_code("pa_hydrogenated_castor_oil"))
            )
            assert inputs[precursor_key]["amount"] == pytest.approx(precursor)
            assert set(inputs) == {
                precursor_key,
                (database, "eo_a"),
                (database, "eo_b"),
                (database, "factory"),
                (database, "electricity"),
                (database, "heat"),
                *((database, f"water_{index}") for index in range(4)),
            }
            for code, amount in (
                ("factory", 0.002),
                ("electricity", 0.7 if original else 0.9),
                ("heat", 2.5),
            ):
                assert inputs[(database, code)]["amount"] == pytest.approx(amount)
            assert inputs[(database, "eo_a")]["amount"] == pytest.approx(
                eo * (0.6 if original else 0.25)
            )
            assert inputs[(database, "eo_b")]["amount"] == pytest.approx(
                eo * (0.4 if original else 0.75)
            )
            assert (database, "donor_alcohol") not in inputs

            # The changed future CO2/mist remain native environmental functions,
            # while the original product reference stays stationary.
            co2 = 0.04 if original else 0.07
            assert record["dated_native_co2_kg"] == pytest.approx(co2)
            assert emissions["co2_fossil"]["amount"] == pytest.approx(co2 * 2 * n / (c + 2 * n))
            assert emissions["co2_bio"]["amount"] == pytest.approx(co2 * c / (c + 2 * n))
            assert emissions["co2_bio"]["name"] == "Carbon dioxide, non-fossil"
            assert emissions["co2_bio"]["unit"] == "kilogram"
            assert emissions["co2_bio"]["categories"] == ("air",)
            if original:
                source_co2 = next(
                    ex
                    for ex in before[database, "ethoxylation"]
                    if ex["input"] == ("biosphere", "co2_fossil")
                )
                expected_properties = deepcopy(source_co2["properties"])
                expected_properties["carbon content, fossil"]["amount"] = 0.0
                expected_properties["carbon content, non-fossil"]["amount"] = 0.27
                assert emissions["co2_bio"]["properties"] == expected_properties
                assert emissions["co2_fossil"]["properties"] == source_co2["properties"]
                assert emissions["co2_bio"]["flow"] == "co2_bio"
                assert emissions["co2_fossil"]["flow"] == "co2_fossil"
            else:
                assert "properties" not in emissions["co2_bio"]
                assert "properties" not in emissions["co2_fossil"]
                assert "flow" not in emissions["co2_bio"]
            assert emissions["mist"]["amount"] == pytest.approx(0.006 if original else 0.009)
            assert emissions["alcohol_water"]["amount"] == pytest.approx(0.001)
            assert set(emissions) == {
                "co2_fossil",
                "co2_bio",
                "eo_air",
                "mist",
                "alcohol_water",
                "water_air",
                "water_surface",
            }
            for index in range(4):
                assert inputs[(database, f"water_{index}")]["amount"] == pytest.approx(
                    -(index + 1) * 0.0001
                )
            for ex in (
                inputs[(database, "eo_a")],
                emissions["co2_bio"],
                emissions["mist"],
                inputs[(database, "water_0")],
            ):
                amount = cast(float, ex["amount"])
                assert ex["loc"] == pytest.approx(math.log(abs(amount)) + 0.03)
                assert ex["scale"] == 0.21
                assert ex["minimum"] == pytest.approx(amount * (2 if amount < 0 else 0.5))
                assert ex["maximum"] == pytest.approx(amount * (0.5 if amount < 0 else 2))
                assert ex["allocation"] == "invented source allocation"
            properties = cast(dict[str, dict[str, object]], production["properties"])
            assert properties["dry mass"]["amount"] == pytest.approx(1.0)
            assert properties["carbon allocation"]["amount"] == pytest.approx(bio_c + fossil_c)
        assert {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        } == before
