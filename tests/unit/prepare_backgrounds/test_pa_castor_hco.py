"""Check complete castor/HCO parent quantities at real source owners."""

import math
from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from pa_castor_hco_fixtures import castor_hco_biosphere, castor_hco_native_inventory

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
    _service_source_closures,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
    _NativeResolverCache,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_castor_hco import (
    _pa_castor_hco_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_finish_residues import (
    _pa_finish_residue_load,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


@pytest.mark.parametrize("original", [True, False])
def test_castor_hco_closes_actual_materials_and_binds_dated_whole_duties(
    tmp_path: Path, original: bool
) -> None:
    """Test crude-oil extraction, refining and HCO residue treatment.

    Crude and refined oil supplies are checked through a Brightway LCI solve.
    HCO direct exchanges and filter-cake quantities are checked separately.
    """
    catalogue = load_provider_catalogue()
    services = {
        str(s["provider_service_id"]): s
        for s in cast(list[dict[str, object]], catalogue["services"])
    }
    selected = ("pa_castor_crude_oil", "pa_hydrogenated_castor_oil")
    names = BrightwayNames(tmp_path / "brightway", "pa_castor_hco", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(castor_hco_biosphere(), searchable=False)
        bd.Database("original").write(
            castor_hco_native_inventory(catalogue, "original", original=True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                castor_hco_native_inventory(catalogue, database, original=False), searchable=False
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
                str(r["method_authority_id"]): r for r in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        castor, hco = (
            _pa_castor_hco_dataset(context, services[sid], "providers") for sid in selected
        )
        reference = cast(dict[str, object], castor["castor_hco_reference"])
        p = cast(dict[str, float], reference["physical_kg"])
        a = cast(dict[str, float], reference["allocation"])
        assert p["crude_oil_dry"] == 1.0
        assert p["seed_dry"] == pytest.approx(1 / (0.47 * 0.96))
        assert p["crude_oil_water"] == pytest.approx(0.0032 / (1 - 0.0032))
        assert p["seed_dry"] == pytest.approx(1 + p["seed_cake_dry"] + p["oil_final_water"])
        assert p["seed_wet"] == pytest.approx(
            p["seed_water_evaporated"] + p["seed_cake_water"] + p["seed_dry"] + p["crude_oil_water"]
        )
        assert a["crude_oil"] + a["seed_cake"] == pytest.approx(1.0)
        assert a["crude_oil"] == pytest.approx(1 / (p["seed_dry"] - p["oil_final_water"]))
        assert a["native_unallocated_dry_seed_kg"] == pytest.approx(2.4 * 0.89 / 0.4)

        castor_rows = cast(list[dict[str, object]], castor["exchanges"])
        castor_amounts: dict[tuple[str, str], float] = {}
        for row in castor_rows:
            key = cast(tuple[str, str], row["input"])
            castor_amounts[key] = castor_amounts.get(key, 0.0) + float(cast(float, row["amount"]))
        assert (database, "soybean") not in castor_amounts
        assert ("biosphere", "correction") not in castor_amounts
        assert castor_amounts[database, "seed"] == pytest.approx(a["crude_oil"] * p["seed_wet"])
        physical_factor = p["seed_dry"] / (2.4 * 0.89)
        allocated_factor = physical_factor * a["crude_oil"]
        assert castor_amounts[database, "hexane_supply"] == pytest.approx(0.003 * allocated_factor)
        assert castor_amounts["biosphere", "hexane"] == pytest.approx(0.0002 * allocated_factor)
        assert castor_amounts["biosphere", "oil_water"] == pytest.approx(0.002 * allocated_factor)
        assert castor_amounts[database, "drying_heat"] == pytest.approx(
            a["crude_oil"] * 1940 * 0.004184 * p["seed_water_evaporated"]
        )
        assert castor_amounts[database, "drying_electricity"] == pytest.approx(
            (0.12 if original else 0.08) * allocated_factor
            + a["crude_oil"] * 0.0155 * 94697 / 5907 * p["seed_water_evaporated"]
        )
        assert all(
            cast(tuple[str, str], row["input"])[0] in {database, "biosphere"}
            for row in castor_rows
            if row["type"] != "production"
        )
        # Full source uncertainty survives nonunit p=2/p=5 and actual local
        # allocation, rather than being replaced by a deterministic coefficient.
        hexane = next(r for r in castor_rows if r["input"] == (database, "hexane_supply"))
        assert hexane["loc"] == pytest.approx(math.log(0.003 * allocated_factor) + 0.02)
        assert hexane["scale"] == 0.19
        assert hexane["allocation"] == "invented source allocation"

        # The separate actual refinery owner preserves its own nonunit output,
        # allocated crude demand and complete native support/receiving records.
        oil_services = {"pa_castor_crude_oil", "pa_castor_oil"}
        provider = PreparedProviderAuthority(
            identity="castor_complete_refinery",
            interface_database="castor_interface",
            catalogue={
                **catalogue,
                "services": [services[sid] for sid in oil_services],
                "routes": [],
            },
            routes=pd.DataFrame(),
            method_authorities=_method_authority_rows(catalogue),
        )
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2025 if original else 2050,
            ordinary_database=database,
            technology_database=database,
            material_recipe=physical_material_recipe(),
            provider=provider,
            resolver_cache=_NativeResolverCache("original", "biosphere"),
        )
        refined = bd.get_node(
            database=frontier.database_name, code=provider_service_code("pa_castor_oil")
        )
        crude = bd.get_node(
            database=frontier.database_name, code=provider_service_code("pa_castor_crude_oil")
        )
        expected = []
        for source in before[database, "refinery"]:
            row = deepcopy(source)
            row["output"] = refined.key
            if row["type"] == "production":
                row["input"] = refined.key
                row.pop("properties", None)
                row.pop("classifications")
            elif row["input"] == (database, "crude_market"):
                row.update(
                    input=crude.key, name=services["pa_castor_crude_oil"]["reference_product"]
                )
                for field in ("activity", "flow", "properties", "classifications"):
                    row.pop(field, None)
            expected.append(row)
        assert sorted((dict(ex) for ex in refined.exchanges()), key=repr) == sorted(
            expected, key=repr
        )
        crude_lca = LCA({crude: 1.0})
        crude_lca.lci()
        marker = bd.get_node(database="biosphere", code="marker")
        crude_marker = float(crude_lca.inventory[crude_lca.dicts.biosphere[marker.id], :].sum())
        refined_lca = LCA({refined: 1.0})
        refined_lca.lci()
        production = 3.0 if original else 7.0
        crude_demand = float(np.float32((1.05 if original else 1.08) * production)) / production
        native_support = math.fsum(
            float(np.float32(amount * production)) / production
            for amount in (0.012, 0.04, 0.003, 0.15, 1e-10, 0.0002)
        ) * float(np.float32(0.007))
        assert float(refined_lca.inventory[refined_lca.dicts.biosphere[marker.id], :].sum()) == (
            pytest.approx(crude_demand * crude_marker + native_support, rel=1e-13)
        )

        h = cast(dict[str, object], hco["castor_hco_reference"])
        cake = cast(dict[str, float], h["cake_components_kg"])
        assert h["cake_kg"] == pytest.approx(math.fsum(cake.values()))
        # Nominal molar masses: castor oil 933.450 and HCO 939.498 g/mol.
        # PBL powder, earth and acid total 0.004 kg/kg feed; the patent's
        # collected cake contains 40% retained oil and 60% non-fat matter.
        feed = 1 / (1 + 6.048 / 933.450 - (0.40 / 0.60) * 0.004)
        assert h["castor_oil_kg"] == pytest.approx(feed)
        assert h["hydrogen_supply_kg"] == pytest.approx(feed * 6.048 / 933.450 * 1.1 / 0.93)
        assert h["catalyst_kg"] == pytest.approx(0.002 * feed)
        assert h["filter_aid_kg"] == pytest.approx(0.0015 * feed)
        assert h["citric_acid_kg"] == pytest.approx(0.0005 * feed)
        assert cake == pytest.approx(
            {
                "oil": (0.40 / 0.60) * 0.004 * feed,
                "Ni": 0.13 * 0.002 * feed,
                "SiO2": 0.87 * 0.002 * feed,
                "unknown_mineral": 0.0015 * feed,
                "citric_acid": 0.0005 * feed,
            }
        )
        incoming = math.fsum(
            float(cast(float, h[k]))
            for k in (
                "castor_oil_kg",
                "castor_water_kg",
                "hydrogen_supply_kg",
                "catalyst_kg",
                "filter_aid_kg",
                "citric_acid_kg",
            )
        )
        outgoing = 1 + math.fsum(
            float(cast(float, h[k])) for k in ("castor_water_kg", "hydrogen_vent_kg", "cake_kg")
        )
        assert incoming == pytest.approx(outgoing, abs=1e-14)
        assert h["castor_water_kg"] == pytest.approx(
            float(cast(float, h["castor_oil_kg"])) * 0.0005 / (1 - 0.0005)
        )
        hco_rows = cast(list[dict[str, object]], hco["exchanges"])
        hco_amounts = {
            cast(tuple[str, str], r["input"]): float(cast(float, r["amount"])) for r in hco_rows
        }
        for service_id, amount in (
            ("pa_castor_oil", h["castor_oil_kg"]),
            ("hco_supported_nickel", h["catalyst_kg"]),
            ("pa_hco_cake_treatment", -float(cast(float, h["cake_kg"]))),
            ("pa_condensate_collection", -float(cast(float, h["castor_water_kg"])) / 1000),
        ):
            assert hco_amounts["providers", provider_service_code(service_id)] == pytest.approx(
                amount
            )
        assert hco_amounts[database, "hydrogen"] == pytest.approx(h["hydrogen_supply_kg"])
        assert hco_amounts[database, "clay"] == pytest.approx(h["filter_aid_kg"])
        assert hco_amounts[database, "citric_acid"] == pytest.approx(0.0005 * feed)
        assert hco_amounts[database, "steam"] == pytest.approx(0.030 * feed)
        assert hco_amounts[database, "drying_electricity"] == pytest.approx(0.010 * feed)
        assert {
            cast(tuple[str, str], r["input"])[1]: r["unit"]
            for r in hco_rows
            if r["input"]
            in ((database, "citric_acid"), (database, "steam"), (database, "drying_electricity"))
        } == {
            "citric_acid": "kilogram",
            "steam": "kilogram",
            "drying_electricity": "kilowatt hour",
        }
        assert ("providers", provider_service_code("pa_ester_operation")) not in hco_amounts
        assert hco_amounts["biosphere", "hydrogen"] == pytest.approx(h["hydrogen_vent_kg"])
        properties = cast(dict[str, dict[str, object]], hco_rows[0]["properties"])
        assert properties["water in wet mass"]["amount"] == 0.0
        # The receiver accounts separately for acid organic matter, retained triglyceride, and
        # silica.
        residue = _pa_finish_residue_load(
            cast(dict[str, object], services["pa_hydrogenated_castor_oil"]["construction"]),
            residue_grade="hco_cake",
        )
        organic_c = cake["oil"] * 684.627 / 939.498 + cake["citric_acid"] * 72.066 / 192.123
        organic_h = cake["oil"] * 110.880 / 939.498 + cake["citric_acid"] * 8.064 / 192.123
        organic_o = cake["oil"] * 143.991 / 939.498 + cake["citric_acid"] * 111.993 / 192.123
        assert residue.active_elements_kg == pytest.approx(
            {
                "C": organic_c,
                "H": organic_h,
                "O": organic_o + cake["SiO2"] * 31.998 / 60.083,
                "Si": cake["SiO2"] * 28.085 / 60.083,
                "Ni": cake["Ni"],
                "N": 0.0,
                "S": 0.0,
                "Na": 0.0,
            }
        )
        assert residue.wet_mass_kg == pytest.approx(h["cake_kg"])
        assert residue.unknown_mineral_kg == pytest.approx(0.0015 * feed)
        assert residue.carrier_water_kg == 0
        assert residue.nonfuel_salt_elements_kg == {}
        assert residue.biogenic_carbon_kg == pytest.approx(organic_c)
        assert residue.net_energy_mj == pytest.approx(
            34.016 * organic_c
            + 124.265 * organic_h
            - 9.8324 * organic_o
            + cake["Ni"] * 239.7 / 58.6934
            - 2.4417 * 9 * organic_h
        )
        assert residue.waste_formed_water_kg == pytest.approx(organic_h * 18.015 / 2.016)
        assert math.fsum(residue.active_elements_kg.values()) + residue.unknown_mineral_kg == (
            pytest.approx(residue.wet_mass_kg)
        )
        assert {
            node.key: [dict(ex) for ex in node.exchanges()]
            for db in {"original", database}
            for node in bd.Database(db)
        } == before
