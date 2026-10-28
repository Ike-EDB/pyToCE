"""PHB fermentation preserves residue operations and substrate-origin boundaries."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _method_authority_rows,
    load_provider_catalogue,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.frontier_database import (
    prepare_provider_frontier,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _NativeResolverCache,
)


def test_phb_frontier_preserves_residue_operations_and_substrate_carbon(tmp_path: Path) -> None:
    """Compile PHB recipes with synthetic glucose supply and signed residue treatment."""

    def stored_amount(amount: float) -> float:
        """Apply bw_processing's float32 coefficient serialization to the expected matrix values."""
        return float(np.float32(amount))

    catalogue = load_provider_catalogue()
    clone_id = "phb_residue_fossil_incineration_operation"
    origins = ("fossil", "biogenic", "atmospheric_dac")
    services = {
        str(service["provider_service_id"]): service
        for service in cast(list[dict[str, Any]], catalogue["services"])
        if service["provider_service_id"] in {clone_id, *(f"phb_{origin}" for origin in origins)}
    }
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="phb_fermentation_and_residue",
        biosphere_database="biosphere",
        source_database="native",
    )
    with selected_brightway_project(names) as bd:
        flows = {
            **{
                f"{species}_{origin}": (
                    species + ", " + origin,
                    "kilogram",
                    ("air", "urban air close to ground"),
                )
                for species in ("Carbon dioxide", "Carbon monoxide", "Methane")
                for origin in ("fossil", "non-fossil")
            },
            "fermentation_fossil": ("Carbon dioxide, fossil", "kilogram", ("air",)),
            "fermentation_non_fossil": ("Carbon dioxide, non-fossil", "kilogram", ("air",)),
            "cod": ("COD, Chemical Oxygen Demand", "kilogram", ("water",)),
            "water": ("Water", "cubic meter", ("water",)),
            "neutral": ("Nitrogen oxides", "kilogram", ("air",)),
        }
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": name,
                    "unit": unit,
                    "categories": categories,
                    "type": "emission",
                }
                for code, (name, unit, categories) in flows.items()
            },
            searchable=False,
        )
        native = {}
        components = services["phb_biogenic"]["construction"]["components"]
        for component in components:
            if "selector" not in component:
                continue
            selector = component["selector"]
            code = component["component_id"]
            native[("native", code)] = {
                "name": selector["name"],
                "reference product": selector["reference_product"],
                "unit": selector["unit"],
                "location": selector["exact_location"],
                "type": "process",
                "exchanges": [{"input": ("native", code), "amount": 1.0, "type": "production"}],
            }
        for origin in origins:
            code = f"glucose_delivered_{origin}"
            selector = {
                "name": f"bounded delivered {origin} glucose",
                "reference_product": "glucose",
                "unit": "kilogram",
                "exact_location": "GLO",
            }
            native[("native", code)] = {
                "name": selector["name"],
                "reference product": "glucose",
                "unit": "kilogram",
                "location": "GLO",
                "type": "process",
                "exchanges": [{"input": ("native", code), "amount": 1.0, "type": "production"}],
            }
            # Bound the recipe at delivered glucose. Its complete upstream
            # manufacture and delivery are covered by their own supplier tests.
            services[code] = {
                "provider_service_id": code,
                "name": selector["name"],
                "reference_product": "glucose",
                "unit": "kilogram",
                "location": "GLO",
                "stage_owner": "virgin_resin_production",
                "foreground_service": False,
                "citation": "ecoinvent_3_12_bounded_glucose_supply",
                "construction": {
                    "kind": "native_activity",
                    "source_role": "native_ecoinvent_provider",
                    "selector": selector,
                },
            }
        residue_key = ("native", "residual_bacterial_biomass_waste_to_energy_analogue")
        residue = native[residue_key]
        residue["exchanges"] = [
            {
                "input": residue_key,
                "amount": -1.0,
                "type": "production",
                "allocation": "native residue reference",
            },
            {
                "input": ("native", "electricity"),
                "amount": 0.25,
                "type": "technosphere",
                "allocation": "native utility",
            },
            {
                "input": ("native", "ash"),
                "amount": -0.1,
                "type": "technosphere",
                "allocation": "native child treatment",
            },
            *[
                {
                    "input": ("biosphere", species + "_non-fossil"),
                    "amount": amount,
                    "type": "biosphere",
                    "name": species + ", non-fossil",
                    "unit": "kilogram",
                    "uncertainty type": 2,
                    "loc": -1.0,
                    "scale": 0.2,
                    "allocation": "native direct residue carbon",
                }
                for species, amount in (
                    ("Carbon dioxide", 1.1),
                    ("Carbon monoxide", 0.028),
                    ("Methane", 0.016),
                )
            ],
            {"input": ("biosphere", "neutral"), "amount": 0.004, "type": "biosphere"},
            # Fuel carbon belongs to another input and retains its native identity.
            {"input": ("biosphere", "Carbon dioxide_fossil"), "amount": 0.03, "type": "biosphere"},
        ]
        native[("native", "electricity")]["exchanges"].append(
            {"input": ("biosphere", "fermentation_fossil"), "amount": 0.07, "type": "biosphere"}
        )
        native[("native", "ash")] = {
            "name": "bounded ash treatment",
            "reference product": "ash",
            "unit": "kilogram",
            "location": "GLO",
            "type": "process",
            "exchanges": [
                {"input": ("native", "ash"), "amount": -1.0, "type": "production"},
                {"input": ("biosphere", "neutral"), "amount": 0.04, "type": "biosphere"},
            ],
        }
        bd.Database("native").write(native, searchable=False)
        native_before = {
            key: [
                dict(exchange) for exchange in bd.get_node(database=key[0], code=key[1]).exchanges()
            ]
            for key in (residue_key, ("native", "ash"), ("native", "electricity"))
        }
        fixture_catalogue = {**catalogue, "services": list(services.values()), "routes": []}
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=2030,
            ordinary_database="native",
            technology_database="native",
            material_recipe=physical_material_recipe(),
            provider=PreparedProviderAuthority(
                identity="bounded_phb_origin_authority",
                interface_database="bounded_interface",
                catalogue=fixture_catalogue,
                routes=pd.DataFrame(),
                method_authorities=_method_authority_rows(fixture_catalogue),
            ),
            resolver_cache=_NativeResolverCache("native", "biosphere"),
        )
        clone = bd.get_node(database=frontier.database_name, code=provider_service_code(clone_id))
        expected_clone = []
        for exchange in native_before[residue_key]:
            expected = {key: value for key, value in exchange.items() if key != "output"}
            if exchange["type"] == "production":
                expected["input"] = clone.key
            elif exchange["input"][1].endswith("_non-fossil"):
                expected["input"] = (
                    "biosphere",
                    exchange["input"][1].replace("_non-fossil", "_fossil"),
                )
                expected["name"] = exchange["name"].replace(", non-fossil", ", fossil")
            expected_clone.append(expected)
        actual_clone = [
            {key: value for key, value in dict(exchange).items() if key != "output"}
            for exchange in clone.exchanges()
        ]
        assert sorted(actual_clone, key=repr) == sorted(expected_clone, key=repr)
        assert (
            json.loads(clone["direct_treatment_carbon_owners_json"])[0]["origin_mapping_status"]
            == "exact_fossil_counterpart"
        )
        carbon_rows = [
            row
            for row in frontier.semantic_rows
            if row.get("ledger_entry_type") == "treatment_carbon_mapping"
        ]
        assert len(carbon_rows) == 3
        assert {row["ultimate_carbon_origin"] for row in carbon_rows} == {"fossil"}
        assert {row["provider_stage_owner"] for row in carbon_rows} == {"virgin_resin_production"}

        for origin in origins:
            parent = bd.get_node(
                database=frontier.database_name, code=provider_service_code(f"phb_{origin}")
            )
            demands = {exchange.input.key: exchange["amount"] for exchange in parent.technosphere()}
            fossil_fraction = 713 / 738 if origin == "fossil" else 0.0
            assert demands[residue_key] == pytest.approx(
                -0.42 * (1 - fossil_fraction), rel=1e-14, abs=0
            )
            if origin == "fossil":
                assert demands[clone.key] == pytest.approx(
                    -0.42 * fossil_fraction, rel=1e-14, abs=0
                )
            else:
                assert clone.key not in demands
            direct_expected = {
                "fermentation_non_fossil": 1.9 * (1 - fossil_fraction),
                "cod": 0.9,
                "water": 0.0331,
            }
            if origin == "fossil":
                direct_expected["fermentation_fossil"] = 1.9 * fossil_fraction
            assert {
                exchange.input.key[1]: exchange["amount"] for exchange in parent.biosphere()
            } == pytest.approx(direct_expected, rel=1e-14, abs=0)
            calculation = LCA({parent: 1.0})
            calculation.lci()
            inventory = {
                bd.get_node(id=identifier).key[1]: float(calculation.inventory[index, :].sum())
                for identifier, index in calculation.dicts.biosphere.items()
            }
            # Record quantities above retain Python floats. The installed matrix
            # serializer first rounds each coefficient to float32; apply only
            # that documented storage conversion to the independent equations.
            fossil_residue = stored_amount(0.42 * fossil_fraction)
            non_fossil_residue = stored_amount(0.42 * (1 - fossil_fraction))
            total_residue = fossil_residue + non_fossil_residue
            electricity = stored_amount(3.26) + total_residue * stored_amount(0.25)
            # The two fossil CO2 exchanges occupy one matrix coordinate;
            # matrix_utils sums their serialized values at float32 precision.
            clone_fossil_co2 = stored_amount(stored_amount(1.1) + stored_amount(0.03))
            expected = {
                "fermentation_fossil": stored_amount(1.9 * fossil_fraction)
                + electricity * stored_amount(0.07),
                "fermentation_non_fossil": stored_amount(1.9 * (1 - fossil_fraction)),
                "Carbon dioxide_fossil": fossil_residue * clone_fossil_co2
                + non_fossil_residue * stored_amount(0.03),
                "Carbon dioxide_non-fossil": non_fossil_residue * stored_amount(1.1),
                "Carbon monoxide_fossil": fossil_residue * stored_amount(0.028),
                "Carbon monoxide_non-fossil": non_fossil_residue * stored_amount(0.028),
                "Methane_fossil": fossil_residue * stored_amount(0.016),
                "Methane_non-fossil": non_fossil_residue * stored_amount(0.016),
                "neutral": total_residue
                * (stored_amount(0.004) + stored_amount(0.1) * stored_amount(0.04)),
                "cod": stored_amount(0.9),
                "water": stored_amount(0.0331),
            }
            assert inventory == pytest.approx(expected, rel=1e-13, abs=1e-15)
        assert native_before == {
            key: [
                dict(exchange) for exchange in bd.get_node(database=key[0], code=key[1]).exchanges()
            ]
            for key in native_before
        }
