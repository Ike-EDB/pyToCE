"""Follow the PA phosphate wash through the complete Swiss treatment chain."""

import json
from math import fsum, isfinite
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from bw2calc import LCA
from numpy.typing import NDArray
from pa_phosphate_water_fixtures import phosphate_water_boundary, phosphate_water_inputs

from pytoce.prepare_backgrounds.preparation.providers.catalogue import provider_service_code
from pytoce.prepare_backgrounds.preparation.providers.pa_phosphate_water import (
    pa_phosphate_water_datasets,
)
from pytoce.shared.runtime.brightway import write_brightway_database


def _emissions(
    bd: Any, dataset: dict[str, object], scale: float
) -> dict[tuple[str, str, tuple[str, ...]], float]:
    """Group compiled elementary rows by their flow identity and compartment."""
    result: dict[tuple[str, str, tuple[str, ...]], float] = {}
    for row in cast(list[dict[str, Any]], dataset["exchanges"]):
        if row["type"] == "biosphere":
            node = bd.get_node(database=row["input"][0], code=row["input"][1])
            key = node["name"], node["unit"], tuple(node["categories"])
            result[key] = result.get(key, 0.0) + row["amount"] * scale
    return result


@pytest.mark.parametrize("parent_amount", (1.0, 2.0))
def test_actual_mixed_wash_keeps_hydraulics_ions_carbon_and_complete_sludge_chain(
    tmp_path: Path, parent_amount: float
) -> None:
    """Scale the compound inputs and solve their synthetic signed treatment graph.

    Both parent amounts use the same composition; only their actual service demand changes. The
    source fixtures have -5/-7 m3 collection/plant
    productions and an independently specified 80% treatment fraction.
    """
    inputs = {name: amount * parent_amount for name, amount in phosphate_water_inputs().items()}
    wet = 3.084670670507235 * parent_amount
    volume = wet / 1000
    carbon = 0.0482705082014361 * parent_amount
    phosphorus = 0.0105552507044127 * parent_amount
    sodium = 0.162092328608040 * parent_amount
    chloride = 0.249944789752801 * parent_amount
    with phosphate_water_boundary(tmp_path) as (bd, context):
        records = pa_phosphate_water_datasets(context, **inputs)
        collection_key = "providers", provider_service_code("pa_phosphate_water_collection_ch")
        plant_key = "providers", provider_service_code("pa_phosphate_water_treatment_ch")
        collection, plant = records[collection_key], records[plant_key]
        assert len(records) == 10
        raw_terms = json.loads(
            cast(
                str,
                next(
                    row["project_terms_json"]
                    for row in context.technology_rows
                    if row["provider_service_id"] == "pa_phosphate_water_collection_ch"
                    and row["component_id"] == "leachate_collection"
                ),
            )
        )
        plant_terms = json.loads(
            cast(
                str,
                next(
                    row["project_terms_json"]
                    for row in context.technology_rows
                    if row["provider_service_id"] == "pa_phosphate_water_treatment_ch"
                    and row["component_id"] == "actual_wastewater_treatment"
                ),
            )
        )
        raw = raw_terms["raw_load_per_caller_reference"]
        water = plant_terms["load_per_caller_reference"]
        assert water["volume_m3"] == pytest.approx(volume, rel=2e-14)
        assert water["water"]["incoming_kg"] == inputs["water_kg"]
        assert fsum(raw["elements_kg"].values()) + inputs["water_kg"] == pytest.approx(wet)
        for element, amount in (("C", carbon), ("P", phosphorus), ("Na", sodium), ("Cl", chloride)):
            assert raw["elements_kg"][element] == pytest.approx(amount, rel=2e-13)
        assert raw["elements_kg"].get("N", 0) == 0
        assert raw["chloride_kg"] == raw["elements_kg"]["Cl"]
        assert raw["carbon"] == [
            {
                "component": "pa_monooleylphosphate_workup",
                "principal": False,
                "origin": "biogenic",
                "carbon_kg": raw["elements_kg"]["C"],
            }
        ]
        # Molecular H/O include H3PO4. COD oxidizes the monoester to CO2, H2O, and phosphoric acid.
        monoester_kmol = inputs["monooleylphosphate_kg"] / (
            18 * 12.011 + 37 * 1.008 + 4 * 15.999 + 30.973761998
        )
        acid_kmol = inputs["phosphoric_acid_kg"] / (3 * 1.008 + 30.973761998 + 4 * 15.999)
        assert raw["elements_kg"]["H"] == pytest.approx(
            (37 * monoester_kmol + 3 * acid_kmol) * 1.008
        )
        assert raw["elements_kg"]["O"] == pytest.approx(4 * (monoester_kmol + acid_kmol) * 15.999)
        assert raw["organic_hydrogen_kg"] == pytest.approx(0.1225 * carbon)
        assert raw["organic_oxygen_kg"] == pytest.approx(0.64 * carbon)
        assert raw["toc_kg"] == pytest.approx(carbon)
        assert raw["doc_kg"] == pytest.approx(0.68 * carbon)
        assert raw["cod_kg_o2"] == pytest.approx(26.5 * monoester_kmol * 31.998)
        assert raw["bod5_kg_o2"] == pytest.approx(1.7586552173908316 * carbon)

        collection_rows = cast(list[dict[str, Any]], collection["exchanges"])
        assert next(row for row in collection_rows if row["type"] == "production")["amount"] == -5
        assert next(row for row in collection_rows if row["input"] == plant_key)["amount"] == -4
        plant_rows = cast(list[dict[str, Any]], plant["exchanges"])
        assert next(row for row in plant_rows if row["type"] == "production")["amount"] == -7
        assert raw_terms["treated_fraction"] == 0.8
        bypass = _emissions(bd, collection, volume / 5)
        treated = _emissions(bd, plant, volume / 7)
        surface = "water", "surface water"
        assert not any(key[0] == "Chlorine" for key in bypass)
        assert bypass["Chloride", "kilogram", surface] == pytest.approx(0.2 * chloride)
        assert treated["Chloride", "kilogram", surface] == pytest.approx(chloride)
        assert bypass["Sodium I", "kilogram", ("water",)] == pytest.approx(0.2 * sodium)
        assert bypass["Phosphorus", "kilogram", ("water",)] == pytest.approx(0.2 * phosphorus)
        assert bypass["TOC, Total Organic Carbon", "kilogram", ("water",)] == pytest.approx(
            0.2 * carbon
        )
        assert bypass["Water", "cubic meter", ("water",)] == pytest.approx(
            0.2 * inputs["water_kg"] / 1000
        )
        assert treated["Sodium I", "kilogram", surface] == pytest.approx(
            (1 - 0.00796981786325856) * sodium
        )
        assert treated["Phosphate", "kilogram", surface] == pytest.approx(
            (1 - 0.5 - 0.8878 * 0.42) * phosphorus * 94.97 / 30.97
        )
        sludge = water["final_sludge"]["elements_kg"]
        assert sludge["P"] == pytest.approx((0.5 + 0.8878 * 0.42) * phosphorus)
        assert sludge["Na"] == pytest.approx(0.00796981786325856 * sodium)
        assert sludge["Cl"] == 0
        assert water["duties"]["iron_sulfate_kg"] == pytest.approx(
            30.6456139178772 * 0.8878 * 0.42 * phosphorus
        )
        assert water["duties"]["polyacrylamide_kg"] > 0
        assert water["duties"]["electricity_kwh"] > 0
        assert water["duties"]["heat_mj"] > 0
        assert water["raw_sludge"]["elements_kg"]["N"] > 0  # Fresh PAM owns this N.
        organic, pam = water["carbon"]
        assert organic["origin"] == "biogenic" and pam["origin"] == "fossil"
        assert organic["input_carbon_kg"] == pytest.approx(carbon)
        assert fsum(
            organic[phase]
            for phase in (
                "biological_air_carbon_kg",
                "digester_gas_carbon_kg",
                "final_sludge_carbon_kg",
                "effluent_carbon_kg",
            )
        ) == pytest.approx(carbon)
        assert pam["input_carbon_kg"] == pytest.approx(
            water["duties"]["polyacrylamide_elements_kg"]["C"]
        )

        # The actual plant retains positive thermal and agricultural sludge
        # outlets. Its children carry the same P/Na and the distinct C origins.
        for fraction, share in (("incineration", 0.9), ("landfarming", 0.1)):
            assert water[f"{fraction}_dry_share"] == pytest.approx(share)
        for service_id, component, share, element_key, carbon_key in (
            (
                "pa_phosphate_sludge_incineration_ch",
                "Swiss_sludge_incineration",
                0.9,
                "elements_kg",
                "carbon",
            ),
            (
                "pa_phosphate_sludge_landfarming_ch",
                "actual_sludge_landfarming",
                0.1,
                "actual_elements_kg",
                "carbon_components",
            ),
        ):
            child = json.loads(
                cast(
                    str,
                    next(
                        row["project_terms_json"]
                        for row in context.technology_rows
                        if row["provider_service_id"] == service_id
                        and row["component_id"] == component
                    ),
                )
            )
            for element in ("P", "Na"):
                assert child[element_key][element] == pytest.approx(share * sludge[element])
            biogenic = next(pool for pool in child[carbon_key] if pool["origin"] == "biogenic")
            assert biogenic["carbon_kg"] == pytest.approx(share * organic["final_sludge_carbon_kg"])
            if component == "Swiss_sludge_incineration":
                # Retained expected active CaO for this phosphate-wash case.
                # The linked market supply below uses the selected 95% grade.
                lime = 5.808074555971866e-6 * parent_amount
                calcium = lime * 40.08 / 56.08
                assert child["duties"]["CaO"] == pytest.approx(lime, rel=2e-12)
                assert child["duties"]["FeCl3"] == 0
                assert child["aqueous_elements_kg"]["Ca"] == pytest.approx(calcium / 2)
                ash = child["residual_ash"]
                assert ash["auxiliary_elements_kg"]["Ca"] == pytest.approx(calcium / 2)
                assert ash["auxiliary_elements_kg"]["O"] == pytest.approx(
                    child["fresh_elements_kg"]["PAM"]["O"]
                    + child["fresh_elements_kg"]["SCR"]["O"]
                    + lime * 16 / 56.08 / 2
                )
                assert (
                    fsum(ash["waste_elements_kg"].values())
                    + fsum(ash["auxiliary_elements_kg"].values())
                    + ash["allocated_hydrogen_kg"]
                    + ash["allocated_oxygen_kg"]
                ) == pytest.approx(ash["treatment_mass_kg"])
                furnace = records["providers", provider_service_code(service_id)]
                furnace_emissions = _emissions(bd, furnace, child["wet_mass_kg"] / 2)
                # Incoming phosphate wash and wastewater treatment auxiliaries contain no
                # Ca; this existing surface-water flow is fresh scrubber Ca only.
                assert furnace_emissions["Calcium II", "kilogram", surface] == pytest.approx(
                    calcium / 2
                )
        write_brightway_database(bd, "providers", records, {})
        node = bd.get_node(database=collection_key[0], code=collection_key[1])
        lca = LCA({node: -volume})
        lca.lci()
        supply_array = cast(NDArray[np.float64] | None, lca.supply_array)
        assert supply_array is not None
        assert all(isfinite(value) for value in supply_array)
        for key in records:
            receiving = bd.get_node(database=key[0], code=key[1])
            assert supply_array[lca.dicts.activity[receiving.id]] > 0
        actual_plant = bd.get_node(database=plant_key[0], code=plant_key[1])
        assert supply_array[lca.dicts.activity[actual_plant.id]] == pytest.approx(
            0.8 * volume / 7, rel=2e-7
        )
        quicklime = bd.get_node(database="source", code="organic_ch_packed_lime")
        # Collection treats 80%; the complete selected market supplies its own
        # full delivery/manufacture once, with no nonexistent donor lime row.
        assert supply_array[lca.dicts.activity[quicklime.id]] == pytest.approx(
            0.8 * 5.808074555971866e-6 * parent_amount / 0.95, rel=2e-7
        )
        for code in ("mibk_water_ch", "mibk_plant_ch", "mibk_rer", "mibk_row"):
            original = bd.get_node(database="source", code=code)
            assert supply_array[lca.dicts.activity[original.id]] == 0
        # Signed synthetic source exchanges retain ordered uncertainty bounds.
        for code in ("mibk_rer", "mibk_row", "mibk_doa"):
            original = bd.get_node(database="source", code=code)
            signed_rows = [ex for ex in original.exchanges() if "minimum" in ex]
            assert any(ex["amount"] < 0 for ex in signed_rows)
            assert all(ex["minimum"] <= ex["amount"] <= ex["maximum"] for ex in signed_rows)
