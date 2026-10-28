"""Receive known carbon-free PA phosphorus/mineral inputs through complete HWI."""

from math import fsum, isfinite
from pathlib import Path
from typing import Any, cast

import pytest
from bw2calc import LCA
from chemical_residue_fixtures import chemical_residue_biosphere, chemical_residue_native_inventory
from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.chemical_residue_incineration import (
    chemical_residue_datasets,
)
from pytoce.prepare_backgrounds.preparation.chemical_residue_load import ChemicalResidueLoad
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.shared.runtime.brightway import write_brightway_database


def test_fully_oxidized_phosphorus_calcium_load_keeps_complete_receiving(tmp_path: Path) -> None:
    """Solve a dry CaO/P2O5 mixture with zero waste carbon and incoming fuel energy.

    This known-oxide reference tests the receiving calculation.
    Full support fuel, fixed operation, native catalysts, native ash support
    and transport remain necessary even for carbon-free input.
    """
    provider = prepared_provider_authority()
    # Invented 0.4 kg CaO + 0.6 kg P2O5, using the receiving source's nominal
    # P31/O16 and Ca40.078 mass convention. Already oxidized solids have zero
    # incoming combustion energy and contain no C/H/N/S/Na or hidden moisture.
    calcium = 0.4 * 40.078 / (40.078 + 16)
    phosphorus = 0.6 * 62 / (62 + 80)
    elements = {
        "C": 0.0,
        "H": 0.0,
        "O": 1 - calcium - phosphorus,
        "N": 0.0,
        "S": 0.0,
        "Na": 0.0,
        "Ca": calcium,
        "P": phosphorus,
    }
    load = ChemicalResidueLoad(1.0, 0.0, elements, 0.0, {}, 0.0, 0.0, waste_formed_water_kg=0.0)
    names = BrightwayNames(tmp_path / "brightway", "pa_inorganic", "biosphere", "source")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(chemical_residue_biosphere(), searchable=False)
        native = chemical_residue_native_inventory(provider.catalogue, "source")
        bd.Database("source").write(native, searchable=False)
        before = {
            node.key: [dict(row) for row in node.exchanges()] for node in bd.Database("source")
        }
        native_resolver = _ActivityResolver(bd, "source")
        context = _RecipeContext(
            background_scenario=None,
            year=2025,
            provider_database="receiving",
            source=native_resolver,
            ordinary=native_resolver,
            technology=native_resolver,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=provider.catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(provider.catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        records = {}
        hkey, akey = ("receiving", "pa_oxide_receiving"), ("receiving", "pa_oxide_ash")
        actual = chemical_residue_datasets(
            context,
            load,
            incineration_key=hkey,
            ash_key=akey,
            residue_name="invented fully oxidized calcium/phosphorus residue",
        )
        q = cast(dict[str, Any], actual[hkey]["chemical_residue_reference"])
        ash_mass = q["ash_mass_kg"]
        assert q["ash"]["P"] == pytest.approx(0.984 * phosphorus)
        assert q["ash"]["Ca"] == pytest.approx(
            0.998 * calcium + q["fresh"]["CaO"]["Ca"] + q["fresh"]["CaCl2"]["Ca"]
        )
        # Oxygen metadata describes the oxide-equivalent body used for terminal demand.
        assert fsum(q["ash"].values()) == pytest.approx(ash_mass, rel=2e-14)
        expected_oxide = (
            0.984 * phosphorus * 142 / 62
            + 0.998 * calcium * (40.078 + 16) / 40.078
            + fsum(q["supplies"][name] for name in ("PAM", "CaO", "TiO2", "V2O5"))
            + q["fresh"]["CaCl2"]["Ca"] * (40.078 + 16) / 40.078
        )
        assert ash_mass == pytest.approx(expected_oxide, rel=2e-14)
        assert q["supplies"]["CaCl2"] == pytest.approx(0.008 * phosphorus / 0.005)
        assert q["nitrogen_ratio"] == 0
        assert q["fuel_ncv_deficit_mj"] == 23
        assert q["oil_kg"] == pytest.approx(23 / 42.6)
        assert q["supplies"]["NH3_fixed"] == 0.002
        assert q["supplies"]["NG_fixed"] == 0.3
        assert q["supplies"]["PAM"] > 0
        for name in (
            "Carbon monoxide, fossil",
            "Carbon monoxide, non-fossil",
            "Carbon dioxide, fossil",
            "Carbon dioxide, non-fossil",
            "Cyanide",
        ):
            assert q["gases"][name] == 0
        rows = cast(list[dict[str, Any]], actual[hkey]["exchanges"])
        by_code = {row["input"][1]: row["amount"] for row in rows}
        assert by_code["cr_phosphorus_air"] == pytest.approx(0.0007 * phosphorus)
        assert by_code["cr_Phosphate_surface"] == pytest.approx(0.0153 * phosphorus * 95 / 31)
        assert "cr_Phosphate_long" not in by_code
        assert by_code["cr_calcium_air"] == pytest.approx(0.00167 * calcium)
        assert by_code["mc_nh3_air"] == pytest.approx(0.00002)
        assert by_code["mc_nox_air"] == pytest.approx(0.001)
        assert by_code["mc_pm_urban"] == pytest.approx(0.0001)
        assert by_code["mc_oil"] == pytest.approx(23 / (0.03125 * 42.6))
        assert by_code["cr_road"] == 0.04
        assert by_code["cr_rail"] == 0.007
        assert by_code["mc_tio2"] == pytest.approx(0.0002)
        assert by_code["mc_v_proxy"] == pytest.approx(0.00005)
        assert by_code[akey[1]] == pytest.approx(-ash_mass)
        arows = cast(list[dict[str, Any]], actual[akey]["exchanges"])
        terminal = {row["input"][1]: row["amount"] * ash_mass for row in arows}
        assert terminal["cr_Phosphate_surface"] == pytest.approx(q["ash"]["P"] * 0.004 / 0.025)
        assert terminal["cr_Phosphate_long"] == pytest.approx(q["ash"]["P"] * 0.006 / 0.025)
        assert terminal["mc_ca_short"] == pytest.approx(q["ash"]["Ca"] * 0.08 / 0.2)
        assert terminal["mc_ca_long"] == pytest.approx(q["ash"]["Ca"] * 0.1 / 0.2)
        assert terminal["mc_tap_a"] == pytest.approx(0.6 * ash_mass)
        assert terminal["mc_cement_a"] + terminal["mc_cement_b"] == pytest.approx(0.4 * ash_mass)
        assert terminal["mc_hydrated_cement"] == pytest.approx(-ash_mass)
        records.update(actual)
        write_brightway_database(bd, "receiving", records, {})
        hwi = bd.get_node(database="receiving", code=hkey[1])
        ash = bd.get_node(database="receiving", code=akey[1])
        assert next(iter(hwi.production()))["amount"] == -1
        assert next(iter(ash.production()))["amount"] == -1
        lca = LCA({hwi: -1})
        lca.lci()
        supply = cast(Any, lca.supply_array)
        assert all(isfinite(value) for value in supply)
        assert supply[lca.dicts.activity[ash.id]] == pytest.approx(ash_mass)
        boiler = bd.get_node(database="source", code="mc_oil")
        co2 = bd.get_node(database="biosphere", code="co2")
        # The boiler retains its fuel-combustion emissions when waste carbon is zero.
        assert lca.inventory[lca.dicts.biosphere[co2.id], lca.dicts.activity[boiler.id]] > 0
        for key, exchanges in before.items():
            assert [
                dict(row) for row in bd.get_node(database=key[0], code=key[1]).exchanges()
            ] == exchanges
