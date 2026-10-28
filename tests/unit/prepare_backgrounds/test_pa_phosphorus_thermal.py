"""Solve actual PA ester/phosphorus residues with complete fuel and ash support."""

from math import fsum, isfinite
from pathlib import Path
from typing import Any, cast

import pytest
from bw2calc import LCA
from chemical_residue_fixtures import chemical_residue_biosphere, chemical_residue_native_inventory
from material_recipe_fixtures import physical_material_recipe

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
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
from pytoce.shared.runtime.brightway import write_brightway_database


@pytest.mark.parametrize(
    "service_id,wet,biogenic",
    [
        ("pa_diglycerol_bottoms", 0.39607774287884906, True),
        ("pa_dioleate_settling", 0.059309904130733435, True),
        ("pa_dimer_monomer", 0.4745777413812858, True),
        ("pa_dimer_mineral_residue", 0.1199733187668906, False),
        ("pa_phosphate_solvent_residue", 0.705062876594644, False),
    ],
)
def test_named_residues_keep_actual_atoms_heat_and_complete_treatment(
    tmp_path: Path, service_id: str, wet: float, biogenic: bool
) -> None:
    """Admit real parent quantities and solve the whole signed receiving graph.

    The five named residues are checked against expected wet masses.
    Synthetic suppliers exercise signed-reference normalization and the retained
    catalyst and PAM supply paths.
    """
    provider = prepared_provider_authority()
    services = {
        str(s["provider_service_id"]): s
        for s in cast(list[dict[str, Any]], provider.catalogue["services"])
    }
    ash_id = services[service_id]["construction"]["ash"]["service_id"]
    names = BrightwayNames(tmp_path / "brightway", "pa_thermal", "biosphere", "source")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(chemical_residue_biosphere(), searchable=False)
        bd.Database("source").write(
            chemical_residue_native_inventory(provider.catalogue, "source"), searchable=False
        )
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
        hkey, akey = (("receiving", provider_service_code(s)) for s in (service_id, ash_id))
        for sid, key in ((service_id, hkey), (ash_id, akey)):
            records[key] = _service_dataset(context, services[sid], services, "receiving")
        ref = cast(dict[str, Any], records[hkey]["chemical_residue_reference"])
        load = ref["load"]
        elements = load["active_elements_kg"]
        assert load["wet_mass_kg"] == pytest.approx(wet, rel=3e-14)
        assert fsum(elements.values()) + load["carrier_water_kg"] + load[
            "unknown_mineral_kg"
        ] + fsum(
            fsum(s.values()) for s in load["nonfuel_salt_elements_kg"].values()
        ) == pytest.approx(wet)
        assert (load["biogenic_carbon_kg"] > 0) is biogenic
        assert ref["ash_mass_kg"] > 0
        assert ref["fuel_ncv_deficit_mj"] == pytest.approx(max(23 * wet - load["net_energy_mj"], 0))
        assert ref["supplies"]["CaCl2"] == pytest.approx(
            0.008 * (elements.get("P", 0) / 0.005 + load["unknown_mineral_kg"])
        )
        rows = cast(list[dict[str, Any]], records[hkey]["exchanges"])
        direct: dict[str, float] = {}
        for row in rows:
            if row["type"] == "biosphere":
                code = row["input"][1]
                direct[code] = direct.get(code, 0.0) + row["amount"] * wet
        if service_id == "pa_phosphate_solvent_residue":
            # Original workup charges equal masses of two one-pass solvents.
            # Molecular C/H/O and Michel (via Muñoz et al. (2007), Eq. 9) NCV are independent of the
            # receiver.
            hexane = 0.3525314382973219 / (6 * 12.011 + 14 * 1.008)
            mibk = 0.3525314382973219 / (6 * 12.011 + 12 * 1.008 + 15.999)
            carbon, hydrogen, oxygen = (
                6 * 12.011 * (hexane + mibk),
                1.008 * (14 * hexane + 12 * mibk),
                15.999 * mibk,
            )
            assert elements["C"] == pytest.approx(carbon)
            assert load["net_energy_mj"] == pytest.approx(
                34.016 * carbon + (124.265 - 9 * 2.4417) * hydrogen - 9.8324 * oxygen
            )
            assert load["waste_formed_water_kg"] == pytest.approx((7 * hexane + 6 * mibk) * 18.015)
            assert load["carrier_water_kg"] == 0
            assert load["biogenic_carbon_kg"] == 0
        ash_rows = cast(list[dict[str, Any]], records[akey]["exchanges"])
        terminal = {
            r["input"][1]: r["amount"] * ref["ash_mass_kg"]
            for r in ash_rows
            if r["type"] == "biosphere"
        }
        if service_id == "pa_diglycerol_bottoms":
            atoms = load["nonfuel_salt_elements_kg"]["Potassium sulfate"]
            sulfate = atoms["S"] + atoms["O"]
            assert direct["cr_potassium_air"] == pytest.approx(0.00301 * atoms["K"])
            assert direct["cr_sulfate_air"] == pytest.approx(0.00301 * sulfate)
            for suffix, fraction in (("surface", 0.2819), ("long", 0.7181)):
                assert terminal[f"cr_Potassium I_{suffix}"] == pytest.approx(
                    atoms["K"] * 0.99699 * fraction
                )
                assert terminal[f"cr_Sulfate_{suffix}"] == pytest.approx(
                    sulfate * 0.99699 * fraction
                )
        if service_id == "pa_dimer_mineral_residue":
            # Both oxidized phosphates release water by dehydration. The NBS crystalline LiH2PO4 and
            # H3PO4 references require heat.
            lithium_phosphate = 0.0118395967896022 / (6.94 + 2 * 1.008 + 30.973761998 + 4 * 15.999)
            phosphoric_acid = 0.00189797069560610 / (3 * 1.008 + 30.973761998 + 4 * 15.999)
            assert load["net_energy_mj"] == pytest.approx(
                -76.5764 * lithium_phosphate - 163.8248 * phosphoric_acid
            )
            assert load["waste_formed_water_kg"] == pytest.approx(
                (lithium_phosphate + 1.5 * phosphoric_acid) * 18.015
            )
            assert ref["fuel_ncv_deficit_mj"] > 23 * wet
            assert direct["cr_lithium_surface"] == pytest.approx(elements["Li"])
            assert "cr_lithium_surface" not in terminal
            assert elements["C"] == 0
            assert ref["gases"]["Carbon dioxide, fossil"] == 0
        write_brightway_database(bd, "receiving", records, {})
        hwi, ash = (bd.get_node(database=k[0], code=k[1]) for k in (hkey, akey))
        assert next(iter(hwi.production()))["amount"] == -1
        assert next(iter(ash.production()))["amount"] == -1
        lca = LCA({hwi: -wet})
        lca.lci()
        supply = cast(Any, lca.supply_array)
        assert all(isfinite(value) for value in supply)
        assert supply[lca.dicts.activity[ash.id]] == pytest.approx(ref["ash_mass_kg"])
