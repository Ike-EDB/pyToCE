"""Compile PA spinning fluid recovery with real native and thermal receiving owners."""

import json
from math import fsum, isfinite, log
from pathlib import Path
from typing import Any, cast

import pytest
from bw2calc import LCA
from chemical_residue_fixtures import (
    apply_premise_chemical_receiving_metadata,
    chemical_residue_biosphere,
    chemical_residue_native_inventory,
)
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


def _spinning_native_inventory(
    catalogue: dict[str, Any], database: str, *, original: bool
) -> dict[tuple[str, str], dict[str, Any]]:
    """Add two invented nonunit native suppliers to complete thermal fixtures.

    Each dated supply retains a distinct native biosphere burden and source
    uncertainty. No fake package receiving activity or licensed row is used.
    """
    data = chemical_residue_native_inventory(catalogue, database)
    service = next(
        s for s in catalogue["services"] if s["provider_service_id"] == "pa6_spinning_support"
    )
    selectors = service["construction"]["supplies"]
    for label, burden in (("factory", 0.03), ("fluid", 0.07)):
        selector = selectors[label]
        key = database, "spinning_" + label
        basis = 2.0 if original else 3.0
        amount = burden * basis * (1.0 if original else 0.6)
        data[key] = {
            "name": selector["name"],
            "reference product": selector["reference_product"],
            "unit": selector["unit"],
            "location": selector["exact_location"],
            "exchanges": [
                {
                    "input": ("biosphere", "mc_co2_fossil"),
                    "type": "biosphere",
                    "amount": amount,
                    "uncertainty type": 2,
                    "loc": log(amount) + 0.02,
                    "scale": 0.13,
                    "allocation": "invented dated spinning supplier",
                },
                {"input": key, "type": "production", "amount": basis},
            ],
        }
    return data


@pytest.mark.parametrize("original", [True, False])
def test_spinning_recovery_closes_external_fluid_and_connected_receiving(
    tmp_path: Path, original: bool
) -> None:
    """Preserve both PA references, dated supplies and the real signed receiver.

    Independent equations check external fluid closure, molecular carbon,
    formed water and thermal energy. The small Brightway solve checks retired-fluid
    routing through the selected HWI/ash graph. Reported air belongs to the parent
    converter; this test checks that the support service adds no direct biosphere
    rows. Internally recovered fluid has no external demand.
    """
    provider = prepared_provider_authority()
    catalogue = provider.catalogue
    services = {
        str(s["provider_service_id"]): s for s in cast(list[dict[str, Any]], catalogue["services"])
    }
    names = BrightwayNames(tmp_path / "brightway", "pa_spinning", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(chemical_residue_biosphere(), searchable=False)
        bd.Database("original").write(
            _spinning_native_inventory(catalogue, "original", original=True), searchable=False
        )
        database = "original" if original else "future"
        if not original:
            bd.Database(database).write(
                _spinning_native_inventory(catalogue, database, original=False), searchable=False
            )
        native_before = {
            (db, "spinning_" + label): [
                dict(row) for row in bd.get_node(database=db, code="spinning_" + label).exchanges()
            ]
            for db in {"original", database}
            for label in ("factory", "fluid")
        }
        records = {}
        if not original:
            apply_premise_chemical_receiving_metadata(bd, database)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario=None if original else "remind_ssp2_npi",
            year=2025 if original else 2050,
            provider_database="receiving",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        for grade in ("pa6", "pa66"):
            for suffix in (
                "spinning_support",
                "spinning_htm_treatment",
                "spinning_htm_treatment_ash",
            ):
                sid = f"{grade}_{suffix}"
                records["receiving", provider_service_code(sid)] = _service_dataset(
                    context, services[sid], services, "receiving"
                )

        for grade, air in (("pa6", 32.5e-6), ("pa66", 30e-6)):
            sid = grade + "_spinning_support"
            skey = "receiving", provider_service_code(sid)
            hkey = "receiving", provider_service_code(grade + "_spinning_htm_treatment")
            akey = "receiving", provider_service_code(grade + "_spinning_htm_treatment_ash")
            support = records[skey]
            ref = cast(dict[str, Any], support["pa_spinning_support_reference"])
            load = ref["residue"]
            # Yantai: 4.025 t spent/21,456 t fiber. Brearley's condenser
            # returns recovered fluid; only retirement and BREF air leave.
            retired = 4.025 / 21_456
            assert ref["retired_fluid_kg"] == pytest.approx(retired)
            assert ref["fresh_fluid_kg"] == pytest.approx(retired + air)
            assert ref["residual_air_kg"] == air
            assert ref["factory_units"] == pytest.approx(4e-10)
            assert load["wet_mass_kg"] == pytest.approx(retired)
            assert ref["fresh_fluid_kg"] == pytest.approx(air + load["wet_mass_kg"])
            assert load["carrier_water_kg"] == 0
            assert load["unknown_mineral_kg"] == 0
            # Nominal 26.5/73.5 biphenyl/ether mass fractions; molecular masses
            # use CIAAW g/mol for C12H10 and C12H10O, respectively.
            molecules = (0.265 * retired / 154.212, 0.735 * retired / 170.211)
            fluid_c = 144.132 * fsum(molecules)
            fluid_h = 10.08 * fsum(molecules)
            fluid_o = 15.999 * molecules[1]
            assert load["active_elements_kg"] == pytest.approx(
                {"C": fluid_c, "H": fluid_h, "O": fluid_o, "N": 0, "S": 0, "Na": 0}
            )
            assert fsum(load["active_elements_kg"].values()) + load["carrier_water_kg"] + load[
                "unknown_mineral_kg"
            ] == pytest.approx(load["wet_mass_kg"])
            assert load["net_energy_mj"] == pytest.approx(
                34.016 * fluid_c + (124.265 - 9 * 2.4417) * fluid_h - 9.8324 * fluid_o
            )
            assert load["waste_formed_water_kg"] == pytest.approx(5 * fsum(molecules) * 18.015)
            assert load["nonfuel_salt_elements_kg"] == {}
            assert load["biogenic_carbon_kg"] == 0

            rows = cast(list[dict[str, Any]], support["exchanges"])
            assert all(row["type"] != "biosphere" for row in rows)
            assert rows[0] == {"input": skey, "type": "production", "amount": 1.0}
            demand = {r["input"]: r["amount"] for r in rows[1:]}
            assert demand == pytest.approx(
                {
                    (database, "spinning_factory"): 4e-10,
                    (database, "spinning_fluid"): retired + air,
                    hkey: -retired,
                }
            )
            hwi = cast(dict[str, Any], records[hkey]["chemical_residue_reference"])
            assert hwi["load"] == load
            assert hwi["ash_mass_kg"] >= load["unknown_mineral_kg"]
            assert cast(list[dict[str, Any]], records[hkey]["exchanges"])[0]["amount"] == -1
            assert cast(list[dict[str, Any]], records[akey]["exchanges"])[0]["amount"] == -1
            traces = [r for r in context.technology_rows if r["provider_service_id"] == sid]
            assert {r["component_id"] for r in traces} == {"factory", "fluid"}
            for trace in traces:
                assert trace["native_database_name"] == database
                native = json.loads(str(trace["retained_biosphere_json"]))
                label = str(trace["component_id"])
                native_row = next(
                    row
                    for row in native_before[database, "spinning_" + label]
                    if row["type"] == "biosphere"
                )
                assert native[0]["amount"] == native_row["amount"]
                assert native[0]["code"] == "mc_co2_fossil"

        write_brightway_database(bd, "receiving", records, {})
        for grade in ("pa6", "pa66"):
            skey = "receiving", provider_service_code(grade + "_spinning_support")
            hkey = "receiving", provider_service_code(grade + "_spinning_htm_treatment")
            akey = "receiving", provider_service_code(grade + "_spinning_htm_treatment_ash")
            reference = cast(dict[str, Any], records[skey]["pa_spinning_support_reference"])
            thermal = cast(dict[str, Any], records[hkey]["chemical_residue_reference"])
            node = bd.get_node(database="receiving", code=skey[1])
            lca = LCA({node: 1.0})
            lca.lci()
            amounts = cast(Any, lca.supply_array)
            assert all(isfinite(value) for value in amounts)
            for key, expected in (
                (hkey, reference["residue"]["wet_mass_kg"]),
                (akey, thermal["ash_mass_kg"]),
                ((database, "spinning_factory"), 4e-10 / (2 if original else 3)),
                (
                    (database, "spinning_fluid"),
                    reference["fresh_fluid_kg"] / (2 if original else 3),
                ),
            ):
                activity = bd.get_node(database=key[0], code=key[1])
                assert amounts[lca.dicts.activity[activity.id]] == pytest.approx(expected)
        for key, before in native_before.items():
            assert [
                dict(row) for row in bd.get_node(database=key[0], code=key[1]).exchanges()
            ] == before
