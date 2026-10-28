"""Connect actual PA precursor residue quantities to complete source treatment."""

import math
from pathlib import Path
from typing import Any, cast

import pytest
from bw2calc import LCA
from chemical_residue_fixtures import chemical_residue_biosphere, chemical_residue_native_inventory
from material_recipe_fixtures import physical_material_recipe
from pa_castor_hco_fixtures import castor_hco_biosphere, castor_hco_native_inventory

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


def test_hco_load_reaches_complete_negative_receiving_graph(tmp_path: Path) -> None:
    """Test HCO residue treatment with synthetic catalyst and polyacrylamide supplies.

    Manufacturing functions supply the precursor quantities. The functional
    unit is the complete manufacturing residue sent to treatment.
    """
    provider = prepared_provider_authority()
    services = {
        str(s["provider_service_id"]): s
        for s in cast(list[dict[str, object]], provider.catalogue["services"])
    }
    service_id = "pa_hco_cake_treatment"
    service = services[service_id]
    construction = cast(dict[str, object], service["construction"])
    ash_id = str(cast(dict[str, object], construction["ash"])["service_id"])
    names = BrightwayNames(tmp_path / "brightway", "pa_residue", "biosphere", "source")
    with selected_brightway_project(names) as bd:
        native = chemical_residue_native_inventory(provider.catalogue, "source")
        biosphere = chemical_residue_biosphere()
        native.update(castor_hco_native_inventory(provider.catalogue, "source", original=True))
        # Both fixtures name the same European gas furnace. Keep the water
        # fixture's key and the castor fixture's complete fuel-based process;
        # two different nodes with that descriptor make source selection ambiguous.
        furnace = native.pop(("source", "drying_heat"))
        native["source", "heat_europe"]["exchanges"] = [
            {**row, "input": ("source", "heat_europe")} if row["type"] == "production" else row
            for row in furnace["exchanges"]
        ]
        biosphere.update(castor_hco_biosphere())
        bd.Database("biosphere").write(biosphere, searchable=False)
        bd.Database("source").write(native, searchable=False)
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
        keys = [("receiving", provider_service_code(s)) for s in (service_id, ash_id)]
        for sid, key in zip((service_id, ash_id), keys, strict=True):
            records[key] = _service_dataset(context, services[sid], services, "receiving")
        reference = cast(dict[str, Any], records[keys[0]]["chemical_residue_reference"])
        load = reference["load"]
        wet, water = load["wet_mass_kg"], load["carrier_water_kg"]
        known = load["active_elements_kg"]
        salt_mass = math.fsum(
            q for salt in load["nonfuel_salt_elements_kg"].values() for q in salt.values()
        )
        assert math.fsum(known.values()) + salt_mass + water + load[
            "unknown_mineral_kg"
        ] == pytest.approx(wet)
        assert load["biogenic_carbon_kg"] > 0
        assert reference["ash_mass_kg"] > load["unknown_mineral_kg"] > 0
        assert reference["supplies"]["TMT"] > 0
        write_brightway_database(bd, "receiving", records, {})
        hwi, ash = (bd.get_node(database=k[0], code=k[1]) for k in keys)
        assert next(iter(hwi.production()))["amount"] == -1
        assert next(iter(ash.production()))["amount"] == -1
        lca = LCA({hwi: -wet})
        lca.lci()
        supply = cast(Any, lca.supply_array)
        assert all(math.isfinite(q) for q in supply)
        assert supply[lca.dicts.activity[ash.id]] == pytest.approx(reference["ash_mass_kg"])
        ash_demand = next(ex for ex in hwi.technosphere() if ex.input.key == keys[1])
        assert ash_demand["amount"] * wet == pytest.approx(-reference["ash_mass_kg"])
