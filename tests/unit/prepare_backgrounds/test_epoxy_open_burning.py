"""Actual cured epoxy burning, including fixed hardener and principal diagnostics."""

import json
from math import fsum
from typing import Any, cast

import pandas as pd
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from pa_open_burning_fixtures import (
    _AIR,
    _INDICATOR,
    open_burning_biosphere,
    open_burning_native_inventory,
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
from pytoce.run_lca.calculation.carbon import direct_treatment_carbon_flows

_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")


@pytest.mark.parametrize("future", (False, True), ids=("original", "future"))
def test_epoxy_burning_uses_cured_material_and_excludes_hardener_from_diagnostic(
    tmp_path, future: bool
) -> None:
    """Bind all three origins using the specified material quantities."""
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    services = {
        row["provider_service_id"]: row
        for row in catalogue["services"]
        if row["construction"]["kind"] == "epoxy_open_burning"
    }
    assert set(services) == {f"epoxy_{origin}_open_burning" for origin in _ORIGINS}
    names = BrightwayNames(tmp_path / "brightway", "epoxy_burning", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(open_burning_biosphere("biosphere"), searchable=False)
        for database in ("original", "future"):
            bd.Database(database).write(
                open_burning_native_inventory(
                    database, original=database == "original", biosphere_database="biosphere"
                ),
                searchable=False,
            )
        before = {
            node.key: [dict(row) for row in node.exchanges()]
            for database in ("original", "future")
            for node in bd.Database(database)
        }
        dated = _ActivityResolver(bd, "future" if future else "original")
        year = 2050 if future else 2025
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=year,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                row["method_authority_id"]: row for row in _method_authority_rows(catalogue)
            },
            complete_source_ids=_service_source_closures(catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        data = {
            ("providers", provider_service_code(service_id)): _service_dataset(
                context, service, services, "providers"
            )
            for service_id, service in services.items()
        }
        bd.Database("providers").write(data, searchable=False)

        # EP1838750B1 Table 6: retained MXDA accompanies one kg principal.
        # EEW186 fixes n=8/71; these rational CHO fractions are the resin's
        # actual nominal formula, independent of the environmental suppliers.
        ratio = 15.44 / 84.56
        principal_c = 1635 / 2201
        mxda_c = ratio * 96 / 136
        elements = {
            "C": principal_c + mxda_c,
            "H": 466 / 6603 + ratio * 12 / 136,
            "O": 1232 / 6603,
            "N": ratio * 28 / 136,
        }
        wet = 1 + ratio
        heat = (
            34.016 * elements["C"]
            + 124.265 * elements["H"]
            - 9.8324 * elements["O"]
            + 6.276 * elements["N"]
        )
        for index, origin in enumerate(_ORIGINS):
            service_id = f"epoxy_{origin}_open_burning"
            node = bd.get_node(database="providers", code=provider_service_code(service_id))
            assert node.rp_exchange()["amount"] == 1
            assert list(node.technosphere()) == []
            trace = next(
                row
                for row in context.semantic_rows
                if row["provider_service_id"] == service_id
                and row["ledger_entry_type"] == "service_construction"
            )
            terms = json.loads(trace["project_terms_json"])
            assert terms["original_source_key"] == ["original", "burning"]
            assert terms["wet_mass_kg_per_principal_kg"] == pytest.approx(wet)
            assert terms["input_elements_kg"] == pytest.approx(elements)
            assert fsum(terms["input_elements_kg"].values()) == pytest.approx(wet)
            assert terms["incorporated_mxda_kg"] == ratio
            # Cleanup acetone leaves manufacture.
            assert terms["incoming_water_kg"] == 0
            assert terms["principal_carbon_kg"] == pytest.approx(
                {item: principal_c if item == origin else 0 for item in _ORIGINS}
            )
            assert terms["auxiliary_carbon_kg"] == pytest.approx({"fossil": mxda_c})
            assert terms["gross_heat_potential_mj"] == pytest.approx(heat)
            rows = {
                (row.input["name"], row.input["unit"], tuple(row.input["categories"])): row
                for row in node.biosphere()
            }
            assert rows["Heat, waste", "megajoule", _AIR]["amount"] == pytest.approx(heat)
            assert rows["Non-hazardous waste disposed", "kilogram", _INDICATOR][
                "amount"
            ] == pytest.approx(wet)
            assert ("PCB", "kilogram", _AIR) not in rows
            assert ("Carbon dioxide, fossil", "kilogram", ("air",)) not in rows
            for name, coefficient in (("Carbon monoxide", 0.0386), ("Methane", 0.0059)):
                assert fsum(
                    row["amount"] for key, row in rows.items() if key[0].startswith(name + ",")
                ) == pytest.approx(wet * coefficient)

            # LCI keeps whole emissions; the ordinary carbon reader separately
            # contracts only principal carbon on the same demand.
            demand = 2.5
            lca = LCA({node: demand})
            lca.lci()
            assert lca.supply_array[lca.dicts.activity[node.id]] == pytest.approx(demand)
            activations = pd.DataFrame(
                [
                    {
                        "year": 2050,
                        "region": "EUR",
                        "life_cycle_stage": "mismanaged_open_burning_treatment",
                        "material": "Others",
                        "constituent": "epoxy",
                        "product": "polymer",
                        "virgin_feedstock_route": origin,
                        "good": "all",
                        "material_cycle_source": "virgin_feedstock",
                        "ultimate_carbon_origin": origin,
                        "lca_fate": "open_burning",
                        "fate_origin": "remind_mfa",
                        "provider_database": "providers",
                        "provider_code": node.key[1],
                        "provider_anchor_year": year,
                        "amount": demand,
                        "mfa_carbon_amount_kg": demand * principal_c,
                    }
                ]
            )
            diagnostic = direct_treatment_carbon_flows(bd, activations)
            assert set(diagnostic.mfa_carbon_origin) == {origin}
            assert set(diagnostic.origin_mapping_status) == {"actual_epoxy_principal_origin"}
            for species, name, carbon in (
                ("carbon dioxide", "Carbon dioxide", 12 / 44),
                ("carbon monoxide", "Carbon monoxide", 12 / 28),
                ("methane", "Methane", 12 / 16),
            ):
                gases = [row for key, row in rows.items() if key[0].startswith(name + ",")]
                total = fsum(row["amount"] for row in gases)
                fossil = fsum(
                    row["amount"] for row in gases if row.input["name"] == name + ", fossil"
                )
                assert fossil == pytest.approx(
                    total * (1 if origin == "fossil" else mxda_c / elements["C"])
                )
                for row in gases:
                    assert float(
                        lca.inventory[lca.dicts.biosphere[row.input.id], :].sum()
                    ) == pytest.approx(demand * row["amount"], rel=2e-7)
                    fractions = row["principal_carbon_origin_fractions"]
                    assert all(
                        share == 0 for position, share in enumerate(fractions) if position != index
                    )
                selected = diagnostic.loc[diagnostic.carbon_species.eq(species)]
                assert fsum(selected.carbon_amount) == pytest.approx(
                    demand * total * carbon * principal_c / elements["C"]
                )
                assert fsum(selected.carbon_amount) < demand * total * carbon
            assert before == {
                node.key: [dict(row) for row in node.exchanges()]
                for database in ("original", "future")
                for node in bd.Database(database)
            }
