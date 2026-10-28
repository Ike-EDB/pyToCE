"""Test epoxy application supplies and residue treatment with synthetic source inventories."""

from math import fsum, log
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from chemical_residue_fixtures import (
    apply_premise_chemical_receiving_metadata,
    chemical_residue_biosphere,
    chemical_residue_native_inventory,
)
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from provider_recipe_fixtures import _process

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


@pytest.mark.parametrize(("year", "ordinary", "factor"), [(2025, "native", 1), (2050, "dated", 2)])
def test_epoxy_frontier_preserves_whole_supplies_and_actual_managed_loss(
    tmp_path: Path, year: int, ordinary: str, factor: int
) -> None:
    """Solve three application declarations with synthetic inventories and nonunit outputs.

    Chemical residue fixtures provide hazardous incineration, ash treatment and
    collection. Synthetic producers and freight services have observable
    emissions; their production amounts and the alkyd mixing output differ
    between dates. The package material recipe determines gross resin and
    its destinations; separate tests verify those equations.
    This test checks their connection to supplies and waste treatment.
    """
    catalogue = cast(dict[str, Any], load_provider_catalogue())
    selected = [
        service
        for service in catalogue["services"]
        if service["construction"]["kind"] == "epoxy_coating"
    ]
    assert {service["construction"]["principal_carbon_origin"] for service in selected} == {
        "fossil",
        "biogenic",
        "atmospheric_dac",
    }
    provider = PreparedProviderAuthority(
        identity="invented_epoxy_application",
        interface_database="bounded_interface",
        catalogue={**catalogue, "services": selected, "routes": []},
        routes=pd.DataFrame(),
        method_authorities=_method_authority_rows(catalogue),
    )
    construction = selected[0]["construction"]
    markers = (
        "mxda",
        "acetone",
        "compressed_air",
        "delivery",
        "mix_power",
        "mix_factory",
        "mix_nitrogen",
        "donor_material",
        "collection_road",
        "collection_rail",
    )

    def inventory(database: str, multiplier: int) -> dict[tuple[str, str], dict[str, Any]]:
        """Add synthetic procurement processes to the treatment inventories."""
        data = chemical_residue_native_inventory(catalogue, database)

        def row(code: str, amount: float, *, bio: bool = False) -> dict[str, Any]:
            """Keep source uncertainty, bounds and allocation observable after scaling."""
            return {
                "input": ("biosphere" if bio else database, code),
                "type": "biosphere" if bio else "technosphere",
                "amount": amount,
                "uncertainty type": 2,
                "loc": log(amount) + 0.02,
                "scale": 0.19,
                "minimum": amount / 2,
                "maximum": amount * 2,
                "allocation": "invented epoxy native allocation",
            }

        def leaf(code: str, product: str, unit: str, marker: str) -> None:
            """Provide a synthetic supplier with a distinguishable solved inventory."""
            data[(database, code)] = {
                **_process(
                    database,
                    code,
                    "invented " + code,
                    product,
                    production_amount=3,
                    exchanges=[row("ep_" + marker, 6 * multiplier, bio=True)],
                ),
                "unit": unit,
            }

        leaf("ep_freight", "transport, freight, lorry", "ton kilometer", "delivery")
        for label, selector in construction["supplies"].items():
            producer = "ep_" + label + "_producer"
            leaf(producer, selector["reference_product"], selector["unit"], label)
            basis = 4 if multiplier == 1 else 7
            data[(database, "ep_" + label)] = {
                **_process(
                    database,
                    "ep_" + label,
                    selector["name"],
                    selector["reference_product"],
                    location=selector["exact_location"],
                    production_amount=basis,
                    exchanges=[
                        row(producer, 1.02 * basis),
                        row("ep_freight", 0.011 * basis),
                        row("ep_freight", 0.017 * basis),
                    ],
                ),
                "unit": selector["unit"],
            }
        for code, product, unit, marker in (
            ("ep_power", "electricity, medium voltage", "kilowatt hour", "mix_power"),
            ("ep_factory", "chemical factory, organics", "unit", "mix_factory"),
            ("ep_nitrogen", "nitrogen, liquid", "kilogram", "mix_nitrogen"),
            ("ep_resin", "alkyd resin, without solvent", "kilogram", "donor_material"),
            ("ep_solvent", "solvent, organic", "kilogram", "donor_material"),
        ):
            leaf(code, product, unit, marker)
        data[(database, "ep_paint_waste")] = _process(
            database,
            "ep_paint_waste",
            "invented treatment of waste paint",
            "waste paint",
            production_amount=-1,
            exchanges=[row("ep_donor_material", 1, bio=True)],
        )
        mixing = construction["mixing_source"]
        basis = 2 if multiplier == 1 else 5
        data[(database, "ep_mixing")] = _process(
            database,
            "ep_mixing",
            mixing["name"],
            mixing["reference_product"],
            location=mixing["exact_location"],
            production_amount=basis,
            exchanges=[
                row("ep_power", 0.11 * multiplier * basis),
                row("ep_factory", 3e-9 * basis),
                row("ep_nitrogen", 0.03 * basis),
                row("cr_nitrogen_air", 0.03 * basis, bio=True),
                row("ep_resin", 0.6 * basis),
                row("ep_solvent", 0.4 * basis),
                {"input": (database, "ep_paint_waste"), "type": "technosphere", "amount": -0.1},
                row("ep_donor_material", 0.2 * basis, bio=True),
            ],
        )
        for code, marker in (("cr_road", "collection_road"), ("cr_rail", "collection_rail")):
            data[(database, code)]["exchanges"].append(row("ep_" + marker, 1, bio=True))
        return data

    names = BrightwayNames(tmp_path / "brightway", "epoxy_coating", "biosphere", "native")
    with selected_brightway_project(names) as bd:
        biosphere = chemical_residue_biosphere()
        biosphere.update(
            {
                ("biosphere", "ep_" + marker): {
                    "name": "invented epoxy " + marker + " inventory",
                    "unit": "kilogram",
                    "categories": ("air",),
                    "type": "emission",
                }
                for marker in markers
            }
        )
        bd.Database("biosphere").write(biosphere, searchable=False)
        for database, multiplier in (("native", 1), ("dated", 2)):
            bd.Database(database).write(inventory(database, multiplier), searchable=False)
        apply_premise_chemical_receiving_metadata(bd, "dated")
        original = {
            node.key: (dict(node), [dict(exchange) for exchange in node.exchanges()])
            for database in ("native", "dated")
            for node in bd.Database(database)
        }
        recipe = physical_material_recipe()
        conversion = next(
            item
            for item in recipe.conversion
            if item.operation == "others_epoxy_coating_conversion"
        )
        frontier = prepare_provider_frontier(
            bd,
            background_scenario="remind_ssp2_npi",
            year=year,
            ordinary_database=ordinary,
            technology_database="dated",
            provider=provider,
            material_recipe=recipe,
            resolver_cache=_NativeResolverCache("native", "biosphere"),
        )
        for service in selected:
            service_id = service["provider_service_id"]
            origin = service["construction"]["principal_carbon_origin"]
            root = bd.get_node(
                database=frontier.database_name, code=provider_service_code(service_id)
            )
            reference = root["epoxy_reference_per_retained_principal_kg"]
            mass = reference["managed_residue"]["mass_kg"]
            assert reference["gross_principal_kg"] == conversion.principal_feed_kg_per_finished_kg
            assert mass == pytest.approx(
                reference["preparation_residue"]["mass_kg"]
                + reference["settled_overspray"]["mass_kg"]
            )
            assert (
                fsum(
                    reference[exit_name]["mass_kg"]
                    for exit_name in ("retained", "airborne_overspray")
                )
                > 0
            )
            hwi = bd.get_node(
                database=frontier.database_name,
                code=f"epoxy_coating__{service_id}__managed_incineration",
            )
            ash = bd.get_node(
                database=frontier.database_name, code=f"epoxy_coating__{service_id}__managed_ash"
            )
            root_rows = {exchange.input.key: dict(exchange) for exchange in root.technosphere()}
            assert set(root_rows) == {
                (ordinary, "ep_" + label) for label in construction["supplies"]
            } | {(ordinary, code) for code in ("ep_power", "ep_factory", "ep_nitrogen")} | {hwi.key}
            assert root_rows[hwi.key]["amount"] == pytest.approx(-mass)
            assert next(iter(hwi.production()))["amount"] == -1
            assert next(iter(ash.production()))["amount"] == -1
            for label, amount in reference["supply_amounts"].items():
                assert root_rows[ordinary, "ep_" + label]["amount"] == pytest.approx(amount)

            # Transfer the actual operation rows on their specified production basis.
            source_mixing = original[ordinary, "ep_mixing"][1]
            production = next(row["amount"] for row in source_mixing if row["type"] == "production")
            scale = reference["gross_binder_kg"] / production
            retained = {
                row.input.key: dict(row) for row in (*root.technosphere(), *root.biosphere())
            }
            for source in source_mixing:
                if source["input"] not in {
                    (ordinary, "ep_power"),
                    (ordinary, "ep_factory"),
                    (ordinary, "ep_nitrogen"),
                    ("biosphere", "cr_nitrogen_air"),
                }:
                    continue
                actual = retained[source["input"]]
                for field in ("amount", "minimum", "maximum"):
                    assert actual[field] == pytest.approx(source[field] * scale)
                assert actual["loc"] == pytest.approx(source["loc"] + log(scale))
                assert actual["scale"] == source["scale"]
                assert actual["allocation"] == source["allocation"]

            treatment = hwi["chemical_residue_reference"]
            hwi_rows = list(hwi.technosphere())
            assert sum(
                row["amount"] for row in hwi_rows if row.input == ash
            ) * mass == pytest.approx(-treatment["ash_mass_kg"])
            assert treatment["ash_mass_kg"] > 0
            assert treatment["nitrogen_ratio"] > 0
            assert len([row for row in hwi_rows if row.input.key == (ordinary, "cr_road")]) == 1
            assert len([row for row in hwi_rows if row.input.key == (ordinary, "cr_rail")]) == 1
            gases = {
                row.input["name"]: row
                for row in hwi.biosphere()
                if "Carbon dioxide" in row.input["name"]
            }
            fossil = gases["Carbon dioxide, fossil"]
            assert fossil["amount"] > 0  # Actual fossil hardener and cleanup survive every origin.
            if origin != "fossil":
                gas = gases["Carbon dioxide, non-fossil"]
                assert gas["amount"] > 0
                expected = (0, 1, 0) if origin == "biogenic" else (0, 0, 1)
                assert gas["principal_carbon_origin_fractions"] == pytest.approx(expected)
            else:
                assert "Carbon dioxide, non-fossil" not in gases

            calculation = LCA({root: 1})
            calculation.lci()
            assert cast(NDArray[np.float64], calculation.supply_array)[
                calculation.dicts.activity[hwi.id]
            ] == pytest.approx(mass)
            assert cast(NDArray[np.float64], calculation.supply_array)[
                calculation.dicts.activity[ash.id]
            ] == pytest.approx(treatment["ash_mass_kg"])
            quantities = reference["supply_amounts"]
            expected_emissions = {
                **{label: amount * 1.02 * 2 * factor for label, amount in quantities.items()},
                "delivery": fsum(quantities.values()) * (0.011 + 0.017) * 2 * factor,
                "mix_power": reference["gross_binder_kg"] * 0.11 * factor * 2 * factor,
                "mix_factory": reference["gross_binder_kg"] * 3e-9 * 2 * factor,
                "mix_nitrogen": reference["gross_binder_kg"] * 0.03 * 2 * factor,
                "donor_material": 0,
                "collection_road": 0.04 * mass,
                "collection_rail": 0.007 * mass,
            }
            for marker, expected in expected_emissions.items():
                node = bd.get_node(database="biosphere", code="ep_" + marker)
                actual = float(calculation.inventory[calculation.dicts.biosphere[node.id], :].sum())
                assert actual == pytest.approx(expected)
        assert {
            node.key: (dict(node), [dict(exchange) for exchange in node.exchanges()])
            for database in ("native", "dated")
            for node in bd.Database(database)
        } == original
