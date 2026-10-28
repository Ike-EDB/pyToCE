"""Registered characterization factors and signed direct emissions in each report."""

import json
from dataclasses import replace
from types import ModuleType
from typing import cast

import numpy as np
import pandas as pd
import pytest
from contribution_fixtures import diagonal_lca

from pytoce.run_lca.calculation.contribution_accumulation import (
    GLOBAL_REGION,
    ContributionMethod,
    _stage_coordinate,
    add_provider_batch,
    begin_matrix_contributions,
    create_contribution_state,
    finish_matrix_contributions,
)
from pytoce.run_lca.calculation.contributions import finalize_contributions
from pytoce.run_lca.calculation.direct_contributions import (
    DirectContribution,
    add_direct_contributions,
    admit_direct_characterization_factors,
)


@pytest.mark.parametrize("supplier_first", (False, True))
def test_actual_direct_flows_and_native_suppliers_close_every_report_view(
    contribution_brightway: ModuleType,
    supplier_first: bool,
) -> None:
    """Mix signed functions and dated shares while preserving real native identities.

    The two ordering cases exercise the same ordinary accumulation contract when
    direct flows are admitted before the supplier matrix or accumulated after it.
    The two regions retain each process's absolute contribution
    when their contribution norms are summed globally.
    """

    bd = contribution_brightway
    lca = diagonal_lca(bd, (2.0, -1.0))
    method_tuple = ("EF v3.1", "climate change")
    bd.Database("direct_biosphere").write(
        {
            ("direct_biosphere", code): {
                "name": name,
                "unit": unit,
                "type": "emission",
                "categories": categories,
            }
            for code, name, unit, categories in (
                ("gas", "modelled gas", "kilogram", ("air", "urban air close to ground")),
                ("water", "modelled water discharge", "cubic meter", ("water", "surface water")),
                ("zero", "zero impact", "kilogram", ("soil",)),
                ("absent", "uncharacterized heat", "megajoule", ("air",)),
            )
        }
    )
    registered = bd.Method(method_tuple)
    registered.write(
        [
            *registered.load(),
            (("direct_biosphere", "gas"), 1.3),
            (("direct_biosphere", "water"), -4.0),
            (("direct_biosphere", "zero"), 0.0),
        ]
    )
    direct_ids = {
        code: int(bd.get_node(database="direct_biosphere", code=code).id)
        for code in ("gas", "water", "zero", "absent")
    }
    supplier = bd.get_node(database="background", code="0")
    shared_flow = int(bd.get_node(database="background_biosphere", code="0").id)
    source_json = json.dumps(
        [
            {
                "database": "background",
                "code": str(index),
                "name": f"activity {index}",
                "location": "GLO",
                "unit": "kilogram",
            }
            for index in (0, 1)
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    activation = {
        "matrix_database": "background",
        "provider_database": "background",
        "provider_code": "0",
        "provider_product_id": int(supplier.id),
        "amount": 1.0,
        "year": 2050,
        "stage_database": "foreground",
        "stage_code": "receiving",
        "life_cycle_stage": "end_of_life_landfill",
        "stage_group": "end_of_life",
        "cutoff_cycle_role": "final_treatment",
        "material_cycle_source": "end_of_life",
        "physical_event_basis": "year of disposal",
        "physical_event_year": 2050,
        "source_flow_names": "end_of_life",
        "lca_fate": "landfill",
        "fate_origin": "managed",
        "material": "Fibres",
        "constituent": "pa6",
        "product": "polymer",
        "virgin_feedstock_route": "mixed",
        "ultimate_carbon_origin": "mixed",
        "good": "all",
        "receiving_function": "wastewater treatment",
        "source_records_json": source_json,
        "endpoint_unit": "kilogram",
    }
    state = create_contribution_state(
        bd,
        {
            region: pd.DataFrame([{**activation, "amount": scale}])
            for region, scale in (("EUR", 1), ("CHA", 2))
        },
        {region: ("foreground", region) for region in ("EUR", "CHA")},
        (ContributionMethod("ef31", "3.1", "climate change", method_tuple, "kg CO2-Eq"),),
        (2050,),
    )
    factors = admit_direct_characterization_factors(state, (*direct_ids.values(), shared_flow))
    assert factors.present.tolist() == [[True, True, True, False, True]]
    precise_factor = float(np.float32(1.3))
    assert factors.factors[0, 0] == precise_factor

    direct_rows = []
    for item in state.activations.itertuples(index=False):
        coordinate = _stage_coordinate(item)
        scale = float(cast(float, item.amount))
        # The gas amount 3 is already the mixed 4 - 1 quantity from one function,
        # including both original source records; source/origin cannot add norms.
        for function, flow_id, amount, unit in (
            ("gas collection", direct_ids["gas"], 3.0, "kilogram"),
            ("gas combustion", direct_ids["gas"], -2.0, "kilogram"),
            ("wastewater treatment", direct_ids["water"], 0.5, "cubic meter"),
            ("wastewater treatment", shared_flow, -0.5, "kilogram"),
            ("wastewater treatment", direct_ids["zero"], 8.0, "kilogram"),
            ("gas combustion", direct_ids["absent"], 35.0, "megajoule"),
        ):
            # Two equal ordinary-anchor shares are already applied to quantities.
            # They retain their branch norms instead of netting different functions.
            row = DirectContribution(
                coordinate, function, source_json, unit, flow_id, amount * scale / 2
            )
            direct_rows.extend((row, row))
        direct_rows.append(
            DirectContribution(
                (coordinate[0], 2035, *coordinate[2:7], 2035, *coordinate[8:]),
                "gas collection",
                source_json,
                "kilogram",
                direct_ids["gas"],
                scale,
            )
        )
    if not supplier_first:
        add_direct_contributions(state, direct_rows)
    matrix = begin_matrix_contributions(state, lca, "background")
    supply = np.asarray(lca.supply_array).reshape((-1, 1))
    add_provider_batch(
        matrix,
        state.activations.iloc[:1],
        (int(supplier.id),),
        supply,
        np.asarray(lca.biosphere_matrix @ supply),
    )
    finish_matrix_contributions(matrix)
    if supplier_first:
        add_direct_contributions(state, direct_rows)
    expected_score = precise_factor - 1
    expected_norm = 5 * precise_factor + 5
    for region, scale in (("EUR", 1), ("CHA", 2)):
        assert state.annual_totals[(region, 2050, 0)] == pytest.approx(
            (scale * expected_score, scale * expected_norm)
        )
        assert state.annual_totals[(region, 2035, 0)] == pytest.approx(
            (scale * precise_factor, scale * precise_factor)
        )
    assert state.characterization_factors[method_tuple][direct_ids["zero"]] == 0
    assert direct_ids["absent"] not in state.characterization_factors[method_tuple]
    tables = dict(finalize_contributions(state))
    for name in ("contributions", "stage_contributions"):
        frame = tables[name]
        for region, scale in (("EUR", 1), ("CHA", 2), (GLOBAL_REGION, 3)):
            selected = frame.loc[frame["region"].eq(region)]
            sums = selected.groupby("contribution_view")[["score", "contribution_norm"]].sum()
            np.testing.assert_allclose(sums["score"], scale * expected_score)
            np.testing.assert_allclose(sums["contribution_norm"], scale * expected_norm)
        labelled = frame.loc[frame["record_role"].eq("modelled_direct")]
        assert set(labelled["receiving_function"]) == {
            "gas collection",
            "gas combustion",
            "wastewater treatment",
        }
        assert set(labelled["source_records_json"]) == {source_json}
        assert all(
            value == ""
            for value in labelled[
                ["provider_database", "provider_code", "activity_database", "activity_code"]
            ]
            .to_numpy()
            .flat
        )
        assert labelled["endpoint_unit"].equals(labelled["elementary_flow_unit"])
        assert set(labelled["elementary_flow_database"]) == {
            "direct_biosphere",
            "background_biosphere",
        }
        supplier_rows = frame.loc[
            frame["contribution_view"].eq("provider_frontier") & frame["record_role"].eq("detail")
        ]
        assert set(supplier_rows["receiving_function"]) == {"wastewater treatment"}
        assert set(supplier_rows["provider_database"]) == {"background"}
        residual = frame.loc[
            frame["contribution_view"].eq("native_activity")
            & frame["record_role"].eq("exact_residual")
        ]
        np.testing.assert_allclose(residual[["score", "contribution_norm"]], 0, atol=1e-12)
    annual = tables["annual_norms"]
    assert set(annual["year"]) == set(range(2019, 2101))
    assert set(annual.loc[annual["contribution_norm"].ne(0), "year"]) == {2035, 2050}
    assert state.direct_values == {}
    assert state.flow_metadata == {}


def test_direct_only_inventory_needs_no_native_producer(
    contribution_brightway: ModuleType,
) -> None:
    """A direct-only process needs no native producer or supply vector."""

    bd = contribution_brightway
    # The registered fixture supplies characterization-factor and flow identities for the
    # direct-emission inventory.
    diagonal_lca(bd, (2.0,))
    flow_id = int(bd.get_node(database="background_biosphere", code="0").id)
    empty = pd.DataFrame(
        columns=(
            "year",
            "matrix_database",
            "provider_database",
            "provider_code",
            "provider_product_id",
            "amount",
            "stage_database",
            "stage_code",
            "life_cycle_stage",
            "stage_group",
            "cutoff_cycle_role",
            "material_cycle_source",
            "physical_event_basis",
            "physical_event_year",
            "source_flow_names",
            "lca_fate",
            "fate_origin",
            "material",
            "constituent",
            "product",
            "virgin_feedstock_route",
            "ultimate_carbon_origin",
            "good",
        )
    )
    method = ContributionMethod(
        "ef31", "3.1", "climate change", ("EF v3.1", "climate change"), "kg CO2-Eq"
    )
    state = create_contribution_state(
        bd, {"EUR": empty}, {"EUR": ("foreground", "EUR")}, (method,), (2050,)
    )
    admit_direct_characterization_factors(state, (flow_id,))
    coordinate = (
        "EUR",
        2050,
        "end_of_life_landfill",
        "end_of_life",
        "final_treatment",
        "end_of_life",
        "year of disposal",
        2050,
        "end_of_life",
        "landfill",
        "managed",
    )
    source = json.dumps(
        [
            {
                "database": "background",
                "code": "0",
                "name": "activity 0",
                "location": "GLO",
                "unit": "kilogram",
            }
        ]
    )
    row = DirectContribution(coordinate, "gas collection", source, "kilogram", flow_id, 1.0)
    add_direct_contributions(
        state, (row, replace(row, receiving_function="gas combustion", amount=-0.25))
    )
    tables = dict(finalize_contributions(state))
    regional = tables["contributions"].loc[lambda frame: frame["region"].eq("EUR")]
    sums = regional.groupby("contribution_view")[["score", "contribution_norm"]].sum()
    np.testing.assert_allclose(sums["score"], 1.5)
    np.testing.assert_allclose(sums["contribution_norm"], 2.5)
    assert state.native_values == {}
