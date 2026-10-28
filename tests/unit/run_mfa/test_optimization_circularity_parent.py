"""Regional optimization controls from direct material realization."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from material_recipe_fixtures import material_source_members

from pytoce.download_data.sources.ecoinvent_materials import derive_material_recipe
from pytoce.run_mfa.model.materials import (
    load_material_composition,
    material_realization_table,
    realize_materials,
)
from pytoce.run_mfa.model.optimization_parents import optimization_circularity_parent
from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue
from pytoce.run_mfa.scenario.parent_relative import compiled_scenario_table_identity
from pytoce.run_mfa.scenario.scenario_contract import (
    ScenarioEvaluation,
    ScenarioFeasibilityError,
)


def _parent_inputs(scale: float) -> tuple[pd.DataFrame, pd.DataFrame, ScenarioEvaluation]:
    """Repeat a balanced three-polymer fixture across the required grid.

    HDPE is the only chemical feed in the discarded mixture. PP production
    consumes recovered propylene; PET is ineligible for this chemical route.
    The independent regional chemical-share witness is 2.6 / 26 = 0.1.
    """
    regions = load_parameter_catalogue("collection_rate").coordinates["region"]
    grid = pd.MultiIndex.from_product(
        [range(2020, 2101), regions], names=["year", "region"]
    ).to_frame(index=False)
    collected = np.array([6.5, 19.5, 0.0]) * scale
    collected_total = sum((Fraction(float(value)) for value in collected), Fraction())
    state = realize_materials(
        year=2020,
        region="EUR",
        materials=("HDPE", "PET", "PP"),
        composition=load_material_composition(),
        recipe=derive_material_recipe(material_source_members(), source_identity="invented_xml"),
        collected=collected,
        route_totals=tuple(
            collected_total * share
            for share in (Fraction(1, 5), Fraction(1, 10), Fraction(3, 10), Fraction(2, 5))
        ),
        processing_requirement=(Fraction(100.0 * scale),) * 3,
        primary_imports=(Fraction(0),) * 3,
        primary_exports=(Fraction(0),) * 3,
    )
    material = grid.merge(
        material_realization_table(state).drop(columns=["year", "region"]), how="cross"
    )
    rows = []
    for name, multiplier in (("use => eol", 1.0), ("eol => collected", 0.65)):
        for material_name, amount in (("HDPE", 10.0), ("PET", 30.0), ("PP", 0.0)):
            for element, fraction in (("C", 0.8), ("Other Elements", 0.2)):
                rows.append(
                    {
                        "foreground_scenario": "fixture",
                        "source_flow_name": name,
                        "material": material_name,
                        "good": "Packaging" if name == "use => eol" else "all",
                        "element": element,
                        "amount": amount * scale * multiplier * fraction,
                        "unit": "Mt",
                    }
                )
    flows = grid.merge(pd.DataFrame(rows), how="cross")
    controls = {
        "collection_rate": 0.65,
        "mechanical_recycling_rate": 0.2,
        "chemical_recycling_rate": 0.1,
        "incineration_rate": 0.3,
        "bio_production_rate": 0.2,
        "daccu_production_rate": 0.1,
    }
    compiled = grid.merge(
        pd.DataFrame(
            [
                {"parameter_id": name, "resolved_value": value, "unit": "fraction"}
                for name, value in controls.items()
            ]
        ),
        how="cross",
    )
    return (
        flows,
        material,
        ScenarioEvaluation(
            foreground_scenario="fixture",
            foreground_scenario_identity="scenario::fixture",
            ssp_scenario="SSP2",
            compiled_parameters=compiled,
            normalized_change_points=(),
            compiled_table_sha256=compiled_scenario_table_identity(compiled),
        ),
    )


@pytest.mark.parametrize("scale", (1.0, 1e-14, 0.0))
def test_regional_parent_preserves_selective_treatment_and_exact_zero_controls(
    scale: float,
) -> None:
    """Use all posttrade waste, retaining controls when actual masses are exactly zero."""
    flows, material, scenario = _parent_inputs(scale)
    result = optimization_circularity_parent(
        flows,
        material,
        physical_state_identity="physical::invented_fixture",
        parent=scenario,
    )
    expected_grid = pd.MultiIndex.from_product(
        [
            range(2020, 2101),
            sorted(load_parameter_catalogue("collection_rate").coordinates["region"]),
        ],
        names=["year", "region"],
    ).to_frame(index=False)
    pd.testing.assert_frame_equal(result.table[["year", "region"]], expected_grid)
    expected = np.broadcast_to([0.65, 0.2, 0.1, 0.3, 0.4, 0.7, 0.2, 0.1], (len(expected_grid), 8))
    np.testing.assert_allclose(result.table.iloc[:, 2:].to_numpy(), expected, rtol=1e-14, atol=0.0)
    if scale:
        local = material.loc[material["year"].eq(2020) & material["region"].eq("EUR")]
        chemical = local.loc[local["flow_name"].eq("chemical_treatment")]
        assert chemical.loc[chemical["source_material"].eq("PET"), "amount"].sum() == 0.0
        assert chemical["amount"].sum() / (6.5 * scale) == pytest.approx(0.4)
        assert result.table["chemical_recycling_rate"].iloc[0] == pytest.approx(0.1)


def test_zero_waste_with_no_compiled_treatment_simplex_cannot_initialize_optimization() -> None:
    """Preserve compiled treatment controls for a valid zero-waste MFA."""
    flows, material, scenario = _parent_inputs(0.0)
    controls = scenario.compiled_parameters
    controls.loc[controls["parameter_id"].eq("mechanical_recycling_rate"), "resolved_value"] = 0.9
    controls.loc[controls["parameter_id"].eq("chemical_recycling_rate"), "resolved_value"] = 0.5
    with pytest.raises(ScenarioFeasibilityError, match="zero collected waste") as caught:
        optimization_circularity_parent(
            flows,
            material,
            physical_state_identity="physical::invented_fixture",
            parent=scenario,
        )
    assert caught.value.code == "optimization_parent_not_representable"
    assert caught.value.context["family"] == "treatment"
