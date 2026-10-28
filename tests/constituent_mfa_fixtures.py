"""Small MFA fixtures and synthetic carbon inputs for LCA boundary tests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from material_carbon_fixtures import prepared_material_carbon
from material_recipe_fixtures import material_source_members

from pytoce.download_data.sources.ecoinvent_materials import (
    EcoinventMaterialRecipe,
    derive_material_recipe,
)
from pytoce.run_lca.calculation.providers import MaterialCarbon
from pytoce.run_mfa.model.cohorts import accepted_default_cohort_result
from pytoce.run_mfa.model.flows import calculate_mfa_flows
from pytoce.run_mfa.model.materials import (
    load_material_composition,
    material_realization_table,
)
from pytoce.run_mfa.model.mfa import MFAEvaluation
from pytoce.run_mfa.model.result_tables import (
    _cohort_material_identity_table,
    _cohort_state_table,
    mfa_flow_table,
    nominal_feedstock_table,
)
from pytoce.run_mfa.remind_mfa.process_inputs import RemindMFAProcessInputs
from pytoce.run_mfa.remind_mfa.translation import NormalizedCohortState
from pytoce.run_mfa.scenario.catalogue import _catalogue_payload


@pytest.fixture(scope="module")
def carbon_authorities(tmp_path_factory: pytest.TempPathFactory):
    """Prepare carbon data and the material recipe from synthetic source records."""
    rows = prepared_material_carbon(tmp_path_factory.mktemp("custody_carbon"))
    composition = load_material_composition()
    origins = ("fossil", "biogenic", "atmospheric_dac")
    values = {
        (row["material"], row["constituent"], row["virgin_feedstock_route"]): tuple(
            float(row[name + "_carbon"]) for name in origins
        )
        for row in rows
    }
    carbon = MaterialCarbon(
        "invented_prepared_carbon",
        tuple((c.material, c.constituent) for c in composition.constituents),
        tuple(
            tuple(values[(c.material, c.constituent, origin)] for origin in origins)
            for c in composition.constituents
        ),
    )
    return carbon, derive_material_recipe(material_source_members(), source_identity="invented_xml")


def empty_model(material: str = "HDPE") -> tuple[NormalizedCohortState, RemindMFAProcessInputs]:
    """Build four years of zero flows for all 12 world regions, families and end-use groups."""
    coordinates = _catalogue_payload()["coordinate_sets"]
    regions, materials, goods = (
        tuple(coordinates[name]) for name in ("region", "material", "good")
    )
    years = (2018, 2019, 2020, 2021)
    nt, nr, nm, ng = len(years), len(regions), len(materials), len(goods)
    shares = np.zeros((nr, nm, ng))
    shares[:, materials.index(material)] = 1
    state = NormalizedCohortState(
        years=years,
        interval_years=np.ones(nt),
        regions=regions,
        materials=materials,
        goods=goods,
        elements=("C", "Other Elements"),
        stock_by_cohort=np.zeros((nt, nt, nr, ng)),
        outflow_by_cohort=np.zeros((nt, nt, nr, ng)),
        annual_stock=np.zeros((nt, nr, ng)),
        annual_inflow=np.zeros((nt, nr, ng)),
        annual_outflow=np.zeros((nt, nr, ng)),
        material_shares=shares,
        carbon_content=np.tile([[0.25], [0.75]], (1, nm)),
        lifetime_mean=np.full(ng, 10.0),
        lifetime_std=np.full(ng, 2.0),
    )
    inputs = RemindMFAProcessInputs(
        final_imports=np.zeros((nt, 2, nr, nm, ng)),
        final_exports=np.zeros((nt, 2, nr, nm, ng)),
        intermediate_imports=np.zeros((nt, 2, nr, nm)),
        intermediate_exports=np.zeros((nt, 2, nr, nm)),
        primary_imports=np.zeros((nt, 2, nr, nm)),
        primary_exports=np.zeros((nt, 2, nr, nm)),
        waste_imports=np.zeros((nt, 2, nr, nm, ng)),
        waste_exports=np.zeros((nt, 2, nr, nm, ng)),
        collection_rate=np.ones((nt, nr)),
        mechanical_recycling_rate=np.zeros((nt, nr)),
        chemical_recycling_rate=np.zeros((nt, nr)),
        incineration_rate=np.ones((nt, nr)),
        bio_production_rate=np.zeros((nt, nr)),
        daccu_production_rate=np.zeros((nt, nr)),
        waste_trade_propensity_ledger=pd.DataFrame(),
    )
    return state, inputs


def evaluate_mfa(
    state: NormalizedCohortState,
    inputs: RemindMFAProcessInputs,
    recipe: EcoinventMaterialRecipe,
    *,
    foreground_scenario: str = "constituent_fixture",
) -> MFAEvaluation:
    """Calculate constituents and cohort/report tables through their real owners."""
    state = replace(
        state,
        annual_stock=state.stock_by_cohort.sum(axis=1),
        annual_outflow=state.outflow_by_cohort.sum(axis=1),
    )
    cohort = accepted_default_cohort_result(state)
    physical = calculate_mfa_flows(
        cohort,
        inputs,
        recipe=recipe,
        composition=load_material_composition(),
        first_year=state.years[0],
        last_year=state.years[-1],
    )
    return MFAEvaluation(
        mfa_flow_table(physical, cohort, foreground_scenario=foreground_scenario),
        pd.DataFrame(),
        pd.concat(
            [material_realization_table(item) for item in physical.material_states],
            ignore_index=True,
        ),
        nominal_feedstock_table(physical),
        _cohort_state_table(cohort, foreground_scenario=foreground_scenario),
        _cohort_material_identity_table(cohort, foreground_scenario=foreground_scenario),
        "invented_physical_state",
        cohort.material_identity_method,
    )
