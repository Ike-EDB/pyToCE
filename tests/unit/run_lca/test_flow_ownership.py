"""MFA quantities and their distinct uses in the LCA inventory."""

import json

import pytest
from constituent_mfa_fixtures import carbon_authorities as carbon_authorities
from constituent_mfa_fixtures import empty_model, evaluate_mfa

from pytoce.run_lca.calculation.flow_coverage import mfa_to_lca_flow_coverage


def test_complete_physical_flow_report_distinguishes_demands_and_duplicate_views(
    carbon_authorities,
) -> None:
    """Account for history, treatment mass, and chemical products with each burden counted once."""
    _, recipe = carbon_authorities
    state, inputs = empty_model()
    region, good = state.regions.index("EUR"), state.goods.index("Packaging")
    state.annual_inflow[0, region, good] = 1
    state.stock_by_cohort[:2, 0, region, good] = 1
    state.outflow_by_cohort[2, 0, region, good] = 1
    state.annual_inflow[2, region, good] = 2
    state.stock_by_cohort[2:, 2, region, good] = 2
    inputs.collection_rate[2, region] = 0.8
    inputs.mechanical_recycling_rate[2, region] = 0.2
    inputs.chemical_recycling_rate[2, region] = 0.05
    inputs.incineration_rate[2, region] = 0.45
    mfa = evaluate_mfa(state, inputs, recipe)
    observed = mfa_to_lca_flow_coverage(
        mfa,
        foreground_scenario="constituent_fixture",
        background_scenario="remind_ssp2_npi",
    ).set_index(["source_table", "source_flow_name", "product"])
    aggregate = observed.loc["flows.parquet"].droplevel("product")
    material = observed.loc["material_realization.parquet"]
    polymer = material.xs("polymer", level="product")

    assert len(aggregate) == 34
    assert len(material) == 32
    assert observed.foreground_scenario.eq("constituent_fixture").all()
    assert observed.background_scenario.eq("remind_ssp2_npi").all()
    assert observed.unit.eq("Mt").all()
    assert observed.first_year.eq(2018).all()
    assert observed.last_year.eq(2021).all()
    assert set(aggregate.loc[aggregate.flow_classification.eq("direct_lca_input")].index) == {
        "processing => fabrication",
        "processing => intermediate_market",
        "eol => mismanaged",
    }
    assert aggregate.flow_classification.eq("aggregate_projection").sum() == 14
    assert aggregate.flow_classification.eq("market_report_alias").sum() == 8
    assert aggregate.constituent_axis.eq("not_applicable").all()
    assert aggregate.loc["collected => reclmech", "total_amount"] == pytest.approx(0.16)
    assert polymer.loc["mechanical_treatment", "total_amount"] == pytest.approx(0.16)
    # Source stage yields plus distinct invented preparation and finishing dust.
    recovered = 0.16 * (31199.69 / 33540) * 0.99 * 0.98 * 0.97
    waterborne = 0.16 * (31199.69 / 33540) * 0.01
    airborne = 0.16 * 0.0001 / (2 * 0.8) + recovered * 0.001
    assert polymer.loc["mechanical_resin", "total_amount"] == pytest.approx(recovered)
    assert polymer.loc["mechanical_incinerated_reject", "total_amount"] == pytest.approx(
        0.16 - recovered - waterborne - airborne
    )
    assert polymer.loc["mechanical_landfilled_reject", "total_amount"] == 0
    assert polymer.loc["mechanical_waterborne_loss", "total_amount"] == pytest.approx(waterborne)
    assert polymer.loc["mechanical_airborne_loss", "total_amount"] == pytest.approx(airborne)
    assert polymer.loc["chemical_treatment", "total_amount"] == pytest.approx(0.04)
    gross = 3 * (0.67 / 0.8 + 0.09 / 0.75 + 0.24 / 0.9)
    assert polymer.loc["new_polymer_domestic", "total_amount"] == pytest.approx(gross - recovered)
    assert polymer.loc["manufacturing_principal_input", "total_amount"] == pytest.approx(gross)
    assert polymer.loc["manufacturing_finished_output", "total_amount"] == pytest.approx(3)
    assert polymer.loc["manufacturing_principal_loss", "total_amount"] == pytest.approx(gross - 3)
    assert aggregate.loc["processing => fabrication", "total_amount"] == pytest.approx(3)
    assert aggregate.loc["eol => mismanaged", "total_amount"] == pytest.approx(0.2)
    assert json.loads(polymer.loc["chemical_treatment", "burden_demand_types_json"]) == [
        "chemical_recycling_preparation",
        "chemical_recycling_conversion",
    ]
    for flow in ("mechanical_resin", "chemical_processed_feed", "chemical_preparation_reject"):
        assert polymer.loc[flow, "flow_classification"] == "already_owned"
    assert json.loads(
        polymer.loc["mechanical_landfilled_reject", "provider_demand_types_json"]
    ) == ["landfill_treatment"]
    assert json.loads(aggregate.loc["sysenv => primary_market", "burden_demand_types_json"]) == []
    assert set(material.loc["hydrothermal_product"].index) == {
        "naphtha",
        "distillate_gas_oil",
        "heavy_gas_oil",
        "heavy_wax_residue",
    }
    assert set(material.loc["cracker_product"].index) == {
        "ethylene",
        "propylene",
        "c4",
        "pyrolysis_gas_btx",
        "hydrogen",
        "external_fuel_gas",
        "external_fuel_oil",
    }
    for flow in ("hydrothermal_product", "cracker_product"):
        assert material.loc[flow, "flow_classification"].eq("physical_process_output").all()
        assert material.loc[flow, "constituent_axis"].eq("source").all()
    for flow in ("recovered_monomer_consumption", "virgin_monomer_consumption"):
        assert set(material.loc[flow].index) == {"ethylene", "propylene"}
        assert material.loc[flow, "flow_classification"].eq("monomer_allocation_input").all()
        assert material.loc[flow, "constituent_axis"].eq("destination").all()
        assert material.loc[flow, "mapping_basis"].eq("kg_destination_polymer_output").all()
    zero = aggregate.loc["waste_market => collected"]
    assert zero.nonzero_row_count == 0
    assert zero.persisted_row_count > 0
    assert zero.total_amount == 0
