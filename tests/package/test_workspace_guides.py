from importlib import resources

from pytoce.shared.runtime.workspace_guides import (
    load_workspace_guide_catalogue,
    render_workspace_guide,
)

EXPECTED_GUIDES = {
    "data_raw": (
        "foreground_scenarios/",
        "ecoinvent/",
        "remind_mfa/",
        "pb_lcia/",
        "premise/",
        "allocated_cc/",
        "methodological_notes/",
    ),
    "data_raw/methodological_notes": (
        "LCI_per_unit.xlsx",
        "pytoce_article.pdf",
        "pytoce_methodological_appendix.pdf",
        "recommended_citations.txt",
    ),
    "data_processed": (
        "ecoinvent/",
        "foreground_scenarios/",
        "remind_mfa/",
        "brightway/",
        "prepared_backgrounds/",
        "allocated_cc/",
        "optimization_response_bank/",
    ),
    "results": ("<foreground_scenario>/",),
    "data_raw/foreground_scenarios": (
        "default_inputs/",
        "schemas/",
        "templates/",
        "scenarios/",
    ),
    "data_raw/foreground_scenarios/default_inputs": (
        "parameter_catalogue.json",
        "default_input_values.parquet",
        "default_input_values_manifest.json",
    ),
    "data_raw/foreground_scenarios/schemas": (
        "foreground_scenario_schema_v1.json",
        "compiled_annual_input_parameters_schema_v1.json",
        "mfa_result_schema_v1.json",
    ),
    "data_raw/foreground_scenarios/templates/examples": (
        "01_stock_volume.yaml",
        "02_lifetime.yaml",
        "03_plastic_family_mix.yaml",
        "04_circularity_treatment.yaml",
        "05_virgin_feedstock.yaml",
        "06_waste_trade.yaml",
        "07_socioeconomic_drivers.yaml",
    ),
    "data_raw/foreground_scenarios/scenarios": (
        "remind_mfa_default.yaml",
        "frozen_2019_default_baseline.yaml",
        "<foreground_scenario>.yaml",
    ),
    "data_raw/ecoinvent": (
        "ecoinvent_3.12_cutoff_ecoSpold02.7z",
        "ecoinvent 3.12_LCIA_implementation.7z",
        "lcia_implementation.xlsx",
        "manifest.json",
    ),
    "data_raw/remind_mfa": (
        "remind_mfa_v0_2_0.tar.gz",
        "input_data_1_0_0.zip",
        "manifest.json",
    ),
    "data_processed/ecoinvent": ("material_recipe.json",),
    "data_raw/pb_lcia": (
        "pb_lcia_0_9_2.tar.gz",
        "manifest.json",
    ),
    "data_processed/foreground_scenarios": ("<foreground_scenario>/",),
    "data_processed/foreground_scenarios/<foreground_scenario>": (
        "compiled_annual_input_parameters.parquet",
        "scenario_manifest.json",
    ),
    "data_processed/remind_mfa": ("code/", "data/", "manifest.json"),
    "results/<foreground_scenario>": (
        "mfa_results/",
        "lca_results/",
        "asr_results/",
        "optimization_results/",
    ),
    "results/<foreground_scenario>/mfa_results": (
        "resolved_annual_input_parameters.parquet",
        "derived_annual_outputs.parquet",
        "flows.parquet",
        "stocks.parquet",
        "material_realization.parquet",
        "nominal_feedstock_equivalents.parquet",
        "cohort_state.parquet",
        "cohort_material_identity.parquet",
        "cohort_composition.parquet",
        "physical_reconciliation_ledger.parquet",
        "stock_balance.parquet",
        "transition_ledger.parquet",
        "waste_trade_propensity_ledger.parquet",
        "sankey/",
        "manifest.json",
    ),
    "results/<foreground_scenario>/mfa_results/sankey": (
        "nodes.parquet",
        "edges.parquet",
    ),
    "data_raw/premise": ("<background_scenario>/",),
    "data_raw/premise/<background_scenario>": (
        "<premise_filename>.csv",
        "manifest.json",
    ),
    "data_processed/prepared_backgrounds": ("shared/", "<background_scenario>/"),
    "data_processed/prepared_backgrounds/shared": (
        "impact_categories.parquet",
        "database_authority.parquet",
        "matrix_authority.parquet",
        "biosphere_authority.parquet",
        "provider_service_routes.parquet",
        "material_carbon.parquet",
        "pa_receiving_reference.json",
        "provider_source_authorities.parquet",
        "provider_method_authorities.parquet",
        "USLCI_DATA_USE_NOTICE.txt",
        "provider_service_manifest.json",
        "preparation_report.json",
        "manifest.json",
    ),
    "data_processed/prepared_backgrounds/<background_scenario>": (
        "background_scenarios.parquet",
        "lcia_methods.parquet",
        "impact_categories.parquet",
        "provider_crosswalk.parquet",
        "provider_technology_crosswalk.parquet",
        "provider_semantic_ledger.parquet",
        "pa_receiving_sources.json",
        "database_authority.parquet",
        "matrix_authority.parquet",
        "biosphere_authority.parquet",
        "preparation_report.json",
        "scenario_report_<anchor_years>.xlsx",
        "change_report_<anchor_years>.xlsx",
        "manifest.json",
    ),
    "results/<foreground_scenario>/lca_results": ("<background_and_selection_readable_name>/",),
    "results/<foreground_scenario>/lca_results/<background_and_selection_readable_name>": (
        "scores.parquet",
        "annual_scores.parquet",
        "annual_life_cycle_stage_scores.parquet",
        "life_cycle_stage_scores.parquet",
        "contributions.parquet",
        "stage_contributions.parquet",
        "structural_paths.parquet",
        "structural_edges.parquet",
        "structural_attribution.parquet",
        "inventory_authority.parquet",
        "mechanical_recycling_loss.parquet",
        "mismanaged_waste_fate.parquet",
        "activation_lineage.parquet",
        "direct_treatment_carbon_flows.parquet",
        "pa_receiving_inventory.parquet",
        "pa_receiving_physical_events.parquet",
        "mismanaged_waste_fate_closure.parquet",
        "mfa_to_lca_flow_coverage.parquet",
        "carbon_consistency.csv",
        "execution_report.json",
        "manifest.json",
    ),
    "data_raw/allocated_cc": ("templates/", "results/"),
    "data_raw/allocated_cc/results": ("<allocated_cc_result>/",),
    "data_processed/allocated_cc": ("<allocated_cc_result>/",),
    "data_processed/optimization_response_bank": ("four_remind/",),
    "results/<foreground_scenario>/optimization_results": (
        "<allocated_cc_result>/<optimization_lcia_selection>/<readable_request_selection>__stock_lp_v6/selected_results/joint_lines_v1_stock_lp_v6__<result_lcia_selection>__grid_20pct__refinement_10_5pct/",
    ),
}

EXPECTED_GUIDES["results/<foreground_scenario>/asr_results/<selection>"] = (
    "annual_asr.parquet",
    "cumulative_gwp100_asr.parquet",
    "manifest.json",
)

OWNER_PACKAGES = (
    "pytoce.set_workspace",
    "pytoce.download_data",
    "pytoce.run_mfa",
    "pytoce.prepare_backgrounds",
    "pytoce.run_lca",
    "pytoce.run_asr",
    "pytoce.run_optimization",
)


def test_workspace_guide_catalogues_cover_the_readable_tree_once() -> None:
    entries = tuple(
        (package, guide)
        for package in OWNER_PACKAGES
        for guide in load_workspace_guide_catalogue(package)
    )
    by_path = {str(guide["workspace_path"]): (package, guide) for package, guide in entries}

    assert len(by_path) == len(entries)
    assert set(by_path) == set(EXPECTED_GUIDES)
    for path, expected_contents in EXPECTED_GUIDES.items():
        package, guide = by_path[path]
        described = tuple(
            item["name"]
            for item in guide["expected_contents"]  # type: ignore[index]
        )
        assert described == expected_contents
        rendered = render_workspace_guide(package, str(guide["guide_id"])).decode("utf-8")
        for heading in (
            "## Files",
            "## Inputs",
            "## Format",
            "## Read with Python",
            "## Reuse",
            "## Notes",
        ):
            assert heading in rendered
        assert "`README.md`" in rendered
        compile(str(guide["pandas_example"]), str(guide["workspace_path"]), "exec")
        if package == "pytoce.run_optimization":
            assert "## Provenance" in rendered

    template_guide = (
        resources.files("pytoce.run_mfa.data")
        .joinpath(
            "workspace",
            "data_raw",
            "foreground_scenarios",
            "templates",
            "README_foreground_scenarios.md",
        )
        .read_text(encoding="utf-8")
    )
    for required in (
        "foreground_scenario.yaml",
        "examples/",
        "default_annual_input_values.csv",
        "parameter_catalogue.json",
        "import pandas as pd",
        'run_mfa(foreground_scenario="your_name", refresh=True)',
        "## Scientific limitations",
    ):
        assert required in template_guide
