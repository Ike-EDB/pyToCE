import json
from pathlib import Path

import pytest

from pytoce.set_workspace.set_workspace import set_workspace

_REPORT_FIELDS = (
    "status",
    "workspace",
    "elapsed_seconds",
    "data_raw",
    "data_processed",
    "results",
    "parameter_catalogue",
    "default_input_parameters",
    "default_annual_input_values",
    "foreground_scenario_templates",
    "foreground_scenarios",
    "allocated_cc_templates",
    "allocated_cc_results",
    "methodological_notes",
    "recommended_citations",
)


def test_set_workspace_creates_exact_roots_without_metadata_tree(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    first = set_workspace(workspace)
    second = set_workspace(workspace)

    assert {path.name for path in workspace.iterdir()} == {
        "data_raw",
        "data_processed",
        "results",
    }
    assert first.status == "computed"
    assert second.status == "reused"
    assert first.parameter_catalogue.is_file()
    assert first.default_input_parameters.is_file()
    assert first.default_annual_input_values.is_file()
    assert tuple(first.as_dict()) == _REPORT_FIELDS
    assert first.allocated_cc_templates == workspace / "data_raw/allocated_cc/templates"
    assert first.allocated_cc_results == (workspace / "data_raw/allocated_cc/results")
    assert {path.name for path in first.allocated_cc_templates.iterdir() if path.is_file()} == {
        "README.md",
        "allocated_cc_result.yaml",
        "annual_allocated_cc.csv",
        "cumulative_gwp100_allocated_cc.csv",
    }
    assert first.allocated_cc_results.is_dir()
    assert (workspace / "data_processed/allocated_cc/README.md").is_file()
    assert (workspace / "data_processed/optimization_response_bank/README.md").is_file()
    assert first.as_dict()["allocated_cc_templates"] == str(first.allocated_cc_templates)
    assert first.as_dict()["allocated_cc_results"] == str(first.allocated_cc_results)
    assert first.methodological_notes == workspace / "data_raw/methodological_notes"
    assert first.recommended_citations == (first.methodological_notes / "recommended_citations.txt")
    assert first.recommended_citations.is_file()
    assert (first.methodological_notes / "pytoce_article.pdf").is_file()
    assert (first.methodological_notes / "pytoce_methodological_appendix.pdf").is_file()
    assert (workspace / "data_raw/allocated_cc/schemas").is_dir()
    assert (first.foreground_scenarios / "remind_mfa_default.yaml").is_file()
    assert (first.foreground_scenarios / "frozen_2019_default_baseline.yaml").is_file()
    assert not (first.foreground_scenarios / "frozen_2019_circularity.yaml").exists()
    assert not (
        first.foreground_scenarios / "frozen_2019_circularity_default_material_demand.yaml"
    ).exists()
    assert not (workspace / "data_processed" / "shared_authorities").exists()
    assert not list(workspace.rglob("selected_authority.json"))
    assert not list(workspace.rglob("summary.log"))
    catalogue = json.loads(first.parameter_catalogue.read_text(encoding="utf-8"))
    defaults_manifest = json.loads(
        (first.default_input_parameters.parent / "default_input_values_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert defaults_manifest["parameter_catalogue_identity"] == catalogue["catalogue_identity"]


def test_refresh_preserves_user_files_and_restores_package_inputs(
    tmp_path: Path,
) -> None:
    report = set_workspace(tmp_path)
    guide = report.foreground_scenario_templates / "README_foreground_scenarios.md"
    scenario_guide = report.foreground_scenarios / "README.md"
    allocated_cc_input_guide = report.allocated_cc_results / "README.md"
    example = report.foreground_scenario_templates / "examples" / "01_stock_volume.yaml"
    lookup = report.default_annual_input_values
    packaged_scenario = report.foreground_scenarios / "remind_mfa_default.yaml"
    user_scenario = report.foreground_scenarios / "user_case.yaml"
    user_template = report.foreground_scenario_templates / "user_notes.yaml"
    catalogue = report.parameter_catalogue
    allocated_cc_template = report.allocated_cc_templates / "allocated_cc_result.yaml"
    allocated_cc_guide = report.data_processed / "allocated_cc" / "README.md"
    lci_workbook = report.methodological_notes / "LCI_per_unit.xlsx"
    accepted_lci_workbook = lci_workbook.read_bytes()
    lci_workbook.write_bytes(b"user documentation notes")
    set_workspace(tmp_path)
    assert lci_workbook.read_bytes() == b"user documentation notes"
    user_result = report.allocated_cc_results / "user_case"
    user_result.mkdir()
    user_bundle = user_result / "allocated_cc_result.yaml"
    user_bundle.write_text("user-owned\n", encoding="utf-8")
    user_allocated_template = report.allocated_cc_templates / "user_notes.csv"
    user_allocated_template.write_text("user-owned template\n", encoding="utf-8")
    accepted_catalogue = catalogue.read_bytes()
    accepted_guide = guide.read_bytes()
    accepted_scenario_guide = scenario_guide.read_bytes()
    accepted_allocated_cc_input_guide = allocated_cc_input_guide.read_bytes()
    accepted_example = example.read_bytes()
    accepted_lookup = lookup.read_bytes()
    accepted_allocated_cc_template = allocated_cc_template.read_bytes()
    accepted_allocated_cc_guide = allocated_cc_guide.read_bytes()
    guide.write_text("user guide notes\n", encoding="utf-8")
    scenario_guide.write_text("changed package guide\n", encoding="utf-8")
    allocated_cc_input_guide.write_text("changed package guide\n", encoding="utf-8")
    example.write_text("user example\n", encoding="utf-8")
    lookup.write_text("user lookup notes\n", encoding="utf-8")
    packaged_scenario.write_text("user scenario notes\n", encoding="utf-8")
    user_scenario.write_text("user-owned\n", encoding="utf-8")
    user_template.write_text("user-owned template\n", encoding="utf-8")
    catalogue.write_text("changed package input\n", encoding="utf-8")
    allocated_cc_template.write_text("changed package template\n", encoding="utf-8")
    allocated_cc_guide.write_text("changed package guide\n", encoding="utf-8")
    refreshed = set_workspace(tmp_path, refresh=True)

    assert guide.read_bytes() == accepted_guide
    assert scenario_guide.read_bytes() == accepted_scenario_guide
    assert allocated_cc_input_guide.read_bytes() == accepted_allocated_cc_input_guide
    assert example.read_bytes() == accepted_example
    assert lookup.read_bytes() == accepted_lookup
    assert packaged_scenario.read_text(encoding="utf-8") != "user scenario notes\n"
    assert "foreground_scenario: remind_mfa_default" in packaged_scenario.read_text(
        encoding="utf-8"
    )
    assert user_scenario.read_text(encoding="utf-8") == "user-owned\n"
    assert user_template.read_text(encoding="utf-8") == "user-owned template\n"
    assert catalogue.read_bytes() == accepted_catalogue
    assert allocated_cc_template.read_bytes() == accepted_allocated_cc_template
    assert allocated_cc_guide.read_bytes() == accepted_allocated_cc_guide
    assert lci_workbook.read_bytes() == accepted_lci_workbook
    assert user_bundle.read_text(encoding="utf-8") == "user-owned\n"
    assert user_allocated_template.read_text(encoding="utf-8") == "user-owned template\n"
    assert refreshed.status == "computed"


def test_invalid_workspace_fails_at_public_boundary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        set_workspace(" ")

    occupied = tmp_path / "occupied"
    occupied.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        set_workspace(occupied)
