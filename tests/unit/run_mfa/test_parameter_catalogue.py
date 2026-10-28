from pathlib import Path

import pytest
import yaml

from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue
from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.run_mfa.scenario.scenario_contract import ScenarioError
from pytoce.set_workspace.set_workspace import set_workspace


def test_catalogue_has_one_role_and_exact_upstream_controls() -> None:
    definitions = load_parameter_catalogue()
    assert isinstance(definitions, tuple)
    by_id = {item.parameter_id: item for item in definitions}
    assert len(by_id) == len(definitions)

    expected = {
        "collection_rate": ("collection_rate", ("year", "region")),
        "mechanical_recycling_rate": (
            "mechanical_recycling_rate",
            ("year", "region"),
        ),
        "chemical_recycling_rate": (
            "chemical_recycling_rate",
            ("year", "region"),
        ),
        "incineration_rate": ("incineration_rate", ("year", "region")),
        "plastic_family_share": (
            "material_shares_in_goods",
            ("year", "region", "material", "good"),
        ),
        "lifetime_mean": ("lifetime_mean", ("year", "good")),
    }
    for parameter_id, (upstream_name, dimensions) in expected.items():
        definition = by_id[parameter_id]
        assert definition.role == "editable_input"
        assert definition.upstream_name == upstream_name
        assert definition.dimensions == dimensions
        assert definition.recomputation_order
        assert definition.invalidated_descendants
        assert definition.upstream_source

    assert by_id["new_polymer_production"].role == "derived_output"
    assert by_id["lifetime_std"].role == "derived_input"
    assert by_id["lifetime_std"].upstream_name == "lifetime_std"
    assert by_id["lifetime_std"].dimensions == ("year", "good")
    assert by_id["lifetime_std"].value_modes == ()
    assert by_id["new_polymer_production"].value_modes == ()
    assert by_id["historical_consumption"].role == "fixed_historical_boundary"
    assert by_id["background_scenario"].role == "background_only_input"
    assert by_id["reuse_rate"].role == "unsupported"
    assert by_id["chemical_recycling_yield"].role == "unsupported"
    assert by_id["emission_capture_rate"].role == "unsupported"
    deferred_mode = "retain_2019_propensity_pretrade_collected"
    assert deferred_mode in by_id["waste_imports"].value_modes
    assert deferred_mode in by_id["waste_exports"].value_modes
    assert all(
        deferred_mode not in definition.value_modes
        for definition in definitions
        if definition.parameter_id not in {"waste_imports", "waste_exports"}
    )


def test_catalogue_exposes_future_optimization_metadata_without_optimizer() -> None:
    definitions = load_parameter_catalogue()
    assert isinstance(definitions, tuple)
    groups = {item.optimization_group for item in definitions if item.role == "editable_input"}
    assert groups == {
        "volume",
        "lifetime",
        "plastic_family_mix",
        "circularity_treatment",
        "virgin_feedstock",
        "waste_trade",
        "socioeconomic_drivers",
    }
    assert all(item.minimum is not None for item in definitions if item.editable)


@pytest.mark.parametrize(
    "parameter", ("mechanical_recycling_yield", "reclmech_loss_uncontrolled_rate")
)
def test_scenario_admission_rejects_obsolete_mechanical_controls(
    tmp_path: Path, parameter: str
) -> None:
    """Reject obsolete user overrides instead of ignoring fixed recipe quantities."""
    workspace = set_workspace(tmp_path)
    scenario = "obsolete_recovery_control"
    payload = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": scenario,
        "ssp_scenario": "SSP2",
        "changes": [
            {
                "parameter": parameter,
                "region": "EUR",
                "material": "HDPE",
                "changes": [{"year": 2020, "mode": "absolute", "value": 0.5}],
            }
        ],
    }
    (workspace.foreground_scenarios / f"{scenario}.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )
    with pytest.raises(ScenarioError, match=f"Unknown foreground parameter '{parameter}'"):
        compile_foreground_scenario(tmp_path, scenario)
