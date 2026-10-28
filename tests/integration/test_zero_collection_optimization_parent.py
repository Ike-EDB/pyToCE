"""Initialize optimization from a valid zero-collection MFA."""

from math import fsum
from pathlib import Path

import pandas as pd
import pytest
import yaml
from mfa_workspace_fixtures import prepare_mfa_workspace

from pytoce import set_workspace
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    read_prepared_manifest,
)
from pytoce.run_lca.calculation.foreground import compile_optimization_activation_context
from pytoce.run_lca.calculation.providers import load_provider_routes
from pytoce.run_mfa.model.mfa import mfa_evaluation_session_identity
from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue
from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.run_mfa.scenario.parent_relative import compiled_scenario_table_identity
from pytoce.run_mfa.scenario.scenario_contract import (
    ScenarioEvaluation,
    ScenarioFeasibilityError,
)
from pytoce.run_optimization.calculation.candidate_authority import (
    CandidateEvaluationError,
    build_candidate_parent_authority,
)
from tests.integration.candidate_mfa_fixtures import _provider_tables
from tests.unit.run_optimization.allocated_cc_fixtures import _methods


@pytest.mark.scientific_integration
def test_zero_collection_parent_retains_unobserved_native_treatment_controls(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> None:
    """A parent MFA result with unobserved treatment controls cannot initialize search."""
    workspace = tmp_path / "workspace"
    foreground = "zero_collection_parent"
    set_workspace(workspace)
    # Zero collected mass requires zero prescribed waste trade. Omitted rules retain absolute
    # defaults. Keep the fixture's virgin-feed and treatment controls; their rates act on zero
    # collected mass.
    controls = {
        "collection_rate": 0.0,
        "waste_imports": 0.0,
        "waste_exports": 0.0,
        "bio_production_rate": 0.05,
        "daccu_production_rate": 0.0,
    }
    scenario_path = workspace / "data_raw/foreground_scenarios/scenarios" / f"{foreground}.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "pytoce_foreground_scenario_schema_v1",
                "foreground_scenario": foreground,
                "ssp_scenario": "SSP2",
                "changes": [
                    {
                        "parameter": parameter,
                        "regions": list(load_parameter_catalogue(parameter).coordinates["region"]),
                        "changes": [{"year": 2020, "mode": "absolute", "value": value}],
                    }
                    for parameter, value in controls.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    # The existing fixture supplies real source-matched stock, material and
    # lifetime controls before its first public MFA calculation. No completed
    # parent, native parameter table or material-balance result is edited.
    parent = prepare_mfa_workspace(
        workspace, remind_mfa_source_root, foreground_scenario=foreground
    )
    compiled = compile_foreground_scenario(workspace, foreground)
    parameters = pd.read_parquet(compiled.compiled_parameters_path)
    scenario = ScenarioEvaluation(
        foreground,
        compiled.foreground_scenario_identity,
        "SSP2",
        parameters,
        compiled.normalized_change_points,
        compiled_scenario_table_identity(parameters),
    )
    _provider_tables(workspace, _methods())
    routes = load_provider_routes(
        read_prepared_manifest(
            workspace / "data_processed/prepared_backgrounds/shared/manifest.json"
        )
    )
    activations = compile_optimization_activation_context(parent.mfa, routes, parent.recipe)
    session_identity = mfa_evaluation_session_identity(
        parent.request,
        load_parameter_catalogue(),
        material_recipe_identity=parent.recipe.identity,
        material_composition_identity=parent.composition.identity,
    )
    with pytest.raises(CandidateEvaluationError, match="cannot be represented") as failure:
        build_candidate_parent_authority(
            scenario,
            mfa_result_identity=str(parent.reference.manifest["identity"]),
            mfa_result_manifest_sha256=parent.reference.sha256,
            parent_mfa=parent.mfa,
            candidate_mfa_session_identity=session_identity,
            provider_routes=routes,
            activation_context=activations,
        )
    cause = failure.value.__cause__
    assert isinstance(cause, ScenarioFeasibilityError)
    assert cause.code == "optimization_parent_not_representable"
    assert cause.context["family"] == "treatment"
    year, region = str(cause.context["coordinate"]).split(":")
    treatment = parameters.loc[
        parameters["year"].eq(int(year))
        & parameters["region"].eq(region)
        & parameters["parameter_id"].isin(
            ("mechanical_recycling_rate", "chemical_recycling_rate", "incineration_rate")
        )
    ]
    assert treatment["value_origin"].eq("remind_mfa_default").all()
    assert fsum(treatment["resolved_value"]) > 1.0
    waste = parent.mfa.material_realization.loc[
        parent.mfa.material_realization["year"].eq(int(year))
        & parent.mfa.material_realization["region"].eq(region)
        & parent.mfa.material_realization["flow_name"].eq("collected_after_trade")
    ]
    assert not waste.empty
    assert waste["amount"].eq(0.0).all()
