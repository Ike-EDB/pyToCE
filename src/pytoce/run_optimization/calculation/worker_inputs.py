"""Load the package results and source files used by optimization workers."""

import json
from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from pytoce.download_data.sources.ecoinvent_materials import (
    EcoinventMaterialRecipe,
    load_material_recipe,
)
from pytoce.prepare_backgrounds.preparation.premise import ACTIVE_BACKGROUND_SCENARIOS
from pytoce.prepare_backgrounds.preparation.prepared_state.authority_tables import (
    read_prepared_manifest,
)
from pytoce.run_lca.calculation.foreground import (
    OptimizationActivationContext,
    compile_optimization_activation_context,
)
from pytoce.run_lca.calculation.lcia import LCIAMethodSelection
from pytoce.run_lca.calculation.providers import (
    ProviderRoutes,
    load_prepared_provider_product,
    load_provider_routes,
)
from pytoce.run_mfa.model.materials import MaterialComposition, load_material_composition
from pytoce.run_mfa.model.mfa import (
    MFAEvaluation,
    mfa_evaluation_session_identity_from_source,
)
from pytoce.run_mfa.model.result_inputs import (
    MFAResultReference,
    read_mfa_result,
    read_mfa_result_manifest,
)
from pytoce.run_mfa.remind_mfa.execution import (
    RemindMFAExecutionRequest,
    prepare_execution,
    prepare_workspace_source,
)
from pytoce.run_mfa.scenario.catalogue import (
    ParameterDefinition,
    load_parameter_catalogue,
)
from pytoce.run_mfa.scenario.compiler import _normalized_change_points
from pytoce.run_mfa.scenario.scenario_contract import ScenarioEvaluation
from pytoce.run_optimization.calculation.candidate_authority import (
    CandidateParentAuthority,
    build_candidate_parent_authority,
)
from pytoce.run_optimization.calculation.candidate_projection import (
    build_response_parent_score_projection,
)
from pytoce.run_optimization.calculation.candidate_projection_arrays import (
    CandidateParentScoreProjection,
)
from pytoce.run_optimization.calculation.pa_candidate_projection import (
    PaCandidateProjection,
    PaCandidateScores,
    build_pa_candidate_projection,
    calculate_pa_candidate_projection,
)
from pytoce.run_optimization.calculation.response_bank import (
    project_optimization_response_bank,
    read_optimization_response_bank,
    select_response_bank_method_registry,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    LoadedOptimizationResponseBank,
    OptimizationResponseBank,
    ResponseBankProjection,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    CANDIDATE_WORKER_INITIALIZER_SCHEMA,
    CANDIDATE_WORKER_METHOD_IDENTITY,
    _arrow_frame,
    _arrow_frame_bytes,
    _canonical_json_bytes,
    _float64_bytes,
)
from pytoce.run_optimization.input.allocated_cc_contract import (
    AllocatedCCResult,
)
from pytoce.shared.runtime.filesystem import payload_sha256

_WORKER_AUTHORITY_PREFIX = "pytoce_optimization_candidate_worker_authority_v3::"
_BACKGROUNDS = tuple(item.background_scenario for item in ACTIVE_BACKGROUND_SCENARIOS)


@dataclass(frozen=True, slots=True)
class _PreparedPhysicalInputs:
    """Store the MFA inputs shared by one optimization worker.

    Attributes:
        payload: Decoded initializer fields and recorded input identities.
        root: Active workspace directory.
        scenario: Completed compiled foreground scenario.
        parameter_catalogue: Foreground parameter definitions used by candidate MFA.
        mfa_request: REMIND-MFA 0.2.0 code, input data 1.0.0, and configuration request.
        recipe: Admitted native unallocated and allocated monomer recipe.
        composition: Packaged constituent composition used by the saved parent.
        parent_mfa: Completed baseline MFA tables and MFA state identity.
        routes: Prepared provider routes used to construct LCA activations.
        optimization_activations: Baseline provider activations and their identity.
        physical_parent: Baseline tables used to build candidate changes.
    """

    payload: Mapping[str, object]
    root: Path
    scenario: ScenarioEvaluation
    parameter_catalogue: tuple[ParameterDefinition, ...]
    mfa_request: RemindMFAExecutionRequest
    recipe: EcoinventMaterialRecipe
    composition: MaterialComposition
    parent_mfa: MFAEvaluation
    routes: ProviderRoutes
    optimization_activations: OptimizationActivationContext
    physical_parent: CandidateParentAuthority


@dataclass(frozen=True, slots=True)
class _PreparedSearchInputs:
    """The shared inputs used by every optimization search."""

    physical: _PreparedPhysicalInputs
    response_bank: LoadedOptimizationResponseBank
    method_registry: pd.DataFrame
    optimization_parent: CandidateParentScoreProjection
    climate_parent: CandidateParentScoreProjection
    optimization_projection: ResponseBankProjection
    climate_projection: ResponseBankProjection
    pa_projection: PaCandidateProjection
    pa_parent: PaCandidateScores


def _prepare_physical_inputs(
    initializer_payload: bytes,
) -> _PreparedPhysicalInputs:
    """Load the shared MFA inputs for one worker."""

    payload = _initializer_payload(initializer_payload)
    root = Path(cast(str, payload["workspace"]))
    return _load_physical_inputs(payload, root, None)


def _prepare_parent_physical_inputs(
    initializer_payload: bytes,
    parent_mfa: MFAEvaluation,
) -> _PreparedPhysicalInputs:
    """Use the MFA inputs already loaded by the main workflow."""

    payload = _initializer_payload(initializer_payload)
    root = Path(cast(str, payload["workspace"]))
    return _load_physical_inputs(payload, root, parent_mfa)


def _load_physical_inputs(
    payload: Mapping[str, object],
    root: Path,
    parent_mfa: MFAEvaluation | None,
) -> _PreparedPhysicalInputs:
    """Load the common MFA inputs from a decoded worker initializer."""

    scenario = _load_parent_scenario(root, payload)
    catalogue = cast(tuple[ParameterDefinition, ...], load_parameter_catalogue())
    mfa_request = prepare_execution(prepare_workspace_source(root))
    recipe = load_material_recipe(root)
    composition = load_material_composition()
    if parent_mfa is None:
        manifest_path = (
            root / "results" / scenario.foreground_scenario / "mfa_results/manifest.json"
        )
        parent_mfa = read_mfa_result(read_mfa_result_manifest(manifest_path))
    routes = load_provider_routes(
        read_prepared_manifest(root / "data_processed/prepared_backgrounds/shared/manifest.json")
    )
    optimization_activations = compile_optimization_activation_context(
        parent_mfa,
        routes,
        recipe,
    )
    physical_parent = build_candidate_parent_authority(
        scenario,
        mfa_result_identity=cast(str, payload["mfa_result_identity"]),
        mfa_result_manifest_sha256=cast(str, payload["mfa_result_manifest_sha256"]),
        parent_mfa=parent_mfa,
        candidate_mfa_session_identity=cast(str, payload["candidate_mfa_session_identity"]),
        provider_routes=routes,
        activation_context=optimization_activations,
    )
    return _PreparedPhysicalInputs(
        payload=payload,
        root=root,
        scenario=scenario,
        parameter_catalogue=catalogue,
        mfa_request=mfa_request,
        recipe=recipe,
        composition=composition,
        parent_mfa=parent_mfa,
        routes=routes,
        optimization_activations=optimization_activations,
        physical_parent=physical_parent,
    )


def prepare_candidate_worker_initializer(
    workspace: Path,
    *,
    foreground_scenario: str,
    mfa_reference: MFAResultReference,
    allocated_cc: AllocatedCCResult,
    allocated_cc_case: tuple[str, ...],
    response_method_union: str = "ef31_pb_lcia_response_v3",
    background_scenarios: tuple[str, ...] = _BACKGROUNDS,
) -> bytes:
    """Bind the selected MFA snapshot to compiled scenario, response and aCC parents."""

    root = workspace
    scenario_directory = root / "data_processed" / "foreground_scenarios" / foreground_scenario
    scenario_manifest_path = scenario_directory / "scenario_manifest.json"
    scenario_content = scenario_manifest_path.read_bytes()
    scenario = json.loads(scenario_content)
    scenario_manifest_sha256 = sha256(scenario_content).hexdigest()
    mfa, mfa_manifest_sha256 = mfa_reference.manifest, mfa_reference.sha256
    mfa_identity_payload = cast(dict[str, object], mfa["identity_payload"])
    response_directory = (
        root
        / "data_processed"
        / "optimization_response_bank"
        / "four_remind"
        / response_method_union
    )
    response_manifest_path = response_directory / "manifest.json"
    response_content = response_manifest_path.read_bytes()
    response = json.loads(response_content)
    response_manifest_sha256 = sha256(response_content).hexdigest()
    response_inputs = response["identity_payload"]
    # Receiving quantities depend on PA reference tables and dated source admission,
    # even when the unchanged parent has no PA events to reveal that dependency.
    prepared_receiving_identity = payload_sha256(
        {
            "shared": response_inputs["prepared_background_shared_identity"],
            "members": {
                parent["background_scenario"]: parent["prepared_background_identity"]
                for parent in response_inputs["member_parents"]
                if parent["background_scenario"] in background_scenarios
            },
        }
    )

    remind_mfa_identity = cast(
        Mapping[str, str],
        mfa_identity_payload["remind_mfa"],
    )
    catalogue = cast(tuple[ParameterDefinition, ...], load_parameter_catalogue())
    payload: dict[str, object] = {
        "schema_version": CANDIDATE_WORKER_INITIALIZER_SCHEMA,
        "workspace": str(root),
        "foreground_scenario": foreground_scenario,
        "compiled_scenario_identity": scenario["identity"],
        "compiled_scenario_manifest_sha256": scenario_manifest_sha256,
        "mfa_result_identity": mfa["identity"],
        "mfa_result_manifest_sha256": mfa_manifest_sha256,
        "candidate_mfa_session_identity": mfa_evaluation_session_identity_from_source(
            remind_mfa_identity,
            catalogue,
            material_recipe_identity=cast(str, mfa_identity_payload["material_recipe_identity"]),
            material_composition_identity=cast(
                str, mfa_identity_payload["material_composition_identity"]
            ),
        ),
        "response_method_union": response_method_union,
        "background_scenarios": list(background_scenarios),
        "response_bank_identity": response["response_bank_identity"],
        "response_bank_manifest_sha256": response_manifest_sha256,
        "prepared_receiving_identity": prepared_receiving_identity,
        "allocated_cc_result": allocated_cc.allocated_cc_result,
        "allocated_cc_result_identity": allocated_cc.allocated_cc_result_identity,
        "allocated_cc_case": list(allocated_cc_case),
    }
    return _canonical_json_bytes(payload)


def candidate_worker_authority_identity(
    payload: Mapping[str, object], *, optimization_projection_identity: str
) -> str:
    """Bind MFA/aCC inputs and actual EF responses, independent of reporting.

    The admitted EF projection includes the mandatory climate category and
    binds provider/native responses, direct factors, units and temporal weights.
    Prepared ancestry binds the receiving quantities before characterization.
    The containing bank may also supply reporting-only methods.
    """

    return _WORKER_AUTHORITY_PREFIX + payload_sha256(
        {
            "method_identity": "pytoce_optimization_candidate_worker_authority_v3",
            "candidate_worker_method_identity": CANDIDATE_WORKER_METHOD_IDENTITY,
            "foreground_scenario": payload["foreground_scenario"],
            "compiled_scenario_identity": payload["compiled_scenario_identity"],
            "compiled_scenario_manifest_sha256": payload["compiled_scenario_manifest_sha256"],
            "mfa_result_identity": payload["mfa_result_identity"],
            "mfa_result_manifest_sha256": payload["mfa_result_manifest_sha256"],
            "candidate_mfa_session_identity": payload["candidate_mfa_session_identity"],
            "optimization_projection_identity": optimization_projection_identity,
            "prepared_receiving_identity": payload["prepared_receiving_identity"],
            "allocated_cc_result": payload["allocated_cc_result"],
            "allocated_cc_result_identity": payload["allocated_cc_result_identity"],
            "allocated_cc_case": payload["allocated_cc_case"],
        }
    )


def _prepare_search_inputs(initializer_payload: bytes) -> _PreparedSearchInputs:
    """Read shared parents once and prepare the two search projections."""

    physical = _prepare_physical_inputs(initializer_payload)
    parent = cast(dict[str, object], physical.payload["pa_parent"])
    shape = tuple(cast(list[int], parent["shape"]))
    scores = PaCandidateScores(
        np.frombuffer(b64decode(cast(str, parent["scores"])), dtype="<f8").reshape(shape),
        np.frombuffer(b64decode(cast(str, parent["norms"])), dtype="<f8").reshape(shape),
        _arrow_frame(b64decode(cast(str, parent["stages"]))),
    )
    return _load_search_inputs(physical, scores)


def _prepare_parent_search_inputs(
    initializer_payload: bytes,
    parent_mfa: MFAEvaluation,
) -> _PreparedSearchInputs:
    """Prepare main-process projections from the already loaded MFA inputs."""

    return _load_search_inputs(
        _prepare_parent_physical_inputs(initializer_payload, parent_mfa), None
    )


def _load_search_inputs(
    physical: _PreparedPhysicalInputs, pa_parent: PaCandidateScores | None
) -> _PreparedSearchInputs:
    """Build search projections from the common worker inputs."""

    loaded_bank = _load_response_bank(physical.root, physical.payload)
    union = loaded_bank.method_registry
    optimization_registry = select_response_bank_method_registry(
        union,
        LCIAMethodSelection(groups=("ef31",), exact_method=()),
    )
    optimization_projection = project_optimization_response_bank(
        loaded_bank, optimization_registry, role="optimization"
    )
    climate_projection = project_optimization_response_bank(
        loaded_bank, _climate_registry(union), role="mandatory_climate"
    )
    prepared = physical.root / "data_processed/prepared_backgrounds"
    shared = read_prepared_manifest(prepared / "shared/manifest.json")
    providers = {
        background: load_prepared_provider_product(
            shared, read_prepared_manifest(prepared / background / "manifest.json")
        )
        for background in cast(list[str], physical.payload["background_scenarios"])
    }
    pa_projection = build_pa_candidate_projection(
        optimization_projection,
        providers,
        physical.routes.open_dump_shares(),
        prepared_receiving_identity=cast(str, physical.payload["prepared_receiving_identity"]),
    )
    # Only the main request evaluates the parent's complete receiving graph.
    # Workers receive its arrays and sparse stage table in their initializer.
    if pa_parent is None:
        parent = physical.optimization_activations
        pa_parent = calculate_pa_candidate_projection(
            pa_projection,
            spinning_events=parent.pa_spinning_events,
            recycling_events=parent.pa_recycling_events,
            disposal_events=parent.pa_disposal_events,
        )
    return _PreparedSearchInputs(
        physical=physical,
        response_bank=loaded_bank,
        method_registry=union,
        optimization_parent=build_response_parent_score_projection(
            physical.physical_parent,
            optimization_projection,
            pa_parent=pa_parent,
            pa_method_registry=optimization_registry,
        ),
        climate_parent=build_response_parent_score_projection(
            physical.physical_parent,
            climate_projection,
            pa_parent=pa_parent,
            pa_method_registry=optimization_registry,
        ),
        optimization_projection=optimization_projection,
        climate_projection=climate_projection,
        pa_projection=pa_projection,
        pa_parent=pa_parent,
    )


def _initializer_with_pa_parent(payload: bytes, parent: PaCandidateScores) -> bytes:
    """Send the once-calculated parent to workers using existing array/table codecs."""
    fields = _initializer_payload(payload)
    fields["pa_parent"] = {
        "shape": list(parent.annual_scores.shape),
        "scores": b64encode(_float64_bytes(parent.annual_scores)).decode("ascii"),
        "norms": b64encode(_float64_bytes(parent.annual_contribution_norms)).decode("ascii"),
        "stages": b64encode(_arrow_frame_bytes(parent.stage_scores)).decode("ascii"),
    }
    return _canonical_json_bytes(fields)


def _load_parent_scenario(root: Path, payload: Mapping[str, object]) -> ScenarioEvaluation:
    """Load the foreground scenario from its completed files."""
    directory = (
        root / "data_processed" / "foreground_scenarios" / cast(str, payload["foreground_scenario"])
    )
    manifest = json.loads((directory / "scenario_manifest.json").read_bytes())
    table = pd.read_parquet(directory / "compiled_annual_input_parameters.parquet")
    compiled_table = cast(Mapping[str, object], manifest["compiled_table"])
    return ScenarioEvaluation(
        foreground_scenario=cast(str, payload["foreground_scenario"]),
        foreground_scenario_identity=cast(str, payload["compiled_scenario_identity"]),
        ssp_scenario="SSP2",
        compiled_parameters=table,
        normalized_change_points=_normalized_change_points(manifest),
        compiled_table_sha256=cast(str, compiled_table["canonical_sha256"]),
    )


def _load_response_bank(
    root: Path,
    payload: Mapping[str, object],
) -> LoadedOptimizationResponseBank:
    """Read completed response tables for one worker."""

    name = cast(str, payload["response_method_union"])
    result = OptimizationResponseBank(
        status="reused",
        identity=cast(str, payload["response_bank_identity"]),
        response_method_union=name,
        result_directory=(
            root / "data_processed" / "optimization_response_bank" / "four_remind" / name
        ),
    )
    return read_optimization_response_bank(
        result, background_scenarios=cast(list[str], payload["background_scenarios"])
    )


def _climate_registry(union: pd.DataFrame) -> pd.DataFrame:
    """Select the EF v3.1 climate-change method required for cumulative ASR."""
    return union.loc[
        union["method_id"].eq("ef31") & union["impact_category_id"].eq("climate_change")
    ].reset_index(drop=True)


def _initializer_payload(payload: bytes) -> dict[str, object]:
    """Decode the initializer created immediately before worker startup."""

    return cast(dict[str, object], json.loads(payload.decode("utf-8")))
