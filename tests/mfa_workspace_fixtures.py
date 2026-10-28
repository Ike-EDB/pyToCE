"""Complete disposable MFA parents for public LCA and optimization boundary tests."""

import json
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.io as pio
import yaml
from material_recipe_fixtures import material_source_members
from remind_mfa_source_fixtures import write_remind_mfa_inputs

from pytoce import run_mfa, set_workspace
from pytoce.download_data.sources.catalogue import load_catalogue
from pytoce.download_data.sources.ecoinvent_materials import (
    EcoinventMaterialRecipe,
    derive_material_recipe,
)
from pytoce.run_mfa.model.cohorts import fixed_lifetime_stock_system
from pytoce.run_mfa.model.materials import (
    MaterialComposition,
    load_material_composition,
)
from pytoce.run_mfa.model.mfa import (
    MFAComputation,
    MFAEvaluation,
    MFAEvaluationSession,
)
from pytoce.run_mfa.model.result_inputs import (
    MFAResultReference,
    read_mfa_result,
    read_mfa_result_manifest,
)
from pytoce.run_mfa.model.results import (
    MFAResultTarget,
    build_mfa_result_target,
    write_mfa_result,
)
from pytoce.run_mfa.remind_mfa.execution import (
    RemindMFAExecutionRequest,
    RemindMFAExecutionSession,
    prepare_execution,
    prepare_workspace_source,
)
from pytoce.run_mfa.remind_mfa.translation import NormalizedCohortState
from pytoce.run_mfa.scenario.catalogue import _catalogue_payload, load_parameter_catalogue
from pytoce.run_mfa.scenario.compiler import (
    CompiledForegroundScenario,
    _compile_payload,
    _manifest,
)
from pytoce.run_mfa.scenario.rules import _ParsedRules
from pytoce.shared.runtime.filesystem import canonical_json_bytes
from pytoce.shared.runtime.workspace_guides import write_workspace_guide

_STOCK_FLOOR_NEW_CLOSING_MT = 0.21987211151099473


@dataclass(frozen=True)
class CompletedMFAFixture:
    """One published and admitted parent, with matching candidate input authorities."""

    request: RemindMFAExecutionRequest
    recipe: EcoinventMaterialRecipe
    composition: MaterialComposition
    reference: MFAResultReference
    mfa: MFAEvaluation


@dataclass(frozen=True)
class StoredMFAFixture:
    """A small real MFA product and the input owners needed to exercise its lifecycle."""

    request: RemindMFAExecutionRequest
    recipe: EcoinventMaterialRecipe
    composition: MaterialComposition
    compiled: CompiledForegroundScenario
    session: MFAEvaluationSession
    target: MFAResultTarget
    computation: MFAComputation


def prepare_stored_mfa(
    workspace: Path, source_root: Path, *, foreground_scenario: str = "direct_fixture"
) -> StoredMFAFixture:
    """Publish two-region, two-polymer science using real owners and invented inputs."""
    set_workspace(workspace)
    renderer = pio.renderers.default
    try:
        request = write_remind_mfa_inputs(
            source_root,
            workspace / "data_raw/remind_mfa" / foreground_scenario,
            regions=("EUR", "USA"),
            materials=("HDPE", "PP"),
        )
    finally:
        pio.renderers.default = renderer
    catalogue = tuple(
        item for item in load_parameter_catalogue() if item.parameter_id == "collection_rate"
    )
    table = pd.MultiIndex.from_product(
        [range(2019, 2101), ["EUR", "USA"]], names=["year", "region"]
    ).to_frame(index=False)
    table = table.assign(
        parameter_id="collection_rate",
        resolved_value=0.65,
        default_value=0.65,
        value_origin="remind_mfa_default",
        value_mode="retain_default",
        parameter_role="editable_input",
        controlling_change_point_year=2019,
        material=None,
        good=None,
        process=None,
        route=None,
        unit="fraction",
    )
    compiled_directory = workspace / "data_processed/foreground_scenarios" / foreground_scenario
    compiled_directory.mkdir(parents=True)
    compiled_path = compiled_directory / "compiled_annual_input_parameters.parquet"
    table.to_parquet(compiled_path, index=False)
    scenario = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": foreground_scenario,
        "ssp_scenario": "SSP2",
        "changes": [],
    }
    source_path = (
        workspace / "data_raw/foreground_scenarios/scenarios" / (foreground_scenario + ".yaml")
    )
    source_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    scenario_manifest = _manifest(
        payload=scenario,
        catalogue=_catalogue_payload(),
        compiled=table,
        parsed=_ParsedRules((), ()),
        definitions={item.parameter_id: item for item in catalogue},
        source_path=source_path,
    )
    write_workspace_guide(compiled_directory, "pytoce.run_mfa", "compiled_scenario")
    manifest_path = compiled_directory / "scenario_manifest.json"
    manifest_bytes = canonical_json_bytes(scenario_manifest)
    manifest_path.write_bytes(manifest_bytes)
    compiled = CompiledForegroundScenario(
        foreground_scenario,
        cast(str, scenario_manifest["identity"]),
        "SSP2",
        "computed",
        compiled_directory,
        manifest_path,
        compiled_path,
        scenario_manifest,
        sha256(manifest_bytes).hexdigest(),
    )
    _, sources = load_catalogue()
    source = next(item for item in sources if item.source_id == "ecoinvent_3_12_cutoff_ecospold02")
    recipe = derive_material_recipe(
        material_source_members(), source_identity=f"{source.source_id}::{source.sha256}"
    )
    recipe_path = workspace / "data_processed/ecoinvent/material_recipe.json"
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_bytes(canonical_json_bytes(recipe.as_dict()))
    composition = load_material_composition()
    target = build_mfa_result_target(
        workspace,
        compiled,
        request,
        material_recipe_identity=recipe.identity,
        material_composition_identity=composition.identity,
    )
    session = MFAEvaluationSession(request, catalogue, recipe=recipe, composition=composition)
    computation = session.compute(
        foreground_scenario=foreground_scenario, compiled=table, scenario_change_points=()
    )
    assert write_mfa_result(target, computation, replacing=False) == "computed"
    return StoredMFAFixture(request, recipe, composition, compiled, session, target, computation)


def prepare_mfa_inputs(
    workspace: Path,
    source_root: Path,
    *,
    foreground_scenario: str,
) -> EcoinventMaterialRecipe:
    """Write complete invented inputs for the real public MFA and worker boundaries.

    This represents inputs already prepared by download_data. Native selectors and
    source-version fields retain the package schema, while every numerical CSV
    and EcoSpold-shaped quantity is invented. The fixture receipt
    records that distinction. No licensed archive or existing study is read.
    Parent and spawned workers reconstruct the same workspace request; no
    replacement request or patched dependency code enters their calculations.
    """
    set_workspace(workspace)
    processed = workspace / "data_processed/remind_mfa"
    code = processed / "code"
    for name in ("remind_mfa", "config"):
        shutil.copytree(
            source_root / name,
            code / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    coordinates = cast(dict[str, list[str]], _catalogue_payload()["coordinate_sets"])
    renderer = pio.renderers.default
    try:
        write_remind_mfa_inputs(
            code,
            processed / "data",
            regions=tuple(coordinates["region"]),
            materials=tuple(coordinates["material"]),
            bio_production_rate=0.0,
        )
    finally:
        pio.renderers.default = renderer
    (processed / "manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "fixture": "real_pinned_code_with_invented_csv_inputs",
                "code_root": "code",
                "data_root": "data",
                "complete": True,
            }
        )
    )

    # Explicit controls make planner values agree with the parent MFA result.
    # Bio-based monomer routes start in 2020, so the invented historical source
    # is fossil-only and this scenario introduces 5% biogenic feed in 2020.
    controls = {
        "collection_rate": 0.65,
        "mechanical_recycling_rate": 0.2,
        "chemical_recycling_rate": 0.1,
        "incineration_rate": 0.3,
        "bio_production_rate": 0.05,
        "daccu_production_rate": 0.0,
    }
    scenario = {
        "schema_version": "pytoce_foreground_scenario_schema_v1",
        "foreground_scenario": foreground_scenario,
        "ssp_scenario": "SSP2",
        "changes": [
            {
                "parameter": name,
                "selector": "uniform_all_h12",
                "changes": [{"year": 2020, "mode": "absolute", "value": value}],
            }
            for name, value in controls.items()
        ],
    }
    scenario_path = (
        workspace / "data_raw/foreground_scenarios/scenarios" / (foreground_scenario + ".yaml")
    )
    # Preserve installed scenario definitions, including the optimization baseline.
    if not scenario_path.exists():
        scenario_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    _, licensed = load_catalogue()
    source = next(item for item in licensed if item.source_id == "ecoinvent_3_12_cutoff_ecospold02")
    recipe = derive_material_recipe(
        material_source_members(), source_identity=f"{source.source_id}::{source.sha256}"
    )
    recipe_path = workspace / "data_processed/ecoinvent/material_recipe.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_bytes(canonical_json_bytes(recipe.as_dict()))
    (workspace / "fixture_inputs.json").write_text(
        json.dumps(
            {
                "source": "invented CSV and EcoSpold-shaped numerical inputs",
                "code": "unchanged REMIND-MFA 0.2.0 runtime and configuration",
                "scenario": yaml.safe_load(scenario_path.read_text(encoding="utf-8")),
                "scope": "disposable public-workflow test; not a scientific result",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return recipe


def _stock_floor_targets(
    state: NormalizedCohortState,
    scenario: dict[str, Any],
    scenario_path: Path,
) -> np.ndarray:
    """Author fixed controls and retain positive prospective closing-cohort inputs."""
    controls = {
        "collection_rate": 0.65,
        "mechanical_recycling_rate": 0.0,
        "chemical_recycling_rate": 0.0,
        "incineration_rate": 0.3,
        "bio_production_rate": 0.0,
        "daccu_production_rate": 0.0,
    }
    for rule in scenario["changes"]:
        if rule["parameter"] in controls:
            rule["changes"] = [
                {
                    "year": 2020,
                    "mode": "absolute",
                    "interpolation": "step",
                    "value": controls[rule["parameter"]],
                }
            ]
    scenario["changes"].extend(
        {
            "parameter": parameter,
            "region": region,
            "changes": [{"year": 2020, "mode": "absolute", "interpolation": "step", "value": 0.0}],
        }
        for parameter in ("waste_imports", "waste_exports")
        for region in state.regions
    )
    # Compile the authored mapping in memory before publishing its final targets.
    evaluation, _ = _compile_payload(scenario, source_path=scenario_path)
    system = fixed_lifetime_stock_system(state, evaluation.compiled_parameters)
    closing = np.stack(
        [
            state.stock_by_cohort[index, index]
            for index, year in enumerate(state.years)
            if year >= 2020
        ]
    )
    assert np.isfinite(closing).all() and np.all(closing > 0.0)
    # Half the independently observed 2020 EUR/Buildings inherited-stock gap
    # between the real L0 and linear L100 owners, in Mt. This is an invented
    # boundary input supplied by this synthetic fixture.
    closing[0, state.regions.index("EUR"), state.goods.index("Buildings & Construction")] = (
        _STOCK_FLOOR_NEW_CLOSING_MT
    )
    return system.inherited_stock + np.einsum("tcg,crg->trg", system.closing_survival, closing)


def prepare_mfa_workspace(
    workspace: Path,
    source_root: Path,
    *,
    foreground_scenario: str,
    stock_floor: bool = False,
) -> CompletedMFAFixture:
    """Publish a source-matched parent or the fixed prospective stock-floor case."""
    recipe = prepare_mfa_inputs(workspace, source_root, foreground_scenario=foreground_scenario)
    request = prepare_execution(prepare_workspace_source(workspace))
    state = RemindMFAExecutionSession(request).cohort_state
    scenario_path = (
        workspace / "data_raw/foreground_scenarios/scenarios" / f"{foreground_scenario}.yaml"
    )
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    # The CSVs deliberately differ from packaged study defaults. Author their
    # actual stock targets (Mt) and material partition so a
    # demand/lifetime intervention cannot introduce unrelated default stocks
    # or polymers when it activates the prospective cohort calculation.
    scenario["changes"].extend(
        {
            "parameter": "stock_projection",
            "region": region,
            "good": good,
            "changes": [
                {
                    "year": year,
                    "mode": "absolute",
                    "interpolation": "step",
                    "value": float(state.annual_stock[t, r, g]),
                }
                for t, year in enumerate(state.years)
                if year >= 2020
            ],
        }
        for r, region in enumerate(state.regions)
        for g, good in enumerate(state.goods)
    )
    scenario["changes"].extend(
        {
            "parameter": "plastic_family_share",
            "region": region,
            "material": material,
            "good": good,
            "changes": [
                {
                    "year": 2020,
                    "mode": "absolute",
                    "interpolation": "step",
                    "value": float(state.material_shares[r, m, g]),
                }
            ],
        }
        for r, region in enumerate(state.regions)
        for m, material in enumerate(state.materials)
        for g, good in enumerate(state.goods)
    )
    # Means are in years. Standard deviations retain the source standard-deviation/mean ratio.
    scenario["changes"].extend(
        {
            "parameter": "lifetime_mean",
            "good": good,
            "changes": [
                {
                    "year": 2020,
                    "mode": "absolute",
                    "interpolation": "step",
                    "value": float(state.lifetime_mean[g]),
                }
            ],
        }
        for g, good in enumerate(state.goods)
    )
    if stock_floor:
        targets = _stock_floor_targets(state, scenario, scenario_path)
        for rule in scenario["changes"]:
            if rule["parameter"] == "stock_projection":
                region = state.regions.index(rule["region"])
                good = state.goods.index(rule["good"])
                for point in rule["changes"]:
                    point["value"] = float(targets[point["year"] - 2020, region, good])
    scenario_path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    receipt_path = workspace / "fixture_inputs.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["scenario"] = scenario
    receipt["cohort_controls"] = (
        "H0 + A0 @ x prospective stock targets; source-matched material partition and lifetimes"
        if stock_floor
        else "Source-matched annual stocks, material partition and lifetime means"
    )
    if stock_floor:
        receipt["stock_floor"] = {
            "year": 2020,
            "region": "EUR",
            "good": "Buildings & Construction",
            "new_closing_stock_mt": _STOCK_FLOOR_NEW_CLOSING_MT,
            "other_new_closing_stocks": "Unchanged positive source closing-cohort diagonal",
            "basis": "Half the independently observed L100 minus L0 inherited-stock gap",
        }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    report = run_mfa(foreground_scenario=foreground_scenario)
    reference = read_mfa_result_manifest(report.results[0].manifest_path)
    mfa = read_mfa_result(reference)
    return CompletedMFAFixture(request, recipe, load_material_composition(), reference, mfa)
