"""Admit plausible numerical LP returns through unchanged full worker physics."""

from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import optimize

from pytoce import run_mfa, set_workspace
from pytoce.download_data.sources.ecoinvent_materials import load_material_recipe
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_mfa.model.cohorts import continue_cohort_stock, fixed_lifetime_stock_system
from pytoce.run_mfa.model.flows import calculate_mfa_flows
from pytoce.run_mfa.model.materials import load_material_composition
from pytoce.run_mfa.model.result_inputs import read_mfa_result, read_mfa_result_manifest
from pytoce.run_mfa.remind_mfa.execution import (
    RemindMFAExecutionSession,
    prepare_execution,
    prepare_workspace_source,
)
from pytoce.run_mfa.remind_mfa.translation import translate_annual_inputs
from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue
from pytoce.run_mfa.scenario.compiler import _compile_payload
from pytoce.run_optimization.calculation.candidate_evaluation import (
    AdmittedCandidateCalculation,
    InfeasibleCandidateCalculation,
)
from pytoce.run_optimization.calculation.candidate_worker import (
    CandidateWorkerRequestContext,
    _demand_stock_multipliers,
    initialize_candidate_worker,
    prepare_candidate_worker_request_context,
)
from pytoce.run_optimization.calculation.demand_lifetime_search import (
    demand_repair_neighbors,
)
from pytoce.run_optimization.calculation.interventions import (
    CircularityDecision,
    DemandLifetimeDecision,
)
from pytoce.run_optimization.calculation.request_search import (
    build_candidate_worker_metric_scopes,
)
from pytoce.run_optimization.calculation.worker_inputs import (
    prepare_candidate_worker_initializer,
)
from pytoce.run_optimization.calculation.worker_protocol import (
    CandidateWorkerResult,
    CandidateWorkerTask,
    decode_candidate_worker_result,
    encode_candidate_worker_task,
)
from pytoce.run_optimization.input.allocated_cc_product import compile_allocated_cc_result
from tests.integration.candidate_mfa_fixtures import _prepare_worker_workspace


def _receiving_fraction(
    workspace: Path,
    adapter: RemindMFAExecutionSession,
    scenario: dict,
    path: Path,
) -> Fraction:
    """Derive the actual 2020 EUR receiving bound before authoring recycling.

    The source contains only HDPE/PP. With zero chemical recycling and trade,
    mechanical inputs can be assigned up to min(collected, gross input / yield)
    for each constituent. Their sum divided by total collected waste is the
    exact feasible mechanical fraction; monomer absorption adds no tighter bound.
    """
    compiled, _ = _compile_payload(scenario, source_path=path)
    cohort = continue_cohort_stock(
        adapter.cohort_state,
        compiled.compiled_parameters,
        optimization_overlay=False,
        record_diagnostics=False,
    )
    translated = translate_annual_inputs(compiled.compiled_parameters, load_parameter_catalogue())
    inputs = adapter.process_inputs(
        translated.numeric_updates,
        carbon_inflow=cohort.carbon_inflow,
        carbon_outflow=cohort.carbon_outflow,
        deferred_waste_trade=translated.deferred_waste_trade,
        record_diagnostics=False,
    )
    recipe = load_material_recipe(workspace)
    flows = calculate_mfa_flows(
        cohort,
        inputs,
        recipe=recipe,
        composition=load_material_composition(),
        first_year=2020,
        last_year=2020,
        record_diagnostics=False,
    )
    eur = next(item for item in flows.material_states if item.region == "EUR")
    recovery = {
        item.name: Fraction(item.recovered_kg_per_input_kg) for item in recipe.mechanical_recovery
    }
    collected = Fraction()
    capacity = Fraction()
    for constituent, waste, gross in zip(
        eur.constituents,
        eur.amounts["collected_after_trade"],
        eur.amounts["manufacturing_principal_input"],
        strict=True,
    ):
        collected += Fraction(float(waste))
        if constituent.mechanical_recovery is not None:
            capacity += min(
                Fraction(float(waste)),
                Fraction(float(gross)) / recovery[constituent.mechanical_recovery],
            )
    return capacity / collected


def _new_parent(
    workspace: Path, *, receiving: bool, receiving_weight: Fraction = Fraction(1, 2)
) -> CandidateWorkerRequestContext:
    """Create a source-based parent and preserve completed fixture products.

    At fixed lifetime, H is inherited stock in 2020. Set P2020=H/(1-u),
    and P_t=P2020/(1-linear_d_t) afterward. The linear candidate then has
    constant stock from 2021 onward. u=1/128 puts its first target below H;
    u=1/32 puts it strictly above H and permits an interior receiving failure.
    Both constants are fixed algebraic fixture choices.
    The receiving weight places the mechanical fraction between the actual
    linear and parent capacities, permitting feasible and exhausted-pool cases.
    """
    set_workspace(workspace)
    adapter = RemindMFAExecutionSession(prepare_execution(prepare_workspace_source(workspace)))
    compiled = pd.read_parquet(
        workspace
        / "data_processed/foreground_scenarios/candidate_worker_parent"
        / "compiled_annual_input_parameters.parquet"
    )
    system = fixed_lifetime_stock_system(adapter.cohort_state, compiled)
    reference = DemandLifetimeDecision(50, 0)
    u = 1.0 / (32 if receiving else 128)
    first_stock = system.inherited_stock[0] / (1.0 - u)
    targets = np.stack(
        [
            first_stock / (1.0 - float(reference.demand_fraction[year - 2019]))
            for year in system.years
        ]
    )
    targets[0] = first_stock
    source = workspace / "data_raw/foreground_scenarios/scenarios/candidate_worker_parent.yaml"
    scenario = yaml.safe_load(source.read_text(encoding="utf-8"))
    foreground = "solver_reference_receiving" if receiving else "solver_interior_stock"
    scenario["foreground_scenario"] = foreground
    controls = {
        "collection_rate": 1.0,
        "mechanical_recycling_rate": 0.0,
        "chemical_recycling_rate": 0.0,
        "incineration_rate": 1.0,
        "bio_production_rate": 0.0,
        "daccu_production_rate": 0.0,
    }
    for rule in scenario["changes"]:
        parameter = rule["parameter"]
        if parameter in controls:
            rule["changes"] = [
                {
                    "year": 2020,
                    "mode": "absolute",
                    "interpolation": "step",
                    "value": controls[parameter],
                }
            ]
        elif parameter == "stock_projection":
            region = system.regions.index(rule["region"])
            good = system.goods.index(rule["good"])
            for point in rule["changes"]:
                point["value"] = float(targets[point["year"] - 2020, region, good])
    scenario["changes"].extend(
        {
            "parameter": parameter,
            "region": region,
            "changes": [{"year": 2020, "mode": "absolute", "interpolation": "step", "value": 0.0}],
        }
        for parameter in ("waste_imports", "waste_exports")
        for region in system.regions
    )
    path = source.with_name(foreground + ".yaml")
    assert not path.exists()
    if receiving:
        parent_capacity = _receiving_fraction(workspace, adapter, scenario, path)
        linear = deepcopy(scenario)
        for rule in linear["changes"]:
            if rule["parameter"] == "stock_projection":
                for point in rule["changes"]:
                    point["value"] *= 1.0 - float(reference.demand_fraction[point["year"] - 2019])
        linear_capacity = _receiving_fraction(workspace, adapter, linear, path)
        assert 0 < linear_capacity < parent_capacity < 1
        rate = float((1 - receiving_weight) * linear_capacity + receiving_weight * parent_capacity)
        assert linear_capacity < Fraction(rate) < parent_capacity
        scenario["changes"] = [
            rule
            for rule in scenario["changes"]
            if rule["parameter"] not in {"mechanical_recycling_rate", "incineration_rate"}
        ]
        scenario["changes"].extend(
            {
                "parameter": parameter,
                "region": region,
                "changes": [
                    {
                        "year": 2020,
                        "mode": "absolute",
                        "interpolation": "step",
                        "value": (rate if region == "EUR" else 0.0)
                        if parameter == "mechanical_recycling_rate"
                        else (1.0 - rate if region == "EUR" else 1.0),
                    },
                    {
                        "year": 2021,
                        "mode": "absolute",
                        "interpolation": "step",
                        "value": 0.0 if parameter == "mechanical_recycling_rate" else 1.0,
                    },
                ],
            }
            for parameter in ("mechanical_recycling_rate", "incineration_rate")
            for region in system.regions
        )
    path.write_text(yaml.safe_dump(scenario), encoding="utf-8")
    report = run_mfa(foreground_scenario=foreground)
    parent = read_mfa_result_manifest(report.results[0].manifest_path)
    allocated_cc = compile_allocated_cc_result(workspace, "worker_result")
    initializer = prepare_candidate_worker_initializer(
        workspace,
        foreground_scenario=foreground,
        mfa_reference=parent,
        allocated_cc=allocated_cc,
        allocated_cc_case=("generous_case",),
        response_method_union="ef31_response_v3",
    )
    return prepare_candidate_worker_request_context(
        initializer,
        allocated_cc,
        parent_mfa=read_mfa_result(parent),
        result_lcia_method=normalized_lcia_method(("ef31",)),
    )


def _evaluate(context: CandidateWorkerRequestContext) -> CandidateWorkerResult:
    """Use the normal worker protocol for one D50/L0 endpoint."""
    intervention = context.planner.plan_intervention(
        demand_lifetime=DemandLifetimeDecision(50, 0), circularity=CircularityDecision.parent()
    )
    task = CandidateWorkerTask(
        intervention, build_candidate_worker_metric_scopes(("demand_lifetime_frontier",))
    )
    worker = initialize_candidate_worker(context.worker_initializer_payload)
    return decode_candidate_worker_result(worker(encode_candidate_worker_task(task)))


@pytest.fixture
def solver_boundary_workspace(
    tmp_path: Path,
    remind_mfa_source_root: Path,
    external_remind_mfa_import_state: None,
) -> Path:
    """Prepare one real synthetic parent and compatible response bank for both boundaries."""
    workspace = tmp_path / "workspace"
    _prepare_worker_workspace(
        workspace,
        remind_mfa_source_root,
        include_generous_case=True,
        include_public_baseline_results=False,
    )
    return workspace


@pytest.mark.scientific_integration
def test_valid_numerical_stock_returns_keep_full_admission_and_physical_deduplication(
    solver_boundary_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only SciPy's public return is simulated; all package science stays real.

    SciPy 1.13.1 HiGHS-DS uses default primal/dual feasibility tolerances 1e-7:
    https://docs.scipy.org/doc/scipy-1.13.1/reference/optimize.linprog-highs-ds.html
    The first double moves one real optimum a fixed 2^-40 into its stock bound,
    verifies submitted primal constraints and objective gap, and retains its
    dual values. This is a constructed approximate numerical solution. The second return
    is the exact zero-departure reference, with zero as an independent global
    lower bound for the nonnegative absolute-departure objective.
    """
    real_linprog = optimize.linprog
    reference = DemandLifetimeDecision(50, 0)
    free = 30
    values = np.array([float(value) for value in reference.demand_fraction[1 : free + 1]])
    calls = []

    def interior_return(c: np.ndarray, **kwargs: object) -> optimize.OptimizeResult:
        """Simulate one admissible near-optimum while keeping actual HiGHS dual evidence."""
        solved = real_linprog(c, **kwargs)
        assert solved.status == 0
        returned = solved.x.copy()
        displacement = 2.0**-40
        assert returned[0] > displacement
        returned[0] -= displacement
        returned[1] += displacement
        returned[free:] = np.abs(np.cumsum(returned[:free]) - values)
        constraints = np.asarray(kwargs["A_ub"])
        bounds = np.asarray(kwargs["b_ub"])
        primal_residual = constraints @ returned - bounds
        dual = solved.ineqlin.marginals
        lower = solved.lower.marginals
        # Zero primal lower bounds, dual signs, and stationarity establish the numerical dual bound.
        assert dual.max() <= 1e-7 and lower.min() >= -1e-7
        assert np.abs(c - constraints.T @ dual - lower).max() <= 1e-7
        objective_gap = float(c @ returned - bounds @ dual)
        assert returned.min() >= 0.0
        assert primal_residual.max() <= 1e-7
        assert -1e-7 <= objective_gap < 1e-7
        calls.append("interior")
        return optimize.OptimizeResult(
            solved,
            x=returned,
            fun=float(c @ returned),
            slack=-primal_residual,
            ineqlin=optimize.OptimizeResult(solved.ineqlin, residual=-primal_residual),
            lower=optimize.OptimizeResult(solved.lower, residual=returned.copy()),
        )

    stock_context = _new_parent(solver_boundary_workspace, receiving=False)
    with monkeypatch.context() as patch:
        patch.setattr(optimize, "linprog", interior_return)
        stock = _evaluate(stock_context)
    assert calls == ["interior"]
    assert stock.repair.status == "stock_projection_feasible"
    assert stock.repair.mfa_evaluation_count == 2
    assert stock.repair.stock_lp_count == 1
    assert stock.repair.pool_evaluation_count == stock.repair.pool_candidate_count == 0
    assert stock.repair.linear_diagnostic is not None
    assert stock.repair.linear_diagnostic.code == "optimization_stock_below_surviving_cohort"
    assert stock.physical.diagnostic is None and stock.scores is not None

    def exact_reference_return(c: np.ndarray, **kwargs: object) -> optimize.OptimizeResult:
        """Return a feasible zero-objective reference, independently optimal for this LP."""
        increments = np.diff(np.concatenate(([0.0], values)))
        returned = np.concatenate((increments, np.zeros(free)))
        exact = tuple(Fraction(float(value)) for value in increments)
        assert tuple(sum(exact[:end], Fraction()) for end in range(1, free + 1)) == tuple(
            reference.demand_fraction[1 : free + 1]
        )
        assert returned.min() >= 0.0 and float(c @ returned) == 0.0
        assert (np.asarray(kwargs["A_ub"]) @ returned - np.asarray(kwargs["b_ub"])).max() <= 1e-7
        calls.append("exact_reference")
        return optimize.OptimizeResult(
            status=0, x=returned, fun=0.0, message="Optimization terminated successfully."
        )

    receiving_context = _new_parent(solver_boundary_workspace, receiving=True)
    with monkeypatch.context() as patch:
        patch.setattr(optimize, "linprog", exact_reference_return)
        receiving = _evaluate(receiving_context)
    assert calls == ["interior", "exact_reference"]
    assert receiving.repair.linear_diagnostic is not None
    assert (
        receiving.repair.linear_diagnostic.code
        == "optimization_material_recovery_capacity_infeasible"
    )
    assert receiving.repair.stock_projection_diagnostic == receiving.repair.linear_diagnostic
    assert receiving.repair.stock_lp_count == 1
    # Reuse the rejected linear MFA trajectory; distinct pool paths account for additional
    # evaluations.
    assert receiving.repair.mfa_evaluation_count == 1 + receiving.repair.pool_evaluation_count
    assert receiving.repair.pool_evaluation_count > 0
    assert receiving.repair.status in {"local_pool_feasible", "repair_unresolved"}
    assert (receiving.scores is not None) == (receiving.repair.status == "local_pool_feasible")


@pytest.mark.scientific_integration
@pytest.mark.parametrize(
    ("receiving_weight", "expected_status"),
    [(Fraction(1, 4), "local_pool_feasible"), (Fraction(1, 2), "repair_unresolved")],
)
def test_rejected_assignment_reuse_preserves_ordered_physical_choice_and_actual_counts(
    solver_boundary_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    receiving_weight: Fraction,
    expected_status: str,
) -> None:
    """Compare against the prior ordered real admission, including a later feasible path.

    The justified public SciPy double shifts two reference fractions by one
    representable step. Its primal residual and gap from the nonnegative LP
    objective's zero lower bound remain below HiGHS' default 1e-7 tolerance.
    The solver stub returns this numerical value.
    Only that dependency return is doubled; receiving physics, trajectories,
    pool generation and admission remain real. No environmental score is needed.
    """
    reference = DemandLifetimeDecision(50, 0)
    free = reference.target_year - 2020
    values = np.array([float(value) for value in reference.demand_fraction[1 : free + 1]])
    near = values.copy()
    near[1:3] = np.nextafter(near[1:3], 0.0)
    np.testing.assert_array_equal(1.0 - values, 1.0 - near)
    calls = []

    def near_reference_return(c: np.ndarray, **kwargs: object) -> optimize.OptimizeResult:
        """Return a numerically admissible near-reference solution with exact float aliases."""
        increments = np.diff(np.concatenate(([0.0], near)))
        returned = np.concatenate((increments, np.abs(near - values)))
        exact = tuple(Fraction(float(value)) for value in increments)
        assert tuple(sum(exact[:end], Fraction()) for end in range(1, free + 1)) == tuple(
            Fraction(float(value)) for value in near
        )
        assert returned.min() >= 0.0
        assert 0.0 <= float(c @ returned) < 1e-7
        assert (np.asarray(kwargs["A_ub"]) @ returned - np.asarray(kwargs["b_ub"])).max() <= 1e-7
        calls.append("near_reference")
        return optimize.OptimizeResult(
            status=0,
            x=returned,
            fun=float(c @ returned),
            message="Optimization terminated successfully.",
        )

    context = _new_parent(
        solver_boundary_workspace, receiving=True, receiving_weight=receiving_weight
    )
    worker = initialize_candidate_worker(context.worker_initializer_payload)
    requested = context.planner.plan_intervention(
        demand_lifetime=reference, circularity=CircularityDecision.parent()
    )
    with monkeypatch.context() as patch:
        patch.setattr(optimize, "linprog", near_reference_return)
        selected, admitted, repair = worker._repair(requested)
    assert calls == ["near_reference"]
    assert repair.status == expected_status
    assert repair.stock_lp_count == 1
    proposal, solved = worker._stock_proposal(reference)
    assert proposal is not None and not solved
    projected = context.planner.plan_intervention(
        demand_lifetime=proposal, circularity=CircularityDecision.parent()
    )
    assert requested.identity != projected.identity
    assert requested.assignment_records == projected.assignment_records
    initial_failure = worker._session.admit_physical(requested)
    projected_failure = worker._session.admit_physical(projected)
    assert isinstance(initial_failure, InfeasibleCandidateCalculation)
    assert isinstance(projected_failure, InfeasibleCandidateCalculation)
    assert repair.linear_diagnostic == initial_failure.physical.diagnostic
    assert repair.stock_projection_diagnostic == projected_failure.physical.diagnostic
    assert repair.stock_projection_diagnostic is not None
    failure_year = dict(repair.stock_projection_diagnostic.coordinates)["year"]
    assert isinstance(failure_year, int)
    pool = demand_repair_neighbors(proposal, failure_year=failure_year)

    # This oracle retains the previous identity-only traversal. Every distinct
    # identity is admitted, including aliases, until its first success.
    seen = {requested.identity, projected.identity}
    controls_seen = {
        _demand_stock_multipliers(reference): requested.assignment_records,
        _demand_stock_multipliers(proposal): projected.assignment_records,
    }
    oracle_selected = projected
    oracle_admitted = projected_failure
    oracle_calls = 0
    rejected_aliases = 0
    for alternative in pool:
        candidate = context.planner.plan_intervention(
            demand_lifetime=alternative, circularity=CircularityDecision.parent()
        )
        if candidate.identity in seen:
            continue
        seen.add(candidate.identity)
        physical = worker._session.admit_physical(candidate)
        oracle_calls += 1
        controls = _demand_stock_multipliers(alternative)
        if controls in controls_seen:
            assert candidate.assignment_records == controls_seen[controls]
            assert isinstance(physical, InfeasibleCandidateCalculation)
            rejected_aliases += 1
        else:
            controls_seen[controls] = candidate.assignment_records
        if isinstance(physical, AdmittedCandidateCalculation):
            oracle_selected, oracle_admitted = candidate, physical
            break
    assert rejected_aliases > 0
    assert repair.pool_candidate_count == len(pool)
    assert repair.pool_evaluation_count == oracle_calls - rejected_aliases
    assert repair.mfa_evaluation_count == 2 + repair.pool_evaluation_count
    assert selected.identity == oracle_selected.identity
    assert selected.assignment_records == oracle_selected.assignment_records
    if isinstance(oracle_admitted, AdmittedCandidateCalculation):
        assert isinstance(admitted, AdmittedCandidateCalculation)
        assert admitted.evaluation is not None and oracle_admitted.evaluation is not None
        assert (
            admitted.evaluation.physical_state_identity
            == oracle_admitted.evaluation.physical_state_identity
        )
        assert repair.status == "local_pool_feasible"
    else:
        assert isinstance(admitted, InfeasibleCandidateCalculation)
        assert admitted.physical == projected_failure.physical
        assert repair.status == "repair_unresolved"

    if expected_status == "local_pool_feasible":
        # One repeat establishes call-local rejection knowledge in a persistent worker.
        repeated_selected, repeated_admitted, repeated = worker._repair(requested)
        assert repeated_selected.identity == selected.identity
        assert type(repeated_admitted) is type(admitted)
        assert repeated.status == repair.status and repeated.stock_lp_count == 0
        assert repeated.mfa_evaluation_count == repair.mfa_evaluation_count
        assert repeated.pool_evaluation_count == repair.pool_evaluation_count
        assert repeated.pool_candidate_count == repair.pool_candidate_count
        assert repeated.linear_diagnostic == repair.linear_diagnostic
        assert repeated.stock_projection_diagnostic == repair.stock_projection_diagnostic
