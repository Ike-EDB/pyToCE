import inspect
from pathlib import Path

import pytest

from pytoce.prepare_backgrounds.preparation.project_biosphere import load_lcia_catalogue
from pytoce.run_optimization import blocks
from pytoce.run_optimization.run_optimization import (
    OptimizationBlock,
    RunOptimizationError,
    RunOptimizationProductReport,
    RunOptimizationReport,
    normalize_run_optimization_request,
    run_optimization,
)


def test_public_signature_remains_the_sixth_workflow_contract() -> None:
    parameters = inspect.signature(run_optimization).parameters

    assert tuple(parameters) == (
        "foreground_scenario",
        "allocated_cc_result",
        "background_scenario",
        "optimization_lcia_method",
        "lcia_method",
        "optimization_block",
        "allocated_cc_case",
        "initial_grid_step_pct",
        "endpoint_refinement_steps_pct",
        "candidate_workers",
        "refresh",
    )
    assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters.values())
    assert OptimizationBlock is blocks.OptimizationBlock
    assert parameters["optimization_block"].default is blocks.OPTIMIZATION_BLOCK_ORDER
    assert parameters["endpoint_refinement_steps_pct"].default == (10, 5)


@pytest.mark.parametrize("steps", ((10, 5), (5,), [10, 5, 1], (19, 3, 1)))
def test_endpoint_refinement_schedule_is_preserved_in_request(steps: tuple[int, ...]) -> None:
    """Public requests preserve exact ordered endpoint increments independently of repair."""
    request = normalize_run_optimization_request(
        foreground_scenario="frozen_2019_default_baseline",
        allocated_cc_result="pyaesa_h12",
        endpoint_refinement_steps_pct=steps,
    )
    assert request.endpoint_refinement_steps_pct == tuple(steps)


@pytest.mark.parametrize(
    "steps", ((), (20,), (25, 5), (5, 10), (5, 5), (0,), (-1,), (101,), (True,), (5.0,), "5", 5)
)
def test_invalid_endpoint_refinement_schedule_is_rejected(steps: object) -> None:
    """Reject ambiguous, unordered or non-integer steps before scientific inputs are opened."""
    with pytest.raises(RunOptimizationError, match="endpoint_refinement_steps_pct"):
        normalize_run_optimization_request(
            foreground_scenario="frozen_2019_default_baseline",
            allocated_cc_result="pyaesa_h12",
            endpoint_refinement_steps_pct=steps,  # type: ignore[arg-type]
        )


def test_request_is_canonical_and_readable() -> None:
    request = normalize_run_optimization_request(
        foreground_scenario="frozen_2019_default_baseline",
        allocated_cc_result="pyaesa_h12",
        optimization_lcia_method="ef31",
        lcia_method=("pb_lcia", "ef31"),
        optimization_block=("integrated_frontier", "demand_lifetime_frontier"),
        allocated_cc_case=("p95", "p5", "p95"),
    )

    assert request.optimization_blocks == (
        "demand_lifetime_frontier",
        "integrated_frontier",
    )
    assert request.allocated_cc_cases == ("p5", "p95")
    assert request.readable_request_selection == (
        "blocks_demand_lifetime_integrated__cases_2_p5_3_p95"
    )
    assert request.result_lcia_selection == "ef31_pb_lcia"


@pytest.mark.parametrize(
    ("foreground", "expected"),
    (
        ("remind_mfa_default", ("remind_mfa_default",)),
        (
            ("study_parent", "frozen_2019_default_baseline", "study_parent"),
            ("frozen_2019_default_baseline", "study_parent"),
        ),
    ),
)
def test_each_named_foreground_is_an_independent_parent(
    foreground: str | tuple[str, ...], expected: tuple[str, ...]
) -> None:
    """Preserve arbitrary readable parents with deterministic order and no duplicates."""
    request = normalize_run_optimization_request(
        foreground_scenario=foreground,
        allocated_cc_result="study_acc",
    )
    assert request.foreground_scenarios == expected


def test_optimization_is_ef_only_while_reporting_may_include_pb() -> None:
    with pytest.raises(RunOptimizationError, match="must be 'ef31'"):
        normalize_run_optimization_request(
            foreground_scenario="frozen_2019_default_baseline",
            allocated_cc_result="pyaesa_h12",
            optimization_lcia_method=("ef31", "pb_lcia"),
        )
    request = normalize_run_optimization_request(
        foreground_scenario="frozen_2019_default_baseline",
        allocated_cc_result="pyaesa_h12",
        optimization_lcia_method="ef31",
        lcia_method=("ef31", "pb_lcia"),
    )
    assert request.result_lcia_selection == "ef31_pb_lcia"


@pytest.mark.parametrize(
    "exact", tuple(method.impact_category_tuple for method in load_lcia_catalogue()[0])
)
def test_reporting_accepts_each_exact_category_from_the_installed_authority(
    exact: tuple[str, ...],
) -> None:
    """Preserve exact reporting selections across all sixteen EF and ten PB categories."""
    request = normalize_run_optimization_request(
        foreground_scenario="frozen_2019_default_baseline",
        allocated_cc_result="pyaesa_h12",
        lcia_method=exact,
    )
    assert request.lcia_method.exact_method == exact
    assert request.lcia_method.groups == ()


@pytest.mark.parametrize(
    "exact",
    (
        ("custom family", "custom category", "custom indicator"),
        ("EF v3.1",),
        ("EF v3.1", "unknown category", "unknown indicator"),
        ("AESA (PBs-LCIA)", "0.9.2"),
        ("AESA (PBs-LCIA)", "0.9.2", "unknown category", "unknown indicator"),
    ),
)
def test_reporting_rejects_unsupported_exact_methods_at_public_admission(
    exact: tuple[str, ...],
) -> None:
    """Reject unsupported or truncated tuples before selecting workspaces or response banks."""
    with pytest.raises(RunOptimizationError, match="lcia_method=.*not supported by optimization"):
        normalize_run_optimization_request(
            foreground_scenario="frozen_2019_default_baseline",
            allocated_cc_result="pyaesa_h12",
            lcia_method=exact,
        )


def test_unknown_or_empty_block_selection_fails_closed() -> None:
    with pytest.raises(RunOptimizationError, match="at least one"):
        normalize_run_optimization_request(
            foreground_scenario="frozen_2019_default_baseline",
            allocated_cc_result="pyaesa_h12",
            optimization_block=(),
        )
    with pytest.raises(RunOptimizationError, match="Unknown"):
        normalize_run_optimization_request(
            foreground_scenario="frozen_2019_default_baseline",
            allocated_cc_result="pyaesa_h12",
            optimization_block=("not_a_block",),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"foreground_scenario": 1}, "one readable name or a sequence"),
        ({"foreground_scenario": ()}, "at least one readable name"),
        ({"allocated_cc_result": "Not readable"}, "lowercase snake_case"),
        ({"lcia_method": ()}, "Invalid lcia_method selection"),
        ({"allocated_cc_case": ()}, "at least one readable name"),
        ({"allocated_cc_case": 1}, "one readable name or a sequence"),
        ({"optimization_block": 1}, "one block or a sequence"),
        ({"refresh": 1}, "Use refresh=True"),
    ),
)
def test_invalid_public_selections_explain_the_required_value(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "foreground_scenario": "frozen_2019_default_baseline",
        "allocated_cc_result": "pyaesa_h12",
    }
    arguments.update(overrides)

    with pytest.raises(RunOptimizationError, match=message):
        normalize_run_optimization_request(**arguments)  # type: ignore[arg-type]


def test_report_contains_selected_products_only() -> None:
    product = RunOptimizationProductReport(
        status="computed",
        foreground_scenario="frozen_2019_default_baseline",
        product_identity="selected-product",
        result_directory=Path("selected_results/ef31__pb_lcia"),
        manifest_path=Path("selected_results/ef31__pb_lcia/manifest.json"),
    )
    report = RunOptimizationReport(
        status="computed",
        workspace=Path("workspace"),
        elapsed_seconds=1.25,
        products=(product,),
    )

    payload = report.as_dict()
    assert payload["status"] == "computed"
    assert payload["products"] == [product.as_dict()]
    assert "candidate_archives" not in payload


@pytest.mark.parametrize("step", (10, 20, 25, 50, 100))
def test_initial_grid_is_admitted_and_stored(step: int) -> None:
    """Expose the supported complete coarse partitions independently of refinement."""
    request = normalize_run_optimization_request(
        foreground_scenario="frozen_2019_default_baseline",
        allocated_cc_result="pyaesa_h12",
        initial_grid_step_pct=step,
        endpoint_refinement_steps_pct=(5,),
    )
    assert request.initial_grid_step_pct == step


@pytest.mark.parametrize("step", (0, 1, 5, 30, 101, True, 20.0, "20"))
def test_invalid_initial_grid_is_rejected(step: object) -> None:
    """Reject unsupported or non-integer coarse grids before producer access."""
    with pytest.raises(RunOptimizationError, match="initial_grid_step_pct"):
        normalize_run_optimization_request(
            foreground_scenario="frozen_2019_default_baseline",
            allocated_cc_result="pyaesa_h12",
            initial_grid_step_pct=step,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("selection", "expected"),
    (
        (
            None,
            (
                "remind_ssp2_ndc",
                "remind_ssp2_npi",
                "remind_ssp2_pkbudg1000",
                "remind_ssp2_pkbudg650",
            ),
        ),
        ("remind_ssp2_npi", ("remind_ssp2_npi",)),
        (
            ("remind_ssp2_pkbudg650", "remind_ssp2_npi", "remind_ssp2_npi"),
            ("remind_ssp2_npi", "remind_ssp2_pkbudg650"),
        ),
    ),
)
def test_background_selection_is_canonical_and_has_a_distinct_readable_owner(
    selection: str | tuple[str, ...] | None, expected: tuple[str, ...]
) -> None:
    """Retain default four-member support and isolate each requested subset's cache."""
    request = normalize_run_optimization_request(
        foreground_scenario="study_parent",
        allocated_cc_result="study_acc",
        background_scenario=selection,
    )
    assert request.background_scenarios == expected
    if selection is None:
        assert "__backgrounds_" not in request.readable_request_selection
    else:
        assert request.readable_request_selection.endswith(
            "__backgrounds_" + "_".join(name.removeprefix("remind_ssp2_") for name in expected)
        )


@pytest.mark.parametrize("selection", ((), "remind_eu_ssp2", "unknown", 2))
def test_invalid_background_selection_fails_at_public_admission(selection: object) -> None:
    """Reject unsupported historical and empty background selections."""
    with pytest.raises(RunOptimizationError, match="background_scenario"):
        normalize_run_optimization_request(
            foreground_scenario="study_parent",
            allocated_cc_result="study_acc",
            background_scenario=selection,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("workers", (1, 3, 8, 61))
def test_worker_count_is_operational_and_does_not_change_request_location(workers: int) -> None:
    """Choose a process bound for compatible candidates."""
    request = normalize_run_optimization_request(
        foreground_scenario="study_parent",
        allocated_cc_result="study_acc",
        candidate_workers=workers,
    )
    default = normalize_run_optimization_request(
        foreground_scenario="study_parent",
        allocated_cc_result="study_acc",
    )
    assert request.candidate_workers == workers
    assert request.readable_request_selection == default.readable_request_selection
    assert default.candidate_workers == 1


@pytest.mark.parametrize("workers", (0, -1, 62, True, 1.0, "3", None))
def test_worker_count_rejects_invalid_public_values(workers: object) -> None:
    """Reject invalid process counts before opening any scientific products."""
    with pytest.raises(
        RunOptimizationError, match="candidate_workers must be an integer from 1 through 61"
    ):
        normalize_run_optimization_request(
            foreground_scenario="study_parent",
            allocated_cc_result="study_acc",
            candidate_workers=workers,  # type: ignore[arg-type]
        )
