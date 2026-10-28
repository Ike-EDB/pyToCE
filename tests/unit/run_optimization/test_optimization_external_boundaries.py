"""External file and scientific-library boundaries used by optimization."""

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from allocated_cc_fixtures import _write_raw_result, _write_shared_authority
from pa_receiving_fixtures import pa_receiving_reference, pa_receiving_sources
from response_bank_fixtures import _write_bank

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    brightway_names,
    selected_brightway_project,
)
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_optimization.calculation.response_bank import (
    response_projection_selection_identity,
)
from pytoce.run_optimization.calculation.response_bank_contract import (
    _DIRECT_COLUMNS,
    _NATIVE_COLUMNS,
    ResponseBankProjection,
)
from pytoce.run_optimization.calculation.response_builder import (
    _ensure_optimization_response_bank,
    response_method_selection,
)
from pytoce.run_optimization.calculation.response_kernel import (
    ResponseBuilderError,
    _provider_rows,
    _service_registry,
)
from pytoce.run_optimization.input.allocated_cc_paths import (
    AllocatedCCError,
    _read_csv,
    _secure_results_root,
)
from pytoce.run_optimization.input.allocated_cc_product import (
    _compile_allocated_cc_result,
    _secure_processed_result_directory,
    compile_allocated_cc_result,
)
from pytoce.run_optimization.runtime import scheduler
from tests.integration.brightway_lca_fixtures import PROVIDER_TECHNICAL_YEARS


def test_response_projection_selection_identity_binds_method_order() -> None:
    """Bind the requested reporting methods to the selected response projection."""

    projection = ResponseBankProjection(
        role="result",
        response_bank_identity="bank::fixture",
        projection_identity="projection::fixture",
        provider_route_identity="routes::fixture",
        provider_service_registry_identity="services::fixture",
        provider_service_registry=pd.DataFrame(),
        method_registry_identity="methods::fixture",
        method_registry=pd.DataFrame(),
        provider_unit_responses_identity="responses::fixture",
        provider_unit_responses=pd.DataFrame(),
        timex_year_weights_identity="weights::fixture",
        timex_year_weights=pd.DataFrame(),
        native_unit_responses_identity="no-pa-native::fixture",
        native_unit_responses=pd.DataFrame(columns=_NATIVE_COLUMNS),
        direct_characterization_factors_identity="no-pa-direct::fixture",
        direct_characterization_factors=pd.DataFrame(columns=_DIRECT_COLUMNS),
    )

    first = response_projection_selection_identity(("ef31", "pb_lcia"), projection)
    second = response_projection_selection_identity(("pb_lcia", "ef31"), projection)

    assert first.startswith("pytoce_optimization_result_lcia_selection_v1::")
    assert first != second


def test_response_builder_reuses_result_completed_while_waiting(tmp_path: Path) -> None:
    """Reuse a complete response bank if another writer finished it before locking."""

    selection = response_method_selection(normalized_lcia_method(("ef31", "pb_lcia")))
    completed = _write_bank(tmp_path, response_method_union=selection.readable_name)

    observed = _ensure_optimization_response_bank(
        tmp_path,
        owner=completed.result_directory.parent,
        selection=selection,
        member_progress=lambda _background: pytest.fail("a completed bank must be reused"),
    )

    assert observed.identity == completed.identity
    assert observed.status == "reused"


def test_response_service_registry_is_unique_and_stably_sorted() -> None:
    """Retain one deterministic row for each prepared provider service."""

    routes = pd.DataFrame.from_records(
        [
            {"provider_service_id": "second", "provider_service_code": "code-b"},
            {"provider_service_id": "first", "provider_service_code": "code-a"},
            {"provider_service_id": "first", "provider_service_code": "code-a"},
        ]
    )

    observed = _service_registry(routes)

    assert observed.to_dict("records") == [
        {"provider_service_id": "first", "provider_service_code": "code-a"},
        {"provider_service_id": "second", "provider_service_code": "code-b"},
    ]


def test_response_kernel_translates_missing_brightway_provider(tmp_path: Path) -> None:
    """Report the exact provider service that is absent from a prepared Brightway database."""

    from bw2data.errors import UnknownObject

    from pytoce.run_lca.calculation.providers import (
        PreparedProviderProduct,
    )

    frontiers = pd.DataFrame(
        {
            "anchor_year": PROVIDER_TECHNICAL_YEARS,
            "database_name": [f"provider_{year}" for year in PROVIDER_TECHNICAL_YEARS],
        }
    )
    services = pd.DataFrame(
        {"provider_service_id": ["service"], "provider_service_code": ["missing-code"]}
    )

    names = brightway_names(tmp_path, source_identity="missing-provider-fixture")
    with selected_brightway_project(names) as bd:
        with pytest.raises(
            ResponseBuilderError,
            match="database='provider_2020', year=2020, service='missing-code'",
        ) as failure:
            _provider_rows(
                bd,
                PreparedProviderProduct(
                    dict(zip(frontiers.anchor_year, frontiers.database_name)),
                    {},
                    pa_receiving_reference=pa_receiving_reference(),
                    pa_receiving_sources_by_year={
                        int(year): pa_receiving_sources(
                            int(year), background_scenario="remind_ssp2_npi"
                        )
                        for year in frontiers.anchor_year
                    },
                ),
                services,
            )
        assert isinstance(failure.value.__cause__, UnknownObject)


def test_response_kernel_reads_brightway_provider_product_identity() -> None:
    """Bind each response activation to the prepared activity and reference product."""

    from pytoce.run_lca.calculation.providers import (
        PreparedProviderProduct,
    )

    class Product:
        """Represent one Brightway reference product."""

        id = 29

    class Exchange:
        """Represent the production exchange returned by Brightway."""

        input = Product()

    class Node:
        """Represent one prepared Brightway provider activity."""

        id = 17

        @staticmethod
        def rp_exchange() -> Exchange:
            """Return the provider's reference-production exchange."""

            return Exchange()

    class BrightwayBoundary:
        """Implement the exact public node lookup used by the response owner."""

        @staticmethod
        def get_node(*, database: str, code: str) -> Node:
            """Return the requested fixture provider."""

            assert database.startswith("provider_")
            assert code == "service-code"
            return Node()

    frontiers = pd.DataFrame(
        {
            "anchor_year": PROVIDER_TECHNICAL_YEARS,
            "database_name": [f"provider_{year}" for year in PROVIDER_TECHNICAL_YEARS],
        }
    )
    services = pd.DataFrame(
        {"provider_service_id": ["service"], "provider_service_code": ["service-code"]}
    )

    rows = _provider_rows(
        BrightwayBoundary(),
        PreparedProviderProduct(
            dict(zip(frontiers.anchor_year, frontiers.database_name)),
            {},
            pa_receiving_reference=pa_receiving_reference(),
            pa_receiving_sources_by_year={
                int(year): pa_receiving_sources(int(year), background_scenario="remind_ssp2_npi")
                for year in frontiers.anchor_year
            },
        ),
        services,
    )

    assert len(rows) == len(PROVIDER_TECHNICAL_YEARS)
    assert rows["provider_activity_id"].eq(17).all()
    assert rows["provider_product_id"].eq(29).all()
    assert rows["amount"].eq(0.0).all()


@pytest.mark.parametrize("content", [b"\xff", b"", b'allocated_cc_case\n"unterminated\n'])
def test_allocated_cc_csv_reader_translates_unreadable_input(
    tmp_path: Path, content: bytes
) -> None:
    """Report an unreadable aCC table at its direct CSV boundary."""

    path = tmp_path / "allocated.csv"
    path.write_bytes(content)

    with pytest.raises(AllocatedCCError, match="Cannot read annual aCC table"):
        _read_csv(path, role="annual aCC table")


def test_allocated_cc_results_root_translates_resolve_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a results directory that the operating system cannot resolve."""

    expected = tmp_path / "data_raw" / "allocated_cc" / "results"
    expected.mkdir(parents=True)
    original_resolve = Path.resolve

    def fail_expected(path: Path, *, strict: bool = False) -> Path:
        """Fail only for the aCC results directory under test."""

        if path == expected:
            raise OSError("cannot resolve")
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_expected)
    with pytest.raises(AllocatedCCError, match="Cannot resolve.*results directory"):
        _secure_results_root(tmp_path)


def _directory_link(link: Path, target: Path) -> None:
    """Create a real directory link using the current operating-system boundary."""

    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"Directory links are unavailable: {exc}")
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            pytest.skip(f"Directory junctions are unavailable: {result.stderr}")


@pytest.mark.parametrize(
    ("linked_component", "message"),
    [
        ("data_processed", "escapes the workspace"),
        ("allocated_cc", "escapes data_processed"),
        ("result", "must be a real direct child"),
    ],
)
def test_processed_allocated_cc_paths_reject_directory_links(
    tmp_path: Path,
    linked_component: str,
    message: str,
) -> None:
    """Reject links and junctions at each writable aCC product component."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    data_processed = workspace / "data_processed"
    allocated_root = data_processed / "allocated_cc"
    result = allocated_root / "fixture"
    if linked_component == "data_processed":
        _directory_link(data_processed, outside)
    elif linked_component == "allocated_cc":
        data_processed.mkdir()
        _directory_link(allocated_root, outside)
    else:
        allocated_root.mkdir(parents=True)
        _directory_link(result, outside)

    with pytest.raises(AllocatedCCError, match=message):
        _secure_processed_result_directory(workspace, "fixture")


def test_allocated_cc_compiler_reuses_product_completed_while_waiting(tmp_path: Path) -> None:
    """Reuse a complete aCC product if another writer finished it before locking."""

    _write_shared_authority(tmp_path)
    _write_raw_result(tmp_path, "fixture")
    original = compile_allocated_cc_result(tmp_path, "fixture")

    observed = _compile_allocated_cc_result(tmp_path, "fixture", original.result_directory)

    assert observed == replace(original, status="reused")


def test_allocated_cc_compiler_removes_staging_after_parquet_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove an incomplete hidden staging directory after serializer failure."""

    _write_shared_authority(tmp_path)
    _write_raw_result(tmp_path, "fixture")

    def fail_parquet(*_args: object, **_kwargs: object) -> None:
        """Represent one failure at the external Parquet serialization boundary."""

        raise OSError("Parquet write failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_parquet)
    with pytest.raises(OSError, match="Parquet write failed"):
        compile_allocated_cc_result(tmp_path, "fixture")

    owner = tmp_path / "data_processed" / "allocated_cc"
    assert not tuple(owner.glob(".fixture.allocated_cc.*"))


def test_candidate_scheduler_reuses_one_external_process_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one spawned-process executor across successive candidate batches."""

    starts: list[bytes] = []

    class Executor:
        """Represent the external process-pool lifecycle with a test double."""

        def __init__(self, **_options: object) -> None:
            starts.append(b"started")

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait and cancel_futures

    monkeypatch.setattr(scheduler, "ProcessPoolExecutor", Executor)
    with scheduler.CandidateProcessPool(worker_initializer_payload=b"initializer") as pool:
        first = pool._get_executor()
        second = pool._get_executor()

    assert first is second
    assert starts == [b"started"]
