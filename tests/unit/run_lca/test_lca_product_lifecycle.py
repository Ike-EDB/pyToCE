"""LCA result-reuse and calculation-process boundary tests."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.run_lca.calculation.lcia import normalized_lcia_method
from pytoce.run_lca.calculation.results import (
    _RESULT_ARTIFACTS,
)
from pytoce.run_lca.run_lca import (
    _calculate_and_write_product,
    _isolated_product,
    _method_selection,
    _Request,
    _year_name,
)
from pytoce.shared.runtime.filesystem import payload_sha256


def _completed_result(directory: Path) -> None:
    """Write the exact manifest-last file set admitted for refresh."""

    directory.mkdir()
    for name in (*_RESULT_ARTIFACTS, "README.md", "execution_report.json"):
        (directory / name).write_bytes(b"")
    manifest = {
        "schema_version": "pytoce_lca_result_schema_v1",
        "complete": True,
        "foreground_scenario": "scenario",
        "background_scenario": "remind_ssp2_npi",
        "artifacts": {name: {} for name in (*_RESULT_ARTIFACTS, "execution_report.json")},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_consecutive_analysis_years_use_one_readable_range() -> None:
    """Compress consecutive years and preserve separate requested years."""

    assert _year_name((2019, 2020, 2021, 2050, 2100)) == "2019-2021_2050_2100"


def test_completed_result_is_admitted_before_owner_scoped_refresh(tmp_path: Path) -> None:
    """Preserve the completed output when refresh cannot find its required MFA parent."""

    directory = tmp_path / "result"
    _completed_result(directory)
    before = {path: path.read_bytes() for path in directory.iterdir()}

    with pytest.raises(FileNotFoundError, match="Run run_mfa"):
        _calculate_and_write_product(
            tmp_path,
            _Request("scenario", "remind_ssp2_npi", directory),
            normalized_lcia_method("ef31"),
            (2019,),
            refresh=True,
        )
    assert {path: path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize("with_uncertainty", [False, True])
def test_method_selection_uses_the_prepared_brightway_project(
    tmp_path: Path, with_uncertainty: bool
) -> None:
    """Preserve native CF data and canonical identities through real method storage."""
    import bw2data as bd

    previous_project = bd.projects.current
    previous_store = bd.projects.dir.parent
    names = BrightwayNames(
        store=tmp_path / "data_processed/brightway",
        project="method_selection",
        biosphere_database="biosphere",
        source_database="source",
    )
    authority = pd.DataFrame.from_records(
        [
            {
                "method_id": "other",
                "method_version": "1",
                "impact_category_id": "other_category",
                "impact_category_tuple_json": '["Other","category"]',
                "unit": "kg",
                "factor_identity": "other_factor",
            }
        ]
    )
    method = ("test method", "climate change")
    factors = [
        (("biosphere", "second"), 1.5),
        (("biosphere", "first"), -0.5, "US"),
        (("biosphere", "first"), 0.0),
    ]
    expected_factors = [
        {"flow_database": "biosphere", "flow_code": "first", "amount": 0.0, "location": None},
        {"flow_database": "biosphere", "flow_code": "first", "amount": -0.5, "location": "US"},
        {"flow_database": "biosphere", "flow_code": "second", "amount": 1.5, "location": None},
    ]
    if with_uncertainty:
        for scale in (0.25, 0.5):
            factors.append(
                (
                    ("biosphere", "second"),
                    {"amount": 1.5, "uncertainty type": np.int64(3), "loc": 1.5, "scale": scale},
                )
            )
            expected_factors.append(
                {
                    "flow_database": "biosphere",
                    "flow_code": "second",
                    "amount": 1.5,
                    "location": None,
                    "uncertainty": {"uncertainty type": 3, "loc": 1.5, "scale": scale},
                }
            )
    with selected_brightway_project(names):
        bd.Database("biosphere").write(
            {
                ("biosphere", code): {
                    "name": code,
                    "unit": "kg",
                    "type": "emission",
                    "categories": ("air",),
                }
                for code in ("first", "second")
            },
            searchable=False,
        )
        registered = bd.Method(method)
        registered.register(unit="kg CO2-Eq")
        registered.write(deepcopy(factors))
        stored_before = registered.load()

    selected = _method_selection(names, authority, normalized_lcia_method(method))
    assert len(selected) == 1
    assert selected[0].method_tuple == method
    assert selected[0].unit == "kg CO2-Eq"
    assert selected[0].factor_identity == payload_sha256(
        {"method_tuple": method, "unit": "kg CO2-Eq", "factors": expected_factors}
    )
    assert bd.projects.current == previous_project
    assert bd.projects.dir.parent == previous_store

    with selected_brightway_project(names):
        assert bd.Method(method).load() == stored_before
        reversed_factors = [
            (line[0], dict(reversed(tuple(line[1].items()))), *line[2:])
            if isinstance(line[1], dict)
            else line
            for line in reversed(factors)
        ]
        bd.Method(method).write(reversed_factors)
        reversed_before = bd.Method(method).load()
    assert _method_selection(names, authority, normalized_lcia_method(method)) == selected
    with selected_brightway_project(names):
        assert bd.Method(method).load() == reversed_before

    if with_uncertainty:
        changed = deepcopy(factors)
        changed[-1][1]["scale"] = 0.75
        with selected_brightway_project(names):
            bd.Method(method).write(changed)
        assert (
            _method_selection(names, authority, normalized_lcia_method(method))[0].factor_identity
            != selected[0].factor_identity
        )

    before_unit_change = _method_selection(names, authority, normalized_lcia_method(method))[0]
    with selected_brightway_project(names):
        registered = bd.Method(method)
        stored_before_unit_change = registered.load()
        # Correct the unit label and preserve characterization-factor values.
        registered.metadata = {**registered.metadata, "unit": "kg CO2 eq"}
    after_unit_change = _method_selection(names, authority, normalized_lcia_method(method))[0]

    assert before_unit_change.unit == "kg CO2-Eq"
    assert after_unit_change.unit == "kg CO2 eq"
    assert after_unit_change.method_tuple == before_unit_change.method_tuple
    assert after_unit_change.factor_identity != before_unit_change.factor_identity
    with selected_brightway_project(names):
        assert bd.Method(method).load() == stored_before_unit_change


@pytest.mark.parametrize("response_lost", [False, True])
def test_calculation_process_failure_without_traceback_keeps_concise_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_lost: bool,
) -> None:
    """Report a subprocess failure even when it cannot return a traceback."""

    def failed_process(
        command: list[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Write the bounded child response returned by a failed interpreter."""

        assert check is False
        if response_lost:
            directory = tmp_path / "result"
            directory.mkdir()
            (directory / "manifest.json").write_text("complete\n", encoding="utf-8")
        else:
            Path(command[-1]).write_text(
                '{"error_type":"MemoryError","message":"not enough memory"}\n',
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", failed_process)

    error = "calculation process failure" if response_lost else "MemoryError: not enough memory"
    with pytest.raises(RuntimeError, match=error) as caught:
        _isolated_product(
            tmp_path,
            _Request("scenario", "remind_ssp2_npi", tmp_path / "result"),
            normalized_lcia_method("ef31"),
            (2019,),
        )

    assert getattr(caught.value, "__notes__", []) == []
    assert "No completed LCA result was recorded" not in str(caught.value)
    if response_lost:
        assert (tmp_path / "result/manifest.json").read_text(encoding="utf-8") == "complete\n"
