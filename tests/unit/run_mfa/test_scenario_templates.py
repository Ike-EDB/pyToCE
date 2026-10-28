import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pytoce.run_mfa.scenario.catalogue import load_parameter_catalogue
from pytoce.run_mfa.scenario.compiler import compile_foreground_scenario
from pytoce.set_workspace.set_workspace import set_workspace


def _scenario_documents(templates: Path) -> list[tuple[Path, dict[str, object]]]:
    paths = [templates / "foreground_scenario.yaml"]
    paths.extend(sorted((templates / "examples").glob("*.yaml")))
    documents: list[tuple[Path, dict[str, object]]] = []
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        documents.append((path, payload))
    return documents


def test_yaml_examples_cover_the_complete_editable_catalogue(tmp_path: Path) -> None:
    report = set_workspace(tmp_path)
    definitions = load_parameter_catalogue()
    assert isinstance(definitions, tuple)
    editable = {
        definition.parameter_id for definition in definitions if definition.role == "editable_input"
    }
    documents = _scenario_documents(report.foreground_scenario_templates)
    examples = documents[1:]

    guide = (report.foreground_scenario_templates / "README_foreground_scenarios.md").read_text(
        encoding="utf-8"
    )
    parameter_reference = guide.split("## Editable parameter reference", maxsplit=1)[1].split(
        "## How changes affect the MFA",
        maxsplit=1,
    )[0]
    documented = {
        match.group(1)
        for match in re.finditer(
            r"^\| `([a-z0-9_]+)` \|",
            parameter_reference,
            flags=re.MULTILINE,
        )
    }
    example_parameters = {
        str(entry["parameter"])
        for _path, payload in examples
        for entry in payload["changes"]
        if "parameter" in entry
    }
    assert documented == editable
    # Landfill and fossil shares complete each group but are derived outputs.
    grouped_parameters = {
        parameter
        for _path, payload in examples
        for entry in payload["changes"]
        if "parameter_group" in entry
        for parameter in entry["trajectory"]["target_values"]
        if parameter in editable
    }
    assert example_parameters | grouped_parameters == editable

    example_modes = {
        str(change["mode"])
        for _path, payload in examples
        for entry in payload["changes"]
        for change in entry.get("changes", [])
    }
    example_modes.update(
        str(entry["trajectory"]["target_mode"])
        for _path, payload in examples
        for entry in payload["changes"]
        if "parameter" in entry and "trajectory" in entry
    )
    baseline = yaml.safe_load(
        (report.foreground_scenarios / "frozen_2019_default_baseline.yaml").read_text(
            encoding="utf-8"
        )
    )
    example_modes.update(
        str(change["mode"]) for entry in baseline["changes"] for change in entry.get("changes", [])
    )
    catalogue_modes = {
        mode
        for definition in definitions
        if definition.role == "editable_input"
        for mode in definition.value_modes
    }
    assert example_modes == catalogue_modes

    example_selectors = {
        str(entry["selector"])
        for _path, payload in examples
        for entry in payload["changes"]
        if "selector" in entry
    }
    catalogue_selectors = {
        selector for definition in definitions for selector in definition.broad_selectors
    }
    assert example_selectors.issubset(catalogue_selectors)
    assert {"uniform_all_h12", "uniform_all_h12_goods"}.issubset(example_selectors)
    assert {
        str(change.get("interpolation", "step"))
        for _path, payload in examples
        for entry in payload["changes"]
        for change in entry.get("changes", [])
    } == {"step"}
    assert {
        str(entry["trajectory"]["mode"])
        for _path, payload in examples
        for entry in payload["changes"]
        if "trajectory" in entry
    } == {"linear_from_2019"}

    coordinate_syntax = {
        key
        for _path, payload in examples
        for entry in payload["changes"]
        for key in ("region", "regions", "material", "materials", "good", "goods")
        if key in entry
    }
    assert coordinate_syntax == {
        "region",
        "regions",
        "materials",
        "good",
        "goods",
    }


@pytest.mark.scientific_integration
def test_every_focused_yaml_example_parses_and_compiles(tmp_path: Path) -> None:
    report = set_workspace(tmp_path)
    for source, payload in _scenario_documents(report.foreground_scenario_templates)[1:]:
        foreground_scenario = str(payload["foreground_scenario"])
        target = report.foreground_scenarios / f"{foreground_scenario}.yaml"
        shutil.copyfile(source, target)
        compiled = compile_foreground_scenario(tmp_path, foreground_scenario)
        assert compiled.foreground_scenario == foreground_scenario
        assert compiled.compiled_parameters_path.is_file()


def test_template_guide_indexes_existing_examples(tmp_path: Path) -> None:
    templates = set_workspace(tmp_path).foreground_scenario_templates
    guide = (templates / "README_foreground_scenarios.md").read_text(encoding="utf-8")
    linked = {match.group(1) for match in re.finditer(r"\]\((examples/[0-9a-z_]+\.yaml)\)", guide)}
    existing = {
        path.relative_to(templates).as_posix() for path in (templates / "examples").glob("*.yaml")
    }
    assert linked == existing

    coverage = guide
    definitions = load_parameter_catalogue()
    editable = {
        definition.parameter_id for definition in definitions if definition.role == "editable_input"
    }
    for parameter_id in editable | {"landfill_share"}:
        assert f"`{parameter_id}`" in coverage
    for mode in {
        mode
        for definition in definitions
        if definition.role == "editable_input"
        for mode in definition.value_modes
    }:
        assert f"`{mode}`" in coverage
    for selector in {
        selector for definition in definitions for selector in definition.broad_selectors
    }:
        assert f"`{selector}`" in coverage
    for coordinate in ("region", "regions", "material", "materials", "good", "goods"):
        assert f"`{coordinate}`" in coverage
    for interpolation in ("step", "linear_from_2019"):
        assert f"`{interpolation}`" in coverage


def test_readable_default_lookup_exactly_matches_compiler_authority(
    tmp_path: Path,
) -> None:
    report = set_workspace(tmp_path)
    lookup = pd.read_csv(
        report.default_annual_input_values,
        keep_default_na=False,
        float_precision="round_trip",
    )
    authority = pd.read_parquet(report.default_input_parameters)
    manifest = json.loads(
        report.default_input_parameters.with_name("default_input_values_manifest.json").read_bytes()
    )

    assert list(lookup.columns) == [
        "parameter_id",
        "year",
        "region",
        "material",
        "good",
        "process_or_route",
        "unit",
        "default_value",
        "default_source_identity",
    ]
    assert len(lookup) == len(authority) == 100_532
    assert set(lookup["default_source_identity"]) == {manifest["identity"]}
    assert not lookup["process_or_route"].any()
    for column in ("parameter_id", "year", "region", "material", "good", "unit"):
        expected = authority[column].fillna("").to_numpy()
        actual = lookup[column].to_numpy()
        assert np.array_equal(actual, expected), column
    assert np.array_equal(
        lookup["default_value"].to_numpy(dtype=float),
        authority["default_value"].to_numpy(dtype=float),
    )
