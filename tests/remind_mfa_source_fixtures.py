"""Synthetic CSV inputs for the REMIND-MFA 0.2.0 dependency."""

import csv
import sys
from collections.abc import Iterator
from dataclasses import replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import plotly.io as pio
import pytest
import yaml

from pytoce.run_mfa.remind_mfa.execution import (
    RemindMFAExecutionRequest,
    RemindMFASource,
    _import_source_module,
    prepare_execution,
)


@pytest.fixture
def external_remind_mfa_import_state() -> Iterator[None]:
    """Restore dependency modules, import paths and renderer after a disposable source run."""
    original_path = list(sys.path)
    renderer = pio.renderers.default
    saved = {
        name: module
        for name, module in sys.modules.copy().items()
        if name == "remind_mfa" or name.startswith("remind_mfa.")
    }
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "remind_mfa" or name.startswith("remind_mfa."):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = original_path
        pio.renderers.default = renderer


def write_remind_mfa_inputs(
    source_root: Path,
    data_root: Path,
    *,
    regions: tuple[str, ...],
    materials: tuple[str, ...],
    bio_production_rate: float = 0.05,
) -> RemindMFAExecutionRequest:
    """Write invented 1950--2100 CSV inputs using REMIND-MFA 0.2.0 parameter definitions.

    HDPE and PP partition product use; other required catalog materials have
    zero shares. Coordinates can be the small unit-test set or the complete
    public-workflow set. The source code is read-only and its exports are disabled.
    """
    coordinates = {
        "t": list(range(1950, 2101)),
        "h": list(range(1950, 2020)),
        "r": list(regions),
        "e": ["C", "Other Elements"],
        "m": list(materials),
        "g": [
            "Buildings & Construction",
            "Household & Leisure & Sports",
            "Electrical/Electronics",
            "Packaging",
            "Textile sector",
            "Transportation",
            "Others",
        ],
    }
    tmp_path = data_root.parent
    data = data_root
    dimensions = data / "plastics" / "dimensions"
    inputs = data / "plastics" / "input_data"
    dimensions.mkdir(parents=True)
    inputs.mkdir()
    for letter, filename in (
        ("t", "time_in_years"),
        ("h", "historic_years"),
        ("e", "elements"),
        ("m", "materials"),
        ("g", "goods_in_use"),
    ):
        with (dimensions / (filename + ".csv")).open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows((item,) for item in coordinates[letter])
    (inputs / "regionmappingH12.csv").write_text(
        "RegionCode\n" + "\n".join(regions) + "\n", encoding="utf-8"
    )
    (inputs / "version.txt").write_text("rev1.0.0_h12_1584d780_mfa", encoding="utf-8")
    source = RemindMFASource(
        code_root=source_root,
        data_root=data,
        code_archive_sha256="eee4f40d0c8512c6edd8d5e7a7993e5708f3bfc13a31e72cbe42ff26a59fea24",
        data_archive_sha256="",
        source_manifest=tmp_path / "synthetic_source.json",
    )
    request = prepare_execution(source)
    config = yaml.safe_load(request.configuration_yaml)
    config["export"]["path"] = str(tmp_path / "disabled_exports")
    config["export"]["docs"]["path"] = str(tmp_path / "disabled_docs")
    config["visualization"]["figures_path"] = str(tmp_path / "disabled_figures")
    cfg_type = _import_source_module(source, "remind_mfa.plastics.plastics_config").PlasticsCfg
    definition_api = _import_source_module(source, "remind_mfa.plastics.plastics_definition")
    definition = definition_api.get_plastics_definition(cfg_type(**config), historic=False)
    shares = np.zeros((len(regions), len(materials), 7))
    shares[:, materials.index("HDPE"), :] = np.linspace(0.2, 0.8, 7)
    shares[:, materials.index("PP"), :] = 1 - shares[:, materials.index("HDPE"), :]
    population_weights = 1 + 0.3 * np.arange(len(regions))
    income_weights = 1 + 0.2 * np.arange(len(regions))
    nominal_carbon = np.linspace(0.8, 0.85, len(materials))
    arrays = {
        "consumption": (1 + np.arange(70)[:, None, None] / 50)
        * population_weights[None, :, None]
        * np.arange(1.0, 8.0)[None, None, :],
        "population": (1000 + np.arange(151)[:, None] * 5) * population_weights,
        "gdppc": (1000 * np.exp(np.arange(151)[:, None] * 0.02)) * income_weights,
        "lifetime_mean": np.array([35.0, 3.0, 8.0, 0.5, 5.0, 13.0, 5.0]),
        "lifetime_std": np.array([7.0, 1.0, 2.0, 0.1, 1.5, 3.0, 1.5]),
        "material_shares_in_goods": shares,
        "carbon_content_materials": np.stack(
            (nominal_carbon, np.linspace(0.2, 0.15, len(materials)))
        ),
    }
    constant_rates = {
        "collection_rate": 0.65,
        "mechanical_recycling_rate": 0.2,
        "chemical_recycling_rate": 0.1,
        "incineration_rate": 0.3,
        "bio_production_rate": bio_production_rate,
        "daccu_production_rate": 0.0,
        "mechanical_recycling_yield": 0.8,
        "reclmech_loss_uncontrolled_rate": 0.25,
        "emission_capture_rate": 0.0,
        "waste_imports": 0.0,
        "waste_exports": 0.0,
    }
    constant_rates.update(
        {
            f"{market}_his_{direction}": 0.0
            for market in ("primary", "intermediate", "manufactured", "final")
            for direction in ("imports", "exports")
        }
    )
    names = {item.letter: item.name for item in definition.dimensions}
    for parameter in definition.parameters:
        letters = parameter.dim_letters
        values = arrays.get(parameter.name)
        if values is None:
            values = np.full(
                tuple(len(coordinates[letter]) for letter in letters),
                constant_rates[parameter.name],
            )
        with (inputs / f"pl_{parameter.name}.cs4r").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            stream.write(
                "* dimensions: (" + ", ".join(names[letter] for letter in letters) + ", value)\n"
            )
            csv.writer(stream).writerows(
                (*coordinate, amount)
                for coordinate, amount in zip(
                    product(*(coordinates[letter] for letter in letters)),
                    values.ravel(),
                    strict=True,
                )
            )
    archive = tmp_path / "synthetic_source.zip"
    with ZipFile(archive, "w") as output:
        for path in sorted(data.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(data).as_posix())
    data_hash = sha256(archive.read_bytes()).hexdigest()
    source.source_manifest.write_text('{"fixture": "synthetic_csv_inputs"}\n', encoding="utf-8")
    return replace(
        request,
        source=replace(source, data_archive_sha256=data_hash),
        configuration_yaml=yaml.safe_dump(config),
        identity=replace(request.identity, input_archive_sha256=data_hash),
    )
