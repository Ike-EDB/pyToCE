"""Temporary Brightway projects for tests of impact contributions."""

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)


@pytest.fixture
def contribution_brightway(tmp_path: Path) -> Iterator[ModuleType]:
    """Select a fresh project confined to this test's temporary directory."""
    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="contribution_fixture",
        biosphere_database="biosphere",
        source_database="background",
    )
    with selected_brightway_project(names) as bd:
        yield bd
