"""Real source APIs for bounded stock and trade boundary tests."""

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import plotly.io as pio
import pytest
from remind_mfa_source_fixtures import write_remind_mfa_inputs

from pytoce.run_mfa.remind_mfa.execution import (
    RemindMFAExecutionRequest,
    RemindMFASource,
    _import_source_module,
)


@pytest.fixture
def remind_trade_modules(
    remind_mfa_source_root: Path, tmp_path: Path
) -> tuple[ModuleType, ModuleType]:
    """Load the actual Trade and extrapolation owners through source admission."""
    source = RemindMFASource(
        code_root=remind_mfa_source_root,
        data_root=tmp_path / "inputs",
        code_archive_sha256="eee4f40d0c8512c6edd8d5e7a7993e5708f3bfc13a31e72cbe42ff26a59fea24",
        data_archive_sha256="",
        source_manifest=tmp_path / "manifest.json",
    )
    return (
        _import_source_module(source, "remind_mfa.common.trade"),
        _import_source_module(source, "remind_mfa.common.trade_extrapolation"),
    )


@pytest.fixture
def remind_source_request(
    remind_mfa_source_root: Path, tmp_path: Path
) -> Iterator[RemindMFAExecutionRequest]:
    """Provide the actual source with two-region, two-polymer CSV inputs."""
    renderer = pio.renderers.default
    try:
        yield write_remind_mfa_inputs(
            remind_mfa_source_root,
            tmp_path / "source_inputs",
            regions=("EUR", "USA"),
            materials=("HDPE", "PP"),
        )
    finally:
        pio.renderers.default = renderer
