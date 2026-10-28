"""Check workflow signatures and lazy imports."""

import subprocess
import sys
from pathlib import Path

import pytest

_PUBLIC_WORKFLOWS = (
    "set_workspace",
    "download_data",
    "prepare_backgrounds",
    "run_mfa",
    "run_lca",
    "run_asr",
    "run_optimization",
)


def test_root_import_is_lazy_and_exports_exact_public_workflows() -> None:
    """Verify workflow signatures, defaults, and lazy dependency loading."""
    script = r"""
import inspect
import sys

scientific_prefixes = (
    "bw2data",
    "bw2io",
    "bw_timex",
    "flodym",
    "ixmp4",
    "madrat",
    "matplotlib",
    "mrmfa",
    "numpy",
    "pandas",
    "plotly",
    "premise",
    "pyam",
    "pyarrow",
    "pydantic",
    "scipy",
    "statsmodels",
)
implementation_prefixes = (
    "pytoce.download_data.sources",
    "pytoce.prepare_backgrounds.preparation",
    "pytoce.run_lca.calculation",
    "pytoce.run_mfa.model",
    "pytoce.run_mfa.remind_mfa",
    "pytoce.run_mfa.scenario",
    "pytoce.run_optimization.calculation",
    "pytoce.set_workspace.runtime",
)
already_loaded = {
    name for name in sys.modules if name.startswith(scientific_prefixes)
}
import pytoce as package
assert package.__all__ == [
    "set_workspace",
    "download_data",
    "prepare_backgrounds",
    "run_mfa",
    "run_lca",
    "run_asr",
    "run_optimization",
]
assert set(package.__all__).issubset(dir(package))
try:
    package.unknown_workflow
except AttributeError as error:
    assert "unknown_workflow" in str(error)
else:
    raise AssertionError("An unknown package workflow must raise AttributeError")
assert "pytoce.run_optimization" not in sys.modules
assert "pytoce.set_workspace.set_workspace" not in sys.modules
assert {
    name for name in sys.modules if name.startswith(scientific_prefixes)
} == already_loaded

function = package.set_workspace
assert "pytoce.set_workspace.set_workspace" in sys.modules
signature = inspect.signature(function)
assert list(signature.parameters) == ["workspace", "refresh"]
assert signature.parameters["refresh"].default is False

download = package.download_data
assert "pytoce.download_data.download_data" in sys.modules
download_signature = inspect.signature(download)
assert list(download_signature.parameters) == ["refresh"]
assert download_signature.parameters["refresh"].default is False

backgrounds = package.prepare_backgrounds
assert "pytoce.prepare_backgrounds.prepare_backgrounds" in sys.modules
background_signature = inspect.signature(backgrounds)
assert list(background_signature.parameters) == [
    "background_scenario", "premise_key"
]
assert background_signature.parameters["background_scenario"].default == (
    "remind_ssp2_ndc",
    "remind_ssp2_npi",
    "remind_ssp2_pkbudg1000",
    "remind_ssp2_pkbudg650",
)
assert background_signature.parameters["premise_key"].default is None

mfa = package.run_mfa
assert "pytoce.run_mfa.run_mfa" in sys.modules
mfa_signature = inspect.signature(mfa)
assert list(mfa_signature.parameters) == ["foreground_scenario", "refresh"]
assert mfa_signature.parameters["foreground_scenario"].default == "remind_mfa_default"
assert mfa_signature.parameters["refresh"].default is False

lca = package.run_lca
assert "pytoce.run_lca.run_lca" in sys.modules
lca_signature = inspect.signature(lca)
assert list(lca_signature.parameters) == [
    "foreground_scenario",
    "background_scenario",
    "lcia_method",
    "upstream_analysis_year",
    "refresh",
]
assert lca_signature.parameters["foreground_scenario"].default == "remind_mfa_default"
assert lca_signature.parameters["background_scenario"].default == (
    "remind_ssp2_ndc",
    "remind_ssp2_npi",
    "remind_ssp2_pkbudg1000",
    "remind_ssp2_pkbudg650",
)
assert lca_signature.parameters["lcia_method"].default == (
    "ef31",
    "pb_lcia",
)
assert lca_signature.parameters["upstream_analysis_year"].default == (2019, 2050, 2100)
assert lca_signature.parameters["refresh"].default is False

asr = package.run_asr
assert "pytoce.run_asr.run_asr" in sys.modules
asr_signature = inspect.signature(asr)
assert list(asr_signature.parameters) == [
    "allocated_cc_result", "foreground_scenario", "background_scenario",
    "lcia_method", "allocated_cc_case", "refresh",
]
assert asr_signature.parameters["allocated_cc_result"].default is inspect.Parameter.empty
assert asr_signature.parameters["foreground_scenario"].default == "remind_mfa_default"
assert asr_signature.parameters["background_scenario"].default == (
    lca_signature.parameters["background_scenario"].default
)
assert asr_signature.parameters["lcia_method"].default == ("ef31", "pb_lcia")
assert asr_signature.parameters["allocated_cc_case"].default is None
assert asr_signature.parameters["refresh"].default is False

optimization = package.run_optimization
assert "pytoce.run_optimization.run_optimization" in sys.modules
optimization_signature = inspect.signature(optimization)
assert list(optimization_signature.parameters) == [
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
]
assert optimization_signature.parameters["foreground_scenario"].default is inspect.Parameter.empty
assert optimization_signature.parameters["allocated_cc_result"].default is inspect.Parameter.empty
assert optimization_signature.parameters["optimization_lcia_method"].default == "ef31"
assert optimization_signature.parameters["lcia_method"].default == ("ef31", "pb_lcia")
assert optimization_signature.parameters["allocated_cc_case"].default is None
assert optimization_signature.parameters["refresh"].default is False

assert str(signature) == (
    "(workspace: str | pathlib.Path, *, refresh: bool = False) -> "
    "pytoce.set_workspace.set_workspace.WorkspaceReport"
)
assert str(download_signature) == (
    "(*, refresh: bool = False) -> "
    "pytoce.download_data.download_data.DownloadDataReport"
)
assert str(background_signature) == (
    "(*, background_scenario: str | collections.abc.Sequence[str] = "
    "('remind_ssp2_ndc', 'remind_ssp2_npi', 'remind_ssp2_pkbudg1000', "
    "'remind_ssp2_pkbudg650'), "
    "premise_key: str | None = None) -> "
    "pytoce.prepare_backgrounds.prepare_backgrounds."
    "PrepareBackgroundsReport"
)
assert str(mfa_signature) == (
    "(*, foreground_scenario: str | collections.abc.Sequence[str] = "
    "'remind_mfa_default', refresh: bool = False) -> "
    "pytoce.run_mfa.run_mfa.RunMFAReport"
)
assert str(lca_signature) == (
    "(*, foreground_scenario: str | collections.abc.Sequence[str] = "
    "'remind_mfa_default', background_scenario: str | "
    "collections.abc.Sequence[str] = ('remind_ssp2_ndc', 'remind_ssp2_npi', "
    "'remind_ssp2_pkbudg1000', 'remind_ssp2_pkbudg650'), lcia_method: str | "
    "collections.abc.Sequence[str] = ('ef31', 'pb_lcia'), "
    "upstream_analysis_year: int | collections.abc.Sequence[int] = "
    "(2019, 2050, 2100), refresh: bool = False) -> "
    "pytoce.run_lca.run_lca.RunLCAReport"
)
assert str(optimization_signature) == (
    "(*, foreground_scenario: str | collections.abc.Sequence[str], "
    "allocated_cc_result: str, background_scenario: str | "
    "collections.abc.Sequence[str] | None = None, "
    "optimization_lcia_method: str | "
    "collections.abc.Sequence[str] = 'ef31', lcia_method: str | "
    "collections.abc.Sequence[str] = ('ef31', 'pb_lcia'), "
    "optimization_block: Union["
    "Literal['demand_lifetime_frontier', 'circularity_feedstock_anchor', "
    "'integrated_frontier'], collections.abc.Sequence["
    "Literal['demand_lifetime_frontier', 'circularity_feedstock_anchor', "
    "'integrated_frontier']]] = ('demand_lifetime_frontier', "
    "'circularity_feedstock_anchor', 'integrated_frontier'), "
    "allocated_cc_case: str | collections.abc.Sequence[str] | None = None, "
    "initial_grid_step_pct: int = 20, "
    "endpoint_refinement_steps_pct: collections.abc.Sequence[int] = (10, 5), "
    "candidate_workers: int = 1, refresh: bool = False) -> "
    "pytoce.run_optimization.run_optimization.RunOptimizationReport"
)

assert {
    name for name in sys.modules if name.startswith(scientific_prefixes)
} == already_loaded
assert not {
    name for name in sys.modules if name.startswith(implementation_prefixes)
}
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_distribution_declares_no_cli_or_entry_point() -> None:
    """Verify that the distribution provides no command line interface or entry point."""
    checkout = Path(__file__).resolve().parents[2]
    source = checkout / "src" / "pytoce"
    assert not (source / "cli.py").exists()
    assert not (source / "__main__.py").exists()
    assert not any(
        'if __name__ == "__main__"' in path.read_text(encoding="utf-8")
        for path in source.rglob("*.py")
    )
    assert "[project.scripts]" not in (checkout / "pyproject.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize("preimported", _PUBLIC_WORKFLOWS)
@pytest.mark.parametrize("requested", _PUBLIC_WORKFLOWS)
def test_root_callable_survives_importing_a_subpackage_first(
    preimported: str, requested: str
) -> None:
    """Resolve every import order with workflow-specific lazy loading."""

    script = """
import importlib
import sys
import pytoce as package
preimported, requested = sys.argv[1:]
importlib.import_module(f"pytoce.{preimported}")
function = getattr(package, requested)
assert callable(function)
assert getattr(package, requested) is function
assert {
    name for name in package.__all__
    if f"pytoce.{name}.{name}" in sys.modules
} == {requested}
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, preimported, requested],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


@pytest.mark.parametrize("requested", _PUBLIC_WORKFLOWS)
def test_preimporting_all_subpackages_keeps_other_workflows_lazy(requested: str) -> None:
    """Load only the requested public workflow after importing all subpackages."""

    script = """
import importlib
import sys
import pytoce as package
for name in package.__all__:
    importlib.import_module(f"pytoce.{name}")
requested = sys.argv[1]
assert callable(getattr(package, requested))
assert {
    name for name in package.__all__
    if f"pytoce.{name}.{name}" in sys.modules
} == {requested}
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, requested],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
