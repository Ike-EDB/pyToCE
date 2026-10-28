"""Wheel, source-distribution, and clean-import release boundary."""

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def test_distributions_have_only_the_public_release_contents(tmp_path: Path) -> None:
    """Build and inspect both archives, then clean-import the installed wheel."""

    root = Path(__file__).resolve().parents[2]
    source_root = tmp_path / "source"
    tracked = (
        subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .split("\0")
    )
    for relative_name in tracked:
        if not relative_name:
            continue
        source = root / relative_name
        if not source.is_file():
            continue
        destination = source_root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    distribution_directory = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(distribution_directory),
        ],
        cwd=source_root,
        check=True,
    )

    wheel = next(distribution_directory.glob("*.whl"))
    source_distribution = next(distribution_directory.glob("*.tar.gz"))
    uslci_notice = "pytoce/prepare_backgrounds/data/USLCI_DATA_USE_NOTICE.txt"
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
        wheel_uslci_notice = archive.read(uslci_notice).decode("utf-8")
        root_notice_name = next(
            name for name in wheel_names if name.endswith(".dist-info/licenses/NOTICE")
        )
        wheel_root_notice = archive.read(root_notice_name).decode("utf-8")
    with tarfile.open(source_distribution, "r:gz") as archive:
        sdist_names = {member.name for member in archive.getmembers()}
        sdist_uslci_name = next(
            name
            for name in sdist_names
            if name.endswith("/src/pytoce/prepare_backgrounds/data/USLCI_DATA_USE_NOTICE.txt")
        )
        stream = archive.extractfile(sdist_uslci_name)
        assert stream is not None
        sdist_uslci_notice = stream.read().decode("utf-8")

    compact_default = (
        "pytoce/run_mfa/data/workspace/data_raw/"
        "foreground_scenarios/default_inputs/default_input_values.parquet"
    )
    waste_trade_propensities = "pytoce/run_mfa/data/catalogues/waste_trade_2019_propensities.csv"
    assert compact_default in wheel_names
    assert waste_trade_propensities in wheel_names
    assert (
        "pytoce/prepare_backgrounds/data/provider_recipes/provider_catalogue.json"
    ) in wheel_names
    assert (
        "pytoce/prepare_backgrounds/data/provider_recipes/chemical_recycling_authority.json"
    ) in wheel_names
    assert (
        "pytoce/prepare_backgrounds/data/provider_recipes/sbr_treatment_authority.json"
    ) in wheel_names
    assert (
        "pytoce/prepare_backgrounds/data/provider_recipes/hybrid_provider_temporal_authority.json"
    ) in wheel_names
    for name in (
        "h12_mismanaged_eol_fate_shares.csv",
        "material_fate_provider_mapping.csv",
        "provider_selector_catalogue.csv",
    ):
        assert f"pytoce/prepare_backgrounds/data/provider_recipes/{name}" in wheel_names
    assert not any(
        name.endswith("/h12_open_dump_infiltration_portfolio.csv")
        for name in (*wheel_names, *sdist_names)
    )
    assert uslci_notice in wheel_names
    for notice in (wheel_uslci_notice, sdist_uslci_notice):
        assert "0e42a306-ee2d-362e-8bc3-580000096459" in notice
        assert "provided by the National Renewable Energy Laboratory" in notice
        assert "provided that this entire notice appears in all copies" in notice
    assert "rev1.0.0_h12_1584d780_mfa" in wheel_root_notice
    assert "CC BY-SA 4.0" in wheel_root_notice
    assert "pytoce/__main__.py" not in wheel_names
    assert not any(name.endswith("entry_points.txt") for name in wheel_names)
    assert all(name.startswith("pytoce/") or ".dist-info/" in name for name in wheel_names)
    assert not any("/tests/" in f"/{name}" for name in wheel_names)

    required_public_files = {
        "README.md",
        "LICENSE",
        "NOTICE",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        ".readthedocs.yaml",
    }
    assert all(
        any(name.endswith(f"/{item}") for name in sdist_names) for item in required_public_files
    )
    assert sum(name.endswith(".ipynb") and "/tutorials/" in name for name in sdist_names) == 6
    assert sum(name.endswith(".nblink") for name in sdist_names) == 6
    assert any(
        item.endswith("/tutorials/optional_workflows/data/pyaesa_plastics/agg_reg_h12_plast.csv")
        for item in sdist_names
    )
    assert all(
        name.split("/", 1)[1].split("/", 1)[0]
        in required_public_files
        | {"src", "docs", "tutorials", "MANIFEST.in", "pyproject.toml", "PKG-INFO", "setup.cfg"}
        for name in sdist_names
        if "/" in name and name.split("/", 1)[1]
    )
    assert not any("/tests/" in f"/{name}" for name in sdist_names)
    assert not any(name.endswith("/src/pytoce/__main__.py") for name in sdist_names)
    install_directory = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_directory),
            str(wheel),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(install_directory)!r}); "
                "import pytoce as package; "
                "from pytoce import download_data, prepare_backgrounds, run_asr, run_lca, "
                "run_mfa, run_optimization, set_workspace; "
                "assert all(callable(item) for item in (set_workspace, download_data, "
                "prepare_backgrounds, run_mfa, run_lca, run_asr, run_optimization)); "
                "assert package.__all__ == "
                "['set_workspace', 'download_data', 'prepare_backgrounds', 'run_mfa', "
                "'run_lca', 'run_asr', 'run_optimization']"
            ),
        ],
        cwd=tmp_path,
        check=True,
    )
