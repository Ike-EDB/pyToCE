from importlib import resources
from typing import Any


def _packaged_files(root: Any, prefix: str = "") -> tuple[str, ...]:
    files: list[str] = []
    for child in root.iterdir():
        if child.name == "__pycache__":
            continue
        relative = f"{prefix}/{child.name}".lstrip("/")
        if child.is_dir():
            files.extend(_packaged_files(child, relative))
        elif child.is_file():
            files.append(relative)
    return tuple(files)


def test_source_package_excludes_generated_provider_tables() -> None:
    package = resources.files("pytoce")
    packaged_files = set(_packaged_files(package))

    assert "prepare_backgrounds/data/provider_recipes/provider_catalogue.json" in packaged_files
    assert (
        "prepare_backgrounds/data/provider_recipes/chemical_recycling_authority.json"
        in packaged_files
    )
    assert (
        "prepare_backgrounds/data/provider_recipes/sbr_treatment_authority.json" in packaged_files
    )
    assert (
        "prepare_backgrounds/data/provider_recipes/hybrid_provider_temporal_authority.json"
        in packaged_files
    )
    assert "run_mfa/data/catalogues/waste_trade_2019_propensities.csv" in packaged_files
    assert not any(
        path.endswith(
            (
                "provider_recipe_nodes.parquet",
                "provider_recipe_exchanges.parquet",
                "provider_route_components.parquet",
                "provider_bridge_crosswalk.parquet",
            )
        )
        for path in packaged_files
    )
