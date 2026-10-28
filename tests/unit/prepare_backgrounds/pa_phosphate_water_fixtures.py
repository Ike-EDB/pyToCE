"""Use synthetic Swiss treatment inventories for PA mixed-water tests.

No wastewater, gas or sludge operation is replaced by a fixture-produced
package record. All receiving declarations come from the real packaged
catalog; the package compiles each operation through its existing owner.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from material_recipe_fixtures import physical_material_recipe
from native_chemical_fixtures import native_chemical_biosphere, native_chemical_inventory

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _service_source_closures,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext


def phosphate_water_inputs() -> dict[str, float]:
    """Supply collected wash per kg final triethanolamine monooleylphosphate."""
    return {
        "water_kg": 2.5833167850919087,
        "sodium_chloride_kg": 0.412037118360841,
        "monooleylphosphate_kg": 0.0778014731007121,
        "phosphoric_acid_kg": 0.0115152939537735,
    }


@contextmanager
def phosphate_water_boundary(
    root: Path,
) -> Iterator[tuple[Any, _RecipeContext]]:
    """Create isolated synthetic Swiss source inventories for the mixed-water calculation."""
    provider = prepared_provider_authority()
    names = BrightwayNames(root / "brightway", "pa_phosphate_water", "biosphere", "source")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(native_chemical_biosphere(), searchable=False)
        bd.Database("source").write(
            native_chemical_inventory(provider.catalogue, "source"), searchable=False
        )
        native = _ActivityResolver(bd, "source")
        context = _RecipeContext(
            background_scenario=None,
            year=2025,
            provider_database="providers",
            source=native,
            ordinary=native,
            technology=native,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=provider.catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                str(row["method_authority_id"]): row for row in provider.method_authorities
            },
            complete_source_ids=_service_source_closures(provider.catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        yield bd, context
