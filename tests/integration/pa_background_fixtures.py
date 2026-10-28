"""Invented connected PA sources for prepared-member and response integration."""

from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import cast

import pandas as pd
import pytest
from openpyxl import Workbook

from pytoce.prepare_backgrounds.preparation.ecoinvent import BrightwayNames
from pytoce.prepare_backgrounds.preparation.providers import (
    pa_municipal_receiving_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    PreparedProviderAuthority,
    _service_dependencies,
    prepared_provider_authority,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_receiving_sources import (
    _SERVICES as LANDFILL_SERVICES,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingReference,
    PaReceivingSources,
    admit_pa_receiving_sources,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    _SERVICES as WATER_SERVICES,
)
from tests.unit.prepare_backgrounds.chemical_residue_fixtures import (
    apply_premise_chemical_receiving_metadata,
)
from tests.unit.prepare_backgrounds.pa_event_receiving_fixtures import (
    _receiving_biosphere,
    _receiving_inventory,
    pa_event_receiving_boundary,
)
from tests.unit.prepare_backgrounds.pa_recycling_receiving_fixtures import RECYCLING_SERVICES


@dataclass(frozen=True)
class PaBackgroundInputs:
    """Disposable real graph and its original reference plus two operating profiles."""

    names: BrightwayNames
    reference: PaReceivingReference
    original: PaReceivingSources
    future: PaReceivingSources


def prepare_pa_background_inputs(workspace: Path) -> PaBackgroundInputs:
    """Reuse the complete native-shaped PA fixture and actual admission owners."""
    root = workspace / "data_processed"
    with pa_event_receiving_boundary(root, original=False) as boundary:
        original = admit_pa_receiving_sources(
            replace(
                boundary.original_context,
                provider_database="frontier",
                technology=boundary.context.technology,
            ),
            boundary.reference,
        )
        return PaBackgroundInputs(
            BrightwayNames(root / "brightway", "pa_receiving", "biosphere", "original"),
            boundary.reference,
            original,
            boundary.sources,
        )


def pa_background_provider(authority: PreparedProviderAuthority) -> PreparedProviderAuthority:
    """Compile five used frontiers while retaining their actual PA source ancestry.

    PA admission reads source declarations for all receivers and both spinning
    providers. The fixture supplies their complete quantity and source closure.
    """
    frontiers = {
        "hydraulic_water_collection_row",
        "hydraulic_water_treatment_row",
        *RECYCLING_SERVICES,
    }
    services = {
        str(row["provider_service_id"]): row
        for row in cast(list[dict[str, object]], authority.catalogue["services"])
    }
    selected = {
        *frontiers,
        *WATER_SERVICES.values(),
        *LANDFILL_SERVICES.values(),
        *pa_municipal_receiving_sources._SERVICES.values(),
        "fibres_pa6_conversion",
        "fibres_pa66_conversion",
    }
    pending = list(selected)
    while pending:
        service = services[pending.pop()]
        construction = cast(dict[str, object], service["construction"])
        dependencies = (
            *_service_dependencies(service),
            *(
                str(construction[key])
                for key in ("formulation_service_id", "incineration_service_id")
                if key in construction
            ),
        )
        for dependency in dependencies:
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    return replace(
        authority,
        identity="provider::bounded_pa_receiving",
        catalogue={
            **authority.catalogue,
            "services": [
                {**row, "foreground_service": key in frontiers, "frontier_service": False}
                for key, row in services.items()
                if key in selected
            ],
            "routes": [],
        },
        routes=pd.DataFrame(),
    )


@pytest.fixture
def bounded_pa_premise(
    isolated_premise: ModuleType,
    inline_premise_process: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supply connected invented PA inputs at premise's public database writer."""
    catalogue = prepared_provider_authority().catalogue
    _, gas_keys = _receiving_biosphere()

    class BoundedPaNewDatabase:
        """Implement premise 2.4.6's constructor, update and database-write boundary."""

        def __init__(self, **kwargs: object) -> None:
            """Accept the production worker's real premise configuration."""

        def update(self, sectors: list[str]) -> None:
            """The invented future inventories already describe dated operations."""

        def write_db_to_brightway(self, name: list[str] | None = None) -> None:
            """Write complete sources and reports; package owners compile and admit PA."""
            import bw2data as bd

            assert name is not None
            for database in name:
                bd.Database(database).write(
                    _receiving_inventory(catalogue, database, future=True, gas_keys=gas_keys),
                    searchable=False,
                )
                apply_premise_chemical_receiving_metadata(bd, database)
            for report_kind in ("scenario_report", "change reports"):
                directory = Path.cwd() / "export" / report_kind
                directory.mkdir(parents=True)
                workbook = Workbook()
                workbook.active.append(["bounded connected PA premise report"])
                workbook.save(directory / "report.xlsx")
                workbook.close()

    monkeypatch.setattr(isolated_premise, "NewDatabase", BoundedPaNewDatabase)
