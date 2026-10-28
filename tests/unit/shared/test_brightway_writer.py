"""Database equivalence and preservation after an interrupted Brightway write."""

from pathlib import Path

import pytest

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.shared.runtime.brightway import (
    database_matches,
    write_brightway_database,
)


def _database_data(database_name: str) -> dict[tuple[str, str], dict[str, object]]:
    """Supply linked nodes with complete admitted exchange metadata."""
    return {
        (database_name, "input"): {
            "name": "bounded input",
            "reference product": "bounded input",
            "unit": "kilogram",
            "location": "GLO",
            "type": "process",
            "exchanges": [
                {
                    "input": (database_name, "input"),
                    "amount": 1.0,
                    "type": "production",
                }
            ],
        },
        (database_name, "output"): {
            "name": "bounded output",
            "reference product": "bounded output",
            "unit": "kilogram",
            "location": "GLO",
            "type": "process",
            "exchanges": [
                {
                    "input": (database_name, "output"),
                    "amount": 1.0,
                    "type": "production",
                },
                {
                    "input": (database_name, "input"),
                    "amount": 2.5,
                    "type": "technosphere",
                    "unit": "kilogram",
                    "uncertainty type": 3,
                    "loc": 2.5,
                    "scale": 0.25,
                    "minimum": 1.0,
                    "maximum": 4.0,
                    "negative": False,
                    "allocation": "invented source allocation",
                    "properties": {"dry mass": {"amount": 0.9}},
                },
            ],
        },
    }


def _canonical_database(bd: object, database_name: str) -> list[dict[str, object]]:
    """Compare scientific records while normalizing the two test database names."""
    rows = []
    for activity in bd.Database(database_name):
        exchanges = sorted(
            (
                {
                    **dict(exchange),
                    "input": ("fixture", str(exchange.input["code"])),
                    "output": ("fixture", str(activity["code"])),
                }
                for exchange in activity.exchanges()
            ),
            key=lambda row: (str(row["type"]), str(row["input"]), float(row["amount"])),
        )
        rows.append(
            {
                "code": str(activity["code"]),
                "name": str(activity["name"]),
                "reference_product": str(activity["reference product"]),
                "unit": str(activity["unit"]),
                "location": str(activity["location"]),
                "exchanges": exchanges,
            }
        )
    return sorted(rows, key=lambda row: str(row["code"]))


def test_compact_writer_matches_official_writer_on_bounded_database(
    tmp_path: Path,
) -> None:
    """Preserve exact node and exchange semantics across the compact writer."""

    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="bounded_writer_equivalence",
        biosphere_database="biosphere",
        source_database="source",
    )
    with selected_brightway_project(names) as bd:
        assert not database_matches(bd, "compact", {"scientific_identity": "bounded_equivalence"})
        bd.Database("official").write(
            _database_data("official"),
            searchable=False,
        )
        write_brightway_database(
            bd,
            "compact",
            _database_data("compact"),
            {"scientific_identity": "bounded_equivalence"},
        )

        assert _canonical_database(bd, "compact") == _canonical_database(
            bd,
            "official",
        )
        assert database_matches(
            bd,
            "compact",
            {"scientific_identity": "bounded_equivalence"},
        )
        assert not database_matches(
            bd,
            "compact",
            {"scientific_identity": "different"},
        )
        assert bd.databases["compact"]["scientific_identity"] == ("bounded_equivalence")

        with pytest.raises(RuntimeError, match="already exists and cannot be replaced"):
            write_brightway_database(
                bd,
                "compact",
                _database_data("compact"),
                {"scientific_identity": "replacement"},
            )

        assert bd.databases["compact"]["scientific_identity"] == "bounded_equivalence"


def test_compact_writer_preserves_a_database_after_processing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave an interrupted database unchanged so its state is not misrepresented."""

    names = BrightwayNames(
        store=tmp_path / "brightway",
        project="bounded_writer_failure",
        biosphere_database="biosphere",
        source_database="source",
    )
    with selected_brightway_project(names) as bd:
        database_type = type(bd.Database("broken"))

        def fail_process(_database: object) -> None:
            """Simulate a Brightway processing failure after data have been written."""
            raise RuntimeError("bounded Brightway processing failure")

        monkeypatch.setattr(database_type, "process", fail_process)
        with pytest.raises(RuntimeError, match="bounded Brightway processing failure"):
            write_brightway_database(
                bd,
                "broken",
                _database_data("broken"),
                {"scientific_identity": "incomplete"},
            )

        assert "broken" in bd.databases
        assert not database_matches(bd, "broken", {"scientific_identity": "incomplete"})
        with pytest.raises(RuntimeError, match="already exists and cannot be replaced"):
            write_brightway_database(
                bd,
                "broken",
                _database_data("broken"),
                {"scientific_identity": "replacement"},
            )
