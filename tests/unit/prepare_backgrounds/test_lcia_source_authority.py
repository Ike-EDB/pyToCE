"""Failure contracts for licensed EF and open PB-LCIA source parsing."""

import json
import tarfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest
import yaml
from openpyxl import Workbook

from pytoce.prepare_backgrounds.preparation import project_biosphere as owner
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    brightway_names,
    selected_brightway_project,
)


def _ef_spec(*, unit: str = "kg CO2-Eq") -> owner.MethodSpec:
    """Return one bounded EF category authority."""
    return owner.MethodSpec(
        order=1,
        method_id="ef31",
        method_version="3.1",
        impact_category_id="climate_change",
        impact_category_tuple=("EF v3.1", "climate", "GWP100"),
        unit=unit,
        source_member="EF-v3.1.xlsx",
    )


def _write_ef_workbook(
    path: Path,
    *,
    indicators: list[tuple[object, ...]],
    factors: list[tuple[object, ...]] | None = None,
) -> None:
    """Write the smallest workbook matching the EF parser's public sheets."""
    book = Workbook()
    indicator_sheet = book.active
    indicator_sheet.title = "Indicators"
    indicator_sheet.append(("method", "category", "indicator", "unit"))
    for row in indicators:
        indicator_sheet.append(row)
    factor_sheet = book.create_sheet("CFs")
    factor_sheet.append(
        ("method", "category", "indicator", "name", "compartment", "subcompartment", "amount")
    )
    for row in factors or []:
        factor_sheet.append(row)
    book.save(path)
    book.close()


def _write_pb_archive(path: Path, members: dict[str, object]) -> None:
    """Write bounded YAML members through the real tar archive boundary."""
    with tarfile.open(path, mode="w:gz") as handle:
        for filename, rows in members.items():
            content = yaml.safe_dump(rows).encode("utf-8")
            member = tarfile.TarInfo(filename)
            member.size = len(content)
            handle.addfile(member, BytesIO(content))


@pytest.mark.parametrize(
    ("indicators", "message"),
    [
        (
            [("EF v3.1", "other", "GWP100", "kg CO2-Eq")],
            "impact-category tuples changed",
        ),
        (
            [("EF v3.1", "climate", "GWP100", "wrong unit")],
            "EF unit changed",
        ),
    ],
)
def test_ef_indicator_schema_changes_fail_before_factor_resolution(
    tmp_path: Path,
    indicators: list[tuple[object, ...]],
    message: str,
) -> None:
    """Reject a missing selected category or changed scientific unit."""
    workbook = tmp_path / "ef.xlsx"
    _write_ef_workbook(workbook, indicators=indicators)

    with pytest.raises(ValueError, match=message):
        owner.resolve_ef_methods(
            workbook,
            biosphere_database="unused_biosphere",
            specs=(_ef_spec(),),
        )


@pytest.mark.parametrize("native_match", (False, True), ids=("missing", "exact"))
def test_ef_factor_resolves_exact_native_identity_from_workbook(
    tmp_path: Path, native_match: bool
) -> None:
    """Preserve signed factors and reject missing native identities through real storage."""
    workbook = tmp_path / "ef.xlsx"
    spec = _ef_spec()
    _write_ef_workbook(
        workbook,
        indicators=[(*spec.impact_category_tuple, spec.unit)],
        factors=[
            ("ignored", "category", "indicator", "Ignored", "air", None, 9.0),
            (*spec.impact_category_tuple, "Carbon dioxide, fossil", "air", None, -2.5),
        ],
    )
    names = brightway_names(tmp_path, source_identity="source::" + "a" * 64)
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "carbon"): {
                    "name": "Carbon dioxide, fossil" if native_match else "Unrelated flow",
                    "categories": ("air",),
                    "unit": "kilogram",
                    "type": "emission",
                }
            },
            searchable=False,
        )
        flow = bd.get_node(database="biosphere", code="carbon")
        if native_match:
            resolved = owner.resolve_ef_methods(
                workbook, biosphere_database="biosphere", specs=(spec,)
            )
            assert resolved[0].factors == ((flow.id, -2.5),)
            expected = {
                "impact_category_tuple": list(spec.impact_category_tuple),
                "factors": [{"database": "biosphere", "code": "carbon", "amount": -2.5}],
            }
            assert (
                resolved[0].factor_identity
                == sha256(
                    (json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n").encode()
                ).hexdigest()
            )
        else:
            with pytest.raises(ValueError, match="EF factor does not resolve exactly.*matches=0"):
                owner.resolve_ef_methods(workbook, biosphere_database="biosphere", specs=(spec,))


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ({"selected.yaml": {"name": "not a row list"}}, "is not a factor table"),
        ({"unselected.yaml": []}, "archive member set changed"),
    ],
)
def test_pb_selected_factor_table_is_required(
    tmp_path: Path, members: dict[str, object], message: str
) -> None:
    """Reject missing or malformed selected factor data before biosphere resolution."""
    archive = tmp_path / "pb.tar.gz"
    _write_pb_archive(archive, members)
    spec = owner.MethodSpec(
        order=1,
        method_id="pb_lcia",
        method_version="0.9.2",
        impact_category_id="bounded",
        impact_category_tuple=("AESA (PBs-LCIA)", "0.9.2", "bounded"),
        unit="point",
        source_member="selected.yaml",
    )
    with pytest.raises(ValueError, match=message):
        owner.resolve_pb_methods(
            archive,
            biosphere_database="unused_biosphere",
            specs=(spec,),
        )


def test_biosphere_category_identity_omits_only_unspecified_subcompartments() -> None:
    """Keep real subcompartments while normalizing the source's unspecified marker."""
    assert owner._biosphere_categories("air", "urban air close to ground") == (
        "air",
        "urban air close to ground",
    )
    assert owner._biosphere_categories("air", "unspecified") == ("air",)
