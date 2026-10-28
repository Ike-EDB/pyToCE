"""Exact EF/PB source resolution and registration tests."""

import json
import tarfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
import yaml
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from pytoce.prepare_backgrounds.preparation import project_biosphere as owner
from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    brightway_names,
    selected_brightway_project,
)


def test_catalogue_keeps_factor_data_workspace_owned_and_pb_exclusions() -> None:
    specs, raw = owner.load_lcia_catalogue()
    assert len(specs) == 26
    assert sum(spec.method_id == "ef31" for spec in specs) == 16
    assert sum(spec.method_id == "pb_lcia" for spec in specs) == 10
    assert [spec.order for spec in specs] == list(range(1, 27))
    assert all(
        "no LT" not in component for spec in specs for component in spec.impact_category_tuple
    )
    assert {
        key for row in cast(list[dict[str, object]], raw["impact_categories"]) for key in row
    }.isdisjoint({"factor_count", "factor_signature", "flow_code"})
    assert "pb_flow_crosswalk" not in raw
    assert raw["excluded_pb_source_members"] == [
        "aesa_BiogeochemicalFlows_N_directlyFixated.yaml",
        "aesa_ChangeBiosphereIntegrity_FunctionalDiversity_Hierarchist_land_use.yaml",
        ("aesa_ChangeBiosphereIntegrity_FunctionalDiversity_Hierarchist_CO2eq_emissions.yaml"),
    ]


def test_registration_preserves_negative_zero_and_positive_factors(
    tmp_path: Path,
) -> None:
    names = brightway_names(tmp_path, source_identity="source::" + "a" * 64)
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "negative"): {
                    "name": "Negative",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("air",),
                },
                ("biosphere", "zero"): {
                    "name": "Zero",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("air",),
                },
                ("biosphere", "positive"): {
                    "name": "Positive",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("air",),
                },
            }
        )
        flows = [
            bd.get_node(database="biosphere", code=code)
            for code in ("negative", "zero", "positive")
        ]
        spec = owner.MethodSpec(
            order=1,
            method_id="test",
            method_version="1",
            impact_category_id="signs",
            impact_category_tuple=("test", "signs"),
            unit="point",
            source_member="test",
        )
        resolved = owner._resolved_method(
            spec,
            [(flows[0], -1.0), (flows[1], 0.0), (flows[2], 2.0)],
        )
        rows = owner.register_methods((resolved,), source_identity="source::test")
        assert rows[0]["factor_identity"] == resolved.factor_identity
        assert dict(bd.Method(("test", "signs")).load()) == {
            flows[0].id: -1.0,
            flows[1].id: 0.0,
            flows[2].id: 2.0,
        }
        assert "A_technosphere_flows" not in bd.databases


@pytest.mark.parametrize("ambiguous", (False, True), ids=("unique", "ambiguous"))
def test_pb_casefold_resolution_requires_one_native_identity(
    tmp_path: Path, ambiguous: bool
) -> None:
    """Parse synthetic PB data and require unique case-insensitive flow matching."""
    archive = tmp_path / "pb.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for filename, rows in (
            (
                "selected.yaml",
                [{"name": "carbon dioxide, fossil", "categories": "air::urban", "amount": -2.5}],
            ),
            ("excluded.yaml", {"invalid": "unselected table must not be parsed"}),
            ("README.txt", "not a factor table"),
        ):
            content = yaml.safe_dump(rows).encode("utf-8")
            member = tarfile.TarInfo(filename)
            member.size = len(content)
            handle.addfile(member, BytesIO(content))
    names = brightway_names(tmp_path, source_identity="source::" + "a" * 64)
    with selected_brightway_project(names) as bd:
        flows = {
            ("biosphere", "carbon"): {
                "name": "Carbon Dioxide, Fossil",
                "unit": "kilogram",
                "type": "emission",
                "categories": ("Air", "Urban"),
            }
        }
        if ambiguous:
            flows[("biosphere", "other")] = {
                "name": "CARBON DIOXIDE, FOSSIL",
                "unit": "kilogram",
                "type": "emission",
                "categories": ("AIR", "URBAN"),
            }
        bd.Database("biosphere").write(flows, searchable=False)
        flow = bd.get_node(database="biosphere", code="carbon")
        spec = owner.MethodSpec(
            order=1,
            method_id="pb_lcia",
            method_version="0.9.2",
            impact_category_id="casefold",
            impact_category_tuple=("AESA (PBs-LCIA)", "0.9.2", "casefold"),
            unit="point",
            source_member="selected.yaml",
        )
        if ambiguous:
            with pytest.raises(ValueError, match="casefold-unique.*casefold_matches=2"):
                owner.resolve_pb_methods(archive, biosphere_database="biosphere", specs=(spec,))
        else:
            resolved = owner.resolve_pb_methods(
                archive, biosphere_database="biosphere", specs=(spec,)
            )
            assert len(resolved) == 1
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


def test_ef_factor_rejects_boolean_amount(tmp_path: Path) -> None:
    """Reject a nonnumeric EF amount at the workbook boundary."""

    workbook = tmp_path / "ef.xlsx"
    book = Workbook()
    indicators = cast(Worksheet, book.active)
    indicators.title = "Indicators"
    indicators.append(("method", "category", "indicator", "unit"))
    indicators.append(("EF v3.1", "climate", "GWP100", "kg CO2-Eq"))
    factors = book.create_sheet("CFs")
    factors.append(
        ("method", "category", "indicator", "name", "compartment", "subcompartment", "amount")
    )
    factors.append(("EF v3.1", "climate", "GWP100", "Carbon dioxide", "air", None, True))
    book.save(workbook)
    book.close()
    spec = owner.MethodSpec(
        order=1,
        method_id="ef31",
        method_version="3.1",
        impact_category_id="climate",
        impact_category_tuple=("EF v3.1", "climate", "GWP100"),
        unit="kg CO2-Eq",
        source_member="implementation.xlsx",
    )

    names = brightway_names(tmp_path, source_identity="source::" + "a" * 64)
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "carbon"): {
                    "name": "Carbon dioxide",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("air",),
                }
            }
        )
        with pytest.raises(ValueError, match="EF factor True.*must be a finite number"):
            owner.resolve_ef_methods(
                workbook,
                biosphere_database="biosphere",
                specs=(spec,),
            )


def test_pb_factor_rejects_nonfinite_amount(tmp_path: Path) -> None:
    """Reject a nonfinite PB-LCIA amount at the YAML boundary."""

    archive = tmp_path / "pb.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for filename, rows in (
            (
                "selected.yaml",
                [{"name": "Carbon dioxide", "categories": "air", "amount": float("nan")}],
            ),
        ):
            content = yaml.safe_dump(rows).encode("utf-8")
            member = tarfile.TarInfo(filename)
            member.size = len(content)
            handle.addfile(member, BytesIO(content))
    spec = owner.MethodSpec(
        order=1,
        method_id="pb_lcia",
        method_version="0.9.2",
        impact_category_id="climate",
        impact_category_tuple=("AESA (PBs-LCIA)", "0.9.2", "climate"),
        unit="point",
        source_member="selected.yaml",
    )

    names = brightway_names(tmp_path, source_identity="source::" + "a" * 64)
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(
            {
                ("biosphere", "carbon"): {
                    "name": "Carbon dioxide",
                    "unit": "kilogram",
                    "type": "emission",
                    "categories": ("air",),
                }
            }
        )
        with pytest.raises(ValueError, match="PB-LCIA factor nan.*must be a finite number"):
            owner.resolve_pb_methods(
                archive,
                biosphere_database="biosphere",
                specs=(spec,),
            )
