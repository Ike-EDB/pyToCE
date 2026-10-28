"""Allocation and source-admission invariants with invented EcoSpold data."""

import json
import shutil
from dataclasses import asdict
from hashlib import sha256
from importlib import resources
from math import fsum
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from jsonschema import Draft202012Validator
from material_recipe_fixtures import (
    ALLOCATION,
    COARSE_PM,
    DRY_MASS,
    ETHYLENE,
    HDPE_FLAKES,
    NAPHTHA,
    NS,
    PET_FLAKES,
    PET_PELLETS,
    PROCESSED,
    REJECT,
    material_source_members,
)

from pytoce.download_data.sources.catalogue import SourceSpec
from pytoce.download_data.sources.ecoinvent_materials import (
    PrincipalExit,
    derive_material_recipe,
    load_material_recipe,
    prepare_material_recipe,
)
from pytoce.shared.runtime.filesystem import payload_sha256


def test_material_reader_preserves_coefficients_and_canonical_identity(
    stored_material_recipe: tuple[Path, dict[str, object]],
) -> None:
    """Read the complete persisted recipe."""
    path, payload = stored_material_recipe
    before = path.read_bytes()
    recipe = load_material_recipe(path.parents[2])
    assert all(isinstance(row.principal_exits, tuple) for row in recipe.conversion)
    assert all(
        isinstance(exit_row, PrincipalExit)
        for row in recipe.conversion
        for exit_row in row.principal_exits
    )
    assert json.loads(json.dumps(recipe.as_dict())) == payload
    assert path.read_bytes() == before
    path.write_text(json.dumps(payload, indent=4, sort_keys=True), encoding="utf-8")
    assert load_material_recipe(path.parents[2]) == recipe


@pytest.mark.parametrize("incomplete", (False, True))
def test_material_reader_preserves_missing_or_partial_owner(
    tmp_path: Path, incomplete: bool
) -> None:
    """Report missing input files and preserve incomplete directories."""
    directory = tmp_path / "data_processed" / "ecoinvent"
    if incomplete:
        directory.mkdir(parents=True)
        (directory / "partial.json").write_bytes(b"unfinished")
    with pytest.raises(FileNotFoundError, match="material recipe") as error:
        load_material_recipe(tmp_path)
    assert "download_data" in str(error.value)
    if incomplete:
        assert "new workspace" in str(error.value)
        assert (directory / "partial.json").read_bytes() == b"unfinished"
    else:
        assert not directory.exists()


def test_physical_requirements_and_yields_have_distinct_allocation_bases() -> None:
    """Read yields across comment fragments with distinct allocation bases."""
    members = material_source_members()
    recipe = derive_material_recipe(members, source_identity="invented-source-v1")
    assert [(p.polymer, p.ethylene_kg_per_kg, p.propylene_kg_per_kg) for p in recipe.polymers] == [
        ("hdpe", 1.0, 0.02),
        ("ldpe", 1.08, 0.02),
        ("pp", 0.1, 0.95),
    ]
    assert recipe.processed_plastic_fraction == 0.8 / 0.9
    assert [p.kg_per_kg_feed for p in recipe.hydrothermal_products] == pytest.approx(
        [0.4, 0.2, 0.1, 0.1], rel=0, abs=1e-16
    )
    assert [p.kg_per_kg_feed for p in recipe.cracker_products] == [
        volume / 14000 for volume in (100, 200, 300, 400, 500, 600, 700)
    ]
    assert fsum(p.kg_per_kg_feed for p in recipe.cracker_products) == 0.2
    assert recipe.cracker_products[4].carbon_kg_per_kg == 0.0
    # Every invented recycling activity produces two kg. Native wet purchases
    # scale with that production; dry feed per reference kg also uses moisture.
    assert {
        item.operation: item.dry_feed_kg_per_reference_kg for item in recipe.recycling_operations
    } == pytest.approx(
        {
            "hdpe": 2.0 * 0.8,
            "ldpe": 2.5 * 0.6,
            "pp": 1.5 * 0.8,
            "pvc": 2.0 * 0.9,
            "pet": 1.8 * 0.75,
            "ps": 1.6 * 0.9,
            "weee": 2.0 * 0.8,
            "pet_finishing": 1.2,
        }
    )
    assert [
        (
            item.operation,
            item.principal_feed_kg_per_finished_kg,
            item.finished_kg_per_native_service_kg,
        )
        for item in recipe.conversion[:7]
    ] == [
        ("conversion_injection", 1 / 0.8, 0.8),
        ("conversion_film", 1 / 0.6, 0.6),
        ("conversion_pipe", 1 / 0.75, 0.75),
        ("conversion_blow", 1 / 0.9, 0.9),
        ("conversion_calendering", 1 / 0.5, 0.5),
        ("conversion_foaming", 1, 1),
        ("fibres_pet_conversion", 1.25, 1),
    ]
    conversions = {item.operation: item for item in recipe.conversion}
    expected_gross = {
        "fibres_pp_conversion": 1.0,
        "fibres_pa6_conversion": 1.0001035 * 100000 / 99999,
        "fibres_pa66_conversion": 100000 / 99999,
        "conversion_tyre": 1.0283,
        "pur_conversion_included_in_finished_products": 1.0,
        "others_esbr_compounding_conversion": 1.0526638382532205,
        "others_epoxy_coating_conversion": 1.6210254438225341,
        "others_road_marking_conversion": 892 / 875,
        # Independent rational witness for this invented .3-styrene/.8-yield source.
        "others_upr_hand_lay_up_conversion": 1.2893924329275226,
    }
    assert len(conversions) == 16
    for name, gross in expected_gross.items():
        assert conversions[name].principal_feed_kg_per_finished_kg == pytest.approx(
            gross, rel=0, abs=5e-16
        )
    assert conversions["others_upr_hand_lay_up_conversion"].finished_kg_per_native_service_kg == (
        pytest.approx(0.3070506950008652, rel=0, abs=1e-16)
    )
    assert {s.member: s.sha256 for s in recipe.datasets} == {
        member: sha256(content).hexdigest() for member, content in members.items()
    }
    payload = recipe.as_dict()
    schema = json.loads(
        resources.files("pytoce.download_data.data")
        .joinpath("ecoinvent_material_recipe_schema_v1.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(json.dumps(payload)))
    identity = payload.pop("identity")
    assert identity == f"{payload['schema_version']}::{payload_sha256(payload)}"
    assert asdict(
        derive_material_recipe(
            dict(reversed(list(members.items()))), source_identity="invented-source-v1"
        )
    ) == asdict(recipe)


def test_manufacturing_exits_preserve_selective_releases_and_sequence() -> None:
    """Principal exits close all sixteen operations and preserve selective losses."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented-source-v1")
    conversions = {row.operation: row for row in recipe.conversion}
    assert len(conversions) == 16
    for row in conversions.values():
        assert fsum(
            exit_row.kg_per_finished_kg for exit_row in row.principal_exits
        ) == pytest.approx(row.principal_feed_kg_per_finished_kg - 1, abs=5e-16, rel=0)
    exits = {
        row.operation: {
            exit_row.name: exit_row.kg_per_finished_kg for exit_row in row.principal_exits
        }
        for row in conversions.values()
    }
    assert exits["fibres_pet_conversion"] == {"principal_remainder": 0.25}
    assert exits["fibres_pp_conversion"] == {}
    assert exits["pur_conversion_included_in_finished_products"] == {}
    # Of 100,000 kg raw PA, 10 kg adheres to equipment, 9 kg returns internally, and 1 kg is
    # destroyed. The remaining 99,999 kg comprises finished PA and, for PA6, caprolactam released to
    # air or collected with water.
    assert exits["fibres_pa6_conversion"] == pytest.approx(
        {
            "caprolactam_air": 0.0000325,
            "caprolactam_collected_water": 0.000071,
            "spinning_equipment_deposits": 1.0001035 / 99999,
        },
        rel=2e-15,
    )
    assert exits["fibres_pa66_conversion"] == pytest.approx(
        {"spinning_equipment_deposits": 1 / 99999}, rel=2e-15
    )

    sbr_gross = conversions["others_esbr_compounding_conversion"].principal_feed_kg_per_finished_kg
    assert exits["others_esbr_compounding_conversion"] == pytest.approx(
        {
            "styrene_air": 3.657420773044308e-6 * sbr_gross,
            "butadiene_air": 3.879806170028767e-7 * sbr_gross,
            "extrusion_dust": 0.0000266 * sbr_gross,
            # Five percent of the pre-trim product, with finished product fixed at one kg.
            "post_release_solid": 1 / 19,
        },
        abs=1e-16,
        rel=0,
    )
    assert exits["others_road_marking_conversion"] == pytest.approx(
        {
            "container_residue": 0.01 * 892 / 875,
            "methyl_methacrylate_air": (202 / 22300) * 892 / 875,
        },
        abs=1e-16,
        rel=0,
    )
    epoxy_gross = conversions["others_epoxy_coating_conversion"].principal_feed_kg_per_finished_kg
    assert exits["others_epoxy_coating_conversion"] == pytest.approx(
        {
            "preparation_residue": epoxy_gross - 1 / 0.675,
            "settled_overspray": 0.95 * (1 / 0.675 - 1),
            "airborne_overspray": 0.05 * (1 / 0.675 - 1),
        },
        abs=3e-16,
        rel=0,
    )

    # Invented XML: glass/resin=1.5, resin styrene=.3 kg/kg,
    # total styrene=.02 kg/kg operation, finished composite=.8 kg/kg operation.
    principal = 0.99675
    charge = 2.517625
    emitted_styrene = 0.02 * charge
    principal_styrene = emitted_styrene * (0.3 * principal) / (0.3 * principal + 0.001)
    fraction_at_trim = (0.8 * charge) / (charge - emitted_styrene - 0.00106)
    finished = fraction_at_trim * (principal - principal_styrene)
    assert exits["others_upr_hand_lay_up_conversion"] == pytest.approx(
        {
            "styrene_air": principal_styrene / finished,
            "post_release_solid": (principal - principal_styrene - finished) / finished,
        },
        abs=2e-16,
        rel=0,
    )


def test_recovery_separates_source_yield_moisture_dust_and_managed_fates() -> None:
    """Close all eleven routes with airborne dust and managed loss partitioned separately."""
    recipe = derive_material_recipe(material_source_members(), source_identity="invented-source-v1")
    # Independent full-route witnesses combine source declarations and invented
    # native PM/feed references. Non-unit productions cancel in these intensities.
    recovered = {
        "hdpe_granulate": 0.875427580824687,
        "ldpe_granulate": 0.86667330501644,
        "pp_granulate": 0.8400346553597812,
        "pet_granulate": (19205.93 / 21930) * 0.99 * 0.99 / 1.2,
        "ps_granulate": 0.5942463695581395,
        "engineering_granulate": 0.9302688,
        "pa_granulate": 0.98,
        "pvc_granulate": 0.81,
        "tps_compound": 20 / 23,
        "phb_granulate": 1 / 1.2,
        "biopolyester_granulate": 0.528165,
    }
    aquatic = {
        "hdpe_granulate": 0.009302233154442457,
        "ldpe_granulate": 0.009209210822898032,
        "pp_granulate": 0.008835987232102143,
        "pet_granulate": 0.008757834017327861,
        "ps_granulate": 0.007455581395348837,
        "biopolyester_granulate": 0.0055,
    }
    air = {
        "hdpe_granulate": 0.0001 / 1.6 + recovered["hdpe_granulate"] * 0.001,
        "ldpe_granulate": recovered["ldpe_granulate"] * 0.003,
        "pp_granulate": 0.0005 / 1.2,
        "pet_granulate": 0.0002 / 1.35 + recovered["pet_granulate"] * 0.002,
        "ps_granulate": 0.0003 / 1.44,
        "engineering_granulate": 0.0004 / 1.6 + recovered["engineering_granulate"] * 0.002,
        "pa_granulate": 0.0004 / 1.6 + 0.98 * 0.002,
        "pvc_granulate": 0.0001 / 1.8,
        "tps_compound": 0.0,
        "phb_granulate": recovered["phb_granulate"] * 0.002,
        "biopolyester_granulate": 0.0004 / 1.6 + 0.528165 * 0.002,
    }
    assert {row.name for row in recipe.mechanical_recovery} == set(recovered)
    for row in recipe.mechanical_recovery:
        name = row.name
        assert row.recovered_kg_per_input_kg == pytest.approx(recovered[name], abs=3e-16, rel=0)
        assert row.waterborne_kg_per_input_kg == pytest.approx(
            aquatic.get(name, 0), abs=1e-17, rel=0
        )
        assert row.airborne_kg_per_input_kg == pytest.approx(air[name], abs=1e-18, rel=0)
        assert row.managed_treatment == ("landfill" if name == "tps_compound" else "incineration")


def test_pellet_feed_changes_only_pet_and_phb_physical_yields() -> None:
    """Keep allocated inventory scaling separate from material-recovery yields."""
    members = material_source_members()
    before = derive_material_recipe(members, source_identity="invented-source-v1")
    member = next(name for name in members if PET_PELLETS in name)
    root = ET.fromstring(members[member])
    feed = root.find(f".//{NS}intermediateExchange[@intermediateExchangeId='{PET_FLAKES}']")
    assert feed is not None
    feed.set("amount", "3.0")  # 1.5 kg feed per kg output; reference production is 2.
    members[member] = ET.tostring(root)
    after = derive_material_recipe(members, source_identity="invented-source-v1")
    assert after.identity != before.identity
    for old, new in zip(before.mechanical_recovery, after.mechanical_recovery, strict=True):
        if old.name in {"pet_granulate", "phb_granulate"}:
            assert new.recovered_kg_per_input_kg == pytest.approx(
                old.recovered_kg_per_input_kg * 0.8, abs=2e-16, rel=0
            )
        else:
            assert new == old


@pytest.mark.parametrize(
    "defect",
    (
        "dry_zero",
        "dry_unit",
        "preparation_dry_zero",
        "dust_negative",
        "dust_excess",
        "dust_missing",
        "dust_unit",
    ),
)
def test_invalid_recycling_source_cannot_publish_a_physical_partition(defect: str) -> None:
    """Dry-feed admission and actual loss capacity are checked at their source owner."""
    members = material_source_members()
    product = HDPE_FLAKES if defect == "preparation_dry_zero" else PET_PELLETS
    member = next(name for name in members if product in name)
    root = ET.fromstring(members[member])
    if "dry" in defect:
        element = root.find(f".//{NS}property[@propertyId='{DRY_MASS}']")
        assert element is not None
        element.set("amount" if defect.endswith("zero") else "unitId", "0")
    else:
        element = root.find(f".//{NS}elementaryExchange[@elementaryExchangeId='{COARSE_PM}']")
        assert element is not None
        if defect == "dust_missing":
            flow_data = root.find(f".//{NS}flowData")
            assert flow_data is not None
            flow_data.remove(element)
        elif defect == "dust_unit":
            element.set("unitId", "cubic-meter-is-not-kilogram")
        else:
            element.set("amount", "-1" if defect == "dust_negative" else "2")
    members[member] = ET.tostring(root)
    with pytest.raises(ValueError, match="Nonpositive|Expected one|Negative|exceed"):
        derive_material_recipe(members, source_identity="invented-source-v1")


@pytest.mark.parametrize(
    ("defect", "message"),
    (
        ("flow_data", "Missing EcoSpold flowData"),
        ("reference", "Expected one EcoSpold outputGroup"),
        ("reference_unit", "must use kg"),
        ("allocation_zero", "Invalid EcoSpold property"),
        ("allocation_above_one", "Invalid EcoSpold property"),
    ),
)
def test_source_structure_and_allocation_are_admitted_once(defect: str, message: str) -> None:
    """Reject invalid external XML before deriving any allocation ratio."""
    members = material_source_members()
    member = next(name for name in members if "371c4bec-" in name)
    root = ET.fromstring(members[member])
    dataset = root.find(f"{NS}childActivityDataset")
    assert dataset is not None
    flow_data = dataset.find(f"{NS}flowData")
    assert flow_data is not None
    reference = next(
        row
        for row in flow_data.findall(f"{NS}intermediateExchange")
        if row.findtext(f"{NS}outputGroup") == "0"
    )
    if defect == "flow_data":
        dataset.remove(flow_data)
    elif defect == "reference":
        flow_data.remove(reference)
    elif defect == "reference_unit":
        reference.set("unitId", "cubic-meter-is-not-kilogram")
    else:
        allocation = reference.find(f"{NS}property[@propertyId='{ALLOCATION}']")
        assert allocation is not None
        # Allocation is a strictly positive fraction.
        allocation.set("amount", "0" if defect == "allocation_zero" else "1.1")
    members[member] = ET.tostring(root)
    with pytest.raises(ValueError, match=message):
        derive_material_recipe(members, source_identity="invented-source-v1")


def test_changing_allocation_changes_physical_demand_and_identity() -> None:
    """Allocated purchases alone cannot determine the LDPE monomer demand."""
    members = material_source_members()
    before = derive_material_recipe(members, source_identity="invented-source-v1")
    member = next(name for name in members if "371c4bec-" in name)
    root = ET.fromstring(members[member])
    allocation = root.find(f".//{NS}property[@propertyId='{ALLOCATION}']")
    assert allocation is not None
    allocation.set("amount", "0.25")
    members[member] = ET.tostring(root)
    after = derive_material_recipe(members, source_identity="invented-source-v1")
    assert after.polymers[1].ethylene_kg_per_kg == 2.16
    assert after.polymers[1].propylene_kg_per_kg == 0.04
    assert after.identity != before.identity
    assert after.polymers[0] == before.polymers[0]
    assert after.polymers[2] == before.polymers[2]


@pytest.mark.parametrize("amount", ["nan", "inf", "-0.1", "0"])
def test_invalid_monomer_input_cannot_become_a_physical_recipe(amount: str) -> None:
    """Admit finite positive source purchases before their normalization."""
    members = material_source_members()
    member = next(name for name in members if "a0b64b03-" in name)
    root = ET.fromstring(members[member])
    exchange = root.find(f".//{NS}intermediateExchange[@intermediateExchangeId='{ETHYLENE}']")
    assert exchange is not None
    exchange.set("amount", amount)
    members[member] = ET.tostring(root)
    with pytest.raises(ValueError, match="Nonfinite|Nonpositive"):
        derive_material_recipe(members, source_identity="invented-source-v1")


@pytest.mark.parametrize(
    ("process", "defect"),
    (
        ("preparation", "excess_output"),
        ("hydrothermal", "excess_output"),
        ("cracker", "excess_output"),
        ("hydrothermal", "zero_feed"),
        ("cracker", "zero_feed"),
    ),
)
def test_named_physical_products_require_positive_sufficient_feed(
    process: str, defect: str
) -> None:
    """Reject invalid source feed signs and yields before publishing a recipe."""
    members = material_source_members()
    for member, content in members.items():
        root = ET.fromstring(content)
        for exchange in root.findall(f".//{NS}intermediateExchange"):
            product = exchange.get("intermediateExchangeId")
            if process == "preparation" and product == REJECT:
                # 0.8 kg processed plus 0.3 kg reject exceeds the 1 kg bale.
                exchange.set("amount", "-0.3")
            elif process == "hydrothermal" and product == PROCESSED:
                if (
                    exchange.findtext(f"{NS}inputGroup") == "5"
                    and float(exchange.get("amount")) > 0
                ):
                    # Halving all allocated feed doubles named yields from 0.8 to 1.6 kg/kg.
                    exchange.set(
                        "amount",
                        "0" if defect == "zero_feed" else str(float(exchange.get("amount")) / 2),
                    )
            elif process == "cracker" and product == NAPHTHA:
                if exchange.findtext(f"{NS}inputGroup") == "5":
                    # One tenth of the feed gives 2 kg named products per kg feed.
                    exchange.set(
                        "amount",
                        "0" if defect == "zero_feed" else str(float(exchange.get("amount")) / 10),
                    )
        members[member] = ET.tostring(root)
    with pytest.raises(ValueError, match="tracked-plastic balance|exceed their feed|Nonpositive"):
        derive_material_recipe(members, source_identity="invented-source-v1")


@pytest.mark.parametrize("state", ("complete", "incomplete"))
def test_material_recipe_preserves_completed_and_incomplete_inputs(
    tmp_path: Path,
    acquisition_catalogue: tuple[tuple[SourceSpec, ...], tuple[SourceSpec, ...], tuple[Path, ...]],
    state: str,
) -> None:
    """Reuse completed sources and preserve incomplete owned directories."""
    _, licensed, archives = acquisition_catalogue
    workspace = tmp_path / "workspace"
    raw = workspace / "data_raw" / "ecoinvent"
    raw.mkdir(parents=True)
    archive = raw / licensed[0].filename
    shutil.copyfile(archives[0], archive)
    output = workspace / "data_processed" / "ecoinvent"
    if state == "complete":
        assert prepare_material_recipe(workspace, licensed[0])
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        archive.unlink()
        assert not prepare_material_recipe(workspace, licensed[0])
        assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    else:
        output.mkdir(parents=True)
        (output / "user.txt").write_bytes(b"unclassified input")
        before = archive.read_bytes()
        with pytest.raises(FileExistsError, match="has no completion file") as error:
            prepare_material_recipe(workspace, licensed[0])
        assert str(output) in str(error.value)
        assert {path.name: path.read_bytes() for path in output.iterdir()} == {
            "user.txt": b"unclassified input"
        }
        assert archive.read_bytes() == before
