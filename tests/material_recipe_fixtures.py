"""Small invented EcoSpold quantities for the recipe input boundary."""

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from pytoce.download_data.sources.ecoinvent_materials import (
    EcoinventMaterialRecipe,
    derive_material_recipe,
    material_recipe_members,
)

NS = "{http://www.EcoInvent.org/EcoSpold02}"
KG = "487df68b-4994-4027-8fdc-a4dc298257b7"
DIMENSIONLESS = "577e242a-461f-44a7-922c-d8e1c3d2bf45"
ALLOCATION = "9b14f020-ce8d-4929-a3b5-d6e44624e6bf"
CARBON = "a301a838-7975-4d89-9e74-8eb77ad03cd1"
DRY_MASS = "3a0af1d6-04c3-41c6-a3da-92c4f61e0eaa"
ETHYLENE = "44e8a892-3178-4352-a898-722e68990fe9"
PROPYLENE = "b658a402-79b9-4a02-a818-f6090adb13f1"
NAPHTHA = "27a3ac7b-e31a-48da-84f5-8dbdcb96e565"
PROCESSED = "21ca75fe-4ca2-52dd-b557-a820c0f7948c"
REJECT = "88afd721-07ec-4910-a004-fa3fecb6bf70"
PET_GRANULATE = "d585421f-a3a1-45bc-bd94-188f65865b9c"
PET_FLAKES = "0804a093-da37-52eb-ba9a-849ddad41900"
PET_PELLETS = "2b6efd7e-d14e-51ef-89c1-620108488d92"
HDPE_FLAKES = "e008ed90-a783-5cee-817b-ddb3ac6076f2"
HDPE_PELLETS = "1e0a3da9-3e39-5cc7-ac23-1f9bfcd4a54d"
LDPE_FLAKES = "1424d99c-387b-537d-8533-b29e34fdad12"
LDPE_PELLETS = "aa4f5f2c-5e5f-5e66-bdc0-cc6a10790be0"
COARSE_PM = "66020d27-7ae4-4e59-83a3-89214b72c40a"
MEDIUM_PM = "b967e1bf-f09b-4c89-8740-ace21db47bba"
UPR_OPERATION = "2e1ed2a3-ca0a-4275-b658-8773f486180d"
UPR_COMPOSITE = "fa6f1b55-8b52-4052-a2c8-0a0ba7a22d17"
UPR_RESIN = "e28d7a4d-fb38-4f5e-aac0-359ecef77b45"
UPR_INPUT = "4bf60736-bc87-4e3d-8d34-ea8bcc46f551"
GLASS_INPUT = "102d2161-b570-42cd-b646-1c44e47d23c5"
STYRENE_INPUT = "0df0db82-8e1d-4053-8cfc-5019470a86a7"
STYRENE_AIR = "d85a44cb-8407-4997-a146-581e4f35f39d"


@dataclass(frozen=True)
class _Source:
    """Invented quantities per reference kg for one named source fixture."""

    inputs: tuple[tuple[str, float], ...] = ()
    output: float = 1.0
    allocation: float | None = None
    finished: float | None = None
    volume: float = 1000.0
    carbon: float = 0.85
    dry_feed: tuple[tuple[str, float], ...] = ()
    air: tuple[tuple[str, float], ...] = ()


def _source_definitions() -> dict[str, _Source]:
    """Bind invented quantities by product identity, independent of member order."""
    definitions = {
        "b685192a-e033-421e-9074-8fe7cb176046": _Source(
            ((ETHYLENE, 1), (PROPYLENE, 0.02)), output=2
        ),
        "947a85fa-114e-4d9e-b36c-3293e48ea5ae": _Source(
            ((ETHYLENE, 0.54), (PROPYLENE, 0.01)), allocation=0.5
        ),
        "66ca2f38-5e51-4546-83c0-d7cef0c55c7c": _Source(((ETHYLENE, 0.1), (PROPYLENE, 0.95))),
        "3222f96e-3ac3-571d-95d4-06fe9caceff1": _Source(
            ((PROCESSED, 0.8), (REJECT, 0.1)), output=-1
        ),
        NAPHTHA: _Source(((PROCESSED, 1),), output=2, allocation=0.4),
        "2b1ad14c-f2b3-58a4-b1f4-9f8199c2504c": _Source(((PROCESSED, 1.5),), allocation=0.3),
        "f708f548-dd69-5319-bb88-9940bb2e7645": _Source(((PROCESSED, 2),), allocation=0.2),
        "5e29207d-289a-5be3-9006-9e3de396a5f2": _Source(((PROCESSED, 1),), allocation=0.1),
        ETHYLENE: _Source(((NAPHTHA, 1),), output=2, volume=100),
        PROPYLENE: _Source(((NAPHTHA, 2),), volume=200),
        "f87b61e1-7feb-4e00-a7c1-d8bc4b08d507": _Source(((NAPHTHA, 3),), volume=300),
        "8cf5755f-4c2c-4367-b1db-a4923a62fadc": _Source(((NAPHTHA, 4),), volume=400),
        "d899ab8e-58d3-47fa-bf64-45989f7da5be": _Source(((NAPHTHA, 5),), volume=500, carbon=0),
        "269c1b54-09c6-4bfb-9d38-c6ac39eac180": _Source(((NAPHTHA, 6),), volume=600),
        "abc2de9f-54b8-467d-936b-428447eb3db6": _Source(((NAPHTHA, 7),), volume=700),
        "68d6dfcf-2089-4586-9bbf-ad75591105cf": _Source(((REJECT, -0.1),), output=2, finished=0.8),
        "579049d0-3db8-4496-a234-2f61ed28a72b": _Source(((REJECT, -0.1),), finished=0.6),
        "406ae9de-8894-4c8b-a601-40b7e849b66d": _Source(((REJECT, -0.1),), finished=0.75),
        "cfe51aa8-55b2-43ef-82f3-0d6c4c80ca67": _Source(((REJECT, -0.1),), finished=0.9),
        "7539e2dc-9694-488e-8cf2-dcf80801892b": _Source(((REJECT, -0.1),), finished=0.5),
        "e6c4c508-f95f-42e0-b0a7-78c703326f9c": _Source(finished=1),
        "2fca690e-d8ae-4b9b-8c48-8f658b6e7268": _Source(
            ((PET_GRANULATE, 1.25), (REJECT, -0.25)), output=2
        ),
        UPR_OPERATION: _Source(output=2, finished=0.8, air=((STYRENE_AIR, 0.02),)),
        UPR_COMPOSITE: _Source(((UPR_INPUT, 0.5), (GLASS_INPUT, 0.75))),
        UPR_RESIN: _Source(((STYRENE_INPUT, 0.3),), output=2),
        HDPE_PELLETS: _Source(((HDPE_FLAKES, 1.2),), output=2, air=((COARSE_PM, 0.001),)),
        LDPE_PELLETS: _Source(((LDPE_FLAKES, 1.4),), output=3, air=((COARSE_PM, 0.003),)),
        PET_PELLETS: _Source(
            ((PET_FLAKES, 1.2),), output=2, dry_feed=((PET_FLAKES, 1),), air=((COARSE_PM, 0.002),)
        ),
    }
    recycling = (
        (HDPE_FLAKES, "a0471645-fb30-570e-9817-116f86bfe832", 2.0, 0.8, ((MEDIUM_PM, 0.0001),)),
        (
            LDPE_FLAKES,
            "a7715d96-f5a1-5240-811d-36f44221efec",
            2.5,
            0.6,
            (),
        ),
        (
            "212c3ba3-1119-5a35-9ac9-6db21e556b3a",
            "218e8d1e-0dd1-5041-89a0-7ce3a64610f9",
            1.5,
            0.8,
            ((COARSE_PM, 0.0003), (MEDIUM_PM, 0.0002)),
        ),
        (
            "fa6cade0-a03d-5fb9-a2fd-24f709fe0133",
            "bf17b7ce-e01a-59a2-bb12-6edb0acc34fd",
            2.0,
            0.9,
            ((MEDIUM_PM, 0.0001),),
        ),
        (PET_FLAKES, "d13ae29b-7d6a-55d2-8380-86343a80fed8", 1.8, 0.75, ((COARSE_PM, 0.0002),)),
        (
            "943bc561-2342-5457-8c2d-0cd3e51ad4b3",
            "6c8a2cce-34a5-5ae8-995e-bc3fe7fe4ca2",
            1.6,
            0.9,
            ((COARSE_PM, 0.0003),),
        ),
        (
            "90062186-804f-5fbb-afac-6ceec863d9e1",
            "147d7548-199e-5261-b50e-3072861b6ecb",
            2.0,
            0.8,
            ((COARSE_PM, 0.0004),),
        ),
    )
    for product, feed, amount, dry, air in recycling:
        definitions[product] = _Source(
            ((feed, amount),), output=2, dry_feed=((feed, dry),), air=air
        )
    return definitions


def material_source_members() -> dict[str, bytes]:
    """Construct valid named members with invented mass, allocation and moisture.

    Different production amounts exercise source normalization. Source normalization follows the
    specified activity yield.
    Formula attributes deliberately cannot be executed as numeric expressions.
    """
    definitions = _source_definitions()
    members = {}
    for member in material_recipe_members():
        activity, product = member.removeprefix("datasets/").removesuffix(".spold").split("_")
        source = definitions[product]
        root = ET.Element(f"{NS}ecoSpold")
        dataset = ET.SubElement(root, f"{NS}childActivityDataset")
        description = ET.SubElement(dataset, f"{NS}activityDescription")
        activity_node = ET.SubElement(description, f"{NS}activity", id=activity)
        if source.finished is not None:
            comment = ET.SubElement(activity_node, f"{NS}generalComment")
            if product == UPR_OPERATION:
                # The native UPR service describes the process before stating
                # its yield in another activity-comment text fragment.
                ET.SubElement(
                    comment, f"{NS}text", index="0"
                ).text = "This invented activity forms a composite product."
            ET.SubElement(
                comment, f"{NS}text", index="1"
            ).text = f"1 kg of this process equals {source.finished} kg of finished plastic."
        flows = ET.SubElement(dataset, f"{NS}flowData")
        reference = ET.SubElement(
            flows,
            f"{NS}intermediateExchange",
            intermediateExchangeId=product,
            id=f"reference-{product}",
            unitId=KG,
            amount=str(source.output),
            productionVolumeAmount=str(source.volume),
        )
        ET.SubElement(reference, f"{NS}unitName").text = "kg"
        ET.SubElement(reference, f"{NS}outputGroup").text = "0"
        ET.SubElement(
            reference,
            f"{NS}property",
            propertyId=CARBON,
            unitId=DIMENSIONLESS,
            amount=str(source.carbon),
        )
        if source.allocation is not None:
            ET.SubElement(
                reference,
                f"{NS}property",
                propertyId=ALLOCATION,
                unitId=DIMENSIONLESS,
                amount=str(source.allocation),
            )
        if source.finished is not None:
            comment = ET.SubElement(reference, f"{NS}comment")
            ET.SubElement(
                comment, f"{NS}text", index="1"
            ).text = "1 kg of this process equals 0.1 kg of a different product."
        for input_product, amount in source.inputs:
            exchange = ET.SubElement(
                flows,
                f"{NS}intermediateExchange",
                id=f"input-{input_product}",
                intermediateExchangeId=input_product,
                unitId=KG,
                amount=str(amount * source.output),
                mathematicalRelation="not evaluated: the numeric amount is authoritative",
            )
            ET.SubElement(exchange, f"{NS}unitName").text = "kg"
            ET.SubElement(exchange, f"{NS}inputGroup").text = "5"
            if input_product in dict(source.dry_feed):
                ET.SubElement(
                    exchange,
                    f"{NS}property",
                    propertyId=DRY_MASS,
                    unitId=KG,
                    amount=str(dict(source.dry_feed)[input_product]),
                )
        for flow, amount in source.air:
            exchange = ET.SubElement(
                flows,
                f"{NS}elementaryExchange",
                id=f"air-{flow}",
                elementaryExchangeId=flow,
                unitId=KG,
                amount=str(amount * source.output),
                mathematicalRelation="not executable",
            )
            ET.SubElement(exchange, f"{NS}unitName").text = "kg"
            ET.SubElement(exchange, f"{NS}outputGroup").text = "4"
        members[member] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return members


def physical_material_recipe() -> EcoinventMaterialRecipe:
    """Derive the complete material recipe from the existing invented source XML."""
    return derive_material_recipe(
        material_source_members(), source_identity="invented-material-source"
    )
