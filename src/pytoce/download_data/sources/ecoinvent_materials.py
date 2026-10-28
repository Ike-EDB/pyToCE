"""Derive pyToCE plastics coefficients from licensed sources.

Monomer requirements differ from allocated LCI purchases. This owner
undoes the admitted granulate allocation, reconstructs hydrothermal product
yields, and combines the seven cracker coproduct datasets on their common feed
basis. Manufacturing coefficients distinguish gross resin from finished output.
Mechanical recovery distinguishes dry principal polymer, waterborne polymer,
dust and managed residue. ``material_constituents.json`` owns the constituent
and forming-portfolio bindings. The fixed process references apply across
12 world regions, years and carbon origins.

The owner reads evaluated ecoinvent 3.12 cut-off XML quantities and documented
process descriptions. Public process studies
supply the identified transfers below. Complete utilities,
auxiliaries and terminal inventories belong to the corresponding LCI owners.
The LCA combines these inventories to calculate life cycle impacts.
The derived numerical product belongs in the user's licensed workspace.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast
from xml.etree import ElementTree

import py7zr
from py7zr.io import BytesIOFactory

from pytoce.shared.runtime.filesystem import (
    atomic_write,
    canonical_json_bytes,
    payload_sha256,
    replace_directory,
)
from pytoce.shared.runtime.workspace_guides import write_workspace_guide

from .catalogue import SourceSpec, source_directory

MATERIAL_RECIPE_SCHEMA = "pytoce_ecoinvent_material_recipe_schema_v1"
MATERIAL_RECIPE_METHOD = "pytoce_ecoinvent_physical_plastics_v5"
# Lase et al. (2022), DOI 10.1016/j.wasman.2022.09.002, SI Table S3.2: kg PE/PP retained per kg
# extrusion feed. This retention also normalizes PE finishing.
POLYOLEFIN_PELLET_RETENTION = 0.97
_NS = "{http://www.EcoInvent.org/EcoSpold02}"
_KG = "487df68b-4994-4027-8fdc-a4dc298257b7"
_DIMENSIONLESS = "577e242a-461f-44a7-922c-d8e1c3d2bf45"
# EcoSpold properties: product allocation fraction, nominal kg C/kg product,
# and kg dry matter per kg wet feed.
_ALLOCATION = "9b14f020-ce8d-4929-a3b5-d6e44624e6bf"
_CARBON_CONTENT = "a301a838-7975-4d89-9e74-8eb77ad03cd1"
_DRY_MASS = "3a0af1d6-04c3-41c6-a3da-92c4f61e0eaa"
# Native kg particulate flows: >10 micrometers and >2.5 to <10 micrometers. These flows record the
# principal-polymer dust transfer.
_COARSE_PM = "66020d27-7ae4-4e59-83a3-89214b72c40a"
_MEDIUM_PM = "b967e1bf-f09b-4c89-8740-ace21db47bba"
_ETHYLENE = "44e8a892-3178-4352-a898-722e68990fe9"
_PROPYLENE = "b658a402-79b9-4a02-a818-f6090adb13f1"
_NAPHTHA = "27a3ac7b-e31a-48da-84f5-8dbdcb96e565"
_PROCESSED_PLASTIC = "21ca75fe-4ca2-52dd-b557-a820c0f7948c"
_PLASTIC_REJECT = "88afd721-07ec-4910-a004-fa3fecb6bf70"

# Exact activity/product UUID pairs select three RER granulate-production records: polyethylene high
# density, polyethylene low density, polypropylene. They supply ethylene/propylene feed for mapped
# PE/PP, including bio-PE and PP fiber. Only LDPE requires undoing source allocation; unity is the
# admitted RER HDPE/PP case.
_POLYMERS = (
    ("hdpe", "a0b64b03-73fd-5f42-a8fa-4f9b05817bfd", "b685192a-e033-421e-9074-8fe7cb176046"),
    ("ldpe", "371c4bec-0597-50d3-8148-ac40ab1416a2", "947a85fa-114e-4d9e-b36c-3293e48ea5ae"),
    ("pp", "94452acd-289d-5485-85b0-8a596afdd9aa", "66ca2f38-5e51-4546-83c0-d7cef0c55c7c"),
)
# GB waste preparation and its four allocated hydrothermal-treatment products supply the PE/PP
# chemical route. Sorting rejects and nonplastic bale content have separate bases; naphtha/DGO
# supply cracker feed.
_PREPARATION = (
    "4811f70c-d14d-55f1-b11d-1dedcea7252f",
    "3222f96e-3ac3-571d-95d4-06fe9caceff1",
)
_HTT_ACTIVITY = "04104c93-6445-5f6c-b81c-f9f915bb786c"
_HTT_PRODUCTS = (
    ("naphtha", _NAPHTHA),
    ("distillate_gas_oil", "2b1ad14c-f2b3-58a4-b1f4-9f8199c2504c"),
    ("heavy_gas_oil", "f708f548-dd69-5319-bb88-9940bb2e7645"),
    ("heavy_wax_residue", "5e29207d-289a-5be3-9006-9e3de396a5f2"),
)
# The KR steam cracker has a naphtha feed, unlike the RER mixed-feed reference.
# HTT naphtha plus DGO supplies naphtha-equivalent feed following
# Ozoemena/Coles (2023), DOI 10.1007/s10924-023-02792-3. All seven siblings are needed
# to reconstruct outputs despite their different operating allocations.
_CRACKER_ACTIVITY = "2319a237-ef6b-57b7-9300-871699ba1d8f"
_CRACKER_PRODUCTS = (
    ("ethylene", _ETHYLENE),
    ("propylene", _PROPYLENE),
    ("c4", "f87b61e1-7feb-4e00-a7c1-d8bc4b08d507"),
    ("pyrolysis_gas_btx", "8cf5755f-4c2c-4367-b1db-a4923a62fadc"),
    ("hydrogen", "d899ab8e-58d3-47fa-bf64-45989f7da5be"),
    ("external_fuel_gas", "269c1b54-09c6-4bfb-9d38-c6ac39eac180"),
    ("external_fuel_oil", "abc2de9f-54b8-467d-936b-428447eb3db6"),
)
# Six RER kg processing services: injection molding, plastic-film extrusion, pipe extrusion, blow
# molding, rigid-sheet calendering, polymer foaming. Their activity comments state finished kg per
# gross process kg. Material portfolios select these functions. PVC's calendered-sheet share uses
# the sheet operation, which excludes the separately supplied principal resin.
_CONVERSIONS = (
    (
        "conversion_injection",
        "4a4d5f11-100e-5803-83e3-809b2120fc2b",
        "68d6dfcf-2089-4586-9bbf-ad75591105cf",
    ),
    (
        "conversion_film",
        "4cec63a7-53cf-5f4b-80e3-979def718e98",
        "579049d0-3db8-4496-a234-2f61ed28a72b",
    ),
    (
        "conversion_pipe",
        "7e5a8c97-439f-5fca-921a-860ad94d4593",
        "406ae9de-8894-4c8b-a601-40b7e849b66d",
    ),
    (
        "conversion_blow",
        "1c8f5a85-838c-5775-81ee-a634c0bf8ab3",
        "cfe51aa8-55b2-43ef-82f3-0d6c4c80ca67",
    ),
    (
        "conversion_calendering",
        "6158b49d-04c8-5fc0-a46a-bd0bf436c836",
        "7539e2dc-9694-488e-8cf2-dcf80801892b",
    ),
    (
        "conversion_foaming",
        "1beca84e-83b9-5be3-a15e-3c4d0a93ab1f",
        "e6c4c508-f95f-42e0-b0a7-78c703326f9c",
    ),
)
# RoW "polyester fibre production, finished": evaluated amorphous-PET input
# per finished fiber kg supplies spinning retention; its full LCI is separate.
_PET_FIBRE = (
    "bf221663-baa6-57d5-a675-0446f29bcc46",
    "2fca690e-d8ae-4b9b-8c48-8f658b6e7268",
)
_PET_GRANULATE = "d585421f-a3a1-45bc-bd94-188f65865b9c"
# Ecoinvent 3.12 "hand lay-up" (activity and reference product), kg service, FR:
# supplies styrene release and finished-service fraction from the 2019--2022
# French-factory reference. Its processing function applies across the 12 world regions.
_UPR_OPERATION = (
    "3575e9d6-42c6-517b-a0da-fbbe9a843a46",
    "2e1ed2a3-ca0a-4275-b658-8773f486180d",
)
# RER "glass fibre reinforced plastic production, polyester resin, hand lay-up",
# reference "glass fibre reinforced plastic, polyester resin, hand lay-up", kg:
# supplies the glass/resin ratio from its one-manufacturer reference.
_UPR_COMPOSITE = (
    "4dd9564f-2b87-55d5-b5ba-8a2fd935cd86",
    "fa6f1b55-8b52-4052-a2c8-0a0ba7a22d17",
)
# RER "orthophthalic acid based unsaturated polyester resin production",
# reference "orthophthalic acid based unsaturated polyester resin", kg:
# supplies the delivered-resin free-styrene reference from five European
# producers (Ernst & Young/cefic UPR and VE resins sector group; inventory
# completed in 2024).
_UPR_RESIN = (
    "393faabd-fc76-5ef5-b47e-789f27e9f919",
    "e28d7a4d-fb38-4f5e-aac0-359ecef77b45",
)
_UPR_INPUT = "4bf60736-bc87-4e3d-8d34-ea8bcc46f551"
_GLASS_INPUT = "102d2161-b570-42cd-b646-1c44e47d23c5"
_STYRENE_INPUT = "0df0db82-8e1d-4053-8cfc-5019470a86a7"
_STYRENE_AIR = "d85a44cb-8407-4997-a146-581e4f35f39d"

# RER tuples give name, activity UUID, output UUID, wet-feed UUID and selected principal-dust flows.
# HDPE/LDPE/PET end in flakes; PP/PS/rigid PVC include pellet extrusion. Empty LDPE preparation PM
# means no selected native row, the source recipe reports these named releases. WEEE supplies ABS
# flakes; its preparation function serves engineering resins, PA and four biopolyesters, preserving
# their actual principal material. The recovery LCI compiler owns the water-medium and
# no-new-masterbatch construction of that transferred line. Allocated wet-feed/dry-mass denominators
# normalize burdens. Recovery follows the material balance.
_RECYCLING = (
    (
        "hdpe",
        "608b3cff-5fd8-5dbe-b259-4dbb352403d8",
        "e008ed90-a783-5cee-817b-ddb3ac6076f2",
        "a0471645-fb30-570e-9817-116f86bfe832",
        (_MEDIUM_PM,),
    ),
    (
        "ldpe",
        "52a8c705-3dfd-593b-9c3d-3a116072c3d8",
        "1424d99c-387b-537d-8533-b29e34fdad12",
        "a7715d96-f5a1-5240-811d-36f44221efec",
        (),
    ),
    (
        "pp",
        "7361f638-287b-5f4e-b9f2-7bbcb35751cd",
        "212c3ba3-1119-5a35-9ac9-6db21e556b3a",
        "218e8d1e-0dd1-5041-89a0-7ce3a64610f9",
        (_COARSE_PM, _MEDIUM_PM),
    ),
    (
        "pvc",
        "9164c69a-5f45-522e-a55b-2c9bb7341703",
        "fa6cade0-a03d-5fb9-a2fd-24f709fe0133",
        "bf17b7ce-e01a-59a2-bb12-6edb0acc34fd",
        (_MEDIUM_PM,),
    ),
    (
        "pet",
        "19881326-f93c-5aef-9750-fa7799f42266",
        "0804a093-da37-52eb-ba9a-849ddad41900",
        "d13ae29b-7d6a-55d2-8380-86343a80fed8",
        (_COARSE_PM,),
    ),
    (
        "ps",
        "dd9d94c4-0a47-5466-8ae4-fa6e665d8262",
        "943bc561-2342-5457-8c2d-0cd3e51ad4b3",
        "6c8a2cce-34a5-5ae8-995e-bc3fe7fe4ca2",
        (_COARSE_PM,),
    ),
    (
        "weee",
        "b03671bd-76bd-515a-9673-8dc43c7b5b94",
        "90062186-804f-5fbb-afac-6ceec863d9e1",
        "147d7548-199e-5261-b50e-3072861b6ecb",
        (_COARSE_PM,),
    ),
)
# Dedicated RER HDPE and LDPE pelletizers supply same-polymer handling PM. The generic non-food RER
# PET pelletizer supplies dry flake/output retention and PM. Its benchmark also serves PHB; its
# finishing/PM function serves engineering resins, PA and four biopolyesters. Food-grade PET LCI
# remains a separate finishing choice.
_HDPE_FINISHING = (
    "6a4c954c-d05c-5055-bbf4-17cf907f36d6",
    "1e0a3da9-3e39-5cc7-ac23-1f9bfcd4a54d",
)
_LDPE_FINISHING = (
    "cf1b331f-1a76-53b8-8aa6-d5d3b0ba5563",
    "aa4f5f2c-5e5f-5e66-bdc0-cc6a10790be0",
)
_PET_FINISHING = (
    "0b3f3af6-ea09-5cf0-8873-11175d2350eb",
    "2b6efd7e-d14e-51ef-89c1-620108488d92",
)
_PET_FLAKES = "0804a093-da37-52eb-ba9a-849ddad41900"


@dataclass(frozen=True, slots=True)
class PolymerMonomerRequirement:
    """Ethylene/propylene inputs per kg new granulate and nominal product carbon.

    ``allocation_factor`` records the original polymer-product allocation.
    Removing it gives the material requirements used by the MFA. LCA uses
    the original allocated exchange amounts."""

    polymer: str
    ethylene_kg_per_kg: float
    propylene_kg_per_kg: float
    carbon_kg_per_kg: float
    allocation_factor: float


@dataclass(frozen=True, slots=True)
class ChemicalProductYield:
    """Product kg/kg processed plastic (HTT) or naphtha-equivalent cracker feed.

    Carbon is the native product kg C/kg product.
    """

    product: str
    kg_per_kg_feed: float
    carbon_kg_per_kg: float


@dataclass(frozen=True, slots=True)
class PrincipalExit:
    """Kg principal material leaving a named operation per kg finished polymer."""

    name: str
    kg_per_finished_kg: float


@dataclass(frozen=True, slots=True)
class ConversionRequirement:
    """Principal feed and native service bases for one kg finished polymer.

    Generic operations describe gross processing: their reported finished
    fraction normalizes the LCA service, and its inverse determines MFA feed.
    PET spinning and the reconstructed operations already report finished
    principal output. Finished PUR includes conversion in its production inventory.
    Named principal exits retain the actual release sequence for carbon custody;
    their terminal LCA treatment remains with the corresponding inventory owner.
    """

    operation: str
    principal_feed_kg_per_finished_kg: float
    finished_kg_per_native_service_kg: float
    principal_exits: tuple[PrincipalExit, ...]


@dataclass(frozen=True, slots=True)
class MechanicalRecovery:
    """Fixed principal outlets per kg dry constituent entering mechanical recovery.

    Waterborne polymer remains reporting where no compatible native
    characterization exists. Airborne principal uses the admitted native dust
    transfer. Managed mass is 1 minus recovered, waterborne and airborne mass.
    TPS uses landfill; the other selected references use incineration. This
    complement defines the managed residue quantity.
    """

    name: str
    recovered_kg_per_input_kg: float
    waterborne_kg_per_input_kg: float
    airborne_kg_per_input_kg: float
    managed_treatment: Literal["incineration", "landfill"]


@dataclass(frozen=True, slots=True)
class RecyclingOperationBasis:
    """Native dry feed per kg reference output used to normalize recovery LCI.

    Preparation ratios normalize the allocated native inventory to its dry feed.
    The admitted PET finishing ratio also sets flakes per finished pellet kg.
    The separate recovery declaration determines actual secondary output.
    """

    operation: str
    dry_feed_kg_per_reference_kg: float


@dataclass(frozen=True, slots=True)
class MaterialSourceDataset:
    """Identify the exact licensed member underlying a coefficient."""

    member: str
    sha256: str
    activity_id: str
    product_id: str


@dataclass(frozen=True, slots=True)
class EcoinventMaterialRecipe:
    """Plastics inputs derived locally from the admitted source archive.

    Preparation values are kg plastic per kg original bale. Hydrothermal
    yields use processed plastic, and cracker yields use naphtha-equivalent
    feed. Conversion records distinguish principal feed per finished kg from
    finished output per native service kg. Report each quantity at its stated process step.
    """

    identity: str
    source_identity: str
    preparation_processed_kg_per_bale_kg: float
    preparation_reject_kg_per_bale_kg: float
    polymers: tuple[PolymerMonomerRequirement, ...]
    hydrothermal_products: tuple[ChemicalProductYield, ...]
    cracker_products: tuple[ChemicalProductYield, ...]
    conversion: tuple[ConversionRequirement, ...]
    mechanical_recovery: tuple[MechanicalRecovery, ...]
    recycling_operations: tuple[RecyclingOperationBasis, ...]
    datasets: tuple[MaterialSourceDataset, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the current material recipe contract for local persistence."""
        return {
            "schema_version": MATERIAL_RECIPE_SCHEMA,
            "method": MATERIAL_RECIPE_METHOD,
            **asdict(self),
        }

    @property
    def processed_plastic_fraction(self) -> float:
        """Return processed/total plastic at the chemical preparation gate.

        Total plastic is processed plus plastic reject, excluding bale water,
        metals and other nonplastics.
        """
        processed = self.preparation_processed_kg_per_bale_kg
        return processed / (processed + self.preparation_reject_kg_per_bale_kg)


def material_recipe_members() -> tuple[str, ...]:
    """Return the exact chemical, manufacturing and recovery sources for MFA."""
    selectors = (
        *((activity, product) for _, activity, product in _POLYMERS),
        _PREPARATION,
        *((_HTT_ACTIVITY, product) for _, product in _HTT_PRODUCTS),
        *((_CRACKER_ACTIVITY, product) for _, product in _CRACKER_PRODUCTS),
        *((activity, product) for _, activity, product in _CONVERSIONS),
        _PET_FIBRE,
        _UPR_OPERATION,
        _UPR_COMPOSITE,
        _UPR_RESIN,
        *((activity, product) for _, activity, product, _, _ in _RECYCLING),
        _HDPE_FINISHING,
        _LDPE_FINISHING,
        _PET_FINISHING,
    )
    return tuple(f"datasets/{activity}_{product}.spold" for activity, product in selectors)


def load_material_recipe(workspace: Path) -> EcoinventMaterialRecipe:
    """Read package-produced coefficients.

    The recipe producer owns source selection, scientific checks and identity.
    Reading this completed JSON preserves its numbers, ordering and provenance.
    """
    path = workspace / "data_processed" / "ecoinvent" / "material_recipe.json"
    if not path.is_file():
        action = (
            "Select a new workspace and run download_data"
            if path.parent.exists()
            else "Run download_data"
        )
        raise FileNotFoundError(
            f"The material recipe {path} is missing. {action} with the "
            "documented licensed ecoinvent inputs before run_mfa."
        )
    payload = json.loads(path.read_bytes())
    return EcoinventMaterialRecipe(
        identity=payload["identity"],
        source_identity=payload["source_identity"],
        preparation_processed_kg_per_bale_kg=payload["preparation_processed_kg_per_bale_kg"],
        preparation_reject_kg_per_bale_kg=payload["preparation_reject_kg_per_bale_kg"],
        polymers=tuple(PolymerMonomerRequirement(**row) for row in payload["polymers"]),
        hydrothermal_products=tuple(
            ChemicalProductYield(**row) for row in payload["hydrothermal_products"]
        ),
        cracker_products=tuple(ChemicalProductYield(**row) for row in payload["cracker_products"]),
        conversion=tuple(
            ConversionRequirement(
                operation=row["operation"],
                principal_feed_kg_per_finished_kg=row["principal_feed_kg_per_finished_kg"],
                finished_kg_per_native_service_kg=row["finished_kg_per_native_service_kg"],
                principal_exits=tuple(
                    PrincipalExit(**exit_row) for exit_row in row["principal_exits"]
                ),
            )
            for row in payload["conversion"]
        ),
        mechanical_recovery=tuple(
            MechanicalRecovery(**row) for row in payload["mechanical_recovery"]
        ),
        recycling_operations=tuple(
            RecyclingOperationBasis(**row) for row in payload["recycling_operations"]
        ),
        datasets=tuple(MaterialSourceDataset(**row) for row in payload["datasets"]),
    )


def _number(element: ElementTree.Element, attribute: str = "amount") -> float:
    """Read an evaluated finite XML number."""
    value = float(element.attrib[attribute])
    if not isfinite(value):
        raise ValueError(f"Nonfinite EcoSpold {attribute} on {element.attrib}")
    return value


def _exchange(
    flow_data: ElementTree.Element,
    product: str,
    *,
    reference: bool = False,
) -> ElementTree.Element:
    """Admit one exact kg reference product or purchased feed by product identity.

    EcoSpold uses outputGroup 0 for the reference and inputGroup 5 for purchases.
    """
    group, value = ("outputGroup", "0") if reference else ("inputGroup", "5")
    selected = [
        node
        for node in flow_data.findall(f"{_NS}intermediateExchange")
        if node.get("intermediateExchangeId") == product and node.findtext(f"{_NS}{group}") == value
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one EcoSpold {group}={value} exchange for {product}")
    result = selected[0]
    if result.get("unitId") != _KG or result.findtext(f"{_NS}unitName") != "kg":
        raise ValueError(f"EcoSpold product {product} must use kg")
    return result


def _property(reference: ElementTree.Element, property_id: str) -> float:
    """Read the source allocation, carbon fraction or kg dry mass per kg feed."""
    selected = [
        node
        for node in reference.findall(f"{_NS}property")
        if node.get("propertyId") == property_id
    ]
    unit = _KG if property_id == _DRY_MASS else _DIMENSIONLESS
    if len(selected) != 1 or selected[0].get("unitId") != unit:
        raise ValueError(f"Expected one EcoSpold property {property_id} with unit {unit}")
    value = _number(selected[0])
    if not 0.0 <= value <= 1.0 or (property_id == _ALLOCATION and value == 0.0):
        raise ValueError(f"Invalid EcoSpold property {property_id}: {value}")
    return value


def _elementary_amount(flow_data: ElementTree.Element, flow: str) -> float:
    """Read one selected nonnegative kg elementary output, preserving its identity."""
    selected = [
        node
        for node in flow_data.findall(f"{_NS}elementaryExchange")
        if node.get("elementaryExchangeId") == flow
    ]
    if len(selected) != 1 or selected[0].get("unitId") != _KG:
        raise ValueError(f"Expected one kg EcoSpold elementary output for {flow}")
    amount = _number(selected[0])
    if amount < 0:
        raise ValueError(f"Negative EcoSpold elementary output for {flow}: {amount}")
    return amount


def _finished_fraction(root: ElementTree.Element) -> float:
    """Read finished kg per gross process kg from the specified activity comment."""
    # The UPR service introduces the process in one text fragment and states
    # the finished fraction in a later fragment of the same activity comment.
    statement = " ".join(
        cast(str, text.text)
        for text in root.findall(
            f"{_NS}childActivityDataset/{_NS}activityDescription/"
            f"{_NS}activity/{_NS}generalComment/{_NS}text"
        )
    )
    return float(statement.split(" equals ", 1)[1].split(" kg ", 1)[0])


def _upr_conversion(
    roots: Mapping[str, ElementTree.Element],
    flows: Mapping[str, ElementTree.Element],
) -> ConversionRequirement:
    """Apply styrene release before trimming the declared UPR/glass formulation.

    Native sources supply glass/resin ratio, delivered-resin styrene, operation
    release and finished-composite fraction for the Others/UPR hand-lay-up case.
    The reference is 1 kg resin-application material before glass and additives.
    ACMA (2012) supplies the 65:535 gelcoat-to-remaining-resin ratio. The
    gelcoat contains 3% silica following Evonik/Vezzetti (2025), with 97%
    delivered orthophthalic resin. Oxytop METOX-50 (2024) supplies the 2 phr
    hardener dose; the selected 6% Co accelerator dose is 0.25 phr within
    ILT's 0.084--0.5 phr range. Both doses are kg per 100 kg application,
    including gelcoat silica.

    Total styrene release follows the complete native operating function.
    Divide it between resin and accelerator styrene, then remove the selected
    volatile carrier/water before homogeneous trimming. Auxiliaries affect
    service throughput and retained composite mass. Principal resin is recorded separately.
    """
    composite = flows[_UPR_COMPOSITE[1]]
    glass_ratio = _number(_exchange(composite, _GLASS_INPUT)) / _number(
        _exchange(composite, _UPR_INPUT)
    )
    resin = flows[_UPR_RESIN[1]]
    resin_styrene_fraction = _number(_exchange(resin, _STYRENE_INPUT)) / _number(
        _exchange(resin, _UPR_RESIN[1], reference=True)
    )
    operation = flows[_UPR_OPERATION[1]]
    styrene_per_service = _elementary_amount(operation, _STYRENE_AIR) / _number(
        _exchange(operation, _UPR_OPERATION[1], reference=True)
    )
    principal = 0.97 * 65 / 600 + 535 / 600
    total = fsum((principal, principal * glass_ratio, 0.03 * 65 / 600, 0.02, 0.0025))
    styrene = styrene_per_service * total
    principal_styrene = resin_styrene_fraction * principal
    # ILT, "Cobalt catalyst 6%", undated specification accessed 10 September 2026:
    # https://ilt.com.pl/en/cobalt-catalyst/#cobalt-catalyst-6
    # The additive contains 0.40 kg styrene and 0.24 kg white spirit/kg.
    # Oxytop METOX-50 (2024) supplies 0.023 kg water/kg hardener.
    # White spirit and this water evaporate in the selected dry endpoint;
    # other additive mass is retained before trimming.
    principal_release = styrene * principal_styrene / (principal_styrene + 0.40 * 0.0025)
    post_release = total - fsum((styrene, 0.24 * 0.0025, 0.023 * 0.02))
    retained_at_trimming = _finished_fraction(roots[_UPR_OPERATION[1]]) * total / post_release
    finished = retained_at_trimming * (principal - principal_release)
    return ConversionRequirement(
        "others_upr_hand_lay_up_conversion",
        principal / finished,
        finished / total,
        (
            PrincipalExit("styrene_air", principal_release / finished),
            PrincipalExit(
                "post_release_solid",
                (1 - retained_at_trimming) * (principal - principal_release) / finished,
            ),
        ),
    )


def _pa_fibre_conversion(pa_grade: Literal["pa6", "pa66"]) -> ConversionRequirement:
    """Close PA fiber principal feed, selective monomer loss and cleaning deposits.

    Haian's Nantong Jiuzhou EIA (2022) supplies 0.01% raw PA adhesion to
    spinning equipment with 90% internal melt return. The remaining dry
    deposits undergo controlled oxidative cleaning. This annual raw-input
    fraction applies to both grades, all years and backgrounds.
    Incoming embedded finish follows the same adhesion separately in
    inventory custody.
    """
    # Polymers BREF 2007, Table 9.4 reports 30--35 g/t caprolactam to air; Table 9.9 reports 25--117
    # g/t under "Discharges to water". Here, the midpoints apply per tonne of finished principal
    # PA6. The water midpoint is modeled as collected caprolactam before the single
    # wastewater-treatment owner. The upstream position follows the model's engineering assumption.
    # Divide by 1e6 for kg/kg. PA66 uses the BREF textile reference.
    exits = (
        (
            PrincipalExit("caprolactam_air", 0.0000325),
            PrincipalExit("caprolactam_collected_water", 0.000071),
        )
        if pa_grade == "pa6"
        else ()
    )
    # Solve G=G0+bG once, with b=.01/100*(1-.90). Returned bulk melt is internal.
    destroyed_fraction = 0.0001 * (1 - 0.90)
    gross = (1 + fsum(exit.kg_per_finished_kg for exit in exits)) / (1 - destroyed_fraction)
    return ConversionRequirement(
        f"fibres_{pa_grade}_conversion",
        gross,
        1.0,
        (*exits, PrincipalExit("spinning_equipment_deposits", destroyed_fraction * gross)),
    )


def _reconstructed_conversions() -> tuple[ConversionRequirement, ...]:
    """Calculate principal feed and exits for the selected finished-product cases.

    Public plant/formulation studies supply tire, fiber, SBR-profile, epoxy
    coating and road-marking references. Their units and assumptions are
    specified below. A service basis of 1 means the reconstructed inventory
    already uses finished principal kg.
    """
    # Others non-tire SBR: EPA 2008 compound-22 workbook gives kg monomer/kg
    # complete compound (AP-42 Table 4.12-2 formulation). Styrene uses mixing
    # X64, extrusion F49 and hot-air E31; butadiene uses mixing X10 and extrusion
    # F10. The factor 1.858 kg compound/kg SBR converts these emission rates
    # to principal feed.
    styrene = 1.858 * fsum((1.4786094907418637e-6, 3.9285095052594246e-8, 4.505773049721235e-7))
    butadiene = 1.858 * fsum((1.3054683728346342e-7, 7.82694259043066e-8))
    # TRC (1995), vol.II PDF p.21: corrected extrusion PM transferred with homogeneous compound
    # composition gives 0.0000266 kg dust/kg SBR feed. Carli (2008), pp.79-81, Fig. 32/Table 15:
    # 550/11000 kg finishing scrap, hence 95% retention AFTER monomer/dust release.
    sbr_finished = 0.95 * (1 - fsum((styrene, butadiene, 0.0000266)))
    sbr_feed = 1 / sbr_finished
    sbr_styrene = styrene * sbr_feed
    sbr_butadiene = butadiene * sbr_feed
    sbr_dust = 0.0000266 * sbr_feed
    # Others epoxy: EP1838750B1 Table 6 comparative 3, pp.9-10, gives GY250/MXDA
    # mass fractions 0.8456/0.1544. GY250 TDS and Merck X1202 give densities
    # 1.17/1.032 kg/L; mixture density uses ideal volume addition.
    mixture_density = 1 / (0.8456 / 1.17 + 0.1544 / 1.032)
    # IERA Elmendorf (1999), pp.2-10--2-12: proportioning-case preparation
    # waste 1400 lb/1700 US gal; 0.45359237 kg/lb and 3.785411784 L/US gal.
    # Transfer this whole preparation function before spraying neat binder.
    preparation_loss = (1400 * 0.45359237 / (1700 * 3.785411784)) / mixture_density
    # NEWMOA/EPA (1998), Table 41 p.101 and Appendix G pp.157-159:
    # the airless-transfer midpoint is (65+70)/2 percent. The selected
    # 95% settled/5% airborne overspray split transfers rough shipyard
    # paint-solids estimates from EPA453-D-94-011a, Appendix C pp.C-9--C-11.
    epoxy_finished = (1 - preparation_loss) * 0.675
    epoxy_feed = 1 / epoxy_finished
    epoxy_overspray = (1 - preparation_loss) * (1 - 0.675) * epoxy_feed
    # Others road marking: Silikal 2016 SM1.01.A R610HW pp.16-17 supplies
    # 0.223 kg reactive binder/kg premix plus 1% hardener. DBC's 2020 MMA mortar
    # EPD pp.6-8 A5 supplies 1% homogeneous container residue and 0.002 kg MMA
    # air/kg ready mixture. Transfer to ambient manual application: residue
    # removes binder first, then MMA leaves premix+hardener (1.01 kg).
    road_binder = 0.223
    road_finished = road_binder * 0.99 - 0.002 * 1.01
    road_feed = road_binder / road_finished
    # Tires: Shanbag/Manjare (2020), DOI 10.13044/j.sdewes.d7.0260, p.27 Table 2: 0.4 compound-stock
    # + 7.5 coated-fabric + 20.4 bead scrap kg/t FINISHED tire. Assume equal principal fraction in
    # aggregate scrap/output; it cancels, so gross SBR-equivalent feed is 1 + scrap/finished.
    # Finished-output utilities stay separate.
    tyre_feed = 1 + (0.4 + 7.5 + 20.4) / 1000
    return (
        # PP fiber: the selected RoW polypropylene nonwoven spunbond operation reworks scrap
        # internally. Unity represents this closed loop.
        ConversionRequirement("fibres_pp_conversion", 1.0, 1.0, ()),
        _pa_fibre_conversion("pa6"),
        _pa_fibre_conversion("pa66"),
        ConversionRequirement(
            "conversion_tyre",
            tyre_feed,
            1.0,
            (PrincipalExit("principal_remainder", tyre_feed - 1),),
        ),
        # The selected finished-product LCI includes PUR fabrication.
        ConversionRequirement("pur_conversion_included_in_finished_products", 1.0, 1.0, ()),
        ConversionRequirement(
            "others_esbr_compounding_conversion",
            sbr_feed,
            1.0,
            (
                PrincipalExit("styrene_air", sbr_styrene),
                PrincipalExit("butadiene_air", sbr_butadiene),
                PrincipalExit("extrusion_dust", sbr_dust),
                PrincipalExit(
                    "post_release_solid",
                    0.05 * (sbr_feed - fsum((sbr_styrene, sbr_butadiene, sbr_dust))),
                ),
            ),
        ),
        ConversionRequirement(
            "others_epoxy_coating_conversion",
            epoxy_feed,
            1.0,
            (
                PrincipalExit("preparation_residue", preparation_loss * epoxy_feed),
                PrincipalExit("settled_overspray", 0.95 * epoxy_overspray),
                PrincipalExit("airborne_overspray", 0.05 * epoxy_overspray),
            ),
        ),
        ConversionRequirement(
            "others_road_marking_conversion",
            road_feed,
            1.0,
            (
                PrincipalExit("container_residue", 0.01 * road_feed),
                PrincipalExit("methyl_methacrylate_air", (0.00202 / road_binder) * road_feed),
            ),
        ),
    )


def _conversion_requirements(
    roots: Mapping[str, ElementTree.Element],
    flows: Mapping[str, ElementTree.Element],
) -> tuple[ConversionRequirement, ...]:
    """Read the selected process descriptions and evaluated PET principal input.

    The six source activities supply finished kg per kg of process in
    their activity comments. For a generic service, f kg finished/kg gross
    gives g=1/f kg principal feed/kg finished and a principal remainder g-1.
    Native waste-treatment quantities retain their separate inventory bases.
    PET fiber uses evaluated resin input per finished output and already has
    a finished-output service basis.
    """
    requirements = []
    for operation, _, product in _CONVERSIONS:
        finished = _finished_fraction(roots[product])
        gross = 1.0 / finished
        requirements.append(
            ConversionRequirement(
                operation, gross, finished, (PrincipalExit("principal_remainder", gross - 1),)
            )
        )
    fibre = flows[_PET_FIBRE[1]]
    output = _number(_exchange(fibre, _PET_FIBRE[1], reference=True))
    feed = _number(_exchange(fibre, _PET_GRANULATE))
    gross = feed / output
    requirements.append(
        ConversionRequirement(
            "fibres_pet_conversion", gross, 1.0, (PrincipalExit("principal_remainder", gross - 1),)
        )
    )
    requirements.extend(_reconstructed_conversions())
    requirements.append(_upr_conversion(roots, flows))
    return tuple(requirements)


def _mechanical_recoveries(
    flows: Mapping[str, ElementTree.Element],
) -> tuple[tuple[MechanicalRecovery, ...], tuple[RecyclingOperationBasis, ...]]:
    """Combine fixed source recovery stages with locally derived principal dust.

    Each stage retains kg target principal per kg target entering that stage.
    The gate describes sorting and preparation after collection.
    Brouwer sorting precedes the specified washing/separation/finishing stages;
    engineering, PA, PVC, TPS and four-biopolyester cases have distinct sources.
    Native preparation quantities normalize the allocated inventory using its wet-feed basis.
    Pure-polymer recovery follows the separate material recipe.
    Generic PET finishing supplies the dry flake/pellet benchmark for PET/PHB;
    other yields use the explicit process transfers below.
    """
    dust: dict[str, float] = {}
    operations = []
    for name, _, product, feed_product, particle_flows in _RECYCLING:
        data = flows[product]
        feed = _exchange(data, feed_product)
        # Original XML: allocated wet kg times kg dry/kg wet, divided by reference kg. Store b for
        # LCI T/b normalization.
        dry_feed = _number(feed) * _property(feed, _DRY_MASS)
        output = _number(_exchange(data, product, reference=True))
        if min(dry_feed, output) <= 0:
            raise ValueError(f"Nonpositive dry recycling feed or reference output for {name}")
        operations.append(RecyclingOperationBasis(name, dry_feed / output))
        # Selected native PM/dry input kg is transferred as principal dust.
        # It remains inside total principal loss, with its native size class.
        dust[name] = fsum(_elementary_amount(data, flow) for flow in particle_flows) / dry_feed
    pet = flows[_PET_FINISHING[1]]
    pet_output = _number(_exchange(pet, _PET_FINISHING[1], reference=True))
    pet_feed = _exchange(pet, _PET_FLAKES)
    pet_dry_feed = _number(pet_feed) * _property(pet_feed, _DRY_MASS)
    hdpe = flows[_HDPE_FINISHING[1]]
    hdpe_output = _number(_exchange(hdpe, _HDPE_FINISHING[1], reference=True))
    ldpe = flows[_LDPE_FINISHING[1]]
    ldpe_output = _number(_exchange(ldpe, _LDPE_FINISHING[1], reference=True))
    if min(pet_output, pet_dry_feed, hdpe_output, ldpe_output) <= 0:
        raise ValueError("Nonpositive pelletising production or dry feed")
    operations.append(RecyclingOperationBasis("pet_finishing", pet_dry_feed / pet_output))
    pet_finish_yield = pet_output / pet_dry_feed
    # Finishing PM is kg/kg pellet output: multiply by recovered output once.
    # Same-polymer PE finishers supply PM, while Lase supplies their retention.
    pet_dust = _elementary_amount(pet, _COARSE_PM) / pet_output
    hdpe_dust = _elementary_amount(hdpe, _COARSE_PM) / hdpe_output
    ldpe_dust = _elementary_amount(ldpe, _COARSE_PM) / ldpe_output

    # Brouwer et al. (2018), DOI 10.1016/j.wasman.2017.10.034, C.2 pp.37-38
    # and F.1 p.46: target Mg in all five saleable fractions / incoming target
    # Mg in the Dutch reference. All five fractions enter compatible processing.
    # The aggregate PE composition is transferred to both HDPE and LDPE.
    pe_sort = 31199.69 / 33540
    pp_sort = 19377.32 / 21930
    pet_sort = 19205.93 / 21930
    ps_sort = 3847.08 / 5160
    # Lase SI Table S3.2, ordered retentions: HDPE cold wash .99, float .98, extrusion .97. LDPE
    # first has .99 negative-NIR retention (Table SI3.1), then the same three stages. PP cold wash
    # .99, float .99, extrusion .97.
    hdpe_recovered = pe_sort * 0.99 * 0.98 * POLYOLEFIN_PELLET_RETENTION
    ldpe_recovered = pe_sort * 0.99 * 0.99 * 0.98 * POLYOLEFIN_PELLET_RETENTION
    pp_recovered = pp_sort * 0.99 * 0.99 * POLYOLEFIN_PELLET_RETENTION
    # PET/PS: transfer Lase's .99 cold wash; Brouwer Table A.1 p.32 supplies measured cold-water
    # SINK retention .99/.83, respectively. PET then uses its dry pelletizer benchmark; PS transfers
    # Lase .97 extrusion retention. The native PS operation already extrudes.
    pet_recovered = pet_sort * 0.99 * 0.99 * pet_finish_yield
    ps_recovered = ps_sort * 0.99 * 0.83 * 0.97
    # ABS/SAN, PC and PMMA: Tinz et al. (2022), DOI 10.3390/waste1010010,
    # pp.129-131, .999 receiving retention then .96 grinding retention for
    # nearly pure industrial sprues; Lase .97 finishing is a further transfer.
    engineering_recovered = 0.999 * 0.96 * 0.97
    # PA6/PA66: Radici Scrap Recovery EPD S-P-00708 rev3 (2018), pp.3-6, .98 whole
    # grinding+extrusion retention.
    pa_recovered = 0.98
    # PLA/PBAT/PBS/PTT: Maga (2019), DOI 10.1016/j.resconrec.2019.05.018,
    # sections 2.3-2.6, PLA target sorting .55; transfer Lase cold wash .99;
    # Terrien et al., Materia Nova (2023), Technologies for the recycling
    # of plastic waste: Mass, energy and climate change balances, supplies
    # .97 dry grinding/extrusion. These factors also apply to PBAT/PBS/PTT.
    biopolyester_recovered = 0.55 * 0.99 * 0.97
    # Waterborne principal is the 1% cold-wash loss after previous selection (LDPE includes negative
    # NIR). Zero rows describe the selected dry/controlled reference. WEEE solvent VOC and PP NMVOC
    # retain their separate emissions accounting. WEEE preparation plus generic PET finishing PM
    # serves engineering/PA/ biopolyester targets. PP/PS/PVC already include their native extrusion
    # PM.
    declarations = (
        (
            "hdpe_granulate",
            hdpe_recovered,
            pe_sort * 0.01,
            dust["hdpe"] + hdpe_recovered * hdpe_dust,
        ),
        (
            "ldpe_granulate",
            ldpe_recovered,
            pe_sort * 0.99 * 0.01,
            dust["ldpe"] + ldpe_recovered * ldpe_dust,
        ),
        ("pp_granulate", pp_recovered, pp_sort * 0.01, dust["pp"]),
        ("pet_granulate", pet_recovered, pet_sort * 0.01, dust["pet"] + pet_recovered * pet_dust),
        ("ps_granulate", ps_recovered, ps_sort * 0.01, dust["ps"]),
        (
            "engineering_granulate",
            engineering_recovered,
            0.0,
            dust["weee"] + engineering_recovered * pet_dust,
        ),
        ("pa_granulate", pa_recovered, 0.0, dust["weee"] + pa_recovered * pet_dust),
        # EC DG XI, Mechanical Recycling of PVC Wastes (2000), pp.80-82:
        # pipe/fitting sorting .9 then mechanical processing .9. Its separate
        # collection rate is outside this gate; rigid-PVC reference is retained.
        ("pvc_granulate", 0.9 * 0.9, 0.0, dust["pvc"]),
        # Surendren (2024), DOI 10.3390/su16177406, Tables 1-3 pp.4-5:
        # .15 kg loss/kg gross OUTPUT, hence input=1.15 and eta=1/1.15.
        # Transfer virgin starch/glycerol processing with proportional losses;
        # technical moisture/protein/solutes retain separate formulation custody.
        ("tps_compound", 1 / 1.15, 0.0, 0.0),
        # PHB: generic PET dry feed/output is the hygroscopic-polyester proxy. Rivas (2017), DOI
        # 10.1590/0104-1428.2406, pp.123-125 establishes reprocessability. PHB conditioning energy
        # belongs to its separate LCI function.
        ("phb_granulate", pet_finish_yield, 0.0, pet_finish_yield * pet_dust),
        (
            "biopolyester_granulate",
            biopolyester_recovered,
            0.55 * 0.01,
            dust["weee"] + biopolyester_recovered * pet_dust,
        ),
    )
    recoveries = []
    for name, recovered, water, air in declarations:
        # This remaining principal mass replaces the overlapping native reject.
        managed = 1 - fsum((recovered, water, air))
        if min(recovered, water, air, managed) < 0:
            raise ValueError(f"Recovery outlets exceed the dry principal feed for {name}")
        recoveries.append(
            MechanicalRecovery(
                name,
                recovered,
                water,
                air,
                "landfill" if name == "tps_compound" else "incineration",
            )
        )
    return tuple(recoveries), tuple(operations)


def derive_material_recipe(
    members: Mapping[str, bytes],
    *,
    source_identity: str,
) -> EcoinventMaterialRecipe:
    """Derive coefficients from the admitted archive's selected members.

    Granulate monomer purchases are divided by reference production and the
    admitted allocation factor. Each hydrothermal yield is reference output
    times allocation divided by its allocated processed-feed purchase. Cracker
    feed is the stable sum of every coproduct's production volume times its
    allocated feed intensity; each product's volume is divided by that sum.
    Generic conversion uses the activity's stated finished fraction; PET fiber
    uses evaluated resin input per finished output.

    These original source references are retained when premise changes dated
    utility suppliers or rounds exchange metadata. Actual LCI purchases keep
    their allocated source amounts. Named HTT heavy oil/wax and nonpolymer cracker
    coproducts are reported as outputs leaving the plastics system.


    Source admission owns archive authentication. This transformation checks
    units and signs and retains member hashes in the derived identity.
    """
    datasets: list[MaterialSourceDataset] = []
    roots: dict[str, ElementTree.Element] = {}
    flows: dict[str, ElementTree.Element] = {}
    for member in material_recipe_members():
        data = members[member]
        root = ElementTree.fromstring(data)
        flow_data = root.find(f"{_NS}childActivityDataset/{_NS}flowData")
        if flow_data is None:
            raise ValueError(f"Missing EcoSpold flowData in {member}")
        activity, product = member.removeprefix("datasets/").removesuffix(".spold").split("_")
        roots[product] = root
        flows[product] = flow_data
        datasets.append(MaterialSourceDataset(member, sha256(data).hexdigest(), activity, product))
    conversion = _conversion_requirements(roots, flows)
    mechanical_recovery, recycling_operations = _mechanical_recoveries(flows)

    polymers: list[PolymerMonomerRequirement] = []
    for polymer, _, product in _POLYMERS:
        flow_data = flows[product]
        reference = _exchange(flow_data, product, reference=True)
        output = _number(reference)
        # kg allocated monomer / kg granulate / product allocation gives the
        # feed; the selected RER HDPE/PP records have unity allocation.
        allocation = _property(reference, _ALLOCATION) if polymer == "ldpe" else 1.0
        ethylene = _number(_exchange(flow_data, _ETHYLENE))
        propylene = _number(_exchange(flow_data, _PROPYLENE))
        if min(output, ethylene, propylene) <= 0.0:
            raise ValueError(f"Nonpositive granulate production or monomer input for {polymer}")
        polymers.append(
            PolymerMonomerRequirement(
                polymer,
                ethylene / output / allocation,
                propylene / output / allocation,
                _property(reference, _CARBON_CONTENT),
                allocation,
            )
        )

    # Negative kg waste-reference/transfer exchanges become positive masses. Store
    # processed/rejected plastic per original bale kg; their sum excludes the bale's other
    # materials. processed_plastic_fraction owns the later tracked-plastic normalization with
    # collection losses counted once.
    preparation = flows[_PREPARATION[1]]
    bale = -_number(_exchange(preparation, _PREPARATION[1], reference=True))
    processed = -_number(_exchange(preparation, _PROCESSED_PLASTIC))
    reject = -_number(_exchange(preparation, _PLASTIC_REJECT))
    if min(bale, processed, reject) <= 0.0 or processed + reject > bale:
        raise ValueError("Invalid native preparation waste signs or tracked-plastic balance")

    # Each allocated HTT sibling's reference kg * allocation / processed-feed kg reconstructs kg
    # product/kg processed plastic. All four outputs are retained. Naphtha and DGO supply the
    # cracker feed.
    hydrothermal: list[ChemicalProductYield] = []
    for name, product in _HTT_PRODUCTS:
        flow_data = flows[product]
        reference = _exchange(flow_data, product, reference=True)
        output = _number(reference)
        feed = _number(_exchange(flow_data, _PROCESSED_PLASTIC))
        allocation = _property(reference, _ALLOCATION)
        if min(output, feed) <= 0.0:
            raise ValueError(f"Nonpositive hydrothermal production or processed feed for {name}")
        hydrothermal.append(
            ChemicalProductYield(
                name, output * allocation / feed, _property(reference, _CARBON_CONTENT)
            )
        )

    # Original sibling reference exchanges carry productionVolumeAmount in
    # kg/year on one period (XML production-volume field). For every sibling,
    # V_j * allocated naphtha kg/reference kg gives annual feed contribution.
    # Sum all seven before V_j/sum(feed): one coproduct alone cannot determine
    # yield. PlasticsEurope's recommendation on Steam Cracker allocation
    # (14 September 2017), pp. 2-3, distinguishes feed allocation from
    # intended-component operating burdens; the native BTX composition/allocation
    # comments preserve that distinction.
    production: list[float] = []
    feed_requirements: list[float] = []
    carbon: list[float] = []
    for name, product in _CRACKER_PRODUCTS:
        flow_data = flows[product]
        reference = _exchange(flow_data, product, reference=True)
        output = _number(reference)
        volume = _number(reference, "productionVolumeAmount")
        feed = _number(_exchange(flow_data, _NAPHTHA))
        if min(output, volume, feed) <= 0.0:
            raise ValueError(f"Nonpositive cracker production or naphtha feed for {name}")
        production.append(volume)
        feed_requirements.append(volume * (feed / output))
        carbon.append(_property(reference, _CARBON_CONTENT))
    total_feed = fsum(feed_requirements)
    cracker = tuple(
        ChemicalProductYield(name, volume / total_feed, fraction)
        for (name, _), volume, fraction in zip(_CRACKER_PRODUCTS, production, carbon, strict=True)
    )
    if (
        fsum(item.kg_per_kg_feed for item in hydrothermal) > 1.0
        or fsum(item.kg_per_kg_feed for item in cracker) > 1.0
    ):
        raise ValueError("Named hydrothermal or cracker products exceed their feed")

    payload = {
        "schema_version": MATERIAL_RECIPE_SCHEMA,
        "method": MATERIAL_RECIPE_METHOD,
        "source_identity": source_identity,
        "preparation_processed_kg_per_bale_kg": processed / bale,
        "preparation_reject_kg_per_bale_kg": reject / bale,
        "polymers": [asdict(item) for item in polymers],
        "hydrothermal_products": [asdict(item) for item in hydrothermal],
        "cracker_products": [asdict(item) for item in cracker],
        "conversion": [asdict(item) for item in conversion],
        "mechanical_recovery": [asdict(item) for item in mechanical_recovery],
        "recycling_operations": [asdict(item) for item in recycling_operations],
        "datasets": [asdict(item) for item in datasets],
    }
    return EcoinventMaterialRecipe(
        identity=f"{MATERIAL_RECIPE_SCHEMA}::{payload_sha256(payload)}",
        source_identity=source_identity,
        preparation_processed_kg_per_bale_kg=processed / bale,
        preparation_reject_kg_per_bale_kg=reject / bale,
        polymers=tuple(polymers),
        hydrothermal_products=tuple(hydrothermal),
        cracker_products=cracker,
        conversion=conversion,
        mechanical_recovery=mechanical_recovery,
        recycling_operations=recycling_operations,
        datasets=tuple(datasets),
    )


def prepare_material_recipe(workspace: Path, spec: SourceSpec) -> bool:
    """Publish the licensed input once; return whether it was created.

    The licensed archive has already been verified. The public py7zr reader
    extracts the selected members into memory. The completed JSON is written
    last in a hidden sibling folder, then published by rename. Later calls
    reuse the completed recipe.
    """
    destination = workspace / "data_processed" / "ecoinvent"
    if (destination / "material_recipe.json").is_file():
        return False
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"The material recipe directory {destination} has no completion file. "
            "Keep it unchanged and select a new workspace before running download_data."
        )
    archive_path = source_directory(workspace, spec) / spec.filename
    members = material_recipe_members()
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        largest_member = max(
            info.uncompressed for info in archive.list() if info.filename in members
        )
        factory = BytesIOFactory(limit=largest_member)
        # py7zr 1.1.3 accepts lists or sets, despite its Collection annotation.
        archive.extract(targets=list(members), factory=factory)
        content = {member: factory.get(member).read() for member in members}
    recipe = derive_material_recipe(content, source_identity=f"{spec.source_id}::{spec.sha256}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".ecoinvent_materials.", dir=destination.parent, ignore_cleanup_errors=True
    ) as temporary:
        staging = Path(temporary)
        write_workspace_guide(staging, "pytoce.download_data", "ecoinvent_material_recipe")
        atomic_write(staging / "material_recipe.json", canonical_json_bytes(recipe.as_dict()))
        replace_directory(staging, destination)
    return True
