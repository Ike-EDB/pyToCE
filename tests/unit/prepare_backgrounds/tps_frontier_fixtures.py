"""Join invented site, thermal and native HWI inventories for the real TPS frontier.

All three packaged formulations, selectors and service constructions are real.
Native quantities are small invented test controls. The one shared
residual-landfill source combines MC support with the full TPS donor assay and
additional elemental rows; it is an explicit test inventory, using synthetic quantities.
"""

from copy import deepcopy
from math import log

from native_hwi_fixtures import (
    native_hwi_biosphere,
    native_hwi_inventory,
)
from tps_site_fixtures import biosphere_inventory as site_biosphere
from tps_site_fixtures import native_inventory as site_native
from tps_treatment_fixtures import biosphere_inventory as thermal_biosphere
from tps_treatment_fixtures import native_inventory as thermal_native

from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    load_provider_catalogue,
)

ORIGINS = ("biogenic", "fossil", "atmospheric_dac")
TPS_SUFFIXES = (
    "landfill",
    "leachate_collection",
    "wastewater_treatment",
    "sludge_incineration_market",
    "sludge_landfarming_market",
    "sludge_incineration",
    "sludge_landfarming",
    "bottom_ash_market",
    "residual_ash_market",
    "bottom_ash_treatment",
    "residual_ash_treatment",
)
ACTIVE_SERVICES = frozenset(
    [f"tps_residue_treatment_{origin}" for origin in ORIGINS]
    + [f"tps_{origin}_{suffix}" for origin in ORIGINS for suffix in TPS_SUFFIXES]
)


def _references(value: object) -> set[str]:
    """Keep real inventory and source-only references in the bounded catalog."""
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"service_id", "formulation_service_id", "incineration_service_id"}:
                result.add(child)
            else:
                result.update(_references(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_references(child))
    return result


def frontier_catalogue() -> dict:
    """Select the real disposal graph while keeping source-only recipes unchanged.

    Only activation flags change on quantity-only records. Their full constructions
    remain available to preparation provenance and actual TPS load extraction.
    The actual RoW hydraulic-water pair is shared with the MC supply chain.
    """
    catalogue = load_provider_catalogue()
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    required = set(ACTIVE_SERVICES)
    pending = list(required)
    while pending:
        current = pending.pop()
        for child in _references(services[current]["construction"]) - required:
            required.add(child)
            pending.append(child)
    selected = []
    for service in catalogue["services"]:
        sid = service["provider_service_id"]
        if sid not in required:
            continue
        if sid not in ACTIVE_SERVICES:
            service["foreground_service"] = False
            service["frontier_service"] = False
        selected.append(service)
    catalogue["services"] = selected
    catalogue["routes"] = [
        row
        for row in catalogue["routes"]
        if row["provider_service_id"].startswith("tps_residue_treatment_")
    ]
    return catalogue


def flow_key(name: str, unit: str, categories: tuple[str, ...]) -> tuple[str, str]:
    """Join identical external elementary semantics, retaining unit distinctions."""
    return "biosphere", "::".join((name, unit, "/".join(categories)))


def biosphere_inventory() -> dict:
    """Combine the three synthetic biospheres by unique selector."""
    result = {}
    for inventory in (site_biosphere(), thermal_biosphere(), native_hwi_biosphere()):
        for row in inventory.values():
            result[flow_key(row["name"], row["unit"], row["categories"])] = row
    return result


def _namespaced(inventory: dict, biosphere: dict, family: str) -> dict:
    """Separate invented supplier codes while preserving native names and units."""
    result = {}
    for key, node in inventory.items():
        for row in node["exchanges"]:
            old = row["input"]
            if row["type"] == "biosphere":
                descriptor = biosphere[old]
                row["input"] = flow_key(
                    descriptor["name"], descriptor["unit"], descriptor["categories"]
                )
            else:
                row["input"] = old[0], family + ":" + old[1]
        result[key[0], family + ":" + key[1]] = node
    return result


def native_inventory(database: str, *, future: bool) -> dict:
    """Build one consistent native source shared by every connected TPS owner.

    The residual-ash source keeps the native fixture's complete 0.4 kg cement and 0.6 kg hydration
    support per reference kg, full native-shaped retirement, and source carbon rows.
    Its intensive 31-element assay is the independently normalized thermal fixture
    assay. Missing elemental rows retain their invented per-kg coefficients after
    conversion from thermal p=-3/-7 to shared HWI p=-5/-10. This joins the declared fixtures while
    retaining the TPS coefficients.

    Source fixtures use the full declarations because the hydraulic fixture shares all four native
    water profiles. The separately bounded frontier activates only its RoW pair.
    """
    result = _namespaced(site_native(database, future=future), site_biosphere(), "site")
    result.update(
        _namespaced(thermal_native(database, future=future), thermal_biosphere(), "thermal")
    )
    result.update(
        _namespaced(
            native_hwi_inventory(load_provider_catalogue(), database, original=not future),
            native_hwi_biosphere(),
            "mc",
        )
    )
    # The dispatcher reads climate only from the original native plant. These
    # invented MAT/MAP/EAT controls use the original workbook's admitted units:
    # degrees Celsius / mm per year / mm per year. Future metadata is absent.
    if not future:
        result[database, "site:plant"]["comment"] = (
            "Invented (MAT) of 12; (MAP) of 1000; (EAT) of 500."
        )
    ash = result[database, "mc:mc_ash"]
    thermal_ash = result[database, "thermal:residual"]
    production = next(row for row in ash["exchanges"] if row["type"] == "production")
    thermal_p = next(row for row in thermal_ash["exchanges"] if row["type"] == "production")
    production["properties"] = deepcopy(thermal_p["properties"])
    keys = {row["input"] for row in ash["exchanges"] if row["type"] == "biosphere"}
    factor = production["amount"] / thermal_p["amount"]
    for row in thermal_ash["exchanges"]:
        if row["type"] != "biosphere" or row["input"] in keys:
            continue
        added = deepcopy(row)
        added["amount"] *= factor
        added["loc"] += log(factor)
        added["minimum"] *= factor
        added["maximum"] *= factor
        ash["exchanges"].append(added)
    # A complete shared source serves both fixtures at the common selector. The site furnace
    # supplies fuel/LHV and dated operating support.
    aliases = {
        (database, "thermal:residual"): (database, "mc:mc_ash"),
        (database, "thermal:actual_pam"): (database, "mc:mc_pam"),
        (database, "mc:heat_europe"): (database, "site:furnace"),
    }
    for old in aliases:
        del result[old]
    for node in result.values():
        for row in node["exchanges"]:
            row["input"] = aliases.get(row["input"], row["input"])
    return result
