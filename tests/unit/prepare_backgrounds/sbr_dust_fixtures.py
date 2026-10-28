"""Synthetic hazardous-waste incineration inputs for SBR dust treatment."""

from math import log
from typing import Any

from native_hwi_fixtures import native_hwi_biosphere, native_hwi_inventory


def sbr_dust_biosphere() -> dict[tuple[str, str], dict[str, Any]]:
    """Provide source flow names and units with synthetic IDs and no impact factors."""
    data = native_hwi_biosphere()
    urban = ("air", "urban air close to ground")
    surface, long = ("water", "surface water"), ("water", "ground-, long-term")
    for code, name, unit, categories in (
        ("dust_cyanide", "Cyanide", "kilogram", urban),
        ("dust_so2", "Sulfur dioxide", "kilogram", urban),
        ("dust_zinc_air", "Zinc II", "kilogram", urban),
        ("dust_heat_air", "Heat, waste", "megajoule", urban),
        ("dust_sulfate_short", "Sulfate", "kilogram", surface),
        ("dust_sulfate_long", "Sulfate", "kilogram", long),
        ("dust_iron_short", "Iron II", "kilogram", surface),
        ("dust_iron_long", "Iron II", "kilogram", long),
    ):
        data["biosphere", code] = {
            "name": name,
            "unit": unit,
            "categories": categories,
            "type": "emission",
        }
    return data


def sbr_dust_native_inventory(
    catalogue: dict[str, Any], database: str, *, original: bool
) -> dict[tuple[str, str], dict[str, Any]]:
    """Add synthetic nitrogen, sulfur, metal and dated supplier inputs.

    Original/future reference amounts are -3/-6 kg waste, +4/+8 kg support and
    -5/-10 kg ash. Wet mass fractions are carbon .3, nitrogen .01 and sulfur .02.
    These and all other declared elements are synthetic; no licensed data are used.
    Future supplier ratios and amounts differ. The treatment calculation retains
    original calibration quantities per kilogram while using dated supplier shares.
    The ash inventory includes water
    so tests can check that the treatment calculation retains that input.
    """
    data = native_hwi_inventory(catalogue, database, original=original)
    if not original:
        # The future fixture uses the generic industrial-heat product label for
        # the shared HWI, fixed-support and ceramic-forming supplier.
        # Their declared heat quantities are unchanged.
        data[database, "mc_ng"]["reference product"] = "heat, district or industrial"

    def exchange(code: str, amount: float, *, bio: bool = False) -> dict[str, Any]:
        """Attach a distinguishable source distribution to the synthetic row."""
        return {
            "input": ("biosphere" if bio else database, code),
            "type": "biosphere" if bio else "technosphere",
            "amount": amount,
            "loc": log(amount) + 0.02,
            "uncertainty type": 2,
            "scale": 0.19,
            "minimum": amount / 3,
            "maximum": 3 * amount,
            "allocation": "invented dust source",
            "comment": "Synthetic HWI test coefficient",
        }

    def leaf(code: str, product: str) -> None:
        """Represent a synthetic supplier with identifiable emissions."""
        data[database, code] = {
            "name": f"invented {code}",
            "reference product": product,
            "unit": "kilogram",
            "location": "RER" if original else "EUR",
            "type": "process",
            "exchanges": [
                {"input": (database, code), "type": "production", "amount": 2},
                exchange("co2", 0.08 if original else 0.12, bio=True),
            ],
        }

    leaf("dust_peroxide", "hydrogen peroxide, without water, in 50% solution state")
    leaf("dust_ferric", "iron(III) chloride, without water, in 40% solution state")
    leaf("dust_second_ti", "titanium dioxide")
    leaf("dust_second_hcl", "hydrochloric acid, without water, in 30% solution state")
    hwi = data[database, "mc_hwi"]
    basis = 3 if original else 6
    hwi["exchanges"].extend(
        exchange(code, amount * basis, bio=bio)
        for code, amount, bio in (
            ("dust_peroxide", 0.0002, False),
            ("dust_ferric", 0.004, False),
            ("mc_nh3_air", 0.00001, True),
            ("mc_nox_air", 0.0002, True),
            ("dust_cyanide", 0.000002, True),
            ("dust_so2", 0.0003, True),
            ("dust_zinc_air", 0.000004, True),
            ("mc_nitrate_short", 0.0001, True),
            ("mc_nitrate_long", 0.0002, True),
            ("dust_sulfate_short", 0.004, True),
            ("dust_sulfate_long", 0.002, True),
        )
    )
    # Oil, CaCl2, generic TMT, cement, support and trace-metal rows are present
    # so tests can check which exchanges the dust calculation removes.
    leaf("dust_calcium_chloride", "calcium chloride")
    hwi["exchanges"].append(exchange("dust_calcium_chloride", 0.03 * basis))
    if not original:
        # Both supplier groups and their total amounts change in the dated input.
        # Original quantities per kilogram and dated supplier shares must survive.
        hwi["exchanges"].append(exchange("dust_second_hcl", 0.06 * basis))
        hwi["exchanges"].append(exchange("dust_second_ti", 0.0003 * basis))
    ash = data[database, "mc_ash"]
    ash_basis = 5 if original else 10
    ash["exchanges"].extend(
        exchange(code, amount * ash_basis, bio=True)
        for code, amount in (
            ("dust_sulfate_short", 0.012),
            ("dust_sulfate_long", 0.024),
            ("dust_iron_short", 0.0005),
            ("dust_iron_long", 0.0025),
        )
    )
    if original:
        ash["exchanges"][0]["properties"].update(
            {"sulfur content": {"amount": 0.04}, "iron content": {"amount": 0.025}}
        )
    return data
