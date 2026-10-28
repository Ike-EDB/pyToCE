"""Keep PBS, PBAT and PTT finished material on their declared product bases.

Wang et al. (2024), DOI 10.1021/acssuschemeng.4c01842, Figure 3(a) and
Table S4 supply the process inputs. The catalog specifies functional-group
balances, reaction water and THF outputs, and the included catalyst/deactivator
reference. These calculations derive dry elemental masses, carbon by origin, and estimated gross
heat of combustion.
PBS uses the separately admitted Rebolledo-Leiva Table SM5 charge and
Ioannidou polymerization/recovery reference, with catalyst inside its product.
PTT uses Van Roijen and Miller's finished-resin functional unit and the
separately admitted Shell catalyst charge, retained as an included equivalent.
"""

from dataclasses import dataclass
from math import fsum
from typing import Any

from .open_burning import _organic_heat_potential


@dataclass(frozen=True, slots=True)
class _PolyesterMaterial:
    """Element masses, carbon and gross heat per kg finished principal polyester.

    Elements and carbon are kg; heat is MJ. The whole product includes the
    specified catalyst/deactivator residue. Each resin calculation separates
    selected monomer/brancher carbon from its fixed fossil inputs.
    """

    elements_kg: dict[str, float]
    selected_carbon_kg: float
    fixed_fossil_carbon_kg: float
    heat_potential_mj: float


def _pbat_material(services: dict[str, Any]) -> _PolyesterMaterial:
    """Derive complete PBAT composition from its declared manufacturing recipe.

    All three origin recipes have the same composition. The biogenic
    declaration supplies these fixed quantities; carbon origin is assigned
    during manufacturing or from the waste entering treatment. Reaction water and THF are separate
    synthesis outputs.
    """
    recipe = services["pbat_biogenic"]
    method = recipe["representative_method"]
    molar = method["molar_masses_g_per_mol"]
    fixed = method["fixed_source_inputs_kg"]
    weights = method["atomic_weights_g_per_mol"]
    atomic = {
        "C": weights["carbon"],
        "H": weights["hydrogen"],
        "N": weights["nitrogen"],
        "O": weights["oxygen"],
    }
    amounts = {
        row["service_id"]: row["amount"]
        for row in recipe["construction"]["components"]
        if "service_id" in row
    }
    outlets = {
        row["source_term"]: row["amount"]
        for row in recipe["construction"]["source_balance_outputs"]
    }
    # Atom counts of the actual charged monomers, brancher and chain extender.
    # All glycerol hydroxyls and both HDI NCO groups are included in the source
    # functional-group reference. THF forms by BDO dehydration; its atoms and
    # those of the declared reaction water are removed once from the product.
    terms = (
        ("bdo", amounts["butanediol_delivered_biogenic"], {"C": 4, "H": 10, "O": 2}),
        (
            "pta",
            amounts["purified_terephthalic_acid_delivered_biogenic"],
            {"C": 8, "H": 6, "O": 4},
        ),
        ("adipic_acid", amounts["adipic_acid_delivered_biogenic"], {"C": 6, "H": 10, "O": 4}),
        ("glycerol", fixed["glycerol"], {"C": 3, "H": 8, "O": 3}),
        ("hdi", fixed["hdi"], {"C": 8, "H": 12, "N": 2, "O": 2}),
        ("water", -outlets["polycondensation reaction water"], {"H": 2, "O": 1}),
        ("tetrahydrofuran", -outlets["tetrahydrofuran purge"], {"C": 4, "H": 8, "O": 1}),
    )
    organic = {
        element: fsum(
            mass * formula.get(element, 0) * weight / molar[name] for name, mass, formula in terms
        )
        for element, weight in atomic.items()
    }
    # The admitted input-equivalent residue uses nominal C16H36O4Ti/340
    # and dry H3PO4/98, inside the existing one-kg finished product. Molecular
    # transformation is unspecified; no TiO2 phase or extra residue is added.
    tbot = {
        element: fixed["titanium_butoxide"] * mass / 340
        for element, mass in {"C": 192, "H": 36, "O": 64, "Ti": 48}.items()
    }
    acid = {
        element: fixed["phosphoric_acid"] * mass / 98
        for element, mass in {"H": 3, "O": 64, "P": 31}.items()
    }
    elements = {
        element: fsum(part.get(element, 0.0) for part in (organic, tbot, acid))
        for element in ("C", "H", "O", "N", "Ti", "P")
    }
    # Preserve the existing carbon arithmetic and its separate HDI origin.
    # Water contains no C. The same net molecular inputs above determine the
    # selected portion, including glycerol and the subtracted THF carbon.
    selected_carbon = atomic["C"] * fsum(
        mass * formula.get("C", 0) / molar[name] for name, mass, formula in terms if name != "hdi"
    )
    fossil_carbon = (
        fixed["hdi"] * 8 * atomic["C"] / molar["hdi"] + fixed["titanium_butoxide"] * 192 / 340
    )
    # Genchel et al. (1973), via NIST Chemistry WebBook: combustion enthalpy -10742 kJ/mol
    # and molar mass 340.322 g/mol give 31.564... MJ/kg input-equivalent residue.
    # This thermochemical reference is separate from the 340 g/mol basis above.
    # Michel (via Muñoz et al. (2007), Eq. 9) heat uses net organic atoms; dry acid contributes no
    # fuel term.
    heat = _organic_heat_potential(organic) + fixed["titanium_butoxide"] * 10742 / 340.322
    return _PolyesterMaterial(elements, selected_carbon, fossil_carbon, heat)


def _pbs_material(services: dict[str, Any]) -> _PolyesterMaterial:
    """Derive the fixed dry PBS product from its admitted normalized recipe.

    Rebolledo-Leiva et al. (2023), supplementary Table SM5, and Ioannidou
    et al. (2022), section 2.1.3.3, constrain the charge and internal BDO
    recovery. The catalog normalizes the backbone and included catalyst to one kg dry product.
    The biogenic declaration has the same quantities as both other
    routes. Manufacturing calculations assign the carbon origins separately.
    """
    method = services["pbs_biogenic"]["representative_method"]
    backbone_mass = method["backbone_normalization_factor"]
    catalyst_mass = method["titanium_butoxide_kg"]
    # C8H12O4 has source molar mass 172.180 g/mol. Its element masses use the
    # same C/H/O convention as the existing PBS carbon and charge calculation.
    backbone = {
        element: backbone_mass * mass / 172.180
        for element, mass in {"C": 96.088, "H": 12.096, "O": 63.996}.items()
    }
    # Included reacted catalyst has the admitted input-equivalent C16H36O4Ti
    # reference, with nominal molar mass 340. Every atom stays inside the
    # existing one-kg dry product.
    catalyst = {
        element: catalyst_mass * mass / 340
        for element, mass in {"C": 192, "H": 36, "O": 64, "Ti": 48}.items()
    }
    elements = {
        element: fsum((backbone.get(element, 0.0), catalyst[element]))
        for element in ("C", "H", "O", "Ti")
    }
    # Apply the organic estimate only to PBS backbone. NIST's liquid TBOT combustion enthalpy 10742
    # kJ/mol / 340.322 g/mol gives MJ/kg residue equivalent; this supplies the transferred catalyst
    # heat estimate.
    heat = _organic_heat_potential(backbone) + catalyst_mass * 10742 / 340.322
    return _PolyesterMaterial(elements, backbone["C"], catalyst["C"], heat)


def _ptt_material(services: dict[str, Any]) -> _PolyesterMaterial:
    """Derive one kg finished PTT with its included catalyst residue.

    PTT represents the OtherBiobased constituent. Van Roijen and Miller (2024)
    define one kg resin; Shell US6326456B2 supplies the catalyst charge,
    assumed fully retained. Purchased monomers and manufacturing outlets
    are separate from this dry product.
    """
    method = services["ptt_source_input_operation"]["representative_method"]
    catalyst_mass = method["titanium_butoxide_kg"]
    # The residue is inside the finished kg. Its complement is nominal PTT C11H10O4/206: C=11*12,
    # H=10*1 and O=4*16 g per mol of repeat units.
    backbone = {
        element: (1 - catalyst_mass) * mass / 206
        for element, mass in {"C": 132, "H": 10, "O": 64}.items()
    }
    # Nominal retained C16H36O4Ti/340 accounts for every catalyst atom, with C12/H1/O16/Ti48.
    catalyst = {
        element: catalyst_mass * mass / 340
        for element, mass in {"C": 192, "H": 36, "O": 64, "Ti": 48}.items()
    }
    elements = {
        element: fsum((backbone.get(element, 0.0), catalyst[element]))
        for element in ("C", "H", "O", "Ti")
    }
    # Michel's (via Doka) organic heat estimate applies to backbone alone. NIST SRD 69 gives liquid
    # TBOT 10742 kJ/mol / 340.322 g/mol = MJ/kg; this qualified heat transfer uses a distinct
    # thermochemical molar mass, and retains the nominal 340 g/mol atom balance.
    heat = _organic_heat_potential(backbone) + catalyst_mass * 10742 / 340.322
    return _PolyesterMaterial(elements, backbone["C"], catalyst["C"], heat)
