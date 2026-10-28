"""Supply PA spinning capital and thermal fluid with internal vent recovery.

The selected control condenses vented fluid and returns it to the heating
circuit. External makeup replaces retired fluid and reported air loss; the
existing spinning utility and factory proxies represent the integrated
operation. Actual retired fluid uses the shared chemical-residue receiver.
"""

import json
from dataclasses import asdict, dataclass
from math import fsum
from typing import cast

from ..chemical_residue_load import ChemicalResidueLoad
from ..hazardous_incineration import reagent_element_masses
from .catalogue import provider_service_code
from .native_recipe import _service_trace
from .recipe_context import _RecipeContext

# CIAAW g/mol, as in the admitted PA ingredient owners. NIST identifies the
# spinning-fluid reference as biphenyl C12H10 and diphenyl ether C12H10O.
_ATOMS = {"C": 12.011, "H": 1.008, "O": 15.999}
_FLUID_FORMULAS = {
    "biphenyl": {"C": 12, "H": 10},
    "diphenyl_ether": {"C": 12, "H": 10, "O": 1},
}


@dataclass(frozen=True, slots=True)
class _PaSpinningSupportReference:
    """External quantities per kg finished principal PA6 or PA66 fiber.

    Fluid quantities use kg and the factory allocation uses units. Retired
    fluid retains its nominal molecular composition, atoms and thermal load;
    upstream allocated supplier burdens are separate from this load.
    """

    pa_grade: str
    factory_units: float
    retired_fluid_kg: float
    residual_air_kg: float
    fresh_fluid_kg: float
    fluid_components_kg: dict[str, float]
    residue: ChemicalResidueLoad


def _pa_spinning_support_reference(
    construction: dict[str, object],
) -> _PaSpinningSupportReference:
    """Calculate nominal external makeup and the actual retired-fluid load.

    Brearley (1952), Figures 7-8, supplies the vent-condensation/internal-return
    reference. At steady state, fresh fluid replaces retired fluid and reported
    air loss; condensed fluid remains within the circuit. Yantai supplies the retirement reference,
    and the BREF supplies the air reference.
    """
    grade = str(construction["pa_grade"])
    # Yantai EIA (2022): 4.025 t/year spent thermal fluid per 21,456 t/year
    # output. Apply this kg/kg reference to PA6 and PA66.
    retired = 4.025 / 21_456
    # BREF 2007 Table 9.4 textile midpoints, g/t -> kg/kg. These residual
    # emissions already belong to the fiber parent's single VOC exchange.
    air = {"pa6": 32.5e-6, "pa66": 30e-6}[grade]
    # Condensed fluid returns to the circuit; only retirement and the reported
    # total air loss require external replacement. No capture fraction or
    # gross volatilization is inferred from the aggregate BREF air quantity.
    # Yantai Table 10.3-4 gives 26.5% biphenyl in its thermal-fluid blend.
    # Brearley uses the same nominal blend.
    compounds = {"biphenyl": 0.265 * retired, "diphenyl_ether": 0.735 * retired}
    molecules = reagent_element_masses(compounds, _FLUID_FORMULAS, _ATOMS)
    fluid_elements = {e: fsum(pool.get(e, 0.0) for pool in molecules.values()) for e in _ATOMS}
    elements = {**fluid_elements, "N": 0.0, "S": 0.0, "Na": 0.0}
    # Michel (via Muñoz et al. (2007), Eq. 9) organic heat factors are MJ/kg element. Its 9 kg
    # water/kg H
    # approximation belongs only to the net calorific value equation.
    energy = fsum(
        fluid_elements[e] * factor
        for e, factor in {"C": 34.016, "H": 124.265, "O": -9.8324}.items()
    )
    # Harvey (1998), NISTIR 5078 Table 1: 2441.7 kJ/kg / 1000 gives the
    # 2.4417 MJ/kg newly vaporized water deduction at the 25 C reference.
    # This nominal retired organic fluid contains no separately assigned water.
    energy -= 2.4417 * 9 * fluid_elements["H"]
    # Actual new water uses the parent's CIAAW molecules: kg H/(2*1.008) * 18.015 gives kg newly
    # formed H2O.
    formed_water = fluid_elements["H"] * (2 * _ATOMS["H"] + _ATOMS["O"]) / (2 * _ATOMS["H"])
    residue = ChemicalResidueLoad(
        wet_mass_kg=retired,
        carrier_water_kg=0.0,
        active_elements_kg=elements,
        net_energy_mj=energy,
        nonfuel_salt_elements_kg={},
        waste_formed_water_kg=formed_water,
    )
    return _PaSpinningSupportReference(
        pa_grade=grade,
        # Native full chemical-factory allocation: 50,000 t/year * 1,000 kg/t
        # * 50 years. This cross-process capital includes final decommissioning.
        factory_units=1 / (50_000 * 1_000 * 50),
        retired_fluid_kg=retired,
        residual_air_kg=air,
        fresh_fluid_kg=retired + air,
        fluid_components_kg=compounds,
        residue=residue,
    )


def _pa_spinning_support_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Bind one kg fiber support to complete dated supplies and signed residue.

    The support reference purchases capital and external fluid makeup once.
    The receiving demand is negative kg for the shared negative
    production waste function. Parent VOC, site operating energy and water
    are already owned by the fiber converter and are not purchased again.
    """
    _service_trace(context, service)
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    selectors = cast(dict[str, dict[str, object]], construction["supplies"])
    reference = _pa_spinning_support_reference(construction)
    key = database_name, provider_service_code(service_id)
    rows: list[dict[str, object]] = [{"input": key, "type": "production", "amount": 1.0}]
    for label, amount in (
        ("factory", reference.factory_units),
        ("fluid", reference.fresh_fluid_kg),
    ):
        node, trace = context.activity(
            selectors[label],
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id=label,
            semantic_class="service_normalization",
        )
        rows.append(
            {
                "input": cast(tuple[str, str], node.key),
                "type": "technosphere",
                "amount": amount,
                "unit": str(node["unit"]),
            }
        )
        trace["project_terms_json"] = json.dumps(
            {"demand_per_finished_principal_kg": amount, "unit": node["unit"]}, sort_keys=True
        )
    receiving = cast(dict[str, object], construction["residue"])
    rows.append(
        {
            "input": (database_name, provider_service_code(str(receiving["service_id"]))),
            "type": "technosphere",
            "amount": -reference.residue.wet_mass_kg,
            "unit": "kilogram",
        }
    )
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "exchanges": rows,
        "pa_spinning_support_reference": asdict(reference),
        "comment": (
            "PA fiber-spinning support per kg finished principal polymer. A vent "
            "condenser returns collected thermal fluid internally, following Brearley. "
            "External makeup combines Yantai spent-fluid/output quantities and BREF "
            "heat-transfer-medium air losses. BREF/Li spinning utilities, water and "
            "the native factory inventory supply cooling, vacuum and return handling. "
            "An aromatic manufacturing proxy supplies the biphenyl/diphenyl-ether "
            "blend; retired fluid receives HWI and ash treatment. Aggregate air "
            "emissions remain with the fiber converter."
        ),
    }
