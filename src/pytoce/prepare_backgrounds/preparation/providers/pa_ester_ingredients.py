"""Supply PA diglycerol dioleate and dimeramide with their actual residues.

Nominal molecular references specify ingredient composition. Native supply
allocation and dated manufacturing functions remain separate from the fixed
charge, reaction and collected-phase quantities.
"""

import json
from dataclasses import asdict, dataclass
from math import fsum
from typing import cast

from .catalogue import provider_service_code
from .native_recipe import _self_production_amount, _service_trace
from .recipe_context import _RecipeContext, _scaled_native_exchange

# Selected nominal formula convention, g/mol of atoms, distinct from the
# integer weights used in native carbon accounts.
_ATOMS = {
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
    "N": 14.007,
    "K": 39.0983,
    "S": 32.06,
    "P": 30.973761998,
    "Li": 6.94,
}
_FORMULAS = {
    "H2O": {"H": 2, "O": 1},
    "CO2": {"C": 1, "O": 2},
    "C3H8O3": {"C": 3, "H": 8, "O": 3},
    "C6H14O5": {"C": 6, "H": 14, "O": 5},
    "C9H20O7": {"C": 9, "H": 20, "O": 7},
    "K2CO3": {"K": 2, "C": 1, "O": 3},
    "H2SO4": {"H": 2, "S": 1, "O": 4},
    "K2SO4": {"K": 2, "S": 1, "O": 4},
    "C18H34O2": {"C": 18, "H": 34, "O": 2},
    "C42H78O7": {"C": 42, "H": 78, "O": 7},
    "Li2CO3": {"Li": 2, "C": 1, "O": 3},
    "LiH2PO4": {"Li": 1, "H": 2, "P": 1, "O": 4},
    "H3PO4": {"H": 3, "P": 1, "O": 4},
    "C36H64O4": {"C": 36, "H": 64, "O": 4},
    "C6H16N2": {"C": 6, "H": 16, "N": 2},
    "C222H396N2O22": {"C": 222, "H": 396, "N": 2, "O": 22},
}
_MASS = {
    species: fsum(_ATOMS[element] * count for element, count in atoms.items())
    for species, atoms in _FORMULAS.items()
}


@dataclass(frozen=True, slots=True)
class _EsterResidue:
    """Known compounds and qualified organic/mineral bodies per delivered kg.

    Compound keys give nominal chemical formulas.
    Diglycerol bottoms use an aggregate organic element balance. Prepared mineral remains separate
    from known salts.
    Carbon origin is independent of both nominal formula and native allocation.
    """

    known_compounds_kg: dict[str, float]
    aggregate_organic_elements_kg: dict[str, float]
    prepared_mineral_kg: float
    fossil_kg_c: float
    biogenic_kg_c: float

    @property
    def wet_kg(self) -> float:
        """Return the complete receiving mass; these selected phases carry no free water."""
        return fsum(
            (
                *self.known_compounds_kg.values(),
                *self.aggregate_organic_elements_kg.values(),
                self.prepared_mineral_kg,
            )
        )


@dataclass(frozen=True, slots=True)
class _PaEsterReference:
    """Fixed manufacture and receiving quantities per kg dry final ingredient.

    Input and residue keys match construction fields. Service quantities are
    kg of their declared operating output or delivered N2.
    Distribution is incoming Li-carbonate mass; its dated market
    supplies freight only. Condensate is additional reaction/process water,
    separate from the native operating function's existing utility effluent.
    """

    native_inputs_kg: dict[str, float]
    service_inputs_kg: dict[str, float]
    condensate_kg: float
    distribution_kg: float
    product_fossil_kg_c: float
    product_biogenic_kg_c: float
    direct_air_kg: dict[str, float]
    residues: dict[str, _EsterResidue]


def _diglycerol_dioleate_reference() -> _PaEsterReference:
    """Join Kao's isolated-polyol and Riken's technical dioleate operations.

    The 3/92/5 feed and approximately 900 g product define the manufacturing
    proxy. Settling quantities follow this operation; the final ingredient
    uses nominal C42H78O7.
    """
    # Kao EP2799129B1 Example 2-4: 600 g reactor glycerol, 230 g internal
    # recycle, 142 g isolated diglycerol, 4.5 g carbonate, 559 g crude mass.
    glycerol = (600 - 230) / 142
    carbonate = 4.5 / 142
    carbonate_moles = carbonate / _MASS["K2CO3"]
    acid = carbonate_moles * _MASS["H2SO4"]
    sulfate = carbonate_moles * _MASS["K2SO4"]
    co2 = carbonate_moles * _MASS["CO2"]
    dehydration_water = (600 + 4.5 - 559) / 142
    condensate = dehydration_water + carbonate_moles * _MASS["H2O"]
    # Known salt/CO2 neutralization cancels separately. Nominal glycerol
    # minus isolated diglycerol and dehydration water determines aggregate
    # organic CHO.
    organic_elements = {
        element: _ATOMS[element]
        * (
            glycerol * _FORMULAS["C3H8O3"][element] / _MASS["C3H8O3"]
            - _FORMULAS["C6H14O5"][element] / _MASS["C6H14O5"]
            - dehydration_water * _FORMULAS["H2O"].get(element, 0) / _MASS["H2O"]
        )
        for element in ("C", "H", "O")
    }
    # Riken JP2024079371A Example 2 uses 269 g technical polyol and 731 g
    # oleic acid for about 900 g technical product, with complete nominal
    # acid conversion chosen for the water/settling receiving reference.
    polyol = 269 / 900
    fatty_acid = 731 / 900
    reaction_water = fatty_acid / _MASS["C18H34O2"] * _MASS["H2O"]
    settling = polyol + fatty_acid - 1 - reaction_water
    # Apply the source feed's nominal 3/92/5 grade to the removed lower phase.
    settled_compounds = {
        species: share * settling
        for species, share in (("C3H8O3", 0.03), ("C6H14O5", 0.92), ("C9H20O7", 0.05))
    }
    settled_carbon = fsum(
        amount * _FORMULAS[species]["C"] * _ATOMS["C"] / _MASS[species]
        for species, amount in settled_compounds.items()
    )
    return _PaEsterReference(
        native_inputs_kg={
            "glycerol": polyol * glycerol,
            "fatty_acid": fatty_acid,
            "potassium_carbonate": polyol * carbonate,
            "sulfuric_acid": polyol * acid,
        },
        # Riken uses nitrogen. Transfer the 2011-2015 Gendorf intensity, via ecoinvent
        # 3.12 sodium hydrogen sulfate production, of 0.019 kg N2/kg output once to this final ester
        # operation.
        service_inputs_kg={"ester_operation": 1 + polyol, "purge": 0.019},
        condensate_kg=polyol * condensate + reaction_water,
        distribution_kg=0.0,
        product_fossil_kg_c=0.0,
        product_biogenic_kg_c=42 * _ATOMS["C"] / _MASS["C42H78O7"],
        direct_air_kg={"Carbon dioxide, fossil": polyol * co2},
        residues={
            "diglycerol_residue": _EsterResidue(
                known_compounds_kg={"K2SO4": polyol * sulfate},
                aggregate_organic_elements_kg={
                    element: polyol * amount for element, amount in organic_elements.items()
                },
                prepared_mineral_kg=0.0,
                fossil_kg_c=0.0,
                biogenic_kg_c=polyol * organic_elements["C"],
            ),
            "settling_residue": _EsterResidue(
                known_compounds_kg=settled_compounds,
                aggregate_organic_elements_kg={},
                prepared_mineral_kg=0.0,
                fossil_kg_c=0.0,
                biogenic_kg_c=settled_carbon,
            ),
        },
    )


def _hmda_dimeramide_reference() -> _PaEsterReference:
    """Supply the technical two-stage dimer function and nominal HMDA condensation.

    JP2005002085A Example 3's two stages share an internal monomer stream;
    their combined output includes retained technical trimer. The final C222
    component separately represents six nominal diacids plus one HMDA, losing
    two waters. Four diacid molecules remain unreacted and ten carboxyl groups
    remain across the complete batch. No pure-dimer yield is inferred from
    the technical manufacture function.
    """
    dimer = 6 * _MASS["C36H64O4"] / _MASS["C222H396N2O22"]
    hmda = _MASS["C6H16N2"] / _MASS["C222H396N2O22"]
    amide_water = 2 * _MASS["H2O"] / _MASS["C222H396N2O22"]
    # Source Figure 1: 1000 g fresh acid -> 535+138 g combined technical
    # output and 327 g final monomer. The 465 g second-stage feed is internal.
    output = 535 + 138
    fatty_acid = 1000 / output
    monomer = 327 / output
    mineral = (50 + 23.2) / output
    lithium_carbonate = (2 + 0.9) / output
    carbonate_moles = lithium_carbonate / _MASS["Li2CO3"]
    # The source adds 5+7 g 75% H3PO4. Its selected native active-only
    # 85%-state market delivers no solvent; all 25% carrier is purchased.
    phosphoric_acid = (5 + 7) * 0.75 / output
    water = (20 + 9.3 + (5 + 7) * 0.25) / output
    lithium_phosphate = 2 * carbonate_moles * _MASS["LiH2PO4"]
    excess_acid = phosphoric_acid - 2 * carbonate_moles * _MASS["H3PO4"]
    # Separate the dry final phase from collected condensate.
    condensate = water + carbonate_moles * _MASS["H2O"]
    monomer_carbon = monomer * 18 * _ATOMS["C"] / _MASS["C18H34O2"]
    return _PaEsterReference(
        native_inputs_kg={
            "fatty_acid": dimer * fatty_acid,
            "prepared_clay": dimer * mineral,
            "lithium_carbonate": dimer * lithium_carbonate,
            "phosphoric_acid": dimer * phosphoric_acid,
            "water": dimer * water,
            "hmda": hmda,
        },
        service_inputs_kg={
            "ester_operation": dimer,
            "amide_operation": 1.0,
            # JP2005002085A uses nitrogen. Apply the 2011-2015 Gendorf intensity, via
            # ecoinvent 3.12 sodium hydrogen sulfate production: 0.019 kg N2/kg combined technical
            # dimer output once.
            "purge": 0.019 * dimer,
        },
        condensate_kg=dimer * condensate + amide_water,
        distribution_kg=dimer * lithium_carbonate,
        product_fossil_kg_c=6 * _ATOMS["C"] / _MASS["C222H396N2O22"],
        product_biogenic_kg_c=216 * _ATOMS["C"] / _MASS["C222H396N2O22"],
        direct_air_kg={"Carbon dioxide, fossil": dimer * carbonate_moles * _MASS["CO2"]},
        residues={
            "monomer_residue": _EsterResidue(
                known_compounds_kg={"C18H34O2": dimer * monomer},
                aggregate_organic_elements_kg={},
                prepared_mineral_kg=0.0,
                fossil_kg_c=0.0,
                biogenic_kg_c=dimer * monomer_carbon,
            ),
            "mineral_residue": _EsterResidue(
                known_compounds_kg={
                    "LiH2PO4": dimer * lithium_phosphate,
                    "H3PO4": dimer * excess_acid,
                },
                aggregate_organic_elements_kg={},
                prepared_mineral_kg=dimer * mineral,
                fossil_kg_c=0.0,
                biogenic_kg_c=0.0,
            ),
        },
    )


def _pa_ester_reference(grade: str) -> _PaEsterReference:
    """Select the ingredient reference shared by manufacture and residue treatment."""
    return {
        "diglycerol_dioleate": _diglycerol_dioleate_reference,
        "hmda_dimeramide": _hmda_dimeramide_reference,
    }[grade]()


def _ester_distribution(
    context: _RecipeContext,
    selector: dict[str, object],
    *,
    service_id: str,
    amount_kg: float,
) -> list[dict[str, object]]:
    """Retain the incoming alcohol/Li-carbonate market's dated freight only.

    The actual selected chemical producer replaces the market's entire supply
    mix. Normalize its complete transport function by its production reference
    once, preserving each exchange's uncertainty and allocation metadata.
    """
    source, trace = context.activity(
        selector,
        source_role="native_ecoinvent_provider",
        service_id=service_id,
        component_id="incoming_chemical_distribution",
        semantic_class="source_backed_portfolio",
    )
    native = context.exchanges(source)
    production = _self_production_amount(source, native, method="PA incoming chemical market")
    transport = [
        exchange
        for exchange in native
        if exchange["type"] == "technosphere"
        and str(context.input(exchange)["reference product"]).startswith("transport,")
    ]
    trace["retained_technosphere_json"] = context.exchange_json(transport)
    trace["removed_technosphere_json"] = context.exchange_json(
        exchange
        for exchange in native
        if exchange["type"] == "technosphere" and exchange not in transport
    )
    trace["removed_biosphere_json"] = context.exchange_json(
        exchange for exchange in native if exchange["type"] == "biosphere"
    )
    trace["retained_biosphere_json"] = "[]"
    trace["project_terms_json"] = json.dumps(
        {"incoming_chemical_kg": amount_kg, "native_market_reference_kg": production},
        sort_keys=True,
    )
    return [_scaled_native_exchange(exchange, amount_kg / production) for exchange in transport]


def _pa_ester_dataset(
    context: _RecipeContext, service: dict[str, object], database_name: str
) -> dict[str, object]:
    """Compile one kg dry PA ingredient with its complete dated supply functions.

    The two admitted charge/reaction references supply their distinct
    operating quantities, nitrogen supply, condensate and residue treatment. Whole native
    chemical activities preserve upstream allocation; the replaced
    Li-carbonate supply mix contributes freight separately. Internal recovered glycerol and monomer
    return to their consuming processes.
    """
    _service_trace(context, service)
    service_id = str(service["provider_service_id"])
    construction = cast(dict[str, object], service["construction"])
    reference = _pa_ester_reference(str(construction["grade"]))
    key = database_name, provider_service_code(service_id)
    carbon = reference.product_fossil_kg_c + reference.product_biogenic_kg_c
    production_properties = {
        "wet mass": {"amount": 1.0, "unit": "kg"},
        "dry mass": {"amount": 1.0, "unit": "kg"},
        "water in wet mass": {"amount": 0.0, "unit": "kg"},
        "carbon allocation": {"amount": carbon, "unit": "kg"},
        "carbon content": {"amount": carbon, "unit": "dimensionless"},
        "carbon content, fossil": {
            "amount": reference.product_fossil_kg_c,
            "unit": "dimensionless",
        },
        "carbon content, non-fossil": {
            "amount": reference.product_biogenic_kg_c,
            "unit": "dimensionless",
        },
    }
    rows: list[dict[str, object]] = [
        {"input": key, "type": "production", "amount": 1.0, "properties": production_properties}
    ]
    sources = cast(dict[str, dict[str, object]], construction["sources"])
    for field, amount in reference.native_inputs_kg.items():
        source, trace = context.activity(
            sources[field],
            source_role="native_ecoinvent_provider",
            service_id=service_id,
            component_id=field,
            semantic_class="unchanged_same_product",
        )
        trace["project_terms_json"] = json.dumps({"supplied_kg": amount}, sort_keys=True)
        # Brightway normalizes the complete supplier through its native production reference.
        rows.append({"input": source.key, "type": "technosphere", "amount": amount})
    if reference.distribution_kg:
        rows.extend(
            _ester_distribution(
                context,
                cast(dict[str, object], construction["freight_source"]),
                service_id=service_id,
                amount_kg=reference.distribution_kg,
            )
        )
    # Water uses the accepted nominal 1000 kg/m3 conversion. Other negative
    # demands carry each receiver's complete wet kg body, with no source-waste
    # dilution or donor carbon substituted for the actual named residue.
    service_demands = {
        **reference.service_inputs_kg,
        "condensate": -reference.condensate_kg / 1000,
        **{field: -residue.wet_kg for field, residue in reference.residues.items()},
    }
    for field, amount in service_demands.items():
        receiving = str(cast(dict[str, object], construction[field])["service_id"])
        rows.append(
            {
                "input": (database_name, provider_service_code(receiving)),
                "type": "technosphere",
                "amount": amount,
            }
        )
    for name, amount in reference.direct_air_kg.items():
        # The selected native fossil-CO2 flow uses the one-level air compartment. Use that declared
        # compartment.
        flow = context.biosphere.flow({"name": name, "unit": "kilogram", "categories": ["air"]})
        rows.append({"input": flow.key, "type": "biosphere", "amount": amount})
    physical = {
        "schema": "pa_ester_reference_schema_v1",
        "grade": str(construction["grade"]),
        **asdict(reference),
        "residue_wet_kg": {field: residue.wet_kg for field, residue in reference.residues.items()},
        "technical_manufacturing_function_is_not_nominal_molecular_assay": True,
    }
    return {
        "name": str(service["name"]),
        "reference product": str(service["reference_product"]),
        "unit": "kilogram",
        "location": str(service["location"]),
        "type": "process",
        "provider_service_id": service_id,
        "provider_stage_owner": str(service["stage_owner"]),
        "pa_ester_reference_json": json.dumps(physical, sort_keys=True, separators=(",", ":")),
        "exchanges": rows,
    }
