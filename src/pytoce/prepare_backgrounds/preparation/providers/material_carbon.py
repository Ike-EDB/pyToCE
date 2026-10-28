"""Derive constituent carbon from the declared polymer basis and native recipes.

Rows describe modeled constituent composition and separate virgin feedstock
route from ultimate carbon origin. Native manufacturing exchanges and
characterization remain unchanged.
The baseline ecoinvent properties are read once during shared preparation; no
prospective anchor changes the composition of a previously produced cohort.
"""

import json
from math import fsum, isfinite
from typing import Any, cast

import numpy as np

from pytoce.run_mfa.model.materials import load_material_composition

from .catalogue import (
    _catalogue_identity,
    _method_authority,
    _service_source_closures,
    _source_authority_closure,
)
from .polyester_material import _pbat_material, _pbs_material, _ptt_material

CARBON_ORIGINS = ("fossil", "biogenic", "atmospheric_dac")
# Shares of one emitted species' carbon attributable to principal
# material, in CARBON_ORIGINS order. Auxiliary carbon occupies the remainder;
# biogenic and DAC entries may share the same native non-fossil elementary flow.
PrincipalCarbonOriginFractions = tuple[float, float, float]
MATERIAL_CARBON_SCHEMA = "pytoce_material_carbon_schema_v1"
MATERIAL_CARBON_COLUMNS = (
    "material",
    "constituent",
    "virgin_feedstock_route",
    "fossil_carbon",
    "biogenic_carbon",
    "atmospheric_dac_carbon",
    "unit",
    "basis",
    "source_inputs_json",
)

# GY 250 in EP1838750B1, Table 6: an unmodified DGEBA homologous-series
# reference with EEW 186 g/eq, rather than pure n=0 DGEBA (EEW 170 g/eq).
# Each molecule has two epoxide equivalents, so its molar mass is 2*EEW.
# With nominal C12/H1/O16, the initial unit is 340 g/mol with 252 g C/mol;
# each homolog increment adds 284 g/mol, including 216 g C/mol.
_EPOXY_EEW_G_PER_EQ = 186.0
_EPOXY_MEAN_REPEAT_UNITS = (2 * _EPOXY_EEW_G_PER_EQ - 340) / 284
_EPOXY_PRINCIPAL_CARBON = (252 + 216 * _EPOXY_MEAN_REPEAT_UNITS) / (
    340 + 284 * _EPOXY_MEAN_REPEAT_UNITS
)

# CIAAW (2024): C12.011/H1.008/O15.999 for glucan C6H10O5 and glycerol C3H8O3.
# Elemental grams per mole divided by formula g/mol gives kg element/kg
# principal material, identically in TPS product and residue accounting.
# Water, protein and retained reagents are separate constituents.
_TPS_PRINCIPAL_ELEMENT_FRACTIONS = {
    "glucan": {"C": 72.066 / 162.141, "H": 10.08 / 162.141, "O": 79.995 / 162.141},
    "glycerol": {"C": 36.033 / 92.094, "H": 8.064 / 92.094, "O": 47.997 / 92.094},
}


def _property_carbon(exchange: Any) -> float:
    """Read kg carbon per kg product from actual exchange properties.

    EcoSpold carbon content is a dry-mass fraction; dry mass is kg per product
    unit. These exchange-specific properties override generic activity metadata.
    """
    properties = exchange.get("properties", {})
    carbon = properties.get("carbon content", {})
    dry = properties.get("dry mass", {})
    if carbon.get("unit") != "dimensionless" or dry.get("unit") != "kg":
        raise ValueError(
            f"Native exchange {exchange.get('name')!r} requires carbon content as a "
            "dry-mass fraction and dry mass in kg. Use the documented ecoinvent 3.12 source."
        )
    c, mass = float(carbon["amount"]), float(dry["amount"])
    if not isfinite(c) or not isfinite(mass) or not 0 < c <= 1 or not 0 < mass <= 1:
        raise ValueError(
            f"Native exchange {exchange.get('name')!r} has invalid carbon content {c!r} "
            f"or dry mass {mass!r}; require fractions in (0, 1]. "
            "Use the documented ecoinvent 3.12 source."
        )
    return c * mass


def _native_inputs(source_database: str, services: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read the six fixed native formulation owners."""
    import bw2data as bd  # type: ignore[import-untyped]

    definitions = {
        "long": (
            "polyether_polyols_long_chain_biogenic",
            {
                "eo": "ethylene oxide",
                "po": "propylene oxide, liquid",
                "glycerine": "glycerine",
            },
        ),
        "short": (
            "polyether_polyols_short_chain_biogenic",
            {
                "eo": "ethylene oxide",
                "po": "propylene oxide, liquid",
                "glycerine": "glycerine",
                "sorbitol_proxy": "glucose",
                "sucrose": "sugar, from sugarcane",
            },
        ),
        "flexible": (
            "pur_flexible_biogenic",
            {
                "polyol": "polyether polyols, long chain",
                "tdi": "toluene diisocyanate",
                "san": "styrene-acrylonitrile copolymer",
                "scrap": "waste polyurethane",
            },
        ),
        "rigid": (
            "pur_rigid_biogenic",
            {
                "polyol": "polyether polyols, short chain",
                "mdi": "methylene diphenyl diisocyanate",
                "pentane": "pentane",
                "scrap": "waste polyurethane",
            },
        ),
        "adhesive": (
            "pur_adhesive_biogenic",
            {
                "polyol": "polyether polyols, long chain",
                "mdi": "methylene diphenyl diisocyanate",
                "additive": "chemical, organic, basic precursors and materials",
            },
        ),
        "upr": ("unsaturated_polyester_resin_biogenic", {}),
    }
    result: dict[str, dict[str, Any]] = {}
    for role, (service_id, ingredients) in definitions.items():
        selector = services[service_id]["construction"]["base"]["selector"]
        # These source-defined baseline locations establish composition.
        node = bd.get_node(
            database=source_database,
            name=selector["name"],
            product=selector["reference_product"],
            location=selector["exact_location"],
            unit="kilogram",
        )
        production = node.rp_exchange()
        basis = float(production["amount"])
        if production["unit"] != "kilogram" or basis <= 0:
            raise ValueError(
                f"Native carbon source {node.key!r} requires positive production in kilograms; "
                "use the documented ecoinvent 3.12 source."
            )
        rows: dict[str, list[Any]] = {}
        for exchange in node.technosphere():
            rows.setdefault(str(exchange.input["reference product"]), []).append(exchange)
        values: dict[str, Any] = {
            "source": {
                field: node[field] for field in ("name", "reference product", "location", "code")
            },
            "production_kg": basis,
        }
        if role in {"long", "short", "upr"}:
            values["carbon"] = _property_carbon(production)
        for field, product in ingredients.items():
            selected = rows.get(product, [])
            direction = -1 if field == "scrap" else 1
            if not selected or any(
                exchange["unit"] != "kilogram" or direction * float(exchange["amount"]) < 0
                for exchange in selected
            ):
                raise ValueError(
                    f"Native carbon source {node.key!r} lacks a valid {product!r} kilogram "
                    "exchange. Use the documented ecoinvent 3.12 source."
                )
            # All regional suppliers belong to the ingredient; the adhesive has two MDI and two
            # polyol exchanges.
            values[field] = direction * fsum(float(row["amount"]) for row in selected) / basis
            if field == "additive":
                values["additive_carbon"] = (
                    fsum(float(row["amount"]) * _property_carbon(row) for row in selected) / basis
                )
        if role in {"flexible", "rigid"}:
            name = "Carbon dioxide, fossil" if role == "flexible" else "Pentane"
            emissions = [
                exchange
                for exchange in node.biosphere()
                if exchange.input["name"] == name
                and tuple(exchange.input["categories"]) == ("air", "urban air close to ground")
            ]
            if not emissions or any(
                exchange["unit"] != "kilogram" or float(exchange["amount"]) < 0
                for exchange in emissions
            ):
                raise ValueError(
                    f"Native carbon source {node.key!r} requires its {name!r} release in kg "
                    "to urban air. Use the documented ecoinvent 3.12 source."
                )
            values["release"] = fsum(float(row["amount"]) for row in emissions) / basis
        if role == "long" and fsum(values[field] for field in ("eo", "po", "glycerine")) == 0:
            raise ValueError(
                f"Native carbon source {node.key!r} has no identified carbon-forming precursor. "
                "Require positive precursor input; use the documented ecoinvent 3.12 source."
            )
        if role == "rigid" and values["release"] > values["pentane"]:
            raise ValueError(
                f"Native carbon source {node.key!r} releases more pentane than its input. "
                "Require release <= input; use the documented ecoinvent 3.12 source."
            )
        if role == "flexible" and values["release"] * 12 / 44 > values["tdi"] * 108 / 174:
            raise ValueError(
                f"Native carbon source {node.key!r} releases more isocyanate-derived carbon "
                "than its TDI input. Require carbon release <= input; "
                "use the documented ecoinvent 3.12 source."
            )
        result[role] = values
    return result


def _admit_product_carbon(carbon: np.ndarray, product: str) -> None:
    """Reject invalid product carbon before aggregation or completed publication."""
    if (
        not np.isfinite(carbon).all()
        or np.any(carbon < 0)
        or np.any(carbon.sum(axis=1) <= 0)
        or np.any(carbon.sum(axis=1) > 1)
    ):
        raise ValueError(
            f"{product} yields invalid retained carbon {carbon.tolist()!r}. "
            "Require finite, nonnegative origins totalling more than 0 and at most "
            "1 kg C/kg product. "
            "Check the native formulation against the documented constituent basis."
        )


def _polyurethane_carbon(
    native: dict[str, dict[str, Any]], glucose: np.ndarray, services: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate grade carbon over identified precursors, then retain finished-formulation carbon.

    Axes are virgin route and ultimate origin. Formula fractions use C12, H1,
    O16 and N14; grade carbon retains native precision. Equal precursor retention,
    homogeneous foam scrap, a sorbitol manufacture proxy and 98% adhesive yield
    are the adopted study assumptions.
    """
    selected = np.eye(3)
    fossil, bio = selected[0], selected[1]
    long, short = native["long"], native["short"]
    glycerine = selected.copy()
    glycerine[0] = bio
    long_weights = (long["eo"] * 24 / 44 + long["po"] * 36 / 58) * selected + long[
        "glycerine"
    ] * 36 / 92 * glycerine
    long_carbon = long["carbon"] * long_weights / long_weights.sum(axis=1)[:, None]
    sorbitol = glucose.copy()
    sorbitol[0] = bio
    # Native comments specify actual TDA/MDA quantities. The source comments define the TDA/MDA
    # dose.
    short_weights = (
        (short["eo"] * 24 / 44 + short["po"] * 36 / 58) * selected
        + short["glycerine"] * 36 / 92 * glycerine
        + short["sorbitol_proxy"] * 72 / 182 * sorbitol
        + short["sucrose"] * 144 / 342 * bio
        + (0.00494 * 84 / 122 + 0.0039 * 156 / 198) * fossil
    )
    short_carbon = short["carbon"] * short_weights / short_weights.sum(axis=1)[:, None]
    flex, rigid, adhesive = native["flexible"], native["rigid"], native["adhesive"]
    san = services["san_biogenic"]["representative_method"]["incorporated_product_kg"]
    san_carbon = san["styrene"] * 96 / 104 + san["acrylonitrile"] * 36 / 53
    # Unknown auxiliary retention is excluded from the principal account;
    # native LCI remains complete. Isocyanate-derived CO2 is removed once.
    flexible_carbon = (
        flex["polyol"] * long_carbon
        + (flex["tdi"] * 108 / 174 + flex["san"] * san_carbon - flex["release"] * 12 / 44)
        * selected
    ) / (1 + flex["scrap"])
    # MDI C15H10N2O2 contains 180/250 = 0.72 kg C/kg with the adopted
    # nominal atomic masses, used for both rigid foam and adhesive below.
    rigid_carbon = (
        rigid["polyol"] * short_carbon
        + rigid["mdi"] * 0.72 * selected
        + (rigid["pentane"] - rigid["release"]) * 60 / 72 * fossil
    ) / (1 + rigid["scrap"])
    adhesive_carbon = 0.98 * (
        adhesive["polyol"] * long_carbon
        + adhesive["mdi"] * 0.72 * selected
        + adhesive["additive_carbon"] * fossil
    )
    _admit_product_carbon(flexible_carbon, "Native flexible foam")
    _admit_product_carbon(rigid_carbon, "Native rigid foam")
    _admit_product_carbon(adhesive_carbon, "Native polyurethane adhesive")
    weights = services["pur_biogenic"]["representative_method"]["portfolio_weights"]
    return (
        weights["flexible polyurethane foam"] * flexible_carbon
        + weights["rigid polyurethane foam"] * rigid_carbon
        + weights["complete polyurethane adhesive analogue"] * adhesive_carbon,
        adhesive_carbon,
    )


def _carbon_provenance(
    catalogue: dict[str, object], services: dict[str, Any], native: dict[str, dict[str, Any]]
) -> dict[str, dict[str, object]]:
    """Bind each constituent basis to its used recipe records and explicit assumptions."""
    basis_services = {
        "abs_asa_san": ("abs_asa_san_biogenic", "abs_biogenic", "san_biogenic"),
        "bio_pa": ("polyamide_6_resin_biogenic",),
        "bio_pe": ("hdpe_polymerization_operation",),
        "bio_pet": ("pet_biogenic",),
        "other_bio_based": ("ptt_biogenic", "ptt_source_input_operation", "starch_fossil"),
        "other_biodegradable": ("tps_fossil", "tps_biogenic", "starch_fossil"),
        "pbat": ("pbat_biogenic",),
        "pbs": ("pbs_biogenic",),
        "pha": ("phb_biogenic", "starch_fossil"),
        "pla": ("pla_biogenic", "starch_fossil"),
        "starch_blends": ("tps_fossil", "tps_biogenic", "starch_fossil"),
        "polypropylene": ("pp_polymerization_operation",),
        "polyethylene_terephthalate": ("pet_amorphous_biogenic",),
        "polyamide_6_resin": ("polyamide_6_resin_biogenic",),
        "polyamide_66_resin": ("polyamide_66_resin_biogenic",),
        "elastomers": ("tyre_sbr_biogenic",),
        "hdpe": ("hdpe_polymerization_operation",),
        "pet": ("pet_biogenic",),
        "pp": ("pp_polymerization_operation",),
        "ps": ("ps_biogenic",),
        "pur": ("pur_biogenic", "san_biogenic", "starch_fossil"),
        "ldpe": ("ldpe_polymerization_operation",),
        "pvc": ("pvc_biogenic",),
        "adhesives": ("pur_adhesive_biogenic",),
        "general_coatings": ("epoxy_resin_biogenic",),
        "marine_coatings": ("epoxy_resin_biogenic",),
        "non_tyre_elastomers": ("emulsion_sbr_biogenic",),
        "other_thermoplastics": ("pmma_biogenic",),
        "pa": ("polyamide_6_resin_biogenic",),
        "pc": ("polycarbonate_biogenic",),
        "road_marking_coatings": ("pmma_biogenic",),
        "sealants": ("pur_adhesive_biogenic",),
        "thermosets": ("unsaturated_polyester_resin_biogenic",),
    }
    basis = {
        "abs_asa_san": "ABS/SAN incorporated monomer masses; C12.011/H1.008/N14.007 atomic masses",
        "bio_pa": "C6H11NO dry PA6 backbone; C72/113",
        "bio_pe": "PE backbone; C6/7; manufacturing auxiliaries excluded from polymer",
        "bio_pet": "C10H8O4 PET backbone; C120/192; commercial grade is a disclosed LCI proxy",
        ("other_bio_based"): (
            "One kg finished PTT: (1-r)*C132/206 backbone with PTA8/11 and PDO3/11 "
            "of carbon and glucose primer retained; r*C192/340 fixed fossil TBOT "
            "equivalent inside product, where r is the admitted catalyst charge"
        ),
        ("other_biodegradable"): (
            "TPS product glucan C6H10O5 and glycerol C3H8O3; C12.011/H1.008/O15.999; "
            "co-delivered water/protein/solutes excluded"
        ),
        ("pbat"): (
            "Source BDO/PTA/adipic/glycerol carbon minus THF; HDI fossil; TBOT "
            "nominal retained ligand C192/340 inside the finished product"
        ),
        ("pbs"): (
            "Source backbone-normalization factor times C96.088/172.180; fossil "
            "nominal retained TBOT ligand C192/340 inside the finished product"
        ),
        "pha": "Source PHB carbon0.558067; synthetic glucose primer retained for fossil/DAC routes",
        "pla": "PLA backbone C0.5; glucose primer retained only for DAC route",
        ("starch_blends"): (
            "Same TPS composition as other biodegradable class; distinct coordinate"
        ),
        ("polypropylene"): ("PP backbone C6/7; manufacturing auxiliaries excluded from polymer"),
        ("polyethylene_terephthalate"): (
            "C10H8O4 PET backbone C120/192; amorphous native grade LCI proxy"
        ),
        "polyamide_6_resin": "C6H11NO PA6 backbone C72/113",
        "polyamide_66_resin": "C12H22N2O2 PA66 backbone C144/226",
        ("elastomers"): (
            "Dry SBR equivalent: styrene0.25/butadiene0.75, C35/39; inferred "
            "representative composition"
        ),
        "hdpe": "PE backbone C6/7; manufacturing auxiliaries excluded from polymer",
        "pet": "C10H8O4 PET backbone C120/192; commercial grade LCI proxy",
        "pp": "PP backbone C6/7; manufacturing auxiliaries excluded from polymer",
        ("ps"): (
            "Finished GPPS/HIPS portfolio: nominal PS C96/104 and PBR C48/54 "
            "share the selected origin; included mineral oil C6/7 remains fossil. "
            "All three components belong to the tracked commercial resin kg."
        ),
        ("pur"): (
            "Grade carbon allocated over identified precursors; homogeneous foam "
            "scrap; retained pentane; adhesive98%yield; unidentified auxiliary "
            "retention excluded"
        ),
        "ldpe": "PE backbone C6/7; manufacturing auxiliaries excluded from polymer",
        "pvc": "C2H3Cl backbone C24/62.5 using nominal Cl35.5; commercial grade LCI proxy",
        "adhesives": "Native PUR adhesive; identified retained precursors and 98% product yield",
        "general_coatings": (
            "DGEBA homologous-series binder at EEW186 g/eq; "
            "structural reference; "
            "MXDA and application auxiliaries separate"
        ),
        "marine_coatings": (
            "DGEBA homologous-series binder at EEW186 g/eq; "
            "structural reference; "
            "MXDA and application auxiliaries separate"
        ),
        "non_tyre_elastomers": "Grade1502 dry SBR: 0.235*C96/104+0.765*C48/54",
        "other_thermoplastics": "PMMA backbone C60/100",
        "pa": "C6H11NO PA6 backbone C72/113",
        "pc": "Polycarbonate backbone C192/254",
        "road_marking_coatings": "PMMA binder C60/100; application auxiliaries are separate",
        "sealants": "Native PUR adhesive representative; same retained-product basis as adhesives",
        "thermosets": "Native delivered UPR resin carbon; formulation additions are separate",
    }
    identity = _catalogue_identity(catalogue)
    parameters = {
        "abs_asa_san": {
            "portfolio_weights": services["abs_asa_san_biogenic"]["portfolio_weights"],
            "incorporated_product_kg": {
                kind: services[f"{kind}_biogenic"]["representative_method"][
                    "incorporated_product_kg"
                ]
                for kind in ("abs", "san")
            },
        },
        "ps": {
            key: services["ps_biogenic"]["representative_method"][key]
            for key in ("provider_allocation", "finished_material_reference")
        },
        "pbat": {
            key: services["pbat_biogenic"]["representative_method"][key]
            for key in (
                "atomic_weights_g_per_mol",
                "molar_masses_g_per_mol",
                "fixed_source_inputs_kg",
            )
        },
        "pbs": {
            key: services["pbs_biogenic"]["representative_method"][key]
            for key in ("backbone_normalization_factor", "titanium_butoxide_kg")
        },
        "other_bio_based": {
            key: services["ptt_source_input_operation"]["representative_method"][key]
            for key in ("titanium_butoxide_kg", "finished_material_reference")
        },
        "pur": services["pur_biogenic"]["representative_method"]["portfolio_weights"],
    }
    for constituent in ("general_coatings", "marine_coatings"):
        parameters[constituent] = {
            "epoxy_equivalent_weight_g_per_eq": _EPOXY_EEW_G_PER_EQ,
            "mean_repeat_units": _EPOXY_MEAN_REPEAT_UNITS,
            "principal_carbon_kg_per_kg": _EPOXY_PRINCIPAL_CARBON,
            "source": "EP1838750B1 Table 6 comparative example 3; GY 250",
        }
    # This kg field records selected monomers and the glycerol brancher only. Fixed fossil TBOT
    # already has its own input record. The recipe also references water treatment in m3.
    parameters["pbat"]["selected_monomer_inputs_kg"] = {
        row["service_id"]: row["amount"]
        for row in services["pbat_biogenic"]["construction"]["components"]
        if row.get("service_id")
        in {
            "butanediol_delivered_biogenic",
            "purified_terephthalic_acid_delivered_biogenic",
            "adipic_acid_delivered_biogenic",
            "glycerine_delivered_biogenic",
        }
    }
    parameters["pbat"]["purge_kg"] = {
        row["source_term"]: row["amount"]
        for row in services["pbat_biogenic"]["construction"]["source_balance_outputs"]
    }
    result: dict[str, dict[str, object]] = {}
    for constituent, service_ids in basis_services.items():
        record: dict[str, object] = {
            "provider_authority_identity": identity,
            "basis": basis[constituent],
            ("formula_convention"): (
                "C12/H1/O16/N14 unless the basis states otherwise; source-reported "
                "coefficients retain their precision"
            ),
            "calculation_reference_services": list(service_ids),
            "parameters": parameters.get(constituent, {}),
        }
        if constituent == "pur":
            record["native_formulations"] = {
                key: value for key, value in native.items() if key != "upr"
            }
            record["source_comment_precursors_kg"] = {"TDA": 0.00494, "MDA": 0.0039}
            record["precursor_convention"] = (
                "Equal-mass glucose manufacture proxy represents sorbitol "
                "C72/182; cane sucrose remains biogenic C144/342; TDA/MDA remain "
                "fossil. Carbon is allocated uniformly across identified precursors."
            )
        elif constituent in {"adhesives", "sealants"}:
            record["native_formulations"] = {key: native[key] for key in ("long", "adhesive")}
            record["adhesive_yield"] = 0.98
        elif constituent == "thermosets":
            record["native_formulations"] = {"upr": native["upr"]}
        result[constituent] = record
    return result


def prepare_material_carbon(
    source_database: str, catalogue: dict[str, object]
) -> tuple[dict[str, object], ...]:
    """Create the complete constituent-by-route carbon table from the native baseline.

    Each row gives three ultimate-origin quantities in kg C per kg
    modeled constituent, including explicit zeros. Native source values and
    the calculation basis accompany the row and participate in table identity.
    Manufacturing providers use the virgin route. Downstream treatment demand
    uses these actual carbon origins at its own allocation boundary.
    """
    services = {
        row["provider_service_id"]: row for row in cast(list[dict[str, Any]], catalogue["services"])
    }
    native = _native_inputs(source_database, services)
    selected = np.eye(3)
    primer = services["starch_fossil"]["representative_method"][
        "biogenic_primer_share_of_dry_amylose"
    ]
    glucose = (1 - primer) * selected
    glucose[:, 1] += primer
    pur, adhesive = _polyurethane_carbon(native, glucose, services)
    pbat_material = _pbat_material(services)
    pbat = pbat_material.selected_carbon_kg * selected
    pbat[:, 0] += pbat_material.fixed_fossil_carbon_kg
    pbs_material = _pbs_material(services)
    pbs = pbs_material.selected_carbon_kg * selected
    pbs[:, 0] += pbs_material.fixed_fossil_carbon_kg
    ptt_material = _ptt_material(services)
    # Eight PTT backbone carbon atoms come from PTA and three from PDO. PDO retains the source
    # glucose primer. Fixed catalyst carbon is fossil.
    ptt = ptt_material.selected_carbon_kg * (8 / 11 * selected + 3 / 11 * glucose)
    ptt[:, 0] += ptt_material.fixed_fossil_carbon_kg
    # ABS/SAN retains the precise atomic-mass convention used in its source
    # calculation; replacing it with integer masses changes the adopted value.
    c, h, n = 12.011, 1.008, 14.007
    monomers = {
        "styrene": 8 * c / (8 * c + 8 * h),
        "acrylonitrile": 3 * c / (3 * c + 3 * h + n),
        "pure_polybutadiene": 4 * c / (4 * c + 6 * h),
    }
    abs_weights = services["abs_asa_san_biogenic"]["portfolio_weights"]
    abs_carbon = fsum(
        abs_weights[kind]
        * fsum(
            amount * monomers[monomer]
            for monomer, amount in services[f"{kind}_biogenic"]["representative_method"][
                "incorporated_product_kg"
            ].items()
        )
        for kind in ("abs", "san")
    )
    ps_method = services["ps_biogenic"]["representative_method"]
    ps_weights = ps_method["provider_allocation"]
    ps_composition = ps_method["finished_material_reference"]["components_kg_per_finished_kg"]
    # Nominal integer masses C12/H1 give PS C96/104, PBR C48/54 and
    # saturated-hydrocarbon oil C6/7. The ACC/Franklin finished-grade reference
    # includes oil inside one commercial kg; no polymer-only normalization.
    ps_selected = fsum(
        ps_weights[grade] * (row["polystyrene"] * 96 / 104 + row["polybutadiene"] * 48 / 54)
        for grade, row in ps_composition.items()
    )
    ps_oil = fsum(
        ps_weights[grade] * row["mineral_oil"] * 6 / 7 for grade, row in ps_composition.items()
    )
    ps = ps_selected * selected
    # Fixed oil retains fossil origin through the existing cohort/trade/recovery graph.
    ps[:, 0] += ps_oil
    tps = np.zeros((3, 3))
    for i, route in enumerate(CARBON_ORIGINS):
        method = services[f"tps_{route}"]["representative_method"]
        starch = method["product_starch_kg" if route == "biogenic" else "product_amylose_kg"]
        tps[i] = (
            starch * _TPS_PRINCIPAL_ELEMENT_FRACTIONS["glucan"]["C"] * glucose[i]
            + method["product_glycerine_kg"]
            * _TPS_PRINCIPAL_ELEMENT_FRACTIONS["glycerol"]["C"]
            * selected[i]
        )
    phb_carbon = services["phb_biogenic"]["representative_method"]["carbon_balance_kg_c"][
        "phb_product"
    ]
    pla = 0.5 * selected
    pla[2] = 0.5 * glucose[2]
    vectors = {
        "abs_asa_san": abs_carbon * selected,
        "bio_pa": 72 / 113 * selected,
        "bio_pe": 6 / 7 * selected,
        "bio_pet": 0.625 * selected,
        "other_bio_based": ptt,
        "other_biodegradable": tps,
        "pbat": pbat,
        "pbs": pbs,
        "pha": phb_carbon * glucose,
        "pla": pla,
        "starch_blends": tps,
        "polypropylene": 6 / 7 * selected,
        "polyethylene_terephthalate": 0.625 * selected,
        "polyamide_6_resin": 72 / 113 * selected,
        "polyamide_66_resin": 144 / 226 * selected,
        "elastomers": 35 / 39 * selected,
        "hdpe": 6 / 7 * selected,
        "pet": 0.625 * selected,
        "pp": 6 / 7 * selected,
        "ps": ps,
        "pur": pur,
        "ldpe": 6 / 7 * selected,
        "pvc": 24 / 62.5 * selected,
        "adhesives": adhesive,
        "general_coatings": _EPOXY_PRINCIPAL_CARBON * selected,
        "marine_coatings": _EPOXY_PRINCIPAL_CARBON * selected,
        "non_tyre_elastomers": (0.235 * 96 / 104 + 0.765 * 48 / 54) * selected,
        "other_thermoplastics": 60 / 100 * selected,
        "pa": 72 / 113 * selected,
        "pc": 192 / 254 * selected,
        "road_marking_coatings": 60 / 100 * selected,
        "sealants": adhesive,
        "thermosets": native["upr"]["carbon"] * selected,
    }
    provenance = _carbon_provenance(catalogue, services, native)
    closures = _service_source_closures(catalogue)
    production_routes = {
        (
            entry["material"],
            entry["constituent"],
            entry["provider_demand_type"],
            entry["product"],
            entry["virgin_feedstock_route"],
        ): entry
        for entry in cast(list[dict[str, Any]], catalogue["routes"])
        if entry["provider_demand_type"]
        in {"virgin_resin_production", "polymerization_operation", "virgin_monomer_supply"}
    }
    rows = []
    for item in load_material_composition().constituents:
        _admit_product_carbon(
            vectors[item.constituent], f"Constituent {item.material}/{item.constituent}"
        )
        details = provenance[item.constituent]
        for i, route in enumerate(CARBON_ORIGINS):
            requests = (
                (
                    ("polymerization", "polymerization_operation", "polymer", "all"),
                    ("ethylene_supply", "virgin_monomer_supply", "ethylene", route),
                    ("propylene_supply", "virgin_monomer_supply", "propylene", route),
                )
                if item.chemical_polymer is not None
                else (("constituent_product", "virgin_resin_production", "polymer", route),)
            )
            components = []
            sources: set[str] = set()
            for role, demand, product, virgin_route in requests:
                entry = production_routes[
                    (item.material, item.constituent, demand, product, virgin_route)
                ]
                service_id = entry["provider_service_id"]
                citation = services[service_id]["citation"]
                route_method = entry.get("method_authority_id", citation)
                components.append(
                    {
                        "role": role,
                        "route_id": entry["route_id"],
                        "provider_service_id": service_id,
                        "method_authority_id": citation,
                        "route_method_authority_id": route_method,
                    }
                )
                sources.update(closures[service_id])
                mapping_sources = json.loads(
                    str(_method_authority(catalogue, route_method)["source_authority_ids_json"])
                )
                sources.update(_source_authority_closure(catalogue, mapping_sources))
            inputs = {
                **details,
                "provider_components": components,
                "source_authority_ids": sorted(sources),
            }
            if item.constituent in {
                "other_bio_based",
                "other_biodegradable",
                "starch_blends",
                "pha",
                "pla",
                "pur",
                "adhesives",
                "sealants",
            }:
                inputs["synthetic_glucan_biogenic_primer_share"] = primer
            if item.constituent in {"other_biodegradable", "starch_blends"}:
                tps_method = services[f"tps_{route}"]["representative_method"]
                inputs["principal_element_mass_fractions"] = _TPS_PRINCIPAL_ELEMENT_FRACTIONS
                inputs["product_glycerine_kg"] = tps_method["product_glycerine_kg"]
                inputs["product_glucan_kg"] = tps_method[
                    "product_starch_kg" if route == "biogenic" else "product_amylose_kg"
                ]
            vector = vectors[item.constituent][i]
            rows.append(
                dict(
                    zip(
                        MATERIAL_CARBON_COLUMNS,
                        (
                            item.material,
                            item.constituent,
                            route,
                            *map(float, vector),
                            "kg C/kg constituent",
                            f"{MATERIAL_CARBON_SCHEMA}:{item.constituent}",
                            json.dumps(inputs, sort_keys=True, separators=(",", ":")),
                        ),
                        strict=True,
                    )
                )
            )
    return tuple(rows)
