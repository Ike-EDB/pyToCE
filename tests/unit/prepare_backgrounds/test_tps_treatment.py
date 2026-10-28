"""Exercise actual TPS load owners and small linked Brightway ash treatment.

These direct-owner tests preserve the real three catalog formulations. Native
inventories are invented in the adjacent fixture; no licensed data or completed
workspace is read. Public site/dispatcher integration is a separate contract.

Owner arithmetic uses relative 2e-12 and zero absolute tolerance so trace flows
remain testable. Real LCI solves use relative 2e-7: bw_processing
create_structured_array stores external exchange amounts as float32, and these
small inventories traverse several such coefficients.
"""

import json
from collections.abc import Callable
from math import fsum, log
from typing import cast

import numpy as np
import pytest
from bw2calc import LCA
from material_recipe_fixtures import physical_material_recipe
from numpy.typing import NDArray
from tps_treatment_fixtures import (
    DRY_FRACTIONS,
    GROUND,
    LONG_WATER,
    ORIGINS,
    RURAL,
    SOIL,
    SURFACE,
    URBAN,
    biosphere_inventory,
    flow_key,
    native_inventory,
    treatment_catalogue,
)

from pytoce.prepare_backgrounds.preparation.ecoinvent import (
    BrightwayNames,
    selected_brightway_project,
)
from pytoce.prepare_backgrounds.preparation.providers.ash_treatment import (
    _MUNICIPAL_ASH_RELEASES,
    _ash_dataset,
    _ash_market_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.catalogue import (
    _method_authority_rows,
    _service_source_closures,
    provider_service_code,
)
from pytoce.prepare_backgrounds.preparation.providers.leachate_load import (
    _ROW_WASTEWATER_PROFILE,
    _wastewater_load,
)
from pytoce.prepare_backgrounds.preparation.providers.native_resolution import (
    _ActivityResolver,
    _FlowResolver,
)
from pytoce.prepare_backgrounds.preparation.providers.recipe_context import _RecipeContext
from pytoce.prepare_backgrounds.preparation.providers.sludge_incineration import (
    _sludge_dataset,
    _sludge_source,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_incineration_load import (
    calculate_sludge_incineration,
)
from pytoce.prepare_backgrounds.preparation.providers.sludge_landfarming import (
    _landfarming_dataset,
)
from pytoce.prepare_backgrounds.preparation.providers.tps_residue_load import (
    _tps_residue_load,
)


@pytest.fixture(scope="module")
def tps_catalogue():
    """Return untouched packaged formulations plus explicit test owner declarations."""
    return treatment_catalogue()


@pytest.fixture(scope="module", params=(False, True), ids=("original", "future"))
def tps_boundary(request, tmp_path_factory, tps_catalogue):
    """Write synthetic source inventories in a disposable Brightway project for each date."""
    future = request.param
    root = tmp_path_factory.mktemp("tps_treatment")
    names = BrightwayNames(root / "brightway", "tps_treatment", "biosphere", "original")
    with selected_brightway_project(names) as bd:
        bd.Database("biosphere").write(biosphere_inventory(), searchable=False)
        bd.Database("original").write(native_inventory("original", future=False), searchable=False)
        database = "future" if future else "original"
        if future:
            bd.Database(database).write(native_inventory(database, future=True), searchable=False)
        dated = _ActivityResolver(bd, database)
        context = _RecipeContext(
            background_scenario="remind_ssp2_npi",
            year=2050 if future else 2025,
            provider_database="providers",
            source=_ActivityResolver(bd, "original"),
            ordinary=dated,
            technology=dated,
            biosphere=_FlowResolver(bd, "biosphere"),
            catalogue=tps_catalogue,
            material_recipe=physical_material_recipe(),
            method_authorities={
                cast(str, r["method_authority_id"]): r
                for r in _method_authority_rows(tps_catalogue)
            },
            complete_source_ids=_service_source_closures(tps_catalogue),
            semantic_rows=[],
            technology_rows=[],
            traced_services=set(),
        )
        yield bd, context, database, future


def _loads(catalogue, origin):
    """Use actual TPS chemistry with independent invented site water and hydraulics."""
    services = {row["provider_service_id"]: row for row in catalogue["services"]}
    residue = _tps_residue_load(services, f"tps_{origin}")
    # Invented site: 2 L hydraulic service, 1.8 kg incoming water,
    # 1% hydraulic evaporation. Water and volume are independent input fields.
    wastewater = _wastewater_load(
        residue.leachate,
        volume_m3=0.002,
        incoming_water_kg=1.8,
        evaporation_fraction=0.01,
        profile=_ROW_WASTEWATER_PROFILE,
    )
    return services, residue, wastewater


@pytest.mark.parametrize("origin", ORIGINS)
def test_tps_material_and_conditional_wastewater_balance(tps_catalogue, origin):
    """Keep formulation atoms/origins, retained reagents and fresh WWT chemicals once."""
    services, residue, water = _loads(tps_catalogue, origin)
    method = services[f"tps_{origin}"]["representative_method"]
    components = {x.component: x for x in residue.components}
    glucan = method["product_starch_kg" if origin == "biogenic" else "product_amylose_kg"]
    glycerol = method["product_glycerine_kg"]
    assert glucan + glycerol == pytest.approx(1, rel=2e-12, abs=0)
    # Molecular references C6H10O5 and C3H8O3, CIAAW conventional g/mol.
    principal_carbon = glucan * (6 * 12.011) / (6 * 12.011 + 10 * 1.008 + 5 * 15.999)
    principal_carbon += glycerol * (3 * 12.011) / (3 * 12.011 + 8 * 1.008 + 3 * 15.999)
    assert fsum(x.input_carbon_kg for x in residue.carbon if x.principal) == pytest.approx(
        principal_carbon, rel=2e-12, abs=0
    )
    assert residue.water_kg == method["co_delivered_water_kg"]
    assert residue.wet_mass_kg == pytest.approx(
        fsum(x.mass_kg for x in residue.components) + residue.water_kg, rel=2e-12, abs=0
    )
    for component in residue.components:
        assert fsum(component.elements_kg.values()) == pytest.approx(
            component.mass_kg, rel=2e-12, abs=0
        )
    for pool in residue.carbon:
        assert pool.input_carbon_kg == pytest.approx(
            pool.methane_carbon_kg + pool.carbon_dioxide_carbon_kg + pool.leachate_carbon_kg,
            rel=2e-12,
            abs=0,
        )
        # Adopt the SP/IVL Fire-LCA, SP Report 2000:13, Ch. 7/Table 63
        # carbohydrate/protein degradation reference: 1% of input C in leachate
        # and 99% in gas. Applying the same extent to glycerol is a model assumption.
        if pool.component in {"glucan", "glycerol", "protein"}:
            assert pool.leachate_carbon_kg == pytest.approx(
                0.01 * pool.input_carbon_kg, rel=2e-12, abs=0
            )
            # Methane-C/gas-C: source starch value 0.5; balanced glycerol
            # C3H8O3 and selected averagine formulas give 7/12 and 25501/49384.
            gas_share = {"glucan": 0.5, "glycerol": 7 / 12, "protein": 25501 / 49384}[
                pool.component
            ]
            assert pool.methane_carbon_kg == pytest.approx(
                0.99 * gas_share * pool.input_carbon_kg, rel=2e-12, abs=0
            )
        else:
            assert pool.leachate_carbon_kg == pool.input_carbon_kg
    assert "Hg" not in residue.elements_kg and "Cd" not in residue.elements_kg
    if origin == "biogenic":
        assert set(components) == {"glucan", "glycerol"}
        assert {x.origin for x in residue.carbon} == {"biogenic"}
        assert water.duties.iron_sulfate_kg == 0
    else:
        assert set(components) == {
            "glucan",
            "glycerol",
            "protein",
            "methanol",
            "hepes",
            "sodium_chloride",
            "magnesium_chloride",
            "zinc_sulfate",
            "thiamine_pyrophosphate",
            "adp",
            "edta",
            "s4379",
        }
        assert components["protein"].mass_kg == method["co_delivered_protein_kg"]
        protein = next(x for x in residue.carbon if x.component == "protein")
        assert protein.origin == "biogenic" and not protein.principal
        # The S4379 powder uses its stated mean chain length of 45.
        powder = components["s4379"]
        p_fraction = 45 * 30.974 / (47 * 22.990 + 45 * 30.974 + 136 * 15.999)
        assert powder.elements_kg["P"] == pytest.approx(
            powder.mass_kg * p_fraction, rel=2e-12, abs=0
        )
        reaction = services[f"methanol_glucan_reaction_{origin}"]
        charge = next(
            x["amount"]
            for x in reaction["construction"]["components"]
            if x.get("service_id") == f"methanol_{origin}"
        )
        route = reaction["representative_method"]["selected_origin_share_of_dry_amylose"]
        recovery = services[f"starch_{origin}"]["representative_method"]
        # Net methanol subtracts 6 CH3OH per glucan repeat before membrane retention.
        unreacted = charge - route * 6 * 32.042 / 162.141
        assert components["methanol"].mass_kg == pytest.approx(
            unreacted
            * recovery["fully_passing_solute_coefficient"]
            * glucan
            / recovery["dry_amylose_kg"],
            rel=2e-12,
            abs=0,
        )
        assert water.duties.iron_sulfate_kg > 0

    # Every C/N/S/P/metal input, including added FeSO4/PAM, has one destination.
    for element, remaining in water.final_sludge.elements_kg.items():
        incoming = residue.leachate.elements_kg.get(element, 0)
        incoming += water.duties.iron_sulfate_elements_kg.get(element, 0)
        incoming += water.duties.polyacrylamide_elements_kg.get(element, 0)
        assert incoming == pytest.approx(
            remaining
            + water.effluent.elements_kg[element]
            + water.biological_air_elements_kg[element]
            + water.digester_gas_elements_kg[element],
            rel=2e-12,
            abs=0,
        )
    # Doka Wastewater 2021: GNIWWT AQ337 is the tertiary treatment share;
    # avgWWT R26 removes an additional 0.42 kg P/kg influent P at that stage.
    # WWcalc D263:D278 supplies 30.6456139178772 kg FeSO4/kg removed P.
    # Apply the coefficient once to the actual tertiary phosphorus removal P3.
    p3 = 0.5064009138103872 * 0.42 * residue.leachate.elements_kg.get("P", 0)
    assert water.tertiary_sludge.elements_kg["P"] == pytest.approx(p3, rel=2e-12, abs=0)
    assert water.duties.iron_sulfate_kg == pytest.approx(30.6456139178772 * p3, rel=2e-12, abs=0)
    iron_before = fsum(
        x.elements_kg["Fe"]
        for x in (water.primary_sludge, water.secondary_sludge, water.tertiary_sludge)
    )
    # Use one mol Fe per mol P, with molar masses of 55.85 and 30.97 g/mol.
    assert water.raw_sludge.elements_kg["Fe"] == pytest.approx(
        iron_before + p3 * 55.85 / 30.97, rel=2e-12, abs=0
    )
    # WWcalc D263:D278: 0.017265521465486582 kg PAM/kg secondary dry sludge.
    assert water.duties.polyacrylamide_kg == pytest.approx(
        0.017265521465486582 * water.secondary_sludge.dry_mass_kg, rel=2e-12, abs=0
    )
    # Doka2021 Section 8.1.1 / avgWWT L79: organic O/C is distinct
    # from FB23's oxide-equivalent dry-mass factor. Fresh undigested PAM
    # retains molecular O; digested carbon uses the source sludge reference.
    organic_o_per_c = 0.628571428571428
    pre_pam_c = fsum(
        part.elements_kg["C"]
        for part in (water.primary_sludge, water.secondary_sludge, water.tertiary_sludge)
    )
    raw_organic_o = pre_pam_c * organic_o_per_c + water.duties.polyacrylamide_elements_kg["O"]
    assert water.raw_sludge.organic_oxygen_kg == pytest.approx(raw_organic_o, rel=2e-12, abs=0)
    assert water.digested_sludge.organic_oxygen_kg == pytest.approx(
        water.digested_sludge.elements_kg["C"] * organic_o_per_c, rel=2e-12, abs=0
    )
    assert water.final_sludge.organic_oxygen_kg == pytest.approx(
        (1 - 0.20525339975098886) * raw_organic_o
        + water.digested_sludge.elements_kg["C"] * organic_o_per_c,
        rel=2e-12,
        abs=0,
    )
    for pool in water.carbon:
        assert pool.input_carbon_kg == pytest.approx(
            pool.biological_air_carbon_kg
            + pool.digester_gas_carbon_kg
            + pool.final_sludge_carbon_kg
            + pool.effluent_carbon_kg,
            rel=2e-12,
            abs=0,
        )
    pam = next(x for x in water.carbon if x.component == "wastewater_polyacrylamide")
    assert not pam.principal and pam.origin == "fossil"
    assert pam.input_carbon_kg == water.duties.polyacrylamide_elements_kg["C"]
    assert fsum(x.final_sludge_carbon_kg for x in water.carbon) == pytest.approx(
        water.final_sludge.elements_kg["C"], rel=2e-12, abs=0
    )
    assert water.incineration_dry_share + water.landfarming_dry_share == pytest.approx(
        1, rel=2e-12, abs=0
    )
    assert (
        water.incinerated_sludge_wet_kg * 0.3 + water.landfarmed_sludge_wet_kg * 0.03
        == pytest.approx(water.final_sludge.dry_mass_kg, rel=2e-12, abs=0)
    )
    assert water.water.hydraulic_evaporation_kg == pytest.approx(0.02, rel=2e-12, abs=0)
    # The secondary/tertiary biological share is GNIWWT AP337 + AQ337.
    # Doka2021 sends 0.245 kg C/kg influent C to biological air and transfers
    # the corresponding organic H using 18.016/(2*1.008) kg H2O/kg H.
    biological = 0.4115654223641656 + 0.5064009138103872
    assert water.water.biological_generated_kg == pytest.approx(
        biological * 0.245 * residue.leachate.organic_hydrogen_kg * 18.016 / (2 * 1.008),
        rel=2e-12,
        abs=0,
    )
    assert water.water.incoming_kg + water.water.biological_generated_kg == pytest.approx(
        water.water.effluent_kg
        + water.water.hydraulic_evaporation_kg
        + water.water.incinerated_sludge_water_kg
        + water.water.landfarmed_sludge_water_kg,
        rel=2e-12,
        abs=0,
    )
    assert water.water.effluent_kg > 0
    # WWcalc D291: 0.019585524957766807 kg screenings/m3. D297/D298 each
    # allocate 50% to plastic/biomass; D294 is the treated volume (here 0.002 m3).
    assert water.duties.grit_plastic_kg == water.duties.grit_biomass_kg
    assert water.duties.grit_plastic_kg * 2 == pytest.approx(
        0.002 * 0.019585524957766807, rel=2e-12, abs=0
    )


@pytest.mark.parametrize("origin", ORIGINS)
def test_tps_thermal_and_ash_linked_inventory(tps_boundary, tps_catalogue, origin):
    """Solve full tiny treatment LCI with actual auxiliary fates and native signs."""
    bd, context, native, future = tps_boundary
    services, _, wastewater = _loads(tps_catalogue, origin)
    sludge_id = f"tps_sludge_incineration_{origin}"
    construction = services[sludge_id]["construction"]
    load = calculate_sludge_incineration(
        _sludge_source(context, construction),
        wastewater,
        f"tps_{origin}",
        source_carbon_origins=tuple(dict.fromkeys(construction["carbon_templates"].values())),
        additional_apc_water_flows=tuple(construction["additional_apc_water_flows"]),
        phosphorus_phase_reference=False,
        # Selected plastic-leachate slag: 11.5628% water / 88.4372% dry.
        bottom_ash_water_to_dry_ratio=0.115628 / 0.884372,
    )
    provider = f"providers_{origin}"
    p = 5 if future else 2
    ash_p, market_p = (7, 6) if future else (3, 4)
    compiled = {
        (provider, provider_service_code(sludge_id)): _sludge_dataset(
            context, services[sludge_id], load, provider
        )
    }
    for fraction, payload in (("bottom", load.bottom_ash), ("residual", load.residual_ash)):
        for role, compiler in (
            ("market", _ash_market_dataset),
            ("treatment", _ash_dataset),
        ):
            service_id = f"tps_{fraction}_ash_{role}_{origin}"
            arguments = (
                {"release_profile": _MUNICIPAL_ASH_RELEASES[fraction]}
                if role == "treatment"
                else {}
            )
            compiled[provider, provider_service_code(service_id)] = cast(
                Callable[..., dict[str, object]], compiler
            )(context, services[service_id], payload, provider, **arguments)
    bd.Database(provider).write(compiled, searchable=False)
    sludge = bd.get_node(database=provider, code=provider_service_code(sludge_id))
    lca = LCA({sludge: -load.wet_mass_kg})
    lca.lci()

    def level(database, code):
        """Read the solved activity amount for one real node, including signed cut-off."""
        node = bd.get_node(database=database, code=code)
        return cast(NDArray[np.float64], lca.supply_array)[lca.dicts.activity[node.id]]

    assert level(provider, provider_service_code(sludge_id)) == pytest.approx(
        load.wet_mass_kg / p, rel=2e-7, abs=0
    )
    assert level(native, "sludge") == 0
    assert load.dry_mass_kg == pytest.approx(
        wastewater.final_sludge.dry_mass_kg * wastewater.incineration_dry_share, rel=2e-12, abs=0
    )
    assert load.water_kg == pytest.approx(0.7 * load.wet_mass_kg, rel=2e-12, abs=0)
    assert load.dry_mass_kg / load.wet_mass_kg == pytest.approx(0.3, rel=2e-12, abs=0)
    assert load.elements_kg["Cl"] == 0
    assert load.duties["FeCl3"] == 0 if origin == "biogenic" else load.duties["FeCl3"] > 0
    assert "TMT" not in load.duties and "TMT" not in load.fresh_apc_elements_kg
    assert level(native, "tmt") == 0
    # Doka 2003, report 13-II, Section 3.3.1: equal thermal/fuel-N shares at
    # 0.01 kg N/kg wet waste give the wet-kg + 100*kg-N driver. The independent
    # invented donor has N=.3*.04 kg/kg and original NH3=.0002 kg/kg.
    n_ratio = (load.wet_mass_kg + 100 * load.elements_kg["N"]) / (1 + 100 * 0.3 * 0.04)
    assert load.duties["NH3"] == pytest.approx(0.0002 * n_ratio, rel=2e-12, abs=0)
    assert load.duties["TiO2"] == pytest.approx(0.00001 * n_ratio, rel=2e-12, abs=0)
    assert load.duties["V2O5"] == pytest.approx(0.000002 * n_ratio, rel=2e-12, abs=0)
    shares = (0.7, 0.3) if future else (0.25, 0.75)
    # Retain the source process-water duty, including its dated supplier mix.
    supplied_water = load.duties["water"]
    for suffix, share in zip(("a", "b"), shares, strict=True):
        assert level(native, f"NH3_{suffix}") == pytest.approx(
            load.duties["NH3"] * share, rel=2e-7, abs=0
        )
        assert level(native, f"water_{suffix}") == pytest.approx(
            supplied_water * share, rel=2e-7, abs=0
        )
        for oxide in ("TiO2", "V2O5"):
            assert level(native, f"{oxide}_{suffix}") == pytest.approx(
                load.duties[oxide] * share, rel=2e-7, abs=0
            )
        assert level(native, f"PAM_{suffix}") == 0
    assert level(native, "actual_pam") == pytest.approx(load.duties["PAM"], rel=2e-7, abs=0)
    # Reference properties are per kg product despite production outputs of -2 or -5 kg.
    production = next(iter(sludge.production()))
    assert production["amount"] == -p
    assert production["properties"]["dry mass"]["amount"] == pytest.approx(0.3, rel=2e-12, abs=0)
    assert production["properties"]["carbon allocation"]["amount"] == pytest.approx(
        load.elements_kg["C"] / load.wet_mass_kg, rel=2e-12, abs=0
    )
    assert not {"flow", "activity", "classifications"}.intersection(production)
    for duty in ("NH3", "TiO2", "V2O5"):
        for suffix, share in zip(("a", "b"), shares, strict=True):
            row = next(
                ex for ex in sludge.technosphere() if ex.input.key == (native, f"{duty}_{suffix}")
            )
            expected = load.duties[duty] * share * p / load.wet_mass_kg
            assert row["amount"] == pytest.approx(expected, rel=2e-12, abs=0)
            assert row["loc"] == pytest.approx(log(expected) + 0.03, rel=2e-12, abs=0)
            assert row["uncertainty type"] == 2
            assert row["scale"] == 0.21 and row["allocation"] == "cut-off control"
            assert row["minimum"] == pytest.approx(expected / 2, rel=2e-12, abs=0)
            assert row["maximum"] == pytest.approx(expected * 2, rel=2e-12, abs=0)

    # N2O and CN follow waste-N; NOx includes the source thermal allocation.
    emissions = load.emissions
    assert emissions["Dinitrogen monoxide", "kilogram", URBAN] == pytest.approx(
        1e-10 * load.elements_kg["N"] / (0.3 * 0.04), rel=2e-12, abs=0
    )
    assert emissions["Nitrogen oxides", "kilogram", URBAN] == pytest.approx(
        1e-10 * n_ratio, rel=2e-12, abs=0
    )
    # Mixed-waste process VOCs persist, but no target or APC chlorine is invented.
    assert emissions["Hexachlorobenzene", "kilogram", URBAN] == pytest.approx(
        1e-10 * load.wet_mass_kg, rel=2e-12, abs=0
    )
    assert emissions["Chloride", "kilogram", SURFACE] == pytest.approx(
        load.aqueous_apc_elements_kg["Cl"], rel=2e-12, abs=0
    )
    assert emissions["COD, Chemical Oxygen Demand", "kilogram", SURFACE] == pytest.approx(
        emissions["DOC, Dissolved Organic Carbon", "kilogram", SURFACE] * 32 / 12, rel=2e-12, abs=0
    )
    # Trace carbon uses the nominal process reference.
    trace_atoms = {
        "Benzene": (6, 6, 0, 0),
        "Benzo(a)pyrene": (20, 12, 0, 0),
        "Hexachlorobenzene": (6, 0, 0, 6),
        "Pentachlorobenzene": (6, 1, 0, 5),
        "Pentachlorophenol": (6, 1, 1, 5),
        "Toluene": (7, 8, 0, 0),
    }
    air_c = fsum(
        emissions[f"{gas}, {o}", "kilogram", URBAN] * cf
        for gas, cf in (
            ("Carbon dioxide", 12 / 44),
            ("Carbon monoxide", 12 / 28),
            ("Methane", 12 / 16),
        )
        for o in ("fossil", "non-fossil")
    )
    air_c += (
        emissions["NMVOC, non-methane volatile organic compounds", "kilogram", URBAN] * 12 / 13.85
    )
    air_c += emissions["Cyanide", "kilogram", URBAN] * 12 / 26
    air_c += fsum(
        emissions[name, "kilogram", URBAN] * (12 * c) / (12 * c + h + 16 * o + 35.45 * cl)
        for name, (c, h, o, cl) in trace_atoms.items()
    )
    solid_c = load.bottom_ash.waste_elements_kg["C"] + load.residual_ash.waste_elements_kg["C"]
    assert air_c + solid_c + emissions[
        "DOC, Dissolved Organic Carbon", "kilogram", SURFACE
    ] == pytest.approx(load.elements_kg["C"], rel=2e-12, abs=0)
    pam_c = load.duties["PAM"] * (3 * 12.011) / (3 * 12.011 + 5 * 1.008 + 14.007 + 15.999)
    assert load.residual_ash.auxiliary_elements_kg["C"] == pytest.approx(pam_c, rel=2e-12, abs=0)
    assert fsum(
        x.carbon_kg for x in load.residual_ash.carbon if x.component == "APC_PAM"
    ) == pytest.approx(pam_c, rel=2e-12, abs=0)
    assert load.residual_ash.auxiliary_elements_kg["Ti"] == pytest.approx(
        load.duties["TiO2"] * 47.867 / (47.867 + 2 * 15.999), rel=2e-12, abs=0
    )
    # The chromium-trioxide manufacturing proxy supplies vanadium oxide, which retains its V
    # identity in ash.
    assert load.residual_ash.auxiliary_elements_kg["V"] == pytest.approx(
        load.duties["V2O5"] * (2 * 50.942) / (2 * 50.942 + 5 * 15.999), rel=2e-12, abs=0
    )
    # Doka 2021 sludgeIncin V23:V61 gives kg oxide-equivalent/kg element.
    # These independent source values cover all elements admitted by TPS WWT.
    # Doka 2003 II Section 2.6.2 sizes ash from those oxides: original H/O
    # remain separate material descriptors and cannot add a second oxygen mass.
    oxide_factors = {
        "C": 1,
        "S": 2.5,
        "N": 4.428571428571429,
        "P": 2.2915724895059735,
        "Cl": 1,
        "Zn": 1.2447605935444392,
        "Fe": 1.4297378599054578,
        "Mg": 1.6581112207963145,
        "Na": 1.3478260869565217,
    }
    # The shared receiver accepts Co/Ca/K. TPS supplies zero incoming loads for those slots.
    for ash in (load.bottom_ash, load.residual_ash):
        assert all(
            mass == 0
            for element, mass in ash.waste_elements_kg.items()
            if element not in oxide_factors
        )
    bottom_dry = fsum(
        mass * oxide_factors[element]
        for element, mass in load.bottom_ash.waste_elements_kg.items()
        if mass != 0
    )
    # Selected bottom-ash reference contains 11.5628% quench water by wet mass.
    quench_water = bottom_dry * 0.115628 / 0.884372
    bottom_mass = bottom_dry + quench_water
    residual_mass = fsum(
        mass * oxide_factors[element]
        for element, mass in load.residual_ash.waste_elements_kg.items()
        if mass != 0
    )
    # Fresh APC Fe/Ca enter residue as source oxide equivalents (V56/V57).
    # Neutral PAM and the TiO2/V2O5 catalyst retain their whole supplied masses.
    residual_mass += (
        load.residual_ash.auxiliary_elements_kg["Fe"] * oxide_factors["Fe"]
        + load.residual_ash.auxiliary_elements_kg["Ca"] * 1.3992015968063873
        + load.duties["PAM"]
        + load.duties["TiO2"]
        + load.duties["V2O5"]
    )
    for payload, expected_mass in (
        (load.bottom_ash, bottom_mass),
        (load.residual_ash, residual_mass),
    ):
        assert payload.allocated_oxygen_kg > 0
        assert payload.treatment_mass_kg == pytest.approx(expected_mass, rel=2e-12, abs=0)
    assert load.bottom_ash.water_kg == pytest.approx(quench_water, rel=2e-12, abs=0)
    # Doka source heat and water follow their allocation rules. The synthetic source's 3:1
    # air/surface-water Heat rows give the 0.75/0.25 partition.
    thermal = load.thermal
    assert emissions["Heat, waste", "megajoule", URBAN] == pytest.approx(
        0.75 * thermal["dissipated_heat_MJ"], rel=2e-12, abs=0
    )
    assert emissions["Heat, waste", "megajoule", SURFACE] == pytest.approx(
        0.25 * thermal["dissipated_heat_MJ"], rel=2e-12, abs=0
    )
    free_water = (
        load.water_kg + load.duties["water"] + thermal["nominal_reaction_water_kg"] - quench_water
    )
    # Doka's furnace water partition is 92.7% air and 7.3% surface water;
    # 1000 kg/m3 converts each independent post-quench mass to its flow unit.
    for compartment, share in ((URBAN, 0.927), (SURFACE, 0.073)):
        assert emissions["Water", "cubic meter", compartment] == pytest.approx(
            share * free_water / 1000, rel=2e-12, abs=0
        )
    assert thermal["NH3_dilution_within_process_water_kg"] <= load.duties["water"]
    assert free_water + quench_water == pytest.approx(
        load.water_kg + supplied_water + thermal["nominal_reaction_water_kg"],
        rel=2e-12,
        abs=0,
    )

    # Independent terminal p and haulage p cancel at their respective owners.
    total_ash = bottom_mass + residual_mass
    for fraction, payload, expected_mass in (
        ("bottom", load.bottom_ash, bottom_mass),
        ("residual", load.residual_ash, residual_mass),
    ):
        terminal_id = provider_service_code(f"tps_{fraction}_ash_treatment_{origin}")
        market_id = provider_service_code(f"tps_{fraction}_ash_market_{origin}")
        assert level(provider, terminal_id) == pytest.approx(expected_mass / ash_p, rel=2e-7, abs=0)
        assert level(provider, market_id) == pytest.approx(
            expected_mass / market_p, rel=2e-7, abs=0
        )
        assert level(native, fraction) == 0 and level(native, fraction + "_market") == 0
        terminal = bd.get_node(database=provider, code=terminal_id)
        actual = dict(payload.waste_elements_kg)
        for element, amount in payload.auxiliary_elements_kg.items():
            actual[element] = actual.get(element, 0) + amount
        reconstructed_dry = (
            fsum(actual.values()) + payload.allocated_hydrogen_kg + payload.allocated_oxygen_kg
        )
        props = next(iter(terminal.production()))["properties"]
        assert props["dry mass"]["amount"] == pytest.approx(
            reconstructed_dry / payload.treatment_mass_kg, rel=2e-12, abs=0
        )
        assert props["carbon content"]["amount"] == pytest.approx(
            actual["C"] / reconstructed_dry, rel=2e-12, abs=0
        )
        for ion, element in (
            ("Phosphate", "P"),
            ("Iron ion", "Fe"),
            ("DOC, Dissolved Organic Carbon", "C"),
        ):
            rows = [ex for ex in terminal.biosphere() if ex.input.key == flow_key(ion, SURFACE)]
            emitted = fsum(ex["amount"] for ex in rows) * payload.treatment_mass_kg / ash_p
            assert emitted == pytest.approx(
                1e-7 * actual[element] / (0.8 * DRY_FRACTIONS[element]), rel=2e-12, abs=0
            )
        assert not any(
            ex.input["name"] in {"Cadmium II", "Mercury II"} for ex in terminal.biosphere()
        )
    factor = 2 if future else 1
    assert level(native, "freight") == pytest.approx(0.05 * factor * total_ash, rel=2e-7, abs=0)
    assert level(native, "support") == pytest.approx(0.02 * factor * total_ash, rel=2e-7, abs=0)
    residual = residual_mass
    assert level(native, "cement") == pytest.approx(0.4 * residual, rel=2e-7, abs=0)
    assert level(native, "hydration") == pytest.approx(0.6 * residual, rel=2e-7, abs=0)
    assert level(native, "hydrated") == pytest.approx(0.8 * residual, rel=2e-7, abs=0)
    terminal = bd.get_node(
        database=provider, code=provider_service_code(f"tps_residual_ash_treatment_{origin}")
    )
    ti = load.residual_ash.auxiliary_elements_kg["Ti"]
    # Doka 2003, report 13-III, Table A8: 0.0004956 short-term and 0.2973
    # cumulative kg Ti/kg incoming Ti. Eq. 5.22 subtracts the short-term part
    # to obtain the additional long-term emission, avoiding double counting.
    for compartment, fraction in ((SURFACE, 0.0004956), (LONG_WATER, 0.2973 - 0.0004956)):
        row = next(
            ex
            for ex in terminal.biosphere()
            if ex.input.key == flow_key("Titanium ion", compartment)
        )
        assert row["amount"] * residual / ash_p == pytest.approx(ti * fraction, rel=2e-12, abs=0)
    # Complete invented upstream emissions include native catalyst suppliers
    # and the retained cement-water-disposal function.
    leaves = [
        node
        for node in bd.Database(native)
        if node["code"]
        not in {"sludge", "bottom", "residual", "bottom_market", "residual_market", "landfarming"}
    ]
    expected_upstream = 0.02 * fsum(level(native, node["code"]) for node in leaves)
    co2 = bd.get_node(database="biosphere", code=flow_key("Carbon dioxide, fossil", URBAN)[1])
    total_fossil = float(lca.inventory[lca.dicts.biosphere[co2.id], :].sum())
    assert total_fossil == pytest.approx(
        expected_upstream + emissions["Carbon dioxide, fossil", "kilogram", URBAN], rel=2e-7, abs=0
    )
    trace = next(
        row
        for row in context.technology_rows
        if row.get("provider_service_id") == sludge_id
        and row.get("component_id") == "actual_sludge_incineration"
    )
    terms = json.loads(trace["project_terms_json"])
    assert terms["formulation_service_id"] == f"tps_{origin}"
    assert "co-treated" in terms["process_emission_boundary"]


@pytest.mark.parametrize("origin", ORIGINS)
def test_tps_landfarming_complete_spreading_and_nutrient_fates(tps_boundary, tps_catalogue, origin):
    """Retain agricultural C/N/P/water fates and a full spreading supplier once."""
    bd, context, native, future = tps_boundary
    services, _, wastewater = _loads(tps_catalogue, origin)
    service_id = f"tps_sludge_landfarming_{origin}"
    provider = f"landfarming_{origin}"
    record = _landfarming_dataset(
        context,
        services[service_id],
        wastewater,
        f"tps_{origin}",
        provider,
        complete_carbon_oxidation=False,
    )
    bd.Database(provider).write(
        {(provider, provider_service_code(service_id)): record}, searchable=False
    )
    node = bd.get_node(database=provider, code=provider_service_code(service_id))
    wet = wastewater.landfarmed_sludge_wet_kg
    fraction = wastewater.landfarming_dry_share
    lca = LCA({node: -wet})
    lca.lci()
    p = 11 if future else 8
    assert cast(NDArray[np.float64], lca.supply_array)[
        lca.dicts.activity[node.id]
    ] == pytest.approx(wet / p, rel=2e-7, abs=0)
    spreading = bd.get_node(database=native, code="spreading")
    spreading_quantity = wet * (2 if future else 1) / 1030
    assert cast(NDArray[np.float64], lca.supply_array)[
        lca.dicts.activity[spreading.id]
    ] == pytest.approx(spreading_quantity, rel=2e-7, abs=0)
    assert len(list(node.technosphere())) == 1
    production = next(iter(node.production()))
    assert production["amount"] == -p
    assert production["properties"]["dry mass"]["amount"] == pytest.approx(0.03, rel=2e-12, abs=0)
    assert production["properties"]["water in wet mass"]["amount"] == pytest.approx(
        0.97, rel=2e-12, abs=0
    )

    emissions = {
        (ex.input["name"], tuple(ex.input["categories"])): ex["amount"] * wet / p
        for ex in node.biosphere()
    }
    nitrogen = fraction * wastewater.final_sludge.elements_kg["N"]
    phosphorus = fraction * wastewater.final_sludge.elements_kg["P"]
    assert emissions["Ammonia", RURAL] * 14 / 17 == pytest.approx(0.1 * nitrogen, rel=2e-12, abs=0)
    assert emissions["Dinitrogen monoxide", RURAL] * 28 / 44 == pytest.approx(
        0.1 * nitrogen, rel=2e-12, abs=0
    )
    assert emissions["Nitrogen oxides", RURAL] * 14 / 46 == pytest.approx(
        0.1 * nitrogen, rel=2e-12, abs=0
    )
    assert emissions["Nitrate", SURFACE] * 14 / 62 == pytest.approx(
        0.1 * nitrogen, rel=2e-12, abs=0
    )
    # Legitimate biogenic formulation has zero P; no zero-P flow is fabricated.
    for compartment in (SURFACE, GROUND):
        assert emissions.get(("Phosphate", compartment), 0) * 31 / 95 == pytest.approx(
            0.1 * phosphorus, rel=2e-12, abs=0
        )
    assert emissions["Water", RURAL] * 1000 == pytest.approx(0.8 * 0.97 * wet, rel=2e-12, abs=0)
    assert emissions["Water", GROUND] * 1000 == pytest.approx(0.2 * 0.97 * wet, rel=2e-12, abs=0)
    for carbon_origin in ("fossil", "non-fossil"):
        carbon = fraction * fsum(
            x.final_sludge_carbon_kg
            for x in wastewater.carbon
            if (x.origin == "fossil") == (carbon_origin == "fossil")
        )
        assert emissions["Carbon dioxide, " + carbon_origin, RURAL] * 12 / 44 == pytest.approx(
            carbon, rel=2e-12, abs=0
        )
    sludge = wastewater.final_sludge
    # Doka Wastewater 2021, WWcalc DU7: empirical gross heat coefficients
    # in MJ/kg element (C 34.016, H 124.265, O -9.8324, N 6.276, S 19.079).
    # H/O follow the source-model allocation; the resulting
    # MJ replaces the invented donor's 99 MJ/kg sludge heat quantity.
    heat = fraction * (
        34.016 * sludge.elements_kg["C"]
        + 124.265 * sludge.allocated_hydrogen_kg
        - 9.8324 * sludge.allocated_oxygen_kg
        + 6.276 * sludge.elements_kg["N"]
        + 19.079 * sludge.elements_kg["S"]
    )
    assert heat > 0
    assert emissions["Heat, waste", SOIL] == pytest.approx(heat, rel=2e-12, abs=0)
    assert production["properties"]["heating value, gross"]["amount"] == pytest.approx(
        heat / wet, rel=2e-12, abs=0
    )
    assert "Cadmium II" not in {name for name, _ in emissions}
    trace = next(
        row
        for row in context.technology_rows
        if row.get("provider_service_id") == service_id
        and row.get("component_id") == "actual_sludge_landfarming"
    )
    terms = json.loads(trace["project_terms_json"])
    assert terms["nitrogen_crop_uptake_kg"] == pytest.approx(0.6 * nitrogen, rel=2e-12, abs=0)
    assert terms["phosphorus_crop_uptake_kg"] == pytest.approx(0.8 * phosphorus, rel=2e-12, abs=0)
    assert terms["load_owner_service_id"] == f"tps_{origin}"
    # The complete external spreading function contributes to the linked inventory.
    co2 = bd.get_node(database="biosphere", code=flow_key("Carbon dioxide, fossil", URBAN)[1])
    assert float(lca.inventory[lca.dicts.biosphere[co2.id], :].sum()) == pytest.approx(
        0.02 * spreading_quantity, rel=2e-7, abs=0
    )
