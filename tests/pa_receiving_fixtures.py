"""Invented detached PA receiving records for prepared-product persistence tests.

These deterministic values are small synthetic scientific inputs. Portable native endpoints identify
their fixture source.
Real native source and supplier selection are tested at their admission owner.
"""

from pytoce.prepare_backgrounds.preparation.providers import (
    pa_municipal_receiving_sources as municipal,
)
from pytoce.prepare_backgrounds.preparation.providers import (
    pa_recycling_receiving_sources as recycling,
)
from pytoce.prepare_backgrounds.preparation.providers.landfill_phases import _LandfillSite
from pytoce.prepare_backgrounds.preparation.providers.pa_dump_receiving import (
    PaDumpReceivingReference,
    PaDumpSite,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_landfill_receiving_sources import (
    PaLandfillReceivingReference,
    PaLandfillReceivingSources,
    PaLandfillSludgeReference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_open_burning_sources import (
    PaOpenBurningReference,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingAshReference,
    PaReceivingFlow,
    PaReceivingLandfarmingReference,
    PaReceivingOperation,
    PaReceivingRow,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_receiving_sources import (
    PaReceivingFuelReference,
    PaReceivingGasReference,
    PaReceivingGasSources,
    PaReceivingReference,
    PaReceivingSources,
)
from pytoce.prepare_backgrounds.preparation.providers.pa_water_receiving_sources import (
    PaWaterReceivingReference,
    PaWaterReceivingSources,
    PaWaterSludgeReference,
)


def _activity(
    database: str, code: str, product: str, unit: str = "kilogram", *, location: str = "CH"
) -> PaReceivingActivity:
    """Give an invented supply an explicit native product, unit and portable key."""
    return PaReceivingActivity((database, code), f"invented {code}", product, unit, location)


def _water_records(
    database: str, frontier: str
) -> tuple[PaWaterReceivingReference, PaWaterReceivingSources]:
    """Provide synthetic water records for completed-product persistence tests.

    These small portfolios exercise nested endpoints, calibration and row
    serialization. The complete CH receiving calculation is covered by
    water_receiving_boundary.
    """
    power = PaReceivingRow(
        _activity(database, "water_power", "electricity, medium voltage", "kilowatt hour"),
        0.2,
        {},
    )
    surface = ("water", "surface water")
    phosphate = PaReceivingFlow(("biosphere", "water_phosphate"), "Phosphate", "kilogram", surface)
    carbon = PaReceivingRow(
        PaReceivingFlow(
            ("biosphere", "water_doc"), "DOC, Dissolved Organic Carbon", "kilogram", surface
        ),
        0.0002,
        {},
    )
    operations = {
        role: PaReceivingOperation(_activity(database, role, product, unit), production, (power,))
        for role, product, unit, production in (
            ("collection", "wastewater, average", "cubic meter", -5.0),
            ("wastewater", "wastewater, average", "cubic meter", -7.0),
            ("sludge_incineration_market", "sewage sludge, 70% water", "kilogram", -1.0),
            ("sludge_landfarming_market", "sewage sludge, 97% water", "kilogram", -1.0),
            ("sludge_incineration", "sewage sludge, 70% water, WWT, WW, average", "kilogram", -2.0),
            ("sludge_landfarming", "sewage sludge, 97% water, WWT, WW, average", "kilogram", -8.0),
            ("bottom_ash_market", "bottom ash, MSWI[F]-WWT, WW, average", "kilogram", -1.0),
            ("residual_ash_market", "residues, MSWI[F]-WWT, WW, average", "kilogram", -1.0),
            ("bottom_ash_treatment", "bottom ash, MSWI[F]-WWT, WW, average", "kilogram", -3.0),
            ("residual_ash_treatment", "residues, MSWI[F]-WWT, WW, average", "kilogram", -4.0),
        )
    }
    reference = PaWaterReceivingReference(
        wastewater=operations["wastewater"].activity,
        evaporation_fraction=0.01,
        sludge=PaWaterSludgeReference(
            PaReceivingOperation(operations["sludge_incineration"].activity, -2.0, (carbon,)),
            {"C": 0.12, "N": 0.012, "P": 0.006},
        ),
        landfarming=PaReceivingLandfarmingReference(
            PaReceivingOperation(operations["sludge_landfarming"].activity, -8.0, (carbon,)),
            0.03,
            0.97,
            {"C": 0.4, "N": 0.04},
        ),
        bottom_ash=PaReceivingAshReference(
            PaReceivingOperation(operations["bottom_ash_treatment"].activity, -3.0, (carbon,)),
            0.8,
            {"C": 0.4},
        ),
        residual_ash=PaReceivingAshReference(
            PaReceivingOperation(operations["residual_ash_treatment"].activity, -4.0, (carbon,)),
            0.8,
            {"C": 0.4},
        ),
    )
    sources = PaWaterReceivingSources(
        **operations,
        pam=_activity(database, "water_pam", "polyacrylamide"),
        biosphere=(phosphate,),
    )
    return reference, sources


def _landfill_records(
    database: str, frontier: str
) -> tuple[PaLandfillReceivingReference, PaLandfillReceivingSources]:
    """Provide synthetic RoW/GLO records for parent-file persistence tests.

    These portfolios exercise eleven role identities, intensive calibration
    and nested serialization.
    """
    power = PaReceivingRow(
        _activity(
            database, "landfill_power", "electricity, low voltage", "kilowatt hour", location="RoW"
        ),
        0.03,
        {},
    )
    liquid = "leachate, SLF, waste plastic, mixture"
    sludge = "sewage sludge, 70% water, WWT-SLF, waste plastic, mixture"
    field = "sewage sludge, 97% water, WWT-SLF, waste plastic, mixture"
    bottom = "bottom ash, MSWI-WWT-SLF, waste plastic, mixture"
    residual = "residues, MSWI-WWT-SLF, waste plastic, mixture"
    operations = {
        role: PaReceivingOperation(
            _activity(database, "landfill_" + role, product, unit, location=location),
            production,
            (power,),
        )
        for role, product, unit, production, location in (
            ("landfill", "waste plastic, mixture", "kilogram", -4.0, "RoW"),
            ("collection", liquid, "cubic meter", -2.0, "RoW"),
            ("wastewater", liquid, "cubic meter", -3.0, "RoW"),
            ("sludge_incineration_market", sludge, "kilogram", -1.0, "RoW"),
            ("sludge_landfarming_market", field, "kilogram", -1.0, "RoW"),
            ("sludge_incineration", sludge, "kilogram", -2.0, "GLO"),
            ("sludge_landfarming", field, "kilogram", -8.0, "RoW"),
            ("bottom_ash_market", bottom, "kilogram", -1.0, "GLO"),
            ("residual_ash_market", residual, "kilogram", -1.0, "GLO"),
            ("bottom_ash_treatment", bottom, "kilogram", -3.0, "GLO"),
            ("residual_ash_treatment", residual, "kilogram", -4.0, "GLO"),
        )
    }
    surface = ("water", "surface water")
    carbon = PaReceivingRow(
        PaReceivingFlow(
            ("biosphere", "landfill_doc"), "DOC, Dissolved Organic Carbon", "kilogram", surface
        ),
        0.0002,
        {},
    )
    reference = PaLandfillReceivingReference(
        landfill=operations["landfill"].activity,
        climate=_LandfillSite(18.0, 1200.0, 1.0),
        leachate_volume_m3_per_wet_kg=0.006,
        wastewater=operations["wastewater"].activity,
        evaporation_fraction=0.0,
        sludge=PaLandfillSludgeReference(
            PaReceivingOperation(operations["sludge_incineration"].activity, -2.0, (carbon,)),
            {"C": 0.12, "N": 0.012, "S": 0.006},
        ),
        landfarming=PaReceivingLandfarmingReference(
            PaReceivingOperation(operations["sludge_landfarming"].activity, -8.0, (carbon,)),
            0.03,
            0.97,
            {"C": 0.4, "N": 0.04},
        ),
        bottom_ash=PaReceivingAshReference(
            PaReceivingOperation(operations["bottom_ash_treatment"].activity, -3.0, (carbon,)),
            0.8,
            {"C": 0.4},
        ),
        residual_ash=PaReceivingAshReference(
            PaReceivingOperation(operations["residual_ash_treatment"].activity, -4.0, (carbon,)),
            0.8,
            {"C": 0.4},
        ),
    )
    sources = PaLandfillReceivingSources(
        **operations,
        pam=_activity(database, "landfill_pam", "polyacrylamide", location="GLO"),
        biosphere=(
            PaReceivingFlow(("biosphere", "landfill_phosphate"), "Phosphate", "kilogram", surface),
        ),
    )
    return reference, sources


def _recycling_records(
    database: str, frontier: str
) -> tuple[recycling.PaRecyclingReceivingReference, recycling.PaRecyclingReceivingSources]:
    """Provide synthetic recycling records for persistence tests."""
    water = PaReceivingRow(_activity(database, "recycling_water", "tap water"), 4.8, {})
    dust = PaReceivingRow(
        PaReceivingFlow(
            ("biosphere", "recycling_dust"),
            "Particulate Matter, > 10 um",
            "kilogram",
            ("air",),
        ),
        0.0004,
        {},
    )
    waste = PaReceivingRow(
        _activity(
            database, "recycling_wastewater", "wastewater from recycling of waste PP", "cubic meter"
        ),
        -0.0048,
        {},
    )
    original = _activity(database, "recycling_preparation", "ABS flakes, recycled", location="RER")
    reference = recycling.PaRecyclingReceivingReference(original, 1.6, 4.8)
    sources = recycling.PaRecyclingReceivingSources(
        preparation=PaReceivingOperation(
            _activity(frontier, "recycling_preparation", "water-only preparation", location="RER"),
            2.0,
            (water, waste, dust),
        ),
        finishing=PaReceivingOperation(
            _activity(frontier, "recycling_finishing", "generic pelletising", location="RER"),
            3.0,
            (dust,),
        ),
        preparation_pm_rows=(2,),
        finishing_pm_rows=(0,),
        preparation_water_rows=(0,),
        preparation_wastewater_rows=(1,),
        pellet_delivery=_activity(
            frontier, "recycling_delivery", "recycled pellet delivery", location="RER"
        ),
        evaporated_water=PaReceivingFlow(
            ("biosphere", "recycling_water_air"), "Water", "cubic meter", ("air",)
        ),
    )
    return reference, sources


def _municipal_records(
    database: str, frontier: str
) -> tuple[municipal.PaMunicipalReceivingReference, municipal.PaMunicipalReceivingSources]:
    """Describe five synthetic municipal roles for serialization tests."""
    power = PaReceivingRow(
        _activity(database, "municipal_power", "electricity, medium voltage", "kilowatt hour"),
        0.07,
        {},
    )
    carbon = PaReceivingFlow(
        ("biosphere", "municipal_co2"),
        "Carbon dioxide, fossil",
        "kilogram",
        ("air", "urban air close to ground"),
    )
    operations = {
        role: PaReceivingOperation(
            _activity(database, "municipal_" + role, product, location="RoW"),
            production,
            (power,),
        )
        for role, product, production in (
            ("incineration", "waste plastic, mixture", -2.0),
            ("bottom_ash_market", "bottom ash, MSWI, waste plastic, mixture", -3.0),
            ("residual_ash_market", "residues, MSWI, waste plastic, mixture", -4.0),
            ("bottom_ash_treatment", "bottom ash, MSWI, waste plastic, mixture", -5.0),
            ("residual_ash_treatment", "residues, MSWI, waste plastic, mixture", -6.0),
        )
    }
    reference = municipal.PaMunicipalReceivingReference(
        operations["incineration"],
        {"C": 0.4, "H": 0.06, "O": 0.1, "N": 0.01, "S": 0.001, "Na": 0.01},
        0.2,
        PaReceivingAshReference(operations["bottom_ash_treatment"], 0.8, {"C": 0.12}),
        PaReceivingAshReference(operations["residual_ash_treatment"], 0.9, {"C": 0.08}),
    )
    sources = municipal.PaMunicipalReceivingSources(
        **operations,
        pam=_activity(database, "municipal_pam", "polyacrylamide"),
        reheat_product="heat, district or industrial",
        biosphere=(carbon,),
    )
    return reference, sources


def _unmanaged_references(
    database: str,
) -> tuple[PaOpenBurningReference, PaDumpReceivingReference]:
    """Retain original unmanaged identities in small persistence-only records."""
    carbon = PaReceivingFlow(
        ("biosphere", "unmanaged_carbon"), "Carbon", "kilogram", ("soil", "industrial")
    )
    sites = tuple(
        PaDumpSite(
            label,
            PaReceivingOperation(
                _activity(database, f"dump_{index}", "waste polyurethane", location="GLO"),
                -(index + 1.0),
                (PaReceivingRow(carbon, 0.001, {}),),
            ),
            _LandfillSite(8.0, precipitation, 1 - 2.0604 / 7),
            index == 0,
        )
        for index, (label, precipitation) in enumerate(
            (
                ("hyperarid infiltration class (-250mm)", 350.0),
                ("dry infiltration class (100mm)", 550.0),
                ("moist infiltration class (300mm)", 900.0),
                ("wet infiltration class (500mm)", 1100.0),
                ("very wet infiltration class (1000mm)", 1900.0),
            )
        )
    )
    return (
        PaOpenBurningReference(
            _activity(database, "open_burning", "waste plastic, mixture", location="GLO"),
            (carbon,),
        ),
        PaDumpReceivingReference(sites, (carbon,)),
    )


def pa_receiving_reference(database: str = "source") -> PaReceivingReference:
    """Return original wet-waste and dry-ash elemental calibration in kg/kg."""
    burning, dump = _unmanaged_references(database)
    return PaReceivingReference(
        hwi=_activity(database, "hwi", "hazardous waste, for incineration"),
        donor_elements_kg={
            "C": 0.3,
            "H": 0.06,
            "O": 0.04,
            "N": 0.01,
            "S": 0.02,
            "P": 0.005,
            "B": 0.001,
            "Cl": 0.05,
            "F": 0.003,
            "Cd": 0.00001,
            "Co": 0.00002,
            "Cr": 0.00003,
            "Cu": 0.00004,
            "Hg": 0.000005,
            "Ni": 0.00002,
            "Pb": 0.00005,
            "Zn": 0.0001,
            "Si": 0.001,
        },
        ash=_activity(database, "ash", "residues, MSWI-WWT-SLF, municipal solid waste"),
        ash_dry_fraction=1.0,
        ash_element_contents={
            "carbon": 0.05,
            "nitrogen": 0.03,
            "phosphorus": 0.025,
            "sulfur": 0.012,
            "sodium": 0.1,
            "calcium": 0.2,
            "vanadium": 0.02,
            "iron": 0.04,
            "nickel": 0.02,
            "silicon": 0.03,
            "aluminium": 0.07,
            "potassium": 0.11,
            "magnesium": 0.035,
        },
        gas=PaReceivingGasReference(
            chp=PaReceivingFuelReference(
                _activity(
                    database,
                    "chp",
                    "heat, central or small-scale, other than natural gas",
                    "megajoule",
                ),
                _activity(database, "biogas", "biogas", "cubic meter"),
                18.0,
            ),
            furnace=PaReceivingFuelReference(
                _activity(
                    database, "furnace", "heat, district or industrial, natural gas", "megajoule"
                ),
                _activity(database, "natural_gas", "natural gas, high pressure", "cubic meter"),
                36.0,
            ),
        ),
        water=_water_records(database, "frontier")[0],
        landfill=_landfill_records(database, "frontier")[0],
        recycling=_recycling_records(database, "frontier")[0],
        municipal=_municipal_records(database, "frontier")[0],
        open_burning=burning,
        dump=dump,
    )


def pa_receiving_sources(
    year: int = 2025,
    *,
    background_scenario: str | None = None,
    source_database: str = "source",
) -> PaReceivingSources:
    """Return normalized original/future operations with distinct source coordinates.

    Rows use each endpoint's unit per positive unit of the operation's reference
    product. HWI rows are per kg wet treatment; gas rows are per MJ. Future
    peroxide is 50% higher so persistence checks can distinguish dated quantities.
    """
    database = source_database if year <= 2025 else f"native_{background_scenario}_{year}"
    frontier = f"provider_{background_scenario}_{year}"
    heat = (
        "heat, district or industrial, natural gas"
        if year <= 2025
        else "heat, district or industrial"
    )
    urban = ("air", "urban air close to ground")
    surface, later = ("water", "surface water"), ("water", "ground-, long-term")

    def supply(code: str, product: str, amount: float, unit: str = "kilogram") -> PaReceivingRow:
        """Describe one deterministic normalized native supply."""
        return PaReceivingRow(_activity(database, code, product, unit), amount, {})

    def release(
        code: str, name: str, categories: tuple[str, ...], amount: float, unit: str = "kilogram"
    ) -> PaReceivingRow:
        """Describe one existing elementary identity with its output unit."""
        return PaReceivingRow(
            PaReceivingFlow(("biosphere", code), name, unit, categories), amount, {}
        )

    hwi_rows = (
        tuple(
            supply(code, product, amount, unit)
            for code, product, amount, unit in (
                ("ammonia", "ammonia, anhydrous, liquid", 0.0001, "kilogram"),
                ("heat", heat, 0.02, "megajoule"),
                (
                    "hcl",
                    "hydrochloric acid, without water, in 30% solution state",
                    0.02,
                    "kilogram",
                ),
                ("naoh", "sodium hydroxide, without water, in 50% solution state", 0.1, "kilogram"),
                (
                    "peroxide",
                    "hydrogen peroxide, without water, in 50% solution state",
                    0.004 if year <= 2025 else 0.006,
                    "kilogram",
                ),
                (
                    "ferric",
                    "iron(III) chloride, without water, in 40% solution state",
                    0.006,
                    "kilogram",
                ),
                ("tmt", "chemical, organic, basic precursors and materials", 0.0001, "kilogram"),
                ("calcium_chloride", "calcium chloride", 0.008, "kilogram"),
                ("lime", "quicklime, milled, packed", 0.03, "kilogram"),
                ("flocculant", "chemical, inorganic", 0.01, "kilogram"),
                ("titanium", "titanium dioxide", 0.0001, "kilogram"),
                ("vanadium_proxy", "chromium", 0.00002, "kilogram"),
            )
        )
        + tuple(
            release(code, name, urban, amount)
            for code, name, amount in (
                ("co2", "Carbon dioxide, fossil", 1.0),
                ("co", "Carbon monoxide, fossil", 0.00001),
                ("cyanide", "Cyanide", 0.000004),
                ("ammonia_air", "Ammonia", 0.000007),
                ("n2o", "Dinitrogen monoxide", 0.00001),
                ("nox", "Nitrogen oxides", 0.0001),
                ("so2", "Sulfur dioxide", 0.00002),
                ("nickel", "Nickel II", 0.000001),
                ("silicon", "Silicon", 0.000002),
            )
        )
        + (
            release("doc_surface", "DOC, Dissolved Organic Carbon", surface, 0.00003),
            release("doc_later", "DOC, Dissolved Organic Carbon", later, 0.002),
        )
    )
    fixed_rows = (
        supply("ammonia", "ammonia, anhydrous, liquid", 0.002),
        supply("heat", heat, 0.3, "megajoule"),
        supply("titanium", "titanium dioxide", 0.0002),
        supply("vanadium_proxy", "chromium", 0.00005),
        release("ammonia_air", "Ammonia", urban, 0.00002),
        release("water_air", "Water", ("air",), 0.0005, "cubic meter"),
        release("water_final", "Water", ("water",), 0.0045, "cubic meter"),
        release(
            "water_resource",
            "Water, well, in ground",
            ("natural resource", "in water"),
            0.005,
            "cubic meter",
        ),
    )
    ash_rows = (
        supply("ash_support", "process-specific burdens, residual material landfill", 1.0),
        supply("cement", "cement, unspecified", 0.4),
        supply("water", "tap water", 0.6),
        release("doc_surface", "DOC, Dissolved Organic Carbon", surface, 0.005),
        release("doc_later", "DOC, Dissolved Organic Carbon", later, 0.02),
        release("silicon_surface", "Silicon", surface, 0.006),
        release("silicon_later", "Silicon", later, 0.03),
    )
    # These invented MJ-reference operations use the same combustion calibration
    # roles as the real gas fixture. Fuel m3 and emission rows share one
    # normalized MJ basis; only the original record supplies fuel MJ/m3.
    rural = ("air", "non-urban air or from high stacks")
    gas_rows = (
        supply("biogas", "biogas", 0.08 if year <= 2025 else 0.10, "cubic meter"),
        *(
            release(code, name, rural, amount)
            for code, name, amount in (
                ("gas_co", "Carbon monoxide, non-fossil", 0.0002),
                ("gas_methane", "Methane, non-fossil", 0.0001),
                ("gas_nmvoc", "NMVOC, non-methane volatile organic compounds", 0.00003),
                ("gas_nox", "Nitrogen oxides", 0.0004),
                ("gas_n2o", "Dinitrogen monoxide", 0.000005),
                ("gas_pt", "Platinum", 2e-9),
            )
        ),
    )
    furnace_rows = (
        supply("natural_gas", "natural gas, high pressure", 0.04, "cubic meter"),
        release("gas_n2o", "Dinitrogen monoxide", rural, 0.000003),
    )
    existing_gas_flows = {
        (row.endpoint.name, row.endpoint.unit, row.endpoint.categories): row.endpoint
        for row in (*gas_rows, *furnace_rows)
        if isinstance(row.endpoint, PaReceivingFlow)
    }
    gas_biosphere = tuple(
        existing_gas_flows.get(
            (name, unit, categories),
            PaReceivingFlow(
                ("biosphere", name + "::" + "/".join(categories)), name, unit, categories
            ),
        )
        for name, unit in (
            *(
                (f"{species}, {origin}", "kilogram")
                for species in ("Carbon dioxide", "Carbon monoxide", "Methane")
                for origin in ("fossil", "non-fossil")
            ),
            ("NMVOC, non-methane volatile organic compounds", "kilogram"),
            ("Particulate Matter, < 2.5 um", "kilogram"),
            ("Nitrogen oxides", "kilogram"),
            ("Dinitrogen monoxide", "kilogram"),
            ("Platinum", "kilogram"),
            ("Sulfur dioxide", "kilogram"),
            ("Ammonia", "kilogram"),
            ("Water", "cubic meter"),
        )
        for categories in (rural, urban)
    ) + (
        PaReceivingFlow(
            ("biosphere", "oxygen"), "Oxygen", "kilogram", ("natural resource", "in air")
        ),
    )
    original = pa_receiving_reference(database)
    return PaReceivingSources(
        background_scenario=background_scenario,
        provider_anchor_year=year,
        general_background_database=database,
        general_background_anchor_year=2025 if year <= 2025 else year,
        hwi=PaReceivingOperation(original.hwi, -3.0, hwi_rows),
        fixed=PaReceivingOperation(
            _activity(
                database, "fixed", "process-specific burdens, hazardous waste incineration plant"
            ),
            4.0,
            fixed_rows,
        ),
        ash=PaReceivingOperation(original.ash, -5.0, ash_rows),
        reheat_product=heat,
        pam=_activity(database, "pam", "polyacrylamide"),
        gas=PaReceivingGasSources(
            chp=PaReceivingOperation(original.gas.chp.activity, 3.0, gas_rows),
            furnace=PaReceivingOperation(original.gas.furnace.activity, 2.0, furnace_rows),
            biosphere=gas_biosphere,
        ),
        water=_water_records(database, frontier)[1],
        landfill=_landfill_records(database, frontier)[1],
        recycling=_recycling_records(database, frontier)[1],
        municipal=_municipal_records(database, frontier)[1],
        spinning_providers={
            grade: _activity(frontier, f"{grade}_conversion", "polyamide fibre conversion")
            for grade in ("pa6", "pa66")
        },
        added_biosphere=tuple(
            PaReceivingFlow(("biosphere", code), name, unit, categories)
            for code, name, unit, categories in (
                (
                    "spinning_nmvoc",
                    "NMVOC, non-methane volatile organic compounds",
                    "kilogram",
                    ("air",),
                ),
                ("co2_bio", "Carbon dioxide, non-fossil", "kilogram", urban),
                ("co_bio", "Carbon monoxide, non-fossil", "kilogram", urban),
                ("water_air", "Water", "cubic meter", ("air",)),
                ("water_final", "Water", "cubic meter", ("water",)),
                ("sodium_air", "Sodium I", "kilogram", urban),
                ("sodium_water", "Sodium I", "kilogram", surface),
                ("chloride", "Chloride", "kilogram", surface),
                ("nitrogen", "Nitrogen", "kilogram", ("air",)),
                ("heat_air", "Heat, waste", "megajoule", urban),
                ("heat_water", "Heat, waste", "megajoule", surface),
                ("titanium_surface", "Titanium ion", "kilogram", surface),
                ("titanium_later", "Titanium ion", "kilogram", later),
            )
        ),
    )
