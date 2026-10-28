"""Supply compound-22 ingredients, utilities and manufacturing air releases."""

from math import fsum, pi
from typing import cast

from .recipe_context import _RecipeContext


def _sbr_compounding_recipe(
    context: _RecipeContext, construction: dict[str, object]
) -> dict[str, object]:
    """Build the SBR manufacturing inventory per kg retained dry principal.

    TRC (1995) and EPA AP-42 define compound 22 as 1 kg dry SBR 1502 plus
    0.858 kg additives. The resin stage supplies the principal kg. The material
    recipe supplies the gross-feed/retained-principal ratio, including releases
    before trimming.

    Carli (2008) supplies 7.1 MJ electricity/kg gross compound for mixing,
    forming and hot-air curing. This complete operation provides the productive
    machinery reference. Its closed-circuit process water is retained within
    the supplied functions.

    Native RER pipe extrusion supplies the maintenance rate of 0.000143 kg
    lubricant/kg gross compound. Lubricant supply and CH waste-oil collection
    and incineration cover the whole consumption. Machinery lubricant is
    separate from compound oil and compressor lubricant.

    EPA mixing, extrusion and hot-air factors and TRC's extrusion metal vector
    supply air emissions. Named substances, unlisted gas and hot-air condensable
    particulate matter are separate, overlapping indicators; their sum is not
    mass loss. Auxiliary organic withdrawals determine retained product
    after mixer dust, extrusion dust and 5% finishing scrap.

    Mixer fan electricity uses EPA throughput and filter pressure, scaled by
    retained filled product. PET filter bags use Donaldson dimensions and weld
    allowances, with felt manufacture, fabrication, delivery and disposal.
    Native compressed-air supply covers pulse demand and compressor maintenance.
    A transferred whole-line factory supplies the capital function.

    Captured mixer powder receives dry-additive HWI, cemented-residue treatment
    and industrial collection. Fossil additives and tallow stearic acid retain
    their carbon origins. Filled finishing scrap receives the composed landfill
    chain and a separate 136 km initial road haul. Retained auxiliaries have
    separate use and recovery calculations.
    """
    basis = next(
        item
        for item in context.material_recipe.conversion
        if item.operation == str(construction["operation"])
    )
    feed = basis.principal_feed_kg_per_finished_kg
    ingredients = cast(list[dict[str, object]], construction["ingredients"])
    auxiliary_feed = feed * fsum(
        float(cast(float, item["amount_kg_per_gross_principal_kg"])) for item in ingredients
    )
    # Productive throughput includes gross principal plus all charged additives; the separate resin
    # service supplies the principal.
    gross_compound = feed * fsum(
        (
            1.0,
            *(float(cast(float, item["amount_kg_per_gross_principal_kg"])) for item in ingredients),
        )
    )
    withdrawals = cast(
        list[dict[str, object]], construction["auxiliary_organic_withdrawal_kg_per_gross_compound"]
    )
    organic_withdrawal = gross_compound * fsum(
        float(cast(float, row["amount"])) for row in withdrawals
    )
    # EPA compound-22 mixer dust is .00045 kg/kg gross compound, entirely from
    # dry additives in the declared reference. Extrusion PM is .0000266 kg/kg
    # gross compound; only its original additive share is withdrawn here.
    # Principal monomer/PM losses already determine feed at the material owner.
    auxiliary_before_trim = (
        auxiliary_feed - 0.00045 * gross_compound - 0.0000266 * auxiliary_feed - organic_withdrawal
    )
    # Carli's 5% final trim leaves .95 of those auxiliaries. The 1 kg retained
    # principal defines this service; P includes its actual retained additives.
    finished_compound = 1.0 + 0.95 * auxiliary_before_trim
    # EPA fabric-filter Eq.1.14 converts acfm * inches H2O * hours to kWh
    # with .000181, including its .65 combined fan/motor efficiency. The EPA
    # rubber-goods case uses 7.5 m3/s, 1000 h/year and 1,700,000 kg/year
    # finished compound. 60 converts seconds/minute; .3048 m/ft converts volume.
    # The admitted 9-inch loss is 3.5 fabric + 1.5 housing + 4 external duct.
    # One fan serves those losses in the separately selected collector proxy.
    fan_kwh = 0.000181 * (7.5 * 60 / 0.3048**3) * 9 * 1000 / 1_700_000 * finished_compound
    # EPA 1977 exhaust is 7.5*3600*1000/1700000 m3/kg finished compound.
    # EPA 1998 filter p.1-48 uses .002 standard m3 pulse air/m3 exhaust;
    # EPA 1992 plywood ACT p.5-9 defines that volume at 70 F/atmospheric pressure.
    # Convert by 273.15/((70-32)*5/9+273.15) to the declared dry 0 C/101325 Pa
    # convention supported by Voegtlin. Native Nm3 temperature is unspecified.
    pulse_air_m3 = 0.029485817473476825 * finished_compound
    lubricant_kg = 0.000143 * gross_compound
    # Donaldson's 2007 media table gives untreated 16 oz/yd2 PET felt.
    # F118107 transfers related-bag dimensions: 4.85-inch diameter, 77-inch
    # length and 7.62-inch flat width. Convert oz to kg, yards/inches to meters.
    # INDRO SIIC-M006 supplies the 15 mm body-weld lap; the same retained
    # allowance on the bottom join is the disclosed geometric transfer.
    felt_density = 16 * 0.028349523125 / 0.9144**2
    diameter, length, flat_width = (4.85 * 0.0254, 77 * 0.0254, 7.62 * 0.0254)
    seam = 0.015
    bag_mass = felt_density * (
        (2 * flat_width + seam) * (length + seam) + pi * (diameter / 2 + seam) ** 2
    )
    # EPA's 7.5 m3/s collector at 5 ft/min needs this total active cloth area;
    # Donaldson supplies 7.6 ft2 active area/bag. Select the three-year midpoint
    # of EPA's 2-4 year reference life; the rubber-goods case's 1.7 million
    # kg/year finished output allocates replacement to actual P.
    cloth_area = 7.5 / (5 * 0.3048 / 60)
    bag_active_area = 7.6 * 0.3048**2
    bag_kg = cloth_area / (bag_active_area * 3 * 1_700_000) * bag_mass * finished_compound
    bag = cast(dict[str, dict[str, object]], construction["replacement_bag"])
    # EPA's mixer emits .00045 kg powder/kg gross compound before its 99.3%
    # capture. Only the captured fraction enters the complete dry-dust HWI;
    # the remaining .7% already belongs to the named manufacturing air rows.
    dust_kg = 0.00045 * 0.993 * gross_compound
    dust = cast(dict[str, dict[str, object]], construction["captured_mixer_dust"])
    scrap = cast(dict[str, dict[str, object]], construction["filled_finishing_scrap"])
    # Carli's 5% final trim /95% retained output gives P/19 kg actual filled
    # cured scrap. Selective upstream losses already determined P above.
    scrap_kg = finished_compound / 19
    return {
        "components": [
            {
                **item,
                "amount": feed * float(cast(float, item["amount_kg_per_gross_principal_kg"])),
            }
            for item in ingredients
        ]
        + [
            {**scrap["treatment"], "amount": -scrap_kg},
            # Native plastic-market disposal transfer: 136 km initial road haul. kg/1000 converts to
            # metric tons.
            {**scrap["haul"], "amount": 0.136 * scrap_kg},
            {**dust["treatment"], "amount": -dust_kg},
            # The selected CH industrial-waste collection contributes road
            # .0258 and rail .0036 tkm/kg dust, before the direct HWI service.
            {**dust["collection_road"], "amount": 0.0258 * dust_kg},
            {**dust["collection_rail"], "amount": 0.0036 * dust_kg},
            {
                **cast(dict[str, object], construction["electricity"]),
                # 3.6 MJ/kWh is the unit conversion.
                "amount": 7.1 / 3.6 * gross_compound + fan_kwh,
            },
            {**cast(dict[str, object], construction["pulse_air"]), "amount": pulse_air_m3},
            {
                **cast(dict[str, object], construction["machinery_lubricant"]),
                "amount": lubricant_kg,
            },
            {
                **cast(dict[str, object], construction["spent_lubricant_treatment"]),
                # Negative waste demand against the native negative kg output.
                "amount": -lubricant_kg,
            },
            {
                **cast(dict[str, object], construction["spent_lubricant_lorry"]),
                # Ecoinvent CH waste-oil market rows 834544-834545: tkm/kg
                # collected oil. Its two treatment purchases are excluded.
                "amount": 0.02577 * lubricant_kg,
            },
            {
                **cast(dict[str, object], construction["spent_lubricant_rail"]),
                "amount": 0.00356 * lubricant_kg,
            },
            {
                **bag["fabrication"],
                # Native filter has .0525 kg PET in, .0025 kg offcut and
                # .05 kg net PET. The child retains its four operations and
                # signed offcut, with their original quantities/uncertainties.
                "amount": bag_kg / 0.05,
            },
            {
                **bag["delivery"],
                # Native filter-market freight is per .4 kg shipped filter.
                # Its five modes remain once; both donor filter purchases go.
                "amount": bag_kg / 0.4,
            },
            {**bag["needle_felt"], "amount": 1.05 * bag_kg},
            # Full native negative-production treatment receives spent PET;
            # fabrication offcuts have collection and disposal inside their
            # retained waste market.
            {**bag["spent_bag"], "amount": -bag_kg},
            {
                **bag["collection_road"],
                # CH PET-market collection is .02577/.00356 tkm per kg bag.
                "amount": 0.02577 * bag_kg,
            },
            {
                **bag["collection_rail"],
                "amount": 0.00356 * bag_kg,
            },
            {
                **cast(dict[str, object], construction["line_factory"]),
                # Ecoinvent's complete RER plastics factory supplies 27,000 t
                # annually for 50 years, including its 25-year machinery renewal.
                # Transfer the whole-line capital function once to actual SBR output.
                "amount": finished_compound / (27_000_000 * 50),
            },
        ],
        "biosphere_terms": [
            {
                **row,
                "amount": gross_compound * float(cast(float, row["amount_kg_per_gross_compound"])),
            }
            for row in cast(list[dict[str, object]], construction["air_emissions"])
        ],
    }
