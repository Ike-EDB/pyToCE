"""Calculate PA finish and caprolactam loads entering wastewater treatment.

Manufacturing and recycling supply their own collected finish, volume and
water. This calculation describes the nonaqueous inlet; the separate
Swiss wastewater inventories cover collection, biology, gas, sludge and ash.
No donor sewage carbon or second purchase of application water is added.
"""

from collections.abc import Mapping
from math import fsum

from .leachate_load import _CarbonOrigin, _LeachateCarbon, _LeachateLoad
from .pa_finish_material import _pa_finish_load, _PAFinishLoad


def _pa_finish_water_load(
    finish: _PAFinishLoad,
    *,
    caprolactam_by_origin_kg: Mapping[_CarbonOrigin, float],
) -> _LeachateLoad:
    """Keep finish moieties and lost principal separate in one actual water load.

    All masses share the caller's reference. Manufacturing supplies
    its collected seven-component finish and PA6 caprolactam by actual origin;
    they enter one wastewater calculation with carbon origins tracked separately.
    The selected recycling wash supplies removed finish.
    Suspended and emulsified material follows the adopted municipal
    co-treatment model.

    Principal caprolactam uses the package's nominal C6H11NO mass convention
    (72 C + 11 H + 14 N + 16 O = 113), preserving PA carbon custody. Auxiliary
    molecules retain their distinct CIAAW reference. The CIAAW caprolactam
    ThOD used by the manufacturing COD partition remains an analytical
    descriptor; it must not independently reassign principal carbon here.
    """
    caprolactam_kg = fsum(caprolactam_by_origin_kg.values())
    elements = dict(finish.elements_kg)
    # Ring opening to PA6 preserves its C6H11NO repeat composition. These
    # nominal atomic masses are the same principal 72/113 convention used by
    # material_carbon.py.
    for element, mass in {"C": 72, "H": 11, "N": 14, "O": 16}.items():
        elements[element] += caprolactam_kg * mass / 113
    carbon_pools = []
    for name, amount in finish.compounds_kg.items():
        component = _pa_finish_load({name: amount})
        carbon_pools.extend(
            (
                _LeachateCarbon(name, False, "fossil", component.fossil_carbon_kg),
                _LeachateCarbon(name, False, "biogenic", component.biogenic_carbon_kg),
            )
        )
    carbon_pools.extend(
        _LeachateCarbon("caprolactam", True, origin, amount * 72 / 113)
        for origin, amount in caprolactam_by_origin_kg.items()
    )
    carbon = fsum(pool.carbon_kg for pool in carbon_pools)
    # C6H11NO + 7.5 O2 -> 6 CO2 + NH3 + 4 H2O. Match the manufacturing owner's CIAAW
    # oxygen-equivalent reference (g/mol).
    caprolactam_thod = 7.5 * (2 * 15.999) / (6 * 12.011 + 11 * 1.008 + 14.007 + 15.999)
    return _LeachateLoad(
        elements_kg=elements,
        carbon=tuple(carbon_pools),
        # Doka Wastewater2021 WWcalc B103/B104 organic H/C and O/C are
        # empirical descriptors, separate from the inlet H and O.
        organic_hydrogen_kg=0.1225 * carbon,
        organic_oxygen_kg=0.64 * carbon,
        toc_kg=carbon,
        # Doka2021 section 19.1.2: selected Swiss residential DOC/TOC.
        # The original workbook's BOD5/TOC is a separate raw-inlet indicator.
        # These are source-inlet indicators.
        doc_kg=0.68 * carbon,
        cod_kg_o2=finish.thod_kg_o2 + caprolactam_kg * caprolactam_thod,
        bod5_kg_o2=1.7586552173908316 * carbon,
    )
