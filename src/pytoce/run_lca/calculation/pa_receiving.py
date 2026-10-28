"""Actual PA fiber quantities before receiving, interpolation and attribution."""

from dataclasses import dataclass
from typing import cast

from ...prepare_backgrounds.preparation.providers.material_carbon import (
    PrincipalCarbonOriginFractions,
)
from ...prepare_backgrounds.preparation.providers.pa_receiving_records import (
    PaReceivingActivity,
    PaReceivingFlow,
)
from ...prepare_backgrounds.preparation.providers.recipe_context import _scaled_native_exchange


@dataclass(frozen=True, slots=True)
class PaReceivingAmount:
    """One actual PA receiving demand or release, in the endpoint's native unit.

    ``function`` identifies the receiving operation for contribution aggregation.
    ``source`` identifies the equation or coefficient source.
    The enclosing event supplies activity year, world region and source anchor.

    Amounts already include the actual load and source-production normalization.
    Native activity keys identify activities; their reference-product keys are
    resolved at the inventory boundary. Metadata preserves provenance and any
    uncertainty scaled to this actual amount. The complete mixed load is evaluated first;
    principal-carbon fractions then attribute the resulting quantities.
    """

    function: str
    source: PaReceivingActivity
    endpoint: PaReceivingActivity | PaReceivingFlow
    amount: float
    metadata: dict[str, object]
    principal_carbon_origin_fractions: PrincipalCarbonOriginFractions | None = None

    def scaled(self, multiplier: float) -> "PaReceivingAmount":
        """Apply a positive load or allocation multiplier to this native-unit amount.

        The existing native-exchange owner scales distribution parameters with
        the quantity, including explicit zero coefficients. Carbon-origin
        fractions and source provenance remain unchanged.
        """
        row = _scaled_native_exchange(
            {
                **self.metadata,
                "amount": self.amount,
                "input": self.endpoint.key,
                "type": "technosphere"
                if isinstance(self.endpoint, PaReceivingActivity)
                else "biosphere",
            },
            multiplier,
        )
        amount = float(cast(float, row.pop("amount")))
        row.pop("input")
        row.pop("type")
        return PaReceivingAmount(
            self.function,
            self.source,
            self.endpoint,
            amount,
            row,
            self.principal_carbon_origin_fractions,
        )


@dataclass(frozen=True, slots=True)
class PaSpinningEvent:
    """One unsplit PA6 or PA66 manufacturing event at its activity year and world region.

    All scalar masses and the three principal-exit vectors are kg. The finished
    principal carbon vector is kg C. Vector order is fossil, biogenic and
    atmospheric DAC. Gross and recovered principal come directly from MFA
    realization, before carbon-origin attribution of LCA provider demands.

    Embedded finish is an inherited dose.
    Its fixed retained composition is supplied by the finish owner. The incoming
    dose has already undergone the actual recycling wash once; maintenance dose
    and principal exits have already undergone the deposit partition.
    Receiving uses these quantities directly. Newly applied finish is
    separate and follows finished principal exactly once.
    """

    year: int
    region: str
    constituent: str
    finished_principal_kg: float
    gross_principal_kg: float
    recovered_principal_kg: float
    embedded_finish_dose_kg: float
    maintenance_principal_by_origin_kg: tuple[float, float, float]
    maintenance_finish_dose_kg: float
    finished_principal_carbon_by_origin_kg: tuple[float, float, float]
    caprolactam_air_by_origin_kg: tuple[float, float, float]
    caprolactam_water_by_origin_kg: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PaRecyclingEvent:
    """One incoming PA fiber mixture and its actual recovered and airborne exits.

    Quantities are kg; principal vectors follow fossil, biogenic and atmospheric
    DAC order. The embedded dose multiplies the grade's retained finish vector,
    separately from the fresh surface dose on incoming principal. The airborne
    dose already includes the unwashed surface remainder and inherited finish.
    Washing, recovery and loss allocation belong to their material-balance calculations;
    receiving uses the dust's allocated finish quantity.
    """

    year: int
    region: str
    constituent: str
    principal_by_origin_kg: tuple[float, float, float]
    embedded_finish_dose_kg: float
    recovered_principal_kg: float
    airborne_principal_by_origin_kg: tuple[float, float, float]
    airborne_finish_dose_kg: float


@dataclass(frozen=True, slots=True)
class PaDisposalEvent:
    """One actual PA end-of-life or dry recycling-reject receiving mixture.

    All masses are kg; principal vectors follow fossil, biogenic and atmospheric
    DAC order. Total finish dose multiplies the original retained compound
    vector. End-of-life fiber includes its surface dose and principal-basis
    moisture regain; dry washed rejects include only their actual outgoing
    embedded dose and their dry mass. Mismanaged fate shares are already
    applied once to every field. Dump-class and temporal shares belong to the
    later selected receiving source, independently of regional material custody.
    """

    year: int
    region: str
    constituent: str
    demand_type: str
    principal_by_origin_kg: tuple[float, float, float]
    total_finish_dose_kg: float
    water_kg: float
