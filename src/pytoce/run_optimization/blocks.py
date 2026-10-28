"""Names accepted by the public optimization workflow."""

from typing import Literal, TypeAlias

OptimizationBlock: TypeAlias = Literal[
    "demand_lifetime_frontier",
    "circularity_feedstock_anchor",
    "integrated_frontier",
]

OPTIMIZATION_BLOCK_ORDER: tuple[OptimizationBlock, ...] = (
    "demand_lifetime_frontier",
    "circularity_feedstock_anchor",
    "integrated_frontier",
)
