# Synthetic inventory for PHB municipal incineration

All quantities, reference production amounts, elemental compositions, supplier
shares and identifiers in `native.json` are synthetic. They reproduce the
selected ecoinvent record structure without distributing licensed data. The
original waste contains nitrogen, chlorine, sulfur and metals that are absent
from dry polyhydroxybutyrate (PHB). Its ash has a deliberately excessive, rounded
nitrate value to test the published limit for complete nitrogen release.

The test combines these records with the synthetic hazardous-waste incineration
inventory and solves the linked catalyst, polyacrylamide, water and ash treatment
inventories. Original and future sources have different negative reference
production amounts and supplier shares. The synthetic ash inventory includes
water to check that the PHB compiler retains each original supply once. This
test does not prescribe adding water to an ecoinvent inventory.
