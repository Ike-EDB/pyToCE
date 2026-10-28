# REMIND-MFA adapter architecture

This directory owns package integration with REMIND-MFA. `execution.py` admits v0.2.0 at commit
`bf95c898e0f6f363196bccb62d34e02d82e4726b`, data archive
`1.0.0 / rev1.0.0_h12_1584d780_mfa`, and the public stock/trade APIs from REMIND-MFA 0.2.0.

The session reconstructs historical cohorts and resolves the default stock
target once. `translation.py` admits complete annual scenario updates and
normalizes required source arrays with explicit units, coordinates and order.
`process_inputs.py` derives final, intermediate/manufactured, primary and waste
trade from each candidate's actual demand and discarded material, using fresh
parameter arrays and the source balancing order. It admits operational rates
before the direct MFA calculation consumes them. After each native trade
balance it requires equal global imports and exports for each retained
year/element/material/good coordinate. Waste trade below the source's machine-epsilon
divisor uses a common power-of-two numerical unit for both sides of each
coordinate, invokes the same native maximum-balance method once, and returns
to Mt. This preserves regional proportions and exact zeros using the source balancing rule. A true zero-sided trade still fails
the unchanged balance check. Floating-point overflow is reported as a numerical failure. Intermediate and
manufactured trade are checked separately before aggregation.

For final, intermediate, manufactured and primary trade, the adapter applies
the source's import-demand cap to the live import array after extrapolation,
then balances both sides to the smaller global total. The adapter applies
this rule through the source's public array and trade methods.
Historical stocks, demand targets and waste treatment retain their source values. Each market's cap is independent; aggregate material
and receiving constraints still belong to the MFA model.

The execution request and candidate-session identity share the current adapter
method identifier. A saved parent's recorded adapter remains parent provenance;
new candidates use the current adapter method. Raw source identities remain part of the session key.

pyToCE cohort continuation, polymer movements, constituent material
calculations and final accumulation belong to `run_mfa/model`. Integration
changes require evidence from the specified source revision and public APIs,
and bounded comparisons.
