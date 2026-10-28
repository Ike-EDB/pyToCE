# `run_mfa` architecture

`run_mfa.py` admits the selected foreground scenarios and material recipe, reuses
matching completed results, and returns `RunMFAReport` with one
`MFAResult` item per scenario. A batch shares one source and historical
preparation session.

## Owners

- `scenario/` admits the parameter catalog and scenario YAML, compiles annual
  inputs, validates controls, and binds their identities.
- `remind_mfa/` imports the public REMIND-MFA 0.2.0 APIs, reconstructs historical
  cohorts, resolves default stock targets and prepares candidate-dependent trade.
- `model/cohorts.py` continues surviving cohorts from their actual age and stock.
- `model/flows.py` calculates polymer movements and the regional treatment
  projection; `model/materials.py` allocates constituent treatment, mechanical
  recovery, paired primary trade and chemical monomer supply and demand.
- `model/stocks.py` accumulates disposal stocks and the separate nominal carbon
  account from the actual preceding closing state.
- `model/mfa.py` coordinates the common MFA calculation and its checks.
  One MFA state identity binds exact source/method, named array coordinates,
  movements, constituent realization and cohort state.
- `model/result_tables.py` formats annual outputs and factorized cohort tables.
- `model/sankey.py` formats the material-flow Sankey tables.
- `model/results.py` owns result identity, staging, refresh and publication.
- `model/result_inputs.py` reads completed package manifests and their recorded
  tables for LCA and optimization. Scientific admission belongs to the source,
  scenario and calculation owners; completed products are trusted.

## Material flow calculations

The source supplies historical stocks/cohorts and default stock/trade inputs.
pyToCE calculates its final flows directly, including the historical prefix
through 2019. Scenario changes start in 2020; existing cohorts retain their
material, region and age. Material shares apply to new inflow, and treatment
and trade controls apply in the year of the activity.

Polymer mass, gross chemical products/monomers, nominal REMIND element
partitions and constituent carbon have distinct meanings. Chemical feedstock
quantities describe monomer inputs; recycled resin quantities describe recovered
polymer. The nominal atmospheric account records signed carbon transfers.
Disposal accumulation starts from the source zero reference before 1950.
End-2100 stock remains in use. Provider selection, allocated inventory
coefficients and characterization belong to LCA.

Candidate evaluation returns the MFA inputs used by foreground construction.
It omits disposal/atmospheric accumulation, Sankey generation and ordinary
reconciliation, transition and waste-trade ledgers. Both calculation modes share
the material flow calculation and admission checks. Ordinary reporting also
checks the recurrence equations for accumulated stocks. Source material/element fractions
are admitted before projection, including at zero stock. Scenario compilation
retains the admitted material-share input precision. Cohort entry uses
those admitted shares, while annual stock balances remain in
`stock_balance.parquet`. Tests verify the deterministic cohort and nominal-element projections.

Optimization treatment shares use all posttrade collected constituent mass as
the denominator. Chemical eligibility constrains allocation within that amount.
Compiled virgin-feed controls select manufacturing routes; zero mass retains valid compiled controls.

The waste available after trade determines treatment quantities. Regional route
controls share one least-change projection onto that total. Nominal element
quantities use the resulting polymer route allocation. The source trade
calculation is retained.

## Cohort Stock Feasibility

The cohort calculation compares each requested stock target with the sum of
surviving existing cohorts. If their difference is negative only within a local
binary64 roundoff allowance, new inflow is zero. Existing cohorts and their
outflows remain unchanged, so actual closing stock can exceed the requested
target by that small amount. The ordinary cohort transition ledger reports
`stock_target_residual` (requested minus actual closing stock) and
`stock_target_roundoff_tolerance`, both in Mt; the residual is negative when
actual stock exceeds the target. Positive
new inflows are unchanged, and a negative difference outside the allowance is
rejected by the MFA. This rule applies from the first MFA calculation, including
linear references and demand repairs. Recycling and material-receiving admission
remain fully required.

For `n` existing-cohort closing terms in a cell, the allowance is
`gamma_(n+2)*abs(requested_stock) + gamma_(n+2)*abs(surviving_stock)`,
where `surviving_stock` is the already calculated sum of the nonnegative closing
cohorts, `gamma_k = k*u/(1-k*u)` and binary64 unit roundoff `u = eps/2`. The allowance covers local floating-point arithmetic roundoff.

## Storage and reuse

Matching completed results are reused from their completion manifests.