# `run_lca` architecture

`run_lca.py` validates the foreground scenarios, background scenarios, impact
methods, and upstream analysis years. It calculates missing combinations and
returns `RunLCAReport`.

## Owners

- `calculation/burdens.py` converts completed MFA flows into dated foreground
  demands.
- `calculation/custody.py` follows constituent carbon origins through transfers,
  manufacturing, recycling and surviving cohorts.
- `calculation/carbon.py` compares those MFA carbon quantities with the declared
  direct treatment carbon exchanges.
- `calculation/foreground.py` writes one foreground database with 12 regional roots.
- `calculation/timex.py` expands each regional root through dated providers and
  creates the annual inventory from 2019 through 2100.
- `calculation/providers.py` reads the prepared provider services for the
  selected background.
- `calculation/lcia.py` resolves registered EF, PB-LCIA or exact method
  selections, native units and factor identities, and characterizes ordinary
  regional inventories for annual and 2019-2100 impact scores. `timex.py` and the
  contribution calculation owners also apply factors when building provider
  responses and detailed contributions.
- `calculation/pa_event_receiving.py` dispatches actual material loads to the
  receiving owners. `pa_projection.py` evaluates each complete mixture at its
  source anchors and records temporal weights; `pa_inventory.py` applies those
  weights once while binding supplies and emissions to native products and
  elementary flows.
- `calculation/direct_contributions.py` characterizes actual receiving
  elementary flows for the requested methods and retains their stage
  attribution.
- `calculation/contribution_accumulation.py` owns matrix accumulation, retained
  detail and contribution state.
- `calculation/contribution_records.py` constructs contribution records and
  materializes them in bounded batches.
- `calculation/stage_scores.py` calculates annual and cumulative stage scores,
  regional/global sums and shares.
- `calculation/contribution_structure.py` builds the additive attribution
  ledger and non-additive paths and edges.
- `calculation/contributions.py` builds detailed regional tables and yields each
  completed product in order, releasing consumed state between products.
- `calculation/results.py` writes tables and the final manifest;
  `result_diagnostics.py` constructs provider lineage, fate allocations and their
  mass residuals for that writer.

## Constituent-carbon ownership

`custody.py` remains one cohesive owner despite exceeding 1,000 lines. Its
annual mixing graph, cohort recurrence, TPS constituent attribution and PA
finish tracking use the same transfers and carbon-origin arrays.
The foreground, burden and carbon-comparison owners consume those results;
provider selection and inventory characterization remain separate.

The PA finish calculation shares the graph and cohort state used by the
principal-carbon equations. Keeping them together makes their conservation
constraints visible in one place. The constituent-custody, carbon-consistency,
and foreground-custody tests cover these relationships.

## Inventory construction and impact assessment

The foreground represents regional flows. Provider geography remains a
separate dimension. Each region has one timeline and one expanded inventory for
each selected background. Global scores are calculated from the 12 regional
scores with stable summation.

Every stage uses its event year. One inventory is characterized by the
selected methods within one result product. A different method selection has
its own result product. Regional inventories follow ecoinvent cut-off accounting;
global results sum the regional environmental burdens.

## Storage and reuse

Results are stored below
`results/<foreground_scenario>/lca_results/<lca_selection>/`. The manifest is
written after all annual, cumulative, contribution, and diagnostic tables.
Completed results are reused directly. Refresh applies only to the selected LCA
result.
