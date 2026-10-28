# `prepare_backgrounds` architecture

`prepare_backgrounds.py` validates the selected SSP2 background scenarios,
prepares missing products, and returns `PrepareBackgroundsReport`.

## Owners

- `preparation/ecoinvent.py` imports the verified ecoinvent 3.12 cut-off
  database with bw2io 0.9.17.
- `preparation/premise.py` prepares the selected REMIND SSP2 databases with
  premise 2.4.6. Each background scenario has separate databases and cache.
- `preparation/project_biosphere.py` registers the 16 EF v3.1 midpoint methods
  and the 10 selected AESA PB-LCIA methods against the native ecoinvent
  biosphere.
- `preparation/providers/` resolves native activities and compiles the dated
  provider services used by the foreground model. The catalog defines names and
  routes; the recipe modules own their inventory construction.
- `preparation/prepared_state/` owns shared preparation, one background
  scenario at a time, authority tables, reports, and final manifests.

## Background scenarios and provider inventories

General backgrounds use ecoinvent through 2025, interpolate to the matching
premise 2030 database during 2026--2029, and then use the matching premise
anchors through 2100. Premise 2020 and 2025 supply selected provider
technologies; their ordinary upstream supplies remain bound to ecoinvent.

Provider services use suitable native inventories and their linked supplies as
provided. A changed or added exchange requires a documented plastics-study
difference in material, functional unit, composition, carbon origin, or selected
technology and year. The recipe records that justified difference and preserves
unaffected native supplies once. Direct elementary exchanges use native ecoinvent
biosphere identities. Characterization retains the admitted LCIA factors.

The default selection contains the four REMIND SSP2 background scenarios.
Completed historical REMIND-EU products remain available for explicit reuse.

## Storage and reuse

Shared products are stored below
`data_processed/prepared_backgrounds/shared/`. Each background scenario is
stored below `data_processed/prepared_backgrounds/<background_scenario>/` and
owns its premise databases, provider tables, reports, and manifest.

The workflow writes its manifest after all required products are complete.
A completed manifest enables direct reuse. Preparation requires unused target
Brightway names. One operating-system lock serializes preparation within each
workspace. Completed backgrounds are immutable.
