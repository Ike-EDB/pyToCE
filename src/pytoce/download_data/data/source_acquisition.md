# Sources used by download_data

`download_data` downloads or verifies these sources:

- REMIND-MFA 0.2.0 at commit
  `bf95c898e0f6f363196bccb62d34e02d82e4726b` under GPL-3.0-only;
- REMIND-MFA input data 1.0.0, revision
  `rev1.0.0_h12_1584d780_mfa`, DOI
  [10.5281/zenodo.18019915](https://doi.org/10.5281/zenodo.18019915),
  under CC BY-SA 4.0;
- AESA (PBs-LCIA) 0.9.2 from the `ei_312` branch at commit
  `00003e27a04bfc5662bf561cf7a092ba5324d67c` under BSD-3-Clause;
- the user-supplied licensed ecoinvent 3.12 Allocation, cut-off by
  classification EcoSpold 2 archive; and
- the user-supplied licensed ecoinvent 3.12 LCIA implementation archive.

Source archives and licensed ecoinvent inventories remain in the workspace. Packaged REMIND-MFA-derived default
inputs retain their declared CC BY-SA 4.0 license and provenance. `download_data`
verifies the licensed archives and extracts the LCIA workbook. It also reads
35 selected EcoSpold records to derive the local plastics recipe
at `data_processed/ecoinvent/material_recipe.json`. The recipe contains three
granulate monomer requirements, four hydrothermal products, seven cracker
products, 16 manufacturing input/output bases, 11 mechanical-recovery
declarations and eight recycling-operation normalization bases. Its schema,
source hashes, coefficients and carbon fractions stay with the licensed
workspace. Manufacturing records keep resin input and finished polymer output
distinct from native waste treatment quantities. Mechanical recovery reports
polymer recovered for reuse, polymer lost to water or air, and polymer sent to
managed waste treatment, per kilogram of incoming dry polymer. The quantities
used to scale native recycling inventories retain their allocation. Recovery yields
follow the separate material balance. The packaged schema defines the input structure.

An existing source registration without this derived product causes its first
creation, with overall status `updated` and source statuses `reused`.
`download_data(refresh=True)` downloads public sources again; a completed material recipe
remains unchanged and is reused from its completion manifest.
