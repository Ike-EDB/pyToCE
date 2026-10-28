# `download_data` architecture

`download_data.py` validates the request, downloads public sources and registers licensed inputs, and
returns `DownloadDataReport`.

`sources/catalogue.py` reads the source catalog and defines the report item.
`sources/network.py` downloads open data. `sources/ecoinvent.py` verifies the
licensed ecoinvent 3.12 EcoSpold and LCIA archives supplied by the user and
extracts the required LCIA workbook.

`sources/ecoinvent_materials.py` derives and publishes the plastics recipe used by MFA
and LCA. It keeps material requirements and recovery
yields distinct from allocated material inputs and the quantities used to scale
processing inventories in LCA.
Source selectors, XML interpretation, typed recipe records, derivation equations
and product identity share this owner so their scientific bases remain together.
A completed recipe is immutable and reused from its completion manifest.

Each source has its own directory, guide, and completion manifest. A completed
manifest identifies a reusable source. With `refresh=True`, open
sources are downloaded to staging and replace only the product owned by that
source after verification.

The user places licensed ecoinvent archives with the documented filenames in
`data_raw/ecoinvent/` before calling the workflow. Network access is tested at
the external transport boundary; ordinary tests use local source files.
