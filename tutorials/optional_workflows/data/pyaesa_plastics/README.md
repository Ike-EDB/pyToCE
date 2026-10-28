# Mapping to the 12 world regions for the pyaesa tutorial

This tutorial data folder provides two pyToCE aggregation files:

- `agg_reg_h12_plast.csv` (regional mapping)
- `agg_sec_h12_plast.csv` (sector mapping)

`agg_reg_h12_plast.csv` maps EXIOBASE 3 regions to the 12 world regions used by pyToCE for
1995–2024, including the nowcasted 2023 and 2024 tables. This data coverage is
distinct from the 1995–2022 regression and historical-structure sampling windows
used for prospective allocation.

`agg_sec_h12_plast.csv` keeps sectors unchanged except for the 4 plastic-related
products that are collapsed to the aggregate sector `Plastics`:

- `Plastics, basic`
- `Secondary plastic for treatment, Re-processing of secondary plastic into new plastic`
- `Plastic waste for treatment: incineration`
- `Plastic waste for treatment: landfill`

Both files are prepared in this tutorial folder and are copied automatically into the
active pyaesa workspace before running the MRIO steps:

- `{PYAESA_WORKSPACE}/data_raw/mrio/exiobase_3/aggregation/agg_reg_h12_plast.csv`
- `{PYAESA_WORKSPACE}/data_raw/mrio/exiobase_3/aggregation/pxp/agg_sec_h12_plast.csv`

How aggregation is done:
- Region mapping: native EXIOBASE countries are assigned directly to their matching pyToCE region with weight 1.
- For Rest-of-World regions, annual shares are computed from the World Bank GDP PPP series (`NY.GDP.MKTP.PP.KD`, constant 2017) and normalized to sum to one across destinations in the 12 world regions for each original region and year.
- Sector mapping: all sectors pass through unchanged except the four plastic rows that are collapsed into `Plastics`.
- pyaesa applies both mappings via `process_mrio` with `agg_reg=True`, `agg_sec=True`, `agg_version='h12_plast'`.

This use of GDP follows the proportional regional disaggregation described by
Wenz et al. (2015). The CSV files list assignments from EXIOBASE regions to the 12 world regions and annual
weights. GDP measures economic size and supplies the regional disaggregation weights. The article Methodological appendix describes the regional
mapping approach.

The custom EF files are created from pyaesa's installed GWP100 and EF 3.1 files
when you run the tutorial.

The World Bank input is the WDI series `NY.GDP.MKTP.PP.KD` used to construct this mapping.

The mapping is an adaptation of World Development Indicators data and is
distributed under CC BY 4.0. The World Bank dataset terms are available at
https://datacatalog.worldbank.org/public-licenses. The package code remains
BSD-3-Clause licensed. Please cite:

Wenz, L., Willner, S. N., Radebach, A., Bierkandt, R., Steckel, J. C., and
Levermann, A. (2015). Regional and sectoral disaggregation of multi-regional
input-output tables: a flexible algorithm. *Economic Systems Research*, 27(2),
194-212. https://doi.org/10.1080/09535314.2014.987731

World Bank. World Development Indicators, GDP PPP in constant 2017
international dollars. https://datacatalog.worldbank.org/search/dataset/0037712/world-development-indicators
