# pyToCE Documentation

**Theory of Change for circular economy.**

pyToCE is a Python package that operationalizes a Theory of Change-based
framework linking inherent circularity performance to absolute environmental
sustainability. Dynamic material flow analysis (MFA) translates circular economy
(CE) performance levels into annual material-flow trajectories. Time-explicit
prospective life cycle assessment (LCA) then quantifies the associated
environmental burdens. Finally, absolute environmental sustainability assessment
(AESA) compares these burdens with allocated carrying capacities. These can be
calculated with pyaesa through the tutorial workflow or supplied by the user.

The framework can be used in two directions: monitoring and target setting. For
monitoring, CE performance levels are first set, and the resulting environmental
burdens are calculated and compared with allocated carrying capacities. For
target setting, allocated carrying capacities are specified first, and the package
searches for combinations of CE performance levels whose environmental burdens
could meet these requirements. CE performance levels are therefore an input to
monitoring and an output of target setting.

pyToCE currently applies to the plastics sector across 12 world regions from
2019 to 2100. The model distinguishes 12 plastic families distributed among
seven end-use groups.

The documentation is organized for two audiences. The user documentation
explains how to install the package, prepare data, run MFA, LCA, monitoring, and target-setting
workflows, and cite data and methodological sources. The developer documentation
is separate and is intended for Python developers modifying package internals.

Start with {doc}`overview` for the package scope and installation. Use
{doc}`workflow_reference` for data sources, prerequisite functions, public
function maps, and study objective routes. Use {doc}`tutorial` for step-by-step
examples and {doc}`api` for the public function reference.

```{toctree}
:maxdepth: 2
:caption: User Documentation

overview
workflow_reference
tutorial
api
publications_and_resources
```

This section is for Python developers modifying package internals.

```{toctree}
:maxdepth: 1
:caption: Developer Documentation

contributing
architecture
```

The [Methodological Notes](publications_and_resources.md) page provides
the LCI workbook, citation guide, article, and Methodological appendix.
