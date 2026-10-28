Tutorials
=========

The tutorials provide step-by-step examples for setting up a workspace,
preparing input data, selecting study objectives, and running monitoring or
target-setting workflows.

The documentation follows the repository tutorial layout: core workflow
notebooks and optional workflow notebooks.

Each notebook selects the workspace and can be opened independently once its
required inputs and parent results are complete.

For **monitoring**, follow Tutorials 1--4, supply allocated carrying capacities,
then follow :doc:`Tutorial 5m <tutorials/core_workflow/5m_run_asr>`.
Monitoring starts from user-defined CE performance trajectories and follows
the causal chain from CE performance to
material flows, environmental burdens, allocated carrying capacities, and
finally absolute environmental sustainability.

For **target setting**, follow Tutorials 1--4 for the starting foreground,
supply allocated carrying capacities, then follow
:doc:`Tutorial 5o <tutorials/core_workflow/5o_run_optimization>`.
Target setting starts from the allocated carrying capacities and then searches
for CE performance levels that allow the resulting environmental
burdens to comply with those allocated carrying capacities (when possible).

For both monitoring and target setting, allocated carrying capacities can be
calculated following the
:doc:`optional pyaesa tutorial <tutorials/optional_workflows/pyaesa_allocated_carrying_capacity>`
or supplied by the user. The pyaesa tutorial requires completed 2019 LCA results
via ``run_lca``.

.. toctree::
   :maxdepth: 2

   tutorials/core_workflow/index
   tutorials/optional_workflows/index

Methodological Notes
--------------------

The repository folder ``src/pytoce/set_workspace/data/methodological_notes/``
contains the methodological resources and recommended citation guide.
``set_workspace(...)`` copies these files into the active workspace under
``data_raw/methodological_notes/``. See :doc:`publications_and_resources` for the
article and Methodological appendix PDFs, LCI workbook, and citation guide.
