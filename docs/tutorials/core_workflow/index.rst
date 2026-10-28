Core Workflow
=============

Start with Tutorials 1--4 to download public inputs, prepare backgrounds, and calculate
material flows and environmental burdens. Then choose the notebook for your
study objective:

* **Monitoring:** supply allocated carrying capacities and follow
  :doc:`5m_run_asr` to assess user-defined CE performance trajectories.
* **Target setting:** supply allocated carrying capacities and follow
  :doc:`5o_run_optimization` to search for CE performance target levels.
  Running ``run_asr`` first is optional.

For both study objectives, calculate allocated carrying capacities following
the :doc:`optional pyaesa tutorial <../optional_workflows/pyaesa_allocated_carrying_capacity>`
or supply your own. The pyaesa tutorial requires completed 2019 LCA results.

For optimization controls, selection rules and result-table definitions,
see the :doc:`workflow reference </workflow_reference>`, particularly
"7. Run the optimization" and "Read the optimization tables".

.. toctree::
   :maxdepth: 1

   1_download_data
   2_prepare_backgrounds
   3_run_mfa
   4_run_lca
   5m_run_asr
   5o_run_optimization
