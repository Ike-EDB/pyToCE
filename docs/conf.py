"""Sphinx configuration for pyToCE documentation."""

import os
from importlib.metadata import version as package_version
from pathlib import Path

import pypandoc

# Put Pandoc on PATH for nbsphinx and nbconvert subprocess calls.
pandoc_directory = Path(pypandoc.get_pandoc_path()).parent
existing_path = os.environ.get("PATH", "")
os.environ["PATH"] = str(pandoc_directory) + os.pathsep + existing_path

project = "pyToCE"
author = "Erwan Ike de Bantel"
copyright = "2026, Erwan Ike de Bantel"
release = package_version("pytoce")
version = release

extensions = [
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
autodoc_typehints = "signature"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
exclude_patterns = [
    "_build",
    ".ipynb_checkpoints",
    "tutorials/optional_workflows/data/pyaesa_plastics/**",
]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/pytoce-logo.svg"
html_css_files = ["layout.css"]

nbsphinx_execute = "never"
nbsphinx_allow_errors = False
suppress_warnings = ["config.cache"]

myst_enable_extensions = [
    "colon_fence",
]

nitpicky = True
nitpick_ignore = [
    ("py:class", "collections.abc.Sequence"),
    ("py:class", "pathlib.Path"),
    # Keep return annotations visible in public function documentation.
    ("py:class", "pytoce.set_workspace.set_workspace.WorkspaceReport"),
    ("py:class", "pytoce.download_data.download_data.DownloadDataReport"),
    ("py:class", "pytoce.prepare_backgrounds.prepare_backgrounds.PrepareBackgroundsReport"),
    ("py:class", "pytoce.run_mfa.run_mfa.RunMFAReport"),
    ("py:class", "pytoce.run_lca.run_lca.RunLCAReport"),
    ("py:class", "pytoce.run_asr.run_asr.RunASRReport"),
    ("py:class", "pytoce.run_optimization.run_optimization.RunOptimizationReport"),
]
