# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# For autodocs
sys.path.insert(0, os.path.abspath('../../../niviz/'))

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'niviz'
copyright = '2020, Jerry Jeyachandra, Ben Selby'
author = 'Jerry Jeyachandra, Ben Selby'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx_autodoc_typehints',
    'sphinx_rtd_theme', 'sphinx.ext.intersphinx', 'sphinx_multiversion',
    'sphinx.ext.githubpages'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['../_templates']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# --- Multiversioning Options -----------------------------------------------
# See the following link for multiversion configuration:
# https://holzhaus.github.io/sphinx-multiversion/master/configuration.html#tag-branch-remote-whitelists

smv_branch_whitelist = r'^dev$'
smv_released_pattern = r'^tags/.*$'
smv_remote_whitelist = r'^origin$'

# --- Autodoc Options -------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True
}

# --- Autodoc Typehint Options --------

# FAILS with an import error on 'F' in pandas._typing
# set_type_checking_flag = True

# --- Intersphinx Options ----------------------------------------------------

intersphinx_mapping = {
    'nipype':
    ('https://nipype.readthedocs.io/en/latest/', None),
    'niworkflows': ('https://www.nipreps.org/niworkflows/', None)
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../_static']
