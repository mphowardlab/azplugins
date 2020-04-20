# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'azplugins'
copyright = '2018-2020, Michael P. Howard'
author = 'Michael P. Howard'
version = '0.9.2'
release = '0.9.2'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo'
]

templates_path = ['_templates']

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = []

# -- Options for autodoc & autosummary ---------------------------------------

autodoc_default_options = {
    'inherited-members': True,
    'show-inheritance': True
}

autodoc_mock_imports = ['hoomd']

autosummary_generate = True

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'hoomd': ('https://hoomd-blue.readthedocs.io/en/stable', None)
}
