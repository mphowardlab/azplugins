# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021, Auburn University
# This file is part of the azplugins project, released under the Modified BSD License.

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'azplugins'
copyright = '2018-2021, Auburn University'
author = 'Michael P. Howard'
version = '0.10.2'
release = '0.10.2'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'nbsphinx'
]

templates_path = ['_templates']

exclude_patterns = ['_build', '**.ipynb_checkpoints']

# ---Options for jupiter notebooks with nbsphinx -----------------------------

nbsphinx_execute = 'never'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = []

# -- Options for autodoc & autosummary ---------------------------------------

autodoc_default_options = {
    'inherited-members': False
}

autodoc_mock_imports = ['azplugins._azplugins']

autosummary_generate = False

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'hoomd': ('https://hoomd-blue.readthedocs.io/en/stable', None)
}
