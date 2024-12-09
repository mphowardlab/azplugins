# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2024, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Sphinx configuration."""

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import os

project = "azplugins"
year = datetime.date.today().year
copyright = f"2018-2020, Michael P. Howard. 2021-{year}, Auburn University."
author = "Michael P. Howard"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

default_role = "any"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "sphinx.ext.mathjax",
]

extensions += ["sphinx.ext.autodoc", "sphinx.ext.autosummary"]
autodoc_docstring_signature = True
autodoc_typehints_format = "short"
autodoc_mock_imports = ["hoomd.azplugins._azplugins"]
if os.getenv("READTHEDOCS"):
    autodoc_mock_imports += [
        "hoomd._hoomd",
        "hoomd.version_config",
        "hoomd.hpmc._hpmc",
        "hoomd.md._md",
        "hoomd.mpcd._mpcd",
    ]

extensions += ["sphinx.ext.napoleon"]
napoleon_include_special_with_doc = True

extensions += ["sphinx.ext.intersphinx"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "gsd": ("https://gsd.readthedocs.io/en/stable/", None),
    "hoomd": ("https://hoomd-blue.readthedocs.io/en/stable/", None),
}

extensions += ["sphinx.ext.todo"]
todo_include_todos = False

extensions += ["sphinx_copybutton"]
copybutton_prompt_text = "$ "
copybutton_remove_prompts = True
copybutton_line_continuation_character = "\\"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
# html_logo = "hoomdblue-logo-vertical.svg"
html_theme_options = {
    # "sidebar_hide_name": True,
    "top_of_page_buttons": [],
    "navigation_with_keys": True,
    "dark_css_variables": {
        "color-brand-primary": "#5187b2",
        "color-brand-content": "#5187b2",
    },
    "light_css_variables": {
        "color-brand-primary": "#406a8c",
        "color-brand-content": "#406a8c",
    },
}
# html_favicon = "hoomdblue-logo-favicon.svg"

pygments_style = "friendly"
pygments_dark_style = "native"
