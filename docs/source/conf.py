# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Include package path
import os
import sys

sys.path.insert(0, os.path.abspath("../../src/hyped"))


# import all modules
import hyped
import hyped.__version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "hyped"
copyright = "2024, open-hyped"
author = "open-hyped"
version = hyped.__version__.__version__
release = hyped.__version__.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",  # Set this to your Google Analytics tracking ID
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#007bff",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": False,
}
