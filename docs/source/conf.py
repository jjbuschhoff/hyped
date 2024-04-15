# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Include package path
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))


# import all modules
import hyped
import hyped.data.graph
import hyped.data.pipe
import hyped.data.processors.base
import hyped.data.processors.features.filter
import hyped.data.processors.features.flatten
import hyped.data.processors.features.format
import hyped.data.processors.sequence.apply_mask
import hyped.data.processors.sequence.chunk
import hyped.data.processors.sequence.extend
import hyped.data.processors.sequence.filter
import hyped.data.processors.sequence.join_str_seq
import hyped.data.processors.spans.apply_idx_spans
import hyped.data.processors.spans.common
import hyped.data.processors.spans.from_bio
import hyped.data.processors.spans.from_word_ids
import hyped.data.processors.spans.idx_spans
import hyped.data.processors.spans.loc_to_glob
import hyped.data.processors.spans.overlaps
import hyped.data.processors.statistics.base
import hyped.data.processors.statistics.report
import hyped.data.processors.statistics.sequence.disc_seq_val_hist
import hyped.data.processors.statistics.sequence.seq_len_hist
import hyped.data.processors.statistics.sequence.seq_val_hist
import hyped.data.processors.statistics.value.disc_hist
import hyped.data.processors.statistics.value.hist
import hyped.data.processors.statistics.value.mean_and_std
import hyped.data.processors.taggers.bio
import hyped.data.processors.taggers.relex
import hyped.data.processors.tokenizers.hf

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "hyped"
copyright = "2024, open-hyped"
author = "open-hyped"
version = hyped.__version__
release = hyped.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]

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
