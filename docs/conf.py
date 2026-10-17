# Configuration file for the Sphinx documentation builder.

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

# Add yapocis to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "Yapocis"
copyright = "2011-2026, Sean D. True and Swapwizard LLC"
author = "Sean D. True"

# The docs are built with the package installed (see .readthedocs.yml)
try:
    release = package_version("yapocis")
except PackageNotFoundError:
    release = "unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# MyST-Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Templates path
templates_path = ["_templates"]

# Patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]

# Custom CSS
html_css_files = [
    "custom.css",
]

# RTD theme context
html_context = {
    "display_github": True,
    "github_user": "seantrue",
    "github_repo": "Yapocis3",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Options for LaTeX/PDF output --------------------------------------------

# xelatex handles the Unicode in the docs (box-drawing diagrams, arrows, check marks)
# that pdflatex drops. Text stays in Sphinx's default GNU FreeFont; code blocks use
# DejaVu Sans Mono, because FreeMono has no box-drawing characters.
latex_engine = "xelatex"
# Sphinx defaults to xindy with xelatex; Read the Docs' xindy lacks its language
# modules ("LATIN9-LANG.XDY has no value"), so build the index with makeindex.
latex_use_xindy = False
latex_elements = {
    "fontpkg": r"""
\setmainfont{FreeSerif}[Extension=.otf, UprightFont=*, ItalicFont=*Italic,
  BoldFont=*Bold, BoldItalicFont=*BoldItalic]
\setsansfont{FreeSans}[Extension=.otf, UprightFont=*, ItalicFont=*Oblique,
  BoldFont=*Bold, BoldItalicFont=*BoldOblique]
\IfFontExistsTF{DejaVuSansMono.ttf}{
  \setmonofont{DejaVuSansMono}[Extension=.ttf, UprightFont=*, BoldFont=*-Bold,
    ItalicFont=*-Oblique, BoldItalicFont=*-BoldOblique, Scale=MatchLowercase]
}{
  \setmonofont{FreeMono}[Extension=.otf, UprightFont=*, ItalicFont=*Oblique,
    BoldFont=*Bold, BoldItalicFont=*BoldOblique]
}
""",
}

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_typehints = "description"

# Optional backends that are not installed where the docs are built
autodoc_mock_imports = ["pycuda", "pymetallic"]

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
