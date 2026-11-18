"""Austronesian cross-lingual NER research infrastructure.

Package layout follows the repository layout documented in README.md. Heavy
dependencies (torch, transformers) are imported lazily inside the functions
that need them, so analysis-only environments can import data and evaluation
modules without the training stack installed.
"""

__version__ = "0.1.0"
