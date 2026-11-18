"""Exception types for the project.

Validation failures (config, data) are raised as these types so callers can
distinguish expected, well-described errors from unexpected crashes.
"""


class AxnerError(Exception):
    """Base class for all project errors."""


class ConfigError(AxnerError):
    """A configuration is missing fields or contains invalid values."""


class DataValidationError(AxnerError):
    """Dataset content does not match expectations (schema, IOB2, sizes)."""


class SamplingError(AxnerError):
    """A sampling request cannot be satisfied (e.g. more rows than available)."""


class ArtifactError(AxnerError):
    """An artifact directory/bundle is missing or inconsistent."""
