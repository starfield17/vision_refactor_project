"""Project-wide custom exceptions."""


class VisionRefactorError(Exception):
    """Base exception for the refactor project."""


class ConfigError(VisionRefactorError):
    """Raised when config parsing or validation fails."""


class DataValidationError(VisionRefactorError):
    """Raised when data contract validation fails."""


class ModelExportError(VisionRefactorError):
    """Raised when model export fails."""


class TransportError(VisionRefactorError):
    """Raised when network transport fails."""
