class DocumentValidationError(ValueError):
    """The uploaded document cannot be processed safely."""


class PipelineUnavailableError(RuntimeError):
    """The configured inference backend is not ready."""
