class DocumentValidationError(ValueError):
    """The uploaded document cannot be processed safely."""


class PipelineUnavailableError(RuntimeError):
    """The configured inference backend is not ready."""


class ProductionExtractionError(RuntimeError):
    """A preloaded production model could not produce a validated extraction."""
