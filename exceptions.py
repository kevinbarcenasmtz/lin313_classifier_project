"""
Custom exceptions for the classification project.
"""


class ClassificationError(Exception):
    """Base exception for classification-related errors."""
    pass


class DataLoadingError(ClassificationError):
    """Raised when dataset loading fails."""
    pass


class ValidationError(ClassificationError):
    """Raised when data validation fails."""
    pass


class APIError(ClassificationError):
    """Raised when API calls fail after retries."""
    pass


class ParsingError(ClassificationError):
    """Raised when LLM response parsing fails."""
    pass

