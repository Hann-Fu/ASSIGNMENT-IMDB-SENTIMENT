class LLMRateLimitError(Exception):
    """Exception raised when API rate limit is exceeded."""
    pass

class LLMServiceError(Exception):
    """Exception raised for LLM service errors."""
    pass