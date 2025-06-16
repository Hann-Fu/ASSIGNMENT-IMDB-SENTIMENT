import sys
import os
import pytest
# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_text():
    """
    Fixture providing sample text for testing.

    Returns:
        str: A sample text string for sentiment analysis.
    """
    return "This is a great product, I love it!"

@pytest.fixture
def sample_negative_text():
    """
    Fixture providing sample negative text for testing.

    Returns:
        str: A sample negative text string for sentiment analysis.
    """
    return "This product is terrible, I hate it!"

@pytest.fixture
def sample_empty_text():
    """
    Fixture providing empty text for testing error cases.

    Returns:
        str: An empty text string.
    """
    return ""

@pytest.fixture
def sample_version():
    """
    Fixture providing sample version for testing.

    Returns:
        str: A sample version string.
    """
    return "1.0.0" 