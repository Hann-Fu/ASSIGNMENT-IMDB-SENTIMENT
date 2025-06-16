import pytest
from utils.token_check import count_tokens


def test_count_tokens_basic():
    """Test basic token counting"""
    result = count_tokens("Hello world")
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_empty():
    """Test empty string"""
    result = count_tokens("")
    assert result == 0


def test_count_tokens_long_text():
    """Test longer text has more tokens"""
    short = count_tokens("Hi")
    long = count_tokens("This is a much longer sentence with many more words")
    assert long > short


def test_count_tokens_special_characters():
    """
    Test token counting with special characters.

    Args:
        None

    Returns:
        None
    """
    text = "Hello! @#$%^&*() 123"
    result = count_tokens(text)
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_different_model():
    """
    Test token counting with different model specification.

    Args:
        None

    Returns:
        None
    """
    text = "Hello world"
    result = count_tokens(text, model="gpt-4o-mini")
    assert isinstance(result, int)
    assert result > 0 