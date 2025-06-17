import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in the text.
    
    Args:
        text (str): The text to count tokens for.
        model (str): The model to use for tokenization.
        
    Returns:
        int: The number of tokens in the text.
        
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))






