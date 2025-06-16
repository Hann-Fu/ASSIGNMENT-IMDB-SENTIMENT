from llm.classify_chain import sentiment_classify_chain

async def llm_sentiment_predict(text: str) -> str:
    """
    Classify the sentiment of a given text.
    Args:
        text: The text to classify.
    Returns:
        The sentiment of the text.
    """
    result = await sentiment_classify_chain.ainvoke({"text": text})
    return result.sentiment

