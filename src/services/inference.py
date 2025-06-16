from ml.tfidfrl.model_predict import rl_model_sentiment_predict
from ml.distilbertmlp.model_predict import bert_model_sentiment_predict
from llm.classify_llm import llm_sentiment_predict
from utils.token_check import count_tokens
from loguru import logger
import asyncio
from openai import OpenAIError, RateLimitError
from exceptions.llm import LLMRateLimitError, LLMServiceError

def sentiment_inference_baseline(text: str, version: str) -> float:
    """
    Predict the sentiment of the text, return the sentiment and the probability of the sentiment.

    Args:
        text (str): The input text for sentiment analysis.

    Returns:
        float: Positive probability.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty or whitespace only")
    if version == "1.0.0":
        return rl_model_sentiment_predict(text)
    
    else:
        raise ValueError("Invalid version.")

def sentiment_inference_bert(text: str, version: str) -> float:
    """
    Predict the sentiment of the text using DistilBERT model, return the positive probability.

    Args:
        text (str): The input text for sentiment analysis.
        version (str): The version of the model.

    Returns:
        float: Positive probability.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty or whitespace only")
    if version == "1.0.0":
        return bert_model_sentiment_predict(text)
    else:
        raise ValueError("Invalid version.")
    


async def sentiment_inference_llm(text: str, version: str) -> str:
    """
    Predict the sentiment of the text using LLM, return the sentiment.

    Args:
        text (str): The input text for sentiment analysis.
        version (str): The version of the model.

    Returns:
        str: Sentiment("positive" or "negative").
    """
    try:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace only")
        if version == "1.0.0":
            # add a token check
            if count_tokens(text) > 900000:
                logger.warning(f"Input text is too long, auto truncate to 400000 characters. Token count: {count_tokens(text)}")
                text = text[:400000]

            result = await llm_sentiment_predict(text)
            return result
        else:
            raise ValueError("Invalid version.")
    
    except RateLimitError as e:
        logger.error(f"LLM rate limit exceeded: {e}")
        raise LLMRateLimitError(f"API rate limit exceeded: {e}")
    
    except (OpenAIError, asyncio.TimeoutError) as e:
        logger.error(f"LLM service error: {e}")
        raise LLMServiceError(f"LLM service temporarily unavailable: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error in LLM inference: {e}")
        raise LLMServiceError(f"LLM service error: {e}")
    