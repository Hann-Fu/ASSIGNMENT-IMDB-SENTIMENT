from ml.tfidfrl.preprocessing import preprocessing_pipeline
from loguru import logger
from utils.load_model_tools import load_baseline_models


_model, _vectorizer = load_baseline_models()

def rl_model_sentiment_predict(text: str) -> float:
    """
    Predicts the sentiment of a given text.

    Args:
        text (str): The input text for sentiment analysis.

    Returns:
        float: Positive probability.
    """

    try:
        cleaned_text = preprocessing_pipeline(text)
        vectorized_text = _vectorizer.transform([cleaned_text])
        probability = _model.predict_proba(vectorized_text)[:,1]

        return float(probability[0])
    
    except (FileNotFoundError, MemoryError, ValueError) as e:
        logger.error(f"Error in sentiment prediction: {e}")
        raise RuntimeError(f"Error in sentiment prediction: {e}")

    except Exception as e:
        logger.error(f"Unexpected error in sentiment prediction: {e}")
        raise RuntimeError(f"Error in sentiment prediction: {e}")

try:
    logger.info("Warming up TfidfRL model...")
    rl_model_sentiment_predict("dummy text for warm-up")
    logger.info("TfidfRL model warmed up successfully.")
except Exception as e:
    logger.warning(f"TfidfRL model warm-up failed: {e}. The first prediction may be slow.")