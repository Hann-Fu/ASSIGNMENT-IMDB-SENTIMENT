from ml.distilbertmlp.preprocessing import preprocessing_pipeline_bert
from utils.load_model_tools import load_bert_model
from loguru import logger
import torch
from torch.nn.functional import softmax
from utils.load_model_tools import load_baseline_models

# Force CPU usage
torch.set_default_device('cpu')

_bert_model, _bert_tokenizer = load_bert_model()

def bert_model_sentiment_predict(text: str) -> float:
    """
    Predicts the sentiment of a given text using DistilBERT model.

    Args:
        text (str): The input text for sentiment analysis.

    Returns:
        float: Positive probability.
    """
    try:   
        # Preprocess the text
        cleaned_text = preprocessing_pipeline_bert(text)
        
        # Tokenize the text
        inputs = _bert_tokenizer(
            cleaned_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Ensure all inputs are on CPU
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = _bert_model(**inputs)
            logits = outputs.logits.to('cpu')  # Ensure output is on CPU
            
            # Apply softmax to get probabilities
            probabilities = softmax(logits, dim=-1)
            
            # Get positive probability (assuming label 1 is positive)
            positive_prob = probabilities[0][1].item()
            
        return float(positive_prob)
        
    except (FileNotFoundError, MemoryError, ValueError) as e:
        logger.error(f"Error in DistilBERT sentiment prediction: {e}")
        raise RuntimeError(f"Error in DistilBERT sentiment prediction: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected error in DistilBERT sentiment prediction: {e}")
        raise RuntimeError(f"Error in DistilBERT sentiment prediction: {e}")

try:
    logger.info("Warming up DistilBERT model...")
    bert_model_sentiment_predict("dummy text for warm-up")
    logger.info("DistilBERT model warmed up successfully.")
except Exception as e:
    logger.warning(f"DistilBERT model warm-up failed: {e}. The first prediction may be slow.")
