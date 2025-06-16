import pickle
import joblib
from loguru import logger
from dotenv import load_dotenv

from vendor.hugging_face_download import download_bert_model_from_hub, download_tfidf_model_from_hub

load_dotenv()

def load_baseline_models() -> tuple:
    """
    Downloads (if needed) and loads the TF-IDF model and vectorizer from the Hugging Face cache.
    
    Returns:
        tuple: A tuple containing (model, vectorizer) if successful.
        
    Raises:
        RuntimeError: If models cannot be loaded using any available strategy.
    """
    # Download the models and get their cache paths
    model_path, vectorizer_path = download_tfidf_model_from_hub()
    
    def load_with_fallback(file_path: str, file_type: str):
        """Load file using multiple strategies with fallback options."""
        logger.info(f"Loading {file_type} from cache: {file_path}")
        
        strategies = [
            lambda p: pickle.load(open(p, 'rb')),
            lambda p: pickle.load(open(p, 'rb'), encoding='latin1'),
            lambda p: joblib.load(p),
            lambda p: pickle.load(open(p, 'rb'), encoding='bytes'),
        ]
        
        for i, load_func in enumerate(strategies, 1):
            try:
                return load_func(file_path)
            except Exception as e:
                if i == len(strategies):  # Last attempt
                    logger.error(f"Failed to load {file_type} from {file_path}")
                    raise RuntimeError(f"Failed to load {file_type} using all strategies")
                logger.warning(f"Loading strategy {i} for {file_type} failed: {e}")
    
    model = load_with_fallback(model_path, "model")
    vectorizer = load_with_fallback(vectorizer_path, "vectorizer")
    
    return model, vectorizer

def load_bert_model() -> tuple:
    """
    Downloads (if needed) and loads the DistilBERT model and tokenizer from the Hugging Face cache.
    
    Returns:
        tuple: A tuple containing (model, tokenizer) if successful.
        
    Raises:
        RuntimeError: If model cannot be loaded.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    # Force CPU usage
    torch.set_default_device('cpu')
    
    # Get the model repository ID
    model_repo_id = download_bert_model_from_hub()
    
    try:
        logger.info(f"Loading DistilBERT model from Hugging Face cache: {model_repo_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_repo_id,
            device_map="cpu",  # Force CPU usage
            torch_dtype=torch.float32  # Use float32 for CPU
        )
        
        # Ensure model is on CPU
        model = model.to('cpu')
        
        # Set model to evaluation mode
        model.eval()
        
        logger.info("DistilBERT model and tokenizer loaded successfully from cache")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load DistilBERT model from {model_repo_id}: {e}")
        raise RuntimeError(f"Failed to load DistilBERT model: {e}")

