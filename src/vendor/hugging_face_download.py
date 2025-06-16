from huggingface_hub import hf_hub_download
from loguru import logger
from dotenv import load_dotenv
import os
load_dotenv()

def download_bert_model_from_hub() -> str:
    """
    Returns the repository ID for the DistilBERT model.
    The transformers library will handle downloading and caching automatically.

    Returns:
        str: The repository ID of the model.
    """
    repo_id = os.getenv('BERT_MODEL_REPO_ID', 'protostarss/distilbert_imdb_full')
    logger.info(f"Using DistilBERT model from Hugging Face Hub: '{repo_id}'")
    
    return repo_id

def download_tfidf_model_from_hub() -> tuple[str, str]:
    """
    Downloads the TF-IDF model and vectorizer from Hugging Face Hub and returns their cache paths.

    Returns:
        tuple[str, str]: A tuple containing the cache paths to the model and vectorizer.
    """
    repo_id = os.getenv('TFIDF_MODEL_REPO_ID', 'protostarss/tfidflr-IMDB')
    logger.info(f"Downloading TF-IDF model and vectorizer from '{repo_id}' to cache...")

    try:
        model_cache_path = hf_hub_download(
            repo_id=repo_id,
            filename="IMDB_tfidflr_model.pkl"
        )
        vectorizer_cache_path = hf_hub_download(
            repo_id=repo_id,
            filename="IMDB_vectorizer.pkl"
        )

        logger.info(f"TF-IDF model cached at: {model_cache_path}")
        logger.info(f"Vectorizer cached at: {vectorizer_cache_path}")
        return model_cache_path, vectorizer_cache_path
    except Exception as e:
        logger.error(f"Failed to download TF-IDF files from Hub: {e}")
        raise









