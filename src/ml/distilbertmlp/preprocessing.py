import re
from bs4 import BeautifulSoup


def preprocessing_pipeline_bert(review: str):
    """
    Clean and preprocess a review string using the full pipeline.

    Args:
        review (str): The raw review text.
    Returns:
        str: The cleaned and preprocessed review text.
    """

    # 1. strip HTML
    review = BeautifulSoup(review, "html.parser").get_text()

    # 2. strip URLs
    review = re.sub(r'http\S+|www\.\S+', '', review, flags=re.IGNORECASE)

    return review
