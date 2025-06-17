import string
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
# from spellchecker import SpellChecker
import emoji
# Import chat-words
from ml.tfidfrl.chatword import chat_words

nltk.download('wordnet')
nltk.download('omw-1.4')  # for improved lemmatization support
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocessing_pipeline(review: str):
    """
    Clean and preprocess a review string using the full pipeline.

    Args:
        review (str): The raw review text.
    Returns:
        str: The cleaned and preprocessed review text.
    """
    # 1. quick lowercase (won’t hurt)
    review = review.lower()

    # 2. strip HTML
    review = BeautifulSoup(review, "html.parser").get_text()

    # 3. strip URLs
    review = re.sub(r'http\S+|www\.\S+', '', review, flags=re.IGNORECASE)

    # 4. strip [bracketed] parts
    review = re.sub(r'\[[^]]*\]', '', review)

    # 5. remove emoji
    review = emoji.replace_emoji(review, replace='')

    # 6. strip punctuation
    review = review.translate(str.maketrans('', '', string.punctuation))

    # 7. chat-word expansion  (done **before** stop-word removal)
    new_text = []
    for word in review.split():
        if word.upper() in chat_words:
            new_text.append(chat_words[word.upper()])
        else:
            new_text.append(word)
    review = " ".join(new_text)

    # 8. lowercase again so expansions are uniform
    review = review.lower()

    # 9. remove stop-words
    review = ' '.join([word for word in review.split() if word not in stop_words])

    # 10. spelling-correction block — takes time
    # spell = SpellChecker()
    # words = review.split()
    # corrected_words = []

    # for word in words:
    #     corrected = spell.correction(word)
    #     # If correction is None, use the original word
    #     corrected_words.append(corrected if corrected else word)

    # review = " ".join(corrected_words)


    # 11. stemming or lemmatization
    # review = ' '.join([ps.stem(word) for word in review.split()])
    review = ' '.join([lemmatizer.lemmatize(word) for word in review.split()])
    
    return review