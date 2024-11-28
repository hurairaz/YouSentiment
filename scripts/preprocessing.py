import logging
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess_comment(comment):
    """
    Preprocesses a given comment by removing punctuation, stopwords, and lemmatizing.
    """
    try:
        comment = str(comment).lower()
        comment = comment.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(comment)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    except Exception as e:
        logging.error(f"Error preprocessing comment: {e}")
        return ""
