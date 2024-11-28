import logging
import joblib
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
temp_dir = tempfile.TemporaryDirectory()

def vectorize_text(data):
    """
    Vectorizes text data using TF-IDF.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(data['Preprocessed Comment'])
    return X, vectorizer


def handle_class_imbalance(X_train, y_train):
    """
    Handles class imbalance using RandomOverSampler.
    """
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.
    """
    lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
    lr_classifier.fit(X_train, y_train)
    logging.info("Logistic Regression model trained successfully.")
    return lr_classifier


def train_multinomial_nb(X_train, y_train):
    """
    Trains a Multinomial Naive Bayes model.
    """
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    logging.info("Multinomial Naive Bayes model trained successfully.")
    return nb_classifier


def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluates a trained model and returns performance metrics.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"])
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, class_report, conf_matrix


def save_model(model, filename):
    """
    Saves a model using joblib in a temporary directory.
    """
    file_path = f"{temp_dir.name}/{filename}.pkl"
    joblib.dump(model, file_path)
    logging.info(f"Model saved as '{file_path}'")


def load_model(filename):
    """
    Loads a model using joblib from the temporary directory.
    """
    file_path = f"{temp_dir.name}/{filename}.pkl"
    logging.info(f"Loading model from '{file_path}'")
    return joblib.load(file_path)
def run_training_pipeline(data):
    """
    Runs the entire training pipeline, trains models, and calculates performance metrics.
    """
    logging.info("Vectorizing text....")
    X, vectorizer = vectorize_text(data)
    y = data['Sentiment']

    logging.info("Splitting data....")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Handling class imbalance....")
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

    logging.info("Training Logistic Regression....")
    lr_model = train_logistic_regression(X_train_resampled, y_train_resampled)

    logging.info("Training Multinomial Naive Bayes....")
    nb_model = train_multinomial_nb(X_train_resampled, y_train_resampled)

    logging.info("Evaluating Logistic Regression....")
    lr_accuracy, lr_class_report, lr_conf_matrix = evaluate_model_performance(lr_model, X_test, y_test)

    logging.info("Evaluating Multinomial Naive Bayes....")
    nb_accuracy, nb_class_report, nb_conf_matrix = evaluate_model_performance(nb_model, X_test, y_test)

    logging.info("\nLogistic Regression Metrics:")
    logging.info(f"Accuracy: {lr_accuracy:.2f}")
    logging.info("Classification Report:")
    logging.info(lr_class_report)

    logging.info("\nMultinomial Naive Bayes Metrics:")
    logging.info(f"Accuracy: {nb_accuracy:.2f}")
    logging.info("Classification Report:")
    logging.info(nb_class_report)

    save_model(vectorizer, "vectorizer")
    save_model(lr_model, "logistic_regression")
    save_model(nb_model, "naive_bayes")

    return {
        "lr": {"accuracy": lr_accuracy, "conf_matrix": lr_conf_matrix, "class_report": lr_class_report},
        "nb": {"accuracy": nb_accuracy, "conf_matrix": nb_conf_matrix, "class_report": nb_class_report}
    }

