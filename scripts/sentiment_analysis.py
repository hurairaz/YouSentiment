import ast
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def calculate_polarity(comment):
    """
    Calculates polarity score for a comment.
    """
    return sia.polarity_scores(comment)


def label_sentiment(score):
    """
    Labels sentiment as Positive, Neutral, or Negative based on compound score.
    """
    if score['compound'] > 0.05:
        return 'Positive'
    elif score['compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def analyze_sentiments(comments):
    """
    Analyzes sentiments of a list of comments.
    """
    polarity_scores = comments.apply(calculate_polarity)
    sentiments = polarity_scores.apply(label_sentiment)
    return sentiments, polarity_scores


def extract_compound_scores(data):
    """
    Ensures Polarity Score is a dictionary and extracts Compound Score.
    """
    data['Polarity Score'] = data['Polarity Score'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    data['Compound Score'] = data['Polarity Score'].apply(
        lambda x: x['compound'] if isinstance(x, dict) and 'compound' in x else None
    )
    return data


def get_top_comments(data):
    """
    Retrieves the top 3 Positive, Negative, and Neutral comments based on Compound Score.
    """
    top_positive_comments = data[data['Sentiment'] == 'Positive'].sort_values(by='Compound Score',
                                                                              ascending=False).head(3)
    top_negative_comments = data[data['Sentiment'] == 'Negative'].sort_values(by='Compound Score',
                                                                              ascending=False).head(3)
    top_neutral_comments = data[data['Sentiment'] == 'Neutral'].sort_values(by='Compound Score', ascending=False).head(
        3)
    return top_positive_comments, top_negative_comments, top_neutral_comments


def calculate_overall_sentiment(data):
    """
    Calculates overall sentiment for the dataset based on the count of sentiments.
    """
    sentiment_counts = data['Sentiment'].value_counts()

    # Determine the sentiment with the highest count
    overall_sentiment = sentiment_counts.idxmax()

    return overall_sentiment
