import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def visualize_sentiment_distribution(data):
    """
    Plots the distribution of sentiments.
    """
    sentiment_counts = data['Sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=['green', 'gray', 'red']
    )
    plt.title('Sentiment Distribution')
    plt.show()


def visualize_sentiments_by_video(data):
    """
    Visualizes sentiment distribution per video.
    """
    video_sentiment = data.groupby(['VideoID', 'Sentiment']).size().unstack(fill_value=0)
    video_sentiment.plot(kind='bar', stacked=True, figsize=(10, 6), color=['red', 'gray', 'green'])
    plt.title('Sentiment Distribution by Video')
    plt.xlabel('Video ID')
    plt.ylabel('Number of Comments')
    plt.legend(title='Sentiment')
    plt.show()


def plot_confusion_matrix(conf_matrix, model_name, labels=["Negative", "Neutral", "Positive"]):
    """
    Plots the confusion matrix for a given model.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def display_classification_report(class_report, model_name):
    """
    Displays the classification report for a given model.
    """
    logging.info(f"\nClassification Report for {model_name}:\n{class_report}")
