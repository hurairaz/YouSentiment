import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def visualize_sentiment_distribution(data):
    """
    Plots the distribution of sentiments with a jet black background.
    """
    sentiment_counts = data['Sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))

    # Set background to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=['#ffeeba', '#d4edda', '#f8d7da' ]
    )

    # Customize text color
    for text in texts:
        text.set_color('white')  # Set label color to white
    for autotext in autotexts:
        autotext.set_color('white')  # Set percentage text color to white

    # Customize title color
    #ax.set_title('Sentiment Distribution', color='white')

    logging.info("Sentiment distribution plotted successfully.")
    return fig


def visualize_sentiments_by_video(data):
    """
    Visualizes sentiment distribution per video with a jet black background.
    """
    video_sentiment = data.groupby(['VideoID', 'Sentiment']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set background to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot the stacked bar chart
    video_sentiment.plot(kind='bar', stacked=True, ax=ax, color=['#f8d7da', '#ffeeba', '#d4edda'])

    # Customize title and axis labels
    #ax.set_title('Sentiment Distribution by Video', color='white')
    ax.set_xlabel('Video ID', color='white')
    ax.set_ylabel('Number of Comments', color='white')

    # Customize ticks
    ax.tick_params(colors='white')

    # Customize legend
    ax.legend(title='Sentiment', facecolor='black', edgecolor='white', labelcolor='white')

    logging.info("Sentiment distribution by video plotted successfully.")
    return fig


def plot_confusion_matrix(conf_matrix, model_name, labels=["Negative", "Neutral", "Positive"]):
    """
    Plots the confusion matrix for a given model with a jet black background.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set background to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Plot the heatmap
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={'label': 'Color Scale'}
    )

    # Customize title and axis labels
    ax.set_title(f"Confusion Matrix: {model_name}", color='white')
    ax.set_xlabel("Predicted Labels", color='white')
    ax.set_ylabel("True Labels", color='white')

    # Customize ticks
    ax.tick_params(colors='white')

    logging.info(f"Confusion matrix plotted for {model_name}.")
    return fig


def display_classification_report(class_report, model_name):
    """
    Logs and displays the classification report for a given model.
    """
    logging.info(f"\nClassification Report for {model_name}:\n{class_report}")
