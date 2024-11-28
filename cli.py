#CLI
import logging
import tempfile
import pandas as pd
from scripts.youtube_scraper import get_yt_videos, get_video_comments
from scripts.preprocessing import preprocess_comment
from scripts.sentiment_analysis import analyze_sentiments, extract_compound_scores, get_top_comments, calculate_overall_sentiment
from scripts.visualization import visualize_sentiment_distribution, visualize_sentiments_by_video, plot_confusion_matrix
from scripts.model_training import run_training_pipeline, load_model

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
temp_dir = tempfile.TemporaryDirectory()

def main():


    search_query = input("Enter a topic for analysis: ")

    logging.info(f"Fetching videos for topic: {search_query}")
    videos = get_yt_videos(search_query)
    if not videos:
        logging.error("No videos found. Try another topic.")
        return

    logging.info("Fetching comments...")
    comments_data = []
    for video in videos:
        comments = get_video_comments(video['videoId'])
        for comment in comments:
            comments_data.append({
                'VideoID': video['videoId'],
                'Title': video['title'],
                'Description': video['description'],
                'Comment': comment
            })

    data = pd.DataFrame(comments_data).dropna()
    data['Preprocessed Comment'] = data['Comment'].apply(preprocess_comment)
    data = data.dropna(subset='Preprocessed Comment')
    data['Sentiment'], data['Polarity Score'] = analyze_sentiments(data['Comment'])
    data = data.dropna(subset='Sentiment')
    data = data.dropna(subset='Polarity Score')
    data = extract_compound_scores(data)
    data = data.dropna(subset='Compound Score')
    logging.info("Saving Data...")
    data_file_path = f"{temp_dir.name}/video_com.csv"
    data.to_csv(data_file_path, index=False)


    logging.info("Calculating top comments and overall sentiment...")
    top_positive_comments, top_negative_comments, top_neutral_comments = get_top_comments(data)
    overall_sentiment = calculate_overall_sentiment(data)

    logging.info("Training models...")
    metrics = run_training_pipeline(data)

    logging.info("Visualizing results...")
    fig = visualize_sentiment_distribution(data)
    fig.show()
    fig = visualize_sentiments_by_video(data)
    fig.show()
    fig = plot_confusion_matrix(metrics["lr"]["conf_matrix"], "Logistic Regression")
    fig.show()
    fig = plot_confusion_matrix(metrics["nb"]["conf_matrix"], "Multinomial Naive Bayes")
    fig.show()

    logging.info(f"Overall sentiment for topic '{search_query}': {overall_sentiment}")

    while True:
        user_comment = input("Enter a comment to analyze sentiment (or type 'exit' to quit): ")
        if user_comment.lower() == 'exit':
            break

        preprocessed_comment = preprocess_comment(user_comment)
        vectorizer = load_model("vectorizer")
        vectorized_comment = vectorizer.transform([preprocessed_comment])
        lr_model = load_model("logistic_regression")
        nb_model = load_model("naive_bayes")

        lr_sentiment = lr_model.predict(vectorized_comment)[0]
        nb_sentiment = nb_model.predict(vectorized_comment)[0]

        logging.info(f"Logistic Regression Sentiment: {lr_sentiment}")
        logging.info(f"Naive Bayes Sentiment: {nb_sentiment}")


if __name__ == "__main__":
    main()
