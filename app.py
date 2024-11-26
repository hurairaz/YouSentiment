import logging
import tempfile
import streamlit as st
import pandas as pd
from scripts.youtube_scraper import get_yt_videos, get_video_comments
from scripts.preprocessing import preprocess_comment
from scripts.sentiment_analysis import (
    analyze_sentiments,
    extract_compound_scores,
    get_top_comments,
    calculate_overall_sentiment,
)
from scripts.visualization import visualize_sentiment_distribution, visualize_sentiments_by_video, plot_confusion_matrix
from scripts.model_training import run_training_pipeline, load_model
from PIL import Image
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
temp_dir = tempfile.TemporaryDirectory()

def main():
    # Page Configuration
    st.set_page_config(page_title="You Sentiment", page_icon="assets/YouSentiment-icon.png", layout="wide")

    # Custom CSS
    custom_css = """
    <style>
    footer {visibility: hidden;}
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .animated-title {
            text-align: center;
            margin-left: -140px;
            font-size: 55px;
            color: #FA5252;
            animation: bounceIn 2s ease-in-out; /* Add bounce-in animation */
        }

        /* Keyframes for bounce-in animation */
        @keyframes bounceIn {
            0% {
                transform: scale(0.5);
                opacity: 0;
            }
            50% {
                transform: scale(1.0);
                opacity: 0.8;
            }
            100% {
                transform: scale(1);
                opacity: 1;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # st.markdown(
    #     """
    #     <h1 style="margin-bottom: -50px; font-size: 2.5rem;">You Sentiment</h1>
    #     """,
    #     unsafe_allow_html=True,
    # )
    img = Image.open("icons8-you-100.png")

    # Display title and image horizontally
    col1, col2, col3 = st.columns([1, 4, 6])  # Adjust column widths for spacing

    with col1:
        st.image(img)  # Adjust width if needed

    with col2:
        st.markdown("<h1 class='animated-title' style='text-align: center; margin-left: -140px; font-size:55px; color:#FA5252;'>You Sentiment</h1>", unsafe_allow_html=True)

    with col3:
        st.write("")

    st.markdown("---")

    # Initialize session state
    if "data" not in st.session_state:
        st.session_state["data"] = None
        st.session_state["metrics"] = None
        st.session_state["models_loaded"] = False

    # Search Bar Section
    search_query = st.text_input("Curious about public opinions? Enter a topic to explore!")

    if search_query:
        if not search_query.strip():
            st.warning("Please enter a valid topic to proceed!")
            return
        else:
            st.markdown("---")
            #st.markdown(f"<h1 class='animated-title' style='text-align: center; font-size:35px; color:#FA5252;'>{search_query}</h1>", unsafe_allow_html=True)

            # Spinner while fetching and analyzing data
            with st.spinner("Fetching and analyzing data..."):
                # Check if data is already fetched
                if st.session_state["data"] is None:
                    logging.info(f"Fetching videos for topic: {search_query}")
                    videos = get_yt_videos(search_query)
                    if not videos:
                        logging.error("No videos found. Try another topic.")
                        st.error("No videos found. Try another search query.")
                        return

                    # Fetch Comments
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

                    # Data Preparation
                    data = pd.DataFrame(comments_data).dropna()
                    data['Preprocessed Comment'] = data['Comment'].apply(preprocess_comment)
                    data = data.dropna(subset=['Preprocessed Comment'])
                    data['Sentiment'], data['Polarity Score'] = analyze_sentiments(data['Preprocessed Comment'])
                    data = extract_compound_scores(data)
                    data = data.dropna(subset=['Compound Score'])

                    # Save Data
                    logging.info("Saving data...")
                    data_file_path = f"{temp_dir.name}/video_com.csv"
                    data.to_csv(data_file_path, index=False)
                    st.session_state["data"] = data
                else:
                    data = st.session_state["data"]

                logging.info("Calculating top comments and overall sentiment...")
                top_positive_comments, top_negative_comments, top_neutral_comments = get_top_comments(data)
                overall_sentiment = calculate_overall_sentiment(data)

                logging.info(f"Overall sentiment for topic '{search_query}': {overall_sentiment}")

                logging.info("Visualizing results...")
                fig1 = visualize_sentiment_distribution(data)
                fig2 = visualize_sentiments_by_video(data)

                st.success("Data fetching and analysis complete!")
                # Stop spinner here
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                color = {
                    "Positive": "#28a745",  # Green
                    "Neutral": "#ffc107",  # Yellow
                    "Negative": "#dc3545"  # Red
                }.get(overall_sentiment, "#6c757d")  # Default gray if sentiment is unknown

                st.markdown(
                    f"""
                    <div style="
                        background-color: {color};
                        border-radius: 10px;
                        padding: 10px;
                        margin-bottom: 20px;
                        text-align: center;
                        font-size: 20px;
                        color: white;
                        font-weight: bold;">
                        Overall Sentiment: {overall_sentiment}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Sentiment Distribution")
                    st.pyplot(fig1)
                with col2:
                    st.markdown("#### Sentiments by Video")
                    st.pyplot(fig2)

                # Top Comments
                st.markdown("### Top Comments")
                st.write("#### Top Positive Comments")
                st.table(top_positive_comments[['Comment', 'Compound Score']])

                st.write("#### Top Negative Comments")
                st.table(top_negative_comments[['Comment', 'Compound Score']])

                st.write("#### Top Neutral Comments")
                st.table(top_neutral_comments[['Comment', 'Compound Score']])

                st.markdown("---")

                # Train Models if not already trained
                if st.session_state["metrics"] is None:
                    logging.info("Training models...")
                    metrics = run_training_pipeline(data)
                    st.session_state["metrics"] = metrics
                else:
                    metrics = st.session_state["metrics"]

                # Analyze User Comment
                st.markdown(f"## Analyze a Comment for '{search_query}'")
                user_comment = st.text_input("Enter your comment to analyze sentiment:")
                if user_comment:
                    preprocessed_comment = preprocess_comment(user_comment)
                    if not st.session_state["models_loaded"]:
                        vectorizer = load_model("vectorizer")
                        lr_model = load_model("logistic_regression")
                        nb_model = load_model("naive_bayes")
                        st.session_state["models"] = {"vectorizer": vectorizer, "lr": lr_model, "nb": nb_model}
                        st.session_state["models_loaded"] = True
                    else:
                        vectorizer = st.session_state["models"]["vectorizer"]
                        lr_model = st.session_state["models"]["lr"]
                        nb_model = st.session_state["models"]["nb"]

                    vectorized_comment = vectorizer.transform([preprocessed_comment])
                    lr_sentiment = lr_model.predict(vectorized_comment)[0]
                    nb_sentiment = nb_model.predict(vectorized_comment)[0]

                    st.markdown("#### Sentiment Prediction")
                    st.write(f"**Logistic Regression Model:** {lr_sentiment}")
                    st.write(f"**Multinomial Naive Bayes Model:** {nb_sentiment}")

                # Display Performance Metrics
                if st.button("Show Model Performance"):
                    logging.info("Displaying performance metrics...")
                    fig3 = plot_confusion_matrix(metrics["lr"]["conf_matrix"], "Logistic Regression")
                    fig4 = plot_confusion_matrix(metrics["nb"]["conf_matrix"], "Multinomial Naive Bayes")
                    lr_accuracy = metrics["lr"]["accuracy"]
                    nb_accuracy = metrics["nb"]["accuracy"]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Logistic Regression Metrics")
                        st.write(f"Accuracy: {lr_accuracy:.2f}")
                        st.pyplot(fig3)
                    with col2:
                        st.markdown("#### Multinomial Naive Bayes Metrics")
                        st.write(f"Accuracy: {nb_accuracy:.2f}")
                        st.pyplot(fig4)

if __name__ == "__main__":
    main()
