#Streamlit App

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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
temp_dir = tempfile.TemporaryDirectory()

# Page Configuration
st.set_page_config(page_title="You Sentiment", page_icon="assets/YouSentiment-icon.png", layout="wide")

# Custom CSS
# custom_css = """
# <style>
# footer {visibility: hidden;}
# header {visibility: hidden;}
# #MainMenu {visibility: hidden;}
# </style>
# """
# st.markdown(custom_css, unsafe_allow_html=True)

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


def fetch_analyze_page():
    """
    Page for fetching and analyzing data.
    """
    img = Image.open("icons8-you-100.png")
    col1, col2, col3 = st.columns([1, 4, 6])  # Adjust column widths for spacing

    with col1:
        st.image(img, width=90)  # Adjust width if needed

    with col2:
        st.markdown(
            "<h1 class='animated-title' style='text-align: center; margin-left: -140px; margin-bottom:20px; font-size:55px; color:#FA5252;'>You Sentiment</h1>",
            unsafe_allow_html=True,
        )
    with col3:
        st.write("")

    st.markdown("---")

    # Initialize session state
    if "data" not in st.session_state:
        st.session_state["data"] = None
        st.session_state["metrics"] = None
        st.session_state["models_loaded"] = False
    st.markdown(
        """
        <style>
        @keyframes colorChange {
            0% { color: #f8d7da; } /* Light red */
            33% { color: #ffeeba; } /* Light yellow */
            66% { color: #d4edda; } /* Light green */
            100% { color: #f8d7da; } /* Back to Light red */
        }
        .animated-header {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            animation: colorChange 6s infinite; /* Change color every 6 seconds */
        }
        </style>
        <h1 class="animated-header">Curious about public sentiments? Enter a topic to explore!</h1>
        """,
        unsafe_allow_html=True
    )
    search_query = st.text_input("")

    if search_query:
        if not search_query.strip():
            st.warning("Please enter a valid topic to proceed!")
            return
        else:
            st.markdown("---")
            # st.markdown(f"<h1 class='animated-title' style='text-align: center; font-size:35px; color:#FA5252;'>{search_query}</h1>", unsafe_allow_html=True)

            with st.spinner(f"Fetching and analyzing data for {search_query}..."):
                if st.session_state["data"] is None:
                    logging.info(f"Fetching videos for topic: {search_query}")
                    videos = get_yt_videos(search_query)
                    if not videos:
                        logging.error("No videos found. Try another topic.")
                        st.error("No videos found. Try another search query.")
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
                    data = data.dropna(subset=['Preprocessed Comment'])
                    data['Sentiment'], data['Polarity Score'] = analyze_sentiments(data['Comment'])
                    data = extract_compound_scores(data)
                    data = data.dropna(subset=['Compound Score'])

                    data_file_path = f"{temp_dir.name}/video_com.csv"
                    data.to_csv(data_file_path, index=False)
                    st.session_state["data"] = data
                    st.session_state["search_query"] = search_query
                    st.session_state["current_page"] = "results"
                    st.success("Data fetching and analysis complete! Click below to view results.")
                    st.button("View Results", on_click=lambda: st.session_state.update({"current_page": "results"}))


def results_page():
    """
    Page to display results and train models.
    """
    data = st.session_state["data"]
    search_query = st.session_state["search_query"]
    header_str = "Topic: " + search_query
    st.header(header_str)
    st.markdown("---")
    st.text("")

    overall_sentiment = calculate_overall_sentiment(data)

    color = {
        "Positive": "#d4edda",  # Light green background
        "Neutral": "#ffeeba",  # Light yellow background
        "Negative": "#f8d7da"  # Light red background
    }.get(overall_sentiment, "#e2e3e5")  # Default gray background

    text_color = {
        "Positive": "#155724",  # Dark green text
        "Neutral": "#856404",  # Dark yellow text
        "Negative": "#721c24"  # Dark red text
    }.get(overall_sentiment, "#383d41")  # Default dark gray text

    st.markdown(
        f"""
        <div style="
            background-color: {color};
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            font-size: 20px;
            color: {text_color};
            font-weight: bold;">
            Overall Sentiment: {overall_sentiment}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.text("")
    st.markdown("---")

    fig1 = visualize_sentiment_distribution(data)
    fig2 = visualize_sentiments_by_video(data)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Sentiment Distribution")
        st.text("")
        st.text("")
        st.pyplot(fig1)
    with col2:
        st.markdown("#### Sentiments by Video")
        st.text("")
        st.text("")
        st.pyplot(fig2)

    st.text("")
    st.text("")
    st.markdown("---")
    st.markdown("""
        <style>
        .chat-bubble {
            padding: 10px;
            margin: 10px 0;
            border-radius: 15px 15px 15px 5px; /* Top-left corner rounded */
            max-width: 70%;
            text-align: left;
            font-size: 1rem;
            word-wrap: break-word;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .positive {
            background-color: #d4edda; /* Light green */
            color: #155724; /* Dark green */
        }
        .neutral {
            background-color: #ffeeba; /* Light yellow */
            color: #856404; /* Dark yellow */
        }
        .negative {
            background-color: #f8d7da; /* Light red */
            color: #721c24; /* Dark red */
        }
        </style>
    """, unsafe_allow_html=True)
    top_positive_comments, top_negative_comments, top_neutral_comments = get_top_comments(data)
    st.markdown("### Top Comments")

    # Top Positive Comments
    st.markdown("#### Top Positive Comments")
    for _, row in top_positive_comments.iterrows():
        comment = row['Comment']
        score = row['Compound Score']
        st.markdown(
            f"""
            <div class="chat-bubble positive">
                <strong>Score: {score:.2f}</strong><br>
                {comment}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top Neutral Comments
    st.markdown("#### Top Neutral Comments")
    for _, row in top_neutral_comments.iterrows():
        comment = row['Comment']
        score = row['Compound Score']
        st.markdown(
            f"""
            <div class="chat-bubble neutral">
                <strong>Score: {score:.2f}</strong><br>
                {comment}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top Negative Comments
    st.markdown("#### Top Negative Comments")
    for _, row in top_negative_comments.iterrows():
        comment = row['Comment']
        score = row['Compound Score']
        st.markdown(
            f"""
            <div class="chat-bubble negative">
                <strong>Score: {score:.2f}</strong><br>
                {comment}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Train Models
    if st.session_state["metrics"] is None:
        logging.info("Training models...")
        metrics = run_training_pipeline(data)
        st.session_state["metrics"] = metrics
    else:
        metrics = st.session_state["metrics"]

    st.markdown("---")
    st.button("Analyze a Comment", on_click=lambda: st.session_state.update({"current_page": "comment_analysis"}))


def comment_analysis_page():
    """
    Page for analyzing a user's comment.
    """
    search_query = st.session_state["search_query"]
    header_str = "Topic: " + search_query
    st.header(header_str)
    st.markdown("---")
    st.text("")
    st.markdown(
        """
        <style>
        @keyframes colorChange {
            0% { color: #f8d7da; } /* Light red */
            33% { color: #ffeeba; } /* Light yellow */
            66% { color: #d4edda; } /* Light green */
            100% { color: #f8d7da; } /* Back to Light red */
        }
        .animated-header {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            animation: colorChange 6s infinite; /* Change color every 6 seconds */
        }
        </style>
        <h1 class="animated-header">Curious about your take on this topic? Test your comment's sentiment and uncover its vibe!</h1>
        """,
        unsafe_allow_html=True
    )
    user_comment = st.text_input("")

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

        st.text("")
        st.markdown("---")
        # Color scheme for sentiment
        sentiment_colors = {
            "Positive": "#d4edda",  # Light green
            "Neutral": "#ffeeba",  # Light yellow
            "Negative": "#f8d7da"  # Light red
        }

        # Generate the HTML for predictions
        predictions_html = f"""
        <div style="
            display: flex;
            flex-direction: column;
            gap: 10px;">
            <div style="
                background-color: {sentiment_colors.get(lr_sentiment, '#d6d8d9')};
                border-radius: 10px;
                padding: 10px;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                color: black;">
                Logistic Regression Model Prediction: {lr_sentiment}
            </div>
            <div style="
                background-color: {sentiment_colors.get(nb_sentiment, '#d6d8d9')};
                border-radius: 10px;
                padding: 10px;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                color: black;">
                Multinomial Naive Bayes Model Prediction: {nb_sentiment}
            </div>
        </div>
        """

        # Display predictions using a single st.markdown
        st.markdown(predictions_html, unsafe_allow_html=True)

        st.text("")
        st.markdown("---")
        if st.button("Show Model Performance"):
            metrics = st.session_state["metrics"]
            logging.info("Displaying performance metrics...")
            fig3 = plot_confusion_matrix(metrics["lr"]["conf_matrix"], "Logistic Regression")
            fig4 = plot_confusion_matrix(metrics["nb"]["conf_matrix"], "Multinomial Naive Bayes")
            lr_accuracy = metrics["lr"]["accuracy"]
            nb_accuracy = metrics["nb"]["accuracy"]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Logistic Regression Metrics")
                st.write(f"Accuracy: {lr_accuracy:.2f}")
                st.text("")
                st.text("")
                st.pyplot(fig3)
            with col2:
                st.markdown("#### Multinomial Naive Bayes Metrics")
                st.write(f"Accuracy: {nb_accuracy:.2f}")
                st.text("")
                st.text("")
                st.pyplot(fig4)


# Navigation Logic
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "fetch_analyze"

if st.session_state["current_page"] == "fetch_analyze":
    fetch_analyze_page()
elif st.session_state["current_page"] == "results":
    results_page()
elif st.session_state["current_page"] == "comment_analysis":
    comment_analysis_page()
