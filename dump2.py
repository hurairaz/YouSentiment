import streamlit as st
import pandas as pd

# CSS for styling the chat bubbles
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

# Initialize DataFrame with dummy comments
data = {
    "Comment": [
        "This video is amazing! I learned so much.",  # Positive
        "I love the way this topic is explained.",  # Positive
        "Great content and very engaging.",  # Positive
        "The video was okay, but nothing special.",  # Neutral
        "Not too bad, but could be better.",  # Neutral
        "I neither liked nor disliked this video.",  # Neutral
        "This is the worst video I've ever seen.",  # Negative
        "Terrible explanation, waste of time.",  # Negative
        "I hate this kind of content, very bad.",  # Negative
    ],
    "Compound Score": [0.85, 0.78, 0.65, 0.05, 0.00, -0.03, -0.75, -0.85, -0.90],
    "Sentiment": ["Positive", "Positive", "Positive", "Neutral", "Neutral", "Neutral", "Negative", "Negative", "Negative"]
}

df = pd.DataFrame(data)

# Split into sentiment-specific DataFrames
top_positive_comments = df[df['Sentiment'] == 'Positive']
top_negative_comments = df[df['Sentiment'] == 'Negative']
top_neutral_comments = df[df['Sentiment'] == 'Neutral']

# Streamlit App
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
