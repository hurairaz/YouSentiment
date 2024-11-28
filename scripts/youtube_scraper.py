import logging
import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure YouTube API
API_KEY = os.getenv('api_key')
youtube = build('youtube', 'v3', developerKey=API_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_yt_videos(search_query, max_results=50):
    """
    Fetches videos based on the search query.
    """
    videos = []
    try:
        video_results = youtube.search().list(
            q=search_query,
            part='id,snippet',
            maxResults=max_results,
            type='video'
        ).execute()

        for video_result in video_results.get('items', []):
            videos.append({
                'videoId': video_result['id']['videoId'],
                'title': video_result['snippet']['title'],
                'description': video_result['snippet']['description']
            })

        logging.info(f"Fetched {len(videos)} videos for the query: '{search_query}'.")

    except Exception as e:
        logging.error(f"An error occurred while fetching videos: {e}")

    return videos


def get_video_comments(video_id, max_results=100):
    """
    Fetches comments for a specific video.
    """
    comments = []
    try:
        comment_results = youtube.commentThreads().list(
            videoId=video_id,
            part='snippet',
            maxResults=max_results
        ).execute()

        for comment_result in comment_results.get('items', []):
            comments.append(comment_result['snippet']['topLevelComment']['snippet']['textDisplay'])

        logging.info(f"Fetched {len(comments)} comments for video ID: {video_id}.")

    except Exception as e:
        logging.error(f"An error occurred while fetching comments for video {video_id}: {e}")

    return comments
