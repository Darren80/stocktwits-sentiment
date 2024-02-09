import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import pytz

# Sample data: Replace with your actual data loading logic
def load_data():
    return pd.DataFrame({
        'datetime': pd.date_range(start="2024-01-01", periods=100, freq='30T'),
        'tweet': ['Tweet {}'.format(i) for i in range(100)],
        'sentiment': ['positive' if i%2 == 0 else 'negative' for i in range(100)],
        'emotion': ['happy' if i%2 == 0 else 'sad' for i in range(100)],
    })
    
# Function to load recent tweets
def load_recent_tweets(file_path, cutoff_time):
    recent_tweets = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tweet = json.loads(line)
            tweet_time = datetime.fromisoformat(tweet['date'].rstrip('Z'))  # Assuming UTC and removing 'Z'
            if tweet_time > cutoff_time:
                recent_tweets.append(tweet)
    return recent_tweets

# Function to display emojis based on count in the main area
def display_emojis_main_area(sentiment_emoji_dict, emotion_emoji_dict, sentiment_counts, emotion_counts):
    # Create two columns for sentiments and emotions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sentiments")
        for key, emoji in sentiment_emoji_dict.items():
            count = sentiment_counts.get(key, 0)
            size = max(10, count * 10)  # Example scaling factor
            st.markdown(f"<h1 style='font-size: {size}px;'>{emoji} x{count}</h1>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Emotions")
        for key, emoji in emotion_emoji_dict.items():
            count = emotion_counts.get(key, 0)
            size = max(10, count * 10)  # Example scaling factor
            st.markdown(f"<h1 style='font-size: {size}px;'>{emoji} x{count}</h1>", unsafe_allow_html=True)

# Mapping of emotions/sentiments to emojis: Update as per your categories
emotion_emojis = {
    'neutral': '😐',    
    'joy': '😂',
    'sadness': '🙁',
    'anger': '😠',
    'fear': '😨',
    'love': '❤️', 
    'surprise': '😮'
}

sentiment_emojis = {
    'positive': '😊',
    'negative': '😠',
    'neutral': '😐'
}

def time_period():
    # Calculate the current time and subtract 6 hours to define the start of the range
    current_time = datetime.utcnow()
    range_start = current_time - timedelta(hours=6)

    # Define the number of 30-minute intervals in a 6-hour range
    total_intervals = 12  # 6 hours * 2 intervals per hour

    # User selects an interval using the slider
    selected_interval_index = st.sidebar.slider("Select Time Interval (0 is most recent)", 0, total_intervals - 1, 0)

    # Calculate the start and end of the selected interval
    # Note: This makes the 0th interval the most recent 30-minute window
    interval_end = current_time - timedelta(minutes=30) * selected_interval_index
    interval_start = interval_end - timedelta(minutes=30)

    # Display the selected interval
    st.write(f"Viewing interval from {interval_start} to {interval_end}")

    # Placeholder for displaying data
    # You would filter your dataset based on interval_start and interval_end here
    # and display the relevant tweets or data.
    st.write("Display tweets or data for the selected interval here.")
# Sidebar for time period selection
time_period()

# Main app logic
file_path = 'processed_english_tweets.jsonl'  # Update with the path to your JSONL file
cutoff_time = datetime.utcnow().replace(tzinfo=pytz.utc) - timedelta(hours=6)  # Last 6 hours

tweets = load_recent_tweets(file_path, cutoff_time)

# Count sentiments and emotions in the interval
sentiment_counts = pd.Series([tweet['sentiment'] for tweet in tweets]).value_counts().to_dict()
emotion_counts = pd.Series([tweet['emotion'] for tweet in tweets]).value_counts().to_dict()

# Display emojis with varying sizes based on counts in the main area
display_emojis_main_area(sentiment_emojis, emotion_emojis, sentiment_counts, emotion_counts)

