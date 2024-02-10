import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
import pytz

# Function to display tweets in a table based on selected sentiment or emotion
def display_tweets(tweets, selected_category, category_name):
    filtered_tweets = [tweet for tweet in tweets if tweet[category_name] == selected_category]
    if filtered_tweets:
        df = pd.DataFrame(filtered_tweets)
        st.table(df[['date', 'cleanContent', 'emotion', 'sentiment']])
    else:
        st.write("No tweets found for this category.")
    
# Function to load recent tweets
def load_tweets_by_timeframe(file_path, start_time, end_time):
    timeframe_tweets = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tweet = json.loads(line)
            tweet_time = datetime.fromisoformat(tweet['date'].rstrip('Z'))  # Convert to datetime, removing 'Z'
            if start_time <= tweet_time <= end_time:
                timeframe_tweets.append(tweet)
    return timeframe_tweets

# print("change: ", change)

def display_sentiments_emotions(tweets, emoji_dict, category, current_counts, changes):
    st.markdown(f"### {category.capitalize()}")

    # Prepare items for a 2-column grid
    items = list(emoji_dict.items())
    num_rows = (len(items) + 1) // 2  # Calculate required rows for a 2-column layout
    
    for i in range(num_rows):
        cols = st.columns(2)  # Create 2 columns for the grid
        for j in range(2):
            index = i * 2 + j
            if index < len(items):
                key, emoji = items[index]
                current_count = current_counts.get(key, 0)
                change = changes.get(key, 0)
                print("change: ", change)
                print(current_counts)
                
                # Handling percentage change correctly
                previous_count = current_count - change  # Back-calculate previous count
                percent_change = (change / previous_count * 100) if previous_count != 0 else float('inf')
                
                arrow = "↗️" if change > 0 else "↘️" if change < 0 else "➖"
                color = "color:green;" if change > 0 else "color:red;" if change < 0 else "color:black;"
                
                # Display emoji, count, and formatted percentage change with arrow
                display_html = f"<span style='font-size: 24px;'>{emoji} x{current_count}</span> <span style='{color}'>{arrow} {abs(percent_change):.2f}%</span>"
                cols[j].markdown(display_html, unsafe_allow_html=True)
                
                # Using an expander to show tweets related to each sentiment/emotion
                with cols[j].expander(f"Show tweets for {key.capitalize()}"):
                    filtered_tweets = tweets[tweets[category] == key]
                    if not filtered_tweets.empty:
                        st.dataframe(filtered_tweets[['date', 'cleanContent']])
                    else:
                        st.write("No tweets found.")



# Mapping of emotions/sentiments to emojis: Update as per your categories
emotion_emojis = {
    'neutral': '😐',    
    'joy': '😂',
    'sadness': '🙁',
    'anger': '😠',
    'fear': '😨',
    'love': '😍', 
    'surprise': '😮'
}

sentiment_emojis = {
    'strong negative': '😡',  # Very angry or upset
    'moderately negative': '😠',  # Angry
    'mildly negative': '🙁',  # Slightly frowning
    'neutral': '😐',  # Neutral face
    'mildly positive': '🙂',  # Slightly smiling
    'moderately positive': '😊',  # Smiling
    'strong positive': '😁'  # Beaming face with smiling eyes
}

def time_period():
    # Calculate the current time and subtract 6 hours to define the start of the range
    current_time = datetime.now(pytz.utc)  # Using now() with timezone awareness
    range_start = current_time - timedelta(hours=6)

    # Define the number of 30-minute intervals in a 6-hour range
    total_intervals = 12  # 6 hours * 2 intervals per hour

    # Generate labels for each interval
    interval_labels = [f"{30 * i} - {30 * (i + 1)} minutes in the past" for i in range(total_intervals)]
    interval_labels.reverse()  # Reverse so that the most recent interval is first

    # User selects an interval using the slider
    selected_interval_label = st.sidebar.select_slider("Select Time Interval", options=interval_labels, value=interval_labels[0])

    # Determine the selected interval index based on the chosen label
    selected_interval_index = interval_labels.index(selected_interval_label)

    # Calculate the start and end times for the selected interval
    interval_end = current_time - timedelta(minutes=30) * selected_interval_index
    interval_start = interval_end - timedelta(minutes=30)

    # Display the selected interval without seconds and timezone
    st.write(f"Viewing interval from {interval_start.strftime('%Y-%m-%d %H:%M')} to {interval_end.strftime('%Y-%m-%d %H:%M')} {current_time.tzname()}")

    st.write("Display tweets or data for the selected interval here.")
    
# Sidebar for time period selection
time_period()

# Main app logic
file_path = 'processed_english_tweets.jsonl'  # Update with the path to your JSONL file
utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)

current_interval_end = utc_now
current_interval_start = utc_now - timedelta(minutes=30)
previous_interval_start = utc_now - timedelta(minutes=60)
previous_interval_end = utc_now - timedelta(minutes=30)

# Load tweets for each interval
current_interval_tweets = pd.DataFrame(load_tweets_by_timeframe(file_path, current_interval_start, current_interval_end))
previous_interval_tweets = pd.DataFrame(load_tweets_by_timeframe(file_path, previous_interval_start, previous_interval_end))

# Debug
print(current_interval_tweets.head())
print(current_interval_tweets.columns)
print(previous_interval_tweets.head())
print(previous_interval_tweets.columns)

def calculate_counts(tweets):
    sentiment_counts = tweets['sentiment'].value_counts().to_dict()
    emotion_counts = tweets['emotion'].value_counts().to_dict()
    return sentiment_counts, emotion_counts

def calculate_changes(current_counts, previous_counts):
    changes = {}
    for key in set(current_counts.keys()).union(previous_counts.keys()):
        current = current_counts.get(key, 0)
        previous = previous_counts.get(key, 0)
        changes[key] = current - previous
    return changes

current_sentiment_counts, current_emotion_counts = calculate_counts(current_interval_tweets)
previous_sentiment_counts, previous_emotion_counts = calculate_counts(previous_interval_tweets)

sentiment_changes = calculate_changes(current_sentiment_counts, previous_sentiment_counts)
emotion_changes = calculate_changes(current_emotion_counts, previous_emotion_counts)

# Assuming you have DataFrames or Series for current counts and changes
# For example: current_sentiment_counts, sentiment_changes, current_emotion_counts, emotion_changes
display_sentiments_emotions(current_interval_tweets, sentiment_emojis, 'sentiment', current_sentiment_counts, sentiment_changes)
display_sentiments_emotions(current_interval_tweets, emotion_emojis, 'emotion', current_emotion_counts, emotion_changes)
