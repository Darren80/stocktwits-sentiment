from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import heapq

def rank_sentences(text):
    stop_words = set(stopwords.words('english'))
    word_frequencies = defaultdict(int)
    
    # Tokenize words and count frequencies, ignoring stopwords
    for word in word_tokenize(text):
        if word.lower() not in stop_words:
            word_frequencies[word.lower()] += 1
    
    # Ensure word_frequencies is not empty before finding the max frequency
    if word_frequencies:
        max_frequency = max(word_frequencies.values())
    else:
        return []
    
    # Normalize frequencies
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)
    
    sentence_scores = defaultdict(int)
    sentences = sent_tokenize(text)
    
    # Score sentences by summing normalized frequencies of their words
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                sentence_scores[sent] += word_frequencies[word]
    
    # Get top n sentences based on their scores
    summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
    return summary_sentences

from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_rank_sentences(text):
    sentences = sent_tokenize(text)
    tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1)

    # Convert matrix to flat list
    sentence_scores = [score[0,0] for score in sentence_scores]

    # Get top n sentences
    top_sentence_indices = sorted(((score, index) for index, score in enumerate(sentence_scores)), reverse=True)[:3]
    top_sentences = [sentences[index] for _, index in top_sentence_indices]

    return top_sentences

# Example usage
key_sentences = tfidf_rank_sentences("Text")
for sentence in key_sentences:
    print(sentence)










import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
import pytz

import streamlit as st
ticker = "TSLA"

def create_ticker_button_row(ticker_symbols):
    col_container = st.columns(len(ticker_symbols))  # Create as many columns as there are ticker symbols
    
    for idx, ticker in enumerate(ticker_symbols):
        # Each button is placed in its own column
        if col_container[idx].button(ticker, key=ticker):  # Ensure each button has a unique key
            # If a button is clicked, update the session state
            st.session_state.selected_ticker = ticker
    
    # Return the selected ticker symbol if any
    return st.session_state.selected_ticker if 'selected_ticker' in st.session_state else None

ticker_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
# Call the function to create a row of ticker buttons and get the selected ticker
selected_ticker = create_ticker_button_row(ticker_symbols)

# Display the selected ticker symbol
if selected_ticker:
    st.write(f"You selected: {selected_ticker}")
    ticker = selected_ticker
else:
    st.write("Select a ticker symbol.")


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

# Function to load and sort news from a .jsonl file
def load_and_sort_news(filename):
    news_items = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            news_item = json.loads(line)
            news_items.append(news_item)
    # Sort news_items by 'publish date', converting the date string to a datetime object for accurate sorting
    news_items.sort(key=lambda x: datetime.strptime(x['published date'], "%a, %d %b %Y %H:%M:%S GMT"), reverse=True)
    return news_items


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
    
    # Define the number of 30-minute intervals in a 6-hour range
    total_intervals = 12  # 6 hours * 2 intervals per hour

    # Generate labels for each interval
    interval_labels = [f"{30 * i} - {30 * (i + 1)} minutes in the past" for i in range(total_intervals)]
    interval_labels.reverse()  # Reverse so that the most recent interval is first

    # User selects an interval using the slider
    selected_interval_label = st.sidebar.select_slider("Select Time Interval", options=interval_labels, value=interval_labels[-1])

    # Determine the selected interval index based on the chosen label
    selected_interval_index = interval_labels.index(selected_interval_label)

    # Calculate the start and end times for the selected interval
    interval_start = current_time - timedelta(minutes=30 * (total_intervals - selected_interval_index))
    interval_end = interval_start + timedelta(minutes=30)

    # Display the selected interval without seconds and timezone
    st.write(f"Viewing interval from {interval_start.strftime('%Y-%m-%d %H:%M')} to {interval_end.strftime('%Y-%m-%d %H:%M')} UTC")

    # Placeholder for displaying tweets or data for the selected interval
    st.write("Display tweets or data for the selected interval here.")

    return [interval_start, interval_end]
    
# Sidebar for time period selection
current_interval_start, current_interval_end = time_period()
previous_interval_end = current_interval_start
previous_interval_start = previous_interval_end - timedelta(minutes=30)
print(current_interval_start)
print(previous_interval_start)

# Main app logic
file_path = f'{ticker}_tweets.jsonl'

# Load tweets for each interval
current_interval_tweets = pd.DataFrame(load_tweets_by_timeframe(file_path, current_interval_start, current_interval_end))
previous_interval_tweets = pd.DataFrame(load_tweets_by_timeframe(file_path, previous_interval_start, previous_interval_end))

# Debug
print("C interval")
print(current_interval_tweets.head())
print(current_interval_tweets.columns)
print("P interval")
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

if not current_interval_tweets.empty and not previous_interval_tweets.empty:
    current_sentiment_counts, current_emotion_counts = calculate_counts(current_interval_tweets)
    previous_sentiment_counts, previous_emotion_counts = calculate_counts(previous_interval_tweets)

    sentiment_changes = calculate_changes(current_sentiment_counts, previous_sentiment_counts)
    emotion_changes = calculate_changes(current_emotion_counts, previous_emotion_counts)

    # Assuming you have DataFrames or Series for current counts and changes
    # For example: current_sentiment_counts, sentiment_changes, current_emotion_counts, emotion_changes
    display_sentiments_emotions(current_interval_tweets, sentiment_emojis, 'sentiment', current_sentiment_counts, sentiment_changes)
    display_sentiments_emotions(current_interval_tweets, emotion_emojis, 'emotion', current_emotion_counts, emotion_changes)


# Load and sort news
filename = f"{ticker}_news.jsonl"
sorted_news = load_and_sort_news(filename)
st.sidebar.title(f"Recent News About {ticker}")

for news_item in sorted_news:
    content = news_item.get('content', False)
            
    # Generate a unique key for each news item
    key = f"{news_item['title']}_{news_item['published date']}"
    title_display = news_item['title']
    
    # If content is not available, append a red "X" to the title
    if not content:
        title_display += '&#10060;'  # Unicode for X mark
        content = "Content not availible."
    
    # Displaying each news item in the sidebar
    with st.sidebar.expander(f"{title_display} ({news_item['published date']}) ({news_item['publisher']['href']})", expanded=False):
            
        key_sentences = rank_sentences(content)
        sentences_html = "<br>".join(key_sentences)
        
        # Create a markdown link for the URL
        link = f"[Read More]({news_item['url']})"
        st.markdown(link, unsafe_allow_html=True)
        
        # Display content in a fixed-size container, allowing for scrolling
        st.markdown(f"**Content:**")
        st.markdown(sentences_html, unsafe_allow_html=True)
    
    # Divider for visual separation (placed outside the expander)
    st.sidebar.markdown("---")






















