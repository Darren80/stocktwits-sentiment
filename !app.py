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
st.set_page_config(layout="wide")
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
    
def parse_date(date_str):
    # Handles both 'date' and 'created_at' formats with timezone awareness
    if date_str.endswith('Z'):
        # If the string ends with 'Z', it's UTC. Replace 'Z' with '+00:00' for consistency
        date_str = date_str.rstrip('Z') + '+00:00'
    # Parse the datetime string into a timezone-aware datetime object
    return datetime.fromisoformat(date_str).astimezone(pytz.utc)
    
# Function to load recent tweets
def load_tweets_by_timeframe(file_paths, start_time, end_time):
    timeframe_tweets = []
    for file_path in file_paths:  # Iterate over both file paths
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                tweet = json.loads(line)
                # Check for 'date' or 'created_at', assuming 'date' takes precedence
                tweet_time_str = tweet.get('date') or tweet.get('created_at')
                if tweet_time_str:
                    tweet_time = parse_date(tweet_time_str)
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
    
    st.markdown(f"### {category.capitalize()} Overview")

    # Display an overview of sentiments/emotions
    overview_html = ""
    for sentiment, emoji in emoji_dict.items():
        count = current_counts.get(sentiment, 0)
        change = changes.get(sentiment, 0)
        arrow = "↗️" if change > 0 else "↘️" if change < 0 else "➖"
        color = "color:green;" if change > 0 else "color:red;" if change < 0 else ""
        overview_html += f"{emoji} {sentiment.capitalize()}: <span style='{color}'>{count} {arrow}</span><br>"

    st.markdown(overview_html, unsafe_allow_html=True)

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
                        for _, tweet in filtered_tweets.iterrows():
                            # Constructing a tweet display format
                            tweet_display = f"""
                            <div style="border: 1px solid #e1e4e8; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
                                <p style="color: #bbb; margin-bottom: 2px;">{tweet['date']}</p>
                                <p style="font-size: 16px; margin-bottom: 2px;">{tweet.get('rawConent', tweet['cleanContent'])}</p>
                                <a href="{tweet['url']}" target="_blank">View Tweet</a>
                            </div>
                            """
                            cols[j].markdown(tweet_display, unsafe_allow_html=True)
                    else:
                        st.write("No tweets found.")

def read_and_count_sentiments(file_paths, lookback_hours=24):
    sentiments_count = {'bullish': 0, 'bearish': 0}
    lookback_limit = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    tweet = json.loads(line)
                    created_at = datetime.strptime(tweet.get('created_at'), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                    if created_at >= lookback_limit:
                        sentiment = tweet.get('entities', {}).get('sentiment', {}).get('basic', '').lower()
                        if sentiment in sentiments_count:
                            sentiments_count[sentiment] += 1
                except json.JSONDecodeError:
                    pass  # Ignore lines that can't be decoded
                except Exception as e:
                    print(f"Error processing tweet: {e}")
    return sentiments_count

def display_sentiments(sentiments_count):
    st.title("Sentiment Analysis - Past 24 hours")
    st.write("### Bullish Sentiments")
    st.write(sentiments_count['bullish'])
    st.write("### Bearish Sentiments")
    st.write(sentiments_count['bearish'])

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
    'strong positive': '😁',  # Beaming face with smiling eyes
    'moderately positive': '😊',  # Smiling
    'mildly positive': '🙂',  # Slightly smiling
    'neutral': '😐',  # Neutral face
    'mildly negative': '🙁',  # Slightly frowning
    'moderately negative': '😠',  # Angry
    'strong negative': '😡',  # Very angry or upset
}

def time_period(time_window=60):
    # Calculate the current time and subtract 6 hours to define the start of the range
    current_time = datetime.now(pytz.utc)  # Using now() with timezone awareness
    
    # Define the number of 30-minute intervals in a 6-hour range
    total_intervals = 12  # 6 hours * 2 intervals per hour

    # Generate labels for each interval
    interval_labels = [f"{time_window * i} - {time_window * (i + 1)} minutes in the past" for i in range(total_intervals)]
    interval_labels.reverse()  # Reverse so that the most recent interval is first

    # User selects an interval using the slider
    selected_interval_label = st.sidebar.select_slider("Select Time Interval", options=interval_labels, value=interval_labels[-1])

    # Determine the selected interval index based on the chosen label
    selected_interval_index = interval_labels.index(selected_interval_label)

    # Calculate the start and end times for the selected interval
    interval_start = current_time - timedelta(minutes=time_window * (total_intervals - selected_interval_index))
    interval_end = interval_start + timedelta(minutes=time_window)

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
file_paths = [f'backup/final_{ticker}_tweets.jsonl', f'backup/final_{ticker}_stocktweets.jsonl']

# Load tweets for each interval
current_interval_tweets = pd.DataFrame(load_tweets_by_timeframe(file_paths, current_interval_start, current_interval_end))
previous_interval_tweets = pd.DataFrame(load_tweets_by_timeframe(file_paths, previous_interval_start, previous_interval_end))

st.header('Summary')
st.write(f'Number of Tweets in Current Interval: {len(current_interval_tweets)}')
st.write(f'Number of Tweets in Previous Interval: {len(previous_interval_tweets)}')

# Raw Data
sentiments_count = read_and_count_sentiments(['backup/trash_TSLA_stocktweets.jsonl', 'backup/TSLA_stocktweets.jsonl'])
display_sentiments(sentiments_count)

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

if not current_interval_tweets.empty: # and not previous_interval_tweets.empty:
    current_sentiment_counts, current_emotion_counts = calculate_counts(current_interval_tweets)
    print(current_emotion_counts)
    print(current_sentiment_counts)
    if previous_interval_tweets.empty:
        previous_sentiment_counts = None 
        previous_emotion_counts = None
    else:
        previous_sentiment_counts, previous_emotion_counts = calculate_counts(previous_interval_tweets)

    sentiment_changes = calculate_changes(current_sentiment_counts, previous_sentiment_counts)
    emotion_changes = calculate_changes(current_emotion_counts, previous_emotion_counts)

    display_sentiments_emotions(current_interval_tweets, sentiment_emojis, 'sentiment', current_sentiment_counts, sentiment_changes)
    display_sentiments_emotions(current_interval_tweets, emotion_emojis, 'emotion', current_emotion_counts, emotion_changes)


# Load and sort news
filename = f"./news/{ticker}_news.jsonl"
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
        
    # Inject custom CSS to reduce padding
    st.sidebar.markdown("""
    <style>
        .css-1d391kg {padding-top: 0rem; padding-bottom: 0rem;}
        .css-1outpf7 {padding-top: 0rem; padding-bottom: 0rem;}
        .st-cm {margin-bottom: 0rem;}
        .css-18e3th9 {padding-bottom: 0rem !important;}
    </style>
    """, unsafe_allow_html=True)
    
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






















