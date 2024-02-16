from collections import Counter
from flask import Flask, render_template, request, send_file
from flask import render_template_string
from bs4 import BeautifulSoup
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import requests
import matplotlib.dates as mdates
import csv
import re
import nltk
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize NLTK components
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        video_url = request.form['video_url']
        comments = get_youtube_comments(video_url)

        # Save comments to a CSV file
        csv_filename = 'comments.csv'
        save_comments_to_csv(comments, csv_filename)

        # Get sentiment data and top keywords for visualization
        sentiment_counts, positive_keywords, negative_keywords = get_sentiment_distribution(comments)

        # Create a sentiment pie chart
        create_pie_chart(sentiment_counts)

        # Create bar charts for top positive and negative keywords
        create_bar_chart(positive_keywords.most_common(), 'Top Positive Keywords')
        create_bar_chart(negative_keywords.most_common(), 'Top Negative Keywords')

        # Create sentiment over time plot
        create_time_series_plot(comments)


        return render_template('home_with_sentiment.html', comments=comments, csv_filename=csv_filename, show_chart_container=True)  

    return render_template('home_with_sentiment.html')

def remove_emojis(text):
    # Emoji pattern to remove emojis from the text
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def extract_keywords(text, num_keywords=1):
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [word.lower() for word in words if word.lower() not in stop_words]
    
    # Remove special characters and single-letter words
    words = [word for word in words if word.isalnum() and len(word) > 1]  # Only alphanumeric characters with length > 1
    
    # Count occurrences of each word
    word_counter = Counter(words)
    
    # Get the top N keywords
    top_keywords = word_counter.most_common(num_keywords)
    
    return top_keywords

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    filtered_text = [word for word in filtered_text if word.isalnum()]
    return ' '.join(filtered_text)

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']

    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

def get_sentiment_distribution(comments):
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    positive_keywords = Counter()
    negative_keywords = Counter()

    for comment in comments:
        sentiment = comment.get('sentiment', 'Neutral')
        sentiment_counts[sentiment] += 1

        # Count keywords for positive and negative sentiments
        if sentiment == 'Positive':
            positive_keywords.update(dict(comment['keywords']))
        elif sentiment == 'Negative':
            negative_keywords.update(dict(comment['keywords']))

    return sentiment_counts, positive_keywords, negative_keywords

def preprocess_comment(comment):
    comment['text'] = remove_emojis(comment['text'])
    comment['text'] = remove_stopwords(comment['text'].lower())
    comment['text'] = re.sub(r'<br>', '', comment['text'])  # Remove <br>
    comment['text'] = re.sub(r'br', '', comment['text'])  # Remove br
    comment['text'] = re.sub(r'\b\d+\b', '', comment['text'])  # Remove numeric values

    # Add sentiment analysis
    comment['sentiment'] = get_sentiment(comment['text'])

    # Extract keywords
    comment['keywords'] = extract_keywords(comment['text'], num_keywords=1)
    
    return comment

def get_youtube_comments(video_url, max_comments=100):
    video_id = video_url.split('v=')[1]
    api_url = f'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key=AIzaSyCvgehl8JBssaajbrGxOMHJIImUEQVeUN8'
    
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        if next_page_token:
            api_url += f'&pageToken={next_page_token}'

        response = requests.get(api_url)
        data = response.json()

        if 'items' not in data:
            break

        for item in data['items']:
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comment_time_str = item['snippet']['topLevelComment']['snippet']['publishedAt']
            
            # Convert timestamp to datetime object
            comment_time = datetime.strptime(comment_time_str, '%Y-%m-%dT%H:%M:%SZ')
            
            comments.append({'text': comment_text, 'time': comment_time, 'sentiment': None})

        if 'nextPageToken' in data:
            next_page_token = data['nextPageToken']
        else:
            break

    # Preprocess comments
    comments = [preprocess_comment(comment) for comment in comments]

    return comments[:max_comments]

def save_comments_to_csv(comments, csv_filename='comments.csv'):
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['time', 'text', 'sentiment', 'keywords']  # Include 'keywords' in fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for comment in comments:
            writer.writerow(comment)

    print(f'Comments saved to {csv_filename}')

def create_bar_chart(top_keywords, title, max_keywords=10):
    labels, values = zip(*top_keywords[:max_keywords])
    indexes = np.arange(len(labels))

    plt.bar(indexes, values, align='center', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(indexes, labels, rotation=45, ha='right')  
    
    plt.xlabel('Keywords')
    plt.ylabel('Frequency (Occurrences)')  # Add a label for the y-axis
    plt.title(title)
    plt.tight_layout()

    plt.savefig(f'static/images/{title.lower().replace(" ", "_")}_bar_chart.png')
    plt.close()

def create_pie_chart(sentiment_data):
    labels = sentiment_data.keys()
    sizes = sentiment_data.values()

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5733', '#808080'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Sentiment Distribution')
    plt.tight_layout()

    plt.savefig('static/images/sentiment_pie_chart.png')  # Save the chart as an image
    plt.close()

def create_time_series_plot(comments):
    time_series_data = {'Positive': [], 'Negative': [], 'Neutral': []}
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

    for comment in comments:
        sentiment = comment.get('sentiment', 'Neutral')
        sentiment_counts[sentiment] += 1
        time_series_data[sentiment].append((comment['time'], sentiment_counts[sentiment]))

    plt.figure(figsize=(12, 6))

    for sentiment, data in time_series_data.items():
        if data:
            # Sample every 5th data point (adjust as needed)
            data = data[::2]
            dates, values = zip(*data)
            plt.scatter(dates, values, label=sentiment)

    plt.xlabel('Time')
    plt.ylabel('Number of Comments')

    # Find the earliest timestamp
    min_time = min(min(data, key=lambda x: x[0])[0] for data in time_series_data.values())
    max_time = max(max(data, key=lambda x: x[0])[0] for data in time_series_data.values())

    # Calculate the number of days between min and max timestamps
    days_interval = (max_time - min_time).days

    # Set x-axis limits
    # plt.xlim(min_time + pd.DateOffset(months=2), max_time + pd.DateOffset(months=1))

    # Set the x-axis locator to have intervals based on the number of days
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=days_interval // 10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Set the y-axis interval to 5 comments
    plt.yticks(np.arange(0, max(sentiment_counts.values()), 10))

    plt.legend()
    plt.tight_layout()
    plt.savefig('static/images/sentiment_over_time.png')
    plt.close()

@app.route('/download_csv')
def download_csv():
    csv_filename = request.args.get('csv_filename', 'comments.csv')
    return send_file(csv_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)