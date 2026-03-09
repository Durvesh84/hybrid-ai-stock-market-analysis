import ssl
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ==========================
# Fix SSL certificate issue
# ==========================
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================
# Download vader lexicon
# ==========================
nltk.download('vader_lexicon')

# ==========================
# Initialize sentiment model
# ==========================
sia = SentimentIntensityAnalyzer()

# ==========================
# Example financial news
# ==========================
news_list = [
    "Apple stock rises after strong quarterly earnings",
    "Tesla shares fall due to weak delivery numbers",
    "Microsoft reports record profits this quarter",
    "Global recession fears push markets lower"
]

# ==========================
# Analyze sentiment
# ==========================
for news in news_list:

    sentiment_score = sia.polarity_scores(news)

    print("\nNews:", news)
    print("Sentiment Score:", sentiment_score)

    if sentiment_score['compound'] >= 0.05:
        print("Sentiment: Positive")

    elif sentiment_score['compound'] <= -0.05:
        print("Sentiment: Negative")

    else:
        print("Sentiment: Neutral")