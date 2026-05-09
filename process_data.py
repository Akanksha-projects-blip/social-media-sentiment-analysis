# ================================
# SOCIAL MEDIA DATA CLEANING SCRIPT
# ================================

# Install required libraries first:
# pip install pandas numpy nltk emoji openpyxl

import pandas as pd
import numpy as np
import re
import string
import nltk
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ================================
# DOWNLOAD NLTK FILES
# ================================

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ================================
# LOAD DATASET
# ================================

df = pd.read_csv(
    'twitter_training.csv',
    header=None,
    names=['ID', 'Brand', 'Sentiment', 'Tweet']
)

print("Original Dataset Shape:", df.shape)

# ================================
# REMOVE MISSING VALUES
# ================================

df.dropna(subset=['Tweet'], inplace=True)

# ================================
# REMOVE DUPLICATES
# ================================

df.drop_duplicates(subset=['Tweet'], inplace=True)

# ================================
# RESET INDEX
# ================================

df.reset_index(drop=True, inplace=True)

# ================================
# TEXT CLEANING SETUP
# ================================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ================================
# CLEANING FUNCTION
# ================================

def clean_text(text):

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)

    # Remove mentions (@username)
    text = re.sub(r'@\\w+', '', text)

    # Remove hashtag symbol (#)
    text = re.sub(r'#', '', text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove numbers
    text = re.sub(r'\\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = re.sub(r'\\s+', ' ', text).strip()

    # Tokenize words
    words = text.split()

    # Remove stopwords and apply lemmatization
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
        and word.isalpha()
        and len(word) > 2
    ]

    # Join cleaned words
    return ' '.join(cleaned_words)

# ================================
# APPLY CLEANING
# ================================

df['Clean_Tweet'] = df['Tweet'].apply(clean_text)

# ================================
# REMOVE EMPTY CLEANED TWEETS
# ================================

df = df[df['Clean_Tweet'].str.strip() != '']

# ================================
# STANDARDIZE SENTIMENT LABELS
# ================================

df['Sentiment'] = df['Sentiment'].str.lower()

sentiment_map = {
    'positive': 'Positive',
    'negative': 'Negative',
    'neutral': 'Neutral',
    'irrelevant': 'Irrelevant'
}

df['Sentiment'] = df['Sentiment'].map(sentiment_map)

# ================================
# FINAL DATASET INFO
# ================================

print("Cleaned Dataset Shape:", df.shape)

print("\nSample Cleaned Data:\n")
print(df[['Tweet', 'Clean_Tweet']].head())

# ================================
# SAVE CLEANED DATA TO EXCEL
# ================================

output_file = 'cleaned_twitter_data.xlsx'

df.to_excel(output_file, index=False)

print(f"\nCleaned dataset saved successfully as: {output_file}")