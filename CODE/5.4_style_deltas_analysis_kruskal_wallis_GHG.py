import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import kruskal

nltk.download('stopwords')
nltk.download('wordnet')

# Load the grouped stylometric dataset - Grouped Features
stylo_df = pd.read_csv("5.4_stylometrics_report_grouped_features_GHG")

# Text preprocessing function for French
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove punctuation and numbers but retain accented characters
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', text)
    # Lowercase
    text = text.lower()
    # Tokenize, remove stopwords, and lemmatize
    stop_words = set(nltk.corpus.stopwords.words('french'))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Process the text data
stylo_df_2 = stylo_df[["id", "model", "source", "text", "author"]].copy()
stylo_df_2['processed_text'] = stylo_df_2['text'].apply(preprocess_text)

# Compute TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(stylo_df_2['processed_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Define stylistic features - Grouped
stylistic_features = ['Structural', 'NER', 'Punctuation', 'TAG', 'Letters', 
                      'Indexes', 'Numbers', 'Function words']

# Define feature columns
feature_columns = stylistic_features

# Initialize a list to store results
results = []

# Get unique authors
unique_authors = stylo_df['author'].unique()

if len(unique_authors) != 2:
    raise ValueError("Expected exactly two author classes for delta calculation.")

author_1, author_2 = unique_authors

# Loop through each feature to calculate delta and Kruskal-Wallis test
for feature in feature_columns:
    feature_data = stylo_df[[feature, 'author']].dropna()

    # Split by author
    data_author_1 = feature_data[feature_data['author'] == author_1][feature]
    data_author_2 = feature_data[feature_data['author'] == author_2][feature]

    # Calculate delta (mean difference)
    delta = data_author_1.mean() - data_author_2.mean()

    # Kruskal-Wallis Test for authors (Outlier T.)
    kruskal_stat, kruskal_p = kruskal(data_author_1, data_author_2)

    # Store results
    results.append({
        'Feature': feature,
        'Delta (Author 1 - Author 2)': f"{delta:.6f}",
        'K-W P-Value (Outlier T.)': f"{kruskal_p:.6f}"
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

results_df
