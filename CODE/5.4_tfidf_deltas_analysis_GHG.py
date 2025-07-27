import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import kruskal
from collections import Counter

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stylo_df_grouped = pd.read_excel("5.4_stylometrics_report_grouped_features_GHG.xlsx")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove punctuation and numbers but retain accented characters
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', text)
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Define a custom list of additional stopwords
    additional_stopwords = []
    
    # Combine NLTK stopwords with the additional custom stopwords
    stop_words = set(stopwords.words('english')).union(set(additional_stopwords))
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Return processed text
    return ' '.join(tokens)


# Process the data
stylo_df_2 = stylo_df_grouped[["id", "model", "source", "text", "author"]].copy()
stylo_df_2['processed_text'] = stylo_df_2['text'].apply(preprocess_text)

# Compute TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(stylo_df_2['processed_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Process the data
stylo_df_2 = stylo_df_grouped[["id", "model", "source", "text", "author"]].copy()
stylo_df_2['processed_text'] = stylo_df_2['text'].apply(preprocess_text)

# Compute TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(stylo_df_2['processed_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Merge the original data with the TF-IDF scores
# Ensure the 'processed_text' column is included in the merged DataFrame
stylo_df_2 = stylo_df_2.reset_index(drop=True)

merged_df = pd.merge(stylo_df_2, tfidf_df, left_index=True, right_index=True)

# Initialize a list to store word-specific results
word_results = []

# Loop through each word in the TF-IDF matrix
for word in tfidf_df.columns:
    if word not in merged_df.columns:
        print(f"Warning: Word '{word}' not found in merged DataFrame.")
        continue

    # Group by author and calculate the delta of the TF-IDF scores
    author_0_scores = merged_df[merged_df['author_x'] == "author_0"][word].values
    author_1_scores = merged_df[merged_df['author_x'] == "author_1"][word].values
    
    # Check if any group is empty
    if len(author_0_scores) == 0 or len(author_1_scores) == 0:
        print(f"Skipping word '{word}' due to empty author group.")
        word_results.append({
            'Word': word,
            'Delta (TF-IDF Author 1 - Author 0)': np.nan,
            'Delta (Occurrences Author 1 - Author 0)': np.nan,
            'Occurrences Author 0': np.nan,
            'Occurrences Author 1': np.nan,
            'K-W P-Value (TF-IDF Author)': np.nan
        })
        continue

    # Compute the delta between the two authors' TF-IDF scores
    delta_tfidf = np.mean(author_1_scores) - np.mean(author_0_scores)

    # Calculate the frequency of the word for each author
    author_0_text = " ".join(merged_df[merged_df['author_x'] == "author_0"]['processed_text'])
    author_1_text = " ".join(merged_df[merged_df['author_x'] == "author_1"]['processed_text'])
    
    author_0_freq = Counter(author_0_text.split())[word]
    author_1_freq = Counter(author_1_text.split())[word]
    
    # Compute the delta of word occurrences between the authors
    delta_occurrences = author_1_freq - author_0_freq

    # Apply Kruskal-Wallis test between the two authors' distributions
    try:
        kruskal_stat, kruskal_p = kruskal(author_1_scores, author_0_scores)
    except ValueError as e:
        # If there is an issue with the test, such as identical distributions
        print(f"Error with Kruskal-Wallis test for word '{word}': {e}")
        kruskal_p = np.nan

    # Append results for this word
    word_results.append({
        'Word': word,
        'Delta (TF-IDF Author 1 - Author 0)': delta_tfidf,
        'Delta (Occurrences Author 1 - Author 0)': delta_occurrences,
        'Occurrences Author 0': author_0_freq,
        'Occurrences Author 1': author_1_freq,
        'K-W P-Value (TF-IDF Author)': kruskal_p
    })

# Convert results to a DataFrame
word_results_df = pd.DataFrame(word_results)