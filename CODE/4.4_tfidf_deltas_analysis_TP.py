from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import ttest_ind, mannwhitneyu, spearmanr, pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob_fr import PatternTagger, PatternAnalyzer
from textblob import Blobber
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

stylo_df_grouped = pd.read_csv("4.4_stylometrics_report_grouped_features_TP")


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
    additional_stopwords = [
        'exemple', 'mot', 'phrase', 'texte', 'maintenant', 'très', 'aussi', 'encore', 'beaucoup', 
        'cela', 'cela', 'voici', 'voilà', 'quelques', 'ceci', 'là', 'tout', 'tout', 'tous', 'autres', 'déjà', 'nom', 'sujet',
        'groupe', 'depuis', 'plus', 'ça', 'ur', 'va', 'or', 'hors', 'objet'
    ]
    
    # Combine NLTK stopwords with the additional custom stopwords
    stop_words = set(stopwords.words('french')).union(set(additional_stopwords))
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Return processed text
    return ' '.join(tokens)

# Initialize Blobber for French sentiment and subjectivity analysis
tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())

# Initialize VADER analyzer for French
french_analyzer = SentimentIntensityAnalyzer()

def compute_sentiment_subjectivity_neutrality(text):
    """
    Compute polarity (sentiment), subjectivity, and neutrality for French text.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0, 0.0  # Neutral sentiment, subjectivity, and neutrality for invalid or empty text
    
    # Use TextBlob for sentiment and subjectivity
    blob = tb(text)
    sentiment = blob.sentiment[0]  # Polarity
    subjectivity = blob.sentiment[1]
    
    # Use VADER for neutrality (French-specific VADER)
    vader_scores = french_analyzer.polarity_scores(text)
    neutrality = vader_scores['neu']  # Neutrality score
    
    return sentiment, subjectivity, neutrality

# Compute and assign sentiment, subjectivity, and neutrality scores for each document
stylo_df_grouped[['sentiment', 'subjectivity', 'neutrality']] = stylo_df_grouped['text'].apply(
    lambda text: pd.Series(compute_sentiment_subjectivity_neutrality(text))
)

# Select necessary columns and preprocess texts
stylo_df_2 = stylo_df_grouped[["id", "model", "text", "author", "sentiment", "subjectivity", "neutrality"]].copy()

# Rename columns to avoid collision
stylo_df_2.rename(columns={
    'sentiment': 'doc_sentiment', 
    'subjectivity': 'doc_subjectivity', 
    'neutrality': 'doc_neutrality'
}, inplace=True)

stylo_df_2['processed_text'] = stylo_df_2['text'].apply(preprocess_text)

# Compute TF-IDF for the processed text
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(stylo_df_2['processed_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Merge original DataFrame with TF-IDF scores using suffixes to handle column collisions
merged_df = pd.merge(
    stylo_df_2, 
    tfidf_df, 
    left_index=True, 
    right_index=True, 
    suffixes=('_orig', '')
)

word_results = []

# Loop through each word in the TF-IDF matrix to compute required metrics
for word in tfidf_df.columns:
    author_0_scores = merged_df[merged_df['author'] == "author_0"][word].values
    author_1_scores = merged_df[merged_df['author'] == "author_1"][word].values
    
    # Skip if any author group is empty
    if len(author_0_scores) == 0 or len(author_1_scores) == 0:
        continue

    # Compute TF-IDF delta between authors
    delta_tfidf = np.mean(author_1_scores) - np.mean(author_0_scores)

    # Calculate word frequency for each author
    author_0_text = " ".join(merged_df[merged_df['author'] == "author_0"]['processed_text'])
    author_1_text = " ".join(merged_df[merged_df['author'] == "author_1"]['processed_text'])
    
    author_0_freq = Counter(author_0_text.split())[word]
    author_1_freq = Counter(author_1_text.split())[word]
    
    # Compute delta in occurrences between authors
    delta_occurrences = author_1_freq - author_0_freq

    # Apply Kruskal-Wallis test
    try:
        _, kruskal_p = kruskal(author_0_scores, author_1_scores)
    except ValueError:
        kruskal_p = np.nan

    # Calculate mean sentiment, subjectivity, and neutrality for documents containing the current word
    pattern = re.compile(r'\b{}\b'.format(re.escape(word)))
    docs_containing_word = merged_df[merged_df['processed_text'].apply(lambda txt: bool(pattern.search(txt)))]
    if not docs_containing_word.empty:
        mean_sentiment = docs_containing_word['doc_sentiment'].mean()
        mean_subjectivity = docs_containing_word['doc_subjectivity'].mean()
        mean_neutrality = docs_containing_word['doc_neutrality'].mean()
    else:
        mean_sentiment = np.nan
        mean_subjectivity = np.nan
        mean_neutrality = np.nan

    # Append metrics for the current word
    word_results.append({
        'Word': word,
        'Delta (TF-IDF Author 1 - Author 0)': delta_tfidf,
        'Delta (Occurrences Author 1 - Author 0)': delta_occurrences,
        'Occurrences Author 0': author_0_freq,
        'Occurrences Author 1': author_1_freq,
        'K-W P-Value (TF-IDF Author)': kruskal_p,
        'Mean Sentiment Score': mean_sentiment,
        'Mean Subjectivity Score': mean_subjectivity,
        'Mean Neutrality Score': mean_neutrality
    })

# Convert list of dictionaries to DataFrame
word_results_df = pd.DataFrame(word_results)


# --- Find Correlation between Delta TF-IDF and Subjectivity/Neutrality Scores ---
# Calculate correlation for subjectivity score
subjectivity_corr, subjectivity_pval = spearmanr(
    word_results_df['Delta (TF-IDF Author 1 - Author 0)'],
    word_results_df['Mean Subjectivity Score']
)

# Calculate correlation for neutrality score
neutrality_corr, neutrality_pval = spearmanr(
    word_results_df['Delta (TF-IDF Author 1 - Author 0)'],
    word_results_df['Mean Neutrality Score']
)

print(f"Spearman correlation (Delta TF-IDF vs Subjectivity): {subjectivity_corr}, p-value: {subjectivity_pval}")
print(f"Spearman correlation (Delta TF-IDF vs Neutrality): {neutrality_corr}, p-value: {neutrality_pval}")