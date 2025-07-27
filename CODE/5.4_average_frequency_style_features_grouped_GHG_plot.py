import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
stylo_df_grouped = pd.read_excel("5.4_stylometrics_report_grouped_features_GHG.xlsx")
# Removing outliers beyond 1.5 * IQR
Q1 = stylo_df_grouped['Numbers'].quantile(0.25)
Q3 = stylo_df_grouped['Numbers'].quantile(0.75)
IQR = Q3 - Q1
stylo_df_grouped = stylo_df_grouped[
    (stylo_df_grouped['Numbers'] >= (Q1 - 1.5 * IQR)) &
    (stylo_df_grouped['Numbers'] <= (Q3 + 1.5 * IQR))
]

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('french'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Process text
stylo_df_2 = stylo_df_grouped[["id", "model", "source", "text", "author"]].copy()
stylo_df_2['processed_text'] = stylo_df_2['text'].apply(preprocess_text)

# Compute TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(stylo_df_2['processed_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Define feature list
significant_features = ['NER', 'Structural', 'TAG', 'Numbers', 'Punctuation', 'Letters', 'Indexes', 'Function words']

# Compute mean TF-IDF
tfidf_mean = tfidf_df.mean(axis=1).rename("TF-IDF")
stylo_df_grouped['TF-IDF'] = tfidf_mean

# Normalize features
scaler = MinMaxScaler()
stylo_df_grouped[significant_features] = scaler.fit_transform(stylo_df_grouped[significant_features])

# Compute deltas between author1 and author0
feature_deltas = stylo_df_grouped.groupby("author")[significant_features].mean().diff().iloc[-1].sort_values()

# Sort features by delta
sorted_features = feature_deltas.index.tolist()

# Melt DataFrame for plotting
data_melted = stylo_df_grouped.melt(id_vars=['author'], value_vars=sorted_features, var_name='Feature', value_name='Normalized Value')

data_melted['author'] = data_melted['author'].replace({'author_1': 'Verifying H', 'author_0': 'Not Verifying H'})

# Rename features for the plot (bold formatting)
feature_mapping = {
    'Letters': r'$\mathbf{Letters}\boldsymbol{*}$',
    'Function words': r'$\mathbf{Function\ words}\boldsymbol{**}$'
}
data_melted['Feature'] = data_melted['Feature'].replace(feature_mapping)

# Plot
plt.figure(figsize=(14, 10))
hue_order = ['Not Verifying H', 'Verifying H']
ax = sns.boxplot(
    data=data_melted,
    x='Feature',
    y='Normalized Value',
    hue='author',
    palette=['red', 'navy'],
    dodge=True,
    showfliers=False,
    boxprops={'alpha': 0.5},
    whiskerprops={'alpha': 1},
    capprops={'alpha': 1},
    medianprops={'alpha': 1},
    hue_order=hue_order
)

sns.stripplot(
    data=data_melted,
    x='Feature',
    y='Normalized Value',
    hue='author',
    palette=['red', 'navy'],
    dodge=True,
    size=3,
    marker='o',
    alpha=0.4,
    ax=ax,
    hue_order=hue_order
)

plt.xlabel('Features', fontsize=22)
plt.ylabel('Averaged Normalized Frequency', fontsize=22)
plt.ylim(-0.05, 1.25)
plt.xticks(rotation=45, fontsize=20, ha='right')
plt.yticks(np.arange(0, 1.1, 0.1), [f"{y:.1f}" for y in np.arange(0, 1.1, 0.1)], fontsize=20)

handles, labels = plt.gca().get_legend_handles_labels()
for handle in handles:
    handle.set_alpha(1)
plt.legend(
    handles[:2], [r'Not Verifying $\mathcal{H}$', r'Verifying $\mathcal{H}$'],  # Ensure correct LaTeX formatting, 
    title=None, 
    loc='upper right', 
    bbox_to_anchor=(1, 1),
    ncol=2,
    fontsize=22
)
plt.tight_layout()
plt.show()