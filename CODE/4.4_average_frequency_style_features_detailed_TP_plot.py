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
stylo_df = pd.read_excel("4.4_stylometrics_report_detailed_features_TP.xlsx")

# Process text
stylo_df_2 = stylo_df[["id", "model", "source", "text", "author"]].copy()

# Define feature list
significant_features = ['avg_w_len', 'tot_short_w', 'tot_upper', 'avg_s_len', 'hapax', 'dis', 'syllable_count', 'avg_w_freqc', 'PER', 'ORG', 'LOC']

# Normalize features
scaler = MinMaxScaler()
stylo_df[significant_features] = scaler.fit_transform(stylo_df[significant_features])

# Compute deltas between author1 and author0
feature_deltas = stylo_df.groupby("author")[significant_features].mean().diff().iloc[-1].sort_values()

# Sort features by delta
sorted_features = feature_deltas.index.tolist()

# Melt DataFrame for plotting
data_melted = stylo_df.melt(id_vars=['author'], value_vars=sorted_features, var_name='Feature', value_name='Normalized Value')

data_melted['author'] = data_melted['author'].replace({'author_1': 'Verifying H', 'author_0': 'Not Verifying H'})

# Rename features for the plot (bold formatting only for those with asterisks)
feature_mapping = {
    'avg_w_len': r'$\mathbf{avg\_word\_length}\boldsymbol{**}$',
    'avg_s_len': r'$\mathbf{avg\_sentence\_length}\boldsymbol{**}$',
    'syllable_count': r'$\mathbf{syllable\_count}\boldsymbol{*}$',
    'avg_w_freqc': r'$\mathbf{avg\_word\_freqc}\boldsymbol{*}$',
    'ORG': r'$\mathbf{organisation\_freqc}\boldsymbol{**}$',
    'PER': r'$\mathbf{person\_freqc}\boldsymbol{**}$',
    'LOC' : 'location_freqc',
    'dis' : 'dis_legomena_ratio',
    'hapax' : 'hapax_legomena_ratio',
    'tot_upper' : 'uppercase_letters_count',
    'tot_short_w' : 'short_words_count'
}
data_melted['Feature'] = data_melted['Feature'].replace(feature_mapping)

# Plot
plt.figure(figsize=(14, 10))  # Increased width for more margin left and right
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
    size=4,
    marker='o',
    alpha=0.6,
    ax=ax,
    hue_order=hue_order
)

plt.xlabel('Subfeatures', fontsize=22)
plt.ylabel('Averaged Normalized Frequency', fontsize=22)
plt.ylim(-0.05, 1.25)
plt.xticks(rotation=45, fontsize=20, ha='right')
plt.yticks(np.arange(0, 1.1, 0.1), [f"{y:.1f}" for y in np.arange(0, 1.1, 0.1)], fontsize=20)

handles, labels = plt.gca().get_legend_handles_labels()
for handle in handles:
    handle.set_alpha(1)
    
legend = plt.legend(
    handles[:2], [r'Not Verifying $\mathcal{H}$', r'Verifying $\mathcal{H}$'], 
    title=None, 
    loc='upper right', 
    bbox_to_anchor=(1, 1),
    ncol=2, 
    fontsize=22,
    frameon=True  # Ensures legend has a box
)
legend.get_frame().set_alpha(1)  # Sets maximum opacity for legend box

plt.tight_layout()
plt.show()
