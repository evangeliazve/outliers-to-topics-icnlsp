import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic import BERTopic


def analyze_outlier_transitions(df, dimentionality, model_name, column_to_process="title"):
    """
    Calculate outlier transitions for each text in cumulative time windows using UMAP and mark outlier-to-topic transitions.
    """
    if column_to_process not in df.columns:
        raise ValueError(f"Column '{column_to_process}' not found in DataFrame.")
    
    # Load the embedding model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[column_to_process].tolist(), show_progress_bar=True)

    # Dimensionality reduction with 10D UMAP
    umap_reducer = UMAP(n_components=10, random_state=42)
    embeddings_10d = umap_reducer.fit_transform(embeddings)

    # Add 10D embedding columns to DataFrame
    for i in range(10):
        df[f'embedding_{i}'] = embeddings_10d[:, i]

    # Sort time windows
    time_windows = sorted(df['year_month'].unique())

    # Initialize BERTopic with 10D UMAP embeddings
    topic_model = BERTopic(embedding_model=model, umap_model=umap_reducer, verbose=True)

    # Assign final topics for the entire dataset
    final_topics, _ = topic_model.fit_transform(df[column_to_process].tolist())
    df['final_topic'] = final_topics

    # Determine final outliers
    final_outliers = set(df[df['final_topic'] == -1].index)

    # Initialize cumulative data
    cumulative_data = pd.DataFrame(columns=df.columns)

    # Track the outlier and topic status for each text
    transition_results = {idx: 'N/A' for idx in df.index}

    for time_window in time_windows:
        # Add data for the current time window to cumulative data
        df_new_window = df[df['year_month'] == time_window]
        cumulative_data = pd.concat([cumulative_data, df_new_window])

        # Assign topics to cumulative data
        if len(cumulative_data) <= 4:
            cumulative_data['topic'] = -1  # Mark all as outliers if too few data points
        else:
            topics, _ = topic_model.fit_transform(cumulative_data[column_to_process].tolist())
            cumulative_data['topic'] = topics

        # Identify current outliers in cumulative data
        current_outliers = set(cumulative_data[cumulative_data['topic'] == -1].index)

        for idx in df_new_window.index:
            if idx in current_outliers and idx not in final_outliers:
                transition_results[idx] = "1"  # Became a topic
            elif idx in current_outliers and idx in final_outliers:
                transition_results[idx] = "0"  # Remained an outlier

    # Add the results to the DataFrame
    df[f"{model_name}_outlier_status"] = df.index.map(transition_results)

    return df[[column_to_process, f"{model_name}_outlier_status"]]

# Main Execution
model_names = [
    "OrdalieTech/Solon-embeddings-large-0.1",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "dangvantuan/sentence-camembert-base",
    "intfloat/multilingual-e5-large",
    "FacebookAI/xlm-roberta-large",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-roberta-large-v1",
    "distilbert/distilbert-base-uncased",
    "intfloat/e5-base-v2"
]

df_fulltext = pd.DataFrame() # to set 

all_model_results = []

column_to_process = "text"
dimentionality = 10

for model_name in model_names:
    updated_df = analyze_outlier_transitions(df_fulltext, model_name, dimentionality,column_to_process=column_to_process)
    all_model_results.append(updated_df)

# Combine results from all models
final_combined_df = pd.concat(all_model_results, axis=1)
final_combined_df = final_combined_df.drop(["text.1", "text.2", "text.3", "text.4", "text.5", "text.6", "text.7", "text.8"], axis=1)