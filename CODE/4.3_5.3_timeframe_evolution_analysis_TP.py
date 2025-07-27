
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic import BERTopic


# Function analysing outlier evolution over time
def analyze_outliers_and_topics_cumulative(df, model_name, dimentionality, column_to_process="text"):
    """
    Analyze outliers in each time window cumulatively using UMAP,
    and track transitions to final topics.
    """
    if column_to_process not in df.columns:
        raise ValueError(f"Column '{column_to_process}' not found in DataFrame.")

    # Load the embedding model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[column_to_process].tolist(), show_progress_bar=True)

    # Dimensionality reduction with UMAP
    umap_reducer = UMAP(n_components=dimentionality, random_state=42)
    embeddings = umap_reducer.fit_transform(embeddings)

    # Assign embeddings to the DataFrame
    for i in range(dimentionality):
        df[f'embedding_dim_{i+1}'] = embeddings[:, i]

    # Sort time windows
    time_windows = sorted(df['year_month'].unique())

    # Initialize BERTopic
    topic_model = BERTopic(embedding_model=model, umap_model=umap_reducer, verbose=True)

    # Assign final topics for the entire dataset
    final_topics, _ = topic_model.fit_transform(df[column_to_process].tolist())
    df['final_topic'] = final_topics

    # Calculate outliers in final topics
    final_outliers = set(df[df['final_topic'] == -1].index)

    # Cumulative data storage
    cumulative_data = pd.DataFrame(columns=df.columns)
    results = []
    cumulative_observations_count = 0

    for idx, time_window in enumerate(time_windows):
        # Add current time window data to cumulative data
        df_new_window = df[df['year_month'] == time_window]
        cumulative_data = pd.concat([cumulative_data, df_new_window])

        # Assign topics to cumulative data
        if len(cumulative_data) <= 4:
            cumulative_data['topic'] = -1  # All outliers if too few data points
        else:
            topics, _ = topic_model.fit_transform(cumulative_data[column_to_process].tolist())
            cumulative_data['topic'] = topics

        # Determine whether this is the last time window
        is_last_window = (idx == len(time_windows) - 1)

        # Select the appropriate column for stats in the last window
        stats_column = 'final_topic' if is_last_window else 'topic'

        # Track outliers in the current cumulative dataset
        current_outliers = set(cumulative_data[cumulative_data[stats_column] == -1].index)

        # Transitioned outliers: those that are no longer outliers in the final assignment
        transitioned_outliers = (
            [idx for idx in current_outliers if df.loc[idx, 'final_topic'] != -1]
            if not is_last_window else []
        )

        # Transition ratio
        transition_ratio = (
            round(len(transitioned_outliers) / len(current_outliers), 4)
            if len(current_outliers) > 0 and not is_last_window
            else None  # Not applicable for the last time window
        )

        # Outlier ratio
        total_observations = len(df_new_window)
        outlier_ratio = (
            round(len(current_outliers) / cumulative_observations_count, 4)
            if cumulative_observations_count > 0
            else 0
        )

        # Number of unique topics (excluding outliers) in the cumulative dataset
        num_topics = cumulative_data[stats_column][cumulative_data[stats_column] != -1].nunique()

        cumulative_observations_count += len(df_new_window)

        # Append results for the current time window
        result = {
            "Model Name": model_name,
            "Time Window": time_window,
            "Number of Observations Treated (Cumulative)": cumulative_observations_count,
            "Number of Outliers": len(current_outliers),
            "Outlier Ratio": outlier_ratio,
            "Number of Topics": num_topics,
        }

        # Add transition-related metrics only if not the last time window
        if not is_last_window:
            result.update({
                "Number Transitioned to Final Topics": len(transitioned_outliers),
                "Transition Ratio to Final Topics": transition_ratio,
            })

        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df



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

df_fulltext = pd.DataFrame() # to be defined

final_results = []

column_to_process = "text"  
dimentionality = 10

for model_name in model_names:
    model_results = analyze_outliers_and_topics_cumulative(df_fulltext, model_name, dimentionality, column_to_process=column_to_process)
    final_results.append(model_results)

# Combine results
final_df = pd.concat(final_results, ignore_index=True)