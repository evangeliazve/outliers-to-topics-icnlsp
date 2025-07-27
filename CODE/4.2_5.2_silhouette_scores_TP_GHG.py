import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.decomposition import PCA
from bertopic import BERTopic


df_fulltext = pd.DataFrame(columns=["text", "year_month"])

# Placeholder for results
results = []

# Model and dimensionality parameters
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
dimensionalities = [2, 3, 5, 10]
timeframes = sorted(df_fulltext['year_month'].unique())  # Unique timeframes

# Loop through models, dimensionalities, and reduction methods
for model_name in model_names:
    # Load the embedding model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df_fulltext['text'].tolist())
    
    for dim in dimensionalities:
        for method in ["UMAP", "PCA"]: 
            # Initialize dimensionality reduction
            if method == "UMAP":
                reducer = UMAP(n_components=dim, random_state=42)
            elif method == "PCA":
                reducer = PCA(n_components=dim)
            
            reduced_embeddings = reducer.fit_transform(embeddings)
            df_fulltext[[f'embedding_{i+1}' for i in range(dim)]] = reduced_embeddings
            
            # Initialize BERTopic
            topic_model = BERTopic(embedding_model=model, verbose=True, umap_model=reducer)
            
            # Cumulative storage for all data up to the current timeframe
            all_data = pd.DataFrame(columns=df_fulltext.columns)
            
            for timeframe in timeframes:
                # Add data for the current timeframe
                df_new_window = df_fulltext[df_fulltext['year_month'] == timeframe]
                all_data = pd.concat([all_data, df_new_window])
                
                if len(all_data) > 4:  
                    # Fit BERTopic cumulatively
                    topics, probs = topic_model.fit_transform(all_data['text'].tolist())
                    all_data['topic'] = topics
                    
                    # Calculate silhouette score
                    clustered_points = all_data[all_data['topic'] != -1]  # Exclude outliers
                    if len(clustered_points['topic'].unique()) > 1:
                        sil_score = silhouette_score(
                            clustered_points[[f'embedding_{i+1}' for i in range(dim)]],
                            clustered_points['topic']
                        )
                    else:
                        sil_score = np.nan
                else:
                    # Handle cases with insufficient data
                    all_data['topic'] = -1
                    sil_score = np.nan
                
                # Store results for the current timeframe
                num_clusters = all_data['topic'].nunique()
                results.append({
                    "model_name": model_name,
                    "dimensionality": dim,
                    "reduction_method": method,
                    "timeframe": timeframe,
                    "timeframe_silhouette_score": sil_score,
                    "number_of_clusters_timeframe": num_clusters
                })

            # Calculate global mean and median silhouette scores
            scores = [r["timeframe_silhouette_score"] for r in results if r["model_name"] == model_name and r["dimensionality"] == dim and r["reduction_method"] == method]
            global_mean = np.nanmean(scores)
            global_median = np.nanmedian(scores)
            
            # Update global scores in results
            for r in results:
                if r["model_name"] == model_name and r["dimensionality"] == dim and r["reduction_method"] == method:
                    r["global_mean_score"] = global_mean
                    r["global_median_score"] = global_median


results_df = pd.DataFrame(results)
