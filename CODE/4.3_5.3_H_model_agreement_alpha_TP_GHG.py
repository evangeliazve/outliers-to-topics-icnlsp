
import pandas as pd
import numpy as np

### For TP
final_combined_df = pd.read_excel("4.3_outlier_to_topic_transitions_model_agreement_TP.xlsx")

### For GHG
#final_combined_df = pd.read_excel("5.3_outlier_to_topic_transitions_model_agreement_GHG.xlsx")


def measure_annotations_agreement(df, model_names):
    """
    Compute annotation agreement between models for each document.
    """
    # Initialize H and alpha columns
    df['H'] = 0.0
    df['alpha'] = 0.0

    # Iterate over each document
    for idx in df.index:
        # Collect all annotations (model results) for the document
        annotations = [df.loc[idx, f"{model}_outlier_status"] for model in model_names]
        
        # Filter out 'N/A' annotations
        filtered_annotations = [ann for ann in annotations if ann != 'N/A']
        
        if 1 not in filtered_annotations:
            # Set values to NaN if no annotation equals 1
            df.loc[idx, 'H'] = None
            df.loc[idx, 'alpha'] = None
        else:
            # Standard calculation for H2 method
            total_annotations = len(filtered_annotations)
            count_label = filtered_annotations.count(1)
            ratio = count_label / total_annotations if total_annotations > 0 else 0
            df.loc[idx, 'H'] = ratio
            df.loc[idx, 'alpha'] = abs(2 * ratio - 1)  # Rescaled value
    
    return df


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

# Apply the measure_annotations_agreement function
final_combined_df_with_agreement = measure_annotations_agreement(final_combined_df, model_names)