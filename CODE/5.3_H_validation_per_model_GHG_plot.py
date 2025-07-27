import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load model agreement file
final_combined_df_with_agreement = pd.read_excel("5.3_outlier_to_topic_transitions_model_agreement_GHG.xslx")

# Identify model columns
model_columns = [col for col in final_combined_df_with_agreement.columns if col.endswith("_outlier_status")]

# Calculate total count of values equal to 1 and 0 across all model columns
total_count_1 = final_combined_df_with_agreement[model_columns].eq(1).sum().sum()
total_count_0 = final_combined_df_with_agreement[model_columns].eq(0).sum().sum()

# Compute the global ratio
global_ratio = total_count_1 / (total_count_1 + total_count_0) if (total_count_1 + total_count_0) > 0 else 0

# Output results
total_count_1, total_count_0, global_ratio


def plot_ratios_custom(df, model_names, custom_labels, colors, custom_order):
    # Check if all model names are in the colors dictionary
    missing_colors = [model for model in model_names if model not in colors]
    if missing_colors:
        print(f"Warning: The following models are missing colors in the dictionary: {', '.join(missing_colors)}")
        # Assign a default color for missing models
        for model in missing_colors:
            colors[model] = '#D3D3D3'  # Assigning gray color to missing models

    # Extract full model names after the last "/"
    legend_labels = [label.split('/')[-1] for label in custom_labels]  # Full names for legend

    # Prepare data for plotting
    ratios = []

    for model in model_names:
        # Filter data for each model column
        column = f"{model}_outlier_status"  # Assuming model columns end with _outlier_status

        # Count rows with 1 and 0
        count_1 = (df[column] == 1).sum()
        count_0 = (df[column] == 0).sum()
        total = count_1 + count_0

        # Calculate ratio of rows equal to 1
        ratio_validating = count_1 / total if total > 0 else 0
        ratios.append(ratio_validating)

    # Sort data based on the specified custom order
    sorted_indices = [model_names.index(model) for model in custom_order]
    ratios = [ratios[i] for i in sorted_indices]
    legend_labels = [legend_labels[i] for i in sorted_indices]
    model_names = [model_names[i] for i in sorted_indices]

    # Map colors to models using the color dictionary
    model_color_mapping = {model: colors[model] for model in model_names}
    bar_colors = [model_color_mapping[model] for model in model_names]

    # Create the bar chart
    x_positions = np.linspace(0, len(legend_labels) - 1, len(legend_labels))  # Closer spacing
    bar_width = 0.75  # Reduced bar width for a more compact look

    plt.figure(figsize=(10, 6))  # Customized size

    for i, (pos, ratio, color) in enumerate(zip(x_positions, ratios, bar_colors)):
        plt.bar(pos, ratio, color=color, width=bar_width, edgecolor="black", label=legend_labels[i])
        # Add annotations (values) in the middle of the bars
        plt.text(
            pos, 
            ratio / 2,  # Position in the middle of the bar
            f"{ratio:.2f}",
            ha="center",
            va="center",
            fontsize=13.5,
            color="black"
        )

    # Add 'Avg. Model' text in the middle of the plot
    plt.text(len(legend_labels) / 5, 1.5, "Avg. Model = 0.81", 
             ha="center", va="center", fontsize=18, 
             color="black", fontweight="bold", fontfamily='Helvetica')

    # Formatting the chart
    plt.xticks([])

    plt.yticks(
        np.arange(0, 1.1, 0.1), 
        [f"{y:.1f}" for y in np.arange(0, 1.1, 0.1)], 
        fontsize=14
    )

    plt.xlabel("Models", fontsize=18)
    plt.ylabel("Mean number of outliers validating $\\mathcal{H}$ per model", fontsize=16)
    plt.ylim(0, 2)  # Increased the upper limit for extra space above y-tick values

    # Adjust legend appearance
    plt.legend(
        fontsize=12.5, 
        loc="upper right", 
        frameon=True,  # Enable thin frame
        framealpha=0.9,  # Slight transparency
        ncol=1, 
        bbox_to_anchor=(1.00, 1)
    )

    # Add vertical space
    plt.subplots_adjust(top=0.85)

    plt.tight_layout()
    plt.show()

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
custom_labels = model_names

# Color mapping dictionary: specify colors here
colors ={'OrdalieTech/Solon-embeddings-large-0.1': '#FFFF00',
 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2': '#FF00FF',
 'dangvantuan/sentence-camembert-base': '#00FF00',
 'intfloat/multilingual-e5-large': '#6B8E23',
 'FacebookAI/xlm-roberta-large': '#00FFFF',
 'sentence-transformers/all-MiniLM-L12-v2': '#FFA500',
 'sentence-transformers/all-roberta-large-v1': '#0000FF',
 'distilbert/distilbert-base-uncased': '#FF0000',
 'intfloat/e5-base-v2': '#999999'}

# Specify the custom order for the models
custom_order = [
    'distilbert/distilbert-base-uncased',
    'OrdalieTech/Solon-embeddings-large-0.1',
    'intfloat/multilingual-e5-large',
    'sentence-transformers/all-MiniLM-L12-v2',
    'intfloat/e5-base-v2',
    'sentence-transformers/all-roberta-large-v1',
    'dangvantuan/sentence-camembert-base',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'FacebookAI/xlm-roberta-large',
]

plot_ratios_custom(final_combined_df_with_agreement, model_names, custom_labels, colors, custom_order)