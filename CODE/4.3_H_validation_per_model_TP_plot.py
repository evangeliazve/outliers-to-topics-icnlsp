import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model agreement data
final_combined_df_with_agreement = pd.read_excel("4.3_outlier_to_topic_transitions_model_agreement_TP.xlsx")

# Define model list
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

# Identify model columns (ending with "_outlier_status")
model_columns = [col for col in final_combined_df_with_agreement.columns if col.endswith("_outlier_status")]

# Calculate total count of values equal to 1 and 0 across all model columns
total_count_1 = final_combined_df_with_agreement[model_columns].eq(1).sum().sum()
total_count_0 = final_combined_df_with_agreement[model_columns].eq(0).sum().sum()

# Compute the global ratio for TP
global_ratio = total_count_1 / (total_count_1 + total_count_0) if (total_count_1 + total_count_0) > 0 else 0

# Output results
total_count_1, total_count_0, global_ratio


def plot_ratios_custom(df, model_names, custom_labels, colors):
    # Extract full model names after the last "/"
    legend_labels = [label.split('/')[-1] for label in custom_labels]  # Full names for legend

    # Prepare data for plotting
    ratios = []
    total_count_1 = 0
    total_count_0 = 0

    for model in model_names:
        # Filter data for each model column
        column = f"{model}_outlier_status"  # Assuming model columns end with _outlier_status

        # Count rows with 1 and 0
        count_1 = (df[column] == 1).sum()
        count_0 = (df[column] == 0).sum()
        total = count_1 + count_0

        # Update total counts
        total_count_1 += count_1
        total_count_0 += count_0

        # Calculate ratio of rows equal to 1
        ratio_validating = count_1 / total if total > 0 else 0
        ratios.append(ratio_validating)
    
    # Compute the global ratio
    global_ratio = total_count_1 / (total_count_1 + total_count_0) if (total_count_1 + total_count_0) > 0 else 0
    
    # Sort data in descending order of ratios
    sorted_indices = np.argsort(ratios)[::-1]
    ratios = [ratios[i] for i in sorted_indices]
    legend_labels = [legend_labels[i] for i in sorted_indices]
    model_names = [model_names[i] for i in sorted_indices]

    # Create the bar chart
    x_positions = np.linspace(0, len(legend_labels) - 1, len(legend_labels))  # Closer spacing
    bar_width = 0.75  # Reduced bar width for a more compact look

    plt.figure(figsize=(10, 6))  # Customized size

    for i, (pos, ratio, color) in enumerate(zip(x_positions, ratios, colors)):
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
    plt.text(len(legend_labels) / 5, 1.5, "Avg. Model = 0.80", 
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


# Example usage
custom_labels = model_names
colors = [
    "#FF0000",  # Bright Red (Flashy)
    "#FFFF00",  # Bright Yellow (Flashy)
    "#00FF00",  # Bright Green (Neon - Flashy)
    "#0000FF",  # Bright Blue (Flashy)
    "#FF00FF",  # Magenta (Flashy)
    "#6B8E23",  # Olive Green (Matte, more subdued)
    "#FFA500",  # Saddle Brown (Matte, subdued)
    "#999999",  # Light Grey (Matte, neutral)
    "#00FFFF",  # Firebrick Red (Matte, rich tone)
]
plot_ratios_custom(final_combined_df_with_agreement, model_names, custom_labels, colors)