import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams['font.family'] = 'Arial'

# Define the model name to load from saved results
model_name = "OrdalieTech/Solon-embeddings-large-0.1"  # Change to desired model

# Define the directory where results are stored
save_dir = "cumulative_clustering_results_TP"
model_dir = os.path.join(save_dir, model_name.replace("/", "_"))

# Define custom time windows
time_windows_custom = [
    '2020-04', '2020-06', '2021-01',
    '2021-04', '2021-05', '2021-07', '2021-09',
    '2022-01', '2024-08'
]

# Visualization settings
n_windows = len(time_windows_custom)
n_cols = 3  # Number of columns for compact layout
n_rows = math.ceil(n_windows / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 12), constrained_layout=True)
axes = axes.flatten()

outlier_color = 'black'  # Fixed color for outliers
available_colors = ["navy", "green"]
x_ticks = np.arange(7.0, 11.1, 1)
y_ticks = np.arange(2.0, 7.1, 1)

# Initialize cumulative points DataFrame
cumulative_points = pd.DataFrame(columns=['embedding_0', 'embedding_1', 'global_topic'])

# Helper function to load data
def load_data(timeframe):
    """Loads cumulative clustering results for a given timeframe."""
    filename = os.path.join(model_dir, f'cumulative_clustering_{timeframe}.csv')
    if not os.path.exists(filename):
        print(f"Warning: File not found - {filename}")
        return None
    return pd.read_csv(filename)

# Helper function to assign topic colors
def assign_topic_colors(topics):
    """Assigns colors to topics using a colormap."""
    return {topic: available_colors[topic % len(available_colors)] for topic in topics if topic != -1}

# Iterate over each time window
for i, time_window in enumerate(time_windows_custom):
    all_data = load_data(time_window)
    if all_data is None:
        continue

    ax = axes[i]

    # Identify new and old points
    all_data['is_new'] = ~all_data[['embedding_0', 'embedding_1']].apply(tuple, axis=1).isin(
        cumulative_points[['embedding_0', 'embedding_1']].apply(tuple, axis=1)
    )
    new_points = all_data[all_data['is_new']]
    old_points = all_data[~all_data['is_new']]
    cumulative_points = pd.concat([cumulative_points, new_points], ignore_index=True)

    # Assign colors to topics
    global_topic_ids = all_data['global_topic'].unique()
    topic_colors = assign_topic_colors(global_topic_ids)

    # Plot points for each topic
    for global_topic_id in global_topic_ids:
        if global_topic_id == -1:  # Handle outliers
            ax.scatter(
                old_points[old_points['global_topic'] == global_topic_id]['embedding_0'],
                old_points[old_points['global_topic'] == global_topic_id]['embedding_1'],
                color=outlier_color, s=100, marker="x", alpha=1.0, label="Old Outliers"
            )
            ax.scatter(
                new_points[new_points['global_topic'] == global_topic_id]['embedding_0'],
                new_points[new_points['global_topic'] == global_topic_id]['embedding_1'],
                color=outlier_color, marker="x", s=80, alpha=1.0, label="New Outliers"
            )
        else:
            ax.scatter(
                old_points[old_points['global_topic'] == global_topic_id]['embedding_0'],
                old_points[old_points['global_topic'] == global_topic_id]['embedding_1'],
                color=topic_colors[global_topic_id], s=10, alpha=0.3, label=f"Old Topic {global_topic_id}"
            )
            ax.scatter(
                new_points[new_points['global_topic'] == global_topic_id]['embedding_0'],
                new_points[new_points['global_topic'] == global_topic_id]['embedding_1'],
                color=topic_colors[global_topic_id], s=100, alpha=1.0, label=f"New Topic {global_topic_id}"
            )

    # Set plot titles, limits, and ticks
    ax.set_title(f'Time Window: {time_window}', fontsize=22)
    ax.set_xlim(7, 11)
    ax.set_ylim(2, 7)

    # Set shared x and y ticks for consistent gridlines
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Customize tick sizes
    ax.tick_params(axis='x', labelsize=24, length=6)  
    ax.tick_params(axis='y', labelsize=24, length=6) 

    # Add gridlines for all plots
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # External plots: show tick labels
    if i % n_cols == 0:  
        ax.set_ylabel('Dimension 2', fontsize=32, labelpad=10)
    else:
        ax.tick_params(axis='y', labelleft=False)  

    if i >= n_windows - n_cols or i >= n_cols * (n_rows - 1):  
        ax.set_xlabel('Dimension 1', fontsize=32, labelpad=10)
    else:
        ax.tick_params(axis='x', labelbottom=False)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()