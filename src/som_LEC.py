# Copyright 2025 Grigorios Piperagkas
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# an implementation of SOM training and visualization for LEC Energy Data aggregation
# Parts/Insights for this code have been supported by LLMs.

import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans

def parse_minisom_weights(som):
    """Parses the weights from a MiniSOM and returns them as a JSON string.

    Args:
        som: The MiniSOM instance.

    Returns:
        A JSON string representing the SOM weights.
    """

    weights = som.get_weights()
    weights_list = weights.tolist()
    # Convert weights to JSON string
    json_string = json.dumps(weights_list)

    return json_string


def transpose_dataframe(dataframe):
    """Transposes a DataFrame, converting rows to columns and vice versa.

    Args:
        dataframe: The DataFrame to transpose.

    Returns:
        The transposed DataFrame.
    """

    return dataframe.T


def generate_input_LEC():
    # Define parameters
    num_houses = 60
    days_in_month = 30
    granularity = 15  # minutes

    # Generate time index
    start_date = pd.Timestamp('2024-10-01')  # Adjust start date as needed
    end_date = start_date + pd.DateOffset(days=days_in_month - 1)
    time_index = pd.date_range(start_date, end_date, freq='15min')
    # Generate base consumption profile with day-night pattern
    time_index_normalized = (time_index - time_index.min()) / pd.Timedelta(hours=24)
    day_night_pattern = (1 - 2*np.sin(2 * np.pi * time_index_normalized))
    # Generate base consumption profile (adjust as needed)
    base_consumption = np.random.uniform(200, 700, size=len(time_index)) * day_night_pattern + np.random.uniform(30, 70, size=len(time_index))

    # Generate individual house consumption profiles with variations
    house_consumption = np.random.normal(base_consumption, 20, size=(num_houses, len(time_index)))

    # Create DataFrame
    df = pd.DataFrame(house_consumption, index=range(1, num_houses + 1), columns=time_index)

    # Add random spikes for occasional high consumption
    spike_probability = 0.08  # Adjust probability as needed
    spike_magnitude = 1500  # Adjust magnitude as needed

    spike_indices = np.random.choice(len(time_index), int(len(time_index) * spike_probability), replace=False)

    # Generate random time shifts for each house
    time_shifts = np.random.randint(-60, 60, size=num_houses)  # Adjust range as needed

    # Apply time shifts to spike indices
    for i in range(num_houses):
        shifted_indices = spike_indices + time_shifts[i]
        shifted_indices = shifted_indices[(shifted_indices >= 0) & (shifted_indices < len(time_index))]
        df.iloc[i, shifted_indices] += spike_magnitude

    # Add random dips for occasional low consumption
    dip_probability = 0.03  # Adjust probability as needed
    dip_magnitude = 150  # Adjust magnitude as needed
    dip_indices = np.random.choice(len(time_index), int(len(time_index) * dip_probability), replace=False)
    df.iloc[:, dip_indices] -= dip_magnitude

    # Cap consumption at reasonable limits (adjust as needed)
    df = df.clip(lower=0)

    transposed_df = transpose_dataframe(df)
    print(transposed_df)

    # Visualize sample consumption data
    plt.figure(figsize=(12, 6))
    plt.plot(df.iloc[0, :])
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption (Wh)')
    plt.title('Generated energy consumption profile ')
    plt.show()

    # Save DataFrame to CSV (adjust file path as needed)
    transposed_df.to_csv('energy_consumption_data_T.csv')
    return transposed_df




def train_som_LEC(transposed_df):
    column_labels = ["EC", "PV", "T", "H"]

    y=[0,1]

    # Normalize the data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(transposed_df)

    num_runs = 30
    num_iteration = 1000
    # Initialize the SOM
    som_x, som_y= 10,10
    # Run multiple experiments
    for run in range(num_runs):
        som = MiniSom(x=som_x, y=som_y, input_len=data_normalized.shape[1], sigma=1.0, learning_rate=0.5)
        som.random_weights_init(data_normalized)
        som.train_random(data_normalized, num_iteration)




    weights = som.get_weights()  # Extract the weight vectors from the SOM
    flattened_weights = weights.reshape(-1, weights.shape[-1])  # Flatten to 2D array for clustering


    # Parse and print the weights as a JSON string
    json_string = parse_minisom_weights(som)
    print(json_string)
    return som, flattened_weights, data_normalized


def plot_results_LEC(som, som_x, som_y, flattened_weights, data_normalized):

    # First plot: SOM clusters
    n_clusters = 5  # Define the number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(flattened_weights)

    cluster_labels_grid = cluster_labels.reshape(som_x, som_y)  # Reshape to SOM grid size

    # Plot the clusters on the SOM grid
    plt.imshow(cluster_labels_grid, cmap='tab10')  # 'tab10' gives distinct colors for up to 10 clusters
    plt.colorbar()
    plt.title('SOM Clusters - LEC')
    plt.show()


    # Second plot: Intensity map for the LEC
    # Visualize the SOM
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar()

    # Collect all the winning node coordinates
    winning_nodes = []
    for x in data_normalized:
        w = som.winner(x)
        winning_nodes.append(w)

    winning_nodes = np.array(winning_nodes)

    # Apply K-means clustering to reduce the number of nodes
    n_clusters = 7  # Number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(winning_nodes)

    # Get the cluster centers (these are the new reduced points)
    cluster_centers = kmeans.cluster_centers_

    # Plot the cluster centers instead of all winning nodes
    for center in cluster_centers:
        plt.plot(center[0] + 0.5, center[1] + 0.5, 'o', markerfacecolor='None',
                markeredgecolor='r', markersize=12, markeredgewidth=2)
    plt.title('Intensity map - LEC')
    plt.show()


    # Third plot: U-matrix with clusters
    # Cluster the data points using K-Means
    # Get the winner nodes for each data point
    winning_nodes = np.array([som.winner(x) for x in data_normalized])

    # Convert the winning nodes to a format suitable for KMeans
    winning_nodes_reshaped = np.array(winning_nodes)

    # Number of clusters
    num_clusters = 7

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(winning_nodes_reshaped)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_
    som_size = 10
    # Calculate the U-Matrix
    weights = som.get_weights()
    u_matrix = np.zeros((som_size, som_size))

    for i in range(som_size):
        for j in range(som_size):
            neighbors = []
            if i > 0:
                neighbors.append(weights[i - 1, j])
            if i < som_size - 1:
                neighbors.append(weights[i + 1, j])
            if j > 0:
                neighbors.append(weights[i, j - 1])
            if j < som_size - 1:
                neighbors.append(weights[i, j + 1])

            # Calculate the average distance to neighboring neurons
            u_matrix[i, j] = np.mean([np.linalg.norm(weights[i, j] - neighbor) for neighbor in neighbors])

    # Plot the U-Matrix
    plt.figure(figsize=(10, 8))
    plt.pcolor(u_matrix.T, cmap='bone_r')  # Transpose to match the coordinate system
    plt.colorbar(label='Distance')
    plt.title('U-Matrix with Clusters - LEC')

    # Overlay the cluster centers
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '*']
    # Use a colormap with enough colors
    colors = matplotlib.colormaps.get_cmap('tab10')
    for i, center in enumerate(cluster_centers):
        plt.plot(center[0] + 0.5, center[1] + 0.5, markers[i % len(markers)], markerfacecolor='None',
                markeredgecolor=colors(i), markersize=15, markeredgewidth=2)
        plt.text(center[0] + 0.5, center[1] + 0.5, f'C{i+1}', color=colors(i), fontsize=12,
                ha='center', va='center')

    plt.show()


    # Fourth plot: heatmap
    # Initialize an activation map
    activation_map = np.zeros((som_size, som_size))

    # Iterate through each node and calculate its activation
    for i in range(som_size):
        for j in range(som_size):
            # Calculate the distance between the node's weights and the input data
            distance = np.linalg.norm(weights[i, j] - data_normalized)
            # Assign a higher activation to nodes that are closer to the input
            activation_map[i, j] = 1 / (1 + distance)  # Or any other activation function

    # Normalize the activation values using min-max scaling
    activation_map_min = np.min(activation_map)
    activation_map_max = np.max(activation_map)
    normalized_activation_map = (activation_map - activation_map_min) / (activation_map_max - activation_map_min)

    # Create the heatmap
    plt.imshow(normalized_activation_map, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap - LEC')
    plt.show()

