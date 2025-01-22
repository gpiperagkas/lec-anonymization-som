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
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
import json
import random
from datetime import datetime, timedelta


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


def generate_input():
    # Initialize a dictionary to hold the data
    energy_data = {}

    # Start date for the dataset
    start_date = datetime(2024, 10, 1)

    # Generate time intervals for 24 hours with 15-minute granularity (96 intervals)
    hours = [f"{h:02d}" for h in range(24)]  # Format hours as "00", "01", ..., "23"
    minutes = ["00", "15", "30", "45"]
    time_intervals = [f"{hour}:{minute}" for hour in hours for minute in minutes]

    # Loop over 30 days to generate monthly data
    for day in range(30):
        current_date = start_date + timedelta(days=day)

        for time_str in time_intervals:
            # Convert the time string to a full datetime string
            timestamp = datetime.strptime(time_str, '%H:%M')
            full_timestamp = (current_date + timedelta(hours=timestamp.hour, minutes=timestamp.minute)).strftime(
                '%Y-%m-%d %H:%M')

            # Generate random energy consumption in Wh (example: between 200 and 500 Wh)
            energy_consumption_wh = random.uniform(200, 700)
            # Generate sample consumption prices according to a predefined value.
            energy_consumption_euro = (energy_consumption_wh / 1000) * 0.30

            # Generate random PV production in Wh (example: between 50 and 300 Wh during the day, less at night)
            if "06:00" <= time_str <= "20:00":
                pv_production_wh = random.uniform(50, 300)
            else:
                pv_production_wh = 0  # No PV production at night
            # Generate sample pv production prices according to a predefined value.
            pv_production_euro = (pv_production_wh / 1000) * 0.15

            # Generate random humidity (example: between 40% and 90%)
            humidity = random.uniform(40, 90)

            # Generate random temperature (example: between 15°C and 30°C, higher during the day)
            if "06:00" <= time_str <= "20:00":
                temperature = random.uniform(20, 30)
            else:
                temperature = random.uniform(15, 20)

            # Populate the dictionary with the timestamp as key and the data as a list
            energy_data[full_timestamp] = [
                round(energy_consumption_wh, 2),  # Wh consumption
                round(energy_consumption_euro, 2),  # EUR consumption
                round(pv_production_wh, 2),  # Wh PV production
                round(pv_production_euro, 2),  # EUR PV production
                round(temperature, 2),  # Temperature
                round(humidity, 2)  # Humidity
            ]

    # Output the total number of rows generated
    print(f"Total data points generated: {len(energy_data)}")

    # Example to print the first few rows to verify
    for timestamp, data in list(energy_data.items())[:10]:
        print(timestamp, data)

    dfinput = pd.DataFrame(energy_data)
    dfinput.to_csv('energy_data_case_1.csv')


    data_list = []

    for timestamp, values in energy_data.items():
        energy = values[0]
        energy_pv = values[2]
        temperature = values[4]
        humidity = values[5]
        # Convert to the data list.
        data_list_element = [energy, energy_pv, temperature, humidity]
        data_list.append(data_list_element)
        print(data_list)

    data = np.array(data_list)

    print(data.shape)  # print shape.
    return data


def train_som(data):
    # Normalize the data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    # Initialize the SOM
    num_runs = 30 # respect the central limit theorem
    num_iteration = 1000 # iterations for training
    som_x, som_y= 10,10 # som grid size

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



def plot_results(som, som_x, som_y, flattened_weights, data, data_normalized):
    column_labels = ["E(Wh)", "Prod(Wh)", "T(C)", "H(%)"]

    #First plot: SOM clusters.
    n_clusters = 5  # Define the number of clusters for som grid plot.
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(flattened_weights)

    cluster_labels_grid = cluster_labels.reshape(som_x, som_y)  # Reshape to SOM grid size

    # Plot the clusters on the SOM grid
    plt.imshow(cluster_labels_grid, cmap='tab10')  # 'tab10' gives distinct colors for up to 10 clusters
    plt.colorbar()
    plt.title('SOM Clusters - single')
    plt.show()

    #Second plot: Intensity map.

    # Visualize the SOM
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar()

    winning_nodes = []
    for x in data_normalized:
        w = som.winner(x)
        winning_nodes.append(w)

    winning_nodes = np.array(winning_nodes)

    # Apply K-means clustering to reduce the number of nodes for intensity map.
    n_clusters = 7  # Number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(winning_nodes)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Plot the cluster centers instead of all winning nodes
    for center in cluster_centers:
        plt.plot(center[0] + 0.5, center[1] + 0.5, 'o', markerfacecolor='None',
                markeredgecolor='r', markersize=12, markeredgewidth=2)
    plt.title('Intensity map - single')

    plt.show()

    # Third plot: heatmap for each labeled variable
    # Initialize dictionaries to hold label counts and feature averages
    neuron_data = defaultdict(list)

    # Map each data point to its closest neuron
    for sample in data_normalized:
        winner = som.winner(sample)
        neuron_data[winner].append(sample)


    # Compute the average feature values for each neuron
    neuron_labels = {}
    for neuron, samples in neuron_data.items():
        average = np.mean(samples, axis=0)
        neuron_labels[neuron] = average
    som_size = 10
    input_len = data.shape[1]
    # Convert neuron labels to a format for visualization
    label_matrix = np.zeros((som_size, som_size, input_len))
    for (x, y), avg in neuron_labels.items():
        label_matrix[x, y] = avg

    fig, axs = plt.subplots(1, input_len, figsize=(15, 5))

    for i in range(input_len):
        ax = axs[i]
        cax = ax.matshow(label_matrix[:, :, i], cmap='coolwarm')
        ax.set_title(column_labels[i])
        plt.colorbar(cax, ax=ax)
    plt.show()


    #Fourth plot: U-matrix with clusters.
    # Assuming 'som' and 'data_normalized' are already defined
    # Cluster the data points using K-Means

    # Get the winner nodes for each data point
    winning_nodes = np.array([som.winner(x) for x in data_normalized])

    # Convert the winning nodes to a format suitable for KMeans
    winning_nodes_reshaped = np.array(winning_nodes)

    # Number of clusters to create (adjust this number as needed)
    num_clusters = 7

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(winning_nodes_reshaped)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

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
    plt.title('U-Matrix with Clusters - single')

    # Overlay the cluster centers
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '*']  # Add more markers if needed
    # Use a colormap with enough colors
    colors = matplotlib.colormaps.get_cmap('tab10')
    for i, center in enumerate(cluster_centers):
        plt.plot(center[0] + 0.5, center[1] + 0.5, markers[i % len(markers)], markerfacecolor='None',
                markeredgecolor=colors(i), markersize=15, markeredgewidth=2)
        plt.text(center[0] + 0.5, center[1] + 0.5, f'C{i+1}', color=colors(i), fontsize=12,
                ha='center', va='center')

    plt.show()


    # Fifth plot: Heatmap.
    # Get the activation map
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
    plt.title('Heatmap - single')
    plt.show()

