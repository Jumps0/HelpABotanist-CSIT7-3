# Label propagation script by: Anastasios “Tasos” Benos & Cody Jackson
# for the AAU Fall 2024 Project "Help a Botanist"

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def setup_data(file_name="datagrid.csv"):
    # Step 1: Load the Dataset
    data = pd.read_csv(file_name)  

    # Step 2: Preprocess the Data
    # Encode the categorical 'soilType' column as a unique identifier
    data['soilType_encoded'] = data['soilType'].astype('category').cat.codes

    # Normalize features (keep the categorical column encoded as-is)
    scaler = MinMaxScaler()
    # Store original latitudes and longitudes before scaling
    data['original_latitude'] = data['latitude']
    data['original_longitude'] = data['longitude']
    data[['latitude', 'longitude', 'tg', 'tx', 'tn', 'hu', 'rr', 'pH_CaCl2', 'soil_moisture']] = scaler.fit_transform(
        data[['latitude', 'longitude', 'tg', 'tx', 'tn', 'hu', 'rr', 'pH_CaCl2', 'soil_moisture']]
    )
    
    return data

# Set up the data
data = setup_data(file_name="datagrid.csv")

# Step 3: Define the Weighted Similarity Function with Categorical Handling
def calculate_similarity(node1, node2, overload=False):
    # List of features to be considered (excluding latitude and longitude)
    features = ['tg', 'tx', 'tn', 'hu', 'rr', 'soilType_encoded', 'pH_CaCl2', 'soil_moisture']
    weights = [1 for _ in features]  # Equal weights

    similarity = 0
    weighted_diff = 0
    distance = 0
    feature_details = []

    # Calculate max distance for normalization
    max_distance = sum(weight * 1 for weight in weights) # Max difference is 1 for normalized features

    #
    #node = data.iloc[i]
    #data.iloc[node2]

    if overload:
        node1 = data.iloc[node1]
        node2 = data.iloc[node2]

    for feature, weight in zip(features, weights):
        if feature == 'soilType_encoded':  # Handle the categorical feature separately
            # If soil types are the same, similarity is 0; otherwise, assign a fixed penalty
            diff = 0 if node1[feature] == node2[feature] else 1
        else:
            # For numerical features, calculate the absolute difference
            diff = abs(node1[feature] - node2[feature])

        weighted_diff += weight * diff
        distance += weighted_diff
        feature_details.append((feature, diff, weighted_diff))

        # Normalize Similarity to [0, 1]
        similarity = 1 - (distance / max_distance)

    return similarity, feature_details  # Return both the similarity and feature-level details

# Find the nearest node in one of the four cardinal directions
def nearest_node(network, reference, direction, max_distance=0.12):
    """Find the nearest node in a specified direction (north, south, east, or west)"""
    ref_lat, ref_lon = reference
    closest_node = None
    smallest_gap = float('inf')

    for candidate in network.nodes:
        # Access the latitude and longitude from the node's attributes
        cand_lat = network.nodes[candidate]['latitude']
        cand_lon = network.nodes[candidate]['longitude']

        distance = None

        if direction == "north" and cand_lat > ref_lat and abs(cand_lon - ref_lon) < 0.01:
            distance = cand_lat - ref_lat
        elif direction == "south" and cand_lat < ref_lat and abs(cand_lon - ref_lon) < 0.01:
            distance = ref_lat - cand_lat
        elif direction == "east" and cand_lon > ref_lon and abs(cand_lat - ref_lat) < 0.01:
            distance = cand_lon - ref_lon
        elif direction == "west" and cand_lon < ref_lon and abs(cand_lat - ref_lat) < 0.01:
            distance = ref_lon - cand_lon

        if distance is not None and 0 < distance <= max_distance and distance < smallest_gap:
            smallest_gap = distance
            closest_node = candidate

    return closest_node

def add_edges_to_neighbors(i, data, graph, report_file = None):
    node = data.iloc[i]
    latitude, longitude = node['latitude'], node['longitude']
    neighbors_info = []

    cardinal_directions = ["north", "south", "east", "west"]

    # Find the closest neighbors in each direction
    for direction in cardinal_directions:
        neighbor = nearest_node(graph, (latitude, longitude), direction)

        if neighbor is not None:  # If a valid neighbor is found
            similarity, feature_details = calculate_similarity(node, data.iloc[neighbor])
            graph.add_edge(i, neighbor, weight=similarity)
            neighbors_info.append((neighbor, similarity, feature_details))

    # Log the relationship details for the node with detailed calculations
    if neighbors_info and report_file != None:
        report_file.write(f"Node {i} ({latitude:.4f}, {longitude:.4f}) has {len(neighbors_info)} neighbors:\n")
        for neighbor, similarity, feature_details in neighbors_info:
            report_file.write(f"  - With Neighbor {neighbor} has similarity of {similarity:.4f}\n")
            report_file.write(f"    Feature-wise details:\n")
            for feature, diff, weighted_diff in feature_details:
                report_file.write(f"      - {feature}: Difference = {diff:.4f} - Weighted Diff = {weighted_diff:.4f}\n")
        report_file.write("\n")

# Step 4: Build the Graph with Limited Neighbor Connections (North, South, East, West), plus add edges
def build_graph(plant="Calluna vulgaris"):
    print(f'...building graph.')
    graph = nx.Graph()

    # Add nodes to the graph with their features
    for i, row in data.iterrows():
        if True:  # binary:
            al = 1 if row[plant] > 0 else 0  # Set the true (correct) label & Make it binary (0 or 1)
            if al == 1:
                actual_label = (0, 1)
            else:
                actual_label = (1, 0)
        else:
            actual_label = row[plant]
        
        pred_label = (-1, -1)  # Start unlabeled
        
        # Create a copy of the row dictionary and remove 'label' if present
        node_data = row.to_dict()
        node_data.pop('label', None)  # Remove 'label' to avoid conflict with 'true_label' and 'pred_label'

        # Add the node to the graph with its features
        graph.add_node(i, **node_data,
                       true_label=actual_label,  # What is actually at this node
                       label=pred_label,         # (Used later) What we will use to predict this node
                       start_empty=False)        # (Used later) If this node should be used to predict others or be predicted from a start state. False = unlabeled
        graph.nodes[i]['name'] = i  # Assign a name to the node (its index)

    # Add edges for each node based on neighbor relations (North, South, East, West)
    for i in range(len(data)):
        add_edges_to_neighbors(i, data, graph)

    return graph

# Example usage:
target_plant = "Rumex acetosa"
graph = build_graph(target_plant)

def plot_graph(graph, title):
    print("I'm in plot_graph")
    pos = {node: (graph.nodes[node]['longitude'], graph.nodes[node]['latitude']) for node in graph.nodes()}
    edge_weights = nx.get_edge_attributes(graph, 'weight')

    if edge_weights:
        edge_colors = [edge_weights[edge] for edge in graph.edges()]
        edge_min = min(edge_colors)
        edge_max = max(edge_colors)
    else:
        edge_colors = []
        edge_min = edge_max = 1  # Default value if no edges

    plt.figure(figsize=(12, 8))
    ax = plt.gca()  # Get current axis for proper association

    # Improve visibility of nodes and edges
    nodes = nx.draw_networkx_nodes(graph, pos, node_size=50, node_color='gray', alpha=0.4, ax=ax)
    edges = nx.draw_networkx_edges(
        graph, pos, edge_color=edge_colors, edge_cmap=plt.cm.Blues, edge_vmin=edge_min, edge_vmax=edge_max, width=2.0,
        ax=ax
    )

    # Add labels for node IDs
    #nx.draw_networkx_labels(graph, pos, font_size=6, font_color='black', font_weight='bold', ax=ax)

    # Add a colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=edge_min, vmax=edge_max))
    sm.set_array([])  # Necessary for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)  # Explicitly associate with the current axis
    cbar.set_label("Edge Weight (Similarity)")

    plt.title(title)
    plt.show()

def get_neighbor_label(graph, node, neighbor, do_sim=True):
    # See what this neighbor is predicting (This is a tuple)
    node_label = graph.nodes[neighbor]['label']
    similarity = 1

    if do_sim:
        # And here we consider modifying the label based on feature data
        similarity, _ = calculate_similarity(node, neighbor, True)

    return np.clip(node_label, 0, 1), similarity # Clamp [0-1]

def calculate_new_label(labels, similarities, do_sim=True):
    new_label = (0, 0)

    if do_sim:
        pre = []
        all_sim = 0

        for i, lb in enumerate(labels): # Go through each label
            # And do the initial vector * weight calculation
            l = lb[0]
            r = lb[1]

            l = l * similarities[i]
            r = r * similarities[i]

            pre.append((l, r)) # Store for next step use

            # Also add up the similarities while we're here
            all_sim += similarities[i]
        
        # Then go through and add up the calculated vectors
        post = (0, 0)
        for v in pre:
            l = v[0]
            r = v[1]

            post_l = post[0]
            post_r = post[1]

            post_l += l
            post_r += r

            post = (post_l, post_r)

        # Then do the final division via numpy
        final = np.divide(post, all_sim, casting="unsafe") # numpy division
        new_label = np.clip(final, 0, 1) # Clamp [0-1]

    else:
        # This is pretty simple, just divide the sum of all the labels by the amount of lables
        count = len(labels)
        sum = (0, 0)

        for lb in labels:
            # Annoying breakdown
            sum_l = sum[0]
            sum_r = sum[1]

            sum_l += lb[0]
            sum_r += lb[1]
            
            sum = (sum_l, sum_r)

        if count > 0: # Set the new label
            final = np.divide(sum, count, casting="unsafe") # numpy division
            new_label = np.clip(final, 0, 1) # Clamp [0-1]

    return new_label

# 5. Perform label propagation
import random as rnd
def labelpropagation(graph, finish_labels=True, only_negatives=False, use_similarity=True, use_fixed_split=True, start_percentage=0.2, unlabeled_base_value=0.5, interations=1000):
    print("...performing label propagation")
    # NOTE: How does 'label' work?
    # 0 = Fully no-occurrence
    # 1 = Fully occurence
    # These values are complimentary where the left side is probability of negative occurrence,
    # and the right side is probability of positive occurrence. They both add up to 100% or 1.

    # 1. Consider all nodes to be "unlabeled"
    # (This is already done in setup)

    # 2. Randomly choose a selection of nodes to have a label
    if use_fixed_split: # Alternatively, use the fixed split defined in the file
        training_nodes = data[data['isTest'] == False].index
        testing_nodes = data[data['isTest'] == True].index 
        for idx in training_nodes:
            graph.nodes[idx]['label'] = graph.nodes[idx]['true_label'] # Set true label
            graph.nodes[idx]['start_empty'] = True
            graph.nodes[idx]['isTest'] = False  # Mark as training node
            graph.nodes[idx]['isTrain'] = True

        for idx in testing_nodes:
            graph.nodes[idx]['label'] = (unlabeled_base_value, unlabeled_base_value) # Set base label
            graph.nodes[idx]['start_empty'] = False
            graph.nodes[idx]['isTest'] = True  # Mark as test node
            graph.nodes[idx]['isTrain'] = False
    else:
        for n in graph.nodes:
            if rnd.random() > start_percentage: # These nodes start with their true value, and are never 're-predicted'
                graph.nodes[n]['label'] = graph.nodes[n]['true_label']
                graph.nodes[n]['start_empty'] = True # (Don't try and re-predict these)
            else: # These nodes start at -1 (unlabeled, or whatever value the user wants)
                graph.nodes[n]['label'] = (unlabeled_base_value, unlabeled_base_value)

    # 3. For each unlabeled node, check its neighbors, and decide what this current node should be
    for _ in range(interations):
        # Go through every node
        for n in graph.nodes:
            # and if it is not part of the base set (aka started with its correct label)
            if graph.nodes[n]['start_empty'] == False:
                # Then get its neighbors, and determine the label based on that
                neighbors = getgraphneighbors(graph, n, 1)
                neighbors = list(dict.fromkeys(neighbors)) # Remove duplicates

                # Then go through the neighbors list to determine the label (ignore unlabel nodes)
                labels = []
                similarities = []

                for nb in neighbors: # Max of 4 so its not that bad
                    if graph.nodes[nb]['label'][0] != -1 and graph.nodes[nb]['label'][1] != -1: # Don't pick nodes that are yet to be labeled
                        node_label, similarity = get_neighbor_label(graph, n, nb, do_sim=use_similarity) # Gather up label and similarity (edge) value

                        labels.append(node_label)
                        similarities.append(similarity)

                # Calculate the node's new label based on its neighbors info
                graph.nodes[n]['label'] = calculate_new_label(labels, similarities, do_sim=use_similarity)

    
    # 4. Finalize the labels. Pick label via majority vote, if tie, pick randomly
        for n in graph.nodes:
            if finish_labels:
                if graph.nodes[n]['start_empty'] == False:
                    # Get left and right of tuple
                    l = graph.nodes[n]['label'][0]
                    r = graph.nodes[n]['label'][1]

                    if l == 0.5 and r == 0.5: # TIE (random)
                        l = rnd.uniform(0, 1)
                        r = 1 if l == 0 else 1
                    else: # VOTE (round to nearest)
                        l = round(l)
                        r = round(r)

                    graph.nodes[n]['label'] = (l, r) # Set new FINAL label
            
            if only_negatives: # Overwrite a label's negative prediction if we KNOW it's actually positive
                l = graph.nodes[n]['label'][0]
                r = graph.nodes[n]['label'][1]
                if(round(l) == 1 and graph.nodes[n]['true_label'] == (0, 1)):
                    graph.nodes[n]['label'] = (0, 1)
    print(graph)
    # 5. Return completed graph
    print("...label propagation completed.")
    return graph

def getgraphneighbors(graph, node, depth, current_depth=0, visited=None):
    if visited is None:
        visited = set()
    
    # If we've reached the target depth, return this node as part of the result
    if current_depth == depth:
        return {node}
    
    visited.add(node)
    neighbors = set()

    # Recursively visit neighbors at the next depth level
    for neighbor in graph.neighbors(node):
        if neighbor not in visited:
            neighbors.update(getgraphneighbors(graph, neighbor, depth, current_depth + 1, visited))

    return neighbors

# Start percentage: 0.X -> XX% start unlabeled, and are used in LP
graph_finished = labelpropagation(graph, finish_labels=True, use_fixed_split=True, use_similarity=True, start_percentage=0.2, interations=100) # Run the function

### 6. Calculate the results
def results(g, update_file=False, file_name="datagrid.csv"):
    g_correct = 0
    g_incorrect = 0
    test_nodes = 0

    for node in g.nodes:
        if g.nodes[node]['start_empty'] == False:
            test_nodes += 1
            if g.nodes[node]['label'] == g.nodes[node]['true_label']:
                g_correct += 1
            else:
                g_incorrect += 1

    accuracy_score = round((g_correct / test_nodes) * 100, 2)

    if update_file:
        data = pd.read_csv(file_name)  

        if "predicted" not in data.columns:
            predicted_index = data.columns.get_loc("isTest") + 1
            data.insert(predicted_index, "predicted", np.nan)
        
        # Update test rows based on graph predictions
        test_indices = data[data['isTest'] == True].index
        
        for idx in test_indices:
            node_label = graph.nodes[idx].get('label', (0, 0))  # Default to (0, 0) if no label
            right_value = node_label[1]  # Right-side value in the tuple
            
            # Update the "predicted" column
            data.at[idx, 'predicted'] = right_value

        # Save the updated file
        data.to_csv(file_name, index=False)

    return accuracy_score

results(graph_finished, update_file=False)
print (results)
def query_location(graph, lat, lon): # Get the final label of a node after the LP process has happened on the graph
    # Calculate Euclidean distance to find the nearest node
    print("Hello I'm query_location")
    distances = np.sqrt((data['latitude'] - lat)**2 + (data['longitude'] - lon)**2)
    nearest_index = distances.idxmin()
    prediction = graph.nodes[nearest_index]['label']
    return prediction[1] # Return right side which is positive probability
