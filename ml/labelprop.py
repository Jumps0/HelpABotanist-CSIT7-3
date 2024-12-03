# A 'from scratch' implementation of label propagation.
# by Cody Jackson
import pandas as pd
import networkx as nx # https://networkx.org/documentation/stable/tutorial.html#graph-attributes
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

### 1. Import the data
data = pd.read_csv("datagrid.csv")
soil_type_mapping = {soil: idx for idx, soil in enumerate(data['soilType'].unique())} # Encode soil types
data['soilType_EC'] = data['soilType'].map(soil_type_mapping)

### 2. Prep the graph
def find_nearest_node(graph, current_node, direction, threshold=0.12):
    lat, lon = current_node
    nearest_node = None
    min_distance = float('inf')
    
    for neighbor in graph.nodes:
        n_lat, n_lon = neighbor
        if direction == "north" and n_lat > lat and abs(n_lon - lon) < 0.01:  # Strictly north
            distance = n_lat - lat
        elif direction == "south" and n_lat < lat and abs(n_lon - lon) < 0.01:  # Strictly south
            distance = lat - n_lat
        elif direction == "east" and n_lon > lon and abs(n_lat - lat) < 0.01:  # Strictly east
            distance = n_lon - lon
        elif direction == "west" and n_lon < lon and abs(n_lat - lat) < 0.01:  # Strictly west
            distance = lon - n_lon
        else:
            continue
        
        # Check if this neighbor is closer than the current nearest and within the threshold
        if 0 < distance <= threshold and distance < min_distance:
            min_distance = distance
            nearest_node = neighbor

    return nearest_node

def node_at_position(graph, lat, lon, tolerance=0.0001):
    target_node = (round(lat, 4), round(lon, 4))
    
    for node in graph.nodes:
        if abs(node[0] - target_node[0]) <= tolerance and abs(node[1] - target_node[1]) <= tolerance:
            return node  # Node found

    return None  # Node not found

def connect_islands(graph):
    graph.add_edge(node_at_position(graph, 57.2999, 10.5999), node_at_position(graph, 57.1999, 10.8999))
    graph.add_edge(node_at_position(graph, 55.7999, 10.5999), node_at_position(graph, 55.8999, 10.6999))
    graph.add_edge(node_at_position(graph, 55.1999, 8.5999), node_at_position(graph, 55.0999, 8.4999))
    graph.add_edge(node_at_position(graph, 54.7999, 8.4999), node_at_position(graph, 54.6999, 8.3999))
    graph.add_edge(node_at_position(graph, 55.0999, 10.6999), node_at_position(graph, 54.9999, 10.7999))
    graph.add_edge(node_at_position(graph, 54.9999, 10.8999), node_at_position(graph, 54.8999, 11.0999))
    graph.add_edge(node_at_position(graph, 55.3999, 10.7999), node_at_position(graph, 55.3999, 11.1999))
    graph.add_edge(node_at_position(graph, 55.7999, 10.5999), node_at_position(graph, 55.5999, 10.4999))

def setup_graph(plant="Calluna vulgaris", binary=True):
    if plant != "positiveOccurrence":
        print(f'Setting up graph for plant {plant}...')

    # Build the graph with nodes and edges
    G = nx.Graph()
    
    # Add each data point as a node
    for idx, row in data.iterrows():
        # > Position
        lat, lon = round(row['latitude'], 4), round(row['longitude'], 4) # Set lat/lon + round them
        # > True Occurrence
        if binary:
            actual_label = 1 if row[plant] > 0 else 0 # Set the true (correct) label & Make it binary (0 or 1)
        else:
            actual_label = row[plant]
        # > Label
        pred_label = -1 # Start unlabeled

        node = (lat, lon) # Set position based on geo-spatial location

        # Initialize node with data as attributes
        G.add_node(node,
                ### --------- FEATURE DATA -------------------- ###
                tg=row['tg'], tx=row['tx'], tn=row['tn'],
                hu=row['hu'], rr=row['rr'],
                soilType=row['soilType_EC'],
                soil_moisture=row['soil_moisture'], pH=row['pH_CaCl2'],
                ### ------------------------------------------- ###
                true_label=actual_label, # What is actually at this node
                label=pred_label, # (Used later) What we will use to predict this node
                start_empty=False) # (Used later) If this node should be used to predict others or be predicted from a start state 

    # Add edges based on nearest neighbor in all 4 directions
    directions = ["north", "south", "east", "west"]
    for node in G.nodes:
        for direction in directions:
            nearest_neighbor = find_nearest_node(G, node, direction)
            if nearest_neighbor:
                G.add_edge(node, nearest_neighbor)

    # And then lastly, manually connected up the "islands" so its one giant graph.
    connect_islands(G)

    return G

graph = setup_graph()

### 3. Do Label propagation!
def jaccard_similarity(lst1, lst2):
    # Explainer: https://www.geeksforgeeks.org/how-to-calculate-jaccard-similarity-in-python/

    # intersection of two sets
    intersection = len([value for value in lst1 if value in lst2])
    # Unions of two sets
    union = len(lst1 + lst2)
     
    return intersection / union

def calc_node(graph, node, neighbor):
    # See what this neighbor is predicting
    node_weight = graph.nodes[neighbor]['label']
    """"
    # And here we consider modifying the weight based on feature data
    similarity = 0 # We do so based on similarity (0 to 1)

    # Extract the values into two np.arrays
    fd_a = ([graph.nodes[node]['tg'], graph.nodes[node]['tx'], graph.nodes[node]['tn'], graph.nodes[node]['hu'], graph.nodes[node]['rr'], 
                    graph.nodes[node]['soil_moisture'], graph.nodes[node]['pH']])
    fd_b = ([graph.nodes[neighbor]['tg'], graph.nodes[neighbor]['tx'], graph.nodes[neighbor]['tn'], graph.nodes[neighbor]['hu'], graph.nodes[neighbor]['rr'], 
                    graph.nodes[neighbor]['soil_moisture'], graph.nodes[neighbor]['pH']])

    # Perform a comparison. There are many different methods of doing this
    similarity = jaccard_similarity(fd_a, fd_b)
    print(f'TEST: Node similarity of [{similarity * 100}].')
    exit()
    # If we have a somewhat decent similarity, then we will give this node a bit of a "boost", depending on its neighbor
    if round(graph.nodes[neighbor]['label']) == 1:
        # Give it a boost upwards
        print()
    else:
        # Give it a boost downwards
        print()
    """
    return np.clip(node_weight, 0, 1) # Clamp [0-1]

def labelprop(graph, start_percentage=0.2, unlabeled_base_value=0.5, interations=1000):
    print("...performing label propagation")
    # NOTE: How does 'label' work?
    # 0 = Fully no-occurrence
    # 1 = Fully occurence
    # and number between these two is a "lean". At the end it will pick 0 or 1 based on which is closer.

    # 1. Consider all nodes to be "unlabeled"
    # (This is already done in setup)

    # 2. Randomly choose a selection of nodes to have a label
    for n in graph.nodes:
        if rnd.random() > start_percentage: # These nodes start with their true value, and are never 're-predicted'
            graph.nodes[n]['label'] = graph.nodes[n]['true_label']
            graph.nodes[n]['start_empty'] = True # (Don't try and re-predict these)
        else: # These nodes start at -1 (unlabeled, or whatever value the user wants)
            graph.nodes[n]['label'] = unlabeled_base_value

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
                sum = 0
                count = 0
                for nb in neighbors: # Max of 4 so its not that bad
                        if graph.nodes[nb]['label'] != -1: # Don't pick nodes that are yet to be labeled
                            node_weight = calc_node(graph, n, nb) # Calculate the node's overall weight (see function for details)
                            sum += node_weight
                            count += 1
                    
                if count > 0: # Set the new label
                    graph.nodes[n]['label'] = np.clip(sum / count, 0, 1) # Clamp [0-1]
    
    # 4. Finalize the labels. Pick label via majority vote, if tie, pick randomly
    for n in graph.nodes:
        #if graph.nodes[node]['start_empty'] == False:
            if graph.nodes[n]['label'] == 0.5: # TIE (random)
                graph.nodes[n]['label'] = rnd.uniform(0, 1)
            else: # VOTE (round to nearest)
                graph.nodes[n]['label'] = round(graph.nodes[n]['label'])

    # 5. Return completed graph
    print("...label propagation completed.")
    return graph

def query_location(graph, lat, lon): # Based on a finished predicting, predict what a location will say for occurrence
    # Calculate Euclidean distance to find the nearest node
    distances = np.sqrt((data['latitude'] - lat)**2 + (data['longitude'] - lon)**2)
    nearest_index = distances.idxmin()
    nearest_prediction = graph.nodes[nearest_index]['label']
    
    print(f"Nearest node at ({data.iloc[nearest_index]['latitude']}, {data.iloc[nearest_index]['longitude']})")
    print(f"Predicted occurrence: {'Occurrence' if nearest_prediction == 1 else 'No Occurrence'}")
    return nearest_prediction

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
graph = labelprop(graph, start_percentage=0.2, interations=1000) # Run the function

### 4. Calculate the results
def results(graph):
    g_correct = 0
    g_incorrect = 0
    test_nodes = 0

    for node in graph.nodes:
        if graph.nodes[node]['start_empty'] == False:
            test_nodes += 1
            if graph.nodes[node]['label'] == graph.nodes[node]['true_label']:
                g_correct += 1
            else:
                g_incorrect += 1

    accuracy_score = round((g_correct / test_nodes) * 100, 2)

    print(f'Scored {accuracy_score}%, from {test_nodes} test nodes, with a graph size of {graph.number_of_nodes()}')
    print(f'Node results: [{g_correct}] predicted correct, [{g_incorrect}] predicted incorrect')

    return accuracy_score

results(graph)

### 5. Visualization of the graph
def visualize(only_pred=False):
    pos = {(x,y):(y,x) for x,y in graph.nodes()}

    # Create a color map based on the final labels (0 or 1)
    node_colors = [graph.nodes[node]['label'] for node in graph.nodes]

    if only_pred:
        # Determine node transparency based on 'start_empty' attribute
        node_alphas = [0.2 if not graph.nodes[node]['start_empty'] else 1.0 for node in graph.nodes]
    else:
        node_alphas = 0.7

    # Draw the graph using the geographic positions
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, node_size=30, with_labels=False, node_color=node_colors,
            cmap=plt.cm.RdYlGn, edge_color='black', alpha=node_alphas)
    plt.title("Label Propagation on the Graph (0 = Incorrect, 1 = Correct)")
    plt.show()

#visualize()
