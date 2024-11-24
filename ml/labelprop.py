# A 'from scratch' implementation of label propagation.
# by Cody Jackson
import pandas as pd
import networkx as nx # https://networkx.org/documentation/stable/tutorial.html#graph-attributes
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

### 1. Import the data
data = pd.read_csv("datagrid_1.csv")

### 2. Prep the graph
# Build the graph with nodes and edges
G = nx.Graph()

# Add each data point as a node
for idx, row in data.iterrows():
    lat, lon = round(row['latitude'], 4), round(row['longitude'], 4) # Set lat/lon + round them
    actual_label = 1 if row['positiveOccurrence'] > 0 else 0 # Set the true (correct) label & Make it binary (0 or 1)
    pred_label = -1 # Start unlabeled
    node = (lat, lon) # Set position based on geo-spatial location
    # Initialize node with data as attributes
    G.add_node(node,
               true_label=actual_label,
               label=pred_label,
               start_empty=False)

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

# Add edges based on nearest neighbor within the threshold in specific directions
directions = ["north", "south", "east", "west"]
for node in G.nodes:
    for direction in directions:
        nearest_neighbor = find_nearest_node(G, node, direction)
        if nearest_neighbor:
            G.add_edge(node, nearest_neighbor)

def node_at_position(graph, lat, lon, tolerance=0.0001):
    target_node = (round(lat, 4), round(lon, 4))
    
    for node in graph.nodes:
        if abs(node[0] - target_node[0]) <= tolerance and abs(node[1] - target_node[1]) <= tolerance:
            return node  # Node found

    return None  # Node not found

# And then lastly, manually connected up the "islands" so its one giant graph.
def connect_islands(graph=G):
    G.add_edge(node_at_position(graph, 57.2999, 10.5999), node_at_position(graph, 57.1999, 10.8999))
    G.add_edge(node_at_position(graph, 55.7999, 10.5999), node_at_position(graph, 55.8999, 10.6999))
    G.add_edge(node_at_position(graph, 55.1999, 8.5999), node_at_position(graph, 55.0999, 8.4999))
    G.add_edge(node_at_position(graph, 54.7999, 8.4999), node_at_position(graph, 54.6999, 8.3999))
    G.add_edge(node_at_position(graph, 55.0999, 10.6999), node_at_position(graph, 54.9999, 10.7999))
    G.add_edge(node_at_position(graph, 54.9999, 10.8999), node_at_position(graph, 54.8999, 11.0999))
    G.add_edge(node_at_position(graph, 55.3999, 10.7999), node_at_position(graph, 55.3999, 11.1999))
    G.add_edge(node_at_position(graph, 55.7999, 10.5999), node_at_position(graph, 55.5999, 10.4999))

connect_islands(G)

### 3. Do Label propagation!
def labelprop(graph, start_percentage=0.5, unlabeled_base_value=0.5, interations=4):
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
                #print(f'Found {len(list(neighbors))} neighbors for this node.')

                # Then go through the neighbors list to determine the label (ignore unlabel nodes)
                sum = 0
                count = 0
                for nb in neighbors: # Max of 4 so its not that bad
                    if graph.nodes[nb]['label'] != -1: # Don't pick nodes that are yet to be labeled
                        sum += graph.nodes[nb]['label']
                        count += 1
                    
                if count > 0: # Set the new label
                    graph.nodes[n]['label'] = np.clip(sum / count, 0, 1) # Clamp [0-1]
    
    # 4. Finalize the labels. Pick label via majority vote, if tie, pick randomly
    for n in graph.nodes:
        if graph.nodes[n]['label'] == 0.5: # TIE (random)
            graph.nodes[n]['label'] = rnd.uniform(0, 1)
        else: # VOTE (round to nearest)
            graph.nodes[n]['label'] = round(graph.nodes[n]['label'])

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

graph = labelprop(graph=G) # Run the function

### 4. Calculate the results
g_correct = 0
g_incorrect = 0

for node in graph.nodes:
    if graph.nodes[node]['label'] == graph.nodes[node]['true_label']:
        g_correct += 1
    else:
        g_incorrect += 1

print(f'Scored {(g_correct / graph.number_of_nodes()) * 100:.2f}% with a graph size of {graph.number_of_nodes()}')
print(f'Node results: [{g_correct}] predicted correct, [{g_incorrect}] predicted incorrect')


### 5. Visualization of the graph
def visualize():
    pos = {(x,y):(y,x) for x,y in G.nodes()}

    # Create a color map based on the final labels (0 or 1)
    node_colors = [G.nodes[node]['label'] for node in G.nodes]

    # Draw the graph using the geographic positions
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_size=30, with_labels=False, node_color=node_colors, cmap=plt.cm.RdYlGn, edge_color='black', alpha=0.7)
    plt.title("Label Propagation on the Graph (0 = Incorrect, 1 = Correct)")
    plt.show()

visualize()