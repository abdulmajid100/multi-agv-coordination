import networkx as nx

# Initialize the directed graph
G = nx.DiGraph()
nodes = list(range(1, 30)) + [9]
G.add_nodes_from(nodes)
edges = [
    (1, 4), (2, 4), (3, 4), (4, 1), (4, 2), (4, 3), (4, 6), (4, 5),
    (5, 4), (6, 4), (4, 11), (11, 4), (11, 10), (11, 21), (11, 12),
    (10, 11), (10, 20), (20, 10), (21, 11), (12, 11), (12, 22),
    (22, 12), (12, 13), (13, 12), (13, 23), (23, 13), (13, 14),
    (14, 13), (14, 24), (24, 14), (14, 15), (15, 14), (15, 25),
    (25, 15), (15, 16), (16, 15), (16, 26), (26, 16), (16, 17),
    (17, 16), (17, 27), (27, 17), (17, 18), (18, 17), (18, 28),
    (28, 18), (18, 19), (19, 18), (19, 29), (29, 19), (9, 16),
    (16, 9)
]
G.add_edges_from(edges)

# Define AGVs and their paths
agv_paths = {
    'AGV1': [1, 4, 11, 21],
    'AGV2': [2, 4, 11, 12, 13, 14, 24],
    'AGV3': [29, 19, 18, 17, 16, 15, 14, 13, 12, 11, 4, 3]
}

# Initialize the reservation status of each node
resource_states = {node: 0 for node in G.nodes()}

# Function to check if an AGV can move to the next node
def can_move(agv, current_node, next_node):
    if resource_states[next_node] == 0:
        return True
    shared_nodes = [n for n in agv_paths[agv] if n in sum(agv_paths.values(), [])]
    if next_node in shared_nodes and resource_states[next_node] == agv:
        return True
    return False

# AGV positions
agv_positions = {agv: 0 for agv in agv_paths.keys()}

# Simulate the movement of AGVs and print their moves
def simulate_agv_movement():
    moves_made = True
    while moves_made:
        moves_made = False
        for agv, path in agv_paths.items():
            current_index = agv_positions[agv]
            if current_index < len(path) - 1:
                current_node = path[current_index]
                next_node = path[current_index + 1]
                if can_move(agv, current_node, next_node):
                    # Reserve the next node for the AGV
                    resource_states[next_node] = agv
                    # Release the current node
                    resource_states[current_node] = 0
                    # Move AGV to next node
                    agv_positions[agv] += 1
                    print(f"{agv} moves from {current_node} to {next_node}")
                    moves_made = True
                else:
                    print(f"{agv} waiting at {current_node}")

# Run the simulation
simulate_agv_movement()