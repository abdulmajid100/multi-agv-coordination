import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    'AGV1': [1,4,11,12,13,14,15,16,17,18,19,29],
    'AGV2': [28,18,17,16,15,14,13,12,11,4,2],
    'AGV3': [29,19,18,17,16,15,14,13,12,11,4,3]
}

# Initialize the reservation status of each node
resource_states = {node: 0 for node in G.nodes()}

# Set initial nodes as reserved
for agv, path in agv_paths.items():
    starting_node = path[0]
    resource_states[starting_node] = agv

def calculate_shared_nodes():
    all_paths = sum(agv_paths.values(), [])
    shared_nodes = {node for node in all_paths if all_paths.count(node) > 1}
    for agv, path in agv_paths.items():
        agv_shared_nodes = [node for node in path if node in shared_nodes]
        print(f"Shared nodes for {agv}: {agv_shared_nodes}")

calculate_shared_nodes()

# Function to check if an AGV can move to the next node
def can_move(agv, current_node, next_node):
    shared_nodes = [n for n in agv_paths[agv] if n in sum(agv_paths.values(), []) and (sum(agv_paths.values(), []).count(n) > 1)]
    if next_node not in shared_nodes:
        return True
    if next_node in shared_nodes:
        if all(resource_states[shared_node] == 0 or resource_states[shared_node] == agv for shared_node in shared_nodes):
            return True
    return False

# Initialize plot
fig, ax = plt.subplots()
pos = nx.kamada_kawai_layout(G)

# Draw the initial graph and AGV positions
nx.draw(G, pos, with_labels=True, arrows=True, node_size=300, ax=ax)
for agv, path in agv_paths.items():
    current_node = path[0]
    nx.draw_networkx_nodes(
        G, pos, nodelist=[current_node],
        node_color='r' if agv == 'AGV1' else 'g' if agv == 'AGV2' else 'b',
        node_size=500, ax=ax
    )

# Update function for animation
def update(frame):
    if frame == 0:
        return  # Skip updating on the first frame to show initial positions

    ax.clear()
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=300, ax=ax)

    # Move each AGV if possible
    for agv, path in agv_paths.items():
        if len(path) > 1:  # Check if there are nodes left in the path
            current_node = path[0]
            next_node = path[1]
            if can_move(agv, current_node, next_node):
                resource_states[next_node] = agv
                resource_states[current_node] = 0
                agv_paths[agv].pop(0)
                print(f"{agv} moves from {current_node} to {next_node}")
            else:
                print(f"{agv} waiting at {current_node}")

    # Draw AGVs at their current positions
    for agv, path in agv_paths.items():
        if path:  # Check if there are nodes left in the path
            current_node = path[0]
            nx.draw_networkx_nodes(
                G, pos, nodelist=[current_node],
                node_color='r' if agv == 'AGV1' else 'g' if agv == 'AGV2' else 'b',
                node_size=500, ax=ax
            )

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(15), repeat=False, interval=2000)

plt.show()