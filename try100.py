import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = list(range(1, 30)) + [9]
G.add_nodes_from(nodes)

# Add edges
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

# Define the sequence of paths for the agent
agent_paths = [
    [1, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29],
    [29,19,18,17,16,15,14,13,12,11,4,5,4,6],
    [6,4,11,12,13,14,15,16,17,18,19,29],
    [29,19,18,17,16,15,14,13,12,11,4,1]
]

# Flatten the paths into a single sequence of nodes
full_path = [node for path in agent_paths for node in path]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G)


def init():
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=500, node_color='lightblue', arrowsize=20)
    return ax.artists


def update(frame):
    ax.clear()

    # Initialize all nodes to lightblue
    node_colors = ['lightblue'] * len(G.nodes)

    # Color the current node red
    current_node = full_path[frame]
    node_colors[list(G.nodes).index(current_node)] = 'red'

    nx.draw(G, pos, ax=ax, with_labels=True, node_size=500, node_color=node_colors, arrowsize=20)

    return ax.artists


# Create and run the animation
ani = animation.FuncAnimation(fig, update, frames=len(full_path), init_func=init, blit=False, repeat=False,
                              interval=1000)
plt.show()