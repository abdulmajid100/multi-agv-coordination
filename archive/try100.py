import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges
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

# Define agents with paths and colors
agents = [
    {'path': [1, 4, 11, 21, 11, 4, 5, 4, 6], 'color': 'red', 'index': 0},
    {'path': [2, 4, 11, 12, 13, 14, 24, 14, 13, 12, 11, 4, 5, 4, 6], 'color': 'green', 'index': 0},
    {'path': [3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 28, 18, 17, 16, 15, 14, 13, 12, 11, 4, 5, 4, 6], 'color': 'blue',
     'index': 0}
]

# Define communication range for connectivity (not directly used but conceptual)
communication_range = 3  # Placeholder for connectivity check


def adjust_agents(agents):
    occupied_positions = set()

    # Determine which nodes are currently occupied
    for agent in agents:
        current_pos = agent['path'][agent['index']]
        occupied_positions.add(current_pos)

    # Decide move or wait for each agent
    for agent in agents:
        next_index = agent['index'] + 1
        if next_index < len(agent['path']):
            next_pos = agent['path'][next_index]
            if next_pos not in occupied_positions:
                # Move if no collision is detected
                occupied_positions.remove(agent['path'][agent['index']])
                agent['index'] += 1
                occupied_positions.add(next_pos)


# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G)


def init():
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=500, node_color='lightblue', arrowsize=20)
    return ax.artists


def update(frame):
    ax.clear()

    # Adjust agents to avoid collisions
    adjust_agents(agents)

    # Initialize all nodes to lightblue
    node_colors = ['lightblue'] * len(G.nodes)

    # Color the current nodes of each agent
    for agent in agents:
        current_node = agent['path'][agent['index']]
        node_colors[list(G.nodes).index(current_node)] = agent['color']

    # Draw the graph with updated node colors
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=500, node_color=node_colors, arrowsize=20)

    return ax.artists


# Determine the number of frames based on the longest path
num_frames = max(len(agent['path']) for agent in agents)

# Create and run the animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, repeat=False, interval=1000)
plt.show()