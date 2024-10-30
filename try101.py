import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the paths for each agent
H_Corridor_Go_1 = [
    [1, 4, 11, 10, 20],
    [1, 4, 11, 21],
    [1, 4, 11, 12, 22],
    [1, 4, 11, 12, 13, 23],
    [1, 4, 11, 12, 13, 14, 24],
    [1, 4, 11, 12, 13, 14, 15, 25],
    [1, 4, 11, 12, 13, 14, 15, 16, 26],
    [1, 4, 11, 12, 13, 14, 15, 16, 17, 27],
    [1, 4, 11, 12, 13, 14, 15, 16, 17, 18, 28],
    [1, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29]
]

H_Corridor_Go_2 = [
    [2, 4, 11, 10, 20],
    [2, 4, 11, 21],
    [2, 4, 11, 12, 22],
    [2, 4, 11, 12, 13, 23],
    [2, 4, 11, 12, 13, 14, 24],
    [2, 4, 11, 12, 13, 14, 15, 25],
    [2, 4, 11, 12, 13, 14, 15, 16, 26],
    [2, 4, 11, 12, 13, 14, 15, 16, 17, 27],
    [2, 4, 11, 12, 13, 14, 15, 16, 17, 18, 28],
    [2, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29]
]

H_Corridor_Go_3 = [
    [3, 4, 11, 10, 20],
    [3, 4, 11, 21],
    [3, 4, 11, 12, 22],
    [3, 4, 11, 12, 13, 23],
    [3, 4, 11, 12, 13, 14, 24],
    [3, 4, 11, 12, 13, 14, 15, 25],
    [3, 4, 11, 12, 13, 14, 15, 16, 26],
    [3, 4, 11, 12, 13, 14, 15, 16, 17, 27],
    [3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 28],
    [3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29]
]

# Combine all paths
all_paths = [H_Corridor_Go_1[0], H_Corridor_Go_2[0], H_Corridor_Go_3[0]]

# Create a directed graph
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

# Position nodes using a layout
pos = nx.spring_layout(G)

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=500, ax=ax)

# Initialize agent positions
agent_positions = [path[0] for path in all_paths]
agent_colors = ['r', 'g', 'b']

# Draw initial agent positions
agent_nodes = [ax.plot([], [], 'o', color=color, markersize=10)[0] for color in agent_colors]

def update(frame):
    for i, path in enumerate(all_paths):
        if frame < len(path):
            agent_positions[i] = path[frame]
        agent_nodes[i].set_data([pos[agent_positions[i]][0]], [pos[agent_positions[i]][1]])
    return agent_nodes

# Create animation
ani = animation.FuncAnimation(fig, update, frames=max(len(path) for path in all_paths), blit=True, repeat=False)

plt.show()