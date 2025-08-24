import networkx as nx
import matplotlib.pyplot as plt

# Grid and environment specifications
grid_size = (10, 10)
obstacles = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
             (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
             (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
             (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
             (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
             (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
             (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8),
             (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8),
             (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8),
             (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8)]
goals = [(0, 9), (9, 9)]
initial_positions = [(9, 9), (0, 9)]

# Create a graph
G = nx.Graph()

# Add nodes for every non-obstacle cell in the grid
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        if (x, y) not in obstacles:
            G.add_node((x, y))

# Add edges between adjacent non-obstacle nodes
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        if (x, y) not in obstacles:
            # Connect to right neighbor
            if x + 1 < grid_size[0] and (x + 1, y) not in obstacles:
                G.add_edge((x, y), (x + 1, y))
            # Connect to bottom neighbor
            if y + 1 < grid_size[1] and (x, y + 1) not in obstacles:
                G.add_edge((x, y), (x, y + 1))

# Visualize the graph
plt.figure(figsize=(8, 8))
pos = {(x, y): (y, -x) for x, y in G.nodes()}  # Convert grid position to plot position
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue')
nx.draw_networkx_nodes(G, pos, nodelist=goals, node_color='green')  # Highlight goals
plt.title("Graph Representation of the Grid")
plt.show()