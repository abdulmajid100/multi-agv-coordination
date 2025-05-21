import networkx as nx
from graph_env_c_new import GraphEnv

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                  (5, 0), (0, 6), (6, 1), (1, 7), (7, 2)])

# Set up the environment
num_agents = 1
goals = [5]  # Goals are nodes 5 and 2
initial_positions = [0]  # Agents start at nodes 0 and 7

env = GraphEnv(G, num_agents, goals, initial_positions)

# Q-learning for this environment would need to be modified to handle:
# 1. Variable action spaces (different number of neighbors for each node)
# 2. Node-based state indexing instead of grid coordinates

# Example random actions for animation demonstration
import random
actions_sequence = []
for _ in range(20):
    actions = []
    for i in range(num_agents):
        num_actions = env.get_valid_actions(i)
        actions.append(random.randint(0, num_actions-1))
    actions_sequence.append(actions)

# Animate the actions
env.animate(actions_sequence)
