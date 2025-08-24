import simpy
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

# Parameters
NUM_AGVS = 3
NUM_NODES = 6
SIMULATION_TIME = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.2  # Exploration rate

# Create a graph map
G = nx.DiGraph()
nodes = {
    0: (0, 0),  # Facility 1
    1: (1, 0),  # Facility 2
    2: (2, 0),  # Facility 3
    3: (0, 1),  # Buffer 1
    4: (1, 1),  # Buffer 2
    5: (2, 1)   # Charging Station
}
edges = [
    (0, 3), (1, 4), (2, 5),  # Facilities to Buffers
    (3, 0), (4, 1), (5, 2),  # Buffers to Facilities
    (3, 4), (4, 5), (5, 3)   # Buffers and Charging Station connections
]
G.add_nodes_from(nodes.keys())
G.add_edges_from(edges)

# Q-Table Initialization
q_table = np.zeros((NUM_NODES, NUM_NODES))

# AGV Class
class AGV:
    def __init__(self, env, name, start_node, q_table):
        self.env = env
        self.name = name
        self.current_node = start_node
        self.destination_node = random.choice(list(nodes.keys()))
        self.q_table = q_table
        self.action_space = list(G.neighbors(self.current_node))
        self.process = env.process(self.run())
        self.path = [self.current_node]  # Track the path for visualization

    def run(self):
        while True:
            # Choose action using epsilon-greedy policy
            if random.uniform(0, 1) < EPSILON:
                action = random.choice(self.action_space)  # Explore
            else:
                action = np.argmax(self.q_table[self.current_node])  # Exploit

            # Move to the next node
            yield self.env.timeout(1)  # Simulate travel time
            self.current_node = action
            self.path.append(self.current_node)

            # Check if destination is reached
            if self.current_node == self.destination_node:
                self.destination_node = random.choice(list(nodes.keys()))  # Set a new destination

            # Update Q-Table
            reward = 10 if self.current_node == self.destination_node else -1
            next_max = np.max(self.q_table[self.current_node])
            self.q_table[self.current_node, action] += LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * next_max - self.q_table[self.current_node, action]
            )

# SimPy Environment
env = simpy.Environment()

# Create AGVs
agvs = [AGV(env, f'AGV{i+1}', start_node=random.choice(list(nodes.keys())), q_table=q_table) for i in range(NUM_AGVS)]

# Visualization
fig, ax = plt.subplots()
pos = nx.spring_layout(G)  # Layout for graph visualization
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', ax=ax)
agv_positions = {f'AGV{i+1}': nodes[agvs[i].current_node] for i in range(NUM_AGVS)}
agv_dots, = ax.plot([], [], 'ro', markersize=10)  # AGV positions

def update(frame):
    # Update AGV positions
    for i, agv in enumerate(agvs):
        agv_positions[f'AGV{i+1}'] = nodes[agv.current_node]
    x = [pos[0] for pos in agv_positions.values()]
    y = [pos[1] for pos in agv_positions.values()]
    agv_dots.set_data(x, y)
    return agv_dots,

# Run the simulation
def run_simulation():
    env.run(until=SIMULATION_TIME)

# Start the simulation
import threading
sim_thread = threading.Thread(target=run_simulation)
sim_thread.start()

# Animate the simulation
ani = animation.FuncAnimation(fig, update, frames=SIMULATION_TIME, interval=500, blit=True)
plt.show()

# Wait for the simulation to finish
sim_thread.join()
