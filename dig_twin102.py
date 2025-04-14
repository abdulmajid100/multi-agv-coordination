import simpy
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

# Parameters
NUM_AGVS = 3
SIMULATION_TIME = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.2  # Exploration rate

# Create graph
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
G.remove_nodes_from([7, 8])

# Create node-to-index mapping
NUM_NODES = len(G.nodes)
node_to_index = {node: i for i, node in enumerate(sorted(G.nodes))}
index_to_node = {i: node for node, i in node_to_index.items()}

# Q-Table Initialization
q_table = np.zeros((NUM_NODES, NUM_NODES))


# AGV Class with DT-driven RL capabilities
class AGV:
    def __init__(self, env, name, start_node, q_table):
        self.env = env
        self.name = name
        self.current_node = start_node
        self.destination_node = random.choice(list(G.nodes))
        self.q_table = q_table
        self.action_space = list(G.neighbors(self.current_node))
        self.process = env.process(self.run())
        self.path = [self.current_node]
        self.waiting = False
        self.wait_time = 0

    def get_state(self):
        # System utilization
        US = self.env.now / SIMULATION_TIME

        # All AGV information
        global agvs
        all_agv_locations = [agv.current_node for agv in agvs]
        all_agv_destinations = [agv.destination_node for agv in agvs]
        all_agv_paths = [agv.path for agv in agvs]

        return {
            'US': US,
            'AGV_L': all_agv_locations,
            'AGV_D': all_agv_destinations,
            'AGV_P': all_agv_paths
        }

    def get_actions(self):
        connected_nodes = list(G.neighbors(self.current_node))
        if not connected_nodes:
            return [{'NC': self.current_node, 'W': 1, 'Tw': 1}]  # Stay at current node if no neighbors

        actions = []
        # Move actions (no waiting)
        for node in connected_nodes:
            actions.append({'NC': node, 'W': 0, 'Tw': 0})

        # Wait actions
        for node in connected_nodes:
            for wait_time in [1, 2]:  # Wait options
                actions.append({'NC': node, 'W': 1, 'Tw': wait_time})

        return actions

    def run(self):
        while True:
            # Update action space
            self.action_space = list(G.neighbors(self.current_node))

            # Check if action space is empty
            if not self.action_space:
                print(f"{self.name} is stuck at node {self.current_node} with no available actions.")
                yield self.env.timeout(1)  # Wait and retry
                continue

            # Choose action using DT-driven RL (handled externally)
            # The actual action selection happens in the dt_driven_q_learning function

            # Simulate movement (simplified)
            yield self.env.timeout(1)

            # Check if destination is reached
            if self.current_node == self.destination_node:
                self.destination_node = random.choice(list(G.nodes))  # Set a new destination


# Function to calculate reward based on simulation results
def calculate_reward(simulation_results):
    # Extract KPIs from simulation
    throughput = simulation_results.get('throughput', 0)
    makespan = simulation_results.get('makespan', 0)
    mean_delivery_time = simulation_results.get('mean_delivery_time', 0)
    collision = simulation_results.get('collision', False)

    # Standard deviations (pre-calculated or from historical data)
    ST = 10  # Standard deviation for throughput
    SM = 100  # Standard deviation for makespan
    STD = 50  # Standard deviation for delivery time

    # Weights
    lambda_T = 0.3  # Weight for throughput
    lambda_M = 0.3  # Weight for makespan
    lambda_TD = 0.4  # Weight for delivery time

    # Calculate reward components
    r_throughput = lambda_T * ((0 - throughput) / ST) ** 2
    r_makespan = lambda_M * ((0 - makespan) / SM) ** 2
    r_delivery = lambda_TD * ((0 - mean_delivery_time) / STD) ** 2

    # Collision factor
    C = -1 if collision else 1

    # Final reward
    reward = C * (r_throughput + r_makespan + r_delivery)

    return reward


# Digital Twin simulation function
def digital_twin_simulation(current_state, action):
    # This is a simplified DT simulation
    # In a real implementation, this would be more complex and accurate

    agv_id = action['agv_id']
    next_node = action['NC']
    wait_time = action['Tw'] if action['W'] == 1 else 0

    # Check for potential collisions
    collision = False
    for other_agv_id, other_location in enumerate(current_state['AGV_L']):
        if other_agv_id != agv_id and other_location == next_node:
            collision = True

    # Calculate simulation metrics
    simulation_time = 1 + wait_time
    if collision and wait_time == 0:
        simulation_results = {
            'throughput': 0,
            'makespan': simulation_time,
            'mean_delivery_time': simulation_time,
            'collision': True
        }
    else:
        # If waiting or no collision
        simulation_results = {
            'throughput': 1,
            'makespan': simulation_time,
            'mean_delivery_time': simulation_time,
            'collision': False
        }

    # Update state
    next_state = current_state.copy()
    next_state['AGV_L'] = current_state['AGV_L'].copy()
    next_state['AGV_L'][agv_id] = next_node

    return next_state, simulation_results


# DT-driven Q-learning algorithm
def dt_driven_q_learning(q_table, node_to_index, index_to_node):
    # Initialize parameters
    alpha = LEARNING_RATE
    gamma = DISCOUNT_FACTOR
    epsilon = EPSILON

    for i, agv in enumerate(agvs):
        # Get current state
        current_state = agv.get_state()
        current_index = node_to_index[agv.current_node]

        # If AGV is waiting, continue its wait
        if agv.waiting:
            agv.wait_time -= 1
            if agv.wait_time <= 0:
                agv.waiting = False
            continue

        # Choose action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            # Explore: random action
            valid_actions = agv.get_actions()
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                continue  # Skip if no valid actions
        else:
            # Exploit: best action based on Q-values
            valid_actions = agv.get_actions()
            if not valid_actions:
                continue  # Skip if no valid actions

            best_action = None
            best_q_value = float('-inf')

            for a in valid_actions:
                action_node = a['NC']
                action_index = node_to_index[action_node]
                q_value = q_table[current_index][action_index]

                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = a

            action = best_action

        # Execute action in Digital Twin simulation
        next_state, simulation_results = digital_twin_simulation(current_state, {
            'agv_id': i,
            'NC': action['NC'],
            'W': action['W'],
            'Tw': action['Tw']
        })

        # Calculate reward
        reward = calculate_reward(simulation_results)

        # Update Q-value
        next_index = node_to_index[action['NC']]
        neighbors = list(G.neighbors(action['NC']))
        if neighbors:
            max_future_q = np.max([q_table[next_index][node_to_index[n]] for n in neighbors])
        else:
            max_future_q = 0

        q_table[current_index][next_index] += alpha * (
                reward + gamma * max_future_q - q_table[current_index][next_index]
        )

        # Move AGV to next node
        agv.current_node = action['NC']
        agv.path.append(action['NC'])

        # If waiting is needed
        if action['W'] == 1:
            agv.waiting = True
            agv.wait_time = action['Tw']
            print(f"{agv.name} waiting at node {agv.current_node} for {action['Tw']} time units")

    return q_table


# SimPy Environment
env = simpy.Environment()

# Create AGVs
agvs = [AGV(env, f'AGV{i + 1}', start_node=random.choice(list(G.nodes)), q_table=q_table) for i in range(NUM_AGVS)]

# Visualization
fig, ax = plt.subplots(figsize=(12, 10))
pos = nx.kamada_kawai_layout(G)  # Use a fixed seed for consistent layouts
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', ax=ax)


# Update function for animation
def update(frame):
    # Step the simulation environment
    env.step()

    # Update Q-table using DT-driven RL
    global q_table
    q_table = dt_driven_q_learning(q_table, node_to_index, index_to_node)

    # Clear previous nodes
    ax.clear()
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', ax=ax)

    # Update AGV positions and draw current nodes
    for i, agv in enumerate(agvs):
        current_node = agv.current_node
        color = 'r' if i == 0 else 'g' if i == 1 else 'b'
        nx.draw_networkx_nodes(
            G, pos, nodelist=[current_node],
            node_color=color,
            node_size=700, ax=ax
        )

        # Display waiting status if applicable
        if agv.waiting:
            plt.annotate(f"Wait: {agv.wait_time}",
                         xy=pos[current_node],
                         xytext=(pos[current_node][0] + 0.05, pos[current_node][1] + 0.05))

        print(f"{agv.name} is at node {current_node}" +
              (f" (waiting: {agv.wait_time})" if agv.waiting else ""))

    return [ax]


# Animate the simulation
ani = animation.FuncAnimation(fig, update, frames=SIMULATION_TIME, interval=1000, blit=False)
plt.show()
