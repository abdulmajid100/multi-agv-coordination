import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

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

# Define AGVs and their tasks
agv_tasks = {
    'AGV1': [[28, 18, 19, 29]],
    'AGV2': [[2, 4, 11, 12, 13, 14, 15, 25]],
    'AGV3': [[3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29]],
    'GEN': [[9, 16, 15, 25]]
}

# Initialize resource reservation for nodes
resource_states = {node: 0 for node in G.nodes()}

# Q-Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Initialize Q-table: states are (current_node, next_node), actions are ['move', 'stop']
Q = {}
for current_node in G.nodes():
    for next_node in G.successors(current_node):
        Q[(current_node, next_node)] = {'move': 0, 'stop': 0}

# Reward function for AGV actions
def get_reward(state, action, deadlock=False):
    if deadlock and action == 'move':
        return -100  # Large negative reward for causing a deadlock
    elif action == 'move':
        return 10  # Reward for successfully moving to the next node
    elif action == 'stop':
        return -1  # Small penalty for waiting
    return 0

# Epsilon-greedy policy to choose an action
def choose_action(state):
    if random.uniform(0, 1) < epsilon:  # Explore
        return random.choice(['move', 'stop'])
    else:  # Exploit
        return max(Q[state], key=Q[state].get)

# Deadlock detection (checks if moving causes conflicts)
def detect_deadlock(agv, current_node, next_node, updated_agv_tasks):
    # Get other AGVs excluding the current one
    other_agvs = [other_agv for other_agv in updated_agv_tasks if other_agv != agv]
    shared_nodes_with_others = {other_agv: [] for other_agv in other_agvs}

    for other_agv, tasks in updated_agv_tasks.items():
        if other_agv != agv and updated_agv_tasks[agv] and len(updated_agv_tasks[agv][0]) > 1 and len(tasks) > 0 and len(tasks[0]) > 1:
            current_task = updated_agv_tasks[agv][0]  # Use the updated task list
            other_current_task = tasks[0]  # Use the updated task list for the other AGV

            # Find shared nodes between the current AGV's task and the other AGV's task
            shared_nodes = set(current_task) & set(other_current_task)
            shared_nodes_with_others[other_agv] = list(shared_nodes)

            # Check if any shared node is occupied by another AGV
            if any(resource_states[shared_node] == other_agv for shared_node in shared_nodes):
                return True  # Return True if a deadlock is detected

    return False  # Return False if no deadlock is detected
#print(detect_deadlock('AGV1', current_node, next_node))
# Update Q-table for a single AGV
def update_q_table(agv):
    if agv_tasks[agv]:
        current_task = agv_tasks[agv][0]
        if len(current_task) > 1:
            current_node = current_task[0]
            next_node = current_task[1]
            state = (current_node, next_node)

            # Decide action (move or stop)
            action = choose_action(state)
            deadlock = detect_deadlock(agv, current_node, next_node, agv_tasks)  # Pass updated agv_tasks

            # Get reward for the chosen action
            reward = get_reward(state, action, deadlock)

            # Update Q-value
            if action == 'move' and not deadlock:
                future_state = (next_node, current_task[2] if len(current_task) > 2 else None)
                future_reward = max(Q[future_state].values()) if future_state in Q else 0
                Q[state][action] += alpha * (reward + gamma * future_reward - Q[state][action])

                # Move AGV to the next node
                resource_states[current_node] = 0
                resource_states[next_node] = agv
                agv_tasks[agv][0].pop(0)  # Pop the completed node
            elif action == 'stop':
                Q[state][action] += alpha * (reward - Q[state][action])  # Update Q-value for stop
        elif len(current_task) == 1:  # Remove task if completed
            agv_tasks[agv].pop(0)

# Plot initialization
fig, ax = plt.subplots()
pos = nx.spring_layout(G)

# Update function for animation
def update(frame):
    ax.clear()
    nx.draw(G, pos, with_labels=True, arrows=True, ax=ax)

    # Update resource states and AGV positions
    for agv in agv_tasks.keys():
        update_q_table(agv)

    # Draw AGVs at their current positions
    for agv, tasks in agv_tasks.items():
        if tasks and tasks[0]:
            current_node = tasks[0][0]
            nx.draw_networkx_nodes(
                G, pos, nodelist=[current_node],
                node_color='r' if agv == 'AGV1' else 'g' if agv == 'AGV2' else 'b' if agv == 'AGV3' else 'orange',
                node_size=500, ax=ax
            )

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(100), repeat=False, interval=1000)

plt.show()