import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Initialize the directed graph
G = nx.DiGraph()
# Define nodes, explicitly excluding 7 and 8
nodes = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# Define edges, ensuring no edges lead to or from nodes 7 and 8
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

# Parameters
num_agents = 4
num_episodes = 1000
max_steps = 20
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # Exploration rate

# Initialize Q-table with the correct size
num_states = len(nodes)
Q = np.zeros((num_states, num_states))  # Q-table size based on states


# Function to choose an action based on epsilon-greedy policy
def choose_action(state):
    valid_actions = list(G.successors(nodes[state]))  # Get valid successors
    if not valid_actions:  # If there are no valid actions, return a random action
        return random.choice(range(num_states))  # Random action
    if random.uniform(0, 1) < epsilon:
        return random.choice(valid_actions)  # Random action (no adjustment needed)
    else:
        valid_action_indices = [nodes.index(a) for a in valid_actions]
        return valid_action_indices[np.argmax(Q[state, valid_action_indices])]  # Return the best action index


# Function to update Q-values
def update_q_value(state, action, reward, next_state):
    best_next_action = np.argmax(Q[next_state])  # Get the best action for the next state
    Q[state, action] += learning_rate * (reward + discount_factor * Q[next_state, best_next_action] - Q[state, action])


# Function to simulate the environment
def simulate_environment():
    total_wait_time = 0
    agents_positions = [random.choice(nodes) for _ in range(num_agents)]

    # Store positions for animation
    all_positions = [[] for _ in range(num_agents)]

    for step in range(max_steps):
        for agent in range(num_agents):
            current_state = nodes.index(agents_positions[agent])  # Get the index of the current state
            action = choose_action(current_state)

            # Get the next state based on the chosen action
            valid_actions = list(G.successors(agents_positions[agent]))  # Get valid actions
            if valid_actions:  # Ensure there are valid actions
                next_node = valid_actions[action % len(valid_actions)]  # Use modulo to avoid out-of-bounds
            else:
                next_node = agents_positions[agent]  # Stay in the same node if no valid actions
            next_state = nodes.index(next_node)  # Get the next state index

            # Reward structure: -1 for each step taken, +10 for reaching node 9
            reward = 10 if agents_positions[agent] == 9 else -1

            update_q_value(current_state, action, reward, next_state)
            agents_positions[agent] = nodes[next_state]  # Update the agent's position

            total_wait_time += reward

            # Store the current position for animation
            all_positions[agent].append(agents_positions[agent])

    return all_positions  # Return the positions of the agents for animation


# Animation function
def animate_environment(frame_num, all_positions, pos):
    plt.cla()  # Clear the previous frame
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', arrows=True)

    # Draw the agents
    for agent in range(num_agents):
        if frame_num < len(all_positions[agent]):  # Check if the frame number is within bounds
            nx.draw_networkx_nodes(G, pos, nodelist=[all_positions[agent][frame_num]], node_color='red', node_size=500)

    plt.title(f"Frame {frame_num}")
    plt.axis('off')  # Turn off the axis


# Run the simulation and animate
def run_simulation():
    all_positions = simulate_environment()  # Get the positions of the agents for animation
    print(f"Final positions of agents after training: {all_positions}")  # Debugging output

    # Set up the animation
    pos = nx.spring_layout(G)  # Precompute layout for consistent animation
    fig = plt.figure(figsize=(10, 8))
    ani = animation.FuncAnimation(fig, animate_environment, frames=max_steps, fargs=(all_positions, pos), interval=500,
                                  blit=False)
    plt.show()


# Execute the simulation
run_simulation()