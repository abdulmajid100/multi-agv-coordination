import numpy as np
import random
import networkx as nx
from graph_env_c_new import GraphEnv

# Q-learning parameters
alpha = 0.05  # Learning rate
gamma = 0.99  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.2  # Minimum exploration rate
epsilon_decay = 0.9973  # Decay rate for epsilon
num_episodes = 2000  # Number of episodes
alpha_start = 1
alpha_decay = 0.01

G = nx.Graph()

# Create a grid-like graph structure based on your original grid layout
grid_size = (10, 10)

# Add all nodes that are not obstacles
obstacles = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (0, 8), (0, 9),
             (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (1, 9),
             (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9),
             (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9),
             (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7), (4, 8), (4, 9),
             (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
             (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 6), (6, 7), (6, 8), (6, 9),
             (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8), (7, 9),
             (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 6), (8, 7), (8, 8), (8, 9),
             (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 6), (9, 7), (9, 8), (9, 9)]

# Add nodes
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        if (x, y) not in obstacles:
            # We'll encode 2D coordinates to a single node ID
            node_id = x * grid_size[1] + y
            G.add_node(node_id, pos=(x, y))

# Add edges (connections between adjacent nodes)
for x in range(grid_size[0]):
    for y in range(grid_size[1]):
        if (x, y) not in obstacles:
            current_node = x * grid_size[1] + y

            # Check adjacent cells
            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            for nx, ny in neighbors:
                if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1] and (nx, ny) not in obstacles:
                    neighbor_node = nx * grid_size[1] + ny
                    G.add_edge(current_node, neighbor_node)
def state_to_index(state, num_nodes):
    """
    Convert a multi-agent state on a graph to a unique index.

    Args:
        state: Array of node positions for each agent
        num_nodes: Total number of nodes in the graph

    Returns:
        A unique integer index representing the state
    """
    num_agents = len(state)
    index = 0
    factor = num_nodes ** (num_agents - 1)

    for i in range(num_agents):
        node_id = state[i]
        index += node_id * factor
        factor //= num_nodes

    return index


def q_learning(env, num_episodes, num_nodes):
    """
    Q-learning algorithm for multi-agent path finding in a graph environment.

    Args:
        env: The graph environment
        num_episodes: Number of episodes to train
        num_nodes: Total number of nodes in the graph

    Returns:
        A list of Q-tables, one for each agent
    """
    # Get maximum action size (largest number of neighbors for any node)
    max_action_size = max(env.get_valid_actions(i) for i in range(env.num_agents))

    # Initialize Q-tables for each agent
    print(num_nodes)
    q_tables = [np.zeros((num_nodes ** env.num_agents, max_action_size)) for _ in range(env.num_agents)]

    previous_total_rewards = float('-inf')

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = np.zeros(env.num_agents)
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
        alpha = max(0.001, alpha_start / (1 + episode * alpha_decay))

        while not done:
            actions = []
            for i in range(env.num_agents):
                valid_actions = env.get_valid_actions(i)

                if random.uniform(0, 1) < epsilon:
                    # Explore: choose a random valid action
                    action = random.randint(0, valid_actions - 1)
                else:
                    # Exploit: choose the best action from Q-table
                    state_index = state_to_index(state, num_nodes)
                    # Only consider valid actions
                    valid_q_values = q_tables[i][state_index][:valid_actions]
                    action = np.argmax(valid_q_values)

                actions.append(action)

            next_state, rewards, done, info = env.step(actions)
            total_rewards += rewards

            # Update Q-values for each agent
            for i in range(env.num_agents):
                state_index = state_to_index(state, num_nodes)
                next_state_index = state_to_index(next_state, num_nodes)

                # Get the number of valid actions for the next state
                next_valid_actions = env.get_valid_actions(i)
                # Find best next action considering only valid actions
                print(next_state_index, next_valid_actions)
                best_next_action = np.argmax(q_tables[i][next_state_index][:next_valid_actions])

                # Calculate temporal difference target and error
                td_target = rewards[i] + gamma * q_tables[i][next_state_index][best_next_action]
                td_error = td_target - q_tables[i][state_index][actions[i]]

                # Update Q-value
                q_tables[i][state_index][actions[i]] += alpha * td_error

            state = next_state

        # Adjust epsilon based on performance
        if np.sum(total_rewards) > np.sum(previous_total_rewards):
            epsilon = max(epsilon_min, epsilon * 0.99)  # Decay faster if improving
        else:
            epsilon = max(epsilon_min, epsilon * 0.999)  # Decay slower if not improving

        previous_total_rewards = total_rewards

        # Print episode progress
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Rewards: {total_rewards}, Epsilon: {epsilon:.4f}")

    return q_tables


def main():
    # Create a graph for the environment
    grid_size = (10, 10)

    # Convert coordinate goals and initial positions to node IDs
    goals_coords = [(9, 5), (0, 5), (5, 9)]
    initial_positions_coords = [(5, 9), (9, 5), (0, 5)]

    goals = [g[0] * grid_size[1] + g[1] for g in goals_coords]
    initial_positions = [pos[0] * grid_size[1] + pos[1] for pos in initial_positions_coords]

    # Create the environment
    num_agents = 3
    env = GraphEnv(G, num_agents, goals, initial_positions)

    # Count total number of nodes
    num_nodes = len(G.nodes())

    # Train with Q-learning
    q_tables = q_learning(env, num_episodes, num_nodes)

    print("Training completed.")

    # Testing trained agents
    state = env.reset()
    done = False
    actions_list = []

    while not done:
        actions = []
        for i in range(num_agents):
            state_index = state_to_index(state, num_nodes)
            valid_actions = env.get_valid_actions(i)
            valid_q_values = q_tables[i][state_index][:valid_actions]
            action = np.argmax(valid_q_values)
            actions.append(action)

        actions_list.append(actions)
        print(f"Agents' positions: {state}")
        state, _, done, _ = env.step(actions)

    # Animate the result
    env.animate(actions_list)


if __name__ == "__main__":
    main()
