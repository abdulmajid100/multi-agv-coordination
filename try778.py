import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from graph_env_c_new import GraphEnv

# Q-learning hyperparameters
alpha = 0.05        # Learning rate
gamma = 0.99        # Discount factor
epsilon_start = 1.0 # Initial exploration rate
epsilon_min = 0.1   # Minimum exploration rate
epsilon_decay = 0.997  # Decay rate for epsilon
num_episodes = 1000 # Number of episodes
alpha_start = 0.1   # Initial learning rate
alpha_decay = 0.01  # Learning rate decay

# Initialize the directed graph
G = nx.DiGraph()
nodes = list(range(1, 30))
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

def state_to_index(state, num_nodes, graph=None):
    """
    Convert a multi-agent state on a graph to a unique index.
    This function maps a multi-dimensional state (positions of multiple agents)
    to a single integer index for use in the Q-table.
    The function incorporates node degree information and adjacency information as features.

    Args:
        state: Array of node positions for each agent
        num_nodes: Total number of nodes in the graph
        graph: Optional networkx graph object to get node degrees and adjacency info

    Returns:
        A unique integer index representing the state
    """
    num_agents = len(state)
    index = 0
    factor = 1

    # Process agents in reverse order for better numerical stability
    for i in range(num_agents - 1, -1, -1):
        # Adjust for 1-based node IDs by subtracting 1
        node_id = state[i] - 1

        # Incorporate node degree and adjacency info if graph is provided
        if graph is None:
            # Get the degree of the node (number of connections)
            node_degree = graph.degree(state[i])

            # Get adjacency information - check if this node is connected to other agent nodes
            adjacency_factor = 0
            for j in range(num_agents):
                if i != j and graph.has_edge(state[i], state[j]):
                    # Add a small value for each connection to another agent's node
                    adjacency_factor += 0.01

            # Use node degree and adjacency as features but scale them down
            degree_factor = node_degree / (10 * num_nodes)  # Scale down the degree influence

            # Add the node_id and the scaled factors multiplied by the factor
            index += (node_id + degree_factor + adjacency_factor) * factor
        else:
            # Simply add the node_id multiplied by the factor
            index += node_id * factor

        # Update the factor for the next agent
        factor *= num_nodes

    # Ensure the index is an integer and within bounds
    return int(index)


def q_learning(env, num_episodes, num_nodes):
    """
    Q-learning algorithm for multi-agent path finding in a graph environment.

    This implementation uses a separate Q-table for each agent and employs
    an epsilon-greedy exploration strategy with adaptive decay.

    Args:
        env: The graph environment
        num_episodes: Number of episodes to train
        num_nodes: Total number of nodes in the graph

    Returns:
        A list of Q-tables, one for each agent
    """
    # Get maximum action size (largest number of neighbors for any node in the graph + 1 for stay action)
    max_neighbors = max(len(list(env.graph.neighbors(node))) for node in env.graph.nodes())
    max_action_size = max_neighbors + 1  # +1 for the "stay" action

    # Initialize Q-tables for each agent
    q_tables = [np.zeros((num_nodes ** env.num_agents, max_action_size)) for _ in range(env.num_agents)]

    previous_total_rewards = float('-inf')

    # Track rewards history for visualization
    rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = np.zeros(env.num_agents)

        # Calculate epsilon and alpha for this episode
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
        alpha_current = max(0.001, alpha_start / (1 + episode * alpha_decay))

        while not done:
            actions = []

            # Determine actions for each agent
            for i in range(env.num_agents):
                valid_actions = env.get_valid_actions(i)
                state_index = state_to_index(state, num_nodes, env.graph)

                if random.uniform(0, 1) < epsilon:
                    # Explore: choose a random valid action
                    action = random.randint(0, valid_actions - 1)
                else:
                    # Exploit: choose the best action from Q-table
                    valid_q_values = q_tables[i][state_index][:valid_actions]
                    action = np.argmax(valid_q_values)

                actions.append(action)

            # Take a step in the environment
            next_state, rewards, done, info = env.step(actions)
            total_rewards += rewards

            # Update Q-values for each agent
            for i in range(env.num_agents):
                state_index = state_to_index(state, num_nodes, env.graph)
                next_state_index = state_to_index(next_state, num_nodes, env.graph)

                # Get the number of valid actions for the next state
                next_valid_actions = env.get_valid_actions(i)

                # Find best next action considering only valid actions
                best_next_action = np.argmax(q_tables[i][next_state_index][:next_valid_actions])

                # Calculate temporal difference target and error
                td_target = rewards[i] + gamma * q_tables[i][next_state_index][best_next_action]
                td_error = td_target - q_tables[i][state_index][actions[i]]

                # Update Q-value
                q_tables[i][state_index][actions[i]] += alpha_current * td_error

            state = next_state

        # Adaptive epsilon decay based on performance
        if np.sum(total_rewards) > np.sum(previous_total_rewards):
            epsilon = max(epsilon_min, epsilon * 0.99)  # Decay faster if improving
        else:
            epsilon = max(epsilon_min, epsilon * 0.999)  # Decay slower if not improving

        previous_total_rewards = total_rewards.copy()  # Create a copy to avoid reference issues

        # Append the sum of rewards to history
        rewards_history.append(np.sum(total_rewards))

        # Calculate average reward over last 100 episodes (or fewer if not enough episodes yet)
        window_size = min(100, episode + 1)
        avg_reward = np.mean(rewards_history[-window_size:])

        # Print episode progress more frequently (every 20 episodes)
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Rewards: {total_rewards}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    # Plot rewards history
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    # Plot moving average for smoother visualization
    if len(rewards_history) > 10:
        window_size = min(100, len(rewards_history))
        moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards_history)), moving_avg, 'r-', label=f'{window_size}-episode Moving Average')
        plt.legend()

    plt.savefig('rewards_history.png')
    plt.show()

    print(f"Final average reward (last 100 episodes): {np.mean(rewards_history[-100:]):.2f}")

    return q_tables, rewards_history


def main():
    """
    Main function to set up the environment, train agents using Q-learning,
    and visualize the results.
    """
    # Define goals and initial positions for agents
    goals = [1, 25, 28]  # Goal nodes for each agent
    initial_positions = [15, 12, 26]  # Starting positions for each agent
    num_agents = len(goals)  # Number of agents

    # Create the environment
    env = GraphEnv(G, num_agents, goals, initial_positions)

    # Count total number of nodes in the graph
    num_nodes = len(G.nodes())

    print(f"Environment created with {num_agents} agents")
    print(f"Goals: {goals}")
    print(f"Initial positions: {initial_positions}")
    print(f"Total nodes in graph: {num_nodes}")

    # Train agents with Q-learning
    print("\nStarting Q-learning training...")
    q_tables, rewards_history = q_learning(env, num_episodes, num_nodes)
    print("Training completed.")

    # Print final statistics about learning
    print(f"Final total reward: {rewards_history[-1]:.2f}")
    print(f"Best episode reward: {max(rewards_history):.2f}")

    # Test the trained agents
    print("\nTesting trained agents...")
    state = env.reset()
    done = False
    actions_list = []
    step_count = 0

    while not done:
        actions = []
        for i in range(num_agents):
            state_index = state_to_index(state, num_nodes, env.graph)
            valid_actions = env.get_valid_actions(i)
            valid_q_values = q_tables[i][state_index][:valid_actions]
            action = np.argmax(valid_q_values)
            actions.append(action)

        # Record actions and take a step
        actions_list.append(actions)
        if step_count % 10 == 0:  # Print every 10 steps to reduce output
            print(f"Step {step_count}: Agents at {state}, Actions: {actions}")

        state, rewards, done, info = env.step(actions)
        step_count += 1

    print(f"Testing completed in {step_count} steps")
    print(f"Final positions: {state}")

    # Animate the result
    print("\nGenerating animation...")
    anim = env.animate(actions_list)
    # Save a reference to the animation to prevent it from being garbage collected
    plt.show()


if __name__ == "__main__":
    main()
