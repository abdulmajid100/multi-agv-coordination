import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from archive.graph_env2 import GraphEnv

# Q-learning hyperparameters
alpha = 0.05  # Learning rate
gamma = 0.99  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.997  # Decay rate for epsilon
num_episodes = 1000  # Number of episodes
alpha_start = 0.1  # Initial learning rate
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


def state_to_index(state, num_nodes):
    """
    Convert a multi-agent state to a unique index for Q-table.
    """
    num_agents = len(state)
    index = 0
    factor = 1

    for i in range(num_agents - 1, -1, -1):
        node_id = state[i] - 1  # Adjust for 1-based node IDs
        index += node_id * factor
        factor *= num_nodes

    return int(index)


def q_learning(env, num_episodes, num_nodes):
    """
    Q-learning algorithm that works with the existing GraphEnv.
    The environment automatically moves agents along shortest paths,
    so we just need to learn from the state transitions and rewards.
    """
    # Since the environment handles movement automatically, we have minimal actions
    # Action 0: Let the environment move the agent (follow shortest path)
    num_actions = 1

    # Initialize Q-table (state_space_size x num_actions)
    state_space_size = num_nodes ** env.num_agents
    q_table = np.zeros((state_space_size, num_actions))

    # Track rewards history
    rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        # Calculate epsilon for this episode
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
        alpha_current = max(0.001, alpha_start / (1 + episode * alpha_decay))

        while not done:
            state_index = state_to_index(state, num_nodes)

            # In this case, we only have one action (let environment handle movement)
            action = 0

            # Take step in environment (it handles the shortest path movement)
            next_state, rewards, done, info = env.step()

            # Calculate total reward for this step
            step_reward = np.sum(rewards)
            total_reward += step_reward

            # Update Q-value
            next_state_index = state_to_index(next_state, num_nodes)

            # Q-learning update
            best_next_action = np.argmax(q_table[next_state_index])
            td_target = step_reward + gamma * q_table[next_state_index][best_next_action]
            td_error = td_target - q_table[state_index][action]
            q_table[state_index][action] += alpha_current * td_error

            state = next_state

        rewards_history.append(total_reward)

        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(
                f"Episode {episode}/{num_episodes}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    # Plot rewards history
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    # Plot moving average
    if len(rewards_history) > 10:
        window_size = min(100, len(rewards_history))
        moving_avg = np.convolve(rewards_history, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards_history)), moving_avg, 'r-',
                 label=f'{window_size}-episode Moving Average')
        plt.legend()

    plt.savefig('rewards_history.png')
    plt.show()

    print(f"Final average reward (last 100 episodes): {np.mean(rewards_history[-100:]):.2f}")

    return q_table, rewards_history


def test_trained_agents(env):
    """Test the environment by running one episode and collecting the path."""
    state = env.reset()
    done = False
    step_count = 0
    states_sequence = [state.copy()]

    print(f"Initial state: {state}")
    print(f"Goals: {env.goals}")

    while not done:
        state, rewards, done, info = env.step()
        states_sequence.append(state.copy())
        step_count += 1

        if step_count % 10 == 0:
            print(f"Step {step_count}: Agents at {state}, Rewards: {rewards}")

    print(f"Testing completed in {step_count} steps")
    print(f"Final positions: {state}")

    return states_sequence


def main():
    """
    Main function to set up the environment, train with Q-learning, and visualize results.
    """
    # Define goals and initial positions for agents
    goals = [1, 25, 28]  # Goal nodes for each agent
    initial_positions = [15, 12, 26]  # Starting positions for each agent
    num_agents = len(goals)

    # Create the environment
    env = GraphEnv(G, num_agents, goals, initial_positions)

    # Count total number of nodes in the graph
    num_nodes = len(G.nodes())

    print(f"Environment created with {num_agents} agents")
    print(f"Goals: {goals}")
    print(f"Initial positions: {initial_positions}")
    print(f"Total nodes in graph: {num_nodes}")

    # Train with Q-learning (learning from the environment's behavior)
    print("\nStarting Q-learning training...")
    q_table, rewards_history = q_learning(env, num_episodes, num_nodes)
    print("Training completed.")

    # Print final statistics
    print(f"Final total reward: {rewards_history[-1]:.2f}")
    print(f"Best episode reward: {max(rewards_history):.2f}")

    # Test the environment
    print("\nTesting the environment...")
    states_sequence = test_trained_agents(env)

    # Animate the result
    print("\nGenerating animation...")
    anim = env.animate()
    plt.show()


if __name__ == "__main__":
    main()
