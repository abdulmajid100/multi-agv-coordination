import numpy as np
import random
import networkx as nx
from graph_env_c import GraphEnv  # Assuming 'graph_env.py' is your new environment file

# Define the graph
G = nx.DiGraph()
nodes = list(range(1, 30)) + [9]
G.add_nodes_from(nodes)
# Q-learning parameters
alpha = 0.001  # Learning rate
gamma = 0.9  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon
num_episodes = 1200  # Number of episodes
alpha_start = 0.1
alpha_decay = 0.01
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

# Environment setup
goals = [4, 18]  # Example goal node
initial_positions = [1, 15]  # Example initial position

env = GraphEnv(G, num_agents=2, goals=goals, initial_positions=initial_positions)


def state_to_index(state, nodes):
    return nodes.index(state[0])  # Assuming single agent


def action_to_index(action):
    return action


def q_learning(env, num_episodes):
    nodes = env.nodes
    q_tables = [np.zeros((len(nodes), len(nodes))) for _ in range(env.num_agents)]
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
                state_index = state_to_index(state, nodes)
                if random.uniform(0, 1) < epsilon:
                    neighbors = list(env.graph.neighbors(state[0]))
                    action = random.choice(range(len(neighbors)))  # Explore
                else:
                    action = np.argmax(q_tables[i][state_index])  # Exploit

                actions.append(action)

            next_state, rewards, done, info = env.step(actions)
            total_rewards += rewards

            for i in range(env.num_agents):
                state_index = state_to_index(state, nodes)
                next_state_index = state_to_index(next_state, nodes)
                best_next_action = np.argmax(q_tables[i][next_state_index])
                td_target = rewards[i] + gamma * q_tables[i][next_state_index][best_next_action]
                td_error = td_target - q_tables[i][state_index][actions[i]]
                q_tables[i][state_index][actions[i]] += alpha * td_error

            state = next_state

        if np.sum(total_rewards) > np.sum(previous_total_rewards):
            epsilon = max(epsilon_min, epsilon * 0.99)
        else:
            epsilon = max(epsilon_min, epsilon * 0.999)

        previous_total_rewards = total_rewards

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Rewards: {total_rewards}, Epsilon: {epsilon:.4f}")

    return q_tables


def main():
    q_tables = q_learning(env, num_episodes)

    print("Training completed.")

    # Testing trained agents
    state = env.reset()
    done = False
    actions_list = []
    while not done:
        actions = []
        for i in range(env.num_agents):
            state_index = state_to_index(state, env.nodes)
            action = np.argmax(q_tables[i][state_index])
            actions.append(action)
        actions_list.append(actions)
        print(f"Agents' positions: {state}")
        state, _, done, _ = env.step(actions)

    env.animate(actions_list)


if __name__ == "__main__":
    main()