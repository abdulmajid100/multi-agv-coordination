import numpy as np
import random
import networkx as nx
from graph_env import GraphEnv  # Ensure 'graph_env_c.py' is structured for graph-based operations

# Define the graph
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.999  # Decay rate for epsilon
num_episodes = 1200  # Number of episodes
alpha_start = 0.1
alpha_decay = 0.01
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

# Environment setup
goals = [11]  # Goals for each agent
initial_positions = [1]  # Starting positions for each agent

env = GraphEnv(G, num_agents=1, goals=goals, initial_positions=initial_positions)

def state_to_index(state, nodes):
    # Convert multi-agent state to a unique index
    indices = [nodes.index(agent_pos) for agent_pos in state]
    return tuple(indices)

def action_to_index(action):
    return action


def q_learning(env, num_episodes):
    nodes = env.nodes
    q_tables = [np.zeros((len(nodes), len(nodes))) for _ in range(env.num_agents)]

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = np.zeros(env.num_agents)
        epsilon = max(1 - episode / 600, 0.01)
        alpha = max(0.001, alpha_start / (1 + episode * alpha_decay))

        while not done:
            actions = []
            for i in range(env.num_agents):
                state_index = nodes.index(state[i])
                neighbors = list(env.graph.neighbors(state[i]))

                if random.uniform(0, 1) < epsilon and neighbors:  # Explore
                    action = random.choice(range(len(neighbors)))
                else:  # Exploit
                    action = np.argmax([q_tables[i][state_index][nodes.index(n)] for n in neighbors])

                actions.append(action)

            next_state, rewards, done, _ = env.step(actions)
            total_rewards += rewards

            for i in range(env.num_agents):
                state_index = nodes.index(state[i])
                next_state_index = nodes.index(next_state[i])
                best_next_action = np.argmax([q_tables[i][next_state_index][nodes.index(n)] for n in neighbors])

                td_target = rewards[i] + gamma * q_tables[i][next_state_index][best_next_action]
                td_error = td_target - q_tables[i][state_index][actions[i]]
                q_tables[i][state_index][actions[i]] += alpha * td_error

            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Rewards: {total_rewards}, Epsilon: {epsilon:.4f}")

    return q_tables


def main():
    env = GraphEnv(G, num_agents=2, goals=goals, initial_positions=initial_positions)
    q_tables = q_learning(env, num_episodes)

    print("Training completed.")

    # Testing trained agents
    state = env.reset()
    done = False
    actions_list = []
    while not done:
        actions = []
        for i in range(env.num_agents):
            state_index = env.nodes.index(state[i])
            neighbors = list(env.graph.neighbors(state[i]))
            action = np.argmax([q_tables[i][state_index][env.nodes.index(n)] for n in neighbors])
            actions.append(action)

        actions_list.append(actions)
        state, _, done, _ = env.step(actions)
        print(f"Agents' positions: {state}")

    env.animate(actions_list)  # Visualize the path taken by the agents

if __name__ == "__main__":
    main()