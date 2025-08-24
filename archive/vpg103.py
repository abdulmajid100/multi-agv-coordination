import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, num_agents):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, num_agents)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Create a directed graph environment
def create_graph():
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
    G.remove_nodes_from([7, 8])
    return G


# Define fixed paths for each AGV
fixed_paths = [
    [1, 4, 11, 12, 22],  # Path for AGV 1
    [2, 4, 11, 12, 13, 14, 15, 25],  # Path for AGV 2
    [3, 4, 11, 12, 13, 14, 15, 16, 26]  # Path for AGV 3
]


def select_actions(policy_net, state_matrix):
    if isinstance(state_matrix, np.ndarray):
        state = torch.FloatTensor(state_matrix.astype(np.float32)).unsqueeze(0)
    elif isinstance(state_matrix, torch.Tensor):
        state = state_matrix.float().unsqueeze(0)
    else:
        raise TypeError("state_matrix must be either a NumPy array or a PyTorch tensor")

    num_agents = 3
    num_actions = 2 ** num_agents - 1
    action_vectors = [list(map(int, bin(i)[2:].zfill(num_agents))) for i in range(1, num_actions + 1)]

    probabilities = policy_net(state).squeeze(0)
    action_index = np.random.choice(len(probabilities), p=probabilities.detach().numpy())
    action_vector = action_vectors[action_index % len(action_vectors)]
    log_prob = torch.log(probabilities[action_index])

    return action_vector, log_prob


def update_policy(policy_net, value_net, policy_optimizer, value_optimizer, rewards, log_probs, states, gamma=0.99):
    discounted_rewards = []
    for t in range(len(rewards)):
        G = sum(gamma ** i * rewards[i + t] for i in range(len(rewards) - t))
        discounted_rewards.append(G)
    discounted_rewards = torch.FloatTensor(discounted_rewards)

    states = torch.FloatTensor(np.array(states))
    values = value_net(states).squeeze()

    advantages = discounted_rewards - values.detach()

    policy_loss = -torch.sum(torch.stack(log_probs) * advantages)
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_loss = F.mse_loss(values, discounted_rewards)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    print(f"Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")


def train_agents(num_agents, num_episodes, fixed_paths):
    G = create_graph()
    state_size = 90
    action_size = 7
    policy_net = PolicyNetwork(state_size, action_size)
    value_net = ValueNetwork(state_size)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.01)
    gamma = 0.99

    agents_paths = [[] for _ in range(num_agents)]
    reward_history = []  # For plotting rewards

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        # Initialize paths for agents
        agv_paths = []
        for path in fixed_paths:
            start_index = random.randint(0, len(path) - 1)
            agv_paths.append(path[start_index:])

        visited_nodes = []
        log_probs = []
        rewards = []
        states = []

        # Initialize state matrix
        state_matrix = np.zeros((num_agents, 30))
        for agent_index, path in enumerate(agv_paths):
            if path:
                for node in path:
                    state_matrix[agent_index, node - 1] += 1
        state_matrix = torch.flatten(torch.from_numpy(state_matrix).float())  # Ensure float dtype

        # Flag for stopping
        done = False

        for step in range(200):
            # Select actions
            action_vector, step_log_prob = select_actions(policy_net, state_matrix)

            for agent_index, action in enumerate(action_vector):
                if action == 1 and agv_paths[agent_index]:
                    current_pos = agv_paths[agent_index][0]
                    next_pos = agv_paths[agent_index][1] if len(agv_paths[agent_index]) > 1 else current_pos

                    # Reward logic
                    if next_pos not in visited_nodes:
                        reward = 10
                    else:
                        reward = -100
                        done = True  # End the episode if penalty occurs

                    visited_nodes.append(next_pos)
                    visited_nodes = visited_nodes[-2:]  # Keep the last 2 visited nodes
                    agv_paths[agent_index] = agv_paths[agent_index][1:] if len(agv_paths[agent_index]) > 1 else []
                else:
                    current_pos = agv_paths[agent_index][0] if agv_paths[agent_index] else None
                    reward = 1  # Default reward for no action

                # Store rewards, log_probs, and states
                log_probs.append(step_log_prob)
                rewards.append(reward)
                states.append(state_matrix.clone())  # Clone to avoid in-place modifications

                if current_pos is not None:
                    agents_paths[agent_index].append(current_pos)

                if done:
                    break

            if done:
                break  # End the episode if flagged

        # Normalize rewards
        #rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

        # Update policy and value networks
        update_policy(policy_net, value_net, policy_optimizer, value_optimizer, rewards, log_probs, states, gamma)

        # Compute total and discounted rewards
        total_reward = sum(rewards)
        reward_history.append(total_reward)
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = sum(gamma ** i * rewards[i + t] for i in range(len(rewards) - t))
            discounted_rewards.append(Gt)
        average_discounted_reward = np.mean(discounted_rewards)

        # Logging for debugging
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
        print(f"Episode {episode + 1}: Average Discounted Reward = {average_discounted_reward:.2f}")

    # Plot rewards after training
    import matplotlib.pyplot as plt
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Time')
    plt.show()

    return agents_paths, G, policy_net


def test_policy(policy_net, num_agents, fixed_paths, num_test_episodes=5):
    G = create_graph()
    test_paths = []

    for episode in range(num_test_episodes):
        print(f"Test Episode {episode + 1}/{num_test_episodes}")

        agv_paths = [path.copy() for path in fixed_paths]
        #print(agv_paths)
        episode_paths = [[] for _ in range(num_agents)]

        state_matrix = np.zeros((num_agents, 30))
        for agent_index, path in enumerate(agv_paths):
            if path:
                for node in path:
                    state_matrix[agent_index, node - 1] += 1

        state_matrix = torch.flatten(torch.from_numpy(state_matrix))

        for step in range(500):
            action_vector, _ = select_actions(policy_net, state_matrix)

            for agent_index, action in enumerate(action_vector):
                #print(agent_index)
                if agv_paths[agent_index]:
                    current_pos = agv_paths[agent_index][0]

                    if action == 1 and agv_paths[agent_index]:
                        agv_paths[agent_index] = agv_paths[agent_index][1:] if len(agv_paths[agent_index]) > 1 else []
                        if agv_paths[agent_index]:
                            current_pos = agv_paths[agent_index][0]
                    episode_paths[agent_index].append(current_pos)
        test_paths.append(episode_paths)
    print(agv_paths)
    print((test_paths))
    return test_paths


def visualize_agents(agents_paths, G):
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots()
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    #plt.show()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'yellow', 'cyan']
    agent_positions = [None] * len(agents_paths)

    def update(frame):
        ax.clear()
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10)

        for i, path in enumerate(agents_paths):
            if frame < len(path):
                agent_positions[i] = path[frame]

        for i, node in enumerate(agent_positions):
            if node is not None:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=[node],
                    node_color=colors[i % len(colors)],
                    node_size=300,
                    ax=ax,
                    label=f'Agent {i + 1}'
                )

        ax.legend()

    ani = animation.FuncAnimation(fig, update, frames=max(len(path) for path in agents_paths), repeat=False,
                                  interval=1000)
    plt.show()


# Main execution
if __name__ == "__main__":
    # Training parameters
    num_agents = 3
    num_episodes = 500

    # Train the agents
    agents_paths, G, trained_policy = train_agents(num_agents, num_episodes, fixed_paths)

    # Test the trained policy
    test_results = test_policy(trained_policy, num_agents, fixed_paths)

    # Visualize training results
    print("Visualizing training results...")
    visualize_agents(agents_paths, G)

    # Visualize test results
    print("Visualizing test results...")
    visualize_agents(test_results[4], G)  # Visualize first test episode