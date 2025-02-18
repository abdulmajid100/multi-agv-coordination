import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import copy

# Define the Policy Network
# Updated Policy Network with Layer Normalization
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, num_actions):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=1, padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 3), stride=1, padding=(0, 1))

        # Dynamically calculate the flattened size
        conv_output_size = self._get_conv_output_size((1, 3, 30))  # Example input shape
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv1(x)
            x = self.conv2(x)
            return x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax here
        return x


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
    [1, 4, 11, 12, 22],
    [2, 4, 11, 12, 13, 14, 15, 25],
    [26, 16, 17, 18, 28]
]
"""fixed_paths = [
    [9, 16, 15, 25],
    [25, 15, 14, 13, 12, 11, 4, 2],
    [11, 12, 13, 14, 15, 16, 17, 27]
]"""


def select_actions(policy_net, state_matrix, num_agents, deterministic=False):
    if isinstance(state_matrix, np.ndarray):
        state = torch.FloatTensor(state_matrix).unsqueeze(0)  # Convert NumPy array to PyTorch tensor
    elif isinstance(state_matrix, torch.Tensor):
        state = state_matrix.float().unsqueeze(0)  # Ensure it's a float tensor
    else:
        raise TypeError("state_matrix must be either a NumPy array or a PyTorch tensor")

    # Reshape the state to have the correct dimensions for the CNN
    #state = state.view(1, num_agents, 1, 30)  # Assuming 3 agents and 30 nodes
    #print(state_matrix.shape)
    num_actions = 2 ** num_agents - 1
    action_vectors = [list(map(int, bin(i)[2:].zfill(num_agents))) for i in range(1, num_actions + 1)]
    #print(state)
    logits = policy_net(state).squeeze(0)  # shape: (num_actions,)
    probabilities = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
    probabilities = probabilities + 1e-8  # Add epsilon for numerical stability
    #print(probabilities)
    probabilities_np = probabilities.detach().numpy()

    if np.any(np.isnan(probabilities_np)):
        raise ValueError("Probabilities contain NaN")

    if deterministic:
        # Select the action with the highest probability
        action_index = np.argmax(probabilities_np)
    else:
        # Sample an action based on the probabilities
        action_index = np.random.choice(len(probabilities_np), p=probabilities_np)

    log_prob = torch.log(probabilities[action_index])
    action_vector = action_vectors[action_index % len(action_vectors)]
    return action_vector, log_prob


def update_policy(policy_net, value_net, policy_optimizer, value_optimizer, rewards, log_probs, states, gamma, ppo_epochs=4, epsilon=0.2):
    # Compute discounted rewards
    discounted_rewards = []
    Gt = 0
    for reward in reversed(rewards):
        Gt = reward + gamma * Gt
        discounted_rewards.insert(0, Gt)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

    # Normalize rewards (optional, improves stability)
    # Normalize rewards only if there is more than one element
    if len(discounted_rewards) > 1:
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
    else:
        discounted_rewards = discounted_rewards  # No normalization needed for a single element

    # Perform multiple PPO updates
    for _ in range(ppo_epochs):
        # Recompute values for states during each epoch
        values = torch.stack([value_net(state.view(-1).float()) for state in states]).squeeze()
        values = values.view(-1)  # Reshape if necessary
        discounted_rewards = discounted_rewards.view(-1)  # Ensure target is also reshaped

        # Compute advantages
        advantages = discounted_rewards - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # Normalize advantages
        else:
            advantages = advantages
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages

        # Recompute log probabilities for the current policy
        new_log_probs = []
        for state, old_log_prob in zip(states, log_probs):
            logits = policy_net(state.unsqueeze(0))  # shape: (num_actions,)
            probabilities = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
            probabilities = probabilities + 1e-8  # Add epsilon for numerical stability
            #probabilities = policy_net(state.unsqueeze(0))  # Forward pass through policy network
            action_index = torch.argmax(torch.exp(old_log_prob))  # Get the action index from the old log prob
            new_log_prob = torch.log(probabilities.squeeze(0)[action_index])  # Recompute log prob
            new_log_probs.append(new_log_prob)

        new_log_probs = torch.stack(new_log_probs)

        # Compute the ratio of new policy to old policy
        old_log_probs = torch.stack(log_probs).detach()  # Detach old log probs to avoid backprop through them
        ratios = torch.exp(new_log_probs - old_log_probs)
        #print(new_log_probs)
        # Clipped surrogate objective
        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        #print(clipped_ratios)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
        #print(policy_loss)
        # Entropy regularization (optional, encourages exploration)
        entropy = -(torch.exp(new_log_probs) * new_log_probs).mean()
        policy_loss -= 0.01 * entropy  # Add entropy bonus

        # Compute value loss
        value_loss = F.mse_loss(values, discounted_rewards)

        # Optimize policy network
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        policy_optimizer.step()

        # Optimize value network
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        value_optimizer.step()

    # Print losses for debugging
    print(f"Policy Loss: {policy_loss.item():.6f}, Value Loss: {value_loss.item():.6f}")

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
def train_agents(num_agents, num_episodes, fixed_paths):
    G = create_graph()
    state_size = 30 * num_agents
    action_size = 2 ** num_agents - 1  # For 3 agents, 7 joint actions
    policy_net = PolicyNetwork(state_size, action_size)
    value_net = ValueNetwork(state_size)

    # Lowered learning rates for stability
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.005)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.005)
    gamma = 0.9  # Discount factor

    agents_paths = [[] for _ in range(num_agents)]
    reward_history = []

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        # Initialize each agent's path (start from a random point in its fixed path)
        agv_paths = copy.deepcopy(fixed_paths)
        visited_nodes = [None] * num_agents
        visited_nodes2 = [None] * num_agents
        log_probs = []
        rewards = []
        states = []

        # Build an initial state matrix (each row is a one-hot for the agent’s current path)
        state_matrix = np.zeros((num_agents, 30), dtype=np.float32)
        for agent_index, path in enumerate(agv_paths):
            if path:
                for node in path:
                    state_matrix[agent_index, node - 1] += 1
        #state_matrix = torch.FloatTensor(state_matrix.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        done = False
        reward = 0
        for step in range(100):
            # Select actions
            try:
                action_vector, step_log_prob = select_actions(policy_net, state_matrix, num_agents)
            except ValueError as e:
                print(f"ValueError in select_actions: {e}")
                return agents_paths, G, policy_net

            # Process each agent’s decision
            for agent_index, action in enumerate(reversed(action_vector)):
                if action == 1 and agv_paths[agent_index]:
                    current_pos = agv_paths[agent_index][0]
                    next_pos = (agv_paths[agent_index][1] if len(agv_paths[agent_index]) > 1 else current_pos)
                    for i in range(len(visited_nodes)):
                        if i != agent_index and (next_pos == visited_nodes[i] or next_pos == visited_nodes2[i]):
                            reward -= 1000  # Penalty for causing a deadlock
                            done = True
                            break
                    if not done:
                        visited_nodes2[agent_index] = current_pos
                        visited_nodes[agent_index] = next_pos
                        if len(agv_paths[agent_index]) > 1:
                            agv_paths[agent_index] = agv_paths[agent_index][1:]
                            reward += 10  # Reward for moving to the next node
                        else:
                            agv_paths[agent_index] = []
                            reward += 20  # Reward for reaching the goal
                            if all(not path for path in agv_paths):
                                reward += 500  # Reward for reaching the goal
                                done = True
                                break
                elif action == 0:
                    if len(agv_paths[agent_index]) >= 1:
                        current_pos = agv_paths[agent_index][0]
                        next_pos = (agv_paths[agent_index][1] if len(agv_paths[agent_index]) > 1
                                    else current_pos)
                    for i in range(len(visited_nodes)):
                        # print(i, "i")
                        # print(agent_index, "j")
                        if i != agent_index and (next_pos != visited_nodes[i] and next_pos != visited_nodes2[i]):
                            # print(agent_index, "agent",next_pos, "next_pos", visited_nodes, "visited_nodes[i]", visited_nodes2, "visited_nodes2[i]")
                            reward -= 5  # Penalty for causing a deadlock
                        else:
                            reward += 1  # Reward for moving to the next node
                    """if len(agv_paths[agent_index]) >= 1:
                        current_pos = agv_paths[agent_index][0]
                        next_pos = agv_paths[agent_index][0]
                        visited_nodes2[agent_index] = current_pos
                        visited_nodes[agent_index] = next_pos"""
                if not all(not path for path in agv_paths):
                    reward -= 10  # Penalty for not reaching the goal
                reward -= 1  # Default reward if no action taken
                    #print(agv_paths[agent_index])
                #reward -= 5 * len(agv_paths[agent_index])  # Penalize longer paths

                # Record current position (if available) for visualization
                if agv_paths[agent_index]:
                    agents_paths[agent_index].append(agv_paths[agent_index][0])
                if done:
                    #print("done1")
                    break
            log_probs.append(step_log_prob)
            rewards.append(reward)
            states.append(state_matrix.clone() if isinstance(state_matrix, torch.Tensor) else torch.from_numpy(state_matrix).clone())

            if done:
                break

            # Update the state matrix at each time step before the next decision
            state_matrix = np.zeros((num_agents, 30), dtype=np.float32)
            for agent_index, path in enumerate(agv_paths):
                if path:
                    state_matrix[agent_index, path[0] - 1] = 1.0
            #state_matrix = torch.FloatTensor(state_matrix.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        # Update policy and value networks using PPO
        update_policy(policy_net, value_net, policy_optimizer, value_optimizer, rewards, log_probs, states, gamma)

        total_reward = sum(rewards)
        reward_history.append(total_reward)

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    # Plot training rewards
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
        episode_paths = [[] for _ in range(num_agents)]

        state_matrix = np.zeros((num_agents, 30), dtype=np.float32)
        for agent_index, path in enumerate(agv_paths):
            if path:
                for node in path:
                    state_matrix[agent_index, node - 1] += 1
        #state_matrix = torch.flatten(torch.from_numpy(state_matrix).float())

        for step in range(100):
            action_vector, _ = select_actions(policy_net, state_matrix, num_agents, deterministic=True)
            for agent_index, action in enumerate(reversed(action_vector)):
                if agv_paths[agent_index]:
                    current_pos = agv_paths[agent_index][0]
                    if action == 1 and agv_paths[agent_index]:
                        agv_paths[agent_index] = agv_paths[agent_index][1:] if len(agv_paths[agent_index]) > 1 else []
                        if agv_paths[agent_index]:
                            current_pos = agv_paths[agent_index][0]
                    episode_paths[agent_index].append(current_pos)
            state_matrix = np.zeros((num_agents, 30), dtype=np.float32)
            for agent_index, path in enumerate(agv_paths):
                if path:
                    state_matrix[agent_index, path[0] - 1] = 1.0
            #state_matrix = torch.FloatTensor(state_matrix.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        test_paths.append(episode_paths)
        #print(episode_paths)
    return test_paths


def visualize_agents(agents_paths, G):
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots()
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
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

    ani = animation.FuncAnimation(fig, update, frames=max(len(path) for path in agents_paths), repeat=False, interval=1000)
    plt.show()


# Main execution
if __name__ == "__main__":
    num_agents = len(fixed_paths)
    num_episodes = 300

    # Train the agents
    agents_paths, G, trained_policy = train_agents(num_agents, num_episodes, fixed_paths)

    # Test the trained policy
    test_results = test_policy(trained_policy, num_agents, fixed_paths)

    print("Visualizing training results...")
    visualize_agents(agents_paths, G)

    print("Visualizing test results...")
    visualize_agents(test_results[0], G)