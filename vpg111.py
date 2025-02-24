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
import scipy.signal

# Define the Policy Network
# Updated Policy Network with Layer Normalization
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, num_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.ln1 = nn.LayerNorm(256, eps=1e-8)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128, eps=1e-8)
        self.fc3 = nn.Linear(128, num_actions)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)
        return x


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.ln1 = nn.LayerNorm(256, eps=1e-8)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128, eps=1e-8)
        self.fc3 = nn.Linear(128, 1)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


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
        state = torch.FloatTensor(state_matrix.astype(np.float32)).unsqueeze(0)
    elif isinstance(state_matrix, torch.Tensor):
        state = state_matrix.float().unsqueeze(0)
    else:
        raise TypeError("state_matrix must be either a NumPy array or a PyTorch tensor")

    # Define action space
    num_actions = 2 ** num_agents - 1
    action_vectors = [list(map(int, bin(i)[2:].zfill(num_agents))) for i in range(1, num_actions + 1)]

    # Get action log probabilities
    log_probabilities = policy_net(state).squeeze(0)  # Output from log_softmax
    probabilities = torch.exp(log_probabilities)  # To get actual probabilities if needed

    probabilities_np = probabilities.detach().numpy()
    if np.any(np.isnan(probabilities_np)):
        raise ValueError("Probabilities contain NaN")

    if deterministic:
        # Select the action with the highest probability
        action_index = torch.argmax(log_probabilities).item()
    else:
        # Sample an action based on the probabilities
        action_index = np.random.choice(len(probabilities_np), p=probabilities_np)

    log_prob = log_probabilities[action_index]  # Use log_softmax output directly
    action_vector = action_vectors[action_index % len(action_vectors)]
    return action_vector, log_prob


def compute_gae(rewards, values, gamma, lambda_):
    advantages = []
    advantage = 0
    for t in range(len(rewards)):
        delta_t = rewards[t] + gamma * (values[t+1] if t < len(rewards) - 1 else 0) - values[t]
        advantage = delta_t + gamma * lambda_ * advantage
        advantages.append(advantage)
    return torch.tensor(advantages, dtype=torch.float32)


def update_policy(policy_net, value_net, policy_optimizer, value_optimizer, rewards, log_probs, states, gamma):
    # Compute discounted rewards
    rewards.append(0)
    discounted_rewards = []
    Gt = 0
    for reward in reversed(rewards):
        Gt = reward + gamma * Gt
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
    discounted_rewards = reversed(discounted_rewards)
    discounted_rewards = discounted_rewards[:-1]
    #print(discounted_rewards)
    #print(rewards)

    values = torch.stack([value_net(state.float()) for state in states]).squeeze()
    #print(values)
    values1 = values.view(-1)
    values = values1.detach().numpy()
    values = np.append(values, 0)
    #print(values)
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    #print(deltas, "delta")
    #values = torch.cat((values, [0]))
    rewards = torch.tensor(rewards, dtype=torch.float32)
    #print(rewards)
    advantages = []
    Gt = 0
    for delta in reversed(deltas):
        Gt = delta + gamma * 0.9 * Gt
        advantages.append(Gt)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = reversed(advantages)
    #print(discounted_rewards)
    """if len(discounted_rewards) > 1:
        discounted_rewards = reversed(discounted_rewards)"""
    #print(discounted_rewards)
    #print(discounted_rewards[0])
    #print(rewards)
    # Normalize rewards
    if len(rewards) > 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    if len(discounted_rewards) > 1:
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
    #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
    #print((states))
    #print(states[0])
    # Compute value loss

    #print(values)

      # Reshape if necessary
    discounted_rewards = discounted_rewards.view(-1)  # Ensure target is also reshaped
    #print(values[0])
    value_loss = F.mse_loss(values1, discounted_rewards)

    # Compute advantages
    #advantages = compute_gae(rewards=rewards, values=values, gamma=gamma, lambda_=0.95)
    #advantages = discounted_rewards - values.detach()
    #advantages = rewards[-1:]
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #print(f"Advantages: {advantages}")

    # Compute policy loss with entropy regularization
    # Compute policy loss with entropy regularization
    entropy = -(torch.stack(log_probs) * torch.stack(log_probs).exp()).mean()
    policy_loss = -torch.stack([log_prob * advantage for log_prob, advantage in zip(log_probs, advantages)]).mean()
    policy_loss -= 0.1 * entropy  # Entropy regularization

    #entropy = -(torch.stack(log_probs) * torch.stack(log_probs).exp()).mean()
    #policy_loss -= 0.01 * entropy  # Add entropy regularization
    #print("policy_loss:", policy_loss, "value_loss:", value_loss)
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

    policy_net.apply(init_weights)
    value_net.apply(init_weights)
    # Lowered learning rates for stability
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.0001)

    # Initialize learning rate schedulers
    #policy_scheduler = torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=150, gamma=1)
    #value_scheduler = torch.optim.lr_scheduler.StepLR(value_optimizer, step_size=150, gamma=1)
    gamma = 0.99  # Discount factor

    agents_paths = [[] for _ in range(num_agents)]
    reward_history = []
    #print("Training")
    for episode in range(num_episodes):
        #print(f"Episode {episode + 1}/{num_episodes}")
        #print("Training")
        # Initialize each agent's path (start from a random point in its fixed path)
        agv_paths = []
        agv_paths = copy.deepcopy(fixed_paths)
        agv_paths = fixed_paths.copy()
        '''for path in fixed_paths:
            start_index = random.randint(0, len(path) - 1)
            agv_paths.append(path[start_index:].copy())'''

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
        state_matrix = torch.flatten(torch.from_numpy(state_matrix).float())
        #print(agv_paths)
        done = False
        reward = 0
        for step in range(100):
            # Select actions
            try:

                #print("Selecting Actions")
                #print(policy_net, state_matrix, num_agents)
                action_vector, step_log_prob = select_actions(policy_net, state_matrix, num_agents)
                #print("action selected")
            except ValueError as e:
                print(f"ValueError in select_actions: {e}")
                return agents_paths, G, policy_net

            # Process each agent’s decision
            for agent_index, action in enumerate(reversed(action_vector)):
                #print(agent_index, "agent_index")
                if action == 1 and agv_paths[agent_index]:
                    #print(action_vector, agent_index)
                    current_pos = agv_paths[agent_index][0]
                    next_pos = (agv_paths[agent_index][1] if len(agv_paths[agent_index]) > 1
                                else current_pos)
                    for i in range(len(visited_nodes)):
                        #print(i, "i")
                        #print(agent_index, "j")
                        if i != agent_index and (next_pos == visited_nodes[i] or next_pos == visited_nodes2[i]):
                            #print(agent_index, "agent",next_pos, "next_pos", visited_nodes, "visited_nodes[i]", visited_nodes2, "visited_nodes2[i]")
                            reward -= 1000  # Penalty for causing a deadlock
                            done = True
                            break  # Exit the loop if the condition is met
                    if not done:
                        #reward += 10  # Reward for moving to the next node
                    #print(visited_nodes, "visited_nodes")
                        visited_nodes2[agent_index] = current_pos
                        visited_nodes[agent_index] = next_pos

                        #print(visited_nodes, "visited_nodes")
                        #print(next_pos, "next_pos")
                        #print(agent_index, "agent_index")
                        #print(i)
                    #print(visited_nodes, "visited_nodes")
                    # Move agent forward
                        if len(agv_paths[agent_index]) > 1:
                            agv_paths[agent_index] = agv_paths[agent_index][1:]
                            #print(agv_paths)
                            reward += 1.0  # Reward for moving to the next node
                            #print(agv_paths)
                        #print(agv_paths[agent_index])
                        else:
                            agv_paths[agent_index] = []
                            reward += 5.0  # Reward for reaching the goal
                            #print(agv_paths)
                            if all(not path for path in agv_paths):
                    #print(agv_paths)
                                reward += 200.0  # Reward for reaching the goal
                                #print(agv_paths, "agv_paths")
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
                            reward -= 5.0  # Penalty for causing a deadlock
                        else:
                            reward += 1.0  # Reward for moving to the next node
                    if len(agv_paths[agent_index]) >= 1:
                        current_pos = agv_paths[agent_index][0]
                        next_pos = agv_paths[agent_index][0]
                        visited_nodes2[agent_index] = current_pos
                        visited_nodes[agent_index] = next_pos
                #reward -= 0.1  # Default reward if no action taken
                    #print(agv_paths[agent_index])
                #reward -= 1 * len(agv_paths[agent_index])  # Penalize longer paths

                # Record current position (if available) for visualization
                if agv_paths[agent_index]:
                    agents_paths[agent_index].append(agv_paths[agent_index][0])
                if done:
                    #print("done1")
                    break


            #reward -= step

            log_probs.append(step_log_prob)
            rewards.append(reward)
            states.append(state_matrix.clone())
            #print(len(rewards))
            #print(len(states))
            #print(states, "states")




            if done:
                #print(f"Breaking at the end of the time step loop. Step: {step}, Agent Paths: {agv_paths}")
                break
            # Update the state matrix at each time step before the next decision
            #print("Updating State Matrix")
            state_matrix = np.zeros((num_agents, 30), dtype=np.float32)
            for agent_index, path in enumerate(agv_paths):
                if path:
                    for node in path:
                        state_matrix[agent_index, node - 1] += 1
            state_matrix = torch.flatten(torch.from_numpy(state_matrix).float())


        #print(states)
        # Convert rewards to tensor (do not normalize here; update_policy will handle it)
        #rewards = torch.tensor(rewards, dtype=torch.float32)
        #print("State Matrix:", state_matrix)  # Debugging: Print state matrix
        #print(states)
        #print(rewards)
        #rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        update_policy(policy_net, value_net, policy_optimizer, value_optimizer, rewards, log_probs, states, gamma)
        #policy_scheduler.step()
        #value_scheduler.step()
        total_reward = sum(rewards)
        reward_history.append(total_reward)

        # Compute a simple discounted reward for logging
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = sum(gamma ** i * rewards[i + t] for i in range(len(rewards) - t))
            discounted_rewards.append(Gt)
        average_discounted_reward = np.mean(discounted_rewards)

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
        print(f"Episode {episode + 1}: Average Discounted Reward = {average_discounted_reward:.2f}")

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
        state_matrix = torch.flatten(torch.from_numpy(state_matrix).float())

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
            state_matrix = torch.flatten(torch.from_numpy(state_matrix).float())
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