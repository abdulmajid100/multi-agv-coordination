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
# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, num_agents):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, num_agents)  # Output 2 probabilities per agent (wait or move)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)  # Reshape and apply softmax for each agent

# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Define the Agent


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
        (28, 18), (18, 19), (19, 18), (19, 29), (29, 19)
    ]
    G.add_edges_from(edges)
    G.remove_nodes_from([7, 8])  # Remove specific nodes
    return G

# Define fixed paths for each AGV
fixed_paths = [
    [1, 4, 11, 12, 22],  # Path for AGV 1
    [2, 4, 11, 12, 13, 14, 15, 25],  # Path for AGV 2
    [3, 4, 11, 12, 13, 14, 15, 16, 26]  # Path for AGV 3
]

# Action selection function
# Action selection function
def select_actions(policy_net, state_matrix):
    # Check if state_matrix is a NumPy array or a PyTorch tensor
    if isinstance(state_matrix, np.ndarray):
        # If it's a NumPy array, convert it to float32 and then to a PyTorch tensor
        state = torch.FloatTensor(state_matrix.astype(np.float32)).unsqueeze(0)  # Add batch dimension
    elif isinstance(state_matrix, torch.Tensor):
        # If it's already a PyTorch tensor, ensure it's float32
        state = state_matrix.float().unsqueeze(0)  # Add batch dimension
    else:
        raise TypeError("state_matrix must be either a NumPy array or a PyTorch tensor")

    # Pass the state to the policy network
    num_agents = 3
    #print(num_agents)
    num_actions = 2 ** num_agents - 1
    action_vectors = [list(map(int, bin(i)[2:].zfill(num_agents))) for i in range(1, num_actions + 1)]
    # Note: [::-1] reverses the action vector so that rightmost corresponds to Agent 1
    #print(action_vectors)
    #state = torch.FloatTensor(state_matrix)
    probabilities = policy_net(state).squeeze(0)  # Ensure probabilities is 1-dimensional
    # print(policy_net(state))
    #print(probabilities)
    action_index = np.random.choice(len(probabilities), p=probabilities.detach().numpy())
    action_vector = action_vectors[action_index % len(action_vectors)]

    log_prob = torch.log(probabilities[action_index])

    return action_vector, log_prob




# Policy update function
def update_policy(policy_net, value_net, policy_optimizer, value_optimizer, rewards, log_probs, states, gamma=0.99):
    # Compute discounted rewards
    discounted_rewards = []
    for t in range(len(rewards)):
        G = sum(gamma ** i * rewards[i + t] for i in range(len(rewards) - t))
        discounted_rewards.append(G)
    discounted_rewards = torch.FloatTensor(discounted_rewards)

    # Compute value estimates for each state
    states = torch.FloatTensor(np.array(states))
    values = value_net(states).squeeze()

    # Compute the advantage
    advantages = discounted_rewards - values.detach()

    # Update the policy network
    policy_loss = -torch.sum(torch.stack(log_probs) * advantages)
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Update the value network
    value_loss = F.mse_loss(values, discounted_rewards)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

# Train agents in the graph environment
def train_agents(num_agents, num_episodes, fixed_paths):
    G = create_graph()
    state_size = 90
    #print(state_size)
    action_size = 7
    policy_net = PolicyNetwork(state_size, action_size)
    value_net = ValueNetwork(state_size)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.01)
    gamma = 0.99  # Discount factor

    agents_paths = [[] for _ in range(num_agents)]  # Initialize agents_paths

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        # Extract sub-paths starting from a random node in the path
        agv_paths = []
        for path in fixed_paths:
            start_index = random.randint(0, len(path) - 1)
            agv_paths.append(path[start_index:])
        #print(agv_paths)
        visited_nodes = set()
        log_probs = []
        rewards = []
        states = []

        # Initialize the state matrix
        state_matrix = np.zeros((num_agents, 30))
        for agent_index, path in enumerate(agv_paths):
            if path:
                for node in path:
                    state_matrix[agent_index, node - 1] += 1

        tensor = torch.from_numpy(state_matrix)
        state_matrix = torch.flatten(tensor)
        #print(state_matrix)
        for step in range(200):  # Limit the number of steps per episode
            #print(state_matrix)
            action_vector, step_log_prob = select_actions(policy_net, state_matrix)
            #print(action_vector)
            for agent_index, action in enumerate(action_vector):  # Action is now correctly mapped
                if action == 1 and agv_paths[agent_index]:  # Move
                    current_pos = agv_paths[agent_index][0]
                    next_pos = agv_paths[agent_index][1] if len(agv_paths[agent_index]) > 1 else current_pos

                    # Reward function
                    reward = 10 if next_pos not in visited_nodes else -100
                    visited_nodes.add(next_pos)

                    # Update state and path
                    agv_paths[agent_index] = agv_paths[agent_index][1:] if len(agv_paths[agent_index]) > 1 else []

                else:  # Wait
                    current_pos = agv_paths[agent_index][0] if agv_paths[agent_index] else None
                    reward = 0

                # Store log probabilities, rewards, and states
                log_probs.append(step_log_prob)
                rewards.append(reward)
                states.append(state_matrix)

                # Append the current position to the agent's path
                if current_pos is not None:
                    agents_paths[agent_index].append(current_pos)

        # Update policies
        update_policy(policy_net, value_net, policy_optimizer, value_optimizer, rewards, log_probs, states, gamma)

    return agents_paths, G

# Visualize the agents' movements
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
        print(agents_paths)
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

# Parameters
num_agents = 3
num_episodes = 100

# Train the agents
agents_paths, G = train_agents(num_agents, num_episodes, fixed_paths)

# Visualize the agents' movements
visualize_agents(agents_paths, G)