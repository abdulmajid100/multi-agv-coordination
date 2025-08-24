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
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


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
class Agent:
    def __init__(self, state_size, action_size):
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.value_net = ValueNetwork(state_size)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.01)
        self.gamma = 0.99  # Discount factor

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probabilities = self.policy_net(state)
        action = np.random.choice(len(probabilities), p=probabilities.detach().numpy())
        return action, torch.log(probabilities[action])

    def update_policy(self, rewards, log_probs, states):
        discounted_rewards = []
        for t in range(len(rewards)):
            G = sum(self.gamma ** i * rewards[i + t] for i in range(len(rewards) - t))
            discounted_rewards.append(G)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        states = torch.FloatTensor(np.array(states))
        values = self.value_net(states).squeeze()

        advantages = discounted_rewards - values.detach()

        policy_loss = -torch.sum(torch.stack(log_probs) * advantages)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        value_loss = F.mse_loss(values, discounted_rewards)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


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


# Train agents in the graph environment
def train_agents(num_agents, num_episodes, state_size, action_size):
    agents = [Agent(state_size, action_size) for _ in range(num_agents)]
    G = create_graph()

    agents_paths = [[] for _ in range(num_agents)]

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        # Generate random nodes for each agent
        states = [random.sample(list(G.nodes), random.randint(3, 5)) for _ in range(num_agents)]
        visited_nodes = set()
        log_probs = [[] for _ in range(num_agents)]
        rewards = [[] for _ in range(num_agents)]
        actions = [[] for _ in range(num_agents)]

        for step in range(200):  # Limit the number of steps per episode
            for agent_index, agent in enumerate(agents):
                # One-hot encode the current state
                state_vector = np.zeros(state_size)
                for node in states[agent_index]:
                    state_vector[node - 1] = 1  # Mark the selected nodes in the state vector

                # Select action: 0 for stop, 1 for move
                action, log_prob = agent.select_action(state_vector)
                actions[agent_index].append(action)

                if action == 1:  # Move
                    current_node = states[agent_index][0]  # Get the current node
                    next_node = random.choice(list(G.successors(current_node))) if list(G.successors(current_node)) else current_node

                    # Check for collision or deadlock
                    if next_node in visited_nodes:
                        reward = -10  # Penalty for collision
                    else:
                        reward = 0  # No penalty for moving to a new node
                        visited_nodes.add(next_node)

                    # Update the state to the new node
                    states[agent_index] = [next_node] + states[agent_index][1:]  # Move to the next node
                else:  # Stop
                    reward = 0  # No reward for stopping

                # Check if the agent reached the last node
                if states[agent_index][0] == states[agent_index][-1]:
                    reward += 10  # Reward for reaching the last node

                # Store log probabilities, rewards, and states
                log_probs[agent_index].append(log_prob)
                rewards[agent_index].append(reward)

        # Update policies for all agents
        for agent_index, agent in enumerate(agents):
            agent.update_policy(rewards[agent_index], log_probs[agent_index], states[agent_index])

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
state_size = 30  # Number of nodes in the graph
action_size = 30  # Number of possible actions (one for each node)

# Train the agents
agents_paths, G = train_agents(num_agents, num_episodes, state_size, action_size)

# Visualize the agents' movements
visualize_agents(agents_paths, G)