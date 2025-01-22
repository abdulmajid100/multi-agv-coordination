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


# Define the Agent
class Agent:
    def __init__(self, state_size, action_size):
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.gamma = 0.99  # Discount factor

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probabilities = self.policy_net(state)
        probabilities = probabilities / probabilities.sum()  # Normalize probabilities
        action = np.random.choice(len(probabilities), p=probabilities.detach().numpy())
        return action, torch.log(probabilities[action])

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        for t in range(len(rewards)):
            G = sum(self.gamma ** i * rewards[i + t] for i in range(len(rewards) - t))
            discounted_rewards.append(G)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        # Normalize the rewards only if there are enough data points
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate loss
        loss = -torch.sum(torch.stack(log_probs) * discounted_rewards)

        # Update the policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Initialize the directed graph
def create_graph():
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
    return G


# Function to animate the agents' movements



# Example of running the agents in the graph environment
def train_agents(num_agents, num_episodes, state_size, action_size):
    agents = [Agent(state_size, action_size) for _ in range(num_agents)]
    G = create_graph()


    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")  # Output progress
        agents_paths = [[] for _ in range(num_agents)]
        for agent_index, agent in enumerate(agents):
            state = random.choice(list(G.nodes))  # Start from a random node
            log_probs = []
            rewards = []
            visited_nodes = set()  # Track visited nodes
            done = False
            steps = 0
            max_steps = 10  # Limit the number of steps per episode

            while not done and steps < max_steps:
                steps += 1
                # Get the current state representation (e.g., current node)
                state_vector = np.zeros(state_size)
                state_vector[state - 1] = 1  # One-hot encoding of the current state

                action, log_prob = agent.select_action(state_vector)
                next_state = random.choice(list(G.successors(state))) if list(
                    G.successors(state)) else state  # Move to a successor or stay

                # Reward function
                reward = -1 if next_state == state else 1  # Penalize staying in the same place
                if next_state in visited_nodes:
                    reward -= 0.5  # Penalize revisiting nodes
                else:
                    reward += 1  # Reward exploring new nodes
                visited_nodes.add(next_state)

                log_probs.append(log_prob)
                rewards.append(reward)
                agents_paths[agent_index].append(next_state)  # Store the path
                state = next_state

                # Check if the goal is reached
                #done = (next_state == 29)  # Example condition to end the episode (reaching node 29)

            if steps >= max_steps:
                print(f"Agent {agent_index + 1} terminated episode due to step limit.")

            agent.update_policy(rewards, log_probs)
            print(
                f"  Agent {agent_index + 1} finished episode with path: {agents_paths[agent_index]}")  # Output agent progress

    return agents_paths, G


# Parameters
num_agents = 3
num_episodes = 10
state_size = 30  # Number of nodes in the graph
action_size = 30  # Number of possible actions (one for each node)

# Train the agents and get their paths
agents_paths, G = train_agents(num_agents, num_episodes, state_size, action_size)
# Animate the agents' movements
print(agents_paths)
# Define positions for nodes in a circular layout
pos = nx.kamada_kawai_layout(G)

fig, ax = plt.subplots()
nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10)

# Define unique colors for each agent
colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'yellow', 'cyan']  # Add more colors if needed
num_agents = len(agents_paths)
agent_positions = [None] * num_agents


def update(frame):
    # Clear the previous frame
    ax.clear()
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=10)

    # Update the position of each agent
    for i, path in enumerate(agents_paths):
        if frame < len(path):  # Ensure the frame index is within the path length
            node = path[frame]
            agent_positions[i] = node


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

    # Add a legend to show agent labels
    ax.legend()


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=max(len(path) for path in agents_paths), blit=False, repeat=False, interval=1000)
plt.show()

