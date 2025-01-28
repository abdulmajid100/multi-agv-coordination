import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# --------------------------
# Visualization
# --------------------------
def visualize_agents(agents_paths, G):
    """
    agents_paths is a list of length num_agents,
    where each element is the list of nodes visited by that agent in each step.
    """
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots()
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', node_size=500, font_size=8)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'yellow', 'cyan']
    agent_positions = [None] * len(agents_paths)

    max_frames = max(len(path) for path in agents_paths)  # Longest path over all agents

    def update(frame):
        ax.clear()
        # Draw the underlying graph
        nx.draw(
            G, pos, ax=ax, with_labels=True,
            node_color='lightblue', node_size=500, font_size=8
        )

        # Update each agent's position at the current frame
        for i, path in enumerate(agents_paths):
            if frame < len(path):
                agent_positions[i] = path[frame]  # the node at this frame for agent i

        # Draw each agent node in a distinct color
        for i, node in enumerate(agent_positions):
            if node is not None:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=[node],
                    node_color=colors[i % len(colors)],
                    node_size=300,
                    ax=ax,
                    label=f'Agent {i + 1}'
                )
        ax.legend(loc='upper right')

    ani = animation.FuncAnimation(
        fig, update, frames=max_frames, repeat=False, interval=1000
    )
    plt.show()


# --------------------------
# Networks
# --------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --------------------------
# Agent
# --------------------------
class Agent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.policy_net = PolicyNetwork(state_size, action_size)
        self.value_net = ValueNetwork(state_size)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

    def select_action(self, state):
        """
        Returns (action, log_probability).
        """
        state_t = torch.FloatTensor(state)
        probs = self.policy_net(state_t)
        action = np.random.choice(len(probs), p=probs.detach().numpy())
        log_prob = torch.log(probs[action])
        return action, log_prob

    def update_policy(self, rewards, log_probs, states, num_agents):
        """
        Slices out each agent's portion of the full arrays using [agent_idx::num_agents].
        Then does a single pass of policy & value update for each agent in turn.
        """
        total_policy_loss = 0
        total_value_loss = 0

        # For each agent, slice out the relevant data
        for agent_idx in range(num_agents):
            # Slice out that agent's rewards
            agent_rewards = rewards[agent_idx::num_agents]

            # Compute discounted returns
            discounted_returns = []
            for t in range(len(agent_rewards)):
                G = 0
                discount = 1
                for r in agent_rewards[t:]:
                    G += r * discount
                    discount *= self.gamma
                discounted_returns.append(G)

            # Convert to float32 for PyTorch
            discounted_returns = np.array(discounted_returns, dtype=np.float32)
            discounted_returns_t = torch.from_numpy(discounted_returns)

            # Slice out states & log_probs for this agent
            agent_states = np.array(states[agent_idx::num_agents], dtype=np.float32)
            agent_states_t = torch.from_numpy(agent_states)
            agent_log_probs = torch.stack(log_probs[agent_idx::num_agents])

            # Value function for each state
            values = self.value_net(agent_states_t).squeeze()  # shape [T]

            # Advantage = discounted_returns - values
            advantages = discounted_returns_t - values.detach()

            # Check for size mismatch
            if len(agent_log_probs) != len(advantages):
                raise RuntimeError(
                    f"Size mismatch: log_probs {len(agent_log_probs)} vs advantages {len(advantages)}"
                )

            # Policy loss
            policy_loss = -torch.sum(agent_log_probs * advantages)
            total_policy_loss += policy_loss

            # Value loss
            value_loss = F.mse_loss(values, discounted_returns_t)
            total_value_loss += value_loss

        # Single backward pass for the policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        # Single backward pass for the value function
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()


# --------------------------
# Environment creation
# --------------------------
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
    G.remove_nodes_from([7, 8])  # example
    return G


def generate_random_paths(G, num_agents):
    """
    Return a list of num_agents paths, each is a list of nodes the agent will try to traverse.
    """
    paths = []
    for _ in range(num_agents):
        path = []
        current_node = random.choice(list(G.nodes))
        while len(path) < len(G.nodes):
            path.append(current_node)
            neighbors = list(G.successors(current_node))
            if not neighbors:
                break
            current_node = random.choice(neighbors)
        paths.append(path)
    return paths


# --------------------------
# Training
# --------------------------
def train_agents(num_agents=3, num_episodes=2, max_steps=20):
    G = create_graph()
    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}
    state_size = len(G.nodes)
    action_size = 5  # For example; you can adjust

    # Create agents
    agents = [Agent(state_size, action_size) for _ in range(num_agents)]

    # We will store the final paths of each agent in the last episode for visualization
    final_trajectories = None

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        # Create random paths
        paths = generate_random_paths(G, num_agents)

        # For visualization: track each agent's actual path over time
        agent_trajectories = [[] for _ in range(num_agents)]

        # These lists will hold data for *all* agents angled
        log_probs = []
        rewards = []
        states = []

        # Run the environment for up to max_steps
        for step in range(max_steps):
            # Build the global state
            global_state = np.zeros(state_size, dtype=np.float32)
            for i, path in enumerate(paths):
                if len(path) > 0:
                    global_state[node_to_index[path[0]]] = i + 1

            # Each agent picks an action
            step_actions = []
            for i, agent in enumerate(agents):
                action, lp = agent.select_action(global_state)
                step_actions.append(action)
                log_probs.append(lp)
                states.append(global_state.copy())  # replicate the same global state for each agent

            # Environment transition (very simple)
            for i, action in enumerate(step_actions):
                # If action != 0 => drop the current node
                if action != 0 and len(paths[i]) > 1:
                    paths[i] = paths[i][1:]
                # Record the agent's node at this step for visualization
                if len(paths[i]) > 0:
                    agent_trajectories[i].append(paths[i][0])
                else:
                    # If path is empty, just record None or the last node
                    agent_trajectories[i].append(None)

            # Common reward = sum(10 if the path is not empty, else -100)
            step_reward = sum(10 if len(path) > 0 else -100 for path in paths)
            # Append it once per agent
            for _ in range(num_agents):
                rewards.append(step_reward)

        # After finishing this episode, we do policy updates for all agents
        for agent in agents:
            agent.update_policy(rewards, log_probs, states, num_agents)

        # If this is the last episode, store agent_trajectories for visualization
        if episode == num_episodes - 1:
            final_trajectories = agent_trajectories

    # Visualize paths from the last episode
    # Remove any trailing None so the animation doesn't show blank steps
    cleaned_trajectories = []
    for traj in final_trajectories:
        # e.g. remove trailing None if some agent finished early
        # or just keep them so the frame counts line up. Here we strip them:
        while len(traj) > 0 and traj[-1] is None:
            traj.pop()
        cleaned_trajectories.append(traj)

    visualize_agents(cleaned_trajectories, G)


if __name__ == "__main__":
    train_agents(num_agents=3, num_episodes=2, max_steps=20)