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
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, num_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.ln1 = nn.LayerNorm(256, eps=1e-8)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128, eps=1e-8)
        self.fc3 = nn.Linear(128, num_actions)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# PPO-specific update function
def update_ppo(policy_net, value_net, policy_optimizer, value_optimizer, states, actions, log_probs_old, rewards, advantages, clip_epsilon, gamma, epochs, batch_size):
    # Convert everything to tensors
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long)
    log_probs_old = torch.stack(log_probs_old)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Perform multiple epochs of updates
    for _ in range(epochs):
        # Shuffle the data
        indices = np.arange(len(states))
        np.random.shuffle(indices)

        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # Mini-batch data
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_log_probs_old = log_probs_old[batch_indices]
            batch_advantages = advantages[batch_indices]

            # Compute new log probabilities
            probabilities = policy_net(batch_states)
            dist = torch.distributions.Categorical(probabilities)
            log_probs_new = dist.log_prob(batch_actions)

            # Compute the ratio (new probability / old probability)
            ratios = torch.exp(log_probs_new - batch_log_probs_old)

            # Compute the clipped surrogate objective
            surrogate1 = ratios * batch_advantages
            surrogate2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Compute value loss
            values = value_net(batch_states).squeeze()
            value_loss = F.mse_loss(values, rewards[batch_indices])

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


# Train agents using PPO
def train_agents_ppo(num_agents, num_episodes, fixed_paths):
    G = create_graph()
    state_size = 30 * num_agents
    action_size = 2 ** num_agents - 1
    policy_net = PolicyNetwork(state_size, action_size)
    value_net = ValueNetwork(state_size)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.005)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.005)

    gamma = 0.9
    clip_epsilon = 0.2
    ppo_epochs = 10
    batch_size = 32

    reward_history = []

    for episode in range(num_episodes):
        agv_paths = copy.deepcopy(fixed_paths)
        visited_nodes = [None] * num_agents
        visited_nodes2 = [None] * num_agents

        log_probs = []
        rewards = []
        states = []
        actions = []

        state_matrix = np.zeros((num_agents, 30), dtype=np.float32)
        for agent_index, path in enumerate(agv_paths):
            if path:
                for node in path:
                    state_matrix[agent_index, node - 1] += 1
        state_matrix = torch.flatten(torch.from_numpy(state_matrix).float())

        done = False
        reward = 0
        for step in range(100):
            # Select actions
            probabilities = policy_net(state_matrix.unsqueeze(0)).squeeze(0)
            dist = torch.distributions.Categorical(probabilities)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Process each agent's decision
            action_vector = [int(x) for x in bin(action.item())[2:].zfill(num_agents)]
            for agent_index, action in enumerate(reversed(action_vector)):
                if action == 1 and agv_paths[agent_index]:
                    current_pos = agv_paths[agent_index][0]
                    next_pos = agv_paths[agent_index][1] if len(agv_paths[agent_index]) > 1 else current_pos
                    if next_pos in visited_nodes or next_pos in visited_nodes2:
                        reward -= 10000
                        done = True
                        break
                    visited_nodes2[agent_index] = current_pos
                    visited_nodes[agent_index] = next_pos
                    if len(agv_paths[agent_index]) > 1:
                        agv_paths[agent_index] = agv_paths[agent_index][1:]
                        reward += 10
                    else:
                        agv_paths[agent_index] = []
                        reward += 100
                        if all(not path for path in agv_paths):
                            reward += 1000
                            done = True
                            break

            log_probs.append(log_prob)
            rewards.append(reward)
            states.append(state_matrix.clone())
            actions.append(action.item())

            if done:
                break

            state_matrix = np.zeros((num_agents, 30), dtype=np.float32)
            for agent_index, path in enumerate(agv_paths):
                if path:
                    state_matrix[agent_index, path[0] - 1] = 1.0
            state_matrix = torch.flatten(torch.from_numpy(state_matrix).float())

        # Compute discounted rewards and advantages
        discounted_rewards = []
        Gt = 0
        for reward in reversed(rewards):
            Gt = reward + gamma * Gt
            discounted_rewards.insert(0, Gt)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        values = torch.stack([value_net(state.float()) for state in states]).squeeze()
        advantages = discounted_rewards - values.detach()

        # Update policy and value networks using PPO
        update_ppo(policy_net, value_net, policy_optimizer, value_optimizer, states, actions, log_probs, discounted_rewards, advantages, clip_epsilon, gamma, ppo_epochs, batch_size)

        total_reward = sum(rewards)
        reward_history.append(total_reward)

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Time')
    plt.show()

    return policy_net, G