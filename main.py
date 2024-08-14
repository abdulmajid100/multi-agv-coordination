import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from collections import namedtuple, deque


# Define the grid environment
class GridEnv(gym.Env):
    def __init__(self):
        super(GridEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)  # Example: 4D state space
        self.action_space = gym.spaces.Discrete(4)  # Example: 4 possible actions (e.g., up, down, left, right)
        self.state = np.zeros(4)

    def reset(self):
        self.state = np.zeros(4)
        return self.state

    def step(self, actions):
        next_state = self.state  # Example: No state transition
        rewards = np.random.rand(2)  # Example: Random rewards for 2 agents
        done = False
        return next_state, rewards, done, {}


# Define a simple MLP for Actor and Critic
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            in_dim = size
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Define Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(Actor, self).__init__()
        self.mlp = MLP(state_dim, action_dim, hidden_sizes)

    def forward(self, state):
        return torch.softmax(self.mlp(state), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(Critic, self).__init__()
        self.mlp = MLP(state_dim + action_dim, 1, hidden_sizes)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.mlp(x)


# Define Attention Critic for MAAC
class AttentionCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, hidden_sizes, num_heads):
        super(AttentionCritic, self).__init__()
        self.num_agents = num_agents
        self.critics = nn.ModuleList([Critic(state_dim, action_dim, hidden_sizes) for _ in range(num_agents)])
        self.attention_heads = nn.ModuleList(
            [nn.Linear(state_dim + action_dim, state_dim + action_dim) for _ in range(num_heads)])

    def forward(self, states, actions):
        # Compute attention scores
        attention_outputs = []
        for i in range(self.num_agents):
            other_agents = [states[j] for j in range(self.num_agents) if j != i]
            for head in self.attention_heads:
                attention_output = head(torch.cat(other_agents, dim=0))
                attention_outputs.append(attention_output)
            q_value = self.critics[i](states[i], actions[i])
            return q_value, attention_outputs


# Define MAAC Algorithm
class MAAC:
    def __init__(self, state_dim, action_dim, num_agents, hidden_sizes, num_heads, lr):
        self.num_agents = num_agents
        self.actors = nn.ModuleList([Actor(state_dim, action_dim, hidden_sizes) for _ in range(num_agents)])
        self.attention_critic = AttentionCritic(state_dim, action_dim, num_agents, hidden_sizes, num_heads)
        self.optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.attention_critic.parameters(), lr=lr)

    def update(self, states, actions, rewards, next_states, done):
        # Update critics
        q_values, _ = self.attention_critic(states, actions)
        target_q_values = rewards + (1 - done) * q_values
        critic_loss = torch.mean((q_values - target_q_values) ** 2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actors
        for i, actor in enumerate(self.actors):
            actor_loss = -torch.mean(self.attention_critic(states, actions)[0])
            self.optimizers[i].zero_grad()
            actor_loss.backward()
            self.optimizers[i].step()


# Training function
def train_maac(env, num_agents, num_episodes, hidden_sizes, num_heads, lr):
    agent_maac = MAAC(env.observation_space.shape[0], env.action_space.n, num_agents, hidden_sizes, num_heads, lr)

    for episode in range(num_episodes):
        states = [env.reset() for _ in range(num_agents)]
        done = False

        while not done:
            actions = [agent_maac.actors[i](torch.tensor(state, dtype=torch.float32).unsqueeze(0)).detach().numpy() for
                       i, state in enumerate(states)]
            next_states, rewards, done, _ = env.step(actions)
            agent_maac.update(states, actions, rewards, next_states, done)
            states = next_states

        print(f"Episode {episode + 1}/{num_episodes} completed")


# Create environment instance
env = GridEnv()

# Training
train_maac(env, num_agents=2, num_episodes=1000, hidden_sizes=[64, 64], num_heads=4, lr=1e-3)
