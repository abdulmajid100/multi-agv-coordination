import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim)
        )

    def forward(self, x):
        return self.model(x)

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)  # Output a single value for Q
        )

    def forward(self, state, action):
        # Ensure state and action have the same batch size
        batch_size = state.size(0)
        action = action.view(batch_size, -1)  # Ensure action is 2D
        x = torch.cat([state, action], dim=-1)  # Concatenate along the last dimension
        return self.model(x)

# Define the MAAC agent
class MAAC:
    def __init__(self, state_dim, action_dim, num_agents, hidden_sizes, lr):
        self.num_agents = num_agents
        self.actors = nn.ModuleList([Actor(state_dim, action_dim, hidden_sizes) for _ in range(num_agents)])
        self.critics = nn.ModuleList([Critic(state_dim, action_dim, hidden_sizes) for _ in range(num_agents)])
        self.optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr) for critic in self.critics]

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        # Ensure actions tensor is 2D
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        # Update critics
        for i in range(self.num_agents):
            self.critic_optimizers[i].zero_grad()
            # Ensure actions have correct dimensions for critic
            action_i = actions[:, i] if actions.dim() == 2 else actions.unsqueeze(1)
            predicted_q = self.critics[i](states, action_i)
            target_q = rewards.unsqueeze(1)  # Ensure target_q has correct dimensions
            critic_loss = nn.functional.mse_loss(predicted_q, target_q)
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # Update actors
        for i in range(self.num_agents):
            self.optimizers[i].zero_grad()
            action_probs = self.actors[i](states)
            q_values = self.critics[i](states, action_probs)
            actor_loss = -q_values.mean()
            actor_loss.backward()
            self.optimizers[i].step()

# Dummy environment class
class DummyEnv:
    def reset(self):
        return np.random.rand(4)

    def step(self, actions):
        next_states = [np.random.rand(4) for _ in actions]
        rewards = np.random.rand(len(actions))
        done = np.random.choice([True, False])
        return next_states, rewards, done, {}

# Training function for MAAC
# Training function for MAAC
def train_maac(env, num_agents, num_episodes, hidden_sizes, lr):
    state_dim = env.reset().shape[0]
    action_dim = 4  # Assuming a fixed action dimension
    agent_maac = MAAC(state_dim, action_dim, num_agents, hidden_sizes, lr)

    for episode in range(num_episodes):
        states = np.array([env.reset() for _ in range(num_agents)])  # Convert to numpy array
        states = torch.tensor(states, dtype=torch.float32)
        done = False

        while not done:
            actions = [agent_maac.actors[i](states[i].unsqueeze(0)).detach().numpy() for i in range(num_agents)]
            next_states, rewards, done, _ = env.step(actions)

            # Prepare tensors for the update
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)  # Convert to numpy array before tensor
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
            dones = torch.tensor([done] * num_agents, dtype=torch.float32)

            # Update the MAAC agent
            agent_maac.update(states, actions, rewards, next_states, dones)

            # Move to the next state
            states = next_states

        print(f"Episode {episode + 1}/{num_episodes} completed")


# Main function
if __name__ == "__main__":
    env = DummyEnv()
    train_maac(env, num_agents=2, num_episodes=1000, hidden_sizes=[64, 64], lr=1e-3)
