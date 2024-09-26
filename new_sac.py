import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from grid_env_c import GridEnv

# Define the SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = self._build_actor(state_dim, action_dim)
        self.critic1 = self._build_critic(state_dim, action_dim)
        self.critic2 = self._build_critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.001)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.001)
        self.action_dim = action_dim

    def _build_actor(self, state_dim, action_dim):
        model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # To normalize actions
        )
        return model

    def _build_critic(self, state_dim, action_dim):
        model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        return model

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action = self.actor(state_tensor)
        action_index = (action + 1) * (self.action_dim / 2)  # Scale action to [0, action_dim]
        return int(action_index.item())

    def update(self, state, action, reward, next_state, done):
        # Placeholder for SAC update logic
        pass  # Implement SAC update logic here


def train_sac(env, num_episodes):
    state_dim = env.grid_size[0] * env.grid_size[1] * env.num_agents  # Modify if needed
    action_dim = 5  # Number of actions (left, up, right, down, wait)
    sac_agent = SACAgent(state_dim, action_dim)

    for episode in range(num_episodes):
        state = env.reset().flatten()  # Flatten the state
        done = False
        total_reward = 0

        while not done:
            # Ensure the state is of the right shape
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension

            action = sac_agent.select_action(state_tensor)
            next_state, reward, done, _ = env.step([action])
            next_state = next_state.flatten()  # Flatten next state

            total_reward += reward

            # Update the agent
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)  # Add batch dimension
            sac_agent.update(state_tensor, action, reward, next_state_tensor, done)

            state = next_state  # Update state for the next step

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


def main():
    grid_size = (5, 5)
    num_agents = 1
    obstacles = [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3)]
    goals = [(4, 0)]
    initial_positions = [(0, 4)]

    env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)
    train_sac(env, num_episodes=1000)

if __name__ == "__main__":
    main()
