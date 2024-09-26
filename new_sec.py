import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Grid Environment
class GridEnv:
    def __init__(self, grid_size, num_agents, obstacles, goals):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obstacles = obstacles
        self.goals = goals
        self.reset()

    def reset(self):
        self.agent_positions = np.random.randint(0, self.grid_size[0], (self.num_agents, 2))
        self.done = False
        return self.get_state()

    def step(self, actions):
        for i in range(self.num_agents):
            if not self.done:
                action_direction = self.action_to_direction(actions[i])
                new_position = self.agent_positions[i] + action_direction
                if self.is_valid_position(new_position):
                    self.agent_positions[i] = new_position
                if self.is_goal_reached(i):
                    self.done = True

        return self.get_state(), self.get_reward(), self.done

    def get_state(self):
        return self.agent_positions.flatten()

    def action_to_direction(self, action):
        if action == 0:   # Left
            return np.array([-1, 0])
        elif action == 1: # Up
            return np.array([0, -1])
        elif action == 2: # Right
            return np.array([1, 0])
        elif action == 3: # Down
            return np.array([0, 1])
        else:            # Wait
            return np.array([0, 0])

    def is_valid_position(self, position):
        x, y = position
        return (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and
                tuple(position) not in self.obstacles)

    def is_goal_reached(self, agent_index):
        return tuple(self.agent_positions[agent_index]) == self.goals[agent_index]

    def get_reward(self):
        return 1 if self.done else -0.1

# Define the SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=0.001):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, state_dim)
        action = self.actor(state_tensor)  # Output will be (1, action_dim)
        return torch.argmax(action, dim=-1).item()  # Return the index of the highest action value

    def update(self, replay_buffer, batch_size):
        # Update logic for SAC can be implemented here.
        pass

# Training Function
def train_sac(env, sac_agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = sac_agent.select_action(state)
            actions = [action for _ in range(env.num_agents)]
            next_state, reward, done = env.step(actions)
            total_reward += reward
            state = next_state

        # Log the progress every 10 episodes
        if (episode + 1) % 1 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Agent Positions: {env.agent_positions}")

# Main Function
def main():
    grid_size = (5, 5)  # 5x5 grid
    num_agents = 1
    obstacles = [(1, 1), (1, 2), (2, 1)]  # Example obstacles
    goals = [(0, 4), (4, 0)]  # Define goals for each agent
    env = GridEnv(grid_size, num_agents, obstacles, goals)

    state_dim = num_agents * 2  # Each agent has x and y coordinates
    action_dim = 5  # Assuming actions are: 0=left, 1=up, 2=right, 3=down, 4=wait
    sac_agent = SACAgent(state_dim, action_dim)

    train_sac(env, sac_agent, num_episodes=1000)

if __name__ == "__main__":
    main()
