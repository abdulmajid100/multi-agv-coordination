import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from env_sac import GridEnv

# SAC hyperparameters
alpha = 0.2  # Entropy coefficient
gamma = 0.99  # Discount factor
tau = 0.005  # Soft update of target networks
lr = 3e-4  # Learning rate
num_episodes = 1
batch_size = 64
replay_buffer_size = 1000000
target_update_interval = 1


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# Neural network for the actor (policy network)
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)  # Softmax for discrete action selection
        return x


# Neural network for the critic (Q-function)
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims + n_actions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = torch.cat([state, action], dim=1)
        q = torch.relu(self.fc1(q))
        q = torch.relu(self.fc2(q))
        q = self.fc3(q)
        return q


# Soft Actor-Critic class
class SACAgent:
    def __init__(self, input_dims, n_actions):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(replay_buffer_size, [input_dims], n_actions)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = ActorNetwork(n_actions, input_dims)
        self.critic_1 = CriticNetwork(input_dims, n_actions)
        self.critic_2 = CriticNetwork(input_dims, n_actions)
        self.target_critic_1 = CriticNetwork(input_dims, n_actions)
        self.target_critic_2 = CriticNetwork(input_dims, n_actions)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

    def choose_action(self, state):
        # Flatten the state if it's a grid
        state = state.flatten()  # Flatten the state into a 1D vector if it's a 2D grid
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        probabilities = self.actor(state)
        action_probs = torch.softmax(probabilities, dim=0)
        action = torch.argmax(action_probs).item()
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).view(-1, 1)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).view(-1, 1)

        with torch.no_grad():
            next_actions = self.actor(states_)
            q1_ = self.target_critic_1(states_, next_actions)
            q2_ = self.target_critic_2(states_, next_actions)
            q_ = torch.min(q1_, q2_)
            target = rewards + self.gamma * q_ * dones

        # Critic update
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_1_loss = torch.nn.functional.mse_loss(q1, target)
        critic_2_loss = torch.nn.functional.mse_loss(q2, target)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Actor update
        new_actions = self.actor(states)
        new_q1 = self.critic_1(states, new_actions)
        actor_loss = -new_q1.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def main():
    grid_size = (5, 5)
    num_agents = 1
    obstacles = [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3), (4, 1), (4, 2),
                 (4, 3)]
    goals = [(4, 0)]
    initial_positions = [(0, 4)]

    env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)

    agent = SACAgent(input_dims=grid_size[0] * grid_size[1], n_actions=5)

    total_rewards = []  # To track total rewards for each episode

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step([action])
            agent.store_transition(state, action, reward[0], next_state, done)
            agent.learn()
            state = next_state
            episode_reward += reward[0]

        total_rewards.append(episode_reward)  # Append the episode reward

        # Print the total reward for each episode
        avg_reward = np.mean(total_rewards[-100:])  # Calculate average reward over the last 100 episodes
        print(f"Episode {episode}, Total Reward: {episode_reward}, Average Reward (last 100): {avg_reward:.2f}")

    print("Training completed.")

    # Testing
    state = env.reset()

    done = False
    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step([action])
        print(action)
        env.animate([[action]])  # You can modify this to visualize the steps

if __name__ == "__main__":
    main()
