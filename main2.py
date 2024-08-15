from grid_env import GridEnv
from q_learning_agent import QLearningAgent
import numpy as np

# Define grid parameters
# Define grid parameters
grid_size = (5, 5)
num_agents = 2
obstacles = [(2, 1), (1, 3)]
goals = [(4, 4), (0, 0)]
initial_positions = [(4, 1), (2, 3)]

# Create environment
env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)

# Define Q-learning parameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 1000

# Initialize Q-learning agents
state_space = (grid_size[0], grid_size[1])  # Example state space size
action_space = 5  # 5 actions (left, up, right, down, wait)
agents = [QLearningAgent(state_space, action_space) for _ in range(num_agents)]

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        actions = [agent.choose_action(tuple(agent_pos)) for agent, agent_pos in zip(agents, state)]
        next_state, rewards, done, _ = env.step(actions)

        for i, agent in enumerate(agents):
            agent.update_q_table(tuple(state[i]), actions[i], rewards[i], tuple(next_state[i]))

        state = next_state
        total_reward += sum(rewards)

    # Decay epsilon after each episode
    for agent in agents:
        agent.decay_epsilon()

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agents[0].epsilon}")

print("Training completed.")
