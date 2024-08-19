import numpy as np
from grid_env import GridEnv
from new_try import QLearningAgent, train_q_learning

# Define environment and agent
grid_size = (5, 5)
num_agents = 2
obstacles = [(2, 2)]
goals = [(4, 4), (0, 0)]
action_space = 5  # 5 possible actions (0=left, 1=up, 2=right, 3=down, 4=wait)

env = GridEnv(grid_size, num_agents, obstacles, goals)
agent = QLearningAgent(grid_size, num_agents, action_space)

# Train the Q-learning agent
train_q_learning(env, agent)

# Function to extract the best policy
def extract_best_policy(agent, start_state, max_steps=100):
    policy_sequence = []
    state = start_state
    for _ in range(max_steps):
        actions = []
        for i in range(agent.num_agents):
            best_action = np.argmax(agent.q_table[state[i]][i])
            actions.append(best_action)
        policy_sequence.append(actions)
        next_state, _, done, _ = env.step(actions)
        state = tuple(next_state)
        if done:
            break
    return policy_sequence

# Get the starting state
start_state = tuple(env.reset())

# Extract the best policy
policy_sequence = extract_best_policy(agent, start_state)

# Animate the best policy
env.animate(policy_sequence)
