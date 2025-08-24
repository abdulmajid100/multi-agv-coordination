# main.py
from grid_env import GridEnv
from q_learning import QLearningAgent
from train_agents import train_agents
from policy_utils import get_actions_from_policy
from animation_utils import animate_results

# Define grid size, obstacles, goals, and initial positions
grid_size = (5, 5)
num_agents = 2
obstacles = [(2, 2), (1, 3)]
goals = [(4, 4), (0, 0)]
initial_positions = [(0, 1), (1, 0)]

# Create the environment
env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)

# Train the agents
trained_agents, _ = train_agents(env)

# Get actions based on the learned policy
actions_sequence = get_actions_from_policy(env, trained_agents)

# Animate the results
animate_results(env, actions_sequence)
