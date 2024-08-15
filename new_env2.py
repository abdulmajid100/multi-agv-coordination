# main.py
from grid_env import GridEnv
import numpy as np

# Define grid parameters
grid_size = (5, 5)
num_agents = 2
obstacles = [(2, 1), (1, 3)]
goals = [(4, 4), (0, 0)]
initial_positions = [(4, 1), (2, 3)]

# Create environment
env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)

# Generate a sequence of random actions for demonstration
actions_sequence = []
state = env.reset()
done = False

while not done:
    actions = [np.random.choice(5) for _ in range(num_agents)]  # Including wait action
    next_state, rewards, done, _ = env.step(actions)

    # Store actions for animation
    actions_sequence.append(actions)

    # Print state, actions, and rewards for debugging
    print(f"State: {state}")
    print(f"Actions: {actions}")
    print(f"Next State: {next_state}")
    print(f"Rewards: {rewards}")
    print(f"Done: {done}")

    # Update the state
    state = next_state

# Animate the environment
env.animate(actions_sequence)
