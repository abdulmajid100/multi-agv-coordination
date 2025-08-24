from grid_env import GridEnv

# Example setup
grid_size = (5, 5)
num_agents = 2
obstacles = [(1, 1), (2, 2)]
goals = [(4, 4), (0, 0)]
initial_positions = [(2, 0), (2, 3)]

# Initialize the environment
env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)

# Example actions sequence
actions_sequence = [
    [2, 2],  # Move agents according to actions
    [1, 1],
    [0, 0],
    [3, 3],

]

# Run animation
env.animate(actions_sequence)

