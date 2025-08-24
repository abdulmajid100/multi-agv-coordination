# RL-Based Traffic Management Guide

This guide explains how to use the reinforcement learning (RL) based traffic management system implemented in `rl_based_traffic_management.py`. The system has been enhanced to support multiple scenarios and custom scenario creation.

## Overview

The RL-based traffic management system uses Q-learning to train agents to navigate from their initial positions to goal positions while avoiding obstacles and other agents. The system now supports:

1. Multiple predefined scenarios with different grid sizes, number of agents, obstacles, and goals
2. Custom scenario creation with user-defined parameters
3. Memory-efficient Q-learning implementation using dictionary-based Q-tables
4. Visualization of agent paths using matplotlib animations

## Running Predefined Scenarios

To run a predefined scenario:

```bash
python rl_based_traffic_management.py --scenario 0
```

Available scenarios:
- 0: Original scenario (10x10 grid, 3 agents)
- 1: Simple scenario (5x5 grid, 2 agents)
- 2: Complex scenario (8x8 grid, 4 agents)

To run all predefined scenarios:

```bash
python rl_based_traffic_management.py --all
```

## Creating Custom Scenarios

To create and run a custom scenario:

```bash
python rl_based_traffic_management.py --custom --grid-size 7 7 --num-agents 3 --random-obstacles 10
```

Parameters:
- `--grid-size`: Size of the grid as width height (default: 5 5)
- `--num-agents`: Number of agents (default: 2)
- `--random-obstacles`: Number of random obstacles to generate (default: 0)

## General Options

- `--episodes`: Number of training episodes (default: 1000)
- `--no-animation`: Disable animation of results

Example:
```bash
python rl_based_traffic_management.py --scenario 1 --episodes 500 --no-animation
```

## How It Works

1. The system creates a grid environment with agents, obstacles, and goals.
2. The Q-learning algorithm trains agents to navigate from their initial positions to their goals.
3. During training, agents learn to avoid obstacles and other agents while minimizing the number of steps.
4. After training, the system tests the learned policy and visualizes the results.

## Implementation Details

- The environment is implemented in `grid_env_c.py`
- The RL algorithm is implemented in `rl_based_traffic_management.py`
- The system uses a dictionary-based Q-table to efficiently store state-action values
- The state representation is based on the positions of all agents
- Actions include moving left, up, right, down, or waiting

## Adding New Scenarios

You can add new predefined scenarios by modifying the `define_scenario` function in `rl_based_traffic_management.py`. Add a new entry to the `scenarios` dictionary with the following structure:

```python
3: {
    "grid_size": (8, 8),
    "num_agents": 3,
    "obstacles": [(1, 1), (2, 2), ...],
    "goals": [(7, 7), (0, 7), (7, 0)],
    "initial_positions": [(3, 0), (0, 3), (7, 3)]
}
```

Also update the `scenarios` dictionary in the `main` function to include a description for your new scenario:

```python
scenarios = {
    0: "Original scenario",
    1: "Simple scenario",
    2: "Complex scenario",
    3: "Your new scenario description"
}
```

## Programmatic API

You can also use the system programmatically by importing the functions and classes:

```python
from rl_based_traffic_management import q_learning, create_custom_scenario
from grid_env_c import GridEnv

# Create a custom scenario
grid_size = (6, 6)
num_agents = 2
obstacles = [(1, 1), (2, 2), (3, 3)]
goals = [(5, 5), (0, 5)]
initial_positions = [(0, 0), (5, 0)]

# Initialize environment
env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)

# Train agents
q_tables = q_learning(env, num_episodes=500)

# Use the trained policy
# ...
```