import numpy as np
import random
from archive.grid_env_c import GridEnv
import itertools

# Q-learning parameters
alpha = 0.05  # Learning rate
gamma = 0.99  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.2  # Minimum exploration rate
epsilon_decay = 0.997  # Decay rate for epsilon
num_episodes = 1000  # Number of episodes
alpha_start = 1
alpha_decay = 0.01

# Define state-action space
def state_to_index(state, grid_size):
    num_agents = len(state)
    flat_state = [coord for agent in state for coord in agent]
    index = 0
    factor = (grid_size[0] * grid_size[1]) ** (num_agents - 1)

    for i in range(num_agents):
        x, y = flat_state[2 * i], flat_state[2 * i + 1]
        index += (x * grid_size[1] + y) * factor
        factor //= (grid_size[0] * grid_size[1])

    return index



def action_to_index(action):
    return action


def q_learning(env, num_episodes):
    """
    Q-learning algorithm for multi-agent pathfinding.

    Args:
        env: The environment to learn in
        num_episodes: Number of episodes to train for

    Returns:
        list: Q-tables for each agent
    """
    # Use a dictionary to store Q-values to save memory for large state spaces
    q_tables = [{} for _ in range(env.num_agents)]
    previous_total_rewards = float('-inf')

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = np.zeros(env.num_agents)
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
        alpha = max(0.001, alpha_start / (1 + episode * alpha_decay))

        while not done:
            actions = []
            for i in range(env.num_agents):
                state_index = state_to_index(state, env.grid_size)

                # Initialize Q-values for new states
                if state_index not in q_tables[i]:
                    q_tables[i][state_index] = np.zeros(5)  # 5 possible actions

                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, 4)  # Explore
                else:
                    action = np.argmax(q_tables[i][state_index])  # Exploit

                actions.append(action)

            next_state, rewards, done, info = env.step(actions)
            total_rewards += rewards

            for i in range(env.num_agents):
                state_index = state_to_index(state, env.grid_size)
                next_state_index = state_to_index(next_state, env.grid_size)

                # Initialize Q-values for new next states
                if next_state_index not in q_tables[i]:
                    q_tables[i][next_state_index] = np.zeros(5)

                best_next_action = np.argmax(q_tables[i][next_state_index])
                td_target = rewards[i] + gamma * q_tables[i][next_state_index][best_next_action]
                td_error = td_target - q_tables[i][state_index][actions[i]]

                q_tables[i][state_index][actions[i]] += alpha * td_error

            state = next_state

        if np.sum(total_rewards) > np.sum(previous_total_rewards):
            epsilon = max(epsilon_min, epsilon * 0.99)  # Decay faster if improving
        else:
            epsilon = max(epsilon_min, epsilon * 0.999)  # Decay slower if not improving

        previous_total_rewards = total_rewards

        # Print episode progress and update epsilon
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Rewards: {total_rewards}, Epsilon: {epsilon:.4f}")

    return q_tables

def q_learning_with_dynamic_scenarios(num_episodes, scenario_batch_size=100, min_grid_size=5, max_grid_size=10, min_agents=1, max_agents=4):
    """
    Q-learning algorithm with dynamic scenario generation to help the agent learn to handle any situation.

    Args:
        num_episodes: Total number of episodes to train for
        scenario_batch_size: Number of episodes to train on each scenario before generating a new one
        min_grid_size: Minimum grid dimension
        max_grid_size: Maximum grid dimension
        min_agents: Minimum number of agents
        max_agents: Maximum number of agents

    Returns:
        list: Q-tables for each agent
    """
    # Use a dictionary to store Q-values to save memory for large state spaces
    q_tables = {}  # We'll use a nested dictionary: {num_agents: [{agent1_qtable}, {agent2_qtable}, ...]}

    # Track performance across different scenario types
    scenario_metrics = {
        'random': {'episodes': 0, 'avg_reward': 0, 'success_rate': 0},
        'maze': {'episodes': 0, 'avg_reward': 0, 'success_rate': 0},
        'rooms': {'episodes': 0, 'avg_reward': 0, 'success_rate': 0},
        'corridors': {'episodes': 0, 'avg_reward': 0, 'success_rate': 0}
    }

    # Generate a variety of obstacle densities to try
    obstacle_densities = [0.1, 0.2, 0.3, 0.4]

    episode = 0
    while episode < num_episodes:
        # Generate a new random scenario
        obstacle_density = random.choice(obstacle_densities)
        scenario_params = generate_random_scenario(
            min_grid_size=min_grid_size,
            max_grid_size=max_grid_size,
            min_agents=min_agents,
            max_agents=max_agents,
            obstacle_density=obstacle_density
        )

        grid_size, num_agents, obstacles, goals, initial_positions = scenario_params

        # Determine the pattern type based on the obstacles
        if len(obstacles) < 10:
            pattern_type = 'random'
        elif any(all(o[0] % 2 == 0 for o in obstacles)):
            pattern_type = 'maze'
        elif any(all(o[0] == obstacles[0][0] or o[1] == obstacles[0][1] for o in obstacles[:5])):
            pattern_type = 'rooms'
        else:
            pattern_type = 'corridors'

        print(f"\nGenerating new scenario (type: {pattern_type}):")
        print(f"Grid Size: {grid_size}, Agents: {num_agents}, Obstacles: {len(obstacles)}")
        print(f"Goals: {goals}")
        print(f"Initial Positions: {initial_positions}")

        # Initialize environment with the new scenario
        env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)

        # Initialize Q-tables for this number of agents if not already done
        if num_agents not in q_tables:
            q_tables[num_agents] = [{} for _ in range(num_agents)]

        # Train on this scenario for a batch of episodes
        batch_size = min(scenario_batch_size, num_episodes - episode)
        total_batch_rewards = np.zeros(num_agents)
        successes = 0

        for batch_episode in range(batch_size):
            state = env.reset()
            done = False
            episode_rewards = np.zeros(num_agents)
            epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** (episode + batch_episode)))
            alpha = max(0.001, alpha_start / (1 + (episode + batch_episode) * alpha_decay))

            step_count = 0
            max_steps = 100  # Prevent infinite loops

            while not done and step_count < max_steps:
                actions = []
                for i in range(num_agents):
                    state_index = state_to_index(state, grid_size)

                    # Initialize Q-values for new states
                    if state_index not in q_tables[num_agents][i]:
                        q_tables[num_agents][i][state_index] = np.zeros(5)  # 5 possible actions

                    if random.uniform(0, 1) < epsilon:
                        action = random.randint(0, 4)  # Explore
                    else:
                        action = np.argmax(q_tables[num_agents][i][state_index])  # Exploit

                    actions.append(action)

                next_state, rewards, done, info = env.step(actions)
                episode_rewards += rewards

                for i in range(num_agents):
                    state_index = state_to_index(state, grid_size)
                    next_state_index = state_to_index(next_state, grid_size)

                    # Initialize Q-values for new next states
                    if next_state_index not in q_tables[num_agents][i]:
                        q_tables[num_agents][i][next_state_index] = np.zeros(5)

                    best_next_action = np.argmax(q_tables[num_agents][i][next_state_index])
                    td_target = rewards[i] + gamma * q_tables[num_agents][i][next_state_index][best_next_action]
                    td_error = td_target - q_tables[num_agents][i][state_index][actions[i]]

                    q_tables[num_agents][i][state_index][actions[i]] += alpha * td_error

                state = next_state
                step_count += 1

            # Check if all agents reached their goals
            if all(env.done):
                successes += 1

            total_batch_rewards += episode_rewards

            # Print progress every 10 episodes
            if batch_episode % 10 == 0:
                print(f"Scenario Episode {batch_episode}/{batch_size}, "
                      f"Total Rewards: {episode_rewards}, "
                      f"Epsilon: {epsilon:.4f}")

        # Update metrics for this scenario type
        avg_reward = np.mean(total_batch_rewards) / batch_size
        success_rate = successes / batch_size

        scenario_metrics[pattern_type]['episodes'] += batch_size
        scenario_metrics[pattern_type]['avg_reward'] = (
            (scenario_metrics[pattern_type]['avg_reward'] * 
             (scenario_metrics[pattern_type]['episodes'] - batch_size) + 
             avg_reward * batch_size) / 
            scenario_metrics[pattern_type]['episodes']
        )
        scenario_metrics[pattern_type]['success_rate'] = (
            (scenario_metrics[pattern_type]['success_rate'] * 
             (scenario_metrics[pattern_type]['episodes'] - batch_size) + 
             success_rate * batch_size) / 
            scenario_metrics[pattern_type]['episodes']
        )

        print(f"\nCompleted training on scenario (type: {pattern_type}):")
        print(f"Average Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")
        print(f"Overall metrics by scenario type:")
        for s_type, metrics in scenario_metrics.items():
            if metrics['episodes'] > 0:
                print(f"  {s_type}: Episodes: {metrics['episodes']}, "
                      f"Avg Reward: {metrics['avg_reward']:.2f}, "
                      f"Success Rate: {metrics['success_rate']:.2f}")

        episode += batch_size

    # Return the Q-tables for all agent configurations
    return q_tables


def create_custom_scenario(grid_size, num_agents, obstacles=None, goals=None, initial_positions=None):
    """
    Create a custom scenario with user-defined parameters.

    Args:
        grid_size (tuple): Size of the grid as (width, height)
        num_agents (int): Number of agents
        obstacles (list): List of obstacle positions as (x, y) tuples
        goals (list): List of goal positions as (x, y) tuples
        initial_positions (list): List of initial agent positions as (x, y) tuples

    Returns:
        tuple: (grid_size, num_agents, obstacles, goals, initial_positions)
    """
    # Set default values if not provided
    if obstacles is None:
        obstacles = []

    if goals is None:
        # Generate random goals if not provided
        goals = []
        for _ in range(num_agents):
            while True:
                goal = (random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1))
                if goal not in obstacles and goal not in goals:
                    goals.append(goal)
                    break

    if initial_positions is None:
        # Generate random initial positions if not provided
        initial_positions = []
        for _ in range(num_agents):
            while True:
                pos = (random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1))
                if pos not in obstacles and pos not in goals and pos not in initial_positions:
                    initial_positions.append(pos)
                    break

    # Validate inputs
    if len(goals) != num_agents:
        raise ValueError(f"Number of goals ({len(goals)}) must match number of agents ({num_agents})")

    if len(initial_positions) != num_agents:
        raise ValueError(f"Number of initial positions ({len(initial_positions)}) must match number of agents ({num_agents})")

    # Check for overlaps
    for pos in initial_positions:
        if pos in obstacles:
            raise ValueError(f"Initial position {pos} overlaps with an obstacle")
        if pos in goals:
            raise ValueError(f"Initial position {pos} overlaps with a goal")

    for goal in goals:
        if goal in obstacles:
            raise ValueError(f"Goal {goal} overlaps with an obstacle")

    return (grid_size, num_agents, obstacles, goals, initial_positions)

def define_scenario(scenario_id=0):
    """
    Define different scenarios for the RL agent to learn.
    Each scenario consists of grid size, number of agents, obstacles, goals, and initial positions.

    Args:
        scenario_id (int): ID of the scenario to use

    Returns:
        tuple: (grid_size, num_agents, obstacles, goals, initial_positions)
    """
    scenarios = {
        # Original scenario
        0: {
            "grid_size": (10, 10),
            "num_agents": 3,
            "obstacles": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (0, 8), (0, 9),
                         (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (1, 9),
                         (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9),
                         (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 7), (3, 8), (3, 9),
                         (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7), (4, 8), (4, 9),
                         (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
                         (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 6), (6, 7), (6, 8), (6, 9),
                         (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8), (7, 9),
                         (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 6), (8, 7), (8, 8), (8, 9),
                         (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 6), (9, 7), (9, 8), (9, 9)],
            "goals": [(9, 5), (0, 5), (5, 9)],
            "initial_positions": [(5, 9), (9, 5), (3, 6)]
        },
        # Simple scenario with smaller grid and fewer obstacles
        1: {
            "grid_size": (5, 5),
            "num_agents": 2,
            "obstacles": [(0, 0), (1, 1), (2, 2), (3, 3)],
            "goals": [(4, 4), (0, 4)],
            "initial_positions": [(0, 1), (4, 0)]
        },
        # Complex scenario with more agents
        2: {
            "grid_size": (8, 8),
            "num_agents": 4,
            "obstacles": [(1, 1), (1, 2), (1, 3), (3, 3), (3, 4), (3, 5), (5, 1), (5, 2), (5, 3)],
            "goals": [(7, 7), (0, 7), (7, 0), (0, 0)],
            "initial_positions": [(3, 0), (0, 3), (7, 3), (3, 7)]
        }
    }

    if scenario_id not in scenarios:
        raise ValueError(f"Scenario ID {scenario_id} not defined")

    scenario = scenarios[scenario_id]
    return (
        scenario["grid_size"],
        scenario["num_agents"],
        scenario["obstacles"],
        scenario["goals"],
        scenario["initial_positions"]
    )

def generate_random_scenario(min_grid_size=5, max_grid_size=10, min_agents=1, max_agents=4, obstacle_density=0.2):
    """
    Generate a random scenario with varying parameters to help the RL agent learn to handle any situation.

    Args:
        min_grid_size (int): Minimum grid dimension
        max_grid_size (int): Maximum grid dimension
        min_agents (int): Minimum number of agents
        max_agents (int): Maximum number of agents
        obstacle_density (float): Percentage of grid cells to be obstacles (0.0 to 1.0)

    Returns:
        tuple: (grid_size, num_agents, obstacles, goals, initial_positions)
    """
    # Randomly determine grid size
    width = random.randint(min_grid_size, max_grid_size)
    height = random.randint(min_grid_size, max_grid_size)
    grid_size = (width, height)

    # Randomly determine number of agents
    num_agents = random.randint(min_agents, max_agents)

    # Calculate number of obstacles based on density
    total_cells = width * height
    max_obstacles = int(total_cells * obstacle_density)

    # Generate obstacle patterns (randomly or with structure)
    pattern_type = random.choice(['random', 'maze', 'rooms', 'corridors'])
    obstacles = []

    if pattern_type == 'random':
        # Random obstacles
        num_obstacles = random.randint(max_obstacles // 2, max_obstacles)
        for _ in range(num_obstacles):
            while True:
                obs = (random.randint(0, width-1), random.randint(0, height-1))
                if obs not in obstacles:
                    obstacles.append(obs)
                    break

    elif pattern_type == 'maze':
        # Simple maze-like pattern
        for i in range(0, width, 2):
            for j in range(0, height, 2):
                if random.random() < 0.7:  # 70% chance to place an obstacle
                    if (i, j) not in obstacles:
                        obstacles.append((i, j))

    elif pattern_type == 'rooms':
        # Room-like structures
        room_count = random.randint(2, 4)
        for _ in range(room_count):
            room_x = random.randint(0, width-3)
            room_y = random.randint(0, height-3)
            room_w = random.randint(2, min(4, width-room_x))
            room_h = random.randint(2, min(4, height-room_y))

            # Create walls around the room
            for i in range(room_x, room_x + room_w):
                obstacles.append((i, room_y))
                obstacles.append((i, room_y + room_h - 1))

            for j in range(room_y + 1, room_y + room_h - 1):
                obstacles.append((room_x, j))
                obstacles.append((room_x + room_w - 1, j))

            # Add a door (remove one obstacle)
            door_wall = random.choice(['top', 'bottom', 'left', 'right'])
            if door_wall == 'top':
                door_pos = (random.randint(room_x, room_x + room_w - 1), room_y)
            elif door_wall == 'bottom':
                door_pos = (random.randint(room_x, room_x + room_w - 1), room_y + room_h - 1)
            elif door_wall == 'left':
                door_pos = (room_x, random.randint(room_y + 1, room_y + room_h - 2))
            else:
                door_pos = (room_x + room_w - 1, random.randint(room_y + 1, room_y + room_h - 2))

            if door_pos in obstacles:
                obstacles.remove(door_pos)

    elif pattern_type == 'corridors':
        # Corridor-like structures
        for i in range(1, width-1, 2):
            for j in range(height):
                if random.random() < 0.8:  # 80% chance to place an obstacle
                    obstacles.append((i, j))

        # Add some horizontal corridors by removing obstacles
        for _ in range(random.randint(1, 3)):
            y = random.randint(0, height-1)
            for x in range(width):
                if (x, y) in obstacles:
                    obstacles.remove((x, y))

    # Ensure we don't have too many obstacles
    if len(obstacles) > max_obstacles:
        obstacles = random.sample(obstacles, max_obstacles)

    # Generate goals and initial positions
    available_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles]

    if len(available_cells) < num_agents * 2:
        # Not enough space for agents and goals, reduce obstacles
        obstacles = random.sample(obstacles, len(obstacles) // 2)
        available_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles]

    # Ensure we have enough cells for agents and goals
    if len(available_cells) < num_agents * 2:
        num_agents = len(available_cells) // 2

    # Select cells for goals and initial positions
    selected_cells = random.sample(available_cells, num_agents * 2)
    goals = selected_cells[:num_agents]
    initial_positions = selected_cells[num_agents:]

    return (grid_size, num_agents, obstacles, goals, initial_positions)

def train_and_test_scenario(scenario_id, episodes=num_episodes):
    """
    Train and test the RL agent on a specific scenario.

    Args:
        scenario_id (int): ID of the scenario to use
        episodes (int): Number of episodes for training

    Returns:
        tuple: (q_tables, actions_list, env) - Learned Q-tables, actions taken during testing, and environment
    """
    grid_size, num_agents, obstacles, goals, initial_positions = define_scenario(scenario_id)

    print(f"\nTraining on Scenario {scenario_id}:")
    print(f"Grid Size: {grid_size}, Agents: {num_agents}")
    print(f"Goals: {goals}")
    print(f"Initial Positions: {initial_positions}")

    env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)
    q_tables = q_learning(env, episodes)

    print(f"Training completed for Scenario {scenario_id}.")

    # Testing trained agents
    state = env.reset()
    done = False
    actions_list = []
    step_count = 0
    max_steps = 100  # Prevent infinite loops during testing

    print(f"\nTesting Scenario {scenario_id}:")
    while not done and step_count < max_steps:
        actions = []
        for i in range(num_agents):
            state_index = state_to_index(state, grid_size)

            # Handle case where state wasn't seen during training
            if state_index not in q_tables[i]:
                q_tables[i][state_index] = np.zeros(5)
                action = random.randint(0, 4)  # Random action for unseen state
            else:
                action = np.argmax(q_tables[i][state_index])

            actions.append(action)

        actions_list.append(actions)
        print(f"Step {step_count}: Agents' positions: {state}, Actions: {actions}")
        state, rewards, done, _ = env.step(actions)
        step_count += 1

    if step_count >= max_steps:
        print(f"Warning: Testing reached maximum steps ({max_steps}) without completion.")
    else:
        print(f"Testing completed for Scenario {scenario_id} in {step_count} steps.")

    return q_tables, actions_list, env

def main():
    """
    Main function to run the RL-based traffic management system.
    Users can choose to run a single scenario, multiple scenarios, or create a custom scenario.
    """
    import argparse

    parser = argparse.ArgumentParser(description='RL-based Traffic Management')

    # Scenario selection arguments
    scenario_group = parser.add_argument_group('Scenario Selection')
    scenario_group.add_argument('--scenario', type=int, default=0, 
                        help='Scenario ID to run (default: 0)')
    scenario_group.add_argument('--all', action='store_true', 
                        help='Run all available scenarios')

    # Custom scenario arguments
    custom_group = parser.add_argument_group('Custom Scenario')
    custom_group.add_argument('--custom', action='store_true',
                        help='Create and run a custom scenario')
    custom_group.add_argument('--grid-size', type=int, nargs=2, default=[5, 5],
                        help='Grid size as width height (default: 5 5)')
    custom_group.add_argument('--num-agents', type=int, default=2,
                        help='Number of agents (default: 2)')
    custom_group.add_argument('--random-obstacles', type=int, default=0,
                        help='Number of random obstacles to generate (default: 0)')

    # General arguments
    parser.add_argument('--episodes', type=int, default=num_episodes, 
                        help=f'Number of training episodes (default: {num_episodes})')
    parser.add_argument('--no-animation', action='store_true', 
                        help='Disable animation of results')

    args = parser.parse_args()

    # Get the number of available scenarios
    scenarios = {
        0: "Original scenario",
        1: "Simple scenario",
        2: "Complex scenario"
    }
    num_scenarios = len(scenarios)

    # Handle custom scenario
    if args.custom:
        print(f"\n{'='*50}")
        print(f"Creating custom scenario:")
        print(f"Grid Size: {args.grid_size}, Agents: {args.num_agents}")

        # Generate random obstacles if requested
        obstacles = []
        if args.random_obstacles > 0:
            grid_size = tuple(args.grid_size)
            total_cells = grid_size[0] * grid_size[1]
            max_obstacles = min(args.random_obstacles, total_cells - args.num_agents * 2)  # Leave space for agents and goals

            for _ in range(max_obstacles):
                while True:
                    obs = (random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1))
                    if obs not in obstacles:
                        obstacles.append(obs)
                        break

            print(f"Generated {len(obstacles)} random obstacles")

        try:
            # Create custom scenario
            scenario_params = create_custom_scenario(
                tuple(args.grid_size),
                args.num_agents,
                obstacles=obstacles
            )

            print(f"Goals: {scenario_params[3]}")
            print(f"Initial Positions: {scenario_params[4]}")
            print(f"{'='*50}")

            # Train and test on custom scenario
            env = GridEnv(*scenario_params)
            q_tables = q_learning(env, args.episodes)

            # Testing
            state = env.reset()
            done = False
            actions_list = []
            step_count = 0
            max_steps = 100

            print(f"\nTesting Custom Scenario:")
            while not done and step_count < max_steps:
                actions = []
                for i in range(args.num_agents):
                    state_index = state_to_index(state, tuple(args.grid_size))

                    if state_index not in q_tables[i]:
                        q_tables[i][state_index] = np.zeros(5)
                        action = random.randint(0, 4)
                    else:
                        action = np.argmax(q_tables[i][state_index])

                    actions.append(action)

                actions_list.append(actions)
                print(f"Step {step_count}: Agents' positions: {state}, Actions: {actions}")
                state, _, done, _ = env.step(actions)
                step_count += 1

            if not args.no_animation:
                env.animate(actions_list)

        except ValueError as e:
            print(f"Error creating custom scenario: {e}")
            return

    # Handle predefined scenarios
    elif args.all or args.scenario in scenarios:
        # Display available scenarios if invalid scenario ID
        if not args.all and args.scenario not in scenarios:
            print("Available scenarios:")
            for sid, desc in scenarios.items():
                print(f"  {sid}: {desc}")
            print(f"Error: Scenario {args.scenario} not found.")
            return

        if args.all:
            print(f"Running all {num_scenarios} available scenarios...")
            for scenario_id in range(num_scenarios):
                print(f"\n{'='*50}")
                print(f"Scenario {scenario_id}: {scenarios[scenario_id]}")
                print(f"{'='*50}")
                q_tables, actions_list, env = train_and_test_scenario(scenario_id, args.episodes)
                if not args.no_animation:
                    env.animate(actions_list)
        else:
            print(f"\n{'='*50}")
            print(f"Running scenario {args.scenario}: {scenarios[args.scenario]}")
            print(f"{'='*50}")
            q_tables, actions_list, env = train_and_test_scenario(args.scenario, args.episodes)
            if not args.no_animation:
                env.animate(actions_list)
    else:
        # Display available options if no valid option was selected
        print("Available scenarios:")
        for sid, desc in scenarios.items():
            print(f"  {sid}: {desc}")
        print("\nUse --custom to create a custom scenario")
        print("Use --all to run all scenarios")
        print("Use --help for more information")
        return

    print("All scenarios completed.")


if __name__ == "__main__":
    main()
