import simpy
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# Q-learning parameters
alpha = 0.05  # Learning rate
gamma = 0.99  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.997  # Decay rate for epsilon
num_episodes = 1000  # Number of episodes
alpha_start = 1
alpha_decay = 0.01

class GridEnvSimPy:
    def __init__(self, env, grid_size, num_agents, obstacles, goals, initial_positions=None):
        self.env = env  # SimPy environment
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obstacles = obstacles
        self.goals = goals
        self.initial_positions = initial_positions
        self.next_agents = []
        self.agents = []
        self.done = []
        self.steps = 0
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)
        if self.initial_positions is None:
            self.agents = [self._get_random_position() for _ in range(self.num_agents)]
        else:
            self.agents = self._validate_initial_positions(self.initial_positions)
        self.next_agents = self.agents.copy()
        self.done = [False] * self.num_agents
        self.steps = 0
        return np.array(self.agents)

    def _validate_initial_positions(self, positions):
        validated_positions = []
        for pos in positions:
            if pos in self.obstacles:
                raise ValueError(f"Initial position {pos} overlaps with an obstacle or goal.")
            validated_positions.append(pos)
        return validated_positions

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        collisions = [False] * self.num_agents

        # Compute next positions based on actions
        next_agents = []
        for agent, action in zip(self.agents, actions):
            next_agents.append(self._apply_action(agent, action))

        # Check for collisions
        for i, next_agent in enumerate(next_agents):
            if self._is_collision(next_agent) or any(np.array_equal(next_agent, other_agent) or np.array_equal(next_agent, self.agents[j])
                                                     for j, other_agent in enumerate(next_agents) if j != i):
                collisions[i] = True

        # Update agent positions and calculate rewards if the episode is not done
        for i, (agent, next_agent) in enumerate(zip(self.agents, next_agents)):
            if not self.done[i]:  # Only update if the agent is not done
                if collisions[i]:
                    rewards[i] -= 10  # Penalty for collision
                    next_agent = agent  # Stay in the same place
                elif self._is_goal(next_agent, i):
                    rewards[i] += 1000  # Reward for reaching the goal
                    self.done[i] = True  # Mark as done
                else:
                    current_distance = self._distance(agent, self.goals[i])
                    new_distance = self._distance(next_agent, self.goals[i])
                    distance_reward = (current_distance - new_distance) * 100 - 50  # Reward for getting closer
                    rewards[i] += distance_reward

                self.next_agents[i] = next_agent
                rewards[i] -= 50  # Penalty for each step taken
            else:
                self.next_agents[i] = agent

        self.agents = self.next_agents.copy()
        self.steps += 1
        done = all(self.done) or self.steps >= 500  # Terminate after 500 steps or if all agents are done

        return np.array(self.agents), rewards, done, {'steps': self.steps}

    def _distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_random_position(self):
        while True:
            pos = tuple(np.random.randint(0, s) for s in self.grid_size)
            if pos not in self.obstacles and pos not in self.goals:
                return pos

    def _apply_action(self, position, action):
        # Define actions: 0=left, 1=up, 2=right, 3=down, 4=wait
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
        if action < len(moves):  # Ensure action is within bounds
            move = moves[action]
            new_position = (position[0] + move[0], position[1] + move[1])
            return new_position if self._is_within_bounds(new_position) else position
        else:
            raise ValueError(f"Invalid action {action}. Action must be between 0 and {len(moves) - 1}.")

    def _is_within_bounds(self, position):
        return 0 <= position[0] < self.grid_size[0] and 0 <= position[1] < self.grid_size[1]

    def _is_collision(self, position):
        if isinstance(position, (list, np.ndarray)):
            position = tuple(position)
        return position in self.obstacles

    def _is_goal(self, position, agent_index):
        if isinstance(position, (list, np.ndarray)):
            position = tuple(position)
        return position == self.goals[agent_index]

    def animate(self, actions_sequence):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Set limits and ticks for the grid
        ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.grid(True)

        # Plot obstacles
        for obs in self.obstacles:
            rect = patches.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, linewidth=1,
                                     edgecolor='black', facecolor='red')
            ax.add_patch(rect)

        # Plot goals
        for goal in self.goals:
            rect = patches.Rectangle((goal[0] - 0.5, goal[1] - 0.5), 1, 1, linewidth=1,
                                     edgecolor='black', facecolor='green')
            ax.add_patch(rect)

        # Initialize agents
        agent_colors = ['blue', 'brown', 'black', 'purple', 'orange']
        agents_patches = [patches.Circle((pos[0], pos[1]), 0.3, color=agent_colors[i % len(agent_colors)])
                          for i, pos in enumerate(self.agents)]
        for patch in agents_patches:
            ax.add_patch(patch)

        def update(frame):
            if frame == 0:
                # Reset and initialize for the first frame
                self.reset()
                for i, agent in enumerate(self.agents):
                    x, y = agent
                    agents_patches[i].center = (x, y)
            else:
                actions = actions_sequence[frame - 1]
                self.agents, _, _, _ = self.step(actions)
                for i, agent in enumerate(self.agents):
                    x, y = agent
                    agents_patches[i].center = (x, y)

            return agents_patches

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(actions_sequence) + 1, blit=True, repeat=False, interval=1000)
        plt.title("Grid Environment Animation")
        plt.show()

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

# Agent process in SimPy
def agent_process(env, agent_id, grid_env, q_table, epsilon, alpha):
    while True:
        # Get current state
        state = grid_env.agents
        state_index = state_to_index(state, grid_env.grid_size)
        
        # Choose action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 4)  # Explore
        else:
            action = np.argmax(q_table[state_index])  # Exploit
            
        # Store the action for later use
        yield env.process(take_action(env, agent_id, action, grid_env, q_table, alpha))
        
        # Check if done
        if all(grid_env.done) or grid_env.steps >= 500:
            break

def take_action(env, agent_id, action, grid_env, q_table, alpha):
    # This is a placeholder for the actual action-taking process
    # In a real implementation, this would interact with the environment
    yield env.timeout(1)  # Simulate time passing
    
    # Update Q-table (simplified)
    # In a real implementation, this would use the actual reward and next state
    state_index = state_to_index(grid_env.agents, grid_env.grid_size)
    # Placeholder for Q-learning update
    # q_table[state_index][action] += alpha * (reward + gamma * max_next_q - q_table[state_index][action])

def q_learning_simpy(grid_size, num_agents, obstacles, goals, initial_positions, num_episodes):
    q_tables = [np.zeros(((grid_size[0] * grid_size[1]) ** num_agents, 5)) for _ in range(num_agents)]
    previous_total_rewards = float('-inf')
    
    for episode in range(num_episodes):
        # Create a new SimPy environment for each episode
        env = simpy.Environment()
        grid_env = GridEnvSimPy(env, grid_size, num_agents, obstacles, goals, initial_positions)
        state = grid_env.reset()
        done = False
        total_rewards = np.zeros(num_agents)
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
        alpha_val = max(0.001, alpha_start / (1 + episode * alpha_decay))
        
        # Create agent processes
        agent_processes = []
        for i in range(num_agents):
            agent_processes.append(env.process(agent_process(env, i, grid_env, q_tables[i], epsilon, alpha_val)))
        
        # Run the simulation until all agents are done or max steps reached
        while not done:
            actions = []
            for i in range(num_agents):
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, 4)  # Explore
                else:
                    state_index = state_to_index(state, grid_size)
                    action = np.argmax(q_tables[i][state_index])  # Exploit
                actions.append(action)
            
            next_state, rewards, done, info = grid_env.step(actions)
            total_rewards += rewards
            
            # Update Q-tables
            for i in range(num_agents):
                state_index = state_to_index(state, grid_size)
                next_state_index = state_to_index(next_state, grid_size)
                
                best_next_action = np.argmax(q_tables[i][next_state_index])
                td_target = rewards[i] + gamma * q_tables[i][next_state_index][best_next_action]
                td_error = td_target - q_tables[i][state_index][actions[i]]
                
                q_tables[i][state_index][actions[i]] += alpha_val * td_error
            
            state = next_state
            
            # Run the SimPy environment for one step
            env.run(until=env.now + 1)
        
        # Adjust epsilon based on performance
        if np.sum(total_rewards) > np.sum(previous_total_rewards):
            epsilon = max(epsilon_min, epsilon * 0.99)  # Decay faster if improving
        else:
            epsilon = max(epsilon_min, epsilon * 0.999)  # Decay slower if not improving
        
        # Update previous total rewards
        previous_total_rewards = total_rewards
        
        # Print episode progress
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Rewards: {total_rewards}, Epsilon: {epsilon:.4f}")
    
    return q_tables

def main():
    grid_size = (10, 10)
    num_agents = 3
    obstacles = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (0, 8), (0, 9),
                 (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (1, 9),
                 (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9),
                 (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 7), (3, 8), (3, 9),
                 (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7), (4, 8), (4, 9),
                 (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
                 (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 6), (6, 7), (6, 8), (6, 9),
                 (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8), (7, 9),
                 (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 6), (8, 7), (8, 8), (8, 9),
                 (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 6), (9, 7), (9, 8), (9, 9)]
    goals = [(9, 5), (0, 5), (5, 9)]
    initial_positions = [(5, 9), (9, 5), (3, 6)]

    # Train agents using SimPy-based Q-learning
    q_tables = q_learning_simpy(grid_size, num_agents, obstacles, goals, initial_positions, num_episodes)

    print("Training completed.")

    # Testing trained agents
    env = simpy.Environment()
    grid_env = GridEnvSimPy(env, grid_size, num_agents, obstacles, goals, initial_positions)
    state = grid_env.reset()
    done = False
    actions_list = []
    
    while not done:
        actions = []
        for i in range(num_agents):
            state_index = state_to_index(state, grid_size)
            action = np.argmax(q_tables[i][state_index])
            actions.append(action)
        actions_list.append(actions)
        print(f"Agents' positions: {state}")
        state, _, done, _ = grid_env.step(actions)

    grid_env.animate(actions_list)  # For animation

if __name__ == "__main__":
    main()