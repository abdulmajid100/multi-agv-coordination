import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

class GridEnv:
    def __init__(self, grid_size, num_agents, obstacles, goals, dirt_density, initial_positions=None):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obstacles = obstacles
        self.goals = goals
        self.dirt_density = dirt_density
        self.initial_positions = initial_positions
        self.next_agents = []  # Initialize next_agents here
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.dirt = self._create_dirt()
        if self.initial_positions is None:
            self.agents = [self._get_random_position() for _ in range(self.num_agents)]
        else:
            self.agents = self._validate_initial_positions(self.initial_positions)
        self.done = [False] * self.num_agents
        self.steps = 0
        self.next_agents = self.agents.copy()  # Initialize next_agents in reset
        return np.array(self.agents)

    def _create_dirt(self, seed=1):
        """Randomly place dirt in the grid based on dirt_density with an optional seed."""
        if seed is not None:
            np.random.seed(1)
        num_cells = self.grid_size[0] * self.grid_size[1]
        num_dirt_cells = int(num_cells * self.dirt_density)
        dirt_cells = set()
        while len(dirt_cells) < num_dirt_cells:
            pos = tuple(np.random.randint(0, s) for s in self.grid_size)
            if pos not in self.obstacles and pos not in self.goals:
                dirt_cells.add(pos)
        return list(dirt_cells)

    def _validate_initial_positions(self, positions):
        validated_positions = []
        for pos in positions:
            if pos in self.obstacles or pos in self.goals:
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
            if self._is_collision(next_agent) or any(
                    np.array_equal(next_agent, other_agent) or np.array_equal(next_agent, self.agents[j])
                    for j, other_agent in enumerate(next_agents) if j != i):
                collisions[i] = True

        # Update agent positions and calculate rewards if the episode is not done
        for i, (agent, next_agent) in enumerate(zip(self.agents, next_agents)):
            if not self.done[i]:  # Only update if the agent is not done
                next_agent_tuple = tuple(next_agent)
                agent_tuple = tuple(agent)

                if collisions[i]:
                    rewards[i] -= 1  # Penalty for collision
                    next_agent = agent  # Stay in the same place
                elif any(np.array_equal(next_agent_tuple, d) or np.array_equal(agent_tuple, d) for d in map(tuple, self.dirt)):
                    rewards[i] += 1000  # Reward for cleaning dirt
                    self.dirt = [d for d in self.dirt if not np.array_equal(d, next_agent_tuple) and not np.array_equal(d, agent_tuple)]  # Remove dirt after cleaning
                    #print(f"Agent {i} cleaned dirt at {next_agent}")  # Debug print
                    if len(self.dirt) == 0:
                        rewards[i] += 10000
                        self.done[i] = True
                elif (len(self.dirt) != 0):
                    rewards[i] -= 10

                #rewards[i] -= 1  # Penalty for each step taken
                self.next_agents[i] = next_agent
            else:
                self.next_agents[i] = agent
        #print(self.dirt)
        self.agents = self.next_agents.copy()
        self.steps += 1
        done = all(self.done) or self.steps >= 5000  # Terminate after cleaning all dirt or 500 steps

        return np.array(self.agents), rewards, done, {'steps': self.steps}

    def _apply_action(self, position, action):
        # Define actions: 0=left, 1=up, 2=right, 3=down, 4=wait
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]  # Corrected move list
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

    def _get_random_position(self):
        while True:
            pos = tuple(np.random.randint(0, s) for s in self.grid_size)
            if pos not in self.obstacles and pos not in self.goals and pos not in self.dirt:
                return pos

    def animate(self, actions_sequence):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Set limits and ticks for the grid
        ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.grid(True)

        # Initialize the environment
        self.reset()

        def plot_environment():
            # Clear previous plot
            ax.clear()
            ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
            ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
            ax.set_xticks(np.arange(self.grid_size[1]))
            ax.set_yticks(np.arange(self.grid_size[0]))
            ax.grid(True)

            # Plot obstacles, goals, dirt, and agents
            for obs in self.obstacles:
                rect = patches.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, linewidth=1,
                                         edgecolor='black', facecolor='red')
                ax.add_patch(rect)

            for goal in self.goals:
                rect = patches.Rectangle((goal[0] - 0.5, goal[1] - 0.5), 1, 1, linewidth=1,
                                         edgecolor='black', facecolor='green')
                ax.add_patch(rect)

            for dirt in self.dirt:
                rect = patches.Rectangle((dirt[0] - 0.5, dirt[1] - 0.5), 1, 1, linewidth=1,
                                         edgecolor='black', facecolor='brown')
                ax.add_patch(rect)

            agent_colors = ['blue', 'brown', 'black', 'purple', 'orange']
            for i, agent in enumerate(self.agents):
                circle = patches.Circle((agent[0], agent[1]), 0.3, color=agent_colors[i % len(agent_colors)])
                ax.add_patch(circle)

        def update(frame):
            # Clear previous plot
            ax.clear()
            ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
            ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
            ax.set_xticks(np.arange(self.grid_size[1]))
            ax.set_yticks(np.arange(self.grid_size[0]))
            ax.grid(True)

            if frame == 0:
                # Plot initial environment
                plot_environment()
            else:
                actions = actions_sequence[frame - 1]
                self.agents, _, _, _ = self.step(actions)
                plot_environment()

            return []

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(actions_sequence) + 1, blit=False, repeat=False,
                                       interval=500)
        plt.title("Grid Environment Animation")
        plt.show()

    def execute_actions(self, actions_sequence):
        state_history = []
        self.reset()
        state_history.append(self.agents.copy())

        for actions in actions_sequence:
            self.agents, _, done, _ = self.step(actions)
            state_history.append(self.agents.copy())
            if done:
                break

        return state_history
