# grid_env.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

class GridEnv:
    def __init__(self, grid_size, num_agents, obstacles, goals, initial_positions=None):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obstacles = obstacles
        self.goals = goals
        self.initial_positions = initial_positions
        self.next_agents = []
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)
        if self.initial_positions is None:
            self.agents = [self._get_random_position() for _ in range(self.num_agents)]
        else:
            self.agents = self._validate_initial_positions(self.initial_positions)
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

        if not self.next_agents:  # If next_agents is empty, initialize it
            self.next_agents = self.agents.copy()

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
                    rewards[i] -= 1  # Penalty for collision
                    next_agent = agent  # Stay in the same place
                elif self._is_goal(next_agent, i):
                    rewards[i] += 1000  # Reward for reaching the goal
                    self.done[i] = True  # Mark as done
                else:
                    current_distance = self._distance(agent, self.goals[i])
                    new_distance = self._distance(next_agent, self.goals[i])
                    distance_reward = (current_distance - new_distance) * 50 # Reward for getting closer
                    rewards[i] += distance_reward

                self.next_agents[i] = next_agent
                rewards[i] -= 1  # Penalty for each step taken
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
