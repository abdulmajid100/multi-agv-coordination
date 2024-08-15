import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

class GridEnv:
    def __init__(self, grid_size, num_agents, obstacles, goals):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.obstacles = obstacles
        self.goals = goals
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.agents = [self._get_random_position() for _ in range(self.num_agents)]
        self.done = [False] * self.num_agents
        self.steps = 0
        return np.array(self.agents)

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        next_agents = []

        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if not self.done[i]:
                next_agent = self._apply_action(agent, action)
                if self._is_collision(next_agent):
                    rewards[i] -= 10  # Penalty for hitting an obstacle
                    next_agent = agent  # Stay in the same place
                elif self._is_goal(next_agent, i):
                    rewards[i] += 100  # Reward for reaching the goal
                    self.done[i] = True  # Mark as done
                next_agents.append(next_agent)
            else:
                next_agents.append(agent)

        self.agents = next_agents
        self.steps += 1
        done = all(self.done) or self.steps >= 100  # Terminate after 100 steps or all agents done
        return np.array(self.agents), rewards, done, {}

    def _get_random_position(self):
        while True:
            pos = tuple(np.random.randint(0, s) for s in self.grid_size)
            if pos not in self.obstacles and pos not in self.goals:
                return pos

    def _apply_action(self, position, action):
        # Define actions: 0=left, 1=up, 2=right, 3=down
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        move = moves[action]
        new_position = (position[0] + move[0], position[1] + move[1])
        return new_position if self._is_within_bounds(new_position) else position

    def _is_within_bounds(self, position):
        return 0 <= position[0] < self.grid_size[0] and 0 <= position[1] < self.grid_size[1]

    def _is_collision(self, position):
        return position in self.obstacles

    def _is_goal(self, position, agent_index):
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
            # (obs[1], self.grid_size[0] - obs[0] - 1) for correct placement
            rect = patches.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, linewidth=1,
                                     edgecolor='black', facecolor='red')
            ax.add_patch(rect)

        # Plot goals
        for goal in self.goals:
            # (goal[1], self.grid_size[0] - goal[0] - 1) for correct placement
            rect = patches.Rectangle((goal[0] - 0.5, goal[1] - 0.5), 1, 1, linewidth=1,
                                     edgecolor='black', facecolor='green')
            ax.add_patch(rect)

        # Initialize agents
        agents_patches = [patches.Circle((0, 0), 0.3, color='blue') for _ in range(self.num_agents)]
        for patch in agents_patches:
            ax.add_patch(patch)

        def update(frame):
            actions = actions_sequence[frame]
            next_state, _, _, _ = self.step(actions)
            for i, agent in enumerate(next_state):
                # Ensure correct plotting coordinates
                x, y = agent
                # (y, self.grid_size[0] - x - 1) for correct placement
                agents_patches[i].center = (x, y)
            return agents_patches

        anim = animation.FuncAnimation(fig, update, frames=len(actions_sequence), blit=True, repeat=False)
        #plt.gca().invert_yaxis()  # Invert the y-axis to match grid coordinates
        plt.title("Grid Environment Animation")
        plt.show()


# Example usage
if __name__ == "__main__":
    grid_size = (5, 5)
    num_agents = 2
    obstacles = [(2, 1), (4, 3)]
    goals = [(4, 4), (0, 0)]

    env = GridEnv(grid_size, num_agents, obstacles, goals)
    actions_sequence = []

    # Generate a random sequence of actions for demonstration
    for _ in range(50):  # 50 frames
        actions = [np.random.choice(4) for _ in range(num_agents)]
        actions_sequence.append(actions)
        env.step(actions)

    env.animate(actions_sequence)
