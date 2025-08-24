import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GraphEnv:
    def __init__(self, graph, num_agents, goals, initial_positions=None):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_agents = num_agents
        self.goals = goals
        self.initial_positions = initial_positions
        self.next_agents = []
        self.reset()

    def reset(self):
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
            if pos not in self.nodes:
                raise ValueError(f"Initial position {pos} is not a valid node in the graph.")
            validated_positions.append(pos)
        return validated_positions

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        collisions = [False] * self.num_agents

        if not self.next_agents:
            self.next_agents = self.agents.copy()

        # Compute next positions based on actions
        next_agents = []
        for agent, action in zip(self.agents, actions):
            next_agents.append(self._apply_action(agent, action))

        # Check for collisions
        for i, next_agent in enumerate(next_agents):
            if any(next_agent == other_agent for j, other_agent in enumerate(next_agents) if j != i):
                collisions[i] = True

        # Update agent positions and calculate rewards if the episode is not done
        for i, (agent, next_agent) in enumerate(zip(self.agents, next_agents)):
            if not self.done[i]:
                if collisions[i]:
                    rewards[i] -= 1  # Penalty for collision
                    next_agent = agent
                elif self._is_goal(next_agent, i):
                    rewards[i] += 1  # Reward for reaching the goal
                    self.done[i] = True
                else:
                    rewards[i] += self._distance_reward(agent, next_agent, i)

                self.next_agents[i] = next_agent
                rewards[i] -= 1  # Penalty for each step taken
            else:
                self.next_agents[i] = agent

        self.agents = self.next_agents.copy()
        self.steps += 1
        done = all(self.done) or self.steps >= 500

        return np.array(self.agents), rewards, done, {'steps': self.steps}

    def _apply_action(self, position, action):
        neighbors = list(self.graph.neighbors(position))
        if action < len(neighbors):
            return neighbors[action]
        else:
            return position  # Invalid action, stay in place
    def _is_goal(self, position, agent_index):
        if isinstance(position, (list, np.ndarray)):
            position = tuple(position)
        return position == self.goals[agent_index]

    def _distance_reward(self, current_position, next_position, agent_index):
        """Calculate reward based on distance reduction to the goal."""
        current_distance = self._distance(current_position, self.goals[agent_index])
        new_distance = self._distance(next_position, self.goals[agent_index])
        return (current_distance - new_distance) * 100  # Reward for getting closer

    def _distance(self, pos1, pos2):
        """Calculate a simple graph-based distance (e.g., Manhattan if applicable)."""
        # For graph-based environments, you might consider shortest path length
        # if using a grid-like graph, otherwise use a heuristic.
        # Here, we use a simple placeholder analogous to Manhattan distance.
        return abs(pos1 - pos2)  # Assuming nodes are indexed uniquely
    def animate(self, actions_sequence):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw the graph
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, ax=ax, with_labels=True, node_size=700, node_color='lightblue')

        # Initialize agent positions
        agent_colors = ['blue', 'brown', 'black', 'purple', 'orange']
        agents_patches = [plt.Circle(pos[node], 0.05, color=agent_colors[i % len(agent_colors)]) for i, node in enumerate(self.agents)]
        for patch in agents_patches:
            ax.add_patch(patch)

        def update(frame):
            if frame == 0:
                self.reset()
                for i, agent in enumerate(self.agents):
                    agents_patches[i].center = pos[agent]
            else:
                actions = actions_sequence[frame - 1]
                self.agents, _, _, _ = self.step(actions)
                for i, agent in enumerate(self.agents):
                    agents_patches[i].center = pos[agent]

            return agents_patches

        anim = animation.FuncAnimation(fig, update, frames=len(actions_sequence) + 1, blit=True, repeat=False, interval=1000)
        plt.title("Graph Environment Animation")
        plt.show()