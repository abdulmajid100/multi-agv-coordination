import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GraphEnv:
    def __init__(self, graph, num_agents, goals, initial_positions=None):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_agents = num_agents
        self.goals = goals
        self.initial_positions = initial_positions
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

    def _get_random_position(self):
        return random.choice(self.nodes)

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        next_agents = []

        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if self.done[i]:
                next_agents.append(agent)
                continue

            neighbors = list(self.graph.neighbors(agent))
            if action < len(neighbors):
                next_agent = neighbors[action]
            else:
                next_agent = agent  # Invalid action, stay in place

            if next_agent == self.goals[i]:
                rewards[i] = 1.0  # Reward for reaching the goal
                self.done[i] = True
            else:
                rewards[i] = -0.1  # Small step penalty to encourage faster goal reaching

            next_agents.append(next_agent)

        self.agents = next_agents
        self.steps += 1
        done = all(self.done) or self.steps >= 500
        return np.array(self.agents), rewards, done, {'steps': self.steps}

    def animate(self, actions_sequence):
        fig, ax = plt.subplots(figsize=(8, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, ax=ax, with_labels=True, node_size=700, node_color='lightgray')

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