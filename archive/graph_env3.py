import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GraphEnv:
    def __init__(self, graph, num_agents, goals, initial_positions=None):
        self.graph = graph
        self.num_agents = num_agents
        self.goals = goals
        self.initial_positions = initial_positions
        self.agents = []
        self.done = [False] * num_agents
        self.reset()

    def reset(self):
        if self.initial_positions is None:
            self.agents = [self._get_random_node() for _ in range(self.num_agents)]
        else:
            self.agents = self._validate_initial_positions(self.initial_positions)
        self.done = [False] * self.num_agents
        return np.array(self.agents)

    def _validate_initial_positions(self, positions):
        for pos in positions:
            if pos not in self.graph.nodes:
                raise ValueError(f"Initial position {pos} is not a valid node in the graph.")
        return positions

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        collisions = [False] * self.num_agents

        next_agents = []
        for agent, action in zip(self.agents, actions):
            if action in self.graph[agent]:
                next_agents.append(action)
            else:
                next_agents.append(agent)

        # Check for collisions
        for i, next_agent in enumerate(next_agents):
            if any(next_agent == other_agent for j, other_agent in enumerate(next_agents) if j != i):
                collisions[i] = True

        # Update agent positions and calculate rewards
        for i, (agent, next_agent) in enumerate(zip(self.agents, next_agents)):
            if not self.done[i]:
                if collisions[i]:
                    rewards[i] -= 1  # Penalty for collision
                elif next_agent == self.goals[i]:
                    rewards[i] += 1  # Reward for reaching the goal
                    self.done[i] = True
                self.agents[i] = next_agent
                rewards[i] -= 1  # Penalty for each step

        done = all(self.done)
        return np.array(self.agents), rewards, done, {}

    def _get_random_node(self):
        return np.random.choice(self.graph.nodes)

    def animate(self, actions_sequence):
        fig, ax = plt.subplots(figsize=(6, 6))
        pos = nx.spring_layout(self.graph)  # Positions for all nodes

        def update(frame):
            ax.clear()
            nx.draw(self.graph, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray')
            agent_positions = {i: pos[self.agents[i]] for i in range(self.num_agents)}
            nx.draw_networkx_nodes(self.graph, pos, nodelist=agent_positions.keys(),
                                   node_color=['blue', 'green'][:self.num_agents])
            return ax,

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(actions_sequence), blit=True, repeat=False, interval=1000)
        plt.show()

# Sample usage
graph = nx.grid_2d_graph(5, 5)  # Create a simple 5x5 grid graph
goals = [(4, 4), (0, 0)]
initial_positions = [(0, 0), (4, 4)]
env = GraphEnv(graph, num_agents=2, goals=goals, initial_positions=initial_positions)