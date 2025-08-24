import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GraphEnv:
    def __init__(self, graph, num_agents, goals, corridor_nodes, initial_positions=None):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_agents = num_agents
        self.goals = goals
        self.corridor_nodes = corridor_nodes
        self.initial_positions = initial_positions
        self.reset()

    def reset(self):
        if self.initial_positions is None:
            self.agents = [self._get_random_position() for _ in range(self.num_agents)]
        else:
            self.agents = self._validate_initial_positions(self.initial_positions)
        self.done = [False] * self.num_agents
        self.steps = 0
        self.corridor_occupied = False
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

            # Check corridor constraint
            if next_agent in self.corridor_nodes:
                if self.corridor_occupied:
                    next_agent = agent  # Wait if corridor is occupied
                    rewards[i] = -0.1  # Penalty for waiting
                else:
                    self.corridor_occupied = True  # Occupy the corridor
            else:
                if agent in self.corridor_nodes:
                    self.corridor_occupied = False  # Free the corridor

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

# Example usage
if __name__ == "__main__":
    # Create a simple graph
    G = nx.Graph()
    G.add_edges_from([
        ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'),
        ('B', 'F'), ('F', 'G'), ('G', 'H'), ('H', 'E')
    ])

    # Define goals and initial positions
    goals = ['E', 'E']
    initial_positions = ['A', 'H']
    corridor_nodes = ['C', 'D']

    # Create environment
    env = GraphEnv(G, num_agents=2, goals=goals, corridor_nodes=corridor_nodes, initial_positions=initial_positions)

    # Define a sequence of actions for animation
    actions_sequence = [
        [1, 0],  # Move agent 0 to B, agent 1 waits
        [1, 0],  # Move agent 0 to C, agent 1 waits
        [0, 0],  # Agent 0 waits at C, agent 1 waits
        [1, 0],  # Move agent 0 to D, agent 1 waits
        [1, 0],  # Move agent 0 to E, agent 1 waits
        [0, 1],  # Agent 0 waits at E, move agent 1 to G
        [0, 1],  # Agent 0 waits at E, move agent 1 to F
        [0, 1],  # Agent 0 waits at E, move agent 1 to B
        [0, 1],  # Agent 0 waits at E, move agent 1 to C
        [0, 1],  # Agent 0 waits at E, move agent 1 to D
        [0, 1],  # Agent 0 waits at E, move agent 1 to E
    ]

    # Animate the sequence
    env.animate(actions_sequence)