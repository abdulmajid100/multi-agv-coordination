import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

class GraphEnv:
    def __init__(self, graph, num_agents, goals, initial_positions=None):
        """
        Initialize a graph-based environment for multi-agent path finding.

        Args:
            graph: A networkx graph representing the environment
            num_agents: Number of agents in the environment
            goals: List of goal nodes for each agent
            initial_positions: Optional initial positions for agents
        """
        self.graph = graph
        self.num_agents = num_agents
        self.goals = goals
        self.initial_positions = initial_positions

        # Precompute shortest paths for each agent
        self.shortest_paths = {}
        for i, goal in enumerate(self.goals):
            self.shortest_paths[i] = nx.single_source_shortest_path(self.graph, goal)

        self.reset()

    def reset(self):
        """Reset the environment to its initial state."""
        if self.initial_positions is None:
            self.agents = [self._get_random_position() for _ in range(self.num_agents)]
        else:
            self.agents = self._validate_initial_positions(self.initial_positions)
        self.done = [False] * self.num_agents
        self.steps = 0
        return np.array(self.agents)

    def _validate_initial_positions(self, positions):
        """Validate that initial positions are valid nodes in the graph."""
        validated_positions = []
        for pos in positions:
            if pos not in self.graph.nodes():
                raise ValueError(f"Initial position {pos} is not a valid node in the graph.")
            validated_positions.append(pos)
        return validated_positions

    def step(self, actions=None):
        """
        Take a step in the environment based on agents' shortest paths.

        Args:
            actions: Ignored in this implementation; agents follow their shortest paths.

        Returns:
            tuple: (new_state, rewards, done, info)
        """
        rewards = np.zeros(self.num_agents)
        next_agents = self.agents.copy()

        for i, agent in enumerate(self.agents):
            if not self.done[i]:  # Only move agents that haven't reached their goals
                # Get the shortest path for this agent
                path = self.shortest_paths[i].get(agent, [])
                if len(path) > 1:  # If there's a next step in the path
                    next_position = path[1]
                else:
                    next_position = agent  # Stay in place if no path exists

                # Check if the agent reaches its goal
                if next_position == self.goals[i]:
                    rewards[i] += 1000  # Reward for reaching the goal
                    self.done[i] = True
                else:
                    rewards[i] -= 1  # Small penalty for each step taken

                next_agents[i] = next_position

        self.agents = next_agents
        self.steps += 1
        done = all(self.done) or self.steps >= 500  # Terminate after 500 steps or if all agents are done
        return np.array(self.agents), rewards, done, {'steps': self.steps}


    def _get_random_position(self):
        """Return a random node from the graph."""
        nodes = list(self.graph.nodes())
        return np.random.choice(nodes)

    def animate(self, actions_sequence=None):
        """Animate the agents moving through the graph."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw the graph
        pos = nx.spring_layout(self.graph, seed=42)  # Position nodes using spring layout
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(self.graph, pos, ax=ax)

        # Draw goals
        goal_nodes = nx.draw_networkx_nodes(self.graph, pos,
                                            nodelist=self.goals,
                                            node_size=500,
                                            node_color='green',
                                            ax=ax)

        # Initialize agents
        agent_colors = ['blue', 'red', 'purple', 'orange', 'brown']
        agent_nodes = []
        for i, agent_pos in enumerate(self.agents):
            color = agent_colors[i % len(agent_colors)]
            agent_node = nx.draw_networkx_nodes(self.graph, pos,
                                                nodelist=[agent_pos],
                                                node_size=300,
                                                node_color=color,
                                                ax=ax)
            agent_nodes.append(agent_node)

        # Title for animation
        title = ax.set_title('Step: 0')

        def update(frame):
            if frame == 0:
                # Reset environment for first frame
                self.reset()
                for i, agent_pos in enumerate(self.agents):
                    agent_nodes[i].set_offsets([pos[agent_pos]])
                title.set_text('Step: 0')
            else:
                # Take actions for this frame
                next_state, _, _, info = self.step()
                for i, agent_pos in enumerate(next_state):
                    agent_nodes[i].set_offsets([pos[agent_pos]])
                title.set_text(f"Step: {info['steps']}")

            return agent_nodes + [title]

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=500,
                                       interval=1000, blit=False, repeat=False)
        plt.tight_layout()

        # Return the animation object before showing the plot
        return anim
