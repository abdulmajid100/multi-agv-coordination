# graph_env.py

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

        # Precompute shortest path distances for reward shaping
        self.shortest_distances = {}
        for i, goal in enumerate(self.goals):
            self.shortest_distances[i] = dict(nx.single_source_shortest_path_length(self.graph, goal))

        self.reset()

    def reset(self):
        """Reset the environment to its initial state."""
        if self.initial_positions is None:
            self.agents = [self._get_random_position() for _ in range(self.num_agents)]
            #print(self.agents)
        else:
            self.agents = self._validate_initial_positions(self.initial_positions)
            #print(self.agents)
        self.done = [False] * self.num_agents
        self.steps = 0
        # Initialize previous positions as None to indicate no previous position yet
        self.previous_positions = [None] * self.num_agents
        return np.array(self.agents)

    def _validate_initial_positions(self, positions):
        """Validate that initial positions are valid nodes in the graph."""
        validated_positions = []
        for pos in positions:
            if pos not in self.graph.nodes():
                raise ValueError(f"Initial position {pos} is not a valid node in the graph.")
            validated_positions.append(pos)
        return validated_positions

    def step(self, actions):
        """
        Take a step in the environment based on agents' actions.

        Args:
            actions: List of actions for each agent

        Returns:
            tuple: (new_state, rewards, done, info)
        """
        rewards = np.zeros(self.num_agents)
        collisions = [False] * self.num_agents

        # Initialize next_agents as a copy of current agents
        next_agents = self.agents.copy()

        # Compute next positions based on actions
        proposed_positions = []
        for agent_idx, (agent, action) in enumerate(zip(self.agents, actions)):
            proposed_positions.append(self._apply_action(agent, action, agent_idx))

        # Check for collisions (agents trying to occupy the same node)
        for i, next_pos in enumerate(proposed_positions):
            if any(next_pos == other_pos for j, other_pos in enumerate(proposed_positions) if j != i):
                collisions[i] = True

        # Update agent positions and calculate rewards
        for i, (agent, proposed_pos) in enumerate(zip(self.agents, proposed_positions)):
            if not self.done[i]:  # Only update if the agent is not done
                if collisions[i]:
                    rewards[i] -= 1000  # Penalty for collision
                    # Agent stays in place (next_agents[i] already equals agent)
                elif proposed_pos == self.goals[i]:
                    rewards[i] += 100  # Reward for reaching the goal
                    self.done[i] = True  # Mark as done
                    # Update previous position before moving
                    self.previous_positions[i] = agent
                    next_agents[i] = proposed_pos  # Move to goal
                else:
                    # Reward shaping based on shortest path distance to goal
                    current_distance = self._distance(agent, i)
                    new_distance = self._distance(proposed_pos, i)
                    distance_reward = (current_distance - new_distance) * 100 - 50 # Reward for getting closer
                    rewards[i] += distance_reward
                    # Update previous position before moving
                    self.previous_positions[i] = agent
                    next_agents[i] = proposed_pos  # Move to new position
            # If agent is done, it stays at its current position (next_agents[i] already equals agent)

        self.agents = next_agents
        self.steps += 1
        done = all(self.done) or self.steps >= 500  # Terminate after 500 steps or if all agents are done

        return np.array(self.agents), rewards, done, {'steps': self.steps}

    def _distance(self, node, agent_idx):
        """Return the shortest path distance from the node to the agent's goal."""
        agent_distances = self.shortest_distances[agent_idx]
        if node in agent_distances:
            return agent_distances[node]
        else:
            # If there's no path to the goal, return a large value
            return float('inf')

    def _get_random_position(self):
        """Return a random node from the graph."""
        nodes = list(self.graph.nodes())
        return np.random.choice(nodes)

    def _apply_action(self, position, action, agent_idx):
        """
        Apply an action to move from the current position.

        Args:
            position: Current node position
            action: Action to take (index representing which neighbor to move to)
            agent_idx: Index of the agent for goal checking

        Returns:
            new_position: Next node position
        """
        neighbors = list(self.graph.neighbors(position))

        # Filter out the previous position to prevent backward movement
        if self.previous_positions[agent_idx] is not None:
            # Remove the previous position from neighbors if it exists
            filtered_neighbors = [n for n in neighbors if n != self.previous_positions[agent_idx]]
            # If there are no valid neighbors after filtering, allow all neighbors
            if filtered_neighbors:
                neighbors = filtered_neighbors

        # Special action for staying in place
        if action == len(neighbors):
            return position

        # Move to a neighboring node
        if action < len(neighbors):
            return neighbors[action]
        else:
            # Invalid action, stay in place
            return position

    def get_valid_actions(self, agent_idx):
        """Get the number of valid actions for an agent in its current position."""
        position = self.agents[agent_idx]
        neighbors = list(self.graph.neighbors(position))

        # Filter out the previous position to prevent backward movement
        if self.previous_positions[agent_idx] is not None:
            # Remove the previous position from neighbors if it exists
            filtered_neighbors = [n for n in neighbors if n != self.previous_positions[agent_idx]]
            # If there are no valid neighbors after filtering, allow all neighbors
            if filtered_neighbors:
                neighbors = filtered_neighbors

        # Number of neighbors plus one for the "stay" action
        return len(neighbors) + 1

    def animate(self, actions_sequence):
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
                actions = actions_sequence[frame - 1]
                next_state, _, _, info = self.step(actions)
                for i, agent_pos in enumerate(next_state):
                    agent_nodes[i].set_offsets([pos[agent_pos]])
                title.set_text(f"Step: {info['steps']}")

            return agent_nodes + [title]

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(actions_sequence) + 1,
                                       interval=1000, blit=False, repeat=False)
        plt.tight_layout()

        # Return the animation object before showing the plot
        # This ensures the caller can save a reference to it
        return anim

    def execute_actions(self, actions_sequence):
        """Execute a sequence of actions and return the history of states."""
        state_history = []
        self.reset()
        state_history.append(self.agents.copy())

        for actions in actions_sequence:
            next_state, _, done, _ = self.step(actions)
            state_history.append(next_state.copy())
            if done:
                break

        return state_history
