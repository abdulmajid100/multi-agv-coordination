import numpy as np
import random

# Optional: Switch backend to address the AttributeError
# import matplotlib
# matplotlib.use('TkAgg')  # Uncomment if necessary

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
NUM_AGENTS = 3
NUM_EPISODES = 5000
MAX_STEPS = 20
ALPHA = 0.1         # Learning rate
GAMMA = 0.9         # Discount factor
EPSILON = 0.1       # Exploration rate

class Environment:
    def __init__(self):
        # Define the graph
        self.graph, self.node_positions = self.create_graph()
        # Define agent paths (predefined)
        self.agent_paths = self.define_agent_paths()
        # Define agent start times
        self.agent_start_times = [0, 2, 4]
        # Initialize agent positions
        self.agent_positions = {}
        # Corridor nodes
        self.corridor_nodes = ['C1', 'C2']
        # Corridor occupancy status
        self.corridor_occupied = False

    def create_graph(self):
        # Simplified graph representation
        # Nodes: Start (S1, S2), Corridor Entrance (E1, E2), Corridor Nodes (C1, C2), Goal (G1, G2)
        graph = {
            'S1': ['E1'],
            'E1': ['C1', 'S1'],  # Entrance for Agent 0
            'C1': ['C2', 'E1'],
            'C2': ['C1', 'E2'],
            'E2': ['C2', 'S2'],  # Entrance for Agents 1 and 2
            'S2': ['E2'],
            'G1': [],
            'G2': []
        }
        # Node positions for visualization
        node_positions = {
            'S1': (0, 0),
            'E1': (1, 0),
            'C1': (2, 0),
            'C2': (3, 0),
            'E2': (4, 0),
            'S2': (5, 1),
            'G1': (5, 0),
            'G2': (0, 1),
        }
        return graph, node_positions

    def define_agent_paths(self):
        # Predefined paths for each agent
        paths = {
            0: ['S1', 'E1', 'C1', 'C2', 'E2', 'G1'],  # Agent 0 path
            1: ['S2', 'E2', 'C2', 'C1', 'E1', 'G2'],  # Agent 1 path
            2: ['S2', 'E2', 'C2', 'C1', 'E1', 'G2']   # Agent 2 path
        }
        return paths

    def reset(self):
        self.agent_positions = {}
        self.corridor_occupied = False
        # Initialize agents at their start positions
        for agent_id in range(NUM_AGENTS):
            self.agent_positions[agent_id] = 'S' + str(1 if agent_id == 0 else 2)
        return self.get_state()

    def get_state(self):
        # State includes positions of agents and corridor status
        state = {
            'positions': self.agent_positions.copy(),
            'corridor_occupied': self.corridor_occupied
        }
        return state

    def step(self, agent_id, action, current_step):
        """
        Perform an action for the agent and update the environment.
        Actions: 'move' or 'wait'
        """
        reward = 0
        done = False

        # Get the agent's current position
        current_pos = self.agent_positions[agent_id]
        path = self.agent_paths[agent_id]
        idx = path.index(current_pos)

        # Agents cannot act before their start time
        if current_step < self.agent_start_times[agent_id]:
            return self.get_state(), 0, False

        if action == 'move':
            # Check if agent has reached the goal
            if current_pos == path[-1]:
                # Agent has already reached the goal
                return self.get_state(), 0, True
            # Determine the next position
            next_pos = path[idx + 1]

            # Check if the next position is a corridor node
            if next_pos in self.corridor_nodes:
                if self.corridor_occupied:
                    # Collision detected
                    reward -= 10  # Negative reward for collision
                    done = True
                else:
                    # Occupy the corridor
                    self.corridor_occupied = True
                    self.agent_positions[agent_id] = next_pos
                    reward += 1  # Small positive reward for moving forward
            else:
                # If leaving the corridor, free it up
                if current_pos in self.corridor_nodes and next_pos not in self.corridor_nodes:
                    self.corridor_occupied = False
                # Move to the next position
                self.agent_positions[agent_id] = next_pos
                # Check if goal is reached
                if next_pos == path[-1]:
                    reward += 10  # Positive reward for reaching the goal
                    done = True
                else:
                    reward += 1  # Small positive reward for moving forward
        elif action == 'wait':
            reward -= 0.1  # Small penalty for waiting to encourage efficiency
            # Agent stays in the same position
        else:
            # Invalid action
            raise ValueError("Invalid action")

        return self.get_state(), reward, done

# Q-table initialization
Q_tables = [{} for _ in range(NUM_AGENTS)]  # One Q-table per agent

# Actions
actions = ['move', 'wait']

env = Environment()

# Training Loop
for episode in range(NUM_EPISODES):
    state = env.reset()
    done_flags = [False for _ in range(NUM_AGENTS)]

    # Simulate each time step
    for current_step in range(MAX_STEPS):
        for agent_id in range(NUM_AGENTS):
            if done_flags[agent_id]:
                continue  # Skip if the agent has finished

            # Construct a state representation for the agent
            agent_state = (state['positions'][agent_id], state['corridor_occupied'])
            # Initialize Q-values for unseen states
            if agent_state not in Q_tables[agent_id]:
                Q_tables[agent_id][agent_state] = dict.fromkeys(actions, 0.0)

            # Epsilon-greedy action selection
            if random.uniform(0, 1) < EPSILON:
                action = random.choice(actions)
            else:
                # Select the action with the highest Q-value
                action_values = Q_tables[agent_id][agent_state]
                max_value = max(action_values.values())
                max_actions = [a for a, v in action_values.items() if v == max_value]
                action = random.choice(max_actions)

            # Take the action
            next_state, reward, done = env.step(agent_id, action, current_step)

            # Update Q-value
            next_agent_state = (next_state['positions'][agent_id], next_state['corridor_occupied'])
            # Initialize Q-values for unseen next states
            if next_agent_state not in Q_tables[agent_id]:
                Q_tables[agent_id][next_agent_state] = dict.fromkeys(actions, 0.0)

            old_value = Q_tables[agent_id][agent_state][action]
            future_rewards = Q_tables[agent_id][next_agent_state].values()
            max_future_reward = max(future_rewards) if future_rewards else 0
            new_value = old_value + ALPHA * (reward + GAMMA * max_future_reward - old_value)
            Q_tables[agent_id][agent_state][action] = new_value

            # Update the state and done flag
            state = next_state
            if done:
                done_flags[agent_id] = True

        # Check if all agents are done
        if all(done_flags):
            break

# After training, simulate an episode to see the learned behavior and animate it
state = env.reset()
done_flags = [False for _ in range(NUM_AGENTS)]
agent_positions_over_time = []

print("Simulation after training:")
for current_step in range(MAX_STEPS):
    print(f"\nTime Step {current_step}")
    agent_positions_snapshot = {}
    for agent_id in range(NUM_AGENTS):
        if done_flags[agent_id]:
            continue  # Skip if the agent has finished

        agent_state = (state['positions'][agent_id], state['corridor_occupied'])
        if agent_state not in Q_tables[agent_id]:
            action = 'wait'  # Default action if state is unknown
        else:
            # Select the action with the highest Q-value
            action_values = Q_tables[agent_id][agent_state]
            max_value = max(action_values.values())
            max_actions = [a for a, v in action_values.items() if v == max_value]
            action = random.choice(max_actions)

        # Take the action
        next_state, reward, done = env.step(agent_id, action, current_step)
        state = next_state
        if done:
            done_flags[agent_id] = True
            print(f"Agent {agent_id} has reached the goal.")
        else:
            print(f"Agent {agent_id} at position {state['positions'][agent_id]} took action '{action}'.")

        # Record the agent's position for animation
        agent_positions_snapshot[agent_id] = state['positions'][agent_id]

    agent_positions_over_time.append(agent_positions_snapshot)

    # Check if all agents are done
    if all(done_flags):
        print("All agents have reached their goals.")
        break

# Visualization using matplotlib
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(8, 6))

# Draw the graph
for node, pos in env.node_positions.items():
    ax.plot(pos[0], pos[1], 'ko', markersize=10)
    ax.text(pos[0] + 0.1, pos[1] + 0.1, node, fontsize=12)

# Draw edges
for node, edges in env.graph.items():
    for edge in edges:
        start_pos = env.node_positions[node]
        end_pos = env.node_positions[edge]
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-')

# Agent colors
agent_colors = ['red', 'blue', 'green']

# Initialize agent plots
agent_plots = []
for agent_id in range(NUM_AGENTS):
    pos = env.node_positions[env.agent_positions[agent_id]]
    plot, = ax.plot([pos[0]], [pos[1]], 'o', color=agent_colors[agent_id], markersize=15)
    agent_plots.append(plot)

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 2)
ax.set_title('Agent Navigation Animation')
ax.axis('off')

def update(frame):
    # Update agent positions
    if frame < len(agent_positions_over_time):
        positions = agent_positions_over_time[frame]
        for agent_id, plot in enumerate(agent_plots):
            if agent_id in positions:
                node = positions[agent_id]
                pos = env.node_positions[node]
                plot.set_data([pos[0]], [pos[1]])  # Wrap pos[0] and pos[1] in lists
            else:
                # Agent has reached the goal
                plot.set_data([], [])
    return agent_plots

ani = FuncAnimation(fig, update, frames=len(agent_positions_over_time), blit=True, interval=1000)

plt.show()