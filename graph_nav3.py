import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

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
        return self.agents  # Return list of tuples instead of NumPy array

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
            if action in self.graph[agent] or action == agent:
                next_agents.append(action)
            else:
                next_agents.append(agent)  # Invalid action, stay in place

        # Check for collisions
        for i, next_agent in enumerate(next_agents):
            if any(next_agent == other_agent for j, other_agent in enumerate(next_agents) if j != i):
                collisions[i] = True

        # Update agent positions and calculate rewards
        for i, (agent, next_agent) in enumerate(zip(self.agents, next_agents)):
            if not self.done[i]:
                if collisions[i]:
                    rewards[i] -= 10  # Penalty for collision
                    next_agent = agent  # Stay in place in case of collision
                elif next_agent == self.goals[i]:
                    rewards[i] += 100  # Reward for reaching the goal
                    self.done[i] = True
                else:
                    rewards[i] -= 1  # Penalty for each step
                self.agents[i] = next_agent

        done = all(self.done)
        return self.agents, rewards, done, {}  # Return list of tuples

    def _get_random_node(self):
        return random.choice(list(self.graph.nodes))

    def animate(self, actions_sequence):
        fig, ax = plt.subplots(figsize=(6, 6))
        pos = {node: (node[0], node[1]) for node in self.graph.nodes}

        frames = []

        state = self.reset()
        for actions in actions_sequence:
            frames.append(state.copy())
            state, _, _, _ = self.step(actions)
        frames.append(state.copy())

        def update(frame):
            ax.clear()
            nx.draw(self.graph, pos, ax=ax, node_color='lightblue', edge_color='gray', with_labels=False)
            for i in range(self.num_agents):
                agent_pos = frames[frame][i]
                ax.plot(agent_pos[0], agent_pos[1], 'o', color=['blue', 'green'][i], markersize=12)
            return ax,

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000, blit=True)
        plt.show()

def get_possible_actions(graph, current_node):
    possible_actions = list(graph.neighbors(current_node))
    possible_actions.append(current_node)  # Include staying in place
    return possible_actions

def state_to_index(state, node_to_index, num_agents):
    index = 0
    num_nodes = len(node_to_index)
    factor = num_nodes ** (num_agents - 1)
    for i in range(num_agents):
        node = tuple(state[i])  # Ensure state[i] is a tuple
        node_idx = node_to_index[node]
        index += node_idx * factor
        if i < num_agents - 1:
            factor //= num_nodes
    return index

def q_learning(env, num_episodes, node_to_index):
    graph = env.graph
    num_agents = env.num_agents
    q_tables = [{} for _ in range(num_agents)]

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = np.zeros(num_agents)
        epsilon = max(1 - episode / 600, epsilon_min)
        alpha = max(0.001, alpha_start / (1 + episode * alpha_decay))

        while not done:
            actions = []
            state_index = state_to_index(state, node_to_index, num_agents)
            for i in range(num_agents):
                possible_actions = get_possible_actions(graph, state[i])
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(possible_actions)  # Explore
                else:
                    q_values = [q_tables[i].get((state_index, a), 0) for a in possible_actions]
                    max_q = max(q_values)
                    max_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
                    action = random.choice(max_actions)
                actions.append(action)

            next_state, rewards, done, _ = env.step(actions)
            total_rewards += rewards

            next_state_index = state_to_index(next_state, node_to_index, num_agents)
            for i in range(num_agents):
                possible_next_actions = get_possible_actions(graph, next_state[i])
                next_q_values = [q_tables[i].get((next_state_index, a), 0) for a in possible_next_actions]
                max_next_q = max(next_q_values) if next_q_values else 0

                td_target = rewards[i] + gamma * max_next_q
                current_q = q_tables[i].get((state_index, actions[i]), 0)
                td_error = td_target - current_q
                q_tables[i][(state_index, actions[i])] = current_q + alpha * td_error
            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Rewards: {total_rewards}, Epsilon: {epsilon:.4f}")
    return q_tables

# Q-learning parameters
alpha_start = 0.1  # Initial learning rate
gamma = 0.9  # Discount factor
epsilon_min = 0.01  # Minimum exploration rate
alpha_decay = 0.01  # Decay rate for alpha
num_episodes = 1200  # Number of episodes

def main():
    # Create a grid graph and remove obstacles
    graph = nx.grid_2d_graph(10, 10)
    obstacles = [(x, y) for x in range(10) for y in range(9)]  # Obstacles covering y = 0 to y = 8
    graph.remove_nodes_from(obstacles)

    num_agents = 2
    goals = [(0, 9), (9, 9)]
    initial_positions = [(9, 9), (0, 9)]

    env = GraphEnv(graph, num_agents, goals, initial_positions)

    node_list = list(graph.nodes)
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    q_tables = q_learning(env, num_episodes, node_to_index)

    print("Training completed.")

    # Testing trained agents
    state = env.reset()
    done = False
    actions_list = []
    while not done:
        actions = []
        state_index = state_to_index(state, node_to_index, num_agents)
        for i in range(num_agents):
            possible_actions = get_possible_actions(graph, state[i])
            q_values = [q_tables[i].get((state_index, a), 0) for a in possible_actions]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
            action = random.choice(max_actions)
            actions.append(action)
        actions_list.append(actions)
        state, _, done, _ = env.step(actions)
        print(f"Agents' positions: {state}")

    env.animate(actions_list)

if __name__ == "__main__":
    main()