import numpy as np
from grid_env import GridEnv


class QLearningAgent:
    def __init__(self, grid_size, num_agents, action_space, alpha=0.001, gamma=0.9, epsilon=0.1):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table with zeros
        self.q_table = np.zeros(grid_size + (num_agents, action_space))

    def choose_action(self, state):
        actions = []
        for i in range(self.num_agents):
            if np.random.rand() < self.epsilon:
                actions.append(np.random.randint(self.action_space))  # Explore
            else:
                actions.append(np.argmax(self.q_table[state[i]][i]))  # Exploit
        return actions

    def update_q_table(self, state, actions, rewards, next_state):
        for i, action in enumerate(actions):
            current_state = state[i]
            next_state_agent = next_state[i]

            if not (0 <= action < self.action_space):
                raise ValueError(
                    f"Action {action} is out of bounds. Action must be between 0 and {self.action_space - 1}.")

            # Get the best next action from the Q-table
            best_next_action = np.argmax(self.q_table[next_state_agent][i])
            td_target = rewards[i] + self.gamma * self.q_table[next_state_agent][i][best_next_action]
            td_error = td_target - self.q_table[current_state][i][action]
            self.q_table[current_state][i][action] += self.alpha * td_error


def train_q_learning(env, agent, episodes=1000):
    for episode in range(episodes):
        state = tuple(env.reset())
        total_reward = 0
        done = False
        while not done:
            actions = agent.choose_action(state)
            next_state, rewards, done, _ = env.step(actions)
            total_reward += sum(rewards)
            next_state = tuple(next_state)
            agent.update_q_table(state, actions, rewards, next_state)
            state = next_state
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")


# Example usage
grid_size = (5, 5)
num_agents = 2
obstacles = [(2, 2)]
goals = [(4, 4), (0, 0)]

env = GridEnv(grid_size, num_agents, obstacles, goals)
action_space = 5  # 5 possible actions (0=left, 1=up, 2=right, 3=down, 4=wait)

agent = QLearningAgent(grid_size, num_agents, action_space)

train_q_learning(env, agent)
