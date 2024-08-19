import numpy as np
import random
from grid_env import GridEnv

# Q-learning parameters
alpha = 0.01  # Learning rate
gamma = 0.9  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.99  # Decay rate for epsilon
num_episodes = 5000  # Number of episodes


# Define state-action space
def state_to_index(state, grid_size):
    num_agents = len(state)
    flat_state = [coord for agent in state for coord in agent]
    index = 0
    factor = (grid_size[0] * grid_size[1]) ** (num_agents - 1)

    for i in range(num_agents):
        x, y = flat_state[2 * i], flat_state[2 * i + 1]
        index += (x * grid_size[1] + y) * factor
        factor //= (grid_size[0] * grid_size[1])

    return index



def action_to_index(action):
    return action


def q_learning(env, num_episodes):
    current_step = 0
    q_tables = [np.zeros(((env.grid_size[0] * env.grid_size[1]) ** env.num_agents, 5)) for _ in range(env.num_agents)]
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = np.zeros(env.num_agents)
        alpha = 0.05 / (1 + episode * 0.005)
        epsilon = max(1 - episode / 4500, 0.01)

        while not done:
            actions = []
            for i in range(env.num_agents):
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, 4)  # Explore
                else:
                    state_index = state_to_index(state, env.grid_size)
                    action = np.argmax(q_tables[i][state_index])  # Exploit

                actions.append(action)

            next_state, rewards, done, info = env.step(actions)
            current_step = info['steps']
            total_rewards += rewards

            for i in range(env.num_agents):
                state_index = state_to_index(state, env.grid_size)
                next_state_index = state_to_index(next_state, env.grid_size)

                best_next_action = np.argmax(q_tables[i][next_state_index])
                td_target = rewards[i] + gamma * q_tables[i][next_state_index][best_next_action]
                td_error = td_target - q_tables[i][state_index][actions[i]]

                q_tables[i][state_index][actions[i]] += alpha * td_error

            state = next_state

        # Print episode progress and update epsilon
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Rewards: {total_rewards}, Epsilon: {epsilon:.4f}")

    return q_tables


def main():
    grid_size = (5, 5)
    num_agents = 3
    obstacles = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1),
                 (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (4, 3)]
    goals = [(4, 4), (0, 4), (2, 4)]
    initial_positions = [(0, 4), (4, 4), (2, 2)]

    env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)
    q_tables = q_learning(env, num_episodes)

    print("Training completed.")

    # Testing trained agents
    state = env.reset()
    done = False
    actions_list = []
    while not done:
        actions = []
        for i in range(num_agents):
            state_index = state_to_index(state, grid_size)
            action = np.argmax(q_tables[i][state_index])
            actions.append(action)
        actions_list.append(actions)
        print(f"Agents' positions: {state}")
        state, _, done, _ = env.step(actions)

    env.animate(actions_list)  # For animation, adjust actions list as needed


if __name__ == "__main__":
    main()
