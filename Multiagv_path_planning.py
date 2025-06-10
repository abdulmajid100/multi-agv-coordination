import numpy as np
import random
from grid_env_c import GridEnv

# Q-learning parameters
alpha = 0.05  # Learning rate
gamma = 0.99  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.2  # Minimum exploration rate
epsilon_decay = 0.997  # Decay rate for epsilon
num_episodes = 1000  # Number of episodes
alpha_start = 1
alpha_decay = 0.01

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
    previous_total_rewards = float('-inf')
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = np.zeros(env.num_agents)
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
        alpha = max(0.001, alpha_start / (1 + episode * alpha_decay))

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

        if np.sum(total_rewards) > np.sum(previous_total_rewards):
            epsilon = max(epsilon_min, epsilon * 0.99)  # Decay faster if improving
        else:
            epsilon = max(epsilon_min, epsilon * 0.999)  # Decay slower if not improving
            # Update previous total rewards
        previous_total_rewards = total_rewards

        # Print episode progress and update epsilon
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Rewards: {total_rewards}, Epsilon: {epsilon:.4f}")

    return q_tables


def main():
    grid_size = (10, 10)
    num_agents = 3
    obstacles = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (0, 8), (0, 9),
                 (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (1, 9),
                 (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8), (2, 9),
                 (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 7), (3, 8), (3, 9),
                 (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7), (4, 8), (4, 9),
                 (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
                 (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 6), (6, 7), (6, 8), (6, 9),
                 (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8), (7, 9),
                 (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 6), (8, 7), (8, 8), (8, 9),
                 (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 6), (9, 7), (9, 8), (9, 9)]
    goals = [(9, 5), (0, 5), (5, 9)]
    initial_positions = [(5, 9), (9, 5), (3, 6)]

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
