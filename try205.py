import numpy as np
import random
from vac_env import GridEnv  # Ensure this imports your GridEnv class correctly

# Q-learning parameters
alpha_start = 1
alpha_decay = 0.01
gamma = 0.9
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
num_episodes = 10000

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

        previous_total_rewards = total_rewards

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Total Rewards: {total_rewards}, Epsilon: {epsilon:.4f}")

    return q_tables

def main():
    grid_size = (7, 7)
    num_agents = 1
    obstacles = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
                 (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
                 (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)
                 ]
    goals = []
    dirt_density = 0.2  # Adjust as needed
    initial_positions = [(6, 6)]

    env = GridEnv(grid_size, num_agents, obstacles, goals, dirt_density, initial_positions)
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
        #print(f"Agents' positions: {state}")
        state, _, done, _ = env.step(actions)

    env.animate(actions_list)  # For animation, adjust actions list as needed

if __name__ == "__main__":
    main()
