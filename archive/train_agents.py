import numpy as np
from grid_env import GridEnv
from q_learning import QLearningAgent


def train_agents(env, episodes=1000):
    state_size = tuple(env.grid_size)  # Grid size as state space
    action_size = 5  # Number of possible actions

    # Initialize two Q-learning agents
    agents = [QLearningAgent(state_size, action_size) for _ in range(env.num_agents)]

    for episode in range(episodes):
        state = env.reset()
        done = [False] * env.num_agents
        episode_actions = []
        episode_rewards = np.zeros(env.num_agents)  # Initialize rewards for this episode

        while not done:
            actions = [agent.choose_action(tuple(pos)) for agent, pos in zip(agents, state)]
            episode_actions.append(actions)
            next_state, rewards, done, _ = env.step(actions)

            # Aggregate rewards
            episode_rewards += rewards

            for i, agent in enumerate(agents):
                agent.update_q_table(tuple(state[i]), actions[i], rewards[i], tuple(next_state[i]))
                agent.decay_epsilon()
                print(f"Agent {i} Q-Table before update: {agent.q_table}")
                print(f"Agent {i} Q-Table after update: {agent.q_table}")
            state = next_state

        # Print rewards for the episode
        print(f"Episode {episode}/{episodes} completed. Rewards: {episode_rewards}")

    return agents, episode_actions
