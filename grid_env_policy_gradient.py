import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from grid_env_c import GridEnv

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create the GridEnv environment
grid_size = (5, 5)
num_agents = 3  # Increased to 3 agents
obstacles = [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3)]
goals = [(4, 0), (4, 4), (0, 0)]  # One goal per agent
initial_positions = [(0, 4), (2, 0), (2, 4)]  # Starting positions for each agent
env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)

# Define a neural network model for each agent
# The input size is 2 (x, y coordinates of the agent)
# The output size is 5 (5 possible actions: left, up, right, down, wait)
models = []
for _ in range(num_agents):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=(2,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax"),  # 5 actions
    ])
    models.append(model)

# Define the loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizers = [tf.keras.optimizers.Nadam(learning_rate=0.01) for _ in range(num_agents)]

def play_one_step(env, obs, models, loss_fn):
    actions = []
    all_grads = []

    # Process each agent
    for agent_idx in range(num_agents):
        with tf.GradientTape() as tape:
            # Convert observation to tensor for this agent
            obs_tensor = tf.convert_to_tensor(obs[agent_idx], dtype=tf.float32)

            # Get action probabilities from the agent's model
            action_probs = models[agent_idx](tf.expand_dims(obs_tensor, axis=0))[0]

            # Sample an action based on the probabilities
            action = tf.random.categorical(tf.math.log([action_probs]), num_samples=1)[0, 0]
            actions.append(int(action))

            # Create a one-hot target for the selected action
            y_target = tf.one_hot(action, 5)

            # Compute the loss
            loss = loss_fn(tf.expand_dims(action, axis=0), tf.expand_dims(action_probs, axis=0))

        # Compute gradients for this agent
        grads = tape.gradient(loss, models[agent_idx].trainable_variables)
        all_grads.append(grads)

    # Take a step in the environment with all agents' actions
    next_obs, rewards, done, info = env.step(actions)

    return next_obs, rewards, done, info, all_grads

def play_multiple_episodes(env, n_episodes, n_max_steps, models, loss_fn):
    # For each agent, we'll track rewards and gradients across episodes
    all_rewards = [[] for _ in range(num_agents)]
    all_grads = [[] for _ in range(num_agents)]

    for episode in range(n_episodes):
        # For each agent, track rewards and gradients in this episode
        current_rewards = [[] for _ in range(num_agents)]
        current_grads = [[] for _ in range(num_agents)]
        obs = env.reset()

        for step in range(n_max_steps):
            obs, rewards, done, info, all_agent_grads = play_one_step(env, obs, models, loss_fn)

            # Store rewards and gradients for each agent
            for agent_idx in range(num_agents):
                current_rewards[agent_idx].append(rewards[agent_idx])
                current_grads[agent_idx].append(all_agent_grads[agent_idx])

            if done:
                break

        # Add this episode's data to our overall tracking
        for agent_idx in range(num_agents):
            all_rewards[agent_idx].append(current_rewards[agent_idx])
            all_grads[agent_idx].append(current_grads[agent_idx])

    return all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    # Process rewards for each agent separately
    all_discounted_rewards = []

    for agent_rewards in all_rewards:
        agent_discounted_rewards = [discount_rewards(rewards, discount_factor)
                                   for rewards in agent_rewards]
        all_discounted_rewards.append(agent_discounted_rewards)

    # Normalize across all agents for better stability
    flat_rewards = np.concatenate([np.concatenate(agent_rewards) for agent_rewards in all_discounted_rewards])
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    if reward_std == 0:
        return all_discounted_rewards

    # Normalize rewards for each agent
    normalized_rewards = []
    for agent_rewards in all_discounted_rewards:
        normalized_agent_rewards = [(rewards - reward_mean) / reward_std
                                   for rewards in agent_rewards]
        normalized_rewards.append(normalized_agent_rewards)

    return normalized_rewards

# Training parameters
n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_factor = 0.95

# Training loop
for iteration in range(n_iterations):
    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, models, loss_fn)

    # Display progress - sum rewards across all agents
    total_rewards = sum(sum(map(sum, agent_rewards)) for agent_rewards in all_rewards)
    print(f"\rIteration: {iteration + 1}/{n_iterations}, "
          f"mean rewards: {total_rewards / (n_episodes_per_update * num_agents):.1f}", end="")

    # Compute discounted and normalized rewards for all agents
    all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

    # For each agent, compute and apply gradients
    for agent_idx in range(num_agents):
        # Compute mean gradients for this agent
        all_mean_grads = []
        for var_index in range(len(models[agent_idx].trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[agent_idx][episode_index][step][var_index]
                 for episode_index, final_rewards in enumerate(all_final_rewards[agent_idx])
                    for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)

        # Apply gradients for this agent
        optimizers[agent_idx].apply_gradients(zip(all_mean_grads, models[agent_idx].trainable_variables))

print("\nTraining completed.")

# Define a policy function that uses the trained models for all agents
def grid_policy(obs, models):
    actions = []
    for agent_idx in range(num_agents):
        obs_tensor = tf.convert_to_tensor(obs[agent_idx], dtype=tf.float32)
        action_probs = models[agent_idx](tf.expand_dims(obs_tensor, axis=0))[0]
        actions.append(int(tf.argmax(action_probs)))
    return actions

# Test the trained policy
obs = env.reset()
done = False
actions_list = []

print("Testing the trained policy:")
step = 0
while not done and step < 100:  # Add step limit to prevent infinite loops
    actions = grid_policy(obs, models)
    actions_list.append(actions)

    # Print positions and actions for all agents
    for agent_idx in range(num_agents):
        print(f"Agent {agent_idx} position: {obs[agent_idx]}, Action: {actions[agent_idx]}")

    obs, rewards, done, info = env.step(actions)
    step += 1
    print(f"Step {step}, Rewards: {rewards}")
    print("-" * 30)

print(f"\nSimulation completed in {step} steps.")
print(f"Final positions: {obs}")
print(f"Goals reached: {all(env._is_goal(agent, i) for i, agent in enumerate(obs))}")

# Animate the agents' paths
env.animate(actions_list)
