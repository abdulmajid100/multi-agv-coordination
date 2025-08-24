import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from grid_env_c import GridEnv
from collections import deque

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create the GridEnv environment
grid_size = (5, 5)
num_agents = 3  # Using multiple agents
obstacles = [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3), (4, 1), (4, 2), (4, 3)]
goals = [(4, 0), (4, 4), (0, 0)]  # One goal for each agent
initial_positions = [(0, 4), (2, 0), (2, 4)]  # Starting positions for each agent
env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)

# Define the online neural network models for each agent
# The input size is 2*num_agents (x, y coordinates of all agents)
# The output size is 5 (5 possible actions: left, up, right, down, wait)
models = []
targets = []

for i in range(num_agents):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="elu", input_shape=(2*num_agents,)),
        tf.keras.layers.Dense(64, activation="elu"),
        tf.keras.layers.Dense(32, activation="elu"),
        tf.keras.layers.Dense(5)  # No activation, raw Q-values
    ])

    # Create a target network (clone of the online network)
    target = tf.keras.models.clone_model(model)
    target.set_weights(model.get_weights())  # Copy the weights

    models.append(model)
    targets.append(target)

# Define the epsilon-greedy policy for multiple agents
def epsilon_greedy_policy(state, agent_idx, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(5)  # Random action (0-4)
    else:
        # Flatten the state for all agents into a single vector
        flat_state = np.array([coord for agent_pos in state for coord in agent_pos])

        # Convert state to tensor and get Q-values for the specific agent
        state_tensor = tf.convert_to_tensor(flat_state, dtype=tf.float32)
        Q_values = models[agent_idx].predict(tf.expand_dims(state_tensor, axis=0), verbose=0)[0]
        return np.argmax(Q_values)  # Optimal action according to the DQN

# Create replay buffers for each agent
replay_buffers = [deque(maxlen=2000) for _ in range(num_agents)]

# Function to play one step and store experience for all agents
def play_one_step(env, state, epsilon):
    actions = []

    # Select actions for all agents using epsilon-greedy policy
    for i in range(num_agents):
        action = epsilon_greedy_policy(state, i, epsilon)
        actions.append(action)

    # Take actions in environment
    next_state, rewards, done, info = env.step(actions)

    # Store experiences in replay buffers for each agent
    flat_state = np.array([coord for agent_pos in state for coord in agent_pos])
    flat_next_state = np.array([coord for agent_pos in next_state for coord in agent_pos])

    for i in range(num_agents):
        replay_buffers[i].append((flat_state, actions[i], rewards[i], flat_next_state, done))

    return next_state, rewards, done, info

# Training parameters
batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
loss_fn = tf.keras.losses.MeanSquaredError()

# Function to sample experiences from replay buffer for a specific agent
def sample_experiences(agent_idx, batch_size):
    if len(replay_buffers[agent_idx]) < batch_size:
        # If not enough experiences, return None
        return None

    indices = np.random.randint(len(replay_buffers[agent_idx]), size=batch_size)
    batch = [replay_buffers[agent_idx][index] for index in indices]
    states = np.array([experience[0] for experience in batch])
    actions = np.array([experience[1] for experience in batch])
    rewards = np.array([experience[2] for experience in batch])
    next_states = np.array([experience[3] for experience in batch])
    dones = np.array([experience[4] for experience in batch])
    return states, actions, rewards, next_states, dones

# Training step function for all agents
def training_step(batch_size):
    # Create optimizers for each agent
    optimizers = [tf.keras.optimizers.Nadam(learning_rate=1e-2) for _ in range(num_agents)]

    # Train each agent separately
    for agent_idx in range(num_agents):
        # Sample experiences for this agent
        experiences = sample_experiences(agent_idx, batch_size)
        if experiences is None:
            continue  # Skip if not enough experiences

        states, actions, rewards, next_states, dones = experiences

        # Compute target Q-values using the target network for this agent
        next_Q_values = targets[agent_idx].predict(next_states, verbose=0)
        max_next_Q_values = next_Q_values.max(axis=1)
        runs = 1.0 - dones  # episode is not done
        target_Q_values = rewards + runs * discount_factor * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)

        # Create mask for actions taken
        mask = tf.one_hot(actions, 5)  # 5 possible actions

        # Compute loss and update model for this agent
        with tf.GradientTape() as tape:
            all_Q_values = models[agent_idx](states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

        # Apply gradients
        grads = tape.gradient(loss, models[agent_idx].trainable_variables)
        optimizers[agent_idx].apply_gradients(zip(grads, models[agent_idx].trainable_variables))

# Training loop
num_episodes = 600
max_steps_per_episode = 200

# Track rewards for visualization (one list per agent)
episode_rewards = [[] for _ in range(num_agents)]
total_episode_rewards = []

for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = np.zeros(num_agents)

    for step in range(max_steps_per_episode):
        # Decrease epsilon over time for exploration-exploitation trade-off
        epsilon = max(1 - episode / 500, 0.01)

        # Play one step
        obs, rewards, done, info = play_one_step(env, obs, epsilon)
        episode_reward += rewards  # Add rewards for all agents

        # Train after collecting enough experiences
        enough_experiences = all(len(replay_buffers[i]) >= batch_size for i in range(num_agents))
        if enough_experiences:
            training_step(batch_size)

        if done:
            break

    # Track episode rewards for each agent
    for i in range(num_agents):
        episode_rewards[i].append(episode_reward[i])
    total_episode_rewards.append(np.sum(episode_reward))

    # Update target networks periodically
    if episode > 50 and episode % 50 == 0:
        for i in range(num_agents):
            targets[i].set_weights(models[i].get_weights())

    # Print progress
    if episode % 10 == 0:
        print(f"\rEpisode: {episode + 1}/{num_episodes}, "
              f"Rewards: {episode_reward}, "
              f"Total: {np.sum(episode_reward):.1f}, "
              f"Epsilon: {epsilon:.3f}", end="")

print("\nTraining completed.")

# Plot rewards over episodes
plt.figure(figsize=(12, 8))

# Plot individual agent rewards
for i in range(num_agents):
    plt.subplot(num_agents + 1, 1, i + 1)
    plt.plot(episode_rewards[i], label=f'Agent {i}')
    plt.ylabel(f'Agent {i} Reward')
    plt.grid(True)
    plt.legend()

# Plot total rewards
plt.subplot(num_agents + 1, 1, num_agents + 1)
plt.plot(total_episode_rewards, label='Total', color='black')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Define a policy function that uses the trained models
def grid_policy(obs, agent_idx):
    # Flatten the state for all agents
    flat_state = np.array([coord for agent_pos in obs for coord in agent_pos])

    # Get Q-values from the agent's model
    state_tensor = tf.convert_to_tensor(flat_state, dtype=tf.float32)
    Q_values = models[agent_idx].predict(tf.expand_dims(state_tensor, axis=0), verbose=0)[0]

    return int(np.argmax(Q_values))

# Test the trained policies
obs = env.reset()
done = False
actions_list = []

print("Testing the trained policies:")
step = 0
while not done and step < 100:  # Add step limit to prevent infinite loops
    actions = []

    # Get actions for all agents
    for i in range(num_agents):
        action = grid_policy(obs, i)
        actions.append(action)

    actions_list.append(actions)

    print(f"Step {step}:")
    for i in range(num_agents):
        print(f"Agent {i} position: {obs[i]}, Action: {actions[i]}")

    obs, rewards, done, info = env.step(actions)
    step += 1

    print(f"Rewards: {rewards}")
    print("-" * 30)

print(f"\nSimulation completed in {step} steps.")
print("Final positions:")
for i in range(num_agents):
    print(f"Agent {i}: {obs[i]}")
    print(f"Goal reached: {env._is_goal(obs[i], i)}")

# Animate the agents' paths
env.animate(actions_list)
