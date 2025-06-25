import tensorflow as tf
import numpy as np
import numpy as np
import random
from grid_env_c import GridEnv

# Q-learning parameters
alpha = 0.05  # Learning rate
gamma = 0.9  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon
num_episodes = 1000  # Number of episodes
alpha_start = 0.1
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
tf.random.set_seed(42)  # extra code – ensures reproducibility on the CPU

input_shape = [2]  # == env.observation_space.shape
n_outputs = 5  # == env.action_space.n

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, 4)  # random action
        return action
    else:
        # Ensure state has the correct shape for model prediction
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        elif len(state.shape) > 2:
            # If state has more than 2 dimensions, reshape it to 2D
            state = state.reshape(state.shape[0], -1)

        Q_values = model.predict(state, verbose=0)[0]
        return Q_values.argmax()  # optimal action according to the DQN

from collections import deque

replay_buffer = deque(maxlen=2000)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = np.empty(max_size, dtype=object)
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.randint(self.size, size=batch_size)
        return self.buffer[indices]

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    return [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]  # [states, actions, rewards, next_states, dones]

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    actions.clear()  # Clear the actions list before appending
    actions.append(action)
    next_state, reward, done, info = env.step(actions)

    # Store the original state and next_state in the replay buffer
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

# extra code – for reproducibility, and to generate the next figure
grid_size = (5, 5)
num_agents = 1
obstacles = [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3), (4, 1), (4, 2),
             (4, 3)]
goals = [(4, 0)]
initial_positions = [(0, 4)]
env = GridEnv(grid_size, num_agents, obstacles, goals, initial_positions)
env.reset()
np.random.seed(42)
tf.random.set_seed(42)
rewards = []
best_score = 0

batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
loss_fn = tf.keras.losses.MeanSquaredError

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    # Ensure states and next_states have the correct shape for the model
    if len(states.shape) == 1:
        states = states.reshape(-1, 2)
    elif len(states.shape) > 2:
        # If states has more than 2 dimensions, reshape it to 2D
        states = states.reshape(states.shape[0], -1)

    if len(next_states.shape) == 1:
        next_states = next_states.reshape(-1, 2)
    elif len(next_states.shape) > 2:
        # If next_states has more than 2 dimensions, reshape it to 2D
        next_states = next_states.reshape(next_states.shape[0], -1)

    next_Q_values = model.predict(next_states, verbose=0)
    max_next_Q_values = next_Q_values.max(axis=1)
    runs = 1.0 - (dones)  # episode is not done or truncated
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# Initialize a list to track total rewards per episode
total_rewards = []

for episode in range(600):
    obs = env.reset()
    actions = []
    total_reward = 0  # Track total reward for the current episode

    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        total_reward += reward  # Accumulate rewards

        if done:
            break

    total_rewards.append(total_reward)  # Store total rewards for the episode

    # Perform training step after the episode ends
    if episode > 50:
        training_step(batch_size)

    # Print progress every 10 episodes
    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

# Optionally, print the final results after training
print(f"Training finished. Average reward over last 100 episodes: {np.mean(total_rewards[-100:]):.2f}")
