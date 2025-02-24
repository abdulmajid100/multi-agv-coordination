import numpy as np
import scipy.signal

# Define parameters
discount = 0.9  # Discount factor
rewards = np.array([1, 2, 3, 4, 5])  # Example rewards

# Apply the lfilter to compute discounted returns
# The filter coefficients
b = [1]  # Numerator coefficients
a = [1, float(-discount)]  # Denominator coefficients

# Reverse the rewards to apply the filter
discounted_returns = scipy.signal.lfilter(b, a, rewards[::-1], axis=0)[::-1]

# Print the results
print("Rewards:", rewards)
print("Discounted Returns:", discounted_returns[:-1])