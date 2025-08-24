# Fix for IndexError in Q-Learning Implementation

## Issue Description

The original code was encountering an `IndexError` during Q-learning training:

```
IndexError: index 68038 is out of bounds for axis 0 with size 24389
```

This error occurred in the `q_learning` function at line 139:

```python
best_next_action = np.argmax(q_tables[i][next_state_index][:next_valid_actions])
```

## Root Cause

The issue was in the `state_to_index` function, which was incorporating node degrees into the index calculation when a graph was provided:

```python
# Incorporate node degree if graph is provided
if graph is not None:
    # Get the degree of the node (number of connections)
    node_degree = graph.degree(state[i])
    # Add the degree to the index calculation
    index += (node_id * node_degree) * factor
```

This could lead to indices larger than the Q-table size, which was initialized with a size of `(num_nodes ** env.num_agents, max_action_size)`.

## Solution

The fix was to remove the node degree multiplication from the index calculation, ensuring that the indices stay within the bounds of the Q-table size:

```python
# Simply add the node_id multiplied by the factor
index += node_id * factor
```

This ensures that the state indices are properly bounded and prevents the IndexError.

## Testing the Fix

To test the fix:

1. Open a terminal or command prompt
2. Navigate to the project directory:
   ```
   cd C:\Users\majmo\PycharmProjects\pythonProject1
   ```
3. Run the test script:
   ```
   python test_fix.py
   ```

The script should execute without the previous IndexError, confirming that the fix works.