"""
This script tests the fixed state_to_index function to verify that it produces
indices within the expected bounds.
"""

import numpy as np
import networkx as nx
from RL_based_traffic import state_to_index, G

def test_state_to_index():
    # Create a test graph similar to the one in RL_based_traffic.py
    num_nodes = len(G.nodes())
    num_agents = 3
    
    # Calculate the expected maximum index
    max_expected_index = num_nodes ** num_agents - 1
    
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of agents: {num_agents}")
    print(f"Maximum expected index: {max_expected_index}")
    
    # Test with various states
    test_states = [
        [1, 15, 28],  # Example from the original code
        [24, 11, 26],  # Another example from the original code
        [29, 29, 29],  # Maximum possible state values
    ]
    
    for state in test_states:
        index = state_to_index(state, num_nodes, G)
        print(f"State {state} -> Index {index}")
        
        # Check if the index is within bounds
        if index <= max_expected_index:
            print(f"  ✓ Index is within bounds (≤ {max_expected_index})")
        else:
            print(f"  ✗ Index exceeds bounds (> {max_expected_index})")
    
    print("\nTest completed.")

if __name__ == "__main__":
    print("Testing the fixed state_to_index function...")
    test_state_to_index()