import numpy as np
import networkx as nx
from RL_based_traffic import state_to_index, G

def test_state_to_index_with_degree():
    """
    Test the modified state_to_index function that incorporates node degree information.
    Verify that the indices are within the expected bounds.
    """
    # Get the number of nodes and agents from the original code
    num_nodes = len(G.nodes())
    num_agents = 3  # As used in the main function
    
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
    
    # Test with and without graph
    print("\nTesting without graph (no degree info):")
    for state in test_states:
        index = state_to_index(state, num_nodes)
        print(f"State {state} -> Index {index}")
        
        # Check if the index is within bounds
        if index <= max_expected_index:
            print(f"  ✓ Index is within bounds (≤ {max_expected_index})")
        else:
            print(f"  ✗ Index exceeds bounds (> {max_expected_index})")
    
    print("\nTesting with graph (including degree info):")
    for state in test_states:
        index = state_to_index(state, num_nodes, G)
        print(f"State {state} -> Index {index}")
        
        # Check if the index is within bounds
        if index <= max_expected_index:
            print(f"  ✓ Index is within bounds (≤ {max_expected_index})")
        else:
            print(f"  ✗ Index exceeds bounds (> {max_expected_index})")
        
        # Print the degree of each node in the state
        print(f"  Node degrees: {[G.degree(node) for node in state]}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    print("Testing the modified state_to_index function with degree information...")
    test_state_to_index_with_degree()