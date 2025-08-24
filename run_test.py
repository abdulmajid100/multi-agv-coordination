"""
This script runs the test_degree_info.py script to verify that the modified
state_to_index function works correctly with node degree information.
"""

import subprocess
import sys

def main():
    print("Running test_degree_info.py to verify the modified state_to_index function...")
    
    try:
        # Run the test script
        result = subprocess.run([sys.executable, "test_degree_info.py"], 
                               capture_output=True, text=True, check=True)
        
        # Print the output
        print("\nTest Output:")
        print(result.stdout)
        
        # Check if the test was successful
        if "Test completed." in result.stdout and "✗ Index exceeds bounds" not in result.stdout:
            print("\nTest passed! The modified state_to_index function works correctly.")
            print("Node degree information has been successfully incorporated without causing index out of bounds errors.")
        else:
            print("\nTest failed! Please check the output for details.")
    
    except subprocess.CalledProcessError as e:
        print(f"\nError running the test: {e}")
        print("\nError output:")
        print(e.stderr)

if __name__ == "__main__":
    main()