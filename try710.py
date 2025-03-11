def calculate_shared_nodes(agent_paths):
    """
    Calculates shared nodes between each pair of agents.

    Parameters:
    - agent_paths: List of lists, where each sublist contains the path of an agent.

    Returns:
    - A dictionary with pairs of agents as keys and their shared nodes as values.
    """
    # Dictionary to store results
    shared_nodes = {}

    # Number of agents
    num_agents = len(agent_paths)

    # Loop through each pair of agents (i and j)
    for i in range(num_agents):
        for j in range(i + 1, num_agents):  # Ensure i < j to avoid duplicates
            agent_i = i + 1  # Agent indices (e.g., 1, 2, 3)
            agent_j = j + 1

            # Find shared nodes between agent i and agent j
            shared = list(set(agent_paths[i]).intersection(agent_paths[j]))

            # Add the shared nodes to the dictionary
            shared_nodes[(agent_i, agent_j)] = shared

    return shared_nodes


# Example usage
agent_paths = [
    [1, 2, 3],  # Path for Agent 1
    [2, 3, 4],  # Path for Agent 2
    [3, 1, 5],  # Path for Agent 3
    [4, 5, 6]   # Path for Agent 4
]

result = calculate_shared_nodes(agent_paths)
print(result)
