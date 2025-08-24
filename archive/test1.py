import numpy as np

# Initialize the state matrix
num_agents = 3
num_nodes = 30
state_matrix = np.zeros((num_agents, num_nodes))

# Example paths for each agent
agent_paths = [
    [1, 4, 11, 12, 22, 23, 22, 24, 11],  # Path for Agent 1
    [2, 4, 11, 12, 13, 14, 15, 25],  # Path for Agent 2
    [3, 4, 11, 12, 13, 14, 15, 16, 26]  # Path for Agent 3
]

# Update the state matrix based on the paths
for agent_index, path in enumerate(agent_paths):
    for node in path:
        state_matrix[agent_index, node - 1] += 1

print(state_matrix)


def select_actions(agents, states):
    num_agents = len(agents)
    num_actions = 2 ** num_agents - 1
    action_vectors = [list(map(int, bin(i)[2:].zfill(num_agents))) for i in range(1, num_actions + 1)]

    actions = []
    log_probs = []
    for agent, state in zip(agents, states):
        state = torch.FloatTensor(state)
        probabilities = agent.policy_net(state)
        action_index = np.random.choice(len(probabilities), p=probabilities.detach().numpy())
        action_vector = action_vectors[action_index]
        actions.append(action_vector)
        log_probs.append(torch.log(probabilities[action_index]))

    return actions, log_probs