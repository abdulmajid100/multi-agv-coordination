import copy

fixed_paths = [[4, 11, 12, 22], [4, 11, 12, 13, 14, 15, 16, 26]]

for episode in range(3):
    agv_paths = copy.deepcopy(fixed_paths)
    print(f"Episode {episode + 1}: Initial agv_paths: {agv_paths}")

    for step in range(30):
        for agent_index in range(len(agv_paths)):
            if len(agv_paths[agent_index]) > 1:
                agv_paths[agent_index] = agv_paths[agent_index][1:]
                print(f"Step {step + 1}, Agent {agent_index}: {agv_paths[agent_index]}")
            else:
                agv_paths[agent_index] = []
                print(f"Step {step + 1}, Agent {agent_index}: Path completed.")