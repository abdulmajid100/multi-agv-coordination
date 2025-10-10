import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize the directed graph
G = nx.DiGraph()
nodes = list(range(1, 30)) + [9]
G.add_nodes_from(nodes)
edges = [
    (1, 4), (2, 4), (3, 4), (4, 1), (4, 2), (4, 3), (4, 6), (4, 5),
    (5, 4), (6, 4), (4, 11), (11, 4), (11, 10), (11, 21), (11, 12),
    (10, 11), (10, 20), (20, 10), (21, 11), (12, 11), (12, 22),
    (22, 12), (12, 13), (13, 12), (13, 23), (23, 13), (13, 14),
    (14, 13), (14, 24), (24, 14), (14, 15), (15, 14), (15, 25),
    (25, 15), (15, 16), (16, 15), (16, 26), (26, 16), (16, 17),
    (17, 16), (17, 27), (27, 17), (17, 18), (18, 17), (18, 28),
    (28, 18), (18, 19), (19, 18), (19, 29), (29, 19), (9, 16),
    (16, 9)
]
G.add_edges_from(edges)

# Define AGVs and their tasks (multiple paths)
agv_tasks = {
    'AGV1': [
        [1, 4, 11, 12, 13, 14, 24],
        [24, 14, 13, 12, 11, 4, 5, 4, 6],
        [6, 4, 11, 10, 20],
        [20, 10, 11, 4, 1]
    ],
    'AGV2': [
        [17, 16, 15, 14, 24],
        [24, 14, 13, 12, 11, 4, 5, 4, 6],
        [6, 4, 11, 12, 13, 14, 15, 25],
        [25, 15, 14, 13, 12, 11, 4, 2]
    ],
    'AGV3': [
        [3, 4, 11, 12, 13, 14, 15, 16, 26],
        [26, 16, 15, 14, 13, 12, 11, 4, 5, 4, 6],
        [6, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29],
        [29, 19, 18, 17, 16, 15, 14, 13, 12, 11, 4, 3]
    ]
}

# Initialize the reservation status of each node
resource_states = {node: 0 for node in G.nodes()}

# Reserve starting nodes for each AGV
for agv, tasks in agv_tasks.items():
    if tasks and tasks[0]:  # Check if there are tasks and nodes in the first task
        starting_node = tasks[0][0]
        resource_states[starting_node] = agv


def can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
    """
    Check if an AGV can move to the next node based on shared node conditions.
    """
    # If the next node is not shared with any other AGV, allow movement
    if all(next_node not in shared_nodes_with_others[other_agv] for other_agv in other_agvs):
        return True

    # If the next node is shared, check if it's free
    for other_agv, shared_nodes in shared_nodes_with_others.items():
        if next_node in shared_nodes and any(resource_states[shared_node] == other_agv for shared_node in shared_nodes):
            return False

    return True


def simulate_digital_twin():
    """
    Simulate the digital twin to generate conflict-free sequences.
    """
    conflict_free_sequences = []

    while any(tasks for tasks in agv_tasks.values()):
        for agv, tasks in agv_tasks.items():
            if tasks and len(tasks[0]) >= 1:  # Check if there are tasks and nodes left
                current_node = tasks[0][0]
                if len(tasks[0]) == 1 and len(tasks) > 1:
                    next_node = tasks[1][0]
                elif len(tasks[0]) == 1:
                    continue
                else:
                    next_node = tasks[0][1]

                other_agvs = [other_agv for other_agv in agv_tasks if other_agv != agv]
                shared_nodes_with_others = {other_agv: [] for other_agv in other_agvs}

                for other_agv in other_agvs:
                    if tasks and agv_tasks[other_agv]:
                        if tasks[0][-1] in agv_tasks[other_agv][0]:
                            if len(agv_tasks[other_agv]) > 1 and len(tasks) > 1:
                                current_task = tasks[0] + tasks[1]
                                other_current_task = agv_tasks[other_agv][0] + agv_tasks[other_agv][1]
                            else:
                                current_task = tasks[0]
                                other_current_task = agv_tasks[other_agv][0]
                            shared_nodes = set(current_task) & set(other_current_task)
                            shared_nodes_with_others[other_agv] = list(shared_nodes)
                        else:
                            current_task = tasks[0]
                            other_current_task = agv_tasks[other_agv][0]
                            shared_nodes = set(current_task) & set(other_current_task)
                            shared_nodes_with_others[other_agv] = list(shared_nodes)

                if can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
                    # Reserve the next node for the AGV
                    resource_states[current_node] = 0
                    resource_states[next_node] = agv

                    # Move AGV to the next node
                    tasks[0].pop(0)
                    if not tasks[0]:  # If the current task is completed
                        tasks.pop(0)
                    conflict_free_sequences.append((agv, current_node, next_node))
                    print(f"{agv} moves from {current_node} to {next_node}")
                else:
                    print(f"{agv} waiting at {current_node}")

    return conflict_free_sequences


# Simulate the digital twin
conflict_free_sequences = simulate_digital_twin()
print("Conflict-Free Sequences:", conflict_free_sequences)

# Visualization (Animation)
fig, ax = plt.subplots()
pos = nx.kamada_kawai_layout(G)


def update(frame):
    ax.clear()
    nx.draw(G, pos, with_labels=True, arrows=True, ax=ax)

    if frame < len(conflict_free_sequences):
        agv, current_node, next_node = conflict_free_sequences[frame]
        nx.draw_networkx_nodes(
            G, pos, nodelist=[next_node],
            node_color='r' if agv == 'AGV1' else 'g' if agv == 'AGV2' else 'b',
            node_size=500, ax=ax
        )
        print(f"Frame {frame}: {agv} moves from {current_node} to {next_node}")


ani = animation.FuncAnimation(fig, update, frames=len(conflict_free_sequences), repeat=False, interval=1000)
plt.show()
