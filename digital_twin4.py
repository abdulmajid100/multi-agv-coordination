import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

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
original_agv_tasks = {
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


def can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node, resource_states):
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


def detect_deadlock(waiting_agvs):
    """
    Detect if there is a deadlock (multiple AGVs waiting for the same node).
    """
    waiting_nodes = [agv_info['next_node'] for agv_info in waiting_agvs.values()]
    conflicts = {}

    for node in set(waiting_nodes):
        conflicting_agvs = [agv for agv, info in waiting_agvs.items() if info['next_node'] == node]
        if len(conflicting_agvs) > 2:
            conflicts[node] = conflicting_agvs

    return conflicts


def resolve_deadlock(conflicts, waiting_agvs, agv_tasks, resource_states, agv_history):
    """
    Resolve deadlocks by making the last AGV to wait backtrack one node using movement history.
    """
    for conflict_node, conflicting_agvs in conflicts.items():
        # Pick the last AGV to enter the waiting state (last in the list)
        last_agv = conflicting_agvs[-1]
        print(f"Last AGV to wait: {last_agv}")

        # Get current node and movement history
        current_node = waiting_agvs[last_agv]['current_node']
        print(f"Current node: {current_node}")

        # Check if AGV has movement history to backtrack
        if len(agv_history[last_agv]) > 1:
            # Get the previous node from history (second to last)
            previous_node = agv_history[last_agv][-2]
            print(f"Previous node from history: {previous_node}")

            # Free the current node and move back
            resource_states[current_node] = 0
            resource_states[previous_node] = last_agv

            # Insert the current node back into the task (so AGV can try again later)
            agv_tasks[last_agv][0].insert(0, previous_node)

            # Remove the last entry from history (backtrack)
            agv_history[last_agv].pop()

            print(f"Resolving conflict at node {conflict_node}: {last_agv} backtracks to {previous_node}")

            # Remove from waiting list
            del waiting_agvs[last_agv]
        else:
            print(f"Cannot backtrack {last_agv} - no movement history available")
            # If no history, just remove from waiting (will try again next iteration)
            del waiting_agvs[last_agv]


def simulate_digital_twin():
    """
    Simulate the digital twin to generate conflict-free sequences.
    """
    # Create deep copies to avoid modifying original data
    agv_tasks = copy.deepcopy(original_agv_tasks)
    resource_states = {node: 0 for node in G.nodes()}

    # Track movement history for each AGV
    agv_history = {agv: [] for agv in agv_tasks.keys()}

    # Reserve starting nodes for each AGV and initialize history
    for agv, tasks in agv_tasks.items():
        if tasks and tasks[0]:
            starting_node = tasks[0][0]
            resource_states[starting_node] = agv
            agv_history[agv].append(starting_node)  # Add starting node to history

    conflict_free_sequences = []
    waiting_agvs = {}
    max_iterations = 1000
    iteration = 0

    while any(tasks for tasks in agv_tasks.values()) and iteration < max_iterations:
        iteration += 1
        moved_this_round = False
        print(agv_tasks, "agv777")
        # Move each AGV if possible
        for agv, tasks in agv_tasks.items():
            if agv in waiting_agvs:
                continue

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

                if can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node, resource_states):
                    # Reserve the next node for the AGV
                    resource_states[current_node] = 0
                    resource_states[next_node] = agv

                    # Move AGV to the next node
                    tasks[0].pop(0)
                    if not tasks[0]:  # If the current task is completed
                        tasks.pop(0)

                    # Add next node to movement history
                    agv_history[agv].append(next_node)

                    conflict_free_sequences.append((agv, current_node, next_node, 'move'))

                    print(f"{agv} moves from {current_node} to {next_node}")
                    moved_this_round = True
                else:
                    # AGV is waiting
                    waiting_agvs[agv] = {
                        'current_node': current_node,
                        'next_node': next_node
                    }
                    conflict_free_sequences.append((agv, current_node, current_node, 'wait'))
                    print(f"{agv} waiting at {current_node}")

        # Detect and resolve deadlocks
        if waiting_agvs:
            conflicts = detect_deadlock(waiting_agvs)
            print(f"AGV History: {agv_history}")
            print(f"Conflicts: {conflicts}")
            if conflicts:
                resolve_deadlock(conflicts, waiting_agvs, agv_tasks, resource_states, agv_history)
                moved_this_round = True
            else:
                # Clear waiting AGVs if no conflicts (they can try again next round)
                waiting_agvs.clear()

        # If no movement happened and no conflicts to resolve, break to avoid infinite loop
        if not moved_this_round and not waiting_agvs:
            break

    return conflict_free_sequences


# Simulate the digital twin
print("Starting digital twin simulation...")
conflict_free_sequences = simulate_digital_twin()
print(f"Simulation complete. Generated {len(conflict_free_sequences)} steps.")
print("Conflict-Free Sequences:", conflict_free_sequences[:10])  # Show first 10 steps

# Initialize plot
fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.kamada_kawai_layout(G)

# Track AGV positions for animation
agv_positions = {'AGV1': None, 'AGV2': None, 'AGV3': None}


def update(frame):
    """
    Update function for animation.
    """
    ax.clear()
    nx.draw(G, pos, with_labels=True, arrows=True, ax=ax, node_color='lightblue',
            node_size=300, font_size=8, font_weight='bold')

    if frame < len(conflict_free_sequences):
        agv, from_node, to_node, action = conflict_free_sequences[frame]

        # Update AGV position
        agv_positions[agv] = to_node

        # Draw all AGVs at their current positions
        for agv_name, position in agv_positions.items():
            if position is not None:
                color = 'red' if agv_name == 'AGV1' else 'green' if agv_name == 'AGV2' else 'blue'
                nx.draw_networkx_nodes(
                    G, pos, nodelist=[position],
                    node_color=color, node_size=500, ax=ax
                )

        # Add title with current action
        action_text = "WAITING" if action == 'wait' else "MOVING"
        ax.set_title(f"Frame {frame}: {agv} {action_text} at node {to_node}", fontsize=12, fontweight='bold')

        print(f"Frame {frame}: {agv} {action} from {from_node} to {to_node}")


# Create animation
if conflict_free_sequences:
    ani = animation.FuncAnimation(fig, update, frames=len(conflict_free_sequences),
                                  repeat=False, interval=1500, blit=False)
    plt.tight_layout()
    plt.show()
else:
    print("No sequences generated. Check the simulation logic.")
    # Show static graph
    nx.draw(G, pos, with_labels=True, arrows=True, node_color='lightblue',
            node_size=300, font_size=8, font_weight='bold')
    plt.title("AGV Navigation Graph")
    plt.show()
