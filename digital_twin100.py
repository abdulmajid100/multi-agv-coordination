import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

# ============= GRAPH INITIALIZATION =============
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

# ============= AGV TASK DEFINITIONS =============
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


# ============= CORE MOVEMENT FUNCTIONS =============
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


# ============= DEADLOCK DETECTION FUNCTIONS =============
def detect_intersection_deadlock(waiting_agvs, waiting_order, agv_tasks):
    """
    Simple intersection deadlock detection.
    Detects when AGVs around an intersection create circular dependencies.
    """
    waiting_nodes = [agv_info['next_node'] for agv_info in waiting_agvs.values()]
    conflicts = {}

    for intersection_node in set(waiting_nodes):
        # Get AGVs waiting for this intersection
        conflicting_agvs = []
        for agv in reversed(waiting_order):
            if agv in waiting_agvs and waiting_agvs[agv]['next_node'] == intersection_node:
                conflicting_agvs.append(agv)

        if len(conflicting_agvs) >= 3:  # Need at least 3 for circular deadlock
            # Check if they form a circular pattern
            if is_circular_deadlock(conflicting_agvs, waiting_agvs, intersection_node, agv_tasks):
                print(f"Circular deadlock at intersection {intersection_node}")
                conflicts[intersection_node] = conflicting_agvs

    return conflicts


def is_circular_deadlock(conflicting_agvs, waiting_agvs, intersection_node, agv_tasks):
    """
    Check if AGVs form a circular pattern around the intersection.
    Example: AGV1(13->14->24), AGV2(24->14->15), AGV3(15->14->13)
    """
    # Get where each AGV wants to go AFTER the intersection
    agv_destinations = {}

    for agv in conflicting_agvs:
        current_node = waiting_agvs[agv]['current_node']

        # Find where this AGV wants to go after the intersection
        current_task = agv_tasks[agv][0] if agv_tasks[agv] else []

        if intersection_node in current_task:
            intersection_index = current_task.index(intersection_node)
            if intersection_index < len(current_task) - 1:
                destination_after_intersection = current_task[intersection_index + 1]
                agv_destinations[agv] = {
                    'current': current_node,
                    'destination': destination_after_intersection
                }

    # Check for circular pattern: A wants B's position, B wants C's position, etc.
    if len(agv_destinations) >= 3:
        current_positions = [info['current'] for info in agv_destinations.values()]
        destinations = [info['destination'] for info in agv_destinations.values()]

        # Simple check: if destinations overlap with current positions, it's likely circular
        overlap = set(current_positions) & set(destinations)
        print(len(overlap), "overlap")
        if len(overlap) >= 3:  # At least 2 positions are both current and destination
            return True

    return False


# Update your main detect_deadlock function
def detect_deadlock(waiting_agvs, waiting_order, agv_tasks):
    """
    Enhanced deadlock detection with intersection circular pattern detection.
    """
    # First check for intersection circular deadlocks
    intersection_conflicts = detect_intersection_deadlock(waiting_agvs, waiting_order, agv_tasks)
    if intersection_conflicts:
        return intersection_conflicts

    """# Fallback to simple conflict detection
    waiting_nodes = [agv_info['next_node'] for agv_info in waiting_agvs.values()]
    conflicts = {}

    for node in set(waiting_nodes):
        conflicting_agvs = []
        for agv in reversed(waiting_order):
            if agv in waiting_agvs and waiting_agvs[agv]['next_node'] == node:
                conflicting_agvs.append(agv)

        if len(conflicting_agvs) >= 3:  # Standard 3+ AGV conflicts
            conflicts[node] = conflicting_agvs

    return conflicts"""


def check_conflict_resolved(conflict_node, resource_states):
    """
    Check if the conflicting node is now occupied by one of the conflicting AGVs.
    """
    return resource_states[conflict_node] != 0


def check_persistent_deadlock(waiting_agvs, backtracked_agvs):
    """
    Check if there are still AGVs waiting for nodes that backtracked AGVs were originally conflicted with.
    """
    persistent_conflicts = {}

    for backtracked_agv, info in backtracked_agvs.items():
        original_conflict_node = info['original_conflict_node']

        # Check if other AGVs are still waiting for this node
        still_waiting = [agv for agv, wait_info in waiting_agvs.items()
                         if wait_info['next_node'] == original_conflict_node]

        if still_waiting:
            persistent_conflicts[original_conflict_node] = {
                'backtracked_agv': backtracked_agv,
                'waiting_agvs': still_waiting
            }

    return persistent_conflicts


# ============= BACKTRACKING FUNCTIONS =============
def start_backtracking(agv, conflict_node, waiting_agvs, waiting_order, agv_tasks, resource_states, agv_history,
                       backtracked_agvs):
    """
    Start the backtracking process for an AGV.
    """
    current_node = waiting_agvs[agv]['current_node']

    # Check if AGV can backtrack within current subtask
    if len(agv_history[agv][-1]) > 1:
        # Get the previous node from current subtask history
        previous_node = agv_history[agv][-1][-2]
        print(f"{previous_node} prev node (same subtask)")

        # Free the current node and move back
        resource_states[current_node] = 0
        resource_states[previous_node] = agv

        # Insert the current node back into the task
        agv_tasks[agv][0].insert(0, previous_node)

        # Remove the last entry from history (backtrack)
        agv_history[agv][-1].pop()

        print(f"{agv} backtracks to {previous_node} (backtrack #1)")

    # Check if AGV can backtrack to previous subtask
    elif len(agv_history[agv]) > 1:
        # Get the last node from previous subtask
        previous_node = agv_history[agv][-2][-1]
        print(f"{previous_node} prev node (previous subtask)")

        # Free the current node and move back
        resource_states[current_node] = 0
        resource_states[previous_node] = agv

        # Insert the current node back into the current task
        agv_tasks[agv].insert(0, [previous_node])

        # Remove the current subtask from history (go back to previous subtask)
        agv_history[agv].pop()

        print(f"{agv} backtracks to previous subtask at {previous_node} (backtrack #1)")

    else:
        print(f"Cannot backtrack {agv} - no movement history available")
        remove_from_waiting(agv, waiting_agvs, waiting_order)
        return

    # Add to backtracked AGVs with conflict info
    backtracked_agvs[agv] = {
        'original_conflict_node': conflict_node,
        'backtrack_count': 1
    }

    # Remove from waiting list
    remove_from_waiting(agv, waiting_agvs, waiting_order)


def backtrack_further(agv, agv_tasks, resource_states, agv_history, backtracked_agvs):
    """
    Make an AGV backtrack one more step.
    """
    # Check if AGV can backtrack further within current subtask
    if len(agv_history[agv][-1]) > 1:
        current_node = agv_history[agv][-1][-1]  # Last node in current subtask
        previous_node = agv_history[agv][-1][-2]  # Second-to-last node in current subtask
        print(f"{previous_node} prev node (same subtask)")

        # Free the current node and move back
        resource_states[current_node] = 0
        resource_states[previous_node] = agv

        # Insert the current node back into the task
        agv_tasks[agv][0].insert(0, previous_node)

        # Remove the last entry from history (backtrack)
        agv_history[agv][-1].pop()

        print(
            f"{agv} backtracks further to {previous_node} (backtrack #{backtracked_agvs[agv]['backtrack_count'] + 1})")

    # Check if AGV can backtrack to previous subtask
    elif len(agv_history[agv]) > 1:
        current_node = agv_history[agv][-1][-1]  # Last node in current subtask
        previous_node = agv_history[agv][-2][-1]  # Last node from previous subtask
        print(f"{previous_node} prev node (previous subtask)")

        # Free the current node and move back
        resource_states[current_node] = 0
        resource_states[previous_node] = agv

        # Insert the current node back into the current task
        agv_tasks[agv].insert(0, [previous_node])

        # Remove the current subtask from history (go back to previous subtask)
        agv_history[agv].pop()

        print(
            f"{agv} backtracks to previous subtask at {previous_node} (backtrack #{backtracked_agvs[agv]['backtrack_count'] + 1})")

    else:
        print(f"{agv} cannot backtrack further - reached starting position")
        return

    # Update backtrack count
    backtracked_agvs[agv]['backtrack_count'] += 1


def check_backtracked_agv_resume(backtracked_agvs, resource_states):
    """
    Check if any backtracked AGV can resume normal operation because
    someone moved to their original conflict node.
    """
    agvs_to_resume = []

    for agv, info in backtracked_agvs.items():
        original_conflict_node = info['original_conflict_node']

        # Check if someone is now occupying the original conflict node
        if resource_states[original_conflict_node] != 0 and resource_states[original_conflict_node] != agv:
            print(f"Node {original_conflict_node} is now occupied! {agv} can resume normal operation100.")
            agvs_to_resume.append(agv)
            print(agvs_to_resume, "AGVS TO RESUME")

    return agvs_to_resume


# ============= DEADLOCK RESOLUTION FUNCTIONS =============
def resolve_deadlock(conflicts, waiting_agvs, waiting_order, agv_tasks, resource_states, agv_history, backtracked_agvs):
    """
    Resolve deadlocks by making the last AGV to wait backtrack progressively.
    """
    for conflict_node, conflicting_agvs in conflicts.items():
        # Pick the last AGV to enter the waiting state (last in the list)
        print(waiting_agvs)
        last_agv = conflicting_agvs[-1]

        # Check if this AGV is already backtracked and waiting
        if last_agv in backtracked_agvs:
            # Check if the conflict is resolved (someone moved to the conflict node)
            if check_conflict_resolved(conflict_node, resource_states):
                print(f"Conflict at {conflict_node} resolved! {last_agv} can resume normal operation.")
                # Remove from backtracked list and waiting list
                del backtracked_agvs[last_agv]
                remove_from_waiting(last_agv, waiting_agvs, waiting_order)
                continue
            else:
                # Conflict still exists, backtrack further if possible
                print(f"Conflict at {conflict_node} still exists. {last_agv} will backtrack further.")
                backtrack_further(last_agv, agv_tasks, resource_states, agv_history, backtracked_agvs)
                continue

        # First time encountering this deadlock - start backtracking
        print(f"New deadlock detected at {conflict_node}. {last_agv} will start backtracking.")
        start_backtracking(last_agv, conflict_node, waiting_agvs, waiting_order, agv_tasks, resource_states, agv_history,
                           backtracked_agvs)

def remove_from_waiting(agv, waiting_agvs, waiting_order):
    """Remove AGV from both waiting_agvs and waiting_order"""
    if agv in waiting_agvs:
        del waiting_agvs[agv]
    if agv in waiting_order:
        waiting_order.remove(agv)

# ============= MAIN SIMULATION FUNCTION =============
def simulate_digital_twin():
    """
    Simulate the digital twin to generate conflict-free sequences.
    """
    # Create deep copies to avoid modifying original data
    agv_tasks = copy.deepcopy(original_agv_tasks)
    resource_states = {node: 0 for node in G.nodes()}

    # Track movement history for each AGV
    agv_history = {agv: [] for agv in agv_tasks.keys()}

    # Track AGVs that are backtracked and waiting
    backtracked_agvs = {}

    # Reserve starting nodes for each AGV and initialize history
    for agv, tasks in agv_tasks.items():
        if tasks and tasks[0]:
            starting_node = tasks[0][0]
            resource_states[starting_node] = agv
            agv_history[agv] = [[starting_node]]  # Initialize with first subtask

    conflict_free_sequences = []
    waiting_agvs = {}
    waiting_order = []
    max_iterations = 1000
    iteration = 0

    while any(tasks for tasks in agv_tasks.values()) and iteration < max_iterations:
        iteration += 1
        moved_this_round = False

        print(f"\n--- Iteration {iteration} ---")
        print(f"Backtracked AGVs: {backtracked_agvs}")

        # Check if any backtracked AGV can resume normal operation
        if backtracked_agvs:
            agvs_to_resume = check_backtracked_agv_resume(backtracked_agvs, resource_states)
            for agv in agvs_to_resume:
                print(f"Resuming normal operation for {agv}")
                del backtracked_agvs[agv]
                moved_this_round = True
        print(agv_history, "AGV HISTORY BEFORE MOVEMENT")
        # Move each AGV if possible
        for agv, tasks in agv_tasks.items():
            # Skip if AGV is backtracked and waiting
            print(agv_tasks[agv], "tasks117")
            if agv in backtracked_agvs:
                continue

            if tasks and len(tasks[0]) >= 1:
                current_node = tasks[0][0]

                if len(tasks[0]) == 1 and len(tasks) > 1:
                    next_node = tasks[1][0]
                elif len(tasks[0]) == 1:
                    continue
                else:
                    next_node = tasks[0][1]

                other_agvs = [other_agv for other_agv in agv_tasks if other_agv != agv]
                shared_nodes_with_others = {other_agv: [] for other_agv in other_agvs}

                # Calculate shared nodes with other AGVs
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

                # Check if AGV can move
                if can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node, resource_states):
                    # Reserve the next node for the AGV
                    resource_states[current_node] = 0
                    resource_states[next_node] = agv

                    # Move AGV to the next node
                    tasks[0].pop(0)
                    subtask_completed = not tasks[0]  # Check BEFORE popping

                    # Add next node to current subtask history
                    if len(tasks[0]) != 1:
                       agv_history[agv][-1].append(next_node)
                    print(agv_history, "AGV HISTORY AFTER MOVE")
                    if subtask_completed:
                        tasks.pop(0)  # Remove completed subtask
                        print(tasks, "tasks after pop")
                        if tasks:  # More subtasks exist
                            agv_history[agv].append([tasks[0][0]])  # Start new subtask with first node
                    conflict_free_sequences.append((agv, current_node, next_node, 'move'))
                    print(f"{agv} moves from {current_node} to {next_node}")
                    moved_this_round = True
                else:
                    # AGV is waiting
                    if agv not in waiting_agvs:  # Only add if not already waiting
                        waiting_agvs[agv] = {
                            'current_node': current_node,
                            'next_node': next_node
                        }
                        waiting_order.append(agv)  # Only add new waiters
                        print(f"{agv} added to waiting list (position {len(waiting_order)})")
                    else:
                        # Update existing waiting AGV info (but don't re-add to waiting_order)
                        waiting_agvs[agv].update({
                            'current_node': current_node,
                            'next_node': next_node
                        })
                    #print(f"{agv} added to waiting list (position {len(waiting_order)})")
                    print(waiting_agvs, "WAITING AGVS")
                    conflict_free_sequences.append((agv, current_node, current_node, 'wait'))
                    print(f"{agv} waiting at {current_node}")

        if waiting_agvs or backtracked_agvs:
            conflicts = detect_deadlock(waiting_agvs, waiting_order, agv_tasks)
            print(f"Conflicts detected: {conflicts}")
            print(agv_history, "AGV HISTORY")

            if conflicts:
                # Instead of resolve_deadlock, directly start backtracking the last AGV
                for conflict_node, conflicting_agvs in conflicts.items():
                    last_agv = conflicting_agvs[-1]  # Last AGV to wait
                    print(f"New deadlock detected at {conflict_node}. {last_agv} will start backtracking.")
                    start_backtracking(last_agv, conflict_node, waiting_agvs, waiting_order, agv_tasks, resource_states,
                                       agv_history, backtracked_agvs)
                    moved_this_round = True

            elif backtracked_agvs:
                # Handle persistent deadlocks
                """persistent_conflicts = check_persistent_deadlock(waiting_agvs, backtracked_agvs)
                if persistent_conflicts:
                    for conflict_node, conflict_info in persistent_conflicts.items():
                        backtracked_agv = conflict_info['backtracked_agv']
                        print(f"Persistent deadlock at {conflict_node}. {backtracked_agv} will backtrack further.")
                        backtrack_further(backtracked_agv, agv_tasks, resource_states, agv_history, backtracked_agvs)
                        moved_this_round = True
                else:
                    # Clear resolved situations
                    waiting_agvs.clear()
                    waiting_order.clear()
                    backtracked_agvs.clear()"""
                agvs_to_resume = check_backtracked_agv_resume(backtracked_agvs, resource_states)
                for agv in agvs_to_resume:
                    print(f"Resuming normal operation for {agv}")
                    del backtracked_agvs[agv]
                    moved_this_round = True
                if backtracked_agvs:
                    for backtracked_agv, info in backtracked_agvs.items():
                        backtrack_further(backtracked_agv, agv_tasks, resource_states, agv_history, backtracked_agvs)
            else:
                # No conflicts - clear everything
                waiting_agvs.clear()
                waiting_order.clear()

        # If no movement happened and no conflicts to resolve, break to avoid infinite loop
        if not moved_this_round and not waiting_agvs and not backtracked_agvs:
            break

    return conflict_free_sequences


# ============= SIMULATION EXECUTION =============
print("Starting digital twin simulation...")
conflict_free_sequences = simulate_digital_twin()
print(conflict_free_sequences, "CONFLICT FREE SEQUENCES")
print(f"Simulation complete. Generated {len(conflict_free_sequences)} steps.")

# ============= VISUALIZATION SETUP =============
fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.kamada_kawai_layout(G)

# Initialize AGV positions with their starting nodes
agv_positions = {}
for agv, tasks in original_agv_tasks.items():
    if tasks and tasks[0]:
        agv_positions[agv] = tasks[0][0]


def build_position_history(conflict_free_sequences):
    position_history = []
    current_positions = {agv: tasks[0][0] for agv, tasks in original_agv_tasks.items()}

    for agv, from_node, to_node, action in conflict_free_sequences:
        current_positions[agv] = to_node
        position_history.append(copy.deepcopy(current_positions))

    return position_history

print(build_position_history(conflict_free_sequences), "POSITION HISTORY")
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

        # Print frame info for debugging
        print(f"Animation Frame {frame}: {agv} {action_text} from {from_node} to {to_node}")


# ============= ANIMATION EXECUTION =============
if conflict_free_sequences:
    ani = animation.FuncAnimation(fig, update, frames=len(conflict_free_sequences),
                                  repeat=False, interval=1500, blit=False)
    plt.tight_layout()
    plt.show()
else:
    print("No sequences generated. Check the simulation logic.")
    nx.draw(G, pos, with_labels=True, arrows=True, node_color='lightblue',
            node_size=300, font_size=8, font_weight='bold')
    plt.title("AGV Navigation Graph")
    plt.show()
