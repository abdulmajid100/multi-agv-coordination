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
def can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node, resource_states,
             forced_waiting_agvs=None):
    """
    Check if an AGV can move to the next node based on shared node conditions.
    Now includes forced waiting check.
    """
    # NEW: Check if AGV is forced to wait
    if forced_waiting_agvs and agv in forced_waiting_agvs:
        wait_info = forced_waiting_agvs[agv]
        if current_node == wait_info['wait_at']:
            conflict_node = wait_info['until_occupied']
            # Can only move if conflict node is occupied by someone else
            if resource_states[conflict_node] == 0 or resource_states[conflict_node] == agv:
                print(f"{agv} is forced to wait at {current_node} until {conflict_node} is occupied")
                return False
            else:
                # Conflict node occupied by another AGV, can proceed
                print(
                    f"{agv} can now proceed from {current_node} (conflict node {conflict_node} occupied by {resource_states[conflict_node]})")
                # Remove from forced waiting
                del forced_waiting_agvs[agv]

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
    """
    agv_destinations = {}

    for agv in conflicting_agvs:
        current_node = waiting_agvs[agv]['current_node']
        current_task = agv_tasks[agv][0] if agv_tasks[agv] else []

        if intersection_node in current_task:
            intersection_index = current_task.index(intersection_node)
            if intersection_index < len(current_task) - 1:
                destination_after_intersection = current_task[intersection_index + 1]
                agv_destinations[agv] = {
                    'current': current_node,
                    'destination': destination_after_intersection
                }

    if len(agv_destinations) >= 3:
        current_positions = [info['current'] for info in agv_destinations.values()]
        destinations = [info['destination'] for info in agv_destinations.values()]
        overlap = set(current_positions) & set(destinations)
        print(len(overlap), "overlap")
        if len(overlap) >= 3:
            return True

    return False


def detect_deadlock(waiting_agvs, waiting_order, agv_tasks):
    """
    Enhanced deadlock detection with intersection circular pattern detection.
    """
    intersection_conflicts = detect_intersection_deadlock(waiting_agvs, waiting_order, agv_tasks)
    if intersection_conflicts:
        return intersection_conflicts
    return {}


# ============= BACKTRACKING FUNCTIONS =============
def start_backtracking(agv, conflict_node, waiting_agvs, waiting_order, agv_tasks, resource_states, agv_history,
                       backtracked_agvs, resolution_points):
    """
    Start the backtracking process for an AGV.
    NEW: Records resolution point information.
    """
    current_node = waiting_agvs[agv]['current_node']

    # NEW: Record the resolution point (where AGV should wait)
    resolution_node = current_node

    # Check if AGV can backtrack within current subtask
    if len(agv_history[agv][-1]) > 1:
        previous_node = agv_history[agv][-1][-2]
        print(f"{previous_node} prev node (same subtask)")

        resource_states[current_node] = 0
        resource_states[previous_node] = agv
        agv_tasks[agv][0].insert(0, previous_node)
        agv_history[agv][-1].pop()
        print(f"{agv} backtracks to {previous_node} (backtrack #1)")

    elif len(agv_history[agv]) > 1:
        previous_node = agv_history[agv][-2][-1]
        print(f"{previous_node} prev node (previous subtask)")

        resource_states[current_node] = 0
        resource_states[previous_node] = agv
        agv_tasks[agv].insert(0, [previous_node])
        agv_history[agv].pop()
        print(f"{agv} backtracks to previous subtask at {previous_node} (backtrack #1)")

    else:
        print(f"Cannot backtrack {agv} - no movement history available")
        remove_from_waiting(agv, waiting_agvs, waiting_order)
        return

    # Add to backtracked AGVs with conflict info
    backtracked_agvs[agv] = {
        'original_conflict_node': conflict_node,
        'resolution_node': resolution_node,  # NEW: Store where AGV should wait
        'backtrack_count': 1
    }

    remove_from_waiting(agv, waiting_agvs, waiting_order)


def backtrack_further(agv, agv_tasks, resource_states, agv_history, backtracked_agvs):
    """
    Make an AGV backtrack one more step.
    """
    if len(agv_history[agv][-1]) > 1:
        current_node = agv_history[agv][-1][-1]
        previous_node = agv_history[agv][-1][-2]
        print(f"{previous_node} prev node (same subtask)")

        resource_states[current_node] = 0
        resource_states[previous_node] = agv
        agv_tasks[agv][0].insert(0, previous_node)
        agv_history[agv][-1].pop()
        print(
            f"{agv} backtracks further to {previous_node} (backtrack #{backtracked_agvs[agv]['backtrack_count'] + 1})")

    elif len(agv_history[agv]) > 1:
        current_node = agv_history[agv][-1][-1]
        previous_node = agv_history[agv][-2][-1]
        print(f"{previous_node} prev node (previous subtask)")

        resource_states[current_node] = 0
        resource_states[previous_node] = agv
        agv_tasks[agv].insert(0, [previous_node])
        agv_history[agv].pop()
        print(
            f"{agv} backtracks to previous subtask at {previous_node} (backtrack #{backtracked_agvs[agv]['backtrack_count'] + 1})")

    else:
        print(f"{agv} cannot backtrack further - reached starting position")
        return

    backtracked_agvs[agv]['backtrack_count'] += 1


def check_backtracked_agv_resume(backtracked_agvs, resource_states):
    """
    Check if any backtracked AGV can resume normal operation.
    """
    agvs_to_resume = []

    for agv, info in backtracked_agvs.items():
        original_conflict_node = info['original_conflict_node']

        if resource_states[original_conflict_node] != 0 and resource_states[original_conflict_node] != agv:
            print(f"Node {original_conflict_node} is now occupied! {agv} can resume normal operation.")
            agvs_to_resume.append(agv)

    return agvs_to_resume


def remove_from_waiting(agv, waiting_agvs, waiting_order):
    """Remove AGV from both waiting_agvs and waiting_order"""
    if agv in waiting_agvs:
        del waiting_agvs[agv]
    if agv in waiting_order:
        waiting_order.remove(agv)


# ============= NEW: FRAME CUTTING FUNCTION =============
def find_frame_to_cut(conflict_free_sequences, agv, resolution_node):
    """
    Find the frame where the AGV first moved FROM the resolution node.
    Returns the frame index, or -1 if not found.
    """
    for i in range(len(conflict_free_sequences)):
        frame = conflict_free_sequences[i]
        if agv in frame:
            from_node, to_node, action = frame[agv]
            if from_node == resolution_node and action == 'move':
                print(f"Found frame {i} where {agv} moved from resolution node {resolution_node}")
                return i
    return -1


# ============= MAIN SIMULATION FUNCTION =============
def simulate_digital_twin():
    """
    Simulate the digital twin to generate conflict-free sequences.
    Now with frame cutting when backtracking completes.
    """
    agv_tasks = copy.deepcopy(original_agv_tasks)
    resource_states = {node: 0 for node in G.nodes()}
    agv_history = {agv: [] for agv in agv_tasks.keys()}
    backtracked_agvs = {}
    resolution_points = []  # NEW: Track all resolution points
    forced_waiting_agvs = {}  # NEW: Track AGVs that should be forced to wait

    for agv, tasks in agv_tasks.items():
        if tasks and tasks[0]:
            starting_node = tasks[0][0]
            resource_states[starting_node] = agv
            agv_history[agv] = [[starting_node]]

    conflict_free_sequences = []
    waiting_agvs = {}
    waiting_order = []
    max_iterations = 1000
    iteration = 0
    deadlock_log = []
    full_iteration_log = []
    while any(tasks for tasks in agv_tasks.values()) and iteration < max_iterations:
        iteration += 1
        moved_this_round = False
        frame_actions = {}

        print(f"\n--- Iteration {iteration} ---")
        print(f"Backtracked AGVs: {backtracked_agvs}")
        print(f"Forced Waiting AGVs: {forced_waiting_agvs}")

        # Check if any backtracked AGV can resume normal operation
        if backtracked_agvs:
            agvs_to_resume = check_backtracked_agv_resume(backtracked_agvs, resource_states)
            for agv in agvs_to_resume:
                print(f"Resuming normal operation for {agv}")

                # NEW: When backtracking completes, record resolution info and cut frames
                resolution_node = backtracked_agvs[agv]['resolution_node']
                conflict_node = backtracked_agvs[agv]['original_conflict_node']

                # Find the frame where this AGV first moved from resolution node
                frame_to_cut = find_frame_to_cut(conflict_free_sequences, agv, resolution_node)

                if frame_to_cut != -1:
                    print(f"Cutting frames from {frame_to_cut} onwards for {agv}")
                    deadlock_log.append({
                        'type': 'resolution',
                        'iteration': iteration,
                        'frame_cut_at': frame_to_cut,
                        'agv': agv,
                        'resolution_node': resolution_node,
                        'conflict_node': conflict_node
                    })

                    # Cut the sequences
                    conflict_free_sequences = conflict_free_sequences[:frame_to_cut]

                    # Add forced waiting rule
                    forced_waiting_agvs[agv] = {
                        'wait_at': resolution_node,
                        'until_occupied': conflict_node
                    }
                    print(
                        f"Added forced waiting: {agv} will wait at {resolution_node} until {conflict_node} is occupied")

                del backtracked_agvs[agv]
                moved_this_round = True

        # Move each AGV if possible
        for agv, tasks in agv_tasks.items():
            if agv in backtracked_agvs:
                current_node = agv_history[agv][-1][-1]
                frame_actions[agv] = (current_node, current_node, 'backtracked')
                continue

            if tasks and len(tasks[0]) >= 1:
                current_node = tasks[0][0]

                if len(tasks[0]) == 1 and len(tasks) > 1:
                    next_node = tasks[1][0]
                elif len(tasks[0]) == 1:
                    frame_actions[agv] = (current_node, current_node, 'completed')
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

                # Check if AGV can move (now includes forced waiting check)
                if can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node, resource_states,
                            forced_waiting_agvs):
                    resource_states[current_node] = 0
                    resource_states[next_node] = agv
                    tasks[0].pop(0)
                    subtask_completed = not tasks[0]

                    if len(tasks[0]) != 1:
                        agv_history[agv][-1].append(next_node)

                    if subtask_completed:
                        tasks.pop(0)
                        if tasks:
                            agv_history[agv].append([tasks[0][0]])

                    frame_actions[agv] = (current_node, next_node, 'move')
                    print(f"{agv} moves from {current_node} to {next_node}")
                    moved_this_round = True
                else:
                    if agv not in waiting_agvs:
                        waiting_agvs[agv] = {
                            'current_node': current_node,
                            'next_node': next_node
                        }
                        waiting_order.append(agv)
                        print(f"{agv} added to waiting list (position {len(waiting_order)})")
                    else:
                        waiting_agvs[agv].update({
                            'current_node': current_node,
                            'next_node': next_node
                        })

                    frame_actions[agv] = (current_node, current_node, 'wait')
                    print(f"{agv} waiting at {current_node}")
        # Record to full log AFTER all deadlock/backtracking/resolution logic
        is_deadlock_this_iter = False
        deadlock_conflict_node = None
        deadlock_agvs_list = None

        if frame_actions:
            full_iteration_log.append({
                'iteration': iteration,
                'frame_actions': copy.deepcopy(frame_actions),
                'is_backtracking': bool(backtracked_agvs),
                'is_deadlock': False,
                'backtracked_agvs': dict(backtracked_agvs),
                'forced_waiting': dict(forced_waiting_agvs)
            })
        if frame_actions:

            conflict_free_sequences.append(frame_actions)

        if waiting_agvs or backtracked_agvs:
            conflicts = detect_deadlock(waiting_agvs, waiting_order, agv_tasks)
            print(f"Conflicts detected: {conflicts}")

            if conflicts:
                for conflict_node, conflicting_agvs in conflicts.items():
                    last_agv = conflicting_agvs[-1]
                    print(f"New deadlock detected at {conflict_node}. {last_agv} will start backtracking.")
                    # Mark this iteration as deadlock in full log
                    if full_iteration_log:
                        full_iteration_log[-1]['is_deadlock'] = True
                        full_iteration_log[-1]['conflict_node'] = conflict_node
                        full_iteration_log[-1]['conflicting_agvs'] = list(conflicting_agvs)

                    deadlock_log.append({
                        'type': 'deadlock',
                        'iteration': iteration,
                        'conflict_node': conflict_node,
                        'conflicting_agvs': list(conflicting_agvs),
                        'backtrack_agv': last_agv
                    })
                    start_backtracking(last_agv, conflict_node, waiting_agvs, waiting_order, agv_tasks, resource_states,
                                       agv_history, backtracked_agvs, resolution_points)
                    moved_this_round = True

            elif backtracked_agvs:
                agvs_to_resume = check_backtracked_agv_resume(backtracked_agvs, resource_states)
                for agv in agvs_to_resume:
                    print(f"Resuming normal operation for {agv}")

                    # NEW: Resolution handling (same as above)
                    resolution_node = backtracked_agvs[agv]['resolution_node']
                    conflict_node = backtracked_agvs[agv]['original_conflict_node']
                    frame_to_cut = find_frame_to_cut(conflict_free_sequences, agv, resolution_node)

                    if frame_to_cut != -1:
                        print(f"Cutting frames from {frame_to_cut} onwards for {agv}")
                        deadlock_log.append({
                            'type': 'resolution',
                            'iteration': iteration,
                            'frame_cut_at': frame_to_cut,
                            'agv': agv,
                            'resolution_node': resolution_node,
                            'conflict_node': conflict_node
                        })
                        conflict_free_sequences = conflict_free_sequences[:frame_to_cut]
                        forced_waiting_agvs[agv] = {
                            'wait_at': resolution_node,
                            'until_occupied': conflict_node
                        }
                        print(
                            f"Added forced waiting: {agv} will wait at {resolution_node} until {conflict_node} is occupied")

                    del backtracked_agvs[agv]
                    moved_this_round = True

                if backtracked_agvs:
                    for backtracked_agv, info in backtracked_agvs.items():
                        backtrack_further(backtracked_agv, agv_tasks, resource_states, agv_history, backtracked_agvs)
            else:
                waiting_agvs.clear()
                waiting_order.clear()

        if not moved_this_round and not waiting_agvs and not backtracked_agvs:
            break

    return conflict_free_sequences, deadlock_log, full_iteration_log



# ============= SIMULATION EXECUTION =============
print("Starting digital twin simulation...")
conflict_free_sequences, deadlock_log, full_iteration_log = simulate_digital_twin()


print(f"Simulation complete. Generated {len(conflict_free_sequences)} frames.")

print("\n=== FULL ITERATION LOG DEBUG ===")
for i, entry in enumerate(full_iteration_log):
    fw = entry['forced_waiting']
    bt = entry['is_backtracking']
    dl = entry['is_deadlock']
    actions = {agv: act for agv, (f, t, act) in entry['frame_actions'].items()}
    if fw or bt or dl:
        print(f"Index {i} | Iter {entry['iteration']} | Actions: {actions} | Deadlock: {dl} | Backtracking: {bt} | ForcedWait: {fw}")

# ============= VISUALIZATION SETUP =============
fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.kamada_kawai_layout(G)

agv_positions = {}
for agv, tasks in original_agv_tasks.items():
    if tasks and tasks[0]:
        agv_positions[agv] = tasks[0][0]


def build_position_history(conflict_free_sequences):
    """Build position history for each AGV from conflict-free sequences (frame-based)."""
    position_history = {agv: [] for agv in original_agv_tasks.keys()}

    for frame in conflict_free_sequences:
        for agv, (from_node, to_node, action) in frame.items():
            position_history[agv].append(to_node)

    return position_history


print(build_position_history(conflict_free_sequences), "POSITION HISTORY")


def update(frame):
    """
    Update function for animation - now handles all AGVs per frame.
    """
    ax.clear()
    nx.draw(G, pos, with_labels=True, arrows=True, ax=ax, node_color='lightblue',
            node_size=300, font_size=8, font_weight='bold')

    if frame < len(conflict_free_sequences):
        frame_actions = conflict_free_sequences[frame]

        status_text = []
        for agv, (from_node, to_node, action) in frame_actions.items():
            agv_positions[agv] = to_node

            action_text = "WAITING" if action == 'wait' else "MOVING" if action == 'move' else action.upper()
            status_text.append(f"{agv}: {action_text} at {to_node}")

        for agv_name, position in agv_positions.items():
            if position is not None:
                color = 'red' if agv_name == 'AGV1' else 'green' if agv_name == 'AGV2' else 'blue'
                nx.draw_networkx_nodes(
                    G, pos, nodelist=[position],
                    node_color=color, node_size=500, ax=ax
                )

        ax.set_title(f"Frame {frame}: " + " | ".join(status_text), fontsize=10, fontweight='bold')
        print(f"Animation Frame {frame}: {status_text}")


print(f"Creating animation with {len(conflict_free_sequences)} frames...")

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(conflict_free_sequences),
    interval=1000,
    repeat=True,
    blit=False
)

plt.show()


# =============================================
# ADD THE TWO GRAPHS BELOW HERE
# =============================================

# ---- GRAPH 1: Network Layout with AGV Starting Positions ----
fig1, ax1 = plt.subplots(figsize=(12, 8))
pos = nx.kamada_kawai_layout(G)

nx.draw(G, pos, with_labels=True, arrows=True, ax=ax1, node_color='lightblue',
        node_size=300, font_size=8, font_weight='bold')

nx.draw_networkx_nodes(G, pos, nodelist=[1], node_color='red', node_size=500, ax=ax1, label='AGV1')
nx.draw_networkx_nodes(G, pos, nodelist=[17], node_color='green', node_size=500, ax=ax1, label='AGV2')
nx.draw_networkx_nodes(G, pos, nodelist=[3], node_color='blue', node_size=500, ax=ax1, label='AGV3')

ax1.legend(fontsize=10)
ax1.set_title("Facility Layout Graph with AGV Starting Positions", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("graph_layout.png", dpi=150)
plt.show()

# ---- GRAPH 2: FULL Iteration Timeline with Algorithmic Detection ----
full_position_history = {'AGV1': [], 'AGV2': [], 'AGV3': []}

for entry in full_iteration_log:
    frame = entry['frame_actions']
    for agv in full_position_history.keys():
        if agv in frame:
            _, to_node, _ = frame[agv]
            full_position_history[agv].append(to_node)
        elif full_position_history[agv]:
            full_position_history[agv].append(full_position_history[agv][-1])

fig2, ax2 = plt.subplots(figsize=(16, 6))
agv_colors = {'AGV1': 'red', 'AGV2': 'green', 'AGV3': 'blue'}

for agv, positions in full_position_history.items():
    ax2.plot(range(len(positions)), positions, marker='o', markersize=3,
             label=agv, color=agv_colors[agv], linewidth=1.5)

y_max = max(max(p) for p in full_position_history.values())
y_min = min(min(p) for p in full_position_history.values())

# ========== ALGORITHMIC DETECTION ==========

# 1. Detect deadlock indices
deadlock_indices = []
for i, entry in enumerate(full_iteration_log):
    if entry['is_deadlock']:
        deadlock_indices.append(i)

# 2. Detect backtracking indices
backtrack_indices = []
for i, entry in enumerate(full_iteration_log):
    if entry['is_backtracking']:
        backtrack_indices.append(i)

# 3. Detect forced waiting indices
# Debug: print what's actually in the frame_actions for forced waiting entries
print("\n=== FORCED WAITING DEBUG ===")
forced_wait_indices = []
for i, entry in enumerate(full_iteration_log):
    if entry['forced_waiting']:
        for agv, wait_info in entry['forced_waiting'].items():
            if agv in entry['frame_actions']:
                frame_data = entry['frame_actions'][agv]
                print(f"Index {i} | Iter {entry['iteration']} | {agv} frame_data: {frame_data} | wait_info: {wait_info}")

# ========== DRAW EVENTS ==========

# Draw deadlock zones
for idx in deadlock_indices:
    ax2.axvspan(idx - 0.5, idx + 0.5, alpha=0.3, color='red')

# Draw backtracking zones (merge consecutive)
if backtrack_indices:
    bt_start = backtrack_indices[0]
    for j in range(1, len(backtrack_indices)):
        if backtrack_indices[j] != backtrack_indices[j - 1] + 1:
            ax2.axvspan(bt_start - 0.5, backtrack_indices[j - 1] + 0.5, alpha=0.2, color='orange')
            bt_start = backtrack_indices[j]
    ax2.axvspan(bt_start - 0.5, backtrack_indices[-1] + 0.5, alpha=0.2, color='orange')

# Resolution: first non-backtracking index after backtracking ends
if backtrack_indices:
    last_bt = backtrack_indices[-1]
    resolution_idx = last_bt + 1
    if resolution_idx < len(full_iteration_log):
        ax2.axvline(x=resolution_idx, color='limegreen', linestyle='--', linewidth=2, alpha=0.8)

# Placeholder for forced waiting - will fix after seeing debug output
# (leaving graph without forced waiting for now)
# ===== FORCED WAITING DETECTION AND DRAWING =====
# ===== 4. FORCED WAITING =====
"""forced_wait_indices = []
for i, entry in enumerate(full_iteration_log):
    if entry['forced_waiting'] and not entry['is_backtracking']:
        frame = entry['frame_actions']
        for agv, wait_info in entry['forced_waiting'].items():
            if agv in frame:
                from_node, to_node, action = frame[agv]
                if action == 'wait':
                    forced_wait_indices.append(i)
                    break

print(f"FORCED WAIT INDICES FOUND: {forced_wait_indices}")

if forced_wait_indices:
    fw_start = forced_wait_indices[0]
    for j in range(1, len(forced_wait_indices)):
        if forced_wait_indices[j] != forced_wait_indices[j - 1] + 1:
            ax2.axvspan(fw_start - 0.5, forced_wait_indices[j - 1] + 0.5, alpha=0.2, color='gold')
            fw_start = forced_wait_indices[j]
    ax2.axvspan(fw_start - 0.5, forced_wait_indices[-1] + 0.5, alpha=0.2, color='gold')

    mid_fw = forced_wait_indices[len(forced_wait_indices) // 2]
    wait_node = full_position_history['AGV1'][mid_fw]
    ax2.annotate(f'AGV1 forced wait\nat node {wait_node}\n(until path to node 24 clear)',
                 xy=(mid_fw, wait_node), fontsize=7, fontstyle='italic', color='goldenrod', ha='center',
                 arrowprops=dict(arrowstyle='->', color='goldenrod', lw=1),
                 xytext=(mid_fw + 5, wait_node + 3))"""


# ========== ANNOTATIONS ==========
if deadlock_indices:
    idx = deadlock_indices[0]
    entry = full_iteration_log[idx]
    ax2.annotate(
        f"Circular Deadlock\nat node {entry.get('conflict_node', '?')}\n{', '.join(entry.get('conflicting_agvs', []))}",
        xy=(idx, y_max), fontsize=8, fontweight='bold', color='darkred', ha='center',
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
        xytext=(idx - 5, y_max + 2))

if backtrack_indices:
    mid = backtrack_indices[len(backtrack_indices) // 2]
    ax2.annotate('Backtracking Phase',
                 xy=(mid, full_position_history['AGV1'][mid]),
                 fontsize=7, fontstyle='italic', color='darkorange', ha='center',
                 arrowprops=dict(arrowstyle='->', color='darkorange', lw=1),
                 xytext=(mid - 1, y_min + 2))

if backtrack_indices:
    res_idx = backtrack_indices[-1] + 1
    if res_idx < len(full_iteration_log):
        ax2.annotate('Resolution\n(Frame cut applied)',
                     xy=(res_idx, y_max), fontsize=8, fontweight='bold', color='darkgreen', ha='left',
                     arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
                     xytext=(res_idx + 2, y_max - 2))

# ========== LEGEND ==========
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='red', linewidth=1.5, label='AGV1'),
    Line2D([0], [0], color='green', linewidth=1.5, label='AGV2'),
    Line2D([0], [0], color='blue', linewidth=1.5, label='AGV3'),
    Patch(facecolor='red', alpha=0.3, label='Circular Deadlock Detected'),
    Patch(facecolor='orange', alpha=0.2, label='Backtracking Phase'),
    #Patch(facecolor='gold', alpha=0.2, label='Forced Waiting Phase'),
    Line2D([0], [0], color='limegreen', linestyle='--', linewidth=2, label='Resolution Point'),
]

ax2.legend(handles=legend_elements, fontsize=9, loc='upper right')
ax2.set_xlabel("Iteration (Time Step)", fontsize=11)
ax2.set_ylabel("Node Position", fontsize=11)
ax2.set_title("Full Simulation Timeline — Circular Deadlock Detection, Backtracking & Resolution",
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("agv_full_timeline_no_cut.png", dpi=150)
plt.show()
