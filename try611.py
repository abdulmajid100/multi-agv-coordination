import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

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
        [4, 5, 4, 6],
        [6, 4, 11, 12, 22],
        [22, 12, 11, 4, 1]
    ],
    'AGV2': [
        [13, 14, 24],
        [24, 14, 13, 12, 11, 4, 5, 4, 6],
        [6, 4, 11, 12, 13, 14, 24],
        [24, 14, 13, 12, 11, 4, 2]
    ],
    'AGV3': [
        [3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 28],
        [28, 18, 17, 16, 15, 14, 13, 12, 11, 4, 5, 4, 6],
        [6, 4, 11, 12, 13, 14, 15, 16, 17, 18, 28],
        [28, 18, 17, 16, 15, 14, 13, 12, 11, 4, 3]
    ],
    'GEN': [
        [15, 14, 13, 23],
        [23, 13, 14, 15, 16, 9]
    ]
}

# Initialize the reservation status of each node
resource_states = {node: 0 for node in G.nodes()}

# Track how long each AGV has been waiting
waiting_counters = {agv: 0 for agv in agv_tasks.keys()}

# Set initial nodes as reserved
for agv, tasks in agv_tasks.items():
    if tasks and tasks[0]:  # Check if there are tasks and nodes in the first task
        starting_node = tasks[0][0]
        resource_states[starting_node] = agv


def is_swap_conflict(agv, current_node, next_node, other_agvs):
    """Check if moving to next_node would cause a swap conflict with another AGV"""
    for other_agv in other_agvs:
        if agv_tasks[other_agv] and len(agv_tasks[other_agv][0]) > 1:
            other_current = agv_tasks[other_agv][0][0]
            other_next = agv_tasks[other_agv][0][1]
            # Check if AGVs are trying to swap positions
            if other_current == next_node and other_next == current_node:
                return True
    return False

def find_alternative_path(agv, start_node, end_node, occupied_nodes):
    """Find an alternative path from start_node to end_node avoiding occupied_nodes"""
    # Create a copy of the graph to work with
    temp_graph = G.copy()

    # Remove occupied nodes from the graph temporarily
    for node in occupied_nodes:
        if node != start_node and node != end_node and temp_graph.has_node(node):
            temp_graph.remove_node(node)

    # Try to find a path in the modified graph
    try:
        path = nx.shortest_path(temp_graph, source=start_node, target=end_node)
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # If no path exists, return None
        return None

def can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
    # Check if the next node is already occupied by another AGV
    if resource_states[next_node] != 0 and resource_states[next_node] != agv:
        return False

    # Check for swap conflicts (two AGVs trying to swap positions)
    if is_swap_conflict(agv, current_node, next_node, other_agvs):
        return False

    # If the next node is not in any shared paths with other AGVs, it's safe to move
    if all(next_node not in shared_nodes_with_others[other_agv] for other_agv in other_agvs):
        return True

    # Check if any other AGV is currently at a shared node that could lead to conflict
    if any(
            next_node in shared_nodes and
            any(resource_states[shared_node] == other_agv for shared_node in shared_nodes)
            for other_agv, shared_nodes in shared_nodes_with_others.items()
    ):
        return False

    return True


# Initialize plot
fig, ax = plt.subplots()
pos = nx.kamada_kawai_layout(G)


# Update function for animation
def update(frame):
    ax.clear()
    nx.draw(G, pos, with_labels=True, arrows=True, ax=ax)

    # Draw AGVs at their current positions
    for agv, tasks in agv_tasks.items():
        if tasks and tasks[0]:  # Check if there are tasks and nodes left in the current task
            current_node = tasks[0][0]
            nx.draw_networkx_nodes(
                G, pos, nodelist=[current_node],
                node_color='r' if agv == 'AGV1' else 'g' if agv == 'AGV2' else 'b' if agv == 'AGV3' else 'orange',
                node_size=500, ax=ax
            )

    # Skip movement on the first frame to show initial positions
    if frame == 0:
        return

    # Check for deadlocks and prioritize AGVs that have been waiting too long
    MAX_WAIT_TIME = 30  # Maximum number of frames an AGV can wait before being prioritized
    SEVERE_DEADLOCK_TIME = 50  # Maximum number of frames before trying alternative paths

    # Find AGVs that have been waiting too long
    deadlocked_agvs = [agv for agv, wait_time in waiting_counters.items() 
                      if wait_time >= MAX_WAIT_TIME and agv_tasks.get(agv) and len(agv_tasks[agv][0]) > 1]

    # Find severely deadlocked AGVs that need alternative paths
    severe_deadlocks = [agv for agv, wait_time in waiting_counters.items() 
                       if wait_time >= SEVERE_DEADLOCK_TIME and agv_tasks.get(agv) and len(agv_tasks[agv][0]) > 1]

    # Try to find alternative paths for severely deadlocked AGVs
    for agv in severe_deadlocks:
        if agv_tasks[agv] and len(agv_tasks[agv][0]) > 1:
            current_node = agv_tasks[agv][0][0]
            # Find the destination node (last node in current task)
            destination_node = agv_tasks[agv][0][-1]

            # Get occupied nodes (nodes with AGVs on them)
            occupied_nodes = [node for node, occupier in resource_states.items() if occupier != 0 and occupier != agv]

            # Try to find an alternative path
            alt_path = find_alternative_path(agv, current_node, destination_node, occupied_nodes)

            if alt_path and len(alt_path) > 1:
                logging.warning(f"Found alternative path for {agv}: {alt_path}")
                # Replace the current task with the alternative path
                agv_tasks[agv][0] = alt_path
                # Reset waiting counter
                waiting_counters[agv] = 0

    # Process deadlocked AGVs first, then others
    if deadlocked_agvs:
        logging.warning(f"Potential deadlock detected. Prioritizing: {deadlocked_agvs}")
        agv_priority_order = deadlocked_agvs + [agv for agv in sorted(agv_tasks.keys()) if agv not in deadlocked_agvs]
    else:
        agv_priority_order = sorted(agv_tasks.keys())
    print(f"AGV priority order: {agv_priority_order}")
    # Process AGVs one by one to determine and execute moves
    for agv in agv_priority_order:
        tasks = agv_tasks[agv]
        print(f"Processing {agv} with tasks: {tasks}")
        if tasks and len(tasks[0]) > 1:  # Check if there are tasks and nodes left in the current task
            #print(f"Processing {agv} with tasks: {tasks}")
            current_node = tasks[0][0]
            next_node = tasks[0][1]
        elif tasks and len(tasks) > 1 and len(tasks[0]) == 1:  # If only one node left in the task
            current_node = tasks[0][0]
            next_node = tasks[1][0]
        elif tasks and len(tasks) == 1 and len(tasks[0]) <= 1:  # If the current task is completed, move to the next task
            next_node = tasks[0][0]
        else:
            continue
            #print(tasks[1])
        other_agvs = [other_agv for other_agv in agv_tasks if other_agv != agv]
        shared_nodes_with_others = {other_agv: [] for other_agv in other_agvs}

        for other_agv in other_agvs:
            if tasks and agv_tasks[other_agv]:
                current_task = tasks[0]
                other_current_task = agv_tasks[other_agv][0]
                if current_task[-1] == other_current_task[-1]:
                    if len(agv_tasks[other_agv]) > 1:
                        other_current_task = agv_tasks[other_agv][0] + agv_tasks[other_agv][1]
                    else:
                        other_current_task = agv_tasks[other_agv][0]
                else:
                    other_current_task = agv_tasks[other_agv][0]
                shared_nodes = set(current_task) & set(other_current_task)
                shared_nodes_with_others[other_agv] = list(shared_nodes)
                if shared_nodes:
                    logging.debug(f"Shared nodes between {agv} and {other_agv}: {shared_nodes_with_others[other_agv]}")

        # Check if the AGV can move and execute the move immediately if possible
        if can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
            # Update resource states
            resource_states[current_node] = 0
            resource_states[next_node] = agv

            # Update AGV task
            agv_tasks[agv][0].pop(0)
            if len(agv_tasks[agv][0]) == 0:
                agv_tasks[agv].pop(0)

            # Reset waiting counter since AGV moved
            waiting_counters[agv] = 0

            logging.info(f"{agv} moves from {current_node} to {next_node}")
        else:
            # Increment waiting counter since AGV couldn't move
            waiting_counters[agv] += 1
            logging.info(f"{agv} waiting at {current_node}, counter: {waiting_counters[agv]}")


# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(200), repeat=False, interval=1000)

plt.show()

# Summary of collision avoidance mechanisms implemented:
# 1. Check if the next node is already occupied by another AGV
# 2. Check for swap conflicts (two AGVs trying to swap positions)
# 3. Process AGVs one by one in a priority order, updating resource states after each move
# 4. Detect deadlocks and prioritize AGVs that have been waiting too long
# 5. Find alternative paths for severely deadlocked AGVs
# 
# These mechanisms work together to prevent AGVs from moving into collision with each other
# by considering the current state of other AGVs when planning movements.
