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
        [20, 10, 11, 4, 5, 4, 6],
        [6, 4, 11, 10, 20],
        [20, 10, 11, 4, 1]
    ],
    'AGV2': [
        [11, 12, 13, 23],
        [23, 13, 12, 11, 4, 5, 4, 6],
        [6, 4, 11, 12, 13, 23],
        [23, 13, 12, 11, 4, 2]
    ],
    'AGV3': [
        [3, 4, 11, 12, 13, 14, 15, 16, 26],
        [26, 16, 15, 14, 13, 12, 11, 4, 5, 4, 6],
        [6, 4, 11, 12, 13, 14, 15, 16, 26],
        [26, 16, 15, 14, 13, 12, 11, 4, 3]
    ]
}

# Initialize the reservation status of each node
resource_states = {node: 0 for node in G.nodes()}

# Set initial nodes as reserved
for agv, tasks in agv_tasks.items():
    if tasks and tasks[0]:  # Check if there are tasks and nodes in the first task
        starting_node = tasks[0][0]
        resource_states[starting_node] = agv

def calculate_shared_nodes():
    for agv, tasks in agv_tasks.items():
        other_agvs = [other_agv for other_agv in agv_tasks if other_agv != agv]
        shared_nodes_with_others = {other_agv: [] for other_agv in other_agvs}

        for other_agv in other_agvs:
            # Calculate shared nodes only for the current tasks
            if tasks and agv_tasks[other_agv]:
                current_task = tasks[0]
                other_current_task = agv_tasks[other_agv][0]
                shared_nodes = set(current_task) & set(other_current_task)
                shared_nodes_with_others[other_agv] = list(shared_nodes)

        print(f"Shared nodes for {agv} with others:")
        for other_agv, shared_nodes in shared_nodes_with_others.items():
            print(f"  With {other_agv}: {shared_nodes}")

calculate_shared_nodes()

# Function to check if an AGV can move to the next node
def can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
    if all(next_node not in shared_nodes_with_others[other_agv] for other_agv in other_agvs):
        return True
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
                node_color='r' if agv == 'AGV1' else 'g' if agv == 'AGV2' else 'b',
                node_size=500, ax=ax
            )

    # Skip movement on the first frame to show initial positions
    if frame == 0:
        return

    # Move each AGV if possible
    for agv, tasks in agv_tasks.items():
        if tasks and len(tasks[0]) > 1:  # Check if there are tasks and nodes left in the current task
            other_agvs = [other_agv for other_agv in agv_tasks if other_agv != agv]
            shared_nodes_with_others = {other_agv: [] for other_agv in other_agvs}

            for other_agv in other_agvs:
                # Calculate shared nodes only for the current tasks
                if tasks and agv_tasks[other_agv]:
                    current_task = tasks[0]
                    other_current_task = agv_tasks[other_agv][0]
                    shared_nodes = set(current_task) & set(other_current_task)
                    shared_nodes_with_others[other_agv] = list(shared_nodes)

            current_node = tasks[0][0]
            next_node = tasks[0][1]
            print(tasks)
            print(shared_nodes_with_others)
            print(resource_states)
            if can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
                # Reserve the next node for the AGV
                resource_states[current_node] = 0

                resource_states[next_node] = agv
                # Release the current node

                # Move AGV to next node
                tasks[0].pop(0)  # Remove the current node from the current task
                print(f"{agv} moves from {current_node} to {next_node}")
            else:
                print(f"{agv} waiting at {current_node}")
        elif tasks:  # If the current task is completed, move to the next task
            last_node = tasks[0][-1]  # Get the last node of the completed task
            tasks.pop(0)  # Remove the completed task
            if tasks:  # If there are more tasks
                tasks[0].insert(0, last_node)  # Insert the last node of the previous task as the starting node

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(100), repeat=False, interval=1000)

plt.show()