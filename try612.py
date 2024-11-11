import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
from threading import Thread

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
        [22, 12, 11, 4, 1],
        [1, 4, 11, 10, 20],
        [20, 10, 11, 4, 5, 4, 6],
        [6, 4, 11, 10, 20]
    ],
    'AGV2': [
        [13, 14, 24],
        [24, 14, 13, 12, 11, 4, 5, 4, 6],
        [6, 4, 11, 12, 13, 14, 24],
        [24, 14, 13, 12, 11, 4, 2],
        [2, 4, 11, 12, 13, 14, 15, 25],
        [25, 15, 14, 13, 12, 11, 4, 5, 4, 6],
        [6, 4, 11, 12, 13, 14, 15, 16, 26]
    ],
    'AGV3': [
        [17, 18, 28],
        [28, 18, 17, 16, 15, 14, 13, 12, 11, 4, 5, 4, 6],
        [6, 4, 11, 12, 13, 14, 15, 16, 17, 18, 28],
        [28, 18, 17, 16, 15, 14, 13, 12, 11, 4, 3],
        [3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29]
    ],
    'GEN': [
        [23, 13, 14, 15, 16, 9],
        [9, 16, 17, 27]
    ]
}

# Function to get tasks as strings for dropdown menus
def get_task_strings(tasks):
    task_strings = []
    for idx, task in enumerate(tasks):
        task_strings.append(f'Task {idx+1}: {task}')
    return task_strings

# Initialize the reservation status of each node
resource_states = {node: 0 for node in G.nodes()}

# Initialize plot variables
fig, ax = plt.subplots()
pos = nx.kamada_kawai_layout(G)

# AGV colors for plotting
agv_colors = {
    'AGV1': 'r',
    'AGV2': 'g',
    'AGV3': 'b',
    'GEN': 'orange'
}

# Create the Tkinter GUI
root = tk.Tk()
root.title("AGV Path Selection")

# Variables to store selected paths
selected_tasks = {}
start_times = {}
agv_paths = {}

# Function to start the animation
def start_animation():
    # Collect selected tasks and start times
    for agv in agv_tasks:
        task_str = selected_tasks[agv].get()
        if task_str != 'None':
            task_idx = int(task_str.split(':')[0].split()[1]) - 1
            agv_paths[agv] = agv_tasks[agv][task_idx][:]
            start_times[agv] = int(agv_start_times[agv].get())
        else:
            agv_paths[agv] = []
            start_times[agv] = None

    # Close the GUI window
    root.destroy()

    # Start the animation in a new thread to prevent blocking
    animation_thread = Thread(target=animate)
    animation_thread.start()

# Create dropdown menus and start time entries for each AGV
selected_tasks = {}
agv_start_times = {}
row = 0
for agv in agv_tasks:
    tk.Label(root, text=f"{agv} Path:").grid(row=row, column=0, padx=5, pady=5, sticky='e')
    task_var = tk.StringVar(root)
    task_var.set('None')
    task_options = ['None'] + get_task_strings(agv_tasks[agv])
    task_menu = ttk.OptionMenu(root, task_var, *task_options)
    task_menu.grid(row=row, column=1, padx=5, pady=5)
    selected_tasks[agv] = task_var

    tk.Label(root, text=f"{agv} Start Time (s):").grid(row=row, column=2, padx=5, pady=5, sticky='e')
    start_time_var = tk.StringVar(root)
    start_time_var.set('0')
    start_time_entry = tk.Entry(root, textvariable=start_time_var, width=5)
    start_time_entry.grid(row=row, column=3, padx=5, pady=5)
    agv_start_times[agv] = start_time_var

    row += 1

start_button = tk.Button(root, text="Start Animation", command=start_animation)
start_button.grid(row=row, column=0, columnspan=4, pady=10)

# Start the Tkinter event loop
root.mainloop()

# Initialize data structures after GUI input
agv_positions = {}
for agv in agv_paths:
    if agv_paths[agv]:
        agv_positions[agv] = agv_paths[agv][0]
        resource_states[agv_positions[agv]] = agv
    else:
        agv_positions[agv] = None

# Function to calculate shared nodes between AGVs
def calculate_shared_nodes():
    shared_nodes_with_others = {}
    for agv in agv_paths:
        other_agvs = [other_agv for other_agv in agv_paths if other_agv != agv]
        shared_nodes_with_others[agv] = {other_agv: [] for other_agv in other_agvs}

        for other_agv in other_agvs:
            if agv_paths[agv] and agv_paths[other_agv]:
                current_task = agv_paths[agv]
                other_current_task = agv_paths[other_agv]
                shared_nodes = set(current_task) & set(other_current_task)
                shared_nodes_with_others[agv][other_agv] = list(shared_nodes)
    return shared_nodes_with_others

# Function to check if an AGV can move to the next node
def can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
    # Check if next_node is free or reserved by the same AGV
    if resource_states.get(next_node, 0) == 0 or resource_states.get(next_node, 0) == agv:
        # Check for conflicts with other AGVs
        for other_agv in other_agvs:
            if next_node in shared_nodes_with_others[agv][other_agv]:
                # Conflict if other AGV is on the shared node or moving to it
                if agv_positions[other_agv] == next_node or \
                        (agv_positions[other_agv] == current_node and agv_paths[other_agv][1] == next_node):
                    return False
        return True
    return False

# Animation function
def animate():
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=True, ax=ax)
    plt.ion()
    plt.show()

    total_frames = 200
    for frame in range(total_frames):
        ax.clear()
        nx.draw(G, pos, with_labels=True, arrows=True, ax=ax)

        # Draw AGVs at their current positions
        for agv in agv_paths:
            if agv_positions[agv]:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=[agv_positions[agv]],
                    node_color=agv_colors.get(agv, 'black'),
                    node_size=500, ax=ax
                )

        plt.draw()
        plt.pause(0.1)

        # Move AGVs if possible
        for agv in agv_paths:
            # Check if AGV is supposed to start at this time
            if start_times[agv] is not None and frame < (start_times[agv] * 10):
                continue  # AGV hasn't started yet

            if agv_positions[agv] and len(agv_paths[agv]) >= 1:
                current_node = agv_positions[agv]
                if len(agv_paths[agv]) > 1:
                    next_node = agv_paths[agv][1]
                else:
                    next_node = None

                other_agvs = [other_agv for other_agv in agv_paths if other_agv != agv]
                shared_nodes_with_others = calculate_shared_nodes()

                if next_node and can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
                    # Move AGV to next node
                    resource_states[current_node] = 0
                    resource_states[next_node] = agv
                    agv_positions[agv] = next_node
                    agv_paths[agv].pop(0)
                elif not next_node:
                    # AGV has reached the end of its path
                    resource_states[current_node] = 0
                    agv_positions[agv] = None
            else:
                agv_positions[agv] = None

    plt.ioff()
    plt.show()