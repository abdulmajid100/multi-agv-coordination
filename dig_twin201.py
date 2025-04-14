import simpy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle
from collections import defaultdict

# Store AGV positions and states
agv_positions = {}
agv_states = {}
segment_occupancy = defaultdict(lambda: None)
time_history = []
position_history = defaultdict(list)
events = []

# Create layout information for segments
segment_layout = {
    1: {'x': 0, 'y': 0, 'length': 10},
    2: {'x': 10, 'y': 0, 'length': 15},
    3: {'x': 25, 'y': 0, 'length': 20}
}


def agv(env, name, path, speed, color):
    """AGV process representing vehicle movement"""
    events.append((env.now, f'{name} starting at {env.now:.2f}'))

    # Initialize AGV state
    agv_states[name] = {'segment': None, 'position': 0, 'color': color}

    current_x = 0  # Starting position

    for segment in path:
        # Request access to path segment
        with segment.request() as req:
            yield req

            # Mark segment as occupied by this AGV
            segment_occupancy[segment.id] = name
            agv_states[name]['segment'] = segment.id

            # Set initial position for this segment
            if segment.id == 1:
                current_x = 0
            else:
                current_x = segment_layout[segment.id]['x']

            agv_states[name]['position'] = current_x

            # Calculate travel time based on segment length and AGV speed
            travel_time = segment.length / speed
            events.append((env.now, f'{name} entering segment {segment.id} at {env.now:.2f}'))

            # Record position at start
            time_history.append(env.now)
            for agv_name, state in agv_states.items():
                position_history[agv_name].append((env.now, state['segment'], state['position']))

            # Simulate travel time with small increments for animation
            steps = 20  # Number of animation steps
            for step in range(steps):
                step_time = travel_time / steps
                distance_per_step = segment.length / steps

                # Update AGV position
                current_x += distance_per_step
                agv_states[name]['position'] = current_x

                # Record position history
                time_history.append(env.now + step * step_time)
                for agv_name, state in agv_states.items():
                    position_history[agv_name].append((env.now + step * step_time,
                                                       state['segment'],
                                                       state['position']))

                yield env.timeout(step_time)

            events.append((env.now, f'{name} exiting segment {segment.id} at {env.now:.2f}'))
            segment_occupancy[segment.id] = None

    events.append((env.now, f'{name} reached destination at {env.now:.2f}'))
    agv_states[name]['segment'] = None


# Create simulation environment
def run_simulation():
    env = simpy.Environment()

    # Create path segments as resources with capacity 1 (only one AGV at a time)
    class PathSegment:
        def __init__(self, env, id, length, capacity=1):
            self.id = id
            self.length = length  # in meters
            self.resource = simpy.Resource(env, capacity)

        def request(self):
            return self.resource.request()

    # Create path segments
    segments = [
        PathSegment(env, 1, 10),  # 10 meters
        PathSegment(env, 2, 15),  # 15 meters
        PathSegment(env, 3, 20)  # 20 meters
    ]

    # Create multiple AGVs with different colors
    colors = ['red', 'blue', 'green']
    for i in range(3):
        speed = random.uniform(1.0, 1.1)  # meters per second
        env.process(agv(env, f'AGV-{i + 1}', segments, speed, colors[i]))

    # Run the simulation
    env.run(until=100)  # Run for 100 time units

    return time_history, position_history, events


# Run simulation to collect data
time_history, position_history, events = run_simulation()

# Create animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                               gridspec_kw={'height_ratios': [3, 1]})

# Draw the path segments
for seg_id, layout in segment_layout.items():
    rect = Rectangle((layout['x'], layout['y'] - 0.5),
                     layout['length'], 1,
                     facecolor='lightgray', edgecolor='black', alpha=0.5)
    ax1.add_patch(rect)
    ax1.text(layout['x'] + layout['length'] / 2, layout['y'] - 1,
             f"Segment {seg_id}", ha='center')

# Set up animation display
agv_markers = {}
agv_text = {}
max_x = sum(seg['length'] for seg in segment_layout.values()) + 5

# Set up axes
ax1.set_xlim(-5, max_x)
ax1.set_ylim(-2, 2)
ax1.set_title("AGV Traffic Simulation")
ax1.set_xlabel("Distance (meters)")
ax1.set_yticks([])

# Event log text box
event_text = ax2.text(0.05, 0.5, "", transform=ax2.transAxes,
                      verticalalignment='center', fontsize=10,
                      bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})
ax2.set_axis_off()

# Unique time points for animation
unique_times = sorted(list(set(time_history)))

# Find position of each AGV at each time point
agv_positions_by_time = {}
for t in unique_times:
    agv_positions_by_time[t] = {}
    for agv_name in position_history.keys():
        # Get the latest position before or at time t
        valid_positions = [(time, seg, pos) for time, seg, pos in position_history[agv_name] if time <= t]
        if valid_positions:
            agv_positions_by_time[t][agv_name] = valid_positions[-1]
        else:
            agv_positions_by_time[t][agv_name] = (0, None, 0)


# Function to initialize animation
def init():
    for agv_name in position_history.keys():
        color = agv_states[agv_name]['color']
        marker, = ax1.plot([], [], 'o', color=color, markersize=10)
        agv_markers[agv_name] = marker
        text = ax1.text(0, 0, agv_name, fontsize=8, ha='center', va='bottom')
        agv_text[agv_name] = text
    return list(agv_markers.values()) + list(agv_text.values()) + [event_text]


# Animation update function
def update(frame):
    t = unique_times[min(frame, len(unique_times) - 1)]

    # Update AGV positions
    for agv_name, marker in agv_markers.items():
        if agv_name in agv_positions_by_time[t]:
            _, segment, position = agv_positions_by_time[t][agv_name]
            if segment is not None:
                marker.set_data([position], [0])
                agv_text[agv_name].set_position((position, 0.2))
                marker.set_visible(True)
                agv_text[agv_name].set_visible(True)
            else:
                marker.set_visible(False)
                agv_text[agv_name].set_visible(False)

    # Update event log
    current_events = [e[1] for e in events if e[0] <= t]
    if current_events:
        # Show last 5 events
        log_text = "\n".join(current_events[-5:])
        event_text.set_text(f"Simulation Time: {t:.2f}\nEvent Log:\n{log_text}")

    return list(agv_markers.values()) + list(agv_text.values()) + [event_text]


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(unique_times),
                              init_func=init, blit=True, interval=50)

plt.tight_layout()
plt.show()

# To save the animation (uncomment to use)
# ani.save('agv_simulation.mp4', writer='ffmpeg', fps=20)
