import simpy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from collections import defaultdict


class AGVSimulation:
    def __init__(self):
        self.env = simpy.Environment()
        self.graph = nx.DiGraph()
        self.segments = {}
        self.position_history = defaultdict(list)
        self.times = []
        self.events = []

    def create_graph_layout(self):
        # Create nodes (locations/intersections)
        nodes = {'A': (0, 0), 'B': (10, 0), 'C': (10, 10),
                 'D': (20, 0), 'E': (20, 10), 'F': (30, 5)}

        for node, pos in nodes.items():
            self.graph.add_node(node, pos=pos)

        # Create edges with distances
        edges = [('A', 'B', 10), ('B', 'C', 10), ('B', 'D', 10),
                 ('C', 'E', 10), ('D', 'F', 12), ('E', 'F', 12)]

        for start, end, length in edges:
            edge_id = f"{start}-{end}"
            self.graph.add_edge(start, end, id=edge_id, length=length)
            self.segments[edge_id] = PathSegment(self.env, edge_id, length)

    def agv_process(self, name, route, speed, color):
        """Process for an AGV traversing a route in the graph"""
        self.events.append((self.env.now, f'{name} starting at {self.env.now:.2f}'))

        # Initialize AGV position at the first node
        start_node = route[0]
        node_pos = self.graph.nodes[start_node]['pos']
        self.position_history[name].append({
            'time': self.env.now,
            'position': node_pos,
            'segment': None
        })
        self.times.append(self.env.now)

        # Traverse each edge in the route
        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

            # Get the edge ID and data
            edge_id = f"{current_node}-{next_node}"

            # Request access to the segment
            segment = self.segments.get(edge_id)
            if segment:
                with segment.request() as req:
                    yield req

                    # Get positions for animation
                    start_pos = self.graph.nodes[current_node]['pos']
                    end_pos = self.graph.nodes[next_node]['pos']
                    edge_data = self.graph.get_edge_data(current_node, next_node)
                    length = edge_data['length']
                    travel_time = length / speed

                    self.events.append((self.env.now,
                                        f'{name} entering segment {edge_id} at {self.env.now:.2f}'))

                    # Simulate movement with smaller steps for animation
                    steps = 20
                    for step in range(steps + 1):
                        current_time = self.env.now + (step * travel_time / steps)
                        progress = step / steps

                        # Calculate position along the path
                        x = start_pos[0] + progress * (end_pos[0] - start_pos[0])
                        y = start_pos[1] + progress * (end_pos[1] - start_pos[1])

                        # Record position
                        self.position_history[name].append({
                            'time': current_time,
                            'position': (x, y),
                            'segment': edge_id
                        })
                        self.times.append(current_time)

                        if step < steps:
                            yield self.env.timeout(travel_time / steps)

                    self.events.append((self.env.now,
                                        f'{name} exiting segment {edge_id} at {self.env.now:.2f}'))
            else:
                self.events.append((self.env.now,
                                    f'{name} cannot find path from {current_node} to {next_node}'))

        self.events.append((self.env.now, f'{name} reached destination at {self.env.now:.2f}'))

    def run_simulation(self):
        # Create the graph layout
        self.create_graph_layout()

        # Define routes for AGVs
        routes = [
            ['A', 'B', 'D', 'F', 'E', 'C'],  # Route for AGV-1
            ['A', 'B', 'C', 'E'],  # Route for AGV-2
            ['A', 'B', 'C']  # Route for AGV-3 (starts later)
        ]

        # Create AGVs with different colors and speeds
        colors = ['red', 'blue', 'green']
        for i in range(3):
            speed = random.uniform(1.0, 1.01)  # meters per second
            # Add a slight delay for each AGV to see them separate more clearly
            yield self.env.timeout(i * 3)
            route = routes[i]
            self.env.process(self.agv_process(f'AGV-{i + 1}', route, speed, colors[i]))

    def animate(self):
        """Create an animation of the simulation"""
        # Sort and get unique time points
        self.times = sorted(set(self.times))

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw the graph network
        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw_networkx(self.graph, pos, with_labels=True, node_size=500,
                         node_color='lightblue', font_weight='bold', ax=ax)

        # Add edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'id')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, ax=ax)

        # Create event text display
        event_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9,
                             bbox=dict(facecolor='white', alpha=0.7))

        # Set up AGV markers with corresponding colors
        agv_markers = {}
        agv_colors = {'AGV-1': 'red', 'AGV-2': 'blue', 'AGV-3': 'green'}

        for name, color in agv_colors.items():
            marker, = ax.plot([], [], 'o', color=color, markersize=10, label=name)
            agv_markers[name] = marker

        ax.legend()

        # Get axis limits from node positions
        x_values = [p[0] for p in pos.values()]
        y_values = [p[1] for p in pos.values()]
        x_min, x_max = min(x_values) - 2, max(x_values) + 2
        y_min, y_max = min(y_values) - 2, max(y_values) + 2
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Animation function
        def update(frame):
            current_time = self.times[min(frame, len(self.times) - 1)]

            # Update AGV positions
            for name, marker in agv_markers.items():
                positions = self.position_history.get(name, [])
                # Find positions at or before current_time
                valid_positions = [p for p in positions if p['time'] <= current_time]

                if valid_positions:
                    # Get the most recent position
                    latest = valid_positions[-1]
                    x, y = latest['position']
                    marker.set_data([x], [y])
                    marker.set_visible(True)
                else:
                    marker.set_visible(False)

            # Update event text
            current_events = [e[1] for e in self.events if e[0] <= current_time]
            if current_events:
                # Show the last 5 events
                text = f"Time: {current_time:.2f}\n"
                text += "\n".join(current_events[-5:])
                event_text.set_text(text)

            return list(agv_markers.values()) + [event_text]

        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.times),
            interval=100, blit=True)

        plt.title("AGV Traffic Simulation")
        plt.tight_layout()
        plt.show()

        return ani  # Return the animation object to prevent garbage collection


# PathSegment class
class PathSegment:
    def __init__(self, env, id, length, capacity=1):
        self.id = id
        self.length = length
        self.resource = simpy.Resource(env, capacity)

    def request(self):
        return self.resource.request()


# Run the simulation
def main():
    sim = AGVSimulation()
    env = sim.env

    # Start the simulation processes
    env.process(sim.run_simulation())

    # Run the simulation
    env.run(until=100)

    # Create and show the animation
    animation = sim.animate()

    return sim, animation


# Execute the simulation
sim, animation = main()
