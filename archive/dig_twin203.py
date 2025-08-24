import simpy
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation  # Add this import
import networkx as nx
import numpy as np
from collections import defaultdict


class AGVSimulation:
    def __init__(self):
        self.env = simpy.Environment()
        self.graph = nx.DiGraph()  # Directed graph
        self.segments = {}
        self.position_history = defaultdict(list)
        self.times = []
        self.events = []

    def create_custom_graph(self):
        """Create a custom graph layout for a warehouse/factory"""

        # Create nodes representing key locations in the facility
        nodes = {
            'Receiving': (0, 0),
            'Storage_A': (20, 0),
            'Storage_B': (20, 15),
            'Storage_C': (40, 15),
            'Assembly': (40, 0),
            'Packing': (60, 0),
            'Shipping': (80, 0),
            'Charging_1': (10, 10),
            'Charging_2': (50, 10),
            'Intersection_1': (20, 10),
            'Intersection_2': (40, 10),
            'Intersection_3': (60, 10),
        }

        # Add nodes to the graph
        for node, pos in nodes.items():
            self.graph.add_node(node, pos=pos)

        # Create edges representing paths between locations
        edges = [
            ('Receiving', 'Storage_A', 20),
            ('Storage_A', 'Assembly', 20),
            ('Assembly', 'Packing', 20),
            ('Packing', 'Shipping', 20),

            ('Receiving', 'Charging_1', 14.1),
            ('Charging_1', 'Storage_B', 14.1),
            ('Storage_B', 'Storage_C', 20),
            ('Storage_C', 'Shipping', 28.3),

            ('Storage_A', 'Intersection_1', 10),
            ('Intersection_1', 'Storage_B', 5),
            ('Storage_B', 'Intersection_2', 20),
            ('Intersection_2', 'Storage_C', 5),
            ('Storage_C', 'Intersection_3', 20),
            ('Intersection_3', 'Shipping', 10),

            ('Intersection_1', 'Charging_1', 10),
            ('Assembly', 'Intersection_2', 10),
            ('Intersection_2', 'Charging_2', 10),
            ('Packing', 'Intersection_3', 10),
        ]

        # Add bidirectional paths
        for start, end, length in edges:
            # Forward direction
            edge_id = f"{start}-{end}"
            self.graph.add_edge(start, end, id=edge_id, length=length)
            self.segments[edge_id] = PathSegment(self.env, edge_id, length)

            # Reverse direction
            reverse_edge_id = f"{end}-{start}"
            self.graph.add_edge(end, start, id=reverse_edge_id, length=length)
            self.segments[reverse_edge_id] = PathSegment(self.env, reverse_edge_id, length)

        return nodes

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

        # Mark arrival at destination
        self.events.append((self.env.now, f'{name} reached destination at {self.env.now:.2f}'))

    def run_simulation(self):
        # Create the custom graph layout
        nodes = self.create_custom_graph()

        # Define different routes for AGVs (example routes)
        routes = [
            # Route for AGV-1: Receiving goods and storing them
            ['Receiving', 'Storage_A', 'Intersection_1', 'Storage_B'],

            # Route for AGV-2: Assembly line supply route
            ['Storage_A', 'Assembly', 'Intersection_2', 'Storage_C'],

            # Route for AGV-3: Delivery to shipping
            ['Storage_B', 'Storage_C', 'Intersection_3', 'Shipping'],

            # Route for AGV-4: Needs to charge during route
            ['Receiving', 'Charging_1', 'Storage_B', 'Intersection_2', 'Charging_2', 'Intersection_3', 'Shipping']
        ]

        # Create AGVs with different colors and speeds
        colors = ['red', 'blue', 'green', 'purple']
        for i in range(len(routes)):
            speed = random.uniform(1.0, 2.0)  # meters per second
            # Add a slight delay for each AGV to see them separate more clearly
            yield self.env.timeout(i * 5)
            route = routes[i]
            self.env.process(self.agv_process(f'AGV-{i + 1}', route, speed, colors[i]))

    def animate(self):
        """Create an animation of the simulation"""
        # Sort and get unique time points
        self.times = sorted(set(self.times))

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 10))

        # Draw the graph network
        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw_networkx(self.graph, pos, with_labels=True, node_size=700,
                         node_color='lightblue', font_weight='bold', font_size=8, ax=ax)

        # Add edge labels - but only for some edges to avoid clutter
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            if u < v:  # Only label one direction to reduce clutter
                edge_labels[(u, v)] = data['length']

        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels,
                                     font_size=7, ax=ax)

        # Create event text display
        event_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9,
                             bbox=dict(facecolor='white', alpha=0.7))

        # Set up AGV markers with corresponding colors
        agv_markers = {}
        agv_texts = {}
        agv_colors = {'AGV-1': 'red', 'AGV-2': 'blue', 'AGV-3': 'green', 'AGV-4': 'purple'}

        for name, color in agv_colors.items():
            marker, = ax.plot([], [], 'o', color=color, markersize=10, label=name)
            agv_markers[name] = marker
            agv_texts[name] = ax.text(0, 0, name, fontsize=8, ha='center', va='bottom', visible=False)

        ax.legend(loc='upper right')

        # Get axis limits from node positions with some padding
        x_values = [p[0] for p in pos.values()]
        y_values = [p[1] for p in pos.values()]
        x_min, x_max = min(x_values) - 5, max(x_values) + 5
        y_min, y_max = min(y_values) - 5, max(y_values) + 5
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

                    # Update text position
                    text = agv_texts[name]
                    text.set_position((x, y + 1))
                    text.set_visible(True)
                else:
                    marker.set_visible(False)
                    agv_texts[name].set_visible(False)

            # Update event text
            current_events = [e[1] for e in self.events if e[0] <= current_time]
            if current_events:
                # Show the last 5 events
                text = f"Time: {current_time:.2f}\n"
                text += "\n".join(current_events[-5:])
                event_text.set_text(text)

            return list(agv_markers.values()) + list(agv_texts.values()) + [event_text]

        # Create the animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.times),
            interval=50, blit=True)

        plt.title("AGV Traffic Simulation - Warehouse Layout")
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


# Create and run the simulation
def main():
    sim = AGVSimulation()
    env = sim.env

    # Start the simulation processes
    env.process(sim.run_simulation())

    # Run the simulation
    env.run(until=300)

    # Create and show the animation
    animation = sim.animate()

    return sim, animation


# Execute the simulation
if __name__ == "__main__":
    sim, animation = main()
