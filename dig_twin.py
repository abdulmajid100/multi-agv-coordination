import simpy
import networkx as nx
import random


def create_graph():
    G = nx.DiGraph()
    nodes = list(range(1, 30))
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
    G.remove_nodes_from([7, 8])
    return G


def agv_process(env, name, graph, start_node):
    current_node = start_node
    while True:
        print(f'{name} is at node {current_node} at time {env.now}')

        # Get the neighbors (possible next nodes) from the current node
        neighbors = list(graph.successors(current_node))

        if not neighbors:
            print(f'{name} has no further nodes to visit and will wait at {current_node}.')
            yield env.timeout(5)  # Wait for some time if no neighbors
            continue

        # Choose a random neighbor to move to
        next_node = random.choice(neighbors)
        travel_time = random.randint(1, 3)  # Simulate travel time between nodes
        print(f'{name} is moving from {current_node} to {next_node} at time {env.now}')

        yield env.timeout(travel_time)  # Simulate the time taken to travel
        current_node = next_node


# Create the simulation environment
env = simpy.Environment()

# Create the directed graph
graph = create_graph()

# Create AGV instances
agv1 = env.process(agv_process(env, 'AGV 1', graph, start_node=1))
agv2 = env.process(agv_process(env, 'AGV 2', graph, start_node=2))

# Run the simulation for a specified time
env.run(until=30)  # Run the simulation for 30 time units