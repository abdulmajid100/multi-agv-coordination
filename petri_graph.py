import networkx as nx
import matplotlib.pyplot as plt

# Define the Petri Net
places = ["P1", "P2", "P3"]
transitions = ["T1", "T2"]
arcs = [("P1", "T1"), ("T1", "P2"), ("P2", "T2"), ("T2", "P3")]

# Define the initial marking
initial_marking = {"P1": 1, "P2": 0, "P3": 0}

# Define the Petri Net structure
petri_net = {
    "places": places,
    "transitions": transitions,
    "arcs": arcs
}

# Function to check if a transition is enabled
def is_enabled(transition, marking, arcs):
    # Check if all input places have enough tokens
    for place, t in arcs:
        if t == transition and marking[place] < 1:
            return False
    return True

# Function to fire a transition
def fire_transition(transition, marking, arcs):
    new_marking = marking.copy()
    # Remove tokens from input places
    for place, t in arcs:
        if t == transition:
            new_marking[place] -= 1
    # Add tokens to output places
    for t, place in arcs:
        if t == transition:
            new_marking[place] += 1
    return new_marking

# Compute the reachability graph
def compute_reachability(petri_net, initial_marking):
    places = petri_net["places"]
    transitions = petri_net["transitions"]
    arcs = petri_net["arcs"]

    # Initialize the reachability graph
    reachability_graph = nx.DiGraph()
    reachability_graph.add_node(tuple(initial_marking.values()))  # Add initial marking as a node

    # Explore all reachable markings
    markings_to_explore = [initial_marking]
    explored_markings = []

    while markings_to_explore:
        current_marking = markings_to_explore.pop(0)
        explored_markings.append(current_marking)

        for transition in transitions:
            if is_enabled(transition, current_marking, arcs):
                new_marking = fire_transition(transition, current_marking, arcs)
                if tuple(new_marking.values()) not in reachability_graph:
                    reachability_graph.add_node(tuple(new_marking.values()))
                    markings_to_explore.append(new_marking)
                reachability_graph.add_edge(tuple(current_marking.values()), tuple(new_marking.values()), label=transition)

    return reachability_graph

# Compute the reachability graph
reachability_graph = compute_reachability(petri_net, initial_marking)

# Visualize the reachability graph
def visualize_reachability_graph(graph):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
    plt.title("Reachability Graph")
    plt.show()

# Visualize the reachability graph
visualize_reachability_graph(reachability_graph)