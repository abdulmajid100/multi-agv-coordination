import simpy
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# This file (try610_simpy.py) is a SimPy implementation of try610.py
# Key differences from try611_simpy.py:
# 1. AGV Tasks: Uses the predefined paths from try610.py instead of random generation
# 2. Functions: Includes both can_move() and can_move2() functions from try610.py
# 3. Movement Logic: Processes AGVs one by one with complex shared node handling
# 4. Shared Nodes: Implements the shared node calculation logic from try610.py

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

class AGVSimulation:
    def __init__(self, env, graph, verbose=True):
        self.env = env
        self.graph = graph
        self.verbose = verbose
        self.resource_states = {node: 0 for node in self.graph.nodes()}
        self.agv_tasks = {
            'AGV1': [
                [1, 4, 11, 10, 20],
                [20, 10, 11, 4, 5, 4, 6],
                [6, 4, 11, 10, 20],
                [20, 10, 11, 4, 1]
            ],
            'AGV2': [
                [4, 11, 12, 13, 14, 24],
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
                [9, 16, 15, 25],
                [25, 15, 16, 9]
            ]
        }
        self.agv_processes = []
        self.collision_count = 0
        self.successful_moves = 0
        self.total_moves = 0
        self.completed_paths = 0
        self.total_paths = sum(len(tasks) for tasks in self.agv_tasks.values())
        self.shared_nodes_list = {}
        self.x = []  # Special list for tracking AGVs with shared nodes (from try610.py)

        # Set initial nodes as reserved
        for agv, tasks in self.agv_tasks.items():
            if tasks and tasks[0]:  # Check if there are tasks and nodes in the first task
                starting_node = tasks[0][0]
                self.resource_states[starting_node] = agv

    def calculate_shared_nodes(self):
        """Calculate shared nodes between AGVs' current tasks (from try610.py)"""
        for agv, tasks in self.agv_tasks.items():
            other_agvs = [other_agv for other_agv in self.agv_tasks if other_agv != agv]
            shared_nodes_with_others = {other_agv: [] for other_agv in other_agvs}

            for other_agv in other_agvs:
                # Calculate shared nodes only for the current tasks
                if tasks and self.agv_tasks[other_agv]:
                    current_task = tasks[0]
                    other_current_task = self.agv_tasks[other_agv][0]
                    shared_nodes = set(current_task) & set(other_current_task)
                    shared_nodes_with_others[other_agv] = list(shared_nodes)

            self.shared_nodes_list[agv] = shared_nodes_with_others
            if self.verbose:
                print(f"Shared nodes for {agv} with others:")
                for other_agv, shared_nodes in shared_nodes_with_others.items():
                    print(f"  With {other_agv}: {shared_nodes}")

    def can_move(self, agv, shared_nodes_with_others, other_agvs, current_node, next_node):
        """Check if an AGV can move to the next node (from try610.py)"""
        if all(next_node not in shared_nodes_with_others[other_agv] for other_agv in other_agvs):
            return True
        elif not any(
            next_node in shared_nodes and
            any(self.resource_states[shared_node] == other_agv for shared_node in shared_nodes)
            for other_agv, shared_nodes in shared_nodes_with_others.items()
        ):
            return True
        return False

    def can_move2(self, agv, shared_nodes_with_others, other_agvs, current_node, next_node):
        """Alternative check if an AGV can move (from try610.py)"""
        if all(next_node not in shared_nodes_with_others[other_agv] for other_agv in other_agvs):
            return True
        elif all(
            next_node not in shared_nodes or
            all(self.resource_states[shared_node] != other_agv for shared_node in shared_nodes)
            for other_agv, shared_nodes in shared_nodes_with_others.items()
        ):
            return True
        return False

    def agv_process(self, agv):
        """SimPy process for an AGV."""
        while self.agv_tasks[agv]:
            s = 0  # Variable from try610.py
            w = 0  # Variable from try610.py

            if len(self.agv_tasks[agv][0]) >= 1 and w == 0:
                other_agvs = [other_agv for other_agv in self.agv_tasks if other_agv != agv]
                shared_nodes_with_others = {other_agv: [] for other_agv in other_agvs}

                # Complex shared node handling from try610.py
                for other_agv in other_agvs:
                    if self.agv_tasks[agv] and self.agv_tasks[other_agv]:
                        if other_agv not in self.x or agv not in self.x:
                            current_task = self.agv_tasks[agv][0]
                            other_current_task = self.agv_tasks[other_agv][0]
                            shared_nodes = set(current_task) & set(other_current_task)
                            shared_nodes_with_others[other_agv] = list(shared_nodes)

                        if 5 in shared_nodes_with_others[other_agv] or (other_agv in self.x and agv in self.x):
                            if agv not in self.x or other_agv not in self.x:
                                self.x.append(agv)
                                self.x.append(other_agv)

                            current_task = self.agv_tasks[agv][0]
                            if len(self.agv_tasks[other_agv]) > 1:
                                other_current_task = self.agv_tasks[other_agv][0] + self.agv_tasks[other_agv][1]
                            else:
                                other_current_task = self.agv_tasks[other_agv][0]
                            shared_nodes = set(current_task) & set(other_current_task)
                            shared_nodes_with_others[other_agv] = list(shared_nodes)

                            if len(self.agv_tasks[agv][0]) == 1:
                                self.x.remove(agv)
                                self.x.remove(other_agv)

                            if self.verbose:
                                print(self.x)
                                print(self.agv_tasks[agv][0])

                current_node = self.agv_tasks[agv][0][0]

                # Determine next node
                if len(self.agv_tasks[agv][0]) > 1:
                    next_node = self.agv_tasks[agv][0][1]
                elif len(self.agv_tasks[agv][0]) == 1:  # If the current task is completed, move to the next task
                    if len(self.agv_tasks[agv]) > 1:
                        next_node = self.agv_tasks[agv][1][0]
                    else:
                        # No next node, end of all tasks
                        self.agv_tasks[agv].pop(0)
                        self.completed_paths += 1
                        break

                if self.verbose:
                    print(f"{agv} current task: {current_task}")
                    print(f"{agv} at node: {current_node}")
                    print(f"Shared nodes: {shared_nodes_with_others}")
                    print(f"Resource states: {self.resource_states}")

                # Check if this is the last node of the last task
                if len(self.agv_tasks[agv]) == 1 and len(self.agv_tasks[agv][0]) == 1:
                    w = 1
                    self.agv_tasks[agv].pop(0)
                    self.completed_paths += 1
                    if self.verbose:
                        print(f"{agv} completed all tasks")
                    break
                else:
                    self.total_moves += 1
                    if self.can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
                        # Reserve the next node for the AGV
                        self.resource_states[current_node] = 0
                        self.resource_states[next_node] = agv

                        self.agv_tasks[agv][0].pop(0)
                        if len(self.agv_tasks[agv][0]) == 0:
                            # Release the current node
                            if len(self.agv_tasks[agv]) > 1:
                                self.agv_tasks[agv].pop(0)  # Remove the completed task
                                self.completed_paths += 1
                                if s == 1:
                                    s = 0

                        self.successful_moves += 1
                        if self.verbose:
                            print(f"{agv} moves from {current_node} to {next_node}")
                    else:
                        self.collision_count += 1
                        if self.verbose:
                            print(f"{agv} waiting at {current_node}")

            # Simulate time passing
            yield self.env.timeout(1)

    def run_simulation(self, max_time=100, progress_interval=10):
        """Run the simulation with a maximum time limit."""
        self.calculate_shared_nodes()

        # Create and start AGV processes
        self.agv_processes = []
        for agv in self.agv_tasks:
            process = self.env.process(self.agv_process(agv))
            self.agv_processes.append(process)

        # Create a process to track and display progress
        self.env.process(self.progress_tracker(max_time, progress_interval))

        # Run the simulation with a time limit
        self.env.run(until=max_time)

        # Calculate metrics
        success_rate = self.successful_moves / self.total_moves if self.total_moves > 0 else 0
        completion_rate = self.completed_paths / self.total_paths if self.total_paths > 0 else 0

        return {
            'success_rate': success_rate,
            'completion_rate': completion_rate,
            'collision_count': self.collision_count,
            'successful_moves': self.successful_moves,
            'total_moves': self.total_moves,
            'completed_paths': self.completed_paths,
            'total_paths': self.total_paths
        }

    def progress_tracker(self, max_time, interval):
        """Track and display simulation progress."""
        while True:
            yield self.env.timeout(interval)
            progress_percent = (self.env.now / max_time) * 100
            if self.verbose:
                print(f"Simulation progress: {self.env.now}/{max_time} time units ({progress_percent:.1f}%)")

            # Check if all AGVs have completed their tasks
            if all(len(tasks) == 0 for tasks in self.agv_tasks.values()):
                if self.verbose:
                    print("All AGVs have completed their tasks!")
                break

def visualize_graph():
    """Visualize the graph structure."""
    plt.figure(figsize=(10, 8))
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=500, node_color='lightblue')
    plt.title("Graph Structure for AGV Navigation")
    plt.savefig("graph_environment.png")
    plt.show()

def run_multiple_simulations(num_simulations=10, max_time=100, verbose=True, simulation_verbose=False):
    """Run multiple simulations and collect statistics."""
    results = []

    for i in range(num_simulations):
        env = simpy.Environment()
        simulation = AGVSimulation(env, G, verbose=simulation_verbose)
        result = simulation.run_simulation(max_time=max_time)
        results.append(result)
        if verbose:
            print(f"Simulation {i+1}/{num_simulations} completed: Success rate = {result['success_rate']:.2f}, Completion rate = {result['completion_rate']:.2f}")

    # Calculate average metrics
    avg_success_rate = sum(r['success_rate'] for r in results) / len(results)
    avg_completion_rate = sum(r['completion_rate'] for r in results) / len(results)
    avg_collision_count = sum(r['collision_count'] for r in results) / len(results)

    print("\nSimulation Results:")
    print(f"Average Success Rate: {avg_success_rate:.4f}")
    print(f"Average Completion Rate: {avg_completion_rate:.4f}")
    print(f"Average Collision Count: {avg_collision_count:.2f}")

    return results

def visualize_results(results):
    """Visualize the simulation results."""
    success_rates = [r['success_rate'] for r in results]
    completion_rates = [r['completion_rate'] for r in results]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(success_rates, bins=10, alpha=0.7)
    plt.axvline(sum(success_rates) / len(success_rates), color='r', linestyle='dashed', linewidth=1)
    plt.title('Distribution of Success Rates')
    plt.xlabel('Success Rate')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(completion_rates, bins=10, alpha=0.7)
    plt.axvline(sum(completion_rates) / len(completion_rates), color='r', linestyle='dashed', linewidth=1)
    plt.title('Distribution of Completion Rates')
    plt.xlabel('Completion Rate')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.show()

def analyze_by_agv_count(max_time=100, verbose=True, simulation_verbose=False):
    """Analyze how the number of AGVs affects performance."""
    # For try610_simpy.py, we'll simulate different scenarios by running multiple times
    # since we can't change the number of AGVs (they're predefined)
    num_runs = [5, 10, 15, 20, 25]
    avg_success_rates = []
    avg_completion_rates = []

    for num_simulations in num_runs:
        if verbose:
            print(f"\nRunning {num_simulations} simulations...")
        results = run_multiple_simulations(num_simulations=num_simulations, max_time=max_time, 
                                          verbose=verbose, simulation_verbose=simulation_verbose)
        avg_success_rates.append(sum(r['success_rate'] for r in results) / len(results))
        avg_completion_rates.append(sum(r['completion_rate'] for r in results) / len(results))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(num_runs, avg_success_rates, 'o-')
    plt.title('Average Success Rate vs. Number of Simulation Runs')
    plt.xlabel('Number of Simulation Runs')
    plt.ylabel('Average Success Rate')

    plt.subplot(1, 2, 2)
    plt.plot(num_runs, avg_completion_rates, 'o-')
    plt.title('Average Completion Rate vs. Number of Simulation Runs')
    plt.xlabel('Number of Simulation Runs')
    plt.ylabel('Average Completion Rate')

    plt.tight_layout()
    plt.savefig('agv_count_analysis.png')
    plt.show()

if __name__ == "__main__":
    # Set a reasonable time limit
    max_time = 100  # Limit simulation to 100 time units

    # Control verbosity
    verbose = True  # Show high-level progress
    simulation_verbose = False  # Hide detailed simulation output

    # Run a single simulation first to show basic functionality
    print("Starting a single AGV simulation based on try610.py...")
    env = simpy.Environment()
    simulation = AGVSimulation(env, G, verbose=simulation_verbose)
    results = simulation.run_simulation(max_time=max_time)

    # Display results of single simulation
    print("\nSingle Simulation Results:")
    print(f"Success Rate: {results['success_rate']:.4f}")
    print(f"Completion Rate: {results['completion_rate']:.4f}")
    print(f"Collision Count: {results['collision_count']}")
    print(f"Successful Moves: {results['successful_moves']}")
    print(f"Total Moves: {results['total_moves']}")
    print(f"Completed Paths: {results['completed_paths']}")
    print(f"Total Paths: {results['total_paths']}")

    # Visualize the graph structure
    visualize_graph()

    # Run multiple simulations and visualize results
    print("\nRunning multiple simulations to evaluate algorithm performance...")
    results = run_multiple_simulations(
        num_simulations=10, 
        max_time=max_time, 
        verbose=verbose, 
        simulation_verbose=simulation_verbose
    )
    visualize_results(results)

    # Analyze performance with different numbers of simulation runs
    print("\nAnalyzing performance with different numbers of simulation runs...")
    analyze_by_agv_count(
        max_time=max_time, 
        verbose=verbose, 
        simulation_verbose=simulation_verbose
    )
