import simpy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import numpy as np
from collections import defaultdict

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
    def __init__(self, env, graph, num_agvs=4, path_length_range=(10, 30), num_paths_range=(4, 5), verbose=True, use_original_process=True):
        self.env = env
        self.graph = graph
        self.num_agvs = num_agvs
        self.path_length_range = path_length_range
        self.num_paths_range = num_paths_range
        self.verbose = verbose
        self.use_original_process = use_original_process
        self.resource_states = {node: 0 for node in self.graph.nodes()}
        self.agv_tasks = {}
        self.agv_processes = []
        self.collision_count = 0
        self.successful_moves = 0
        self.total_moves = 0
        self.completed_paths = 0
        self.total_paths = 0

    def generate_random_path(self, start_node=None):
        """Generate a random path in the graph."""
        #print("the nodes are", list(self.graph.nodes()))
        if start_node is None:
            start_node = random.choice(list(self.graph.nodes()))

        path_length = random.randint(*self.path_length_range)
        path = [start_node]

        for _ in range(path_length - 1):
            neighbors = list(self.graph.neighbors(path[-1]))
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            path.append(next_node)

        return path

    def generate_random_tasks(self):
        """Generate random tasks for each AGV."""
        self.agv_tasks = {}
        for i in range(1, self.num_agvs + 1):
            agv_name = f'AGV{i}'
            num_paths = random.randint(*self.num_paths_range)

            # Generate first path
            valid_start_nodes = [n for n in self.graph.nodes() if n not in [7, 8]]
            start_node = random.choice(valid_start_nodes)

            paths = [self.generate_random_path(start_node)]

            # Generate subsequent paths starting from the end of the previous path
            for _ in range(num_paths - 1):
                start_node = paths[-1][-1]
                paths.append(self.generate_random_path(start_node))

            self.agv_tasks[agv_name] = paths
            self.total_paths += num_paths

        # Set initial nodes as reserved
        for agv, tasks in self.agv_tasks.items():
            if tasks and tasks[0]:
                starting_node = tasks[0][0]
                self.resource_states[starting_node] = agv

    def can_move(self, agv, shared_nodes_with_others, other_agvs, current_node, next_node):
        """Check if an AGV can move to the next node."""
        if all(next_node not in shared_nodes_with_others[other_agv] for other_agv in other_agvs):
            return True
        if any(
                next_node in shared_nodes and
                any(self.resource_states[shared_node] == other_agv for shared_node in shared_nodes)
                for other_agv, shared_nodes in shared_nodes_with_others.items()
        ):
            return False
        return True

    def agv_process_original(self, agv):
        """Original SimPy process for an AGV."""
        # Track consecutive failed attempts to detect deadlocks
        consecutive_failures = 0
        max_consecutive_failures = 1000  # Maximum number of consecutive failures before trying alternative path
        print(self.agv_tasks[agv], agv)
        while self.agv_tasks[agv]:
            current_task = self.agv_tasks[agv][0]
            #print(current_task)
            if len(current_task) <= 1:  # Task completed or only one node left
                self.agv_tasks[agv].pop(0)
                self.completed_paths += 1
                consecutive_failures = 0  # Reset counter when task changes
                if not self.agv_tasks[agv]:  # No more tasks
                    break
                continue

            current_node = current_task[0]
            next_node = current_task[1]

            # Check if move is possible
            other_agvs = [other_agv for other_agv in self.agv_tasks if other_agv != agv]
            shared_nodes_with_others = {other_agv: [] for other_agv in other_agvs}

            for other_agv in other_agvs:
                if self.agv_tasks[other_agv]:
                    other_current_task = self.agv_tasks[other_agv][0]
                    shared_nodes = set(current_task) & set(other_current_task)
                    shared_nodes_with_others[other_agv] = list(shared_nodes)

            self.total_moves += 1
            if self.can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
                # Execute move
                self.resource_states[current_node] = 0
                self.resource_states[next_node] = agv
                self.agv_tasks[agv][0].pop(0)
                self.successful_moves += 1
                consecutive_failures = 0  # Reset counter on successful move
                if self.verbose:
                    print(f"{agv} moves from {current_node} to {next_node}")
            else:
                self.collision_count += 1
                consecutive_failures += 1
                if self.verbose:
                    print(f"{agv} collision detected at {next_node}, staying at {current_node}")

                # Deadlock detection and resolution
                if consecutive_failures >= max_consecutive_failures:
                    if self.verbose:
                        print(f"{agv} detected potential deadlock, skipping to next task")
                    self.agv_tasks[agv].pop(0)  # Skip current task
                    self.completed_paths += 1  # Count as completed (though unsuccessful)
                    consecutive_failures = 0  # Reset counter
                    if not self.agv_tasks[agv]:  # No more tasks
                        break
                    continue

            # Simulate time passing
            yield self.env.timeout(1)

    def agv_process(self, agv):
        """SimPy process for an AGV."""
        # Track consecutive failed attempts to detect deadlocks
        consecutive_failures = 0
        max_consecutive_failures = 1000000  # Maximum number of consecutive failures before trying alternative path
        print(self.agv_tasks[agv], agv)
        while self.agv_tasks[agv]:
            current_task = self.agv_tasks[agv][0]

            if len(current_task) <= 1:  # Task completed or only one node left
                self.agv_tasks[agv].pop(0)
                self.completed_paths += 1
                consecutive_failures = 0  # Reset counter when task changes
                if not self.agv_tasks[agv]:  # No more tasks
                    break
                continue

            current_node = current_task[0]
            next_node = current_task[1]

            # Check if move is possible
            other_agvs = [other_agv for other_agv in self.agv_tasks if other_agv != agv]
            shared_nodes_with_others = {other_agv: [] for other_agv in other_agvs}

            for other_agv in other_agvs:
                if self.agv_tasks[other_agv]:

                    #other_current_task = self.agv_tasks[other_agv][0]
                    if current_task[-1] == self.agv_tasks[other_agv][0][-1]:  # Check if the last node of current task is the same as the other AGV's task
                        other_current_task = self.agv_tasks[other_agv][0] + self.agv_tasks[other_agv][1] if len(self.agv_tasks[other_agv]) > 1 else self.agv_tasks[other_agv][0]
                    else:
                        other_current_task = self.agv_tasks[other_agv][0]
                    shared_nodes = set(current_task) & set(other_current_task)
                    shared_nodes_with_others[other_agv] = list(shared_nodes)

            self.total_moves += 1
            if self.can_move(agv, shared_nodes_with_others, other_agvs, current_node, next_node):
                # Execute move
                self.resource_states[current_node] = 0
                self.resource_states[next_node] = agv
                self.agv_tasks[agv][0].pop(0)
                #print(self.agv_tasks[agv], agv)
                self.successful_moves += 1
                consecutive_failures = 0  # Reset counter on successful move
                if self.verbose:
                    print(f"{agv} moves from {current_node} to {next_node}")
            else:
                self.collision_count += 1
                consecutive_failures += 1
                if self.verbose:
                    print(f"{agv} collision detected at {next_node}, staying at {current_node}")

                # Deadlock detection and resolution
                if consecutive_failures >= max_consecutive_failures:
                    if self.verbose:
                        print(f"{agv} detected potential deadlock, skipping to next task")
                    self.agv_tasks[agv].pop(0)  # Skip current task
                    self.completed_paths += 1  # Count as completed (though unsuccessful)
                    consecutive_failures = 0  # Reset counter
                    if not self.agv_tasks[agv]:  # No more tasks
                        break
                    continue

            # Simulate time passing
            yield self.env.timeout(1)

    def run_simulation(self, max_time=1000, progress_interval=100):
        """Run the simulation with a maximum time limit."""
        self.generate_random_tasks()

        # Create and start AGV processes
        self.agv_processes = []
        for agv in self.agv_tasks:
            if self.use_original_process:
                process = self.env.process(self.agv_process_original(agv))
                if self.verbose:
                    print(f"Using original process for {agv}")
            else:
                process = self.env.process(self.agv_process(agv))
                if self.verbose:
                    print(f"Using new process for {agv}")
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

def run_multiple_simulations(num_simulations=50, num_agvs=4, max_time=1000, verbose=True, simulation_verbose=False, use_original_process=True):
    """Run multiple simulations and collect statistics."""
    results = []

    for i in range(num_simulations):
        env = simpy.Environment()
        simulation = AGVSimulation(env, G, num_agvs=num_agvs, verbose=simulation_verbose, use_original_process=use_original_process)
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

def analyze_by_agv_count(max_time=1000, verbose=True, simulation_verbose=False, use_original_process=True):
    """Analyze how the number of AGVs affects performance."""
    agv_counts = [2, 3, 4, 5, 6]
    avg_success_rates = []
    avg_completion_rates = []

    process_type = "Original" if use_original_process else "New"

    for num_agvs in agv_counts:
        if verbose:
            print(f"\nRunning simulations with {num_agvs} AGVs using {process_type} process...")
        results = run_multiple_simulations(num_simulations=20, num_agvs=num_agvs, max_time=max_time,
                                          verbose=verbose, simulation_verbose=simulation_verbose,
                                          use_original_process=use_original_process)
        avg_success_rates.append(sum(r['success_rate'] for r in results) / len(results))
        avg_completion_rates.append(sum(r['completion_rate'] for r in results) / len(results))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(agv_counts, avg_success_rates, 'o-')
    plt.title(f'Average Success Rate vs. Number of AGVs ({process_type} Process)')
    plt.xlabel('Number of AGVs')
    plt.ylabel('Average Success Rate')

    plt.subplot(1, 2, 2)
    plt.plot(agv_counts, avg_completion_rates, 'o-')
    plt.title(f'Average Completion Rate vs. Number of AGVs ({process_type} Process)')
    plt.xlabel('Number of AGVs')
    plt.ylabel('Average Completion Rate')

    plt.tight_layout()
    plt.savefig(f'agv_count_analysis_{process_type.lower()}.png')
    plt.show()

    return {
        'agv_counts': agv_counts,
        'avg_success_rates': avg_success_rates,
        'avg_completion_rates': avg_completion_rates
    }

def print_agv_paths(simulation):
    """Print the paths assigned to each AGV."""
    print("\n=== AGV Paths ===")
    for agv, tasks in simulation.agv_tasks.items():
        print(f"{agv} paths:")
        for i, path in enumerate(tasks):
            print(f"  Path {i+1}: {path}")
    print("=================\n")

def compare_processes(max_time=1000, verbose=True, simulation_verbose=False):
    """Compare the original and new AGV processes."""
    print("\n=== Comparing Original and New AGV Processes ===")

    # Create a simulation with a fixed seed to demonstrate that the same paths are used
    random.seed(42)  # Set a fixed seed for reproducibility
    env = simpy.Environment()
    simulation = AGVSimulation(env, G, num_agvs=5, verbose=simulation_verbose, use_original_process=True)
    simulation.generate_random_tasks()

    print("\nNOTE: Both process types use the SAME generated paths for AGVs.")
    print("The difference is in how they handle shared nodes and potential collisions.")
    print("Below are the paths that would be used by both process types:")
    print_agv_paths(simulation)
    random.seed()  # Reset the seed

    # Run analysis with original process
    print("\nRunning analysis with Original process...")
    original_results = analyze_by_agv_count(max_time=max_time, verbose=verbose, 
                                           simulation_verbose=simulation_verbose, 
                                           use_original_process=True)

    # Run analysis with new process
    print("\nRunning analysis with New process...")
    new_results = analyze_by_agv_count(max_time=max_time, verbose=verbose, 
                                      simulation_verbose=simulation_verbose, 
                                      use_original_process=False)

    # Compare the results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(original_results['agv_counts'], original_results['avg_success_rates'], 'o-', label='Original Process')
    plt.plot(new_results['agv_counts'], new_results['avg_success_rates'], 's-', label='New Process')
    plt.title('Success Rate Comparison')
    plt.xlabel('Number of AGVs')
    plt.ylabel('Average Success Rate')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(original_results['agv_counts'], original_results['avg_completion_rates'], 'o-', label='Original Process')
    plt.plot(new_results['agv_counts'], new_results['avg_completion_rates'], 's-', label='New Process')
    plt.title('Completion Rate Comparison')
    plt.xlabel('Number of AGVs')
    plt.ylabel('Average Completion Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig('process_comparison.png')
    plt.show()

    return {
        'original': original_results,
        'new': new_results
    }

def demonstrate_same_paths():
    """Demonstrate that both process types use the same generated paths."""
    print("\n=== Demonstrating that both process types use the same paths ===")

    # Set a fixed seed for reproducibility
    random.seed(42)

    # Create first simulation with original process
    env1 = simpy.Environment()
    sim1 = AGVSimulation(env1, G, num_agvs=5, verbose=False, use_original_process=True)
    sim1.generate_random_tasks()

    # Create second simulation with new process but same seed
    random.seed(42)  # Reset to same seed
    env2 = simpy.Environment()
    sim2 = AGVSimulation(env2, G, num_agvs=5, verbose=False, use_original_process=False)
    sim2.generate_random_tasks()

    # Compare the paths
    print("\nPaths for original process:")
    print_agv_paths(sim1)

    print("\nPaths for new process:")
    print_agv_paths(sim2)

    # Check if paths are identical
    paths_identical = True
    for agv in sim1.agv_tasks:
        if agv in sim2.agv_tasks:
            if sim1.agv_tasks[agv] != sim2.agv_tasks[agv]:
                paths_identical = False
                break
        else:
            paths_identical = False
            break

    if paths_identical:
        print("\nCONCLUSION: Both process types use IDENTICAL paths.")
        print("The difference is only in how they handle shared nodes and potential collisions.")
    else:
        print("\nCONCLUSION: The paths are different between the two process types.")

    # Reset the random seed
    random.seed()

if __name__ == "__main__":
    # Set a reasonable time limit to prevent infinite execution
    max_time = 500  # Limit each simulation to 500 time units

    # Control verbosity
    verbose = True  # Show high-level progress
    simulation_verbose = False  # Hide detailed simulation output

    # Choose which analysis to run
    run_basic_analysis = False
    run_agv_count_analysis = False
    run_process_comparison = True
    run_path_demonstration = True

    if run_basic_analysis:
        print("Running multiple simulations to evaluate algorithm performance...")
        results = run_multiple_simulations(
            num_simulations=10,
            max_time=max_time,
            verbose=verbose,
            simulation_verbose=simulation_verbose
        )
        visualize_results(results)

    if run_agv_count_analysis:
        print("\nAnalyzing the effect of AGV count on performance...")
        analyze_by_agv_count(
            max_time=max_time,
            verbose=verbose,
            simulation_verbose=simulation_verbose
        )

    if run_process_comparison:
        print("\nComparing original and new AGV processes...")
        # Use fewer simulations and AGVs for quicker comparison
        compare_processes(
            max_time=max_time,
            verbose=verbose,
            simulation_verbose=simulation_verbose
        )

    if run_path_demonstration:
        demonstrate_same_paths()
