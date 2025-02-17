"""
AdvancedTruckRouter Module (ALNS-based)
-----------------------------------------
This module implements an ALNS (Adaptive Large Neighborhood Search) heuristic for improving the initial CVRP
solution obtained over capacity-aware subcluster centroids (from HDBSCAN and capacitated splitting).
It uses three operators:
  1. Destroy Operator: Randomly removes a percentage of non-depot nodes from the solution.
  2. Repair Operator: Reinserts the removed nodes using a greedy insertion heuristic.
  3. (Optionally) Swap/Merge Operator: Could be added later to further refine the solution.

The acceptance of new solutions is governed by a simulated annealing criterion.
  
Inspiration and References:
    - Chen, S., et al. (2018), "An adaptive large neighborhood search heuristic for dynamic vehicle routing problems",
      Computers and Electrical Engineering, 67, 596–607. (For ALNS framework; see p.600, lines 5–15.)
    - Simoni et al. (2020), "Optimization and analysis of a robot-assisted last mile delivery system", Transportation Research Part E,
      p.67, lines 15–20. (For local search operators.)
      
This module is implemented in an OOP style with one class per file.
"""

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import random
import math
import copy
import time
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Import our base TruckRouter to reuse its CVRP solver for an initial solution.
from components.truck_router.truck_router import TruckRouter


class AdvancedTruckRouter:
    """
    AdvancedTruckRouter uses an ALNS metaheuristic to improve an initial CVRP solution for truck routing.
    The truck acts as a mobile depot serving subcluster centroids from customer data.
    """

    def __init__(
        self,
        file_path,
        min_cluster_size,
        truck_capacity,
        num_vehicles=25,
        iterations=1000,
        destruction_rate=0.3,
        initial_temperature=1000,
        cooling_rate=0.995,
    ):
        """
        Initialize the ALNS-based AdvancedTruckRouter.

        :param file_path: Path to the Solomon dataset.
        :param min_cluster_size: Minimum cluster size for HDBSCAN.
        :param truck_capacity: Truck capacity.
        :param num_vehicles: Number of vehicles available for the CVRP solver.
        :param iterations: Number of ALNS iterations.
        :param destruction_rate: Fraction of nodes (non-depot) to remove in each destroy step.
        :param initial_temperature: Starting temperature for simulated annealing.
        :param cooling_rate: Cooling rate for simulated annealing.
        """
        self.file_path = file_path
        self.min_cluster_size = min_cluster_size
        self.truck_capacity = truck_capacity
        self.num_vehicles = num_vehicles
        self.iterations = iterations
        self.destruction_rate = destruction_rate
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

        # We'll use our TruckRouter's CVRP solver to generate an initial solution.
        self.router = TruckRouter(
            file_path, min_cluster_size, truck_capacity, num_vehicles
        )

    def total_distance(self, routes, distance_matrix):
        """
        Compute total distance of a set of routes.

        :param routes: List of routes (each route is a list of node indices).
        :param distance_matrix: 2D list or numpy array of distances.
        :return: Total distance (float).
        """
        total = 0
        for route in routes:
            route_distance = 0
            for i in range(len(route) - 1):
                route_distance += distance_matrix[route[i]][route[i + 1]]
            total += route_distance
        return total

    def destroy(self, solution, distance_matrix):
        """
        Destroy operator: Remove a fraction of non-depot nodes from the solution.

        :param solution: List of routes.
        :param distance_matrix: The distance matrix (unused here, but may be used in a more advanced operator).
        :return: (partial_solution, removed_nodes)
        """
        partial_solution = copy.deepcopy(solution)
        removed_nodes = []

        # For each route, remove a fraction (destruction_rate) of nodes (excluding depot at index 0 at both ends)
        for r_idx, route in enumerate(partial_solution):
            # Do not remove if route length <= 2 (only depot)
            if len(route) <= 2:
                continue
            # Determine indices eligible for removal (exclude first and last depot)
            eligible_indices = list(range(1, len(route) - 1))
            num_remove = max(1, int(len(eligible_indices) * self.destruction_rate))
            indices_to_remove = random.sample(eligible_indices, num_remove)
            # Remove in descending order so that indices remain valid
            for idx in sorted(indices_to_remove, reverse=True):
                removed_nodes.append(route.pop(idx))
        return partial_solution, removed_nodes

    def repair(self, partial_solution, removed_nodes, data):
        """
        Repair operator: Reinsert removed nodes into the solution using a greedy insertion heuristic.

        :param partial_solution: List of routes after destruction.
        :param removed_nodes: List of nodes to be reinserted.
        :param data: VRP data model dictionary.
        :return: New solution (list of routes).
        """
        new_solution = copy.deepcopy(partial_solution)
        # For each removed node, try to insert it into the route that minimizes the increase in cost,
        # while respecting capacity constraints.
        # Note: Here we assume that the capacity constraints are enforced in the data["demands"] and data["vehicle_capacities"].
        for node in removed_nodes:
            best_increase = math.inf
            best_route_idx = None
            best_insert_pos = None

            for r_idx, route in enumerate(new_solution):
                # Calculate current load for this route.
                current_load = sum(
                    data["demands"][n] for n in route if n != data["depot"]
                )
                # Check capacity feasibility.
                if (
                    current_load + data["demands"][node]
                    > data["vehicle_capacities"][r_idx]
                ):
                    continue
                # Try inserting the node between each pair of consecutive nodes.
                for i in range(1, len(route)):
                    prev_node = route[i - 1]
                    next_node = route[i]
                    cost_increase = (
                        data["distance_matrix"][prev_node][node]
                        + data["distance_matrix"][node][next_node]
                        - data["distance_matrix"][prev_node][next_node]
                    )
                    if cost_increase < best_increase:
                        best_increase = cost_increase
                        best_route_idx = r_idx
                        best_insert_pos = i

            # If we found a feasible insertion, insert the node.
            if best_route_idx is not None:
                new_solution[best_route_idx].insert(best_insert_pos, node)
            else:
                # If no feasible route found, create a new route (depot, node, depot)
                new_solution.append([data["depot"], node, data["depot"]])
        return new_solution

    def alns(self, data, initial_solution):
        """
        The core ALNS algorithm that iteratively destroys and repairs the current solution.

        :param data: VRP data model dictionary.
        :param initial_solution: Initial solution as a list of routes.
        :return: The best solution found and its total distance.
        """
        # Compute the distance matrix from the data.
        distance_matrix = np.array(data["distance_matrix"])
        current_solution = initial_solution
        best_solution = current_solution
        best_distance = self.total_distance(best_solution, distance_matrix)
        current_distance = best_distance
        temperature = self.initial_temperature

        for it in range(self.iterations):
            # Destroy phase: remove some nodes from current_solution.
            partial_solution, removed_nodes = self.destroy(
                current_solution, distance_matrix
            )
            # Repair phase: reinsert the removed nodes greedily.
            new_solution = self.repair(partial_solution, removed_nodes, data)
            new_distance = self.total_distance(new_solution, distance_matrix)

            # Acceptance criterion (simulated annealing)
            if new_distance < current_distance:
                current_solution = new_solution
                current_distance = new_distance
                if new_distance < best_distance:
                    best_solution = new_solution
                    best_distance = new_distance
            else:
                # Accept with probability exp(-(delta)/temperature)
                delta = new_distance - current_distance
                acceptance_prob = math.exp(-delta / temperature)
                if random.random() < acceptance_prob:
                    current_solution = new_solution
                    current_distance = new_distance

            # Cooling
            temperature *= self.cooling_rate

        return best_solution, best_distance

    def solve_alns(self):
        """
        Full workflow: Load cluster data, build VRP data model, generate an initial solution using our CVRP solver,
        then improve the solution using ALNS.

        :return: Tuple (best_solution, best_distance, data, nodes) where:
                 - best_solution: List of routes representing the improved solution.
                 - best_distance: Total distance of the best solution.
                 - data: The VRP data model dictionary.
                 - nodes: 2D numpy array of node coordinates.
        """
        # Load cluster data from the base TruckRouter workflow.
        depot, final_labels, final_centroids, cluster_demands = (
            self.router.load_cluster_data()
        )
        # Build nodes: depot is first, then subcluster centroids.
        nodes = np.vstack([depot, final_centroids])
        # Build demands array: assume depot demand is 0; the rest, order by sorted keys.
        unique_labels = sorted(cluster_demands.keys())
        demands = [0]
        for label in unique_labels:
            demands.append(cluster_demands[label])

        # Create VRP data model using our base router method.
        data, nodes = self.router.create_vrp_data_model(
            depot, final_centroids, cluster_demands
        )

        # Get an initial solution using the CVRP solver.
        initial_solution = self.router.solve_cvrp(data)["routes"]
        if initial_solution is None:
            raise Exception("Initial CVRP solution not found.")

        print(
            "Initial CVRP Total Distance:",
            self.total_distance(initial_solution, np.array(data["distance_matrix"])),
        )

        # Apply ALNS to improve the initial solution.
        best_solution, best_distance = self.alns(data, initial_solution)

        # Plot improved solution.
        self.router.plot_truck_routes(best_solution, nodes)

        return best_solution, best_distance, data, nodes


if __name__ == "__main__":
    # Usage Example for AdvancedTruckRouter using ALNS
    import sys
    import os

    sys.path.append(os.getcwd())

    file_path = "dataset/c101.txt"
    min_cluster_size = 5
    truck_capacity = 200
    num_vehicles = 25

    # Initialize the ALNS-based advanced truck router.
    alns_router = AdvancedTruckRouter(
        file_path,
        min_cluster_size,
        truck_capacity,
        num_vehicles,
        iterations=500,
        destruction_rate=0.3,
        initial_temperature=1000,
        cooling_rate=0.995,
    )

    start_time = time.time()
    best_solution, best_distance, data, nodes = alns_router.solve_alns()
    end_time = time.time()

    print("Best ALNS Solution Routes:")
    for idx, route in enumerate(best_solution):
        print(f"Route {idx+1}: {route}")
    print(f"Best Total Distance: {best_distance}")
    print(f"ALNS Runtime: {end_time - start_time:.2f} seconds")
