"""
LocalSearchTruckRouter Module
-------------------------------
This module implements a local search heuristic to improve a CVRP solution
for truck routing. It includes multiple operators:
    1. Intra-route 2-opt: Optimize a single route by reordering stops.
    2. Inter-route Swap: Swap nodes between routes if capacity constraints allow.
    3. Route Merge: Merge two routes if their combined demand is under capacity and distance is reduced.
    
The algorithm iteratively applies these operators to the initial CVRP solution (obtained via TruckRouter)
until no further improvements are observed or a maximum number of iterations is reached.

Inspiration and References:
    - Simoni et al. (2020), "Optimization and analysis of a robot-assisted last mile delivery system", Transportation Research Part E,
      p. 67, lines 15–20.
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
      (For classical 2-opt and swap operators.)
      
This module is implemented in an OOP style with one class per file.
"""

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import math
import copy
import random
import time
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Import our base TruckRouter CVRP solver to generate the initial solution.
from components.truck_router.truck_router import TruckRouter


class LocalSearchTruckRouter:
    """
    LocalSearchTruckRouter improves an initial CVRP solution using a set of local search operators.
    The operators include intra-route 2-opt, inter-route swap, and route merge.
    """

    def __init__(
        self,
        file_path,
        min_cluster_size,
        truck_capacity,
        num_vehicles=25,
        max_iterations=1000,
        improvement_threshold=1e-3,
    ):
        """
        Initialize the LocalSearchTruckRouter.

        :param file_path: Path to the Solomon dataset.
        :param min_cluster_size: Minimum cluster size for HDBSCAN.
        :param truck_capacity: Truck capacity.
        :param num_vehicles: Number of vehicles allowed in the CVRP.
        :param max_iterations: Maximum iterations for the local search.
        :param improvement_threshold: Minimum improvement (relative) to continue searching.
        """
        self.file_path = file_path
        self.min_cluster_size = min_cluster_size
        self.truck_capacity = truck_capacity
        self.num_vehicles = num_vehicles
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold

        # Use base TruckRouter (CVRP version) for initial solution and data model
        self.base_router = TruckRouter(
            file_path, min_cluster_size, truck_capacity, num_vehicles
        )

    def total_distance(self, routes, distance_matrix):
        """Compute total distance of given routes."""
        total = 0
        for route in routes:
            route_distance = 0
            for i in range(len(route) - 1):
                route_distance += distance_matrix[route[i]][route[i + 1]]
            total += route_distance
        return total

    def intra_route_2opt(self, route, distance_matrix):
        """
        Perform 2-opt improvement on a single route.

        :param route: List of node indices (starting and ending with depot).
        :param distance_matrix: 2D array of distances.
        :return: Improved route and its distance.
        """
        best_route = route[:]
        best_distance = self.route_distance(best_route, distance_matrix)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    if j - i == 1:  # no change if adjacent
                        continue
                    new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                    new_distance = self.route_distance(new_route, distance_matrix)
                    if new_distance < best_distance - self.improvement_threshold:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
            # End while if no improvement found.
        return best_route, best_distance

    def route_distance(self, route, distance_matrix):
        """Compute the distance of a single route."""
        dist = 0
        for i in range(len(route) - 1):
            dist += distance_matrix[route[i]][route[i + 1]]
        return dist

    def inter_route_swap(self, routes, data):
        """
        Attempt to swap one node from one route with a node from another route to improve overall distance,
        while respecting capacity constraints.

        :param routes: List of routes.
        :param data: VRP data model (contains "demands", "vehicle_capacities", and "depot").
        :return: Modified routes if improvement found, else original routes.
        """
        best_routes = copy.deepcopy(routes)
        best_total = self.total_distance(best_routes, np.array(data["distance_matrix"]))
        improved = False

        # For each pair of routes (i, j)
        for i in range(len(best_routes)):
            for j in range(i + 1, len(best_routes)):
                route1 = best_routes[i]
                route2 = best_routes[j]
                # Skip routes that have only depot (empty)
                if len(route1) <= 2 or len(route2) <= 2:
                    continue
                # Try swapping each node (excluding depots) between the two routes
                for idx1 in range(1, len(route1) - 1):
                    for idx2 in range(1, len(route2) - 1):
                        node1 = route1[idx1]
                        node2 = route2[idx2]
                        # Check capacity feasibility:
                        load1 = sum(
                            data["demands"][n] for n in route1 if n != data["depot"]
                        )
                        load2 = sum(
                            data["demands"][n] for n in route2 if n != data["depot"]
                        )
                        new_load1 = (
                            load1 - data["demands"][node1] + data["demands"][node2]
                        )
                        new_load2 = (
                            load2 - data["demands"][node2] + data["demands"][node1]
                        )
                        if (
                            new_load1 > data["vehicle_capacities"][i]
                            or new_load2 > data["vehicle_capacities"][j]
                        ):
                            continue
                        # Perform swap
                        new_route1 = route1[:]
                        new_route2 = route2[:]
                        new_route1[idx1] = node2
                        new_route2[idx2] = node1
                        new_total = self.route_distance(
                            new_route1, np.array(data["distance_matrix"])
                        ) + self.route_distance(
                            new_route2, np.array(data["distance_matrix"])
                        )
                        old_total = self.route_distance(
                            route1, np.array(data["distance_matrix"])
                        ) + self.route_distance(
                            route2, np.array(data["distance_matrix"])
                        )
                        if new_total < old_total - self.improvement_threshold:
                            best_routes[i] = new_route1
                            best_routes[j] = new_route2
                            best_total = self.total_distance(
                                best_routes, np.array(data["distance_matrix"])
                            )
                            improved = True
        return best_routes if improved else routes

    def route_merge(self, routes, data):
        """
        Attempt to merge two routes if their combined demand is under capacity,
        and reorder the merged route using 2-opt to reduce total distance.

        :param routes: List of routes.
        :param data: VRP data model dictionary.
        :return: New set of routes (list of routes).
        """
        merged = False
        new_routes = copy.deepcopy(routes)
        num_routes = len(new_routes)
        # Try each pair of routes
        for i in range(num_routes):
            for j in range(i + 1, num_routes):
                route1 = new_routes[i]
                route2 = new_routes[j]
                # Skip if either route is trivial
                if len(route1) <= 2 or len(route2) <= 2:
                    continue
                # Compute combined load (excluding depot)
                load1 = sum(data["demands"][n] for n in route1 if n != data["depot"])
                load2 = sum(data["demands"][n] for n in route2 if n != data["depot"])
                if load1 + load2 > data["vehicle_capacities"][i]:
                    continue
                # Merge routes: remove the ending depot from route1 and starting depot from route2, then concatenate
                merged_route = route1[:-1] + route2[1:]
                # Re-optimize merged route using 2-opt
                merged_route, merged_distance = self.intra_route_2opt(
                    merged_route, np.array(data["distance_matrix"])
                )
                old_distance = self.route_distance(
                    route1, np.array(data["distance_matrix"])
                ) + self.route_distance(route2, np.array(data["distance_matrix"]))
                if merged_distance < old_distance - self.improvement_threshold:
                    # Merge successful: update route i and remove route j
                    new_routes[i] = merged_route
                    new_routes[j] = []  # mark for removal
                    merged = True
        # Remove empty routes
        new_routes = [r for r in new_routes if len(r) > 0]
        return new_routes if merged else routes

    def improve_solution(self, data, initial_solution):
        """
        Iteratively improve the initial solution using local search operators until no significant improvement
        is found or maximum iterations are reached.

        :param data: VRP data model dictionary.
        :param initial_solution: Initial solution as a list of routes.
        :return: Improved solution and its total distance.
        """
        current_solution = copy.deepcopy(initial_solution)
        distance_matrix = np.array(data["distance_matrix"])
        best_distance = self.total_distance(current_solution, distance_matrix)
        iterations_without_improvement = 0

        for it in range(self.max_iterations):
            new_solution = copy.deepcopy(current_solution)
            # Apply intra-route 2-opt to each route.
            for idx, route in enumerate(new_solution):
                if len(route) > 3:
                    improved_route, _ = self.intra_route_2opt(route, distance_matrix)
                    new_solution[idx] = improved_route

            # Apply inter-route swap operator.
            new_solution = self.inter_route_swap(new_solution, data)

            # Apply route merge operator.
            new_solution = self.route_merge(new_solution, data)

            new_distance = self.total_distance(new_solution, distance_matrix)
            improvement = best_distance - new_distance
            if improvement > self.improvement_threshold:
                current_solution = new_solution
                best_distance = new_distance
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            if iterations_without_improvement >= 100:
                break  # stop if no improvement for 100 consecutive iterations
        return current_solution, best_distance

    def solve_local_search(self):
        """
        Full workflow: Obtain an initial CVRP solution using the base TruckRouter,
        then improve it using local search operators.

        :return: Tuple (best_solution, best_distance, data, nodes)
        """
        # Obtain initial solution using base CVRP solver.
        depot, final_labels, final_centroids, cluster_demands = (
            self.base_router.load_cluster_data()
        )
        nodes = np.vstack([depot, final_centroids])
        data, nodes = self.base_router.create_vrp_data_model(
            depot, final_centroids, cluster_demands
        )
        initial_solution = self.base_router.solve_cvrp(data)["routes"]
        if initial_solution is None:
            raise Exception("Initial CVRP solution not found.")
        print(
            "Initial CVRP Total Distance:",
            self.total_distance(initial_solution, np.array(data["distance_matrix"])),
        )

        # Apply local search improvement.
        best_solution, best_distance = self.improve_solution(data, initial_solution)

        # Plot the improved solution.
        self.base_router.plot_truck_routes(best_solution, nodes)
        return best_solution, best_distance, data, nodes


if __name__ == "__main__":
    # Usage Example for LocalSearchTruckRouter
    import sys
    import os

    sys.path.append(os.getcwd())

    file_path = "dataset/c101.txt"
    min_cluster_size = 5
    truck_capacity = 200
    num_vehicles = 25

    ls_router = LocalSearchTruckRouter(
        file_path,
        min_cluster_size,
        truck_capacity,
        num_vehicles,
        max_iterations=1000,
        improvement_threshold=1e-3,
    )

    start_time = time.time()
    best_solution, best_distance, data, nodes = ls_router.solve_local_search()
    end_time = time.time()

    print("Best Local Search Solution Routes:")
    for idx, route in enumerate(best_solution):
        print(f"Route {idx+1}: {route}")
    print(f"Best Total Distance: {best_distance}")
    print(f"Local Search Runtime: {end_time - start_time:.2f} seconds")
