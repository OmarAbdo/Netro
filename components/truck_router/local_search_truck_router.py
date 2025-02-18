"""
LocalSearchTruckRouter Module (Enhanced Debug Version)
--------------------------------------------------------
This module implements an enhanced local search heuristic to improve a CVRP solution for truck routing.
It includes multiple operators:
    1. Intra-route 2-opt: Optimizes a single route by reordering stops.
    2. Inter-route Swap: Attempts to swap nodes between routes if capacity constraints allow.
    3. Route Merge: Merges two routes if their combined demand is under capacity and reoptimizes the merged route.
    
An adaptive operator selection mechanism is incorporated to choose between these operators based on their recent performance.
The algorithm iteratively applies the chosen operator until no significant improvement is found or the maximum iterations is reached.

Parameter adjustments:
    - Maximum iterations increased to 2000.
    - Destruction rate increased to 0.5.
    - Improvement threshold lowered to 1e-4.

Inspiration and References:
    - Chen, S., et al. (2018), "An adaptive large neighborhood search heuristic for dynamic vehicle routing problems",
      Computers and Electrical Engineering, 67, 596–607.
    - Simoni et al. (2020), "Optimization and analysis of a robot-assisted last mile delivery system", Transportation Research Part E.
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".

This module is implemented in an OOP style.
"""

import sys
import os
import numpy as np
import math
import copy
import random
import time
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

sys.path.append(os.getcwd())
from components.truck_router.truck_router import TruckRouter


class LocalSearchTruckRouter:
    """
    LocalSearchTruckRouter improves an initial CVRP solution using a set of local search operators.
    Adaptive Operator Selection chooses among:
        - Intra-route 2-opt
        - Inter-route swap
        - Route merge
    """

    def __init__(
        self,
        file_path,
        min_cluster_size,
        truck_capacity,
        num_vehicles=25,
        max_iterations=2000,
        improvement_threshold=1e-4,
        destruction_rate=0.5,
    ):
        self.file_path = file_path
        self.min_cluster_size = min_cluster_size
        self.truck_capacity = truck_capacity
        self.num_vehicles = num_vehicles
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.destruction_rate = destruction_rate

        self.base_router = TruckRouter(
            file_path, min_cluster_size, truck_capacity, num_vehicles
        )
        self.operators = {
            "2opt": self.apply_2opt,
            "swap": self.apply_swap,
            "merge": self.apply_merge,
        }
        self.operator_weights = {op: 1.0 for op in self.operators.keys()}

    def total_distance(self, routes, distance_matrix):
        total = 0
        for route in routes:
            for i in range(len(route) - 1):
                total += distance_matrix[route[i]][route[i + 1]]
        return total

    def route_distance(self, route, distance_matrix):
        return sum(
            distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)
        )

    # ------------------ Operator Implementations ------------------
    def apply_2opt(self, routes, distance_matrix, data):
        new_routes = copy.deepcopy(routes)
        for idx, route in enumerate(new_routes):
            if len(route) > 3:
                improved_route, _ = self.intra_route_2opt(route, distance_matrix)
                new_routes[idx] = improved_route
        return new_routes

    def apply_swap(self, routes, distance_matrix, data):
        return self.inter_route_swap(routes, data)

    def apply_merge(self, routes, distance_matrix, data):
        return self.route_merge(routes, data)

    def intra_route_2opt(self, route, distance_matrix):
        best_route = route[:]
        best_distance = self.route_distance(best_route, distance_matrix)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    if j - i == 1:
                        continue
                    new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                    new_distance = self.route_distance(new_route, distance_matrix)
                    if new_distance < best_distance - self.improvement_threshold:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
            if not improved:
                break
        return best_route, best_distance

    def inter_route_swap(self, routes, data):
        best_routes = copy.deepcopy(routes)
        distance_matrix = np.array(data["distance_matrix"])
        improved = False
        for i in range(len(best_routes)):
            for j in range(i + 1, len(best_routes)):
                route1 = best_routes[i]
                route2 = best_routes[j]
                if len(route1) <= 2 or len(route2) <= 2:
                    continue
                for idx1 in range(1, len(route1) - 1):
                    for idx2 in range(1, len(route2) - 1):
                        node1 = route1[idx1]
                        node2 = route2[idx2]
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
                        new_route1 = route1[:]
                        new_route2 = route2[:]
                        new_route1[idx1] = node2
                        new_route2[idx2] = node1
                        new_total = self.route_distance(
                            new_route1, distance_matrix
                        ) + self.route_distance(new_route2, distance_matrix)
                        old_total = self.route_distance(
                            route1, distance_matrix
                        ) + self.route_distance(route2, distance_matrix)
                        if new_total < old_total - self.improvement_threshold:
                            best_routes[i] = new_route1
                            best_routes[j] = new_route2
                            improved = True
        return best_routes if improved else routes

    def route_merge(self, routes, data):
        merged = False
        new_routes = copy.deepcopy(routes)
        distance_matrix = np.array(data["distance_matrix"])
        num_routes = len(new_routes)
        for i in range(num_routes):
            for j in range(i + 1, num_routes):
                route1 = new_routes[i]
                route2 = new_routes[j]
                if len(route1) <= 2 or len(route2) <= 2:
                    continue
                load1 = sum(data["demands"][n] for n in route1 if n != data["depot"])
                load2 = sum(data["demands"][n] for n in route2 if n != data["depot"])
                if load1 + load2 > data["vehicle_capacities"][i]:
                    continue
                merged_route = route1[:-1] + route2[1:]
                merged_route, merged_distance = self.intra_route_2opt(
                    merged_route, distance_matrix
                )
                old_distance = self.route_distance(
                    route1, distance_matrix
                ) + self.route_distance(route2, distance_matrix)
                if merged_distance < old_distance - self.improvement_threshold:
                    new_routes[i] = merged_route
                    new_routes[j] = []  # mark for removal
                    merged = True
        new_routes = [r for r in new_routes if len(r) > 0]
        return new_routes if merged else routes

    def improve_solution(self, data, initial_solution):
        current_solution = copy.deepcopy(initial_solution)
        distance_matrix = np.array(data["distance_matrix"])
        best_distance = self.total_distance(current_solution, distance_matrix)
        iterations_without_improvement = 0

        for it in range(self.max_iterations):
            # Adaptive operator selection
            operators = list(self.operators.keys())
            total_weight = sum(self.operator_weights[op] for op in operators)
            rnd = random.uniform(0, total_weight)
            cumulative = 0
            selected_operator = None
            for op in operators:
                cumulative += self.operator_weights[op]
                if rnd <= cumulative:
                    selected_operator = op
                    break

            new_solution = self.operators[selected_operator](
                current_solution, distance_matrix, data
            )
            new_distance = self.total_distance(new_solution, distance_matrix)
            delta = best_distance - new_distance

            if delta > self.improvement_threshold:
                current_solution = new_solution
                best_distance = new_distance
                iterations_without_improvement = 0
                self.operator_weights[selected_operator] += 0.1  # reward
            else:
                iterations_without_improvement += 1

            if (it + 1) % 100 == 0:
                total = sum(self.operator_weights.values())
                for op in self.operator_weights:
                    self.operator_weights[op] /= total

            if iterations_without_improvement >= 200:
                break
        return current_solution, best_distance

    def solve_local_search(self):
        depot, final_labels, final_centroids, cluster_demands = (
            self.base_router.load_cluster_data()
        )
        nodes = np.vstack([depot, final_centroids])
        data, nodes = self.base_router.create_vrp_data_model(
            depot, final_centroids, cluster_demands
        )
        sol_dict = self.base_router.solve_cvrp(data)
        initial_solution = sol_dict["routes"]
        if initial_solution is None:
            raise Exception("Initial CVRP solution not found.")
        print(
            "Initial CVRP Total Distance:",
            self.total_distance(initial_solution, np.array(data["distance_matrix"])),
        )
        best_solution, best_distance = self.improve_solution(data, initial_solution)
        # Plot the improved solution
        self.base_router.plot_truck_routes(best_solution, nodes)
        return best_solution, best_distance, data, nodes


if __name__ == "__main__":
    file_path = "dataset/c101.txt"
    min_cluster_size = 5
    truck_capacity = 200
    num_vehicles = 25

    ls_router = LocalSearchTruckRouter(
        file_path,
        min_cluster_size,
        truck_capacity,
        num_vehicles,
        max_iterations=2000,
        improvement_threshold=1e-4,
        destruction_rate=0.5,
    )
    start_time = time.time()
    best_solution, best_distance, data, nodes = ls_router.solve_local_search()
    end_time = time.time()

    print("Best Local Search Solution Routes:")
    for idx, route in enumerate(best_solution):
        print(f"Route {idx+1}: {route}")
    print(f"Best Total Distance: {best_distance}")
    print(f"Local Search Runtime: {end_time - start_time:.2f} seconds")
