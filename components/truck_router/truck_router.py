"""
AdvancedTruckRouter Module
---------------------------
This module implements the AdvancedTruckRouter class using an Adaptive Large Neighborhood Search (ALNS)
heuristic to solve the Traveling Salesman Problem (TSP) for truck routing among cluster centroids.

Inspiration and References:
    - Chen, S., Chen, R., Wang, G.-G., Gao, J., & Sangaiah, A.K. (2018), "An adaptive large neighborhood search heuristic 
      for dynamic vehicle routing problems", Computers and Electrical Engineering, 67, 596–607.
      (For the ALNS framework; see p. 600, lines 5–15.)
    - Rinaldi, M., Primatesta, S., Bugaj, M., Rostáš, J., & Guglieri, G. (2023), "Development of Heuristic Approaches for 
      Last-Mile Delivery TSP with a Truck and Multiple Drones", Drones, 7, 407.
      (For hybrid metaheuristic ideas; see p. 10, lines 10–20.)
      
This module follows SOLID principles (one class per file).
"""

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import random
import math
import matplotlib.pyplot as plt


class AdvancedTruckRouter:
    """
    AdvancedTruckRouter uses an ALNS heuristic to compute a near-optimal route for the truck
    among the cluster centroids (and the depot). The depot is assumed to be the first point in the input list.
    """

    def __init__(self, coords, iterations=1000, destruction_rate=0.2, random_seed=42):
        """
        Initialize the router with coordinates and ALNS parameters.

        :param coords: List or numpy array of 2D coordinates (first coordinate is the depot).
        :param iterations: Number of ALNS iterations.
        :param destruction_rate: Fraction of nodes to remove in the destruction phase.
        :param random_seed: Seed for reproducibility.
        """
        self.coords = np.array(coords)
        self.iterations = iterations
        self.destruction_rate = destruction_rate
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def euclidean_distance(self, a, b):
        """Compute Euclidean distance between two points a and b."""
        return np.linalg.norm(np.array(a) - np.array(b))

    def total_route_distance(self, route):
        """
        Compute total travel distance for a given route.

        :param route: List of indices representing the order in which nodes are visited.
        :return: Total distance.
        """
        distance = 0.0
        for i in range(len(route) - 1):
            distance += self.euclidean_distance(
                self.coords[route[i]], self.coords[route[i + 1]]
            )
        return distance

    def nearest_neighbor_solution(self):
        """
        Generate an initial TSP route using the nearest neighbor heuristic.
        Starts at depot (index 0) and returns to depot.

        :return: A route (list of indices).
        """
        n = len(self.coords)
        unvisited = list(range(1, n))
        route = [0]  # start at depot
        current = 0

        while unvisited:
            next_node = min(
                unvisited,
                key=lambda j: self.euclidean_distance(
                    self.coords[current], self.coords[j]
                ),
            )
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        route.append(0)  # return to depot
        return route

    def destruction_operator(self, route):
        """
        Remove a fraction of nodes (excluding depot) from the route.

        :param route: Current route (list of indices).
        :return: A tuple (remaining_route, removed_nodes)
        """
        # Exclude depot (first and last positions)
        indices = list(range(1, len(route) - 1))
        num_to_remove = max(1, int(self.destruction_rate * len(indices)))
        removed_indices = random.sample(indices, num_to_remove)
        removed_nodes = [route[i] for i in removed_indices]
        remaining_route = [
            route[i] for i in range(len(route)) if i not in removed_indices
        ]
        return remaining_route, removed_nodes

    def repair_operator(self, route, removed_nodes):
        """
        Reinsert removed nodes into the route using a greedy insertion heuristic.

        :param route: Current route (list of indices) after destruction.
        :param removed_nodes: List of nodes (indices) that were removed.
        :return: Repaired route (list of indices).
        """
        route = route.copy()
        for node in removed_nodes:
            best_insertion = None
            best_cost_increase = float("inf")
            # Try inserting between every pair of consecutive nodes
            for i in range(1, len(route)):
                prev_node = route[i - 1]
                next_node = route[i]
                cost_increase = (
                    self.euclidean_distance(self.coords[prev_node], self.coords[node])
                    + self.euclidean_distance(self.coords[node], self.coords[next_node])
                    - self.euclidean_distance(
                        self.coords[prev_node], self.coords[next_node]
                    )
                )
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_insertion = i
            route.insert(best_insertion, node)
        return route

    def solve_tsp_alns(self):
        """
        Solve the TSP using ALNS heuristic.

        :return: A tuple (best_route, best_distance)
        """
        # Generate initial solution using nearest neighbor
        current_route = self.nearest_neighbor_solution()
        current_distance = self.total_route_distance(current_route)
        best_route = current_route
        best_distance = current_distance

        for iteration in range(self.iterations):
            # Destruction: remove a fraction of nodes (excluding depot)
            partial_route, removed_nodes = self.destruction_operator(current_route)
            # Repair: reinsert removed nodes in best possible positions
            new_route = self.repair_operator(partial_route, removed_nodes)
            new_distance = self.total_route_distance(new_route)

            # Acceptance Criterion: if new solution is better, accept it
            if new_distance < current_distance:
                current_route = new_route
                current_distance = new_distance
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
            else:
                # Optionally, use simulated annealing acceptance criteria (omitted for simplicity)
                pass

            # (Optional) Print iteration info every 100 iterations
            if iteration % 100 == 0:
                print(
                    f"Iteration {iteration}: Current Distance = {current_distance:.2f}, Best Distance = {best_distance:.2f}"
                )

        return best_route, best_distance

    def plot_route(self, route):
        """
        Plot the TSP route on a 2D plot.

        :param route: List of indices representing the route.
        """
        plt.figure(figsize=(8, 8))
        x_coords = [self.coords[i][0] for i in route]
        y_coords = [self.coords[i][1] for i in route]
        plt.plot(x_coords, y_coords, marker="o")
        plt.scatter(
            self.coords[0][0], self.coords[0][1], c="red", marker="s", label="Depot"
        )
        plt.title("Optimized Truck Route (ALNS)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Usage Example for AdvancedTruckRouter
    import sys
    import os

    sys.path.append(os.getcwd())

    # For demonstration, assume we have cluster centroids from CustomerAnalyzer.
    # Here, we simulate with sample centroids (first point is depot).
    # In practice, these would be the centroids of clusters determined by HDBSCAN.
    centroids = [[40, 40], [42, 45], [50, 48], [55, 42], [47, 38], [60, 44]]  # Depot

    # Create an instance of AdvancedTruckRouter with the centroids.
    router = AdvancedTruckRouter(
        coords=centroids, iterations=1000, destruction_rate=0.3, random_seed=42
    )

    # Solve the TSP using ALNS
    best_route, best_distance = router.solve_tsp_alns()
    print("\nOptimized Truck Route (indices):")
    print(best_route)
    print(f"Total Route Distance: {best_distance:.2f}")

    # Plot the optimized route
    router.plot_route(best_route)
